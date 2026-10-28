"""JPL-driven Newtonian test-particle fits and variational uncertainty propagation.

States are heliocentric ICRF in AU and AU/day, epochs are MJD TDB. Planetary
positions follow the supplied SPK; indirect accelerations account for the Sun's
motion. This is not a full solar-system ephemeris fit or a non-gravitational model.
"""
from functools import lru_cache
import warnings
import os

from astropy import units as au
from astropy.time import Time
from astropy.utils import iers
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import least_squares

from .catalogue import unit_vectors
from .fit import Geometry, check_deadline, local_uncertainty, orbital_elements
from .kernels import ARCSEC, C

AU_KM = 149597870.7
GM_SCALE = 86400.**2 / AU_KM**3
SUN_MU = 132712440041.279419 * GM_SCALE
# DE440 gravitational parameters, km^3/s^2; giant planets include satellites.
BODY_IDS = (1, 2, 399, 301, 4, 5, 6, 7, 8, 9)
BODY_MU = np.array([22031.868551, 324858.592, 398600.435507, 4902.800118,
                    42828.375816, 126712764.1, 37940584.8418, 5794556.4,
                    6836527.10058, 975.5]) * GM_SCALE


class Ephemeris:
    def __init__(self, path, start, end, step=.5):
        from jplephem.spk import SPK
        self.kernel = SPK.open(str(path))
        self.min_mjd = max(segment.start_jd for segment in self.kernel.segments) - 2400000.5
        self.max_mjd = min(segment.end_jd for segment in self.kernel.segments) - 2400000.5
        self.masses = BODY_MU
        self.sun_mu = SUN_MU
        start, end = max(start, self.min_mjd), min(end, self.max_mjd)
        if start >= end:
            raise ValueError('Requested dates are outside the JPL ephemeris')
        grid = np.linspace(start, end, int(np.ceil((end-start)/step)) + 1)
        positions, velocities = self.exact(grid)
        self.spline = CubicHermiteSpline(grid, positions, velocities, axis=0, extrapolate=False)
        self.start, self.end = float(grid[0]), float(grid[-1])
        self.step = step

    def exact(self, t):
        jd = np.asarray(t) + 2400000.5
        sun, vsun = self.kernel[0, 10].compute_and_differentiate(jd)
        positions, velocities = [], []
        for body in BODY_IDS:
            if body in (399, 301):
                p, v = self.kernel[0, 3].compute_and_differentiate(jd)
                q, w = self.kernel[3, body].compute_and_differentiate(jd)
                p, v = p + q, v + w
            else:
                p, v = self.kernel[0, body].compute_and_differentiate(jd)
            positions.append(np.moveaxis((p-sun)/AU_KM, 0, -1))
            velocities.append(np.moveaxis((v-vsun)/AU_KM, 0, -1))
        return np.stack(positions, axis=-2), np.stack(velocities, axis=-2)

    def positions(self, t):
        return self.spline(t)

    def earth(self, t):
        jd = np.asarray(t) + 2400000.5
        p = self.kernel[0, 3].compute(jd) + self.kernel[3, 399].compute(jd) - self.kernel[0, 10].compute(jd)
        return np.moveaxis(p / AU_KM, 0, -1)


def acceleration(position, planets, masses, sun_mu=SUN_MU):
    radius = np.linalg.norm(position)
    if radius < .00465:
        raise ValueError('Trajectory intersects the Sun')
    delta = planets - position
    distances = np.linalg.norm(delta, axis=1)
    if np.any(distances < 1e-8):
        raise ValueError('Unresolved planetary encounter')
    solar = -sun_mu * position / radius**3
    result = solar + np.sum(masses[:, None] * (delta / distances[:, None]**3 -
                            planets / np.linalg.norm(planets, axis=1)[:, None]**3), axis=0)
    eye = np.eye(3)
    gradient = sun_mu * (3*np.outer(position, position)/radius**5 - eye/radius**3)
    gradient += np.sum(masses[:, None, None] *
        (3*delta[:, :, None]*delta[:, None, :]/distances[:, None, None]**5 -
         eye/distances[:, None, None]**3), axis=0)
    return result, gradient


class TrajectoryDomainError(ValueError):
    """Reception/retarded epochs unavailable in the prepared force ephemeris."""
    def __init__(self, invalid):
        super().__init__('Light-time epoch outside prepared ephemeris interval')
        self.invalid = np.asarray(invalid, dtype=bool)


class FitDeferred(Exception):
    """A cheap initializer needs a later unrestricted fit; not a rejection."""
    def __init__(self, rms_arcsec, evaluations):
        self.rms_arcsec, self.evaluations = float(rms_arcsec), int(evaluations)
        super().__init__('Planetary pilot deferred for a full fit')


class Trajectory:
    def __init__(self, state, epoch, start, end, ephemeris, *, deadline=np.inf,
                 rtol=2e-11, max_step=10., backend=None):
        self.epoch, self.ephemeris = epoch, ephemeris
        self.deadline, self.rtol, self.max_step = deadline, rtol, max_step
        self.backend = backend or os.environ.get('ITF_NBODY_BACKEND', 'scipy')
        if self.backend not in ('scipy', 'scipy_compiled_rhs', 'compiled'):
            raise ValueError('Unknown N-body backend')
        self.initial = np.r_[state, np.eye(6).ravel()]
        self.start = self.end = epoch
        # Each segment carries the STM relative to the ORIGINAL state epoch.
        self.segments = []
        self.extend(start, end)

    def _integrate(self, initial, epoch, target):
        if self.backend == 'compiled' and hasattr(self.ephemeris, 'spline'):
            from .nbody_fast import CompiledSolution
            return CompiledSolution(initial, epoch, target-epoch, self.ephemeris,
                rtol=self.rtol, max_step=self.max_step, deadline=self.deadline)

        def derivative(dt, y):
            check_deadline(self.deadline)
            acc, gradient = acceleration(y[:3], self.ephemeris.positions(epoch+dt),
                                         self.ephemeris.masses, self.ephemeris.sun_mu)
            matrix = y[6:].reshape(6, 6)
            return np.r_[y[3:6], acc, np.vstack((matrix[3:], gradient @ matrix[:3])).ravel()]

        if self.backend == 'scipy_compiled_rhs' and hasattr(self.ephemeris, 'spline'):
            from .nbody_fast import derivative as compiled_derivative
            spline = self.ephemeris.spline
            def derivative(dt, y):
                check_deadline(self.deadline)
                return compiled_derivative(epoch+dt, y, spline.x, spline.c,
                    self.ephemeris.masses, self.ephemeris.sun_mu)

        result = solve_ivp(derivative, (0., target-epoch), initial, method='DOP853',
            rtol=self.rtol, atol=1e-12, dense_output=True, max_step=self.max_step)
        if not result.success:
            raise ValueError('N-body integration failed: ' + result.message)
        return result.sol

    def extend(self, start, end):
        """Append missing intervals without reintegrating already covered epochs."""
        check_deadline(self.deadline)
        start, end = min(start, self.epoch), max(end, self.epoch)
        if start < self.ephemeris.start or end > self.ephemeris.end:
            raise ValueError('Trajectory outside prepared ephemeris interval')
        for origin, target in ((self.start, start), (self.end, end)):
            if (target >= self.start and target <= self.end) or abs(target-origin) < 1e-12:
                continue
            state, stm = self.evaluate([origin])
            solution = self._integrate(np.r_[state[0], stm[0].ravel()], origin, target)
            self.segments.append((min(origin, target), max(origin, target), origin, solution))
            self.start, self.end = min(self.start, target), max(self.end, target)

    def evaluate(self, times):
        times = np.atleast_1d(times).astype(float)
        if np.any(times < self.start-1e-9) or np.any(times > self.end+1e-9):
            raise ValueError('Requested light-time epoch outside trajectory')
        out = np.tile(self.initial, (len(times), 1))
        for low, high, origin, solution in self.segments:
            use = (times >= low-1e-9) & (times <= high+1e-9) & (times != self.epoch)
            if np.any(use):
                out[use] = solution(times[use]-origin).T
        return out[:, :6], out[:, 6:].reshape(-1, 6, 6)

    def states(self, times):
        """Light-time Newton steps need six states, not all 36 STM entries."""
        times = np.atleast_1d(times).astype(float)
        if np.any(times < self.start-1e-9) or np.any(times > self.end+1e-9):
            raise ValueError('Requested light-time epoch outside trajectory')
        out = np.tile(self.initial[:6], (len(times), 1))
        for low, high, origin, solution in self.segments:
            use = (times >= low-1e-9) & (times <= high+1e-9) & (times != self.epoch)
            if np.any(use):
                out[use] = (solution.states(times[use]-origin) if hasattr(solution, 'states')
                            else solution(times[use]-origin)[:6]).T
        return out

    def apparent(self, times, observers):
        times = np.atleast_1d(times).astype(float)
        emitted = times.copy()
        for _ in range(12):
            invalid = ((emitted < self.ephemeris.start) | (emitted > self.ephemeris.end) |
                       ~np.isfinite(emitted))
            if np.any(invalid):
                raise TrajectoryDomainError(invalid)
            # Light time is state-dependent; a fixed 64-day pad excluded fast modes.
            self.extend(float(emitted.min()), float(emitted.max()))
            states = self.states(emitted)
            vector = states[:, :3] - observers
            distance = np.linalg.norm(vector, axis=1)
            unit = vector / distance[:, None]
            denominator = 1 + np.einsum('ni,ni->n', unit, states[:, 3:])/C
            delta = (emitted-times+distance/C)/denominator
            emitted -= delta
            if np.max(np.abs(delta)) < 1e-10:
                break
        else:
            raise ValueError('Retarded light-time iteration did not converge')
        invalid = (emitted < self.ephemeris.start) | (emitted > self.ephemeris.end)
        if np.any(invalid):
            raise TrajectoryDomainError(invalid)
        self.extend(float(emitted.min()), float(emitted.max()))
        states, matrices = self.evaluate(emitted)
        vector = states[:, :3] - observers
        distance = np.linalg.norm(vector, axis=1)
        unit = vector / distance[:, None]
        dtime = -np.einsum('ni,nij->nj', unit, matrices[:, :3]) / C
        dtime /= (1 + np.einsum('ni,ni->n', unit, states[:, 3:])/C)[:, None]
        dvector = matrices[:, :3] + states[:, 3:, None]*dtime[:, None, :]
        tangent = np.eye(3)[None] - unit[:, :, None]*unit[:, None, :]
        jacobian = np.einsum('nij,njk->nik', tangent, dvector)/distance[:, None, None]
        return unit, distance, jacobian


class JPLGeometry(Geometry):
    def __init__(self, catalogue, sites, ephemeris):
        super().__init__(catalogue, sites)
        self.ephemeris = ephemeris

    def prepare(self, indices, deadline=np.inf):
        """Exact SPK Earth and ground geometry, batched by station."""
        missing = np.array(sorted(set(map(int, indices))-self.prepared.keys()), dtype=int)
        if not len(missing):
            return
        tracks = self.catalogue.tracks
        for code in np.unique(tracks['site'][missing]):
            if code not in self.sites:
                continue
            selected = missing[tracks['site'][missing] == code]
            for first in range(0, len(selected), 4096):
                check_deadline(deadline)
                batch = selected[first:first+4096]
                blocks = [self.catalogue.obs[tracks[i]['start']:tracks[i]['start']+tracks[i]['count']] for i in batch]
                data = np.concatenate(blocks)
                utc = Time(data['t'], format='mjd', scale='utc')
                with iers.conf.set_temp('auto_download', False), warnings.catch_warnings():
                    warnings.simplefilter('ignore', iers.IERSWarning)
                    offset, _ = self.sites[code].get_gcrs_posvel(utc)
                times = utc.tdb.mjd
                observer = self.ephemeris.earth(times) + offset.xyz.to_value(au.au).T
                unit = unit_vectors(data['ra_mas']/3600000, data['dec_mas']/3600000)
                begin = 0
                for index, block in zip(batch, blocks):
                    end = begin+len(block)
                    self.prepared[int(index)] = (block, times[begin:end], unit[begin:end], observer[begin:end])
                    begin = end
        # Bound persistent metadata; already cached track() tuples remain valid.
        while len(self.prepared) > 50000:
            self.prepared.pop(next(iter(self.prepared)))

    @lru_cache(maxsize=12000)
    def track(self, index):
        if index in self.prepared:
            return self.prepared[index]
        data, sites, _ = self.catalogue.observations([index])
        if sites[0] not in self.sites:
            raise ValueError('Observer coordinates unavailable')
        utc = Time(data['t'], format='mjd', scale='utc')
        with iers.conf.set_temp('auto_download', False), warnings.catch_warnings():
            warnings.simplefilter('ignore', iers.IERSWarning)
            offset, _ = self.sites[sites[0]].get_gcrs_posvel(utc)
        tdb = utc.tdb.mjd
        observer = self.ephemeris.earth(tdb) + offset.xyz.to_value(au.au).T
        return data, tdb, unit_vectors(data['ra_mas']/3600000, data['dec_mas']/3600000), observer


def covariance_factor(jacobian):
    scale = np.maximum(np.linalg.norm(jacobian, axis=0), 1e-30)
    _, singular, vt = np.linalg.svd(jacobian / scale, full_matrices=False)
    if len(singular) != 6 or singular[-1] <= singular[0]*max(jacobian.shape)*np.finfo(float).eps:
        raise ValueError('N-body state covariance is rank deficient')
    return (vt.T / singular) / scale[:, None]


def fit_nbody(geometry, indices, state, epoch, *, sigma=.3, rms_limit=.6,
              deadline=np.inf, max_nfev=35, fast_start=False, pilot_gate_arcsec=None,
              observation_covariance=None, allow_two_tracklets=False):
    t, observed, observers, labels = geometry.get(indices)
    minimum = 4 if allow_two_tracklets and len(set(indices)) == 2 else 6
    if len(t) < minimum or len(np.unique(labels)) < len(indices):
        return None  # A duplicate designation must not buy extra linkage evidence.
    if observation_covariance is not None and observation_covariance.n != len(t):
        raise ValueError('Observation covariance must match geometry.get order and deduplication')
    east = np.column_stack((-observed[:, 1], observed[:, 0], np.zeros(len(t))))
    east /= np.linalg.norm(east, axis=1)[:, None]
    north = np.cross(observed, east)
    if fast_start and np.ptp(t) > 120.:
        # Centering the Cartesian solve reduces the position/velocity correlation
        # of decade-long extensions. Propagate the initial orbit precisely once.
        midpoint = float((t.min()+t.max())/2)
        transfer = Trajectory(state, epoch, min(epoch, midpoint), max(epoch, midpoint),
                              geometry.ephemeris, deadline=deadline)
        state = transfer.evaluate([midpoint])[0][0]
        epoch = midpoint
    start = max(geometry.ephemeris.start, min(float(t.min()), epoch)-.05)
    end = min(geometry.ephemeris.end, max(float(t.max()), epoch)+.01)
    last_state, cache = None, None
    proposal_accuracy = False

    def evaluate(parameters):
        nonlocal last_state, cache
        check_deadline(deadline)
        if last_state is None or not np.array_equal(parameters, last_state):
            trajectory = Trajectory(parameters, epoch, start, end, geometry.ephemeris, deadline=deadline,
                                    **(dict(rtol=2e-8, max_step=256.) if proposal_accuracy else {}))
            predicted, _, jacobian = trajectory.apparent(t, observers)
            residual = np.column_stack((np.sum((predicted-observed)*east, axis=1),
                                        np.sum((predicted-observed)*north, axis=1))).ravel()*ARCSEC/sigma
            jac = np.stack((np.einsum('ni,nij->nj', east, jacobian),
                            np.einsum('ni,nij->nj', north, jacobian)), axis=1).reshape(-1, 6)*ARCSEC/sigma
            if observation_covariance is not None:
                residual = observation_covariance.whiten(residual*sigma)
                jac = observation_covariance.whiten(jac*sigma)
            last_state, cache = parameters.copy(), (residual, jac, predicted)
        return cache

    pilot_iterations = 0
    if fast_start and end-start > 120.:
        # A less expensive planetary solve supplies only an initializer. Every
        # acceptance, covariance and residual is recomputed by the strict solver.
        # Pilot failure does not reject an association: restore its original start.
        proposal_accuracy = True
        try:
            pilot = least_squares(lambda x: evaluate(x)[0], state, jac=lambda x: evaluate(x)[1],
                x_scale='jac', max_nfev=min(4, max_nfev) if pilot_gate_arcsec else max_nfev,
                ftol=1e-5, xtol=1e-7, gtol=1e-4)
            # A correlated likelihood's whitened residual is not angular RMS.
            _, _, pilot_predicted = evaluate(pilot.x)
            pilot_miss = np.arctan2(np.linalg.norm(np.cross(pilot_predicted, observed), axis=1),
                                    np.sum(pilot_predicted*observed, axis=1))*ARCSEC
            pilot_rms = np.sqrt(np.mean(pilot_miss**2))
            if pilot_gate_arcsec and pilot_rms > pilot_gate_arcsec:
                raise FitDeferred(pilot_rms, pilot.nfev)
            if np.isfinite(pilot.x).all() and np.isfinite(pilot.fun).all():
                state = pilot.x
                pilot_iterations = pilot.nfev
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            pass
        proposal_accuracy = False
        last_state, cache = None, None
    result = least_squares(lambda x: evaluate(x)[0], state, jac=lambda x: evaluate(x)[1],
                           x_scale='jac', max_nfev=max_nfev, ftol=1e-7 if fast_start else 1e-9,
                           xtol=1e-8 if fast_start else 1e-10, gtol=1e-7)
    # The optimizer already evaluated this exact state. Reuse the same apparent
    # positions for acceptance instead of integrating the final orbit again.
    residual, jac, predicted = evaluate(result.x)
    miss = np.arctan2(np.linalg.norm(np.cross(predicted, observed), axis=1),
                      np.sum(predicted*observed, axis=1))*ARCSEC
    rms = float(np.sqrt(np.mean(miss**2)))
    if not result.success:
        raise ValueError('N-body optimizer did not converge')
    if rms > rms_limit or np.max(miss) > 2*rms_limit or any(
            np.sqrt(np.mean(miss[labels == label]**2)) > rms_limit for label in np.unique(labels)):
        return None
    elements = orbital_elements(result.x)
    if elements is None:
        return None
    factor = covariance_factor(jac)
    return dict(tracklet_indices=sorted(map(int, indices)), observations=len(t),
                range_au=float(np.linalg.norm(result.x[:3]-geometry.ephemeris.earth(epoch))),
                span_days=float(np.ptp(t)), rms_arcsec=rms, max_residual_arcsec=float(np.max(miss)),
                state_au_au_per_day=result.x.tolist(), epoch_mjd_tdb=float(epoch),
                covariance_factor=factor.tolist(), chi2=float(residual@residual),
                iterations=result.nfev, pilot_iterations=pilot_iterations,
                **(dict(astrometric_error_model=observation_covariance.identity,
                        measurement_log_determinant=observation_covariance.logdet)
                   if observation_covariance is not None else {}), **elements)
