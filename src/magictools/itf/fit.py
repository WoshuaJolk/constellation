"""Topocentric, light-time-corrected multistart orbit fitting and explicit scores."""
from functools import lru_cache
import math
from pathlib import Path
import time
import warnings
from types import SimpleNamespace

import erfa
from astropy import units as au
from astropy.coordinates import EarthLocation, get_body_barycentric_posvel
from astropy.time import Time
from astropy.utils import iers
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, logsumexp

from .catalogue import unit_vectors
from .derivatives import residual_jacobian, solve_lm, state_geometry
from .kernels import ARCSEC, MU, directions, position


class BudgetExpired(Exception):
    pass


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise BudgetExpired


def read_sites(path):
    sites = {}
    for line in Path(path).read_text().splitlines():
        try:
            code = line[:3]
            longitude, rc, rs = float(line[4:13]), float(line[13:21]), float(line[21:30])
        except ValueError:
            continue
        if not (np.isfinite([longitude, rc, rs]).all() and 0 <= longitude <= 360
                and 0 <= rc <= 1.1 and abs(rs) <= 1.1):
            continue
        theta = np.deg2rad(longitude)
        sites[code.encode()] = EarthLocation.from_geocentric(
            6378137*rc*np.cos(theta), 6378137*rc*np.sin(theta), 6378137*rs, unit=au.m)
    if not sites:
        raise ValueError('No terrestrial observatory coordinates in MPC ObsCodes file')
    return sites


class Geometry:
    def __init__(self, catalogue, sites):
        self.catalogue, self.sites = catalogue, sites
        self.prepared = {}

    def prepare(self, indices, deadline=np.inf):
        """Batch Earth/site astrometry once per station, before parallel orbit fits."""
        indices = np.array(sorted(set(map(int, indices))-self.prepared.keys()), dtype=int)
        if not len(indices):
            return
        tracks = self.catalogue.tracks
        for code in np.unique(tracks['site'][indices]):
            check_deadline(deadline)
            if code not in self.sites:
                continue
            selected = indices[tracks['site'][indices] == code]
            # Bound temporary astropy arrays; retain only the prepared geometry.
            for offset in range(0, len(selected), 4096):
                check_deadline(deadline)
                batch = selected[offset:offset+4096]
                blocks = [self.catalogue.obs[tracks[i]['start']:tracks[i]['start']+tracks[i]['count']] for i in batch]
                data = np.concatenate(blocks)
                utc = Time(data['t'], format='mjd', scale='utc')
                with iers.conf.set_temp('auto_download', False), warnings.catch_warnings():
                    warnings.simplefilter('ignore', iers.IERSWarning)
                    offset_position, _ = self.sites[code].get_gcrs_posvel(utc)
                tdb = utc.tdb
                earth_heliocentric, _ = erfa.epv00(tdb.jd1, tdb.jd2)
                observers = earth_heliocentric['p']+offset_position.xyz.to_value(au.au).T
                unit = unit_vectors(data['ra_mas']/3600000, data['dec_mas']/3600000)
                times = tdb.mjd
                start = 0
                for i, block in zip(batch, blocks):
                    end = start+len(block)
                    self.prepared[int(i)] = (block, times[start:end], unit[start:end], observers[start:end])
                    start = end

    @lru_cache(maxsize=12000)
    def track(self, index):
        if index in self.prepared:
            return self.prepared[index]
        data, sites, _ = self.catalogue.observations([index])
        code = sites[0]
        if code not in self.sites:
            raise ValueError(f'No terrestrial observer coordinates for {code!r}')
        t = Time(data['t'], format='mjd', scale='utc')
        # No hidden IERS/network download. Builtin ephemerides are approximate.
        with iers.conf.set_temp('auto_download', False), warnings.catch_warnings():
            warnings.simplefilter('ignore', iers.IERSWarning)
            offset, _ = self.sites[code].get_gcrs_posvel(t)
        earth, _ = get_body_barycentric_posvel('earth', t.tdb, ephemeris='builtin')
        sun, _ = get_body_barycentric_posvel('sun', t.tdb, ephemeris='builtin')
        observer = (earth.xyz-sun.xyz+offset.xyz).to_value(au.au).T
        u = unit_vectors(data['ra_mas']/3600000, data['dec_mas']/3600000)
        return data, t.tdb.mjd, u, observer

    def get(self, indices):
        parts = [self.track(int(i)) for i in indices]
        data, t, u, observer = [np.concatenate([p[k] for p in parts]) for k in range(4)]
        labels = np.concatenate([np.full(len(p[0]), i, dtype=int) for p, i in zip(parts, indices)])
        sites = np.concatenate([np.full(len(p[0]), self.catalogue.tracks[i]['site'], dtype='S3')
                                for p, i in zip(parts, indices)])
        # Deduplicate cross-designation resubmissions as well.
        key = np.rec.fromarrays([data['t'], data['ra_mas'], data['dec_mas'], sites])
        _, unique = np.unique(key, return_index=True)
        unique = unique[np.argsort(t[unique])]
        return t[unique], u[unique], observer[unique], labels[unique]


def gauss_seeds(t, u, observer):
    """All positive real roots of the classical three-direction Gauss polynomial."""
    indices = [0, len(t)//2, len(t)-1]
    l, r = u[indices].T, observer[indices].T
    tau1, tau3 = t[indices[0]]-t[indices[1]], t[indices[2]]-t[indices[1]]
    tau = tau3-tau1
    if min(-tau1, tau3) < 1e-4 or abs(np.linalg.det(l)) < 1e-15:
        return []
    inv = np.linalg.inv(l)
    p = inv@r
    a1, a3 = tau3/tau, -tau1/tau
    b1, b3 = a1*(tau*tau-tau3*tau3)/6, a3*(tau*tau-tau1*tau1)/6
    a = p[1, 0]*a1-p[1, 1]+p[1, 2]*a3
    b = MU*(p[1, 0]*b1+p[1, 2]*b3)
    e, r2 = l[:, 1]@r[:, 1], r[:, 1]@r[:, 1]
    roots = np.roots([1, 0, -(a*a+2*a*e+r2), 0, 0, -2*b*(a+e), 0, 0, -b*b])
    result = []
    for root in roots:
        if abs(root.imag) > 1e-7 or root.real <= .03:
            continue
        radius = root.real
        c1, c3 = a1+MU*b1/radius**3, a3+MU*b3/radius**3
        rho = (inv@(r@np.array([-c1, 1, -c3])))/np.array([c1, -1, c3])
        if np.any(rho <= 0):
            continue
        xyz = l*rho+r
        f1, f3 = 1-.5*MU*tau1*tau1/radius**3, 1-.5*MU*tau3*tau3/radius**3
        g1, g3 = tau1-MU*tau1**3/(6*radius**3), tau3-MU*tau3**3/(6*radius**3)
        denominator = f1*g3-f3*g1
        if abs(denominator) > 1e-12:
            v = (f1*xyz[:, 2]-f3*xyz[:, 0])/denominator
            # Gauss uses reception epochs; differential correction handles light time.
            result.append((np.r_[xyz[:, 1], v], float(t[indices[1]])))
    return result


def orbital_elements(state):
    r, v = state[:3], state[3:]
    radius = np.linalg.norm(r)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    if hn < 1e-14 or radius < .01:
        return None
    eccentricity = np.linalg.norm(np.cross(v, h)/MU-r/radius)
    energy = np.dot(v, v)/2-MU/radius
    obl = np.deg2rad(23.439291111)
    hz = -np.sin(obl)*h[1]+np.cos(obl)*h[2]
    return dict(eccentricity=float(eccentricity), perihelion_au=float(hn*hn/(MU*(1+eccentricity))),
                semimajor_axis_au=float(-MU/(2*energy)) if abs(energy) > 1e-16 else None,
                inclination_deg=float(np.rad2deg(np.arccos(np.clip(hz/hn, -1, 1)))),
                conic='hyperbolic' if energy > 0 else 'elliptic')


def linear_chi2(t, u, labels, sigma):
    total = 0.
    for label in np.unique(labels):
        selected = labels == label
        tt, uu = t[selected], u[selected]
        design = np.column_stack([np.ones(len(tt)), tt-tt.mean()])
        residual = uu-design@np.linalg.lstsq(design, uu, rcond=None)[0]
        total += np.sum(residual**2)*(ARCSEC/sigma)**2
    return float(total)


def local_uncertainty(state, jacobian, epoch_offset, observer):
    """Formal, local Gaussian errors at fixed astrometric sigma, not a posterior.

    Column scaling avoids a units-dependent rank/condition diagnostic. Work with
    the covariance square root so nearly singular normal equations are never
    formed and small residuals do not artificially shrink the specified noise.
    """
    scales = np.maximum(np.linalg.norm(jacobian, axis=0), 1e-30)
    _, singular, vt = np.linalg.svd(jacobian/scales, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*max(jacobian.shape)*np.finfo(float).eps))
    diagnostics = dict(scaled_jacobian_rank=rank, scaled_jacobian_condition=None,
                       range_sigma_au=None, inclination_sigma_deg=None,
                       energy_sigma_au2_per_day2=None, energy_sign_resolved_3sigma=False,
                       local_linearization_weak=True,
                       uncertainty_kind='local linear Gaussian; fixed astrometric sigma; no model error or mode integration')
    if rank < 6:
        return diagnostics
    factor = (vt.T/singular)/scales[:, None]
    p, _, position_jac = state_geometry(state, epoch_offset)
    delta = p-observer
    distance = np.linalg.norm(delta)
    range_sigma = float(np.linalg.norm((delta/distance)@position_jac@factor))
    r, v = state[:3], state[3:]
    radius = np.linalg.norm(r)
    energy = np.dot(v, v)/2-MU/radius
    energy_sigma = float(np.linalg.norm(np.r_[MU*r/radius**3, v]@factor))
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    obl = np.deg2rad(23.439291111)
    pole = np.array([0., -np.sin(obl), np.cos(obl)])
    cosine = np.clip(h@pole/hn, -1., 1.)
    sine = np.sqrt(max(0., 1-cosine*cosine))
    inclination_sigma = None
    if sine > 1e-10:
        dh = np.column_stack([np.cross(np.eye(3)[k], v) for k in range(3)]+
                             [np.cross(r, np.eye(3)[k]) for k in range(3)])
        gradient = -(pole-cosine*h/hn)/(hn*sine)*180/np.pi
        inclination_sigma = float(np.linalg.norm(gradient@dh@factor))
    diagnostics.update(scaled_jacobian_condition=float(singular[0]/singular[-1]),
                       range_sigma_au=range_sigma, inclination_sigma_deg=inclination_sigma,
                       energy_sigma_au2_per_day2=energy_sigma,
                       energy_sign_resolved_3sigma=bool(abs(energy) > 3*energy_sigma),
                       local_linearization_weak=bool(range_sigma > .5*distance or
                           inclination_sigma is None or inclination_sigma > 10.))
    return diagnostics


def fit_chain(geometry, indices, *, sigma=.3, rms_limit=.6, prior=.001,
              deadline=np.inf, extra_seeds=(), max_nfev=80, jacobian_backend="compiled",
              broad_starts=False, start_policy="full", allow_two_tracklets=False):
    check_deadline(deadline)
    if start_policy not in ('full', 'scout', 'provided'):
        raise ValueError('Unknown orbit-start policy')
    t, u, observers, labels = geometry.get(indices)
    minimum = 4 if allow_two_tracklets and len(set(indices)) == 2 else 6
    if len(t) < minimum or len(np.unique(labels)) < len(indices):
        return None
    ra = np.arctan2(u[:, 1], u[:, 0])
    east = np.column_stack([-np.sin(ra), np.cos(ra), np.zeros(len(t))])
    north = np.cross(u, east)
    middle = len(t)//2
    epoch = float(t[middle])
    seeds = list(extra_seeds)
    if start_policy != 'provided':
        seeds += gauss_seeds(t, u, observers)
    # Range hypotheses cover nearby through distant objects; radial branches are explicit.
    dt = t[-1]-t[0]
    hypotheses = [(distance, vr) for distance in (.1, .5, 1., 2., 4., 10., 40., 150.)
                  for vr in (0., -.01, .01)]
    if broad_starts:
        hypotheses += [(distance, vr) for distance in (.01, .03, 300., 1000.)
                       for vr in (0., -.01, .01)]
        hypotheses += [(distance, vr) for distance in (.03, .1, 1., 10.) for vr in (-.05, .05)]
    if start_policy == 'scout':
        # An explicit first sweep, never a replacement for the retained full-grid
        # queue. Gauss branches plus nearby/main-belt/distant/extreme-range starts.
        hypotheses = [(distance, 0.) for distance in (.03, 1., 40., 1000.)]
    elif start_policy == 'provided':
        hypotheses = []
    for distance, vr in hypotheses:
        r = observers[middle]+distance*u[middle]
        v = (observers[-1]-observers[0]+distance*(u[-1]-u[0]))/dt
        seeds.append((np.r_[r, v+vr*u[middle]], epoch))
    solutions, canonical_states = [], []
    attempts = 0
    for seed, seed_epoch in seeds:
        check_deadline(deadline)
        if not np.isfinite(seed).all():
            continue
        offsets = t-seed_epoch

        def numerical_residual(state):
            check_deadline(deadline)
            prediction = directions(state, offsets, observers)
            if not np.isfinite(prediction).all():
                return np.full(2*len(t), 1e12)
            denominator = np.sum(prediction*u, axis=1)
            return np.column_stack([np.arctan2(np.sum(prediction*east, axis=1), denominator),
                                    np.arctan2(np.sum(prediction*north, axis=1), denominator)]).ravel()*ARCSEC/sigma

        cached_state, cached_value = None, None
        def evaluate(state):
            nonlocal cached_state, cached_value
            check_deadline(deadline)
            if cached_state is None or not np.array_equal(state, cached_state):
                cached_state = state.copy()
                cached_value = residual_jacobian(state, offsets, observers, u, east, north, sigma)
            return cached_value
        if jacobian_backend in ('compiled', 'analytic'):
            residual = lambda state: evaluate(state)[0]
            jacobian = lambda state: evaluate(state)[1]
        elif jacobian_backend == 'finite_difference':
            residual, jacobian = numerical_residual, '2-point'
        else:
            raise ValueError('Unknown Jacobian backend')
        attempts += 1
        if jacobian_backend == 'compiled':
            x, fun, jac, success = solve_lm(seed, offsets, observers, u, east, north, sigma, max_nfev)
            result = SimpleNamespace(x=x, fun=fun, jac=jac, success=success)
        else:
            result = least_squares(residual, seed, jac=jacobian, method='lm', x_scale='jac', max_nfev=max_nfev,
                                   ftol=1e-7, xtol=1e-9, gtol=1e-7)
        chi2 = float(result.fun@result.fun)
        rms = sigma*math.sqrt(chi2/len(t))
        if not result.success or rms > rms_limit:
            continue
        per_observation = sigma*np.linalg.norm(result.fun.reshape(-1, 2), axis=1)
        if np.max(per_observation) > max(3*sigma, 2*rms_limit):
            continue
        if any(np.sqrt(np.mean(per_observation[labels == label]**2)) > rms_limit
               for label in np.unique(labels)):
            continue
        elements = orbital_elements(result.x)
        if elements is None or elements['perihelion_au'] < .005:
            continue
        # Collapse repeated numerical solutions, preserving materially different modes.
        # Compare states at their own epochs through prediction AND physical range.
        distance = float(np.linalg.norm(position(result.x, epoch-seed_epoch)-observers[middle]))
        canonical_position = position(result.x, epoch-seed_epoch)
        canonical_velocity = (position(result.x, epoch-seed_epoch+.001)-
                              position(result.x, epoch-seed_epoch-.001))/.002
        duplicate = next((i for i, (p, v) in enumerate(canonical_states)
                          if np.linalg.norm(canonical_position-p) < max(1e-5, distance*1e-3)
                          and np.linalg.norm(canonical_velocity-v) < 1e-6), None)
        if duplicate is not None:
            if solutions[duplicate]['chi2'] <= chi2:
                continue
            del solutions[duplicate]
            del canonical_states[duplicate]
        canonical_states.append((canonical_position, canonical_velocity))
        solutions.append(dict(state_au_au_per_day=result.x.tolist(), epoch_mjd_tdb=seed_epoch,
                              chi2=chi2, rms_arcsec=rms, range_au=distance,
                              max_residual_arcsec=float(np.max(per_observation)),
                              jacobian_rank=int(np.linalg.matrix_rank(result.jac)),
                              _jacobian=result.jac, **elements))
    if not solutions:
        return None
    solutions.sort(key=lambda s: s['chi2'])
    best = solutions[0]
    k = len(indices)
    null_chi2 = linear_chi2(t, u, labels, sigma)
    # BIC is an asymptotic approximation, not a calibrated false-association probability.
    log_bf = .5*(null_chi2-best['chi2']+(4*k-6)*np.log(2*len(t)))
    log_odds = math.log(prior)-math.log1p(-prior)+log_bf
    logs = np.array([-.5*s['chi2'] for s in solutions])
    for solution, weight in zip(solutions, np.exp(logs-logsumexp(logs))):
        solution['relative_mode_likelihood'] = float(weight)
        solution['local_uncertainty'] = local_uncertainty(
            np.array(solution['state_au_au_per_day']), solution.pop('_jacobian'),
            epoch-solution['epoch_mjd_tdb'], observers[middle])
    return dict(tracklet_indices=sorted(map(int, indices)), observations=len(t),
                span_days=float(t[-1]-t[0]), rms_arcsec=best['rms_arcsec'],
                inclination_deg=best['inclination_deg'], modes=solutions, seeds_attempted=attempts,
                jacobian_backend=jacobian_backend, orbit_start_policy=start_policy,
                full_grid_refinement_pending=start_policy != 'full',
                independent_tracklet_chi2=null_chi2, log_bayes_factor_bic=float(log_bf),
                log_odds_bic=float(log_odds), probability_bic=float(expit(log_odds)),
                probability_kind='uncalibrated BIC approximation; independent linear-tracklet null',
                sigma_arcsec=sigma, prior_link_probability=prior)
