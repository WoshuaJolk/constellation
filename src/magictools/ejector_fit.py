"""Constrained common-origin fits in measured parallax/proper-motion/RV space.

This is a multistart local optimizer, not a discovery probability, exhaustive
catalogue search, or exclusion certificate. Angular positions are held fixed.
Shared objects have one fitted state across every supplied encounter epoch.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .galactic import GalacticPotential, KMS_TO_PC_MYR
from .io import _fields, _named_rows, _provenance, read_json
from .stellar_inputs import icrs_to_galactocentric


class EncounterProblem:
    def __init__(self, spec):
        _fields(spec, ['schema_version', 'epoch_label', 'frame', 'potential', 'objects', 'encounters', 'fit'], ['notes'])
        if spec['schema_version'] != 1 or not isinstance(spec['epoch_label'], str) or not spec['epoch_label']:
            raise ValueError('Require schema_version 1 and a common epoch label')
        _named_rows(spec['objects'], 'objects')
        _named_rows(spec['encounters'], 'encounters')
        self.spec = spec
        self.ids = [r['id'] for r in spec['objects']]
        if not 2 <= len(self.ids) <= 32 or not 1 <= len(spec['encounters']) <= 16:
            raise ValueError('Require 2--32 objects and 1--16 encounters')
        self.n = len(self.ids)
        self.potential = GalacticPotential(spec['potential'])
        settings = spec['fit']
        _fields(settings, ['starts_myr', 'max_step_myr', 'max_sigma', 'min_parallax_mas', 'max_iterations', 'constraint_tolerance_pc'])
        for key in ['max_step_myr', 'max_sigma', 'min_parallax_mas', 'constraint_tolerance_pc']:
            if not np.isfinite(settings[key]) or settings[key] <= 0:
                raise ValueError('Fit scales and tolerances must be finite and positive')
        if not isinstance(settings['max_iterations'], int) or not 1 <= settings['max_iterations'] <= 2000:
            raise ValueError('max_iterations must be 1--2000')
        self.means, self.factors, angles = [], [], []
        for row in spec['objects']:
            _fields(row, ['id', 'ra_deg', 'dec_deg', 'mean', 'covariance'], ['notes'])
            mean, cov = np.asarray(row['mean'], float), np.asarray(row['covariance'], float)
            if mean.shape != (4,) or cov.shape != (4, 4) or not np.isfinite(mean).all() or not np.isfinite(cov).all():
                raise ValueError('Require finite mean (parallax, pmra, pmdec, RV) and 4x4 covariance')
            if mean[0] <= settings['min_parallax_mas'] or not np.allclose(cov, cov.T, rtol=1e-12, atol=1e-14):
                raise ValueError('Require positive observed parallax above bound and symmetric covariance')
            try:
                factor = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError as error:
                raise ValueError('Observation covariance must be positive definite') from error
            if not np.isfinite([row['ra_deg'], row['dec_deg']]).all() or not 0 <= row['ra_deg'] < 360 or not -90 <= row['dec_deg'] <= 90:
                raise ValueError('Invalid angular position')
            self.means.append(mean); self.factors.append(factor)
            angles.append([row['ra_deg'], row['dec_deg']])
        self.means, self.factors = np.array(self.means), np.array(self.factors)
        # Astropy supplies the explicit affine frame rotation. Directions and
        # tangent bases then avoid a coordinate-object allocation per fit step.
        basis = icrs_to_galactocentric(np.array([[0, 0, 1, 0, 0, 0], [90, 0, 1, 0, 0, 0], [0, 90, 1, 0, 0, 0.]]), spec['frame'])
        frame = spec['frame']
        self.observer = np.array([-np.sqrt(frame['galcen_distance_pc']**2-frame['z_sun_pc']**2), 0., frame['z_sun_pc']])
        rotation = (basis[:, :3] - self.observer).T
        self.solar_velocity = np.array(frame['galcen_v_sun_kms'])
        ra, dec = np.deg2rad(angles).T
        self.radial = np.column_stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)]) @ rotation.T
        self.east = np.column_stack([-np.sin(ra), np.cos(ra), np.zeros(self.n)]) @ rotation.T
        self.north = np.column_stack([-np.sin(dec)*np.cos(ra), -np.sin(dec)*np.sin(ra), np.cos(dec)]) @ rotation.T
        self.events, time_bounds, used = [], [], set()
        for event in spec['encounters']:
            _fields(event, ['id', 'members', 'lookback_bounds_myr', 'radius_pc'])
            members = event['members']
            if not isinstance(members, list) or len(members) < 2 or len(set(members)) != len(members) or not set(members) <= set(self.ids):
                raise ValueError('Encounter members must be distinct known objects')
            bounds = np.asarray(event['lookback_bounds_myr'], float)
            radius = event['radius_pc']
            if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1] <= 100:
                raise ValueError('Require positive increasing lookback bounds <= 100 Myr')
            if not np.isfinite(radius) or radius < 0:
                raise ValueError('Encounter radius must be finite and nonnegative')
            steps = int(np.ceil(bounds[1] / settings['max_step_myr']))
            if steps > 10000:
                raise ValueError('Encounter exceeds 10,000 integration steps')
            self.events.append((np.array([self.ids.index(i) for i in members]), radius, steps))
            time_bounds.append(bounds); used.update(members)
        if used != set(self.ids):
            raise ValueError('Every supplied object must participate in an encounter')
        self.starts = np.asarray(settings['starts_myr'], float)
        if self.starts.ndim != 2 or self.starts.shape[1] != len(self.events) or not 1 <= len(self.starts) <= 128 or not np.isfinite(self.starts).all():
            raise ValueError('starts_myr must have 1--128 rows and one column per encounter')
        time_bounds = np.array(time_bounds)
        if np.any(self.starts < time_bounds[:, 0]) or np.any(self.starts > time_bounds[:, 1]):
            raise ValueError('Starting ages lie outside encounter bounds')
        s = settings['max_sigma']
        self.bounds = [(-s, s)] * (4*self.n)
        for i in range(self.n):
            # Cholesky's first row depends only on the first latent coordinate.
            self.bounds[4*i] = (max(-s, (settings['min_parallax_mas']-self.means[i, 0])/self.factors[i, 0, 0]), s)
        self.bounds += [tuple(b) for b in time_bounds]

    def observations(self, x):
        x = np.atleast_2d(x)
        return self.means + np.einsum('nij,bnj->bni', self.factors, x[:, :4*self.n].reshape(-1, self.n, 4))

    def states(self, x):
        obs = self.observations(x)
        distance = 1000 / obs[..., 0]
        position = self.observer + distance[..., None]*self.radial
        velocity = (self.solar_velocity + obs[..., 3, None]*self.radial
                    + 4.740470463533349 * (distance/1000)[..., None]
                    * (obs[..., 1, None]*self.east + obs[..., 2, None]*self.north))
        return np.concatenate([position, velocity], axis=-1)

    def positions(self, x, refinement=1):
        x = np.atleast_2d(x)
        states = self.states(x)
        output = []
        for e, (indices, radius, steps) in enumerate(self.events):
            p = states[:, indices, :3].copy()
            v = states[:, indices, 3:].copy()*KMS_TO_PC_MYR
            steps *= refinement
            h = -x[:, 4*self.n+e, None, None]/steps
            if not self.potential.components:
                p += v*h*steps
            else:
                acceleration = self.potential.acceleration(p)
                for _ in range(steps):
                    v += h*acceleration/2
                    p += h*v
                    acceleration = self.potential.acceleration(p)
                    v += h*acceleration/2
            output.append(p)
        return output

    def constraints(self, x):
        eq, inequality = [], []
        for p, (_, radius, _) in zip(self.positions(x), self.events):
            if radius == 0:
                eq.append((p[:, 1:] - p[:, :1]).reshape(len(p), -1))
            else:
                inequality.append((radius**2 - np.sum((p-p.mean(axis=1, keepdims=True))**2, axis=-1))/max(1., radius))
        batch = len(np.atleast_2d(x))
        return tuple(np.concatenate(v, axis=1) if v else np.empty((batch, 0)) for v in (eq, inequality))

    def constraint_jacobian(self, x, index):
        # Batched central differences advance all perturbed trajectories together.
        delta = 1e-4 * np.maximum(1., np.abs(x))
        bounds = np.array(self.bounds)
        upper = np.minimum(x+delta, bounds[:, 1])
        lower = np.maximum(x-delta, bounds[:, 0])
        plus, minus = np.tile(x, (len(x), 1)), np.tile(x, (len(x), 1))
        np.fill_diagonal(plus, upper); np.fill_diagonal(minus, lower)
        values = self.constraints(np.r_[plus, minus])[index]
        return ((values[:len(x)]-values[len(x):])/(upper-lower)[:, None]).T


def fit_encounters(spec):
    problem = EncounterProblem(spec)
    n = 4*problem.n
    settings = spec['fit']
    constraints = []
    for index, kind in enumerate(['eq', 'ineq']):
        if problem.constraints(np.r_[np.zeros(n), problem.starts[0]])[index].shape[1]:
            constraints.append(dict(type=kind, fun=lambda x, i=index: problem.constraints(x)[i][0],
                                    jac=lambda x, i=index: problem.constraint_jacobian(x, i)))
    trials, solutions = [], []
    for ages in problem.starts:
        x0 = np.r_[np.zeros(n), ages]
        result = minimize(lambda x: np.dot(x[:n], x[:n]), x0,
                          jac=lambda x: np.r_[2*x[:n], np.zeros(len(problem.events))],
                          bounds=problem.bounds, constraints=constraints, method='SLSQP',
                          options=dict(maxiter=settings['max_iterations'], ftol=1e-10))
        coarse = problem.positions(result.x)
        fine = problem.positions(result.x, refinement=2)
        event_reports, violation, numerical_error = [], 0., 0.
        for p, q, event, (_, radius, _) in zip(coarse, fine, spec['encounters'], problem.events):
            center = q[0].mean(axis=0)
            spread = float(np.linalg.norm(q[0]-center, axis=1).max())
            # Exact-origin constraints use separation from the first member.
            residual = float(np.linalg.norm(q[0, 1:]-q[0, :1], axis=1).max()) if radius == 0 else max(0., spread-radius)
            violation = max(violation, residual)
            numerical_error = max(numerical_error, float(np.linalg.norm(p-q, axis=-1).max()))
            event_reports.append(dict(id=event['id'], members=event['members'], origin_pc=center.tolist(), max_centroid_radius_pc=spread, constraint_violation_pc=residual))
        tolerance = settings['constraint_tolerance_pc']
        feasible = bool(violation <= tolerance and numerical_error <= tolerance)
        converged = bool(result.success and feasible)
        trial = dict(start_myr=ages.tolist(), optimizer_success=bool(result.success), message=str(result.message),
                     feasible=feasible, converged=converged, chi2=float(result.fun), iterations=int(result.nit),
                     constraint_violation_pc=violation, half_step_max_position_difference_pc=numerical_error)
        trials.append(trial)
        if converged:
            fitted = problem.observations(result.x)[0]
            for j, row in enumerate(event_reports): row['lookback_myr'] = float(result.x[n+j])
            boundary = [i for i, (v, (lo, hi)) in enumerate(zip(result.x, problem.bounds)) if min(v-lo, hi-v) < 1e-5]
            solutions.append(dict(chi2=float(result.fun), events=event_reports,
                fitted_objects=[dict(id=key, mean=fitted[i].tolist(), marginal_sigma_shift=((fitted[i]-problem.means[i])/np.sqrt(np.diag(problem.factors[i]@problem.factors[i].T))).tolist()) for i, key in enumerate(problem.ids)],
                active_bound_parameter_indices=boundary, constraint_violation_pc=violation,
                half_step_max_position_difference_pc=numerical_error))
    solutions.sort(key=lambda row: row['chi2'])
    return dict(tool='ejector_fit', schema_version=1, epoch_label=spec['epoch_label'],
                status='feasible_local_fit' if solutions else 'no_converged_feasible_fit',
                best_fit=solutions[0] if solutions else None, trials=trials,
                settings=settings, full_objective_complete=False, discovery_probability=None,
                interpretation=['Chi2 is squared Mahalanobis displacement of supplied measurements; no prior or p-value is applied.',
                    'Angular positions are fixed; covariance covers parallax, pmra, pmdec, radial velocity, independently between objects.',
                    'Shared objects have one fitted state across all encounters; separate encounter times allow repeated launches.',
                    'Multistart local optimization is not a global optimum or exclusion certificate; failed fits are inconclusive.',
                    'A feasible encounter is compatibility under the supplied model, not evidence of causal ejection.',
                    'Finite-radius events constrain every member relative to their arithmetic centroid, not a designated source surface.',
                    'Half-step checks estimate numerical error but do not rigorously bound integration error.'])


def run_fit(manifest_path):
    path = Path(manifest_path).resolve()
    result = fit_encounters(read_json(path))
    result['provenance'] = _provenance([path])
    return result
