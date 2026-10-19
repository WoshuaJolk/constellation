"""Catalogue-wide profile fits, complete-search controls, and source tracebacks.

Every combination in the supplied finite family is attempted. Local fits do
not certify that every compatible origin was found. Controls measure a declared
randomization model, not the Galactic population or a causal ejection rate.
"""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
from itertools import combinations
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .ejector_fit import EncounterProblem, fit_encounters
from .galactic import GalacticPotential, KMS_TO_PC_MYR, integrate_stars
from .io import _fields, _named_rows, _provenance, read_json
from .stellar_inputs import icrs_to_galactocentric


def rotated_catalogue(objects, rng):
    """Rotate each proper-motion vector AND its complete covariance coherently."""
    rows = deepcopy(objects)
    for row, angle in zip(rows, rng.uniform(0, 2*np.pi, len(rows))):
        transform = np.eye(4)
        c, s = np.cos(angle), np.sin(angle)
        transform[1:3, 1:3] = [[c, -s], [s, c]]
        row['mean'] = (transform@np.asarray(row['mean'])).tolist()
        row['covariance'] = (transform@np.asarray(row['covariance'])@transform.T).tolist()
    return rows


def family_spec(spec, indices, objects=None):
    objects = spec['objects'] if objects is None else objects
    selected = [objects[i] for i in indices]
    search = spec['search']
    return dict(schema_version=1, epoch_label=spec['epoch_label'], frame=spec['frame'],
                potential=spec['potential'], objects=selected, fit=spec['fit'],
                encounters=[dict(id='common_origin', members=[r['id'] for r in selected],
                                 lookback_bounds_myr=search['lookback_bounds_myr'], radius_pc=search['radius_pc'])])


def validate_search(spec):
    _fields(spec, ['schema_version', 'epoch_label', 'frame', 'potential', 'objects', 'search', 'fit',
                   'controls', 'archive_output', 'catalogue_description'], ['counterparts'])
    if spec['schema_version'] != 1 or not isinstance(spec['catalogue_description'], str) or not spec['catalogue_description']:
        raise ValueError('Require schema 1 and a nonempty description of catalogue selection')
    _named_rows(spec['objects'], 'objects')
    search = spec['search']
    _fields(search, ['group_size', 'lookback_bounds_myr', 'radius_pc', 'report_chi2_max', 'max_hypotheses', 'workers'])
    for key, lo, hi in [('group_size', 3, min(6, len(spec['objects']))), ('max_hypotheses', 1, 100000), ('workers', 1, 32)]:
        if type(search[key]) is not int or not lo <= search[key] <= hi:
            raise ValueError('Invalid '+key)
    if not np.isfinite(search['report_chi2_max']) or search['report_chi2_max'] < 0:
        raise ValueError('report_chi2_max must be finite and nonnegative')
    count = comb(len(spec['objects']), search['group_size'])
    if count > search['max_hypotheses']:
        raise ValueError(f'Family has {count} hypotheses, exceeding explicit limit; no truncated search is performed')
    _fields(spec['controls'], ['method', 'n_controls', 'seed'])
    c = spec['controls']
    if c['method'] != 'proper_motion_rotation':
        raise ValueError('Supported control method: proper_motion_rotation')
    for key, hi in [('n_controls', 9999), ('seed', 2**32-1)]:
        if type(c[key]) is not int or not 0 <= c[key] <= hi:
            raise ValueError('Invalid '+key)
    # Validate EVERY row before creating output, even if it is not in group zero.
    size = search['group_size']
    for start in range(0, len(spec['objects']), size):
        indices = list(range(start, min(start+size, len(spec['objects']))))
        if len(indices) < size:
            indices += list(range(size-len(indices)))
        EncounterProblem(family_spec(spec, indices))
    return count


def _fit_job(task):
    indices, fit_spec, threshold = task
    result = fit_encounters(fit_spec)
    best = result['best_fit']
    ledger = dict(indices=list(indices), status=result['status'], chi2=best['chi2'] if best else None,
                  converged_starts=sum(t['converged'] for t in result['trials']),
                  starts=len(result['trials']), iterations=[t['iterations'] for t in result['trials']],
                  optimizer_messages=[t['message'] for t in result['trials']],
                  active_bounds=best['active_bound_parameter_indices'] if best else [])
    return ledger, best if best and best['chi2'] <= threshold else None


def search_family(spec, objects=None, pool=None):
    objects = spec['objects'] if objects is None else objects
    indices = combinations(range(len(objects)), spec['search']['group_size'])
    tasks = ((i, family_spec(spec, i, objects), spec['search']['report_chi2_max']) for i in indices)
    outputs = map(_fit_job, tasks) if pool is None else pool.map(_fit_job, tasks, chunksize=1)
    ledger, candidates = [], []
    for row, best in outputs:
        ledger.append(row)
        if best is not None:
            candidates.append(dict(indices=row['indices'], star_ids=[objects[i]['id'] for i in row['indices']], best_fit=best))
    candidates.sort(key=lambda r: r['best_fit']['chi2'])
    costs = [r['chi2'] for r in ledger if r['chi2'] is not None]
    return ledger, candidates, min(costs) if costs else None


def control_rank(cost, minimum_costs):
    """Search-wide randomization rank; ties count against the observed result."""
    if not minimum_costs:
        return None
    tolerance = 1e-8*max(1., abs(cost))
    return (1+sum(v is not None and v <= cost+tolerance for v in minimum_costs))/(len(minimum_costs)+1)


def validate_candidate(spec, candidate):
    """Reintegrate the fitted state using independent adaptive DOP853."""
    problem = EncounterProblem(family_spec(spec, candidate['indices']))
    best = candidate['best_fit']
    z = np.concatenate([np.linalg.solve(f, np.asarray(row['mean'])-m)
                        for row, m, f in zip(best['fitted_objects'], problem.means, problem.factors)])
    age = best['events'][0]['lookback_myr']
    states = problem.states(np.r_[z, age])[0]
    def rhs(t, y):
        y = y.reshape(-1, 6)
        return np.c_[y[:, 3:]*KMS_TO_PC_MYR, problem.potential.acceleration(y[:, :3])/KMS_TO_PC_MYR].ravel()
    solution = solve_ivp(rhs, (0., -age), states.ravel(), method='DOP853', rtol=1e-11, atol=1e-11)
    if not solution.success:
        raise ValueError('Independent candidate integration failed')
    historical = solution.y[:, -1].reshape(-1, 6)
    p = historical[:, :3]
    radius = float(np.linalg.norm(p-p.mean(axis=0), axis=1).max())
    if spec['search']['radius_pc'] == 0:
        residual = float(np.linalg.norm(p[1:]-p[:1], axis=1).max())
    else:
        residual = max(0., radius-spec['search']['radius_pc'])
    velocities = historical[:, 3:]
    return dict(passed=bool(residual <= spec['fit']['constraint_tolerance_pc']),
                constraint_violation_pc=residual, max_centroid_radius_pc=radius,
                origin_pc=p.mean(axis=0).tolist(),
                relative_rms_speed_kms=float(np.sqrt(np.mean(np.sum((velocities-velocities.mean(axis=0))**2, axis=1)))),
                member_velocities_kms=velocities.tolist(),
                interpretation='Independent numerical check of this fitted trajectory, not causal confirmation')


def load_counterparts(path, frame):
    data = read_json(path)
    _fields(data, ['schema_version', 'epoch_label', 'description', 'coverage', 'objects', 'excluded'])
    if data['schema_version'] != 1:
        raise ValueError('Counterpart schema must be 1')
    _named_rows(data['objects'], 'counterparts')
    values = []
    for row in data['objects']:
        _fields(row, ['id', 'icrs', 'match_radius_pc'], ['metadata'])
        if not np.isfinite(row['match_radius_pc']) or row['match_radius_pc'] <= 0:
            raise ValueError('Counterpart match radius must be finite and positive')
        values.append(row['icrs'])
    states = icrs_to_galactocentric(np.asarray(values, float), frame)
    return data, states


def match_counterparts(candidate, data, states, potential, max_step):
    """Nominal source orbits only: no fabricated covariance or source probability."""
    age = candidate['best_fit']['events'][0]['lookback_myr']
    origin = np.array(candidate['independent_validation']['origin_pc'])
    past = integrate_stars(states, [0., -age], potential, max_step/2)[-1]
    separation = np.linalg.norm(past[:, :3]-origin, axis=1)
    limits = np.array([r['match_radius_pc'] for r in data['objects']])
    inside = np.flatnonzero(separation <= limits)
    nearest = np.argsort(separation)[:min(5, len(states))]
    def row(i):
        obj = data['objects'][i]
        return dict(id=obj['id'], separation_pc=float(separation[i]), match_radius_pc=obj['match_radius_pc'],
                    within_radius=bool(separation[i] <= limits[i]), present_icrs=obj['icrs'],
                    historical_velocity_kms=past[i, 3:].tolist(), metadata=obj.get('metadata', {}))
    return dict(matches=[row(i) for i in inside[np.argsort(separation[inside])]], nearest=[row(i) for i in nearest],
                objects_tested=len(states), excluded_count=len(data['excluded']), coverage=data['coverage'],
                interpretation='Nominal orbit proximity to fitted origin. Cluster extent is a match scale, not astrometric uncertainty. No-match is not a dark source.')


def run_search(manifest_path):
    path = Path(manifest_path).resolve()
    spec = read_json(path)
    count = validate_search(spec)
    inputs = [path]
    sources, source_states = None, None
    if 'counterparts' in spec:
        source_path = (path.parent/spec['counterparts']).resolve()
        sources, source_states = load_counterparts(source_path, spec['frame'])
        if sources['epoch_label'] != spec['epoch_label']:
            raise ValueError('Counterpart and stellar epochs must match exactly; convert epochs upstream')
        inputs.append(source_path)
    output = (path.parent/spec['archive_output']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        with (output/name).open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    save('manifest.json', spec)
    potential = GalacticPotential(spec['potential'])
    # A reusable nominal orbit library; uncertainty belongs to the fit likelihood,
    # not a label falsely implying these single paths are posterior ensembles.
    obs = np.array([[r['ra_deg'], r['dec_deg'], 1000/r['mean'][0], *r['mean'][1:]] for r in spec['objects']])
    states = icrs_to_galactocentric(obs, spec['frame'])
    times = np.linspace(0., spec['search']['lookback_bounds_myr'][1], 121)
    np.savez_compressed(output/'nominal_tracebacks.npz', source_ids=np.array([r['id'] for r in spec['objects']]),
                        lookback_myr=times, states_pc_kms=integrate_stars(states, -times, potential, spec['fit']['max_step_myr']/2))
    rng = np.random.default_rng(spec['controls']['seed'])
    minima = []
    with ProcessPoolExecutor(spec['search']['workers']) as pool:
        ledger, candidates, minimum = search_family(spec, pool=pool)
        save('observed_fits.json', ledger)
        for i in range(spec['controls']['n_controls']):
            control, _, score = search_family(spec, rotated_catalogue(spec['objects'], rng), pool)
            save(f'control_{i:04d}.json', control)
            minima.append(score)
    for candidate in candidates:
        candidate['search_wide_rotation_rank'] = control_rank(candidate['best_fit']['chi2'], minima)
        candidate['independent_validation'] = validate_candidate(spec, candidate)
        candidate['counterparts'] = (match_counterparts(candidate, sources, source_states, potential, spec['fit']['max_step_myr'])
                                     if sources is not None and candidate['independent_validation']['passed'] else None)
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
    result = dict(tool='ejector_search', schema_version=1, provenance=_provenance(inputs), source_code_sha256=code,
        catalogue_description=spec['catalogue_description'], catalogue_size=len(spec['objects']), epoch_label=spec['epoch_label'],
        hypotheses_per_catalogue=count, catalogues_searched=1+len(minima), minimum_chi2=minimum,
        converged_groups=sum(r['chi2'] is not None for r in ledger), reported_candidates=len(candidates), candidates=candidates,
        search=spec['search'], fit=spec['fit'], potential=spec['potential'],
        controls=dict(**spec['controls'], minimum_chi2=minima, minimum_rank_resolution=1/(len(minima)+1) if minima else None,
                      assumption='Proper-motion position angles are independently uniform at fixed sky position, parallax, RV and proper-motion magnitude; covariances rotate with the vector. This is NOT established for the real selected catalogue.'),
        archive=str(output), nominal_tracebacks='nominal_tracebacks.npz', full_objective_complete=False,
        discovery_probability=None,
        completion=dict(all_declared_combinations_attempted=True, all_requested_controls_completed=True,
                        all_reported_candidates_independently_validated=all(c['independent_validation']['passed'] for c in candidates),
                        all_survey_input_available=False, globally_exhaustive_uncertainty_search=False,
                        astrophysical_false_association_calibration=False, source_visibility_complete=False, causal_confirmation=False),
        interpretation=[
            'Each combination is fitted at every declared start; failed local fits are inconclusive, not excluded origins.',
            'Low chi2 is measurement compatibility, not encounter posterior mass or evidence of an ejection mechanism.',
            'Control ranks include every group and age fit in this run, not choices made across different runs or models.',
            'Rotation controls are model diagnostics, not astrophysical false-alarm probabilities; literature-selected samples violate simple exchangeability.',
            'Source matching uses supplied nominal present states propagated backward. No counterpart is never proof of a dark ejector.',
            'The archived nominal paths do not marginalize over measurement, frame or Galactic-potential uncertainty.'])
    # No completion sentinel is emitted until every search and follow-up finishes.
    save('complete.json', result)
    return result
