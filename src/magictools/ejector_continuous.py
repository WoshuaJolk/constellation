"""Sparse, continuous-time co-convergence on piecewise-linear orbit paths.

Completeness refers to the supplied joint draws and their linear interpolants,
not to unknown stars, unsampled posteriors, or the exact Galactic flow.
"""
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .galactic import GalacticPotential, integrate_stars
from .io import _fields, _provenance, read_json


def ball_interval(offset, displacement, radius):
    """Closed interval u in [0,1] where |offset + u*displacement| <= R.

    Vectorized over all leading dimensions. Empty intervals have lo > hi.
    Closest-point form avoids cancellation of the quadratic discriminant.
    Floating-point calculation, not an interval-arithmetic certificate.
    """
    a = np.sum(displacement * displacement, axis=-1)
    moving = a > 0
    center = -np.sum(offset * displacement, axis=-1) / np.where(moving, a, 1)
    closest = offset + center[..., None] * displacement
    slack = radius**2 - np.sum(closest * closest, axis=-1)
    half = np.sqrt(np.maximum(slack, 0) / np.where(moving, a, 1))
    lo, hi = np.maximum(0, center - half), np.minimum(1, center + half)
    inside = np.sum(offset * offset, axis=-1) <= radius**2
    lo = np.where(moving, lo, np.where(inside, 0., 1.))
    hi = np.where(moving, hi, np.where(inside, 1., 0.))
    hi = np.where(moving & (slack < 0), -1., hi)
    return lo, hi


def sparse_pairs(start, end, radius, required, max_pair_checks):
    """Necessary pair gate, including encounters between output epochs.

    A moving point stays in its midpoint sphere of radius half its displacement.
    Thus a close pair must have midpoint separation <= 2R + h_i + h_j.
    KD-tree query uses max(h_j), then exact line-segment closest approach.
    Count each pair once per joint draw, irrespective of encounter duration.
    """
    codes = []
    checks = 0
    n_stars = start.shape[1]
    for p, q in zip(start, end):
        mid = (p + q) / 2
        half = np.linalg.norm(q - p, axis=-1) / 2
        tree = cKDTree(mid)
        bound = np.nextafter(2 * radius + 2 * half.max(), np.inf)
        # Count first so a dense query cannot allocate an unbounded pair array.
        count = (int(tree.count_neighbors(tree, bound)) - n_stars) // 2
        checks += count
        if checks > max_pair_checks:
            raise ValueError('Pair-check cap exceeded; no truncated result is reported')
        pairs = tree.query_pairs(bound, output_type='ndarray')
        if not len(pairs):
            continue
        i, j = pairs.T
        keep = np.linalg.norm(mid[i] - mid[j], axis=-1) <= np.nextafter(2 * radius + half[i] + half[j], np.inf)
        i, j = i[keep], j[keep]
        lo, hi = ball_interval(p[i] - p[j], (q[i] - p[i]) - (q[j] - p[j]), 2 * radius)
        keep = lo <= hi
        codes.append(i[keep].astype(np.int64) * n_stars + j[keep])
    adjacency = defaultdict(set)
    if codes:
        unique, counts = np.unique(np.concatenate(codes), return_counts=True)
        for code in unique[counts >= required]:
            i, j = divmod(int(code), n_stars)
            adjacency[i].add(j)
            adjacency[j].add(i)
    return adjacency, checks


def sparse_cliques(adjacency, size):
    def visit(chosen, remaining):
        if len(chosen) == size:
            yield tuple(chosen)
            return
        while len(remaining) >= size - len(chosen):
            node = min(remaining)
            remaining.remove(node)
            yield from visit(chosen + [node], remaining & adjacency[node])
    yield from visit([], set(adjacency))


def support_regions(lo, hi, required):
    """All closed intervals (including singleton tangencies) with enough draws."""
    valid = lo <= hi
    lo, hi = lo[valid], hi[valid]
    if len(lo) < required:
        return []
    boundaries = np.unique(np.r_[lo, hi])
    regions = []
    for index, left in enumerate(boundaries):
        count = int(np.count_nonzero((lo <= left) & (hi >= left)))
        if count >= required:
            regions.append((float(left), float(left), count))
        if index + 1 < len(boundaries):
            right = boundaries[index + 1]
            mid = (left + right) / 2
            count = int(np.count_nonzero((lo <= mid) & (hi >= mid)))
            if count >= required:
                regions.append((float(left), float(right), count))
    return regions


def scan_segments(segments, times, radius_pc, n_samples, group_size=3, min_support=0.2,
                  max_group_evaluations=1000000, max_pair_checks=20000000):
    """Return every qualifying group and every qualifying interpolated time range.

    segments yields (sample, star, 6) endpoint pairs. Memory is independent of
    time-grid length apart from qualifying results. Caps fail, never truncate.
    """
    times = np.asarray(times, float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError('Lookback grid must start at zero and strictly increase')
    if not np.isfinite(radius_pc) or radius_pc <= 0 or not 0 < min_support <= 1:
        raise ValueError('Positive radius and support in (0,1] required')
    if not isinstance(group_size, int) or not 3 <= group_size <= 6 or n_samples < 1:
        raise ValueError('Require group size 3--6 and positive sample count')
    required = max(1, int(np.ceil(min_support * n_samples - 1e-12)))
    best, ranges = {}, defaultdict(list)
    evaluated = checks = segment_count = 0
    for index, (state0, state1) in enumerate(segments):
        segment_count += 1
        if index >= len(times) - 1:
            raise ValueError('More segments than time intervals')
        if (state0.shape != state1.shape or state0.ndim != 3 or state0.shape[0] != n_samples
                or state0.shape[-1] != 6 or not np.isfinite(state0).all() or not np.isfinite(state1).all()):
            raise ValueError('Segment states must be finite (samples, stars, 6) arrays')
        p, q = state0[..., :3], state1[..., :3]
        adjacency, used = sparse_pairs(p, q, radius_pc, required, max_pair_checks)
        checks += used
        for group in sparse_cliques(adjacency, group_size):
            evaluated += 1
            if evaluated > max_group_evaluations:
                raise ValueError('Group-evaluation cap exceeded; no truncated result is reported')
            a = p[:, group] - p[:, group].mean(axis=1, keepdims=True)
            b = q[:, group] - q[:, group].mean(axis=1, keepdims=True) - a
            low, high = ball_interval(a, b, radius_pc)
            lo, hi = low.max(axis=1), high.min(axis=1)
            regions = support_regions(lo, hi, required)
            if not regions:
                continue
            dt = times[index + 1] - times[index]
            # Within a constant-support region, ensemble RMS is a convex quadratic.
            denominator = np.sum(b * b)
            optimum = -np.sum(a * b) / denominator if denominator else 0.
            for left, right, _ in regions:
                ranges[group].append([float(times[index] + dt * left), float(times[index] + dt * right)])
                u = float(np.clip(optimum, left, right))
                accepted = (lo <= u) & (hi >= u)
                count = int(accepted.sum())
                rms2 = float(np.mean(np.sum((a + u * b)**2, axis=-1)))
                score = count / n_samples + float(np.exp(-rms2 / (2 * radius_pc**2))) / (2 * (n_samples + 1))
                if group in best and score <= best[group]['score']:
                    continue
                points = p[:, group] + u * (q[:, group] - p[:, group])
                velocities = state0[:, group, 3:] + u * (state1[:, group, 3:] - state0[:, group, 3:])
                relative_speed = np.sqrt(np.mean(np.sum((velocities - velocities.mean(axis=1, keepdims=True))**2, axis=-1), axis=1))
                best[group] = dict(indices=group, lookback_myr=float(times[index] + dt * u),
                    relative_rms_speed_given_convergence_percentiles_kms=np.quantile(relative_speed[accepted], [.16,.5,.84]).tolist(),
                    peak_on_search_boundary=bool((index == 0 and u == 0) or (index == len(times)-2 and u == 1)),
                    score=score, support=count / n_samples, support_count=count,
                    ensemble_rms_radius_pc=float(np.sqrt(rms2)),
                    origin_given_convergence_percentiles_pc=np.quantile(points.mean(axis=1)[accepted], [.16, .5, .84], axis=0).tolist())
    if segment_count != len(times) - 1:
        raise ValueError('Fewer segments than time intervals')
    for group, row in best.items():
        merged = []
        for left, right in sorted(ranges[group]):
            if merged and left <= merged[-1][1]:
                merged[-1][1] = max(right, merged[-1][1])
            else:
                merged.append([left, right])
        row['convergence_intervals_myr'] = merged
    return sorted(best.values(), key=lambda x: (-x['score'], x['indices'])), dict(
        group_segment_evaluations=evaluated, broad_phase_pair_checks=checks, segments=segment_count)


def orbit_segments(states, times, potential, max_step, diagnostics, archive=None):
    """Stream fine trajectories, comparing an independently advanced coarse path.

    Midpoint curvature is measured against the chord; it is a diagnostic, not a
    rigorous bound on all subinterval errors. Coarse/fine errors accumulate over
    the full integration (neither path is reset onto the other).
    """
    coarse, fine = states.copy(), states.copy()
    diagnostics.update(half_step_max_position_difference_pc=0., max_midpoint_chord_error_pc=0.)
    if archive is not None:
        archive[0] = fine
    for index, dt in enumerate(np.diff(times)):
        coarse = integrate_stars(coarse, [0., -dt], potential, max_step)[-1]
        refined = integrate_stars(fine, [0., -dt / 2, -dt], potential, max_step / 2)
        end = refined[-1]
        error = float(np.linalg.norm(coarse[..., :3] - end[..., :3], axis=-1).max())
        chord = float(np.linalg.norm(refined[1, ..., :3] - (fine[..., :3] + end[..., :3]) / 2, axis=-1).max())
        diagnostics['half_step_max_position_difference_pc'] = max(error, diagnostics['half_step_max_position_difference_pc'])
        diagnostics['max_midpoint_chord_error_pc'] = max(chord, diagnostics['max_midpoint_chord_error_pc'])
        if archive is not None:
            archive[index + 1] = end
        yield fine, end
        fine = end


def rotate_tangential_velocities(states, observer, angles):
    """Rotate heliocentric tangential velocity, preserving RV and speed per draw.

    One angle per star is shared across its cloud. This conditional null keeps
    the radial selection geometry that whole-velocity permutations destroy.
    It still assumes exchangeable/uniform tangential angles, not a fitted
    Galactic velocity distribution.
    """
    observer = np.asarray(observer, float)
    angles = np.asarray(angles, float)
    if observer.shape != (6,) or not np.isfinite(observer).all() or angles.shape != (states.shape[1],) or not np.isfinite(angles).all():
        raise ValueError('Finite observer six-vector and one rotation angle per star required')
    position = states[..., :3] - observer[:3]
    distance = np.linalg.norm(position, axis=-1)
    if np.any(distance <= 0):
        raise ValueError('A star is at the observer position')
    direction = position / distance[..., None]
    velocity = states[..., 3:] - observer[3:]
    radial = np.sum(velocity * direction, axis=-1)[..., None] * direction
    transverse = velocity - radial
    rotated = transverse * np.cos(angles)[None, :, None] + np.cross(direction, transverse) * np.sin(angles)[None, :, None]
    result = states.copy()
    result[..., 3:] = observer[3:] + radial + rotated
    return result


def run_continuous_ejector(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    _fields(manifest, ['schema_version', 'epoch_label', 'catalogue', 'potential', 'search', 'tracebacks_output'], ['counterparts'])
    if manifest['schema_version'] != 2 or not isinstance(manifest['epoch_label'], str) or not manifest['epoch_label']:
        raise ValueError('Require schema_version 2 and common epoch label')
    source = (path.parent / manifest['catalogue']).resolve()
    with np.load(source, allow_pickle=False) as data:
        states = data['states_pc_kms'].copy()
        ids = data['source_ids'].tolist()
        strata = data['control_strata'].tolist() if 'control_strata' in data else ['all'] * len(ids)
        rotation_groups = data['rotation_groups'].tolist() if 'rotation_groups' in data else None
    if (states.ndim != 3 or len(states) < 1 or states.shape[1:] != (len(ids), 6) or len(ids) < 3
            or not np.isfinite(states).all() or len(ids) != len(set(ids))
            or any(not isinstance(x, str) or not x for x in ids)
            or len(strata) != len(ids) or any(not isinstance(x, str) or not x for x in strata)):
        raise ValueError('Catalogue needs finite joint draws, unique string IDs and valid control strata')
    rotation_inverse = None
    if rotation_groups is not None:
        if len(rotation_groups) != len(ids) or any(not isinstance(x, str) or not x for x in rotation_groups):
            raise ValueError('rotation_groups needs a nonempty string per star')
        rotation_labels, rotation_inverse = np.unique(rotation_groups, return_inverse=True)
    counterpart_source = None
    counterpart_states, counterpart_ids = None, []
    if 'counterparts' in manifest:
        counterpart_source = (path.parent / manifest['counterparts']).resolve()
        with np.load(counterpart_source, allow_pickle=False) as data:
            counterpart_states = data['states_pc_kms'].copy()
            counterpart_ids = data['source_ids'].tolist()
        if (counterpart_states.shape != (len(states), len(counterpart_ids), 6)
                or not np.isfinite(counterpart_states).all()
                or len(set(counterpart_ids)) != len(counterpart_ids)
                or any(not isinstance(x, str) or not x for x in counterpart_ids)
                or set(ids) & set(counterpart_ids)):
            raise ValueError('Counterparts need distinct IDs and matching finite joint draws')
    settings = manifest['search']
    _fields(settings, ['lookback_max_myr', 'time_step_myr', 'radius_pc'],
            ['group_size', 'min_support', 'max_step_myr', 'n_controls', 'seed', 'max_group_evaluations', 'max_pair_checks', 'control_method', 'observer_state_pc_kms'])
    settings = {'group_size': 3, 'min_support': .2, 'max_step_myr': .02,
                'n_controls': 0, 'seed': 2026, 'control_method': 'velocity_permutation', 'max_group_evaluations': 1000000,
                'max_pair_checks': 20000000, **settings}
    for key in ['lookback_max_myr', 'time_step_myr', 'radius_pc', 'max_step_myr']:
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key, low, high in [('n_controls', 0, 9999), ('seed', 0, 2**32-1), ('group_size', 3, min(6, len(ids))),
                           ('max_group_evaluations', 1, 100000000), ('max_pair_checks', 1, 1000000000)]:
        if isinstance(settings[key], bool) or not isinstance(settings[key], int) or not low <= settings[key] <= high:
            raise ValueError(f'Invalid {key}')
    if not 0 < settings['min_support'] <= 1:
        raise ValueError('min_support must be in (0,1]')
    if settings['control_method'] not in {'velocity_permutation', 'tangent_rotation'}:
        raise ValueError('Unknown control_method')
    if settings['control_method'] == 'tangent_rotation':
        rotate_tangential_velocities(states, settings.get('observer_state_pc_kms'), np.zeros(len(ids)))
    count = int(np.ceil(settings['lookback_max_myr'] / settings['time_step_myr']))
    if count > 100000 or settings['lookback_max_myr'] > 100:
        raise ValueError('Maximum 100 Myr and 100000 output intervals')
    times = np.linspace(0, settings['lookback_max_myr'], count + 1)
    potential = GalacticPotential(manifest['potential'])
    groups = [np.flatnonzero(np.asarray(strata) == name) for name in sorted(set(strata))]
    if settings['n_controls'] and settings['control_method'] == 'velocity_permutation' and all(len(g) == 1 for g in groups):
        raise ValueError('All control strata are singletons')
    destination = (path.parent / manifest['tracebacks_output']).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    # A completed index is written only after every observed/control search succeeds.
    archive = np.lib.format.open_memmap(destination / 'states.npy', mode='w+', dtype='float64', shape=(len(times),) + states.shape)
    np.save(destination / 'lookback_myr.npy', times)
    np.save(destination / 'source_ids.npy', np.asarray(ids))
    diagnostic = {}
    options = {k: settings[k] for k in ['radius_pc', 'group_size', 'min_support', 'max_group_evaluations', 'max_pair_checks']}
    rows, counts = scan_segments(orbit_segments(states, times, potential, settings['max_step_myr'], diagnostic, archive),
                                times, n_samples=len(states), **options)
    archive.flush()
    del archive
    def check_numerics(d):
        if max(d.values()) > .1 * settings['radius_pc']:
            raise ValueError('Numerical trajectory/chord diagnostic exceeds 10% of radius; reduce integration/output steps')
    check_numerics(diagnostic)
    maxima, control_diagnostics = [], []
    rng = np.random.default_rng(settings['seed'])
    for _ in range(settings['n_controls']):
        if settings['control_method'] == 'tangent_rotation':
            angles = (rng.uniform(0, 2*np.pi, len(rotation_labels))[rotation_inverse]
                      if rotation_inverse is not None else rng.uniform(0, 2*np.pi, len(ids)))
            scrambled = rotate_tangential_velocities(states, settings['observer_state_pc_kms'], angles)
        else:
            permutation = np.arange(len(ids))
            for indices in groups:
                permutation[indices] = rng.permutation(indices)
            scrambled = states.copy()
            scrambled[..., 3:] = states[:, permutation, 3:]
        d = {}
        control, _ = scan_segments(orbit_segments(scrambled, times, potential, settings['max_step_myr'], d),
                                  times, n_samples=len(states), **options)
        check_numerics(d)
        control_diagnostics.append(d)
        maxima.append(control[0]['score'] if control else 0.)
    for row in rows:
        row['supplied_counterpart_comparisons'] = []
        if counterpart_states is not None and len(counterpart_ids):
            t = row['lookback_myr']
            j = min(int(np.searchsorted(times, t, side='right') - 1), len(times) - 2)
            u = (t - times[j]) / (times[j+1] - times[j])
            saved = np.load(destination / 'states.npy', mmap_mode='r')
            points = saved[j][:, row['indices'], :3] * (1-u) + saved[j+1][:, row['indices'], :3] * u
            centers = points.mean(axis=1)
            accepted = np.linalg.norm(points - centers[:, None], axis=-1).max(axis=1) <= np.nextafter(settings['radius_pc'], np.inf)
            comparison = (integrate_stars(counterpart_states, [0., -t], potential, settings['max_step_myr']/2)[-1]
                          if t else counterpart_states)
            for k, name in enumerate(counterpart_ids):
                separation = np.linalg.norm(comparison[accepted, k, :3] - centers[accepted], axis=-1)
                row['supplied_counterpart_comparisons'].append(dict(id=name,
                    separation_percentiles_pc=np.quantile(separation, [.16, .5, .84]).tolist() if len(separation) else None,
                    conditional_proximity_fraction=float(np.mean(separation <= settings['radius_pc'])) if len(separation) else None))
        row['present_close_comoving_pairs'] = []
        for i,j in combinations(row['indices'], 2):
            separation = np.linalg.norm(states[:, i, :3] - states[:, j, :3], axis=-1)
            relative_speed = np.linalg.norm(states[:, i, 3:] - states[:, j, 3:], axis=-1)
            fraction = float(np.mean((separation < 1.) & (relative_speed < 5.)))
            if fraction > 0:
                row['present_close_comoving_pairs'].append(dict(star_ids=[ids[i], ids[j]],
                    sample_fraction_within_1pc_and_5kms=fraction,
                    separation_percentiles_pc=np.quantile(separation,[.16,.5,.84]).tolist(),
                    relative_speed_percentiles_kms=np.quantile(relative_speed,[.16,.5,.84]).tolist()))
        row['star_ids'] = [ids[i] for i in row.pop('indices')]
        row['search_wide_control_rank_fraction'] = ((1 + sum(x >= row['score'] for x in maxima)) / (len(maxima) + 1)) if maxima else None
        row['search_wide_velocity_permutation_p'] = row['search_wide_control_rank_fraction'] if settings['control_method'] == 'velocity_permutation' else None
    result = dict(schema_version=2, tool='find_ejector', provenance=_provenance([path, source] + ([counterpart_source] if counterpart_source else [])),
        epoch_label=manifest['epoch_label'], potential=manifest['potential'], settings=settings,
        search_summary=dict(catalogue_size=len(ids), n_samples=len(states), qualifying_groups=len(rows), **counts),
        candidates=rows, numerics=diagnostic, tracebacks_directory=str(destination),
        controls=dict(method=settings['control_method'], max_search_scores=maxima, numerics=control_diagnostics,
            rotation_groups_supplied=rotation_groups is not None,
            rotation_group_count=len(set(rotation_groups)) if rotation_groups is not None else len(ids),
            assumption=('Uniform/exchangeable heliocentric tangential angles at fixed positions, RVs and speeds; not a Galactic population model'
                        if settings['control_method'] == 'tangent_rotation' else
                        'Velocity clouds exchangeable within supplied strata; destroys position/velocity covariance; not an astrophysical false-discovery probability')),
        completeness='Every fixed-size group and every qualifying time interval in supplied joint draws of piecewise-linear paths, subject to floating-point arithmetic',
        full_catalogue_objective_complete=False,
        interpretation='Kinematic candidates only. No proof of ejection, no counterpart survey, no dark-source claim; unknown RV and model priors belong in catalogue provenance.')
    import json
    (destination / 'complete.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
