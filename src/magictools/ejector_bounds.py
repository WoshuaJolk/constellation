"""Validated continuous-time outer enclosures for a bounded encounter search.

Arb ball arithmetic encloses the smooth ODE with Picard tubes and integral
endpoint bounds. Empty intersections can exclude groups; nonempty intersections
are unresolved. The domain is fixed-angle, finite-cost measurement space in a
specified frame and potential, not all physical histories of the real stars.
"""
from itertools import combinations
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .ejector_fit import EncounterProblem, fit_encounters
from .ejector_search import family_spec, validate_search, load_counterparts, match_counterparts
from .galactic import G_PC_MYR, KMS_TO_PC_MYR, GalacticPotential
from .io import _provenance, read_json


def _arb():
    try:
        from flint import arb, arb_mat, ctx
    except ImportError as error:
        raise ImportError('ejector_bound requires python-flint; install magictools-astro[research]') from error
    return arb, arb_mat, ctx


def initial_boxes(spec):
    arb, arb_mat, _ = _arb()
    size = spec['search']['group_size']
    # The fixed float64 frame coefficients are explicit inputs to this bounded
    # model. Interval arithmetic does not claim to certify Astropy's rotation.
    frame_coefficients, boxes = [], []
    budget = arb(spec['search']['report_chi2_max'])
    for row in spec['objects']:
        single = dict(spec)
        single['objects'] = [row, {**row, 'id': row['id']+'__basis_only'}]
        local = family_spec(single, [0, 1])
        p = EncounterProblem(local)
        covariance = np.asarray(row['covariance'], float)
        if not np.array_equal(covariance, covariance.T):
            raise ValueError('Certified domain needs exactly symmetric covariance')
        for k in range(1, 5):
            if not arb_mat(covariance[:k, :k].tolist()).det() > 0:
                raise ValueError('Arb could not certify positive-definite observation covariance')
        intervals = [arb(m)+arb(0, (budget*arb(covariance[j, j])).sqrt()) for j, m in enumerate(row['mean'])]
        # Group total Mahalanobis cost <= C implies every marginal displacement
        # <= sqrt(C * C_ii), regardless of within-star correlations.
        parallax = intervals[0]
        floor = arb(spec['fit']['min_parallax_mas'])
        if parallax.lower() < floor:
            parallax = floor.union(parallax.upper())
        if not parallax.lower() > 0:
            raise ValueError('Parallax enclosure must stay strictly positive')
        distance = arb(1000)/parallax
        position, velocity = [], []
        for axis in range(3):
            position.append(arb(p.observer[axis])+distance*arb(p.radial[0, axis]))
            speed = (arb(p.solar_velocity[axis])+intervals[3]*arb(p.radial[0, axis])
                     + arb(4.740470463533349)/parallax*(intervals[1]*arb(p.east[0, axis])+intervals[2]*arb(p.north[0, axis])))
            velocity.append(speed*arb(KMS_TO_PC_MYR))
        boxes.append(position+velocity)
        frame_coefficients.append(dict(id=row['id'], observer_pc=p.observer.tolist(), solar_velocity_kms=p.solar_velocity.tolist(),
            radial=p.radial[0].tolist(), east=p.east[0].tolist(), north=p.north[0].tolist()))
    return boxes, frame_coefficients


def acceleration(position, specification):
    arb, _, _ = _arb()
    x, y, z = position
    force = [arb(0), arb(0), arb(0)]
    for c in specification['components']:
        direction = [x, y, z]
        if c['type'] == 'plummer':
            soft = (x*x+y*y+z*z+arb(c['scale_pc'])**2).sqrt()
            factor = arb(G_PC_MYR)*arb(c['mass_msun'])/(soft**3)
        elif c['type'] == 'miyamoto_nagai':
            soft = (z*z+arb(c['b_pc'])**2).sqrt()
            height = arb(c['a_pc'])+soft
            denominator = (x*x+y*y+height*height).sqrt()**3
            direction[2] = z*height/soft
            factor = arb(G_PC_MYR)*arb(c['mass_msun'])/denominator
        else:
            factor = (arb(c['speed_kms'])*arb(KMS_TO_PC_MYR))**2/(x*x+y*y+z*z+arb(c['core_pc'])**2)
        for axis in range(3): force[axis] -= factor*direction[axis]
    if not all(f.is_finite() for f in force):
        raise ValueError('Force enclosure became nonfinite; shorten integration steps or tighten domain')
    return force


def validated_step(state, h, potential, max_iterations=32):
    """Enclose every trajectory from state throughout [h, 0], then its endpoint.

    A Picard self-inclusion encloses the entire continuous path. Endpoint
    positions use x0+h*v0+integral acceleration, with its positive h²/2 weight;
    velocities use v0+integral acceleration. No sampled-path assumption is used.
    """
    arb, _, _ = _arb()
    span = arb(0).union(h)
    derivative = state[3:]+acceleration(state[:3], potential)
    tube = [s+span*f for s, f in zip(state, derivative)]
    for _ in range(max_iterations):
        tube = [v+arb(0, (v.rad()+arb('1e-20'))/8) for v in tube]
        force = acceleration(tube[:3], potential)
        images = [s+span*f for s, f in zip(state, tube[3:]+force)]
        if all(t.contains(v) for t, v in zip(tube, images)):
            endpoint = [state[j]+h*state[j+3]+h*h*force[j]/2 for j in range(3)]
            endpoint += [state[j+3]+h*force[j] for j in range(3)]
            if not all(v.is_finite() for v in endpoint):
                raise ValueError('Nonfinite endpoint enclosure')
            return tube, endpoint
        tube = [t.union(v) for t, v in zip(tube, images)]
    raise ValueError('Picard enclosure did not close; reduce max_step_myr. No exclusion certificate is issued.')


def outward_float_bounds(value):
    """Prove converted binary64 endpoints enclose the Arb ball before use."""
    arb, _, _ = _arb()
    lower, upper = value.lower(), value.upper()
    lo, hi = np.nextafter(float(lower), -np.inf), np.nextafter(float(upper), np.inf)
    if not np.isfinite([lo, hi]).all():
        raise ValueError('Nonfinite float enclosure')
    # Usually one outward ulp suffices. The checked comparisons are the guarantee,
    # rather than an assumption about a ball-to-float conversion's rounding.
    while not arb(lo) <= lower: lo = np.nextafter(lo, -np.inf)
    while not arb(hi) >= upper: hi = np.nextafter(hi, np.inf)
    return lo, hi


def bound_family(spec):
    validate_search(spec)
    arb, _, ctx = _arb()
    old_precision = ctx.prec
    ctx.prec = 128
    try:
        states, coefficients = initial_boxes(spec)
        age_max = spec['search']['lookback_bounds_myr'][1]
        steps = int(np.ceil(age_max/spec['fit']['max_step_myr']))
        h = -arb(age_max)/steps
        radius = arb(spec['search']['radius_pc'])
        groups = np.array(list(combinations(range(len(states)), spec['search']['group_size'])), dtype=int)
        unresolved = np.zeros(len(groups), dtype=bool)
        first_possible = np.full(len(groups), -1, dtype=int)
        bounds = np.empty((steps, len(states), 3, 2))
        enclosure_failures = []
        for k in range(steps):
            for i, state in enumerate(states):
                if state is None:
                    bounds[k, i] = [-np.inf, np.inf]
                    continue
                try:
                    tube, states[i] = validated_step(state, h, spec['potential'])
                    for axis in range(3):
                        bounds[k, i, axis] = outward_float_bounds(tube[axis]+arb(0, radius))
                except ValueError as error:
                    # An unbounded outer set is safe. A failed enclosure must
                    # never become a discarded star or a false exclusion.
                    bounds[k, i] = [-np.inf, np.inf]
                    states[i] = None
                    enclosure_failures.append(dict(id=spec['objects'][i]['id'], segment=k, reason=str(error)))
            # Include a segment touching the lower age bound conservatively.
            if arb(age_max)*(k+1)/steps < arb(spec['search']['lookback_bounds_myr'][0]):
                continue
            selected = bounds[k, groups]
            possible = np.all(selected[..., 0].max(axis=1) <= selected[..., 1].min(axis=1), axis=1)
            first_possible[possible & ~unresolved] = k
            unresolved |= possible
        ids = [r['id'] for r in spec['objects']]
        def records(mask):
            return [dict(star_ids=[ids[i] for i in groups[j]], first_possible_segment=int(first_possible[j]) if mask[j] else None)
                    for j in range(len(groups)) if mask[j]]
        result = dict(tool='ejector_bound', schema_version=1, hypotheses=len(groups),
            excluded_groups=[[ids[i] for i in g] for g in groups[~unresolved]], unresolved_groups=records(unresolved),
            excluded_count=int((~unresolved).sum()), unresolved_count=int(unresolved.sum()),
            lookback_bounds_myr=spec['search']['lookback_bounds_myr'], radius_pc=spec['search']['radius_pc'],
            total_mahalanobis_cost_at_most=spec['search']['report_chi2_max'], steps=steps, arb_precision_bits=128,
            unbounded_tube_fallbacks=enclosure_failures,
            galactic_gravity_constant_pc3_msun_myr2=G_PC_MYR, kms_to_pc_myr=KMS_TO_PC_MYR,
            fixed_frame_coefficients=coefficients, potential=spec['potential'],
            statement='For excluded groups, no continuous-time centroid-radius encounter exists in the stated age interval and bounded measurement domain under this fixed smooth ODE model.',
            domain=['Fixed observed RA/Dec; exact supplied symmetric Gaussian covariance; total group Mahalanobis cost <= the declared bound.',
                    'Parallax >= min_parallax_mas; per-coordinate max_sigma fit bounds are relaxed, enlarging the domain.',
                    'Fixed supplied Galactic potential, unit constants and float64 frame coefficients recorded here; no frame/potential uncertainty.',
                    'Outer marginal observation boxes enlarge the correlated cost ellipsoid. Empty common-center box intersection excludes; overlap never proves an encounter.'],
            method='128-bit Arb arithmetic, Picard self-inclusion over every full time segment, integral endpoint bounds, checked outward binary64 conversion, and necessary common-center box intersections.',
            limitations=['This is a model-conditional exclusion, not a proof about all physical histories or a new mathematical theorem.',
                         'Unresolved groups may be impossible; interval wrapping and ignored covariance/total-cost coupling can be conservative.',
                         'The implementation and Arb library are trusted numerical software; no separate formal proof assistant verifies this program.'],
            full_objective_complete=False)
        return result, bounds
    finally:
        ctx.prec = old_precision


def run_bounds(manifest_path, max_refinement_nodes=0, verify_witnesses=False):
    if type(max_refinement_nodes) is not int or not 0 <= max_refinement_nodes <= 1000000:
        raise ValueError('Refinement node limit must be between 0 and 1,000,000')
    path = Path(manifest_path).resolve()
    spec = read_json(path)
    destination = (path.parent/(spec['archive_output']+'-bounds')).resolve()
    if destination.exists(): raise FileExistsError('Bounds output already exists')
    result, bounds = bound_family(spec)
    result['refinements'] = []
    result['certified_witnesses'] = []
    remaining = []
    for group in result['unresolved_groups']:
        members = group['star_ids']
        if verify_witnesses and spec['search']['radius_pc'] > 0:
            indices = [next(i for i, row in enumerate(spec['objects']) if row['id'] == key) for key in members]
            fitted_spec = family_spec(spec, indices)
            # An exact-center fit is only a proposal. A positive-radius witness
            # is accepted only after independent interval ODE/cost verification.
            fitted_spec['encounters'][0]['radius_pc'] = 0.
            fitted = fit_encounters(fitted_spec)['best_fit']
            if fitted is not None and fitted['chi2'] <= spec['search']['report_chi2_max']:
                witness = certify_witness(spec, members, fitted)
                if witness['certified']:
                    result['certified_witnesses'].append(witness)
                    continue
        if max_refinement_nodes:
            refined = refine_group(spec, members, max_refinement_nodes)
            result['refinements'].append(refined)
            if refined['excluded']:
                result['excluded_groups'].append(members)
                continue
        remaining.append(group)
    result['unresolved_groups'] = remaining
    result['excluded_count'] = len(result['excluded_groups'])
    result['unresolved_count'] = len(remaining)
    result['certified_compatible_count'] = len(result['certified_witnesses'])
    result['all_groups_decided_under_model'] = not remaining
    result['refinement_node_limit_per_group'] = max_refinement_nodes
    result['witness_search_requested'] = verify_witnesses
    result['decision_scope'] = 'Existence of at least one encounter per fixed-size group in the bounded model. A witness does not enumerate every possible origin of that group.'
    extra_inputs = []
    if 'counterparts' in spec and result['certified_witnesses']:
        source_path = (path.parent/spec['counterparts']).resolve()
        sources, source_states = load_counterparts(source_path, spec['frame'])
        if sources['epoch_label'] != spec['epoch_label']:
            raise ValueError('Counterpart and stellar epochs must match exactly')
        extra_inputs.append(source_path)
        for witness in result['certified_witnesses']:
            candidate = dict(best_fit=dict(events=[dict(lookback_myr=witness['lookback_myr'])]),
                             independent_validation=dict(origin_pc=np.mean(witness['origin_enclosure_pc'], axis=1).tolist()))
            witness['counterparts'] = match_counterparts(candidate, sources, source_states,
                                                         GalacticPotential(spec['potential']), spec['fit']['max_step_myr'])
    from flint import __version__ as flint_version
    result['python_flint_version'] = flint_version
    result['provenance'] = _provenance([path]+extra_inputs+[Path(__file__).with_name(name) for name in
        ['ejector_bounds.py', 'ejector_search.py', 'ejector_fit.py', 'galactic.py', 'stellar_inputs.py']])
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination/'enclosures.npz', expanded_position_bounds_pc=bounds,
                        source_ids=np.array([r['id'] for r in spec['objects']]))
    result['enclosures'] = str(destination/'enclosures.npz')
    (destination/'complete.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def certify_witness(spec, member_ids, fitted, max_step_myr=.002):
    """Certify cost and finite-radius common origin for a proposed fitted state."""
    arb, arb_mat, ctx = _arb()
    previous_precision = ctx.prec
    ctx.prec = 128
    try:
        age = fitted['events'][0]['lookback_myr']
        lower, upper = spec['search']['lookback_bounds_myr']
        if not lower <= age <= upper or spec['search']['radius_pc'] <= 0:
            raise ValueError('Witness needs an in-domain age and strictly positive radius')
        lookup = {r['id']: r for r in spec['objects']}
        observations = {r['id']: r['mean'] for r in fitted['fitted_objects']}
        if set(observations) != set(member_ids) or len(fitted['fitted_objects']) != len(member_ids):
            raise ValueError('Witness observations must match the declared group')
        cost = arb(0)
        objects = []
        for key in member_ids:
            row = lookup[key]
            d = [arb(a)-arb(b) for a, b in zip(observations[key], row['mean'])]
            precision = arb_mat(row['covariance']).inv()
            cost += sum((d[i]*precision[i, j]*d[j] for i in range(4) for j in range(4)), arb(0))
            objects.append({**row, 'mean': observations[key]})
        local = {**spec, 'objects': objects, 'search': {**spec['search'], 'report_chi2_max': 0.}}
        states, _ = initial_boxes(local)
        steps = int(np.ceil(age/min(max_step_myr, spec['fit']['max_step_myr'])))
        h = -arb(age)/steps
        for _ in range(steps):
            states = [validated_step(s, h, spec['potential'])[1] for s in states]
        center = [sum((s[j] for s in states), arb(0))/len(states) for j in range(3)]
        radii = []
        for state in states:
            squared = sum((abs(state[j]-center[j]).upper()**2 for j in range(3)), arb(0))
            radii.append(outward_float_bounds(squared.sqrt())[1])
        cost_upper = outward_float_bounds(cost)[1]
        radius_upper = max(radii)
        certified = (arb(cost_upper) <= arb(spec['search']['report_chi2_max'])
                     and arb(radius_upper) <= arb(spec['search']['radius_pc'])
                     and all(row['mean'][0] >= spec['fit']['min_parallax_mas'] for row in objects))
        return dict(star_ids=member_ids, certified=bool(certified), lookback_myr=age,
                    total_mahalanobis_cost_upper=cost_upper, centroid_radius_upper_pc=radius_upper,
                    integration_steps=steps, fitted_objects=fitted['fitted_objects'],
                    origin_enclosure_pc=[outward_float_bounds(c) for c in center],
                    interpretation='Existence witness under the fixed recorded model, verified by Arb continuous-time enclosures. Not a causal ejection certificate.')
    finally:
        ctx.prec = previous_precision


def refine_group(spec, member_ids, max_nodes=1000):
    """Branch on latent Gaussian coordinates; never turn a node cap into exclusion.

    Each branch encloses continuous physical trajectories. Box cost lower bounds
    and a covering binary subdivision retain every point of the cost ball.
    Other stars' trajectory boxes are cached when a single star is subdivided.
    """
    if type(max_nodes) is not int or not 1 <= max_nodes <= 1000000:
        raise ValueError('max_nodes must be an integer between 1 and 1,000,000')
    if len(set(member_ids)) != len(member_ids) or len(member_ids) != spec['search']['group_size']:
        raise ValueError('Refinement requires one distinct group of the declared size')
    local = dict(spec)
    lookup = {r['id']: r for r in spec['objects']}
    local['objects'] = [lookup[key] for key in member_ids]
    validate_search(local)
    arb, _, ctx = _arb()
    previous_precision = ctx.prec
    ctx.prec = 128
    try:
        _, coefficients = initial_boxes(local)
        n = len(member_ids)
        budget = arb(spec['search']['report_chi2_max'])
        age_max = spec['search']['lookback_bounds_myr'][1]
        steps = int(np.ceil(age_max/spec['fit']['max_step_myr']))
        h = -arb(age_max)/steps
        radius = arb(spec['search']['radius_pc'])
        skip = np.array([arb(age_max)*(k+1)/steps < arb(spec['search']['lookback_bounds_myr'][0]) for k in range(steps)])
        factors, marginal = [], []
        for row in local['objects']:
            c = row['covariance']
            factor = [[arb(0) for _ in range(4)] for _ in range(4)]
            for i in range(4):
                for j in range(i+1):
                    value = arb(c[i][j])-sum((factor[i][k]*factor[j][k] for k in range(j)), arb(0))
                    factor[i][j] = value.sqrt() if i == j else value/factor[j][j]
            if not all(x.is_finite() for row_f in factor for x in row_f):
                raise ValueError('Failed to enclose exact covariance Cholesky factor')
            factors.append(factor)
            marginal.append([arb(m)+arb(0, (budget*arb(c[i][i])).sqrt()) for i, m in enumerate(row['mean'])])

        @lru_cache(maxsize=4096)
        def trajectory(i, box):
            intervals = [arb(box[2*j]).union(arb(box[2*j+1])) for j in range(4)]
            row, coeff = local['objects'][i], coefficients[i]
            obs = [arb(row['mean'][j])+sum((factors[i][j][k]*intervals[k] for k in range(4)), arb(0)) for j in range(4)]
            obs = [x.intersection(m) for x, m in zip(obs, marginal[i])]
            floor = arb(spec['fit']['min_parallax_mas'])
            if obs[0].upper() < floor:
                return None
            if obs[0].lower() < floor:
                obs[0] = floor.union(obs[0].upper())
            if not obs[0].lower() > 0:
                raise ValueError('Refined parallax enclosure is not positive')
            pos, vel = [], []
            for axis in range(3):
                pos.append(arb(coeff['observer_pc'][axis])+1000/obs[0]*arb(coeff['radial'][axis]))
                v = (arb(coeff['solar_velocity_kms'][axis])+obs[3]*arb(coeff['radial'][axis])
                     + arb(4.740470463533349)/obs[0]*(obs[1]*arb(coeff['east'][axis])+obs[2]*arb(coeff['north'][axis])))
                vel.append(v*arb(KMS_TO_PC_MYR))
            state = pos+vel
            bounds = np.empty((steps, 3, 2))
            for k in range(steps):
                try:
                    tube, state = validated_step(state, h, spec['potential'])
                    for axis in range(3): bounds[k, axis] = outward_float_bounds(tube[axis]+arb(0, radius))
                except ValueError:
                    bounds[k:] = [-np.inf, np.inf]
                    break
            return bounds

        # A heuristic decides only which covered box to split, never exclusion.
        # Nominal ballistic derivatives identify coordinates with large spatial
        # leverage; actual exclusions always use validated Galactic dynamics.
        problem = EncounterProblem(family_spec(local, list(range(n))))
        def split_dimension(box, age):
            priorities = []
            for i in range(n):
                plx, pmra, pmdec, _ = local['objects'][i]['mean']
                c = coefficients[i]
                radial, east, north = (np.array(c[k]) for k in ['radial', 'east', 'north'])
                derivatives = np.column_stack([
                    -1000/plx**2*radial + age*KMS_TO_PC_MYR*4.740470463533349/plx**2*(pmra*east+pmdec*north),
                    -age*KMS_TO_PC_MYR*4.740470463533349/plx*east,
                    -age*KMS_TO_PC_MYR*4.740470463533349/plx*north,
                    -age*KMS_TO_PC_MYR*radial]) @ problem.factors[i]
                priorities.extend(np.linalg.norm(derivatives, axis=0)*(box[4*i:4*i+4, 1]-box[4*i:4*i+4, 0]))
            return int(np.argmax(priorities))

        _, root_radius = outward_float_bounds(budget.sqrt())
        root = np.tile([-root_radius, root_radius], (4*n, 1))
        pending, leaves = [(root, 0)], []
        visited = 0
        spatial, cost, invalid = 0, 0, 0
        while pending and visited < max_nodes:
            box, depth = pending.pop()
            visited += 1
            minimum_cost = sum((arb(0) if lo <= 0 <= hi else arb(lo if lo > 0 else hi)**2 for lo, hi in box), arb(0))
            kind = None
            if minimum_cost > budget:
                cost += 1; kind = 'cost_lower_bound'
            else:
                paths = [trajectory(i, tuple(box[4*i:4*i+4].ravel())) for i in range(n)]
                if any(p is None for p in paths):
                    invalid += 1; kind = 'parallax_lower_bound'
                else:
                    paths = np.array(paths)
                    possible = np.all(paths[..., 0].max(axis=0) <= paths[..., 1].min(axis=0), axis=1) & ~skip
                    if not possible.any():
                        spatial += 1; kind = 'disjoint_continuous_tubes'
            if kind is not None:
                leaves.append(dict(bounds=box.tolist(), reason=kind, depth=depth))
                continue
            age = (np.flatnonzero(possible).mean()+.5)*age_max/steps
            j = split_dimension(box, age)
            mid = (box[j, 0]+box[j, 1])/2
            if not box[j, 0] < mid < box[j, 1]:
                pending.append((box, depth))
                break
            lower, upper = box.copy(), box.copy()
            lower[j, 1], upper[j, 0] = mid, mid
            pending.extend([(upper, depth+1), (lower, depth+1)])
        return dict(star_ids=member_ids, excluded=not pending, nodes_visited=visited, max_nodes=max_nodes,
                    terminal_leaves=leaves, pending_boxes=[dict(bounds=b.tolist(), depth=d) for b, d in pending],
                    spatial_leaves=spatial, cost_leaves=cost, invalid_parallax_leaves=invalid,
                    trajectory_cache=trajectory.cache_info()._asdict(),
                    interpretation='A covered latent-cost domain was subdivided. Exclusion requires the entire queue to be discharged; a node cap leaves the group unresolved.')
    finally:
        ctx.prec = previous_precision
