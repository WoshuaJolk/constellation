"""Research batch preparation/selection with the existing float64 fit policy.

No orbit starts or fitted observations are removed. Packing preserves each
start's epoch; selection preserves ordered duplicate replacement and ties.
"""
import numpy as np
from numba import njit
from magictools.itf.fit import prepare_fit, finish_prepared, check_deadline
from magictools.itf.fit_bookkeeping import candidate_summary, duplicate_index


@njit(cache=True, nogil=True)
def _pack_group(start, begin, seeds, epochs, t, observer, u, east, north,
                states, case_index, offsets, observers, observed, easts, norths):
    n = len(t)
    for k in range(len(seeds)):
        c = start+k
        states[c] = seeds[k]
        for j in range(n):
            p = begin+k*n+j
            case_index[p] = c
            offsets[p] = t[j]-epochs[k]
            observers[p] = observer[j]
            observed[p] = u[j]
            easts[p] = east[j]
            norths[p] = north[j]


def prepare_bank(geometry, jobs, *, compact_geometry=False):
    plans, groups, seeds, epochs = [], [], [], []
    count = 0
    policy = None
    allowed = {'deadline', 'extra_seeds', 'broad_starts', 'start_policy', 'allow_two_tracklets'}
    accepted_keys = allowed | {'sigma', 'rms_limit', 'prior', 'max_nfev', 'jacobian_backend'}
    for key, settings in jobs:
        if set(settings)-accepted_keys:
            raise ValueError('Unsupported batched fitting setting')
        if settings.get('jacobian_backend', 'compiled') != 'compiled':
            raise ValueError('GPU adapter requires compiled policy')
        requested_limit = settings.get('max_nfev', 80)
        if int(requested_limit) != requested_limit:
            raise ValueError('Iteration limit must be integral, not silently truncated')
        current = (float(settings.get('sigma', .3)), int(requested_limit))
        if policy is not None and current != policy:
            raise ValueError('GPU batch requires one sigma and iteration limit')
        policy = current
        plan = prepare_fit(geometry, key, **{k:v for k,v in settings.items() if k in allowed})
        plans.append(plan)
        if plan is None:
            states, times = np.empty((0, 6)), np.empty(0)
        else:
            states = np.asarray([x[0] for x in plan.seeds]).reshape(-1, 6)
            if states.dtype != np.float64:
                raise ValueError('GPU seed bank must already be float64; no silent precision conversion')
            times = np.asarray([x[1] for x in plan.seeds])
            if times.dtype.kind not in 'fiu' or times.dtype.itemsize > 8:
                raise ValueError('GPU epochs must be representable in the float64 input model')
            times = times.astype(np.float64, copy=False)
            finite = np.isfinite(states).all(axis=1)
            states, times = states[finite], times[finite]
        groups.append((count, count+len(states)))
        count += len(states)
        seeds.append(states); epochs.append(times)
    bounds = np.zeros(count+1, dtype=np.int64)
    for plan, (lo, hi) in zip(plans, groups):
        if hi > lo:
            bounds[lo+1:hi+1] = bounds[lo]+np.arange(1, hi-lo+1)*len(plan.t)
    nobs = int(bounds[-1])
    bank = dict(states=np.empty((count, 6)), case_index=np.empty(nobs, dtype=np.int64),
                offsets=np.empty(nobs), sigma=np.array(policy[0] if policy else .3))
    if compact_geometry:
        # One exact vector per chain detection, reused across its starting states.
        live = [(p, group) for p, group in zip(plans, groups) if group[1] > group[0]]
        for name, attr in [('observers','observers'),('observed','u'),('east','east'),('north','north')]:
            bank[name] = np.concatenate([getattr(p,attr) for p,_ in live]) if live else np.empty((0,3))
        bank['geometry_index'] = np.empty(nobs, dtype=np.int64)
        origin = 0
        for plan, (lo,hi), states, times, (_,settings) in zip(plans,groups,seeds,epochs,jobs):
            check_deadline(settings.get('deadline',np.inf))
            if hi > lo:
                n = len(plan.t)
                bank['states'][lo:hi] = states
                bank['case_index'][bounds[lo]:bounds[hi]] = np.repeat(np.arange(lo,hi),n)
                bank['offsets'][bounds[lo]:bounds[hi]] = (plan.t[None,:]-times[:,None]).ravel()
                bank['geometry_index'][bounds[lo]:bounds[hi]] = np.tile(np.arange(origin,origin+n),hi-lo)
                origin += n
    else:
        for name in ['observers', 'observed', 'east', 'north']:
            bank[name] = np.empty((nobs, 3))
        for plan, (lo, hi), states, times, (_, settings) in zip(plans, groups, seeds, epochs, jobs):
            check_deadline(settings.get('deadline', np.inf))
            if hi > lo:
                _pack_group(lo, bounds[lo], states, times, plan.t, plan.observers, plan.u,
                            plan.east, plan.north, bank['states'], bank['case_index'], bank['offsets'],
                            bank['observers'], bank['observed'], bank['east'], bank['north'])
    if hasattr(geometry, 'observation_indices'):
        observation_index = np.empty(nobs, dtype=np.int64)
        for plan, (lo, hi) in zip(plans, groups):
            if hi > lo:
                ids = geometry.observation_indices(plan.indices)
                if len(ids) != len(plan.t) or not np.array_equal(geometry.times[ids], plan.t):
                    raise ValueError('Observation-index mapping differs from fitted input')
                observation_index[bounds[lo]:bounds[hi]] = np.tile(ids, hi-lo)
        bank['observation_index'] = observation_index
    all_epochs = np.concatenate(epochs) if epochs else np.empty(0)
    return bank, plans, groups, all_epochs, bounds, (policy[1] if policy else 80)


@njit(cache=True, nogil=True)
def select_group(states, flags, residuals, bounds, lo, hi, labels, epochs,
                 canonical_epoch, observer, sigma, rms_limit):
    selected = np.empty(hi-lo, dtype=np.int64)
    values = np.empty((hi-lo, 10))
    positions, velocities = np.empty((hi-lo, 3)), np.empty((hi-lo, 3))
    count = 0
    for c in range(lo, hi):
        if not flags[c]:
            continue
        good, chi2, rms, maximum, distance, position, velocity, elements = candidate_summary(
            states[c], residuals[2*bounds[c]:2*bounds[c+1]], sigma, rms_limit,
            labels, canonical_epoch-epochs[c], observer)
        if not good:
            continue
        duplicate = duplicate_index(positions, velocities, count, position, velocity, distance)
        if duplicate >= 0:
            if values[duplicate, 0] <= chi2:
                continue
            count -= 1
            for j in range(duplicate, count):
                selected[j] = selected[j+1]
                values[j] = values[j+1]
                positions[j] = positions[j+1]
                velocities[j] = velocities[j+1]
        selected[count] = c
        values[count, :4] = np.array([chi2, rms, maximum, distance])
        values[count, 4:10] = elements
        positions[count], velocities[count] = position, velocity
        count += 1
    return selected[:count], values[:count]


def finish_group(plan, settings, group, epochs, bounds, states, flags, residuals, jacobians):
    if plan is None:
        return None
    check_deadline(settings.get('deadline', np.inf))
    lo, hi = group
    sigma = float(settings.get('sigma', .3))
    selected, values = select_group(states, flags, residuals, bounds, lo, hi, plan.labels,
        epochs, plan.epoch, plan.observers[plan.middle], sigma, settings.get('rms_limit', .6))
    solutions = []
    for c, v in zip(selected, values):
        chi2, rms, maximum, distance = v[:4]
        eccentricity, perihelion, axis, inclination, energy, hn = v[4:10]
        solutions.append(dict(state_au_au_per_day=states[c].tolist(), epoch_mjd_tdb=float(epochs[c]),
            chi2=float(chi2), rms_arcsec=float(rms), range_au=float(distance),
            max_residual_arcsec=float(maximum), _jacobian=jacobians[2*bounds[c]:2*bounds[c+1]],
            eccentricity=float(eccentricity), perihelion_au=float(perihelion),
            semimajor_axis_au=float(axis) if np.isfinite(axis) else None,
            inclination_deg=float(inclination), conic='hyperbolic' if energy > 0 else 'elliptic'))
    return finish_prepared(plan, solutions, hi-lo, sigma=sigma, prior=settings.get('prior', .001))
