"""Full-catalogue, uncertainty-guided recovery before and after short-arc seeds."""
import argparse
from collections import Counter, defaultdict, deque, OrderedDict
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import warnings

from astropy.time import Time
import numpy as np
from scipy.special import expit

from .catalogue import Catalogue, sha256
from .coverage import observing_coverage, site_longitudes, write_coverage
from .fit import BudgetExpired, check_deadline, linear_chi2, local_uncertainty, read_sites
from .kernels import ARCSEC
from .nbody import Ephemeris, JPLGeometry, Trajectory, TrajectoryDomainError, FitDeferred, fit_nbody


class NightIndex:
    """All eligible ITF nights, with measured motion carried to each night center."""
    def __init__(self, catalogue, sites, ephemeris, deadline=np.inf):
        tracks = catalogue.tracks
        self.tracks = tracks
        valid = ((tracks['count'] >= 2) & np.isfinite(tracks['u']).all(axis=1) &
                 np.isfinite(tracks['v']).all(axis=1) & np.isin(tracks['site'], list(sites)))
        valid &= (tracks['t'] > ephemeris.start+64) & (tracks['t'] < ephemeris.end-.1)
        selected = np.flatnonzero(valid)
        days = np.floor(tracks['t'][selected]).astype(int)
        order = np.argsort(days, kind='stable')
        selected, days = selected[order], days[order]
        unique, starts = np.unique(days, return_index=True)
        ends = np.r_[starts[1:], len(days)]
        units, ordered_indices = [], []
        self.utc = unique.astype(float) + .5
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.tdb = Time(self.utc, format='mjd', scale='utc').tdb.mjd
        self.observers = ephemeris.earth(self.tdb)
        for day, start, end in zip(unique, starts, ends):
            check_deadline(deadline)
            indices = selected[start:end]
            unit = tracks['u'][indices] + tracks['v'][indices]*(day+.5-tracks['t'][indices])[:, None]
            unit /= np.linalg.norm(unit, axis=1)[:, None]
            spatial_order = np.argsort(unit[:, 0], kind='stable')
            units.append(unit[spatial_order])
            ordered_indices.append(indices[spatial_order])
        self.point_indices = np.concatenate(ordered_indices) if ordered_indices else np.empty(0, dtype=np.int64)
        self.point_units = np.concatenate(units) if units else np.empty((0, 3))
        self.night_starts, self.night_ends = starts, ends
        self.fast_retrieval = False
        self.trajectory_cache = OrderedDict()
        self.trajectory_cache_size = 0
        from .reuse import NightPredictionCache,IndexedObserverCache
        self.work_reuse = True
        self.night_predictions = NightPredictionCache(len(self.tdb))
        self.motion_observers = IndexedObserverCache(len(tracks))
        self.window_cache = OrderedDict()
        self.counts = dict(indexed_tracklets=len(selected), unindexed_tracklets=len(tracks)-len(selected),
                           indexed_nights=len(unique), catalogue_tracklets=len(tracks))

    def proposals(self, fitted, ephemeris, *, minimum_gap=365., max_cone_deg=2.,
                  max_candidates=32, maximum_gap=0., retrieval='position', position_reserve=0, deadline=np.inf):
        self.arc_candidates = []
        epoch = fitted['epoch_mjd_tdb']
        window_key=(epoch,maximum_gap)
        cached_window=self.window_cache.get(window_key) if self.work_reuse else None
        if cached_window is None:
            window = np.abs(self.tdb-epoch) <= maximum_gap if maximum_gap else np.ones(len(self.tdb), dtype=bool)
            nights = np.flatnonzero(window)
            times, observers = self.tdb[nights], self.observers[nights]
            if self.work_reuse:
                self.window_cache[window_key]=(window,nights,times,observers)
                while len(self.window_cache)>16:self.window_cache.popitem(last=False)
        else:window,nights,times,observers=cached_window
        if not len(times):
            return [], dict(nights_considered=len(self.tdb), nights_in_time_window=0,
                outside_time_window_nights=len(self.tdb), nights_queried=0,
                uncertainty_gated_nights=0, short_gap_nights=0, raw_proposals=0,
                proposals_discarded_at_cap=0)
        key = (epoch, tuple(fitted['state_au_au_per_day']), self.fast_retrieval)
        cached = self.trajectory_cache.pop(key, None)
        trajectory = cached[0] if cached else None
        if cached:
            self.trajectory_cache_size -= cached[1]
        low, high = max(ephemeris.start, float(times.min())-.05), min(ephemeris.end, float(times.max())+.05)
        if trajectory is not None:
            trajectory.deadline = deadline
            trajectory.extend(low, high)
        else:
            trajectory = Trajectory(np.array(fitted['state_au_au_per_day']), epoch,
                                    low, high, ephemeris, deadline=deadline,
                                    **(dict(rtol=2e-8, max_step=256.) if self.fast_retrieval else {}))
        self.last_trajectory, self.trajectory_key = trajectory, key
        if not hasattr(trajectory,'night_reuse_token'):trajectory.night_reuse_token=object()
        # Retain central/probe integrations across overlapping search windows.
        # Count and byte limits keep distant, highly subdivided trajectories bounded.
        size = sum(getattr(solution, field).nbytes
                   for _, _, _, solution in trajectory.segments
                   for field in ('boundaries', 'old', 'dense') if hasattr(solution, field))
        self.trajectory_cache[key] = trajectory, size
        self.trajectory_cache_size += size
        while len(self.trajectory_cache) > 32 or (self.trajectory_cache_size > 128*1024**2 and len(self.trajectory_cache) > 1):
            _, (_, removed_size) = self.trajectory_cache.popitem(last=False)
            self.trajectory_cache_size -= removed_size
        unavailable = 0
        while len(times):
            try:
                predicted, distance, jacobian = (self.night_predictions.apparent((key,trajectory.night_reuse_token),trajectory,nights,times,observers)
                    if self.work_reuse else trajectory.apparent(times, observers))
                break
            except TrajectoryDomainError as exc:
                unavailable += int(exc.invalid.sum())
                nights, times, observers = nights[~exc.invalid], times[~exc.invalid], observers[~exc.invalid]
        if not len(times):
            return [], dict(nights_considered=len(self.tdb), ephemeris_unavailable_nights=unavailable,
                nights_in_time_window=int(window.sum()), outside_time_window_nights=int((~window).sum()),
                nights_queried=0, raw_proposals=0, proposals_discarded_at_cap=0)
        factor = np.array(fitted['covariance_factor'])
        # Frobenius norm bounds the largest angular standard deviation.
        spread = np.linalg.norm(jacobian @ factor, axis=(1, 2))
        radius = 5*spread + 600/ARCSEC + 4.27e-5/distance
        # The fast lane only proposes associations. Give its numerical prediction
        # and cone-boundary errors an extra margin; all innovation tests and fits
        # below use the original precise planetary dynamics. This is a measured
        # retrieval heuristic, not a rigorous global integration-error bound.
        cone_limit = np.deg2rad(max_cone_deg)*(1.05 if self.fast_retrieval else 1.)
        selected = (np.abs(times-epoch) >= minimum_gap) & (radius <= cone_limit)
        counts = dict(nights_considered=len(self.tdb), nights_in_time_window=int(window.sum()),
                      ephemeris_unavailable_nights=unavailable,
                      outside_time_window_nights=int((~window).sum()), nights_queried=int(selected.sum()),
                      uncertainty_gated_nights=int(np.sum((radius > np.deg2rad(max_cone_deg)) &
                                                         (np.abs(times-epoch) >= minimum_gap))),
                      excluded_window_uncertainty_gates=int(np.sum((radius > np.deg2rad(max_cone_deg)) &
                                                                  (np.abs(times-epoch) < minimum_gap))),
                      short_gap_nights=int(np.sum(np.abs(times-epoch) < minimum_gap)),
                      raw_proposals=0, proposals_discarded_at_cap=0)
        if self.fast_retrieval:
            radius = radius*1.01 + 30/ARCSEC
        candidates, candidate_times = {}, {}
        members = set(fitted['tracklet_indices'])
        selected_offsets = np.flatnonzero(selected)
        for first in range(0, len(selected_offsets), 512):
            check_deadline(deadline)
            offsets = selected_offsets[first:first+512]
            from .night_query import query_nights
            points, local_offsets, distances = query_nights(self.point_units, self.night_starts,
                self.night_ends, nights[offsets], predicted[offsets], 2*np.sin(radius[offsets]/2))
            if not len(points):
                continue
            offsets = offsets[local_offsets]
            indices = self.point_indices[points]
            scores = distances/radius[offsets]
            epochs = times[offsets] + self.tracks['t'][indices]-self.utc[nights[offsets]]
            for index, score, t in zip(indices.tolist(), scores.tolist(), epochs.tolist()):
                if index not in members:
                    candidates[index] = score
                    candidate_times[index] = t
        ordered = sorted(candidates, key=lambda index: (candidates[index], index))
        counts['raw_proposals'] = len(ordered)
        if retrieval == 'motion' and len(ordered) > max_candidates:
            position_order = ordered
            indices = np.array(ordered)
            t = np.array([candidate_times[i] for i in ordered])
            h = .01
            observers_plus,observers_minus=(self.motion_observers.pair(ephemeris,indices,t,h)
                if self.work_reuse else (ephemeris.earth(t+h),ephemeris.earth(t-h)))
            plus, dist, jp = trajectory.apparent(t+h, observers_plus)
            minus, _, jm = trajectory.apparent(t-h, observers_minus)
            velocity = (plus-minus)/(2*h)
            uncertainty = np.linalg.norm(((jp-jm)/(2*h))@factor, axis=(1, 2))
            tracks = self.tracks[indices]
            measurement = np.maximum(.3, tracks['rms'])/ARCSEC * np.sqrt(12/tracks['count'])/np.maximum(tracks['span'], 1e-5)
            # Geocentric ranking allows terrestrial rotational velocity; final fits
            # use exact station geometry. This ranks; it never rejects a proposal.
            scale = 5*uncertainty + measurement + 2.7e-4/dist + 1/ARCSEC
            rate_score = np.linalg.norm(velocity-tracks['v'], axis=1)/scale
            scores = {int(i): float(candidates[i]**2 + r*r) for i, r in zip(indices, rate_score)}
            motion_order = sorted(ordered, key=lambda i: (scores[i], i))
            # Protect an explicit positional prefix, then explore motion ranking.
            ordered = list(position_order[:min(position_reserve, max_candidates)])
            protected = set(ordered)
            ordered.extend(i for i in motion_order if i not in protected)
            counts['motion_ranked_proposals'] = len(indices)
            counts['position_reserved_proposals'] = len(protected)
        # Retain an independent arc-anchor lane before the single-tracklet cap.
        self.arc_candidates = [i for i in ordered if i in getattr(self, 'arc_lookup', {})]
        counts['arc_anchors_before_tracklet_cap'] = len(self.arc_candidates)
        counts['proposals_discarded_at_cap'] = max(0, len(ordered)-max_candidates)
        return ordered[:max_candidates], counts


def screening_trajectory(trajectory, times):
    if trajectory.rtol > 2e-11 or trajectory.max_step > 10.:
        precise = getattr(trajectory, 'screening_trajectory', None)
        if precise is None:
            precise = Trajectory(trajectory.initial[:6], trajectory.epoch,
                max(trajectory.ephemeris.start, float(times.min())-.05),
                min(trajectory.ephemeris.end, float(times.max())+.05),
                trajectory.ephemeris, deadline=trajectory.deadline, backend=trajectory.backend)
            trajectory.screening_trajectory = precise
        precise.deadline = trajectory.deadline
        trajectory = precise
    return trajectory


def prepare_screening(trajectory, geometry, indices):
    """Batch exact observer predictions once for each proposed orbit branch."""
    if not indices:
        return
    t, _, observers, _ = geometry.get(sorted(indices))
    if not len(t):
        return
    precise = screening_trajectory(trajectory, t)
    predicted, _, jacobian = precise.apparent(t, observers)
    trajectory.screen_predictions = {tuple([float(time_), *observer]): (unit, jac)
        for time_, observer, unit, jac in zip(t, observers, predicted, jacobian)}


def screen_extension(seed, trajectory, geometry, candidate, sigma=.3):
    """Joint astrometric innovation; only the full N-body fit accepts a link."""
    t, observed, observers, _ = geometry.get([candidate] if np.isscalar(candidate) else candidate)
    cache = getattr(trajectory, 'screen_predictions', {})
    cached = [cache.get(tuple([float(time_), *observer])) for time_, observer in zip(t, observers)]
    if cached and all(item is not None for item in cached):
        predicted = np.array([item[0] for item in cached])
        jacobian = np.array([item[1] for item in cached])
    else:
        predicted, _, jacobian = screening_trajectory(trajectory, t).apparent(t, observers)
    east = np.column_stack((-observed[:, 1], observed[:, 0], np.zeros(len(t))))
    east /= np.linalg.norm(east, axis=1)[:, None]
    north = np.cross(observed, east)
    residual = np.column_stack((np.sum((observed-predicted)*east, axis=1),
                                np.sum((observed-predicted)*north, axis=1))).ravel()*ARCSEC/sigma
    jac = np.stack((np.einsum('ni,nij->nj', east, jacobian),
                    np.einsum('ni,nij->nj', north, jacobian)), axis=1).reshape(-1, 6)*ARCSEC/sigma
    factor = np.array(seed['covariance_factor'])
    design = jac @ factor
    u, singular, vt = np.linalg.svd(design, full_matrices=False)
    update = vt.T @ ((singular/(1+singular**2)) * (u.T @ residual))
    after = residual-design@update
    chi2 = float(after@after+update@update)
    if chi2 > 25*len(residual):
        return None, chi2
    return np.array(seed['state_au_au_per_day']) + factor@update, chi2


def arc_lookup(groups):
    lookup = defaultdict(list)
    for group in groups:
        members = tuple(sorted(set(map(int, group))))
        for member in members:
            lookup[member].append(members)
    return lookup


def attach_arcs(index, groups, catalogue):
    index.arc_groups = groups
    index.arc_lookup = arc_lookup(groups)


def proposed_arcs(proposals, members, lookup, limit=32):
    """Whole source arcs anchored by a retrieved tracklet; singletons stay separate."""
    members = set(members)
    ranked = {}
    for rank, proposal in enumerate(proposals):
        for group in lookup.get(proposal, ()):
            additions = tuple(i for i in group if i not in members)
            if len(additions) < 2:
                continue
            ranked[additions] = min(rank, ranked.get(additions, rank))
    ordered = sorted(ranked, key=lambda group: (ranked[group], -len(group), group))
    return ordered[:limit], len(ordered)


def seed_bank(report, max_relative_range=.05, allow_weak=False):
    seeds = []
    for row in report['candidates']:
        if not row['maximal']:
            continue
        for mode_index, mode in enumerate(row['modes']):
            error = mode['local_uncertainty']
            if (error['local_linearization_weak'] and not allow_weak) or error['range_sigma_au'] is None:
                continue
            relative = error['range_sigma_au']/mode['range_au']
            if relative <= max_relative_range:
                seeds.append((relative, -row['span_days'], row['rank'], mode_index, row, mode))
    return sorted(seeds, key=lambda item: item[:4])


def read_recovery_input(path, max_relative_range=.05, allow_weak=False, link_arcs=False):
    """Keep eligible modes and arc memberships, not every fitted report in RAM.

    Mode indices and ordering match seed_bank exactly. Ineligible maximal rows
    still enter the arc index, so streaming does not narrow arc-merge proposals.
    """
    from .report_io import metadata, candidates
    header = metadata(path)
    seeds, groups = [], []
    for row in candidates(path):
        if link_arcs and row['maximal']:
            groups.append(row['tracklet_indices'])
        selected = seed_bank({'candidates': [row]}, max_relative_range, allow_weak)
        if selected:
            # A worker needs the chosen mode, not every sibling mode on its row.
            compact = {key: value for key, value in row.items() if key != 'modes'}
            seeds.extend(item[:4] + (compact, item[5]) for item in selected)
    seeds.sort(key=lambda item: item[:4])
    return header, seeds, groups


def candidate_row(fitted, geometry, seed_rank, seed_mode, sigma):
    indices = fitted['tracklet_indices']
    t, u, observers, labels = geometry.get(indices)
    null = linear_chi2(t, u, labels, sigma)
    log_bf = .5*(null-fitted['chi2']+(4*len(indices)-6)*np.log(2*len(t)))
    log_odds = math.log(.001/.999) + log_bf
    factor = np.array(fitted['covariance_factor'])
    uncertainty = local_uncertainty(np.array(fitted['state_au_au_per_day']), np.linalg.inv(factor),
                                    0, geometry.ephemeris.earth(fitted['epoch_mjd_tdb']))
    mode = dict(fitted, relative_mode_likelihood=1., local_uncertainty=uncertainty)
    catalogue = geometry.catalogue
    observations, codes, _ = catalogue.observations(indices)
    longitudes = site_longitudes({code: geometry.sites[code] for code in set(codes) if code in geometry.sites})
    return dict(fitted, tracklets=[catalogue.tracks[i]['id'].decode().strip() for i in indices],
                stations=sorted({catalogue.tracks[i]['site'].decode() for i in indices}),
                modes=[mode], probability_bic=float(expit(log_odds)),
                log_odds_bic=float(log_odds), probability_kind='uncalibrated BIC approximation',
                seed_rank=seed_rank, seed_mode=seed_mode, sigma_arcsec=sigma,
                dynamics='Newtonian Sun + JPL planets and Moon; variational covariance',
                **observing_coverage(observations['t'], [longitudes.get(code, np.nan) for code in codes]))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def mark_maximal(rows):
    members = [set(row['tracklet_indices']) for row in rows]
    membership = defaultdict(set)
    for k, group in enumerate(members):
        for tracklet in group:
            membership[tracklet].add(k)
    for rank, row in enumerate(rows, 1):
        row['rank'] = rank
        group = members[rank-1]
        possible = min((membership[i] for i in group), key=len)
        row['maximal'] = not any(group < members[k] for k in possible)


def merge_candidate_rows(previous, incoming):
    """Preserve distinct recovered orbital alternatives on the same observation set.

    Only identical epoch/state records are collapsed. Source likelihood values are
    retained as diagnostics, never renormalized into an implied posterior.
    """
    if previous is None:
        return incoming
    if previous['tracklet_indices'] != incoming['tracklet_indices']:
        raise ValueError('Cannot merge different chain memberships')
    best = dict(previous if previous['chi2'] <= incoming['chi2'] else incoming)
    modes = {}
    for row in (previous, incoming):
        for mode in row['modes']:
            key = (mode['epoch_mjd_tdb'], tuple(mode['state_au_au_per_day']))
            if key not in modes or mode['chi2'] < modes[key]['chi2']:
                modes[key] = mode
    best['modes'] = sorted(modes.values(), key=lambda mode: mode['chi2'])
    best['mode_weights_kind'] = 'Source-fit values retained; not a normalized merged posterior'
    return best


def export(db, directory, catalogue, identity, counts, status, elapsed):
    rows = [json.loads(row[0]) for row in db.execute('SELECT body FROM candidates')]
    rows.sort(key=lambda row: (-row['span_days'], row['chi2']/row['observations'], row['tracklet_indices']))
    mark_maximal(rows)
    statuses = dict(db.execute('SELECT status, COUNT(*) FROM seeds GROUP BY status'))
    completed = statuses.get('searched', 0) + statuses.get('seed_refit_rejected', 0)
    deferred = statuses.get('searched_deferred_fits', 0)
    attempted = sum(statuses.values())
    counts = dict(counts, seeds_completed=completed, seeds_attempted=attempted,
                  seeds_searched=statuses.get('searched', 0), seed_refits_rejected=statuses.get('seed_refit_rejected', 0),
                  seeds_retryable=attempted-completed-deferred, seeds_unattempted=counts['eligible_seed_modes']-attempted,
                  seeds_with_deferred_fits=deferred, seeds_first_pass_completed=completed+deferred,
                  first_pass_scan_complete=completed+deferred == counts['eligible_seed_modes'],
                  seed_attempts=db.execute('SELECT COALESCE(SUM(attempts),0) FROM seeds').fetchone()[0],
                  candidate_chains=len(rows), seeds_remaining=counts['eligible_seed_modes']-completed,
                  full_catalogue_index_complete=True,
                  full_catalogue_scan_complete=completed == counts['eligible_seed_modes'])
    report = dict(status=status, input=catalogue.manifest, identity=identity, candidates=rows,
                  counts=counts, timings={'invocation_seconds': elapsed},
                  exhaustive=False, probabilities_calibrated=False, seed_statuses=statuses,
                  ranking='Descending observed arc span, then chi2 per observation',
                  limitations=['Local Gaussian state covariance; not an integrated or calibrated posterior.',
                    'Optional fast retrieval uses relaxed integration with padded cones; its recall is empirical, not guaranteed.',
                    'Optional fit cascade saves unpromoted extension inputs under deferred-fits; these are not rejected associations.',
                    'Maximal source modes enter under the configured range-uncertainty/weak-mode eligibility limits.',
                    'Distinct recovered modes are preserved; source weights are not a normalized merged posterior.',
                    'Five-sigma cones include a 600-arcsecond screening allowance and terrestrial parallax.',
                    'Daily linear motion reduction and maximum-cone/candidate budgets bound retrieval.',
                    'No asteroid perturbers, relativity, comet non-gravitational forces, or systematic astrometric error model.',
                    'Newtonian fits use JPL interpolation for forces and exact SPK Earth positions for observers.'])
    atomic_json(directory/'report.json', report)
    atomic_json(directory/'progress.json', dict(status=status, counts=counts, elapsed_seconds=elapsed,
                                                updated_utc=time.time()))
    lines = ['Arc recovery candidates; uncalibrated BIC scores; no confirmed identities.']
    for row in rows:
        lines.append(f'{row["rank"]}. '+ ' + '.join(row['tracklets']) +
                     f' ({", ".join(row["stations"])}), {row["span_days"]:.1f} d, '
                     f'{row["nights"]} nights, {row["oppositions_estimate"]} estimated seasons, '
                     f'{row["observations"]} obs, {row["rms_arcsec"]:.3f}" rms')
    (directory/'candidates.txt').write_text('\n'.join(lines)+'\n')
    return report


def recover_seed(seed, geometry, index, ephemeris, args, deadline):
    from .reuse import ExactFitCache
    index.work_reuse=not getattr(args,'no_work_reuse',False) and hasattr(index,'night_predictions') and hasattr(geometry,'catalogue')
    if index.work_reuse and not hasattr(geometry,'exact_fit_cache'):
        output=getattr(args,'output',None)
        identity=dict(source=geometry.catalogue.manifest['source_sha256'],
            ephemeris=sha256(args.ephemeris) if hasattr(args,'ephemeris') else str(type(ephemeris)),
            implementation={name:sha256(Path(__file__).with_name(name)) for name in ('longarc.py','nbody.py','nbody_fast.py','reuse.py')},
            backend=getattr(args,'backend',os.environ.get('ITF_NBODY_BACKEND','scipy')))
        geometry.exact_fit_cache=ExactFitCache(Path(output)/'exact-fit-cache.sqlite' if output else None,identity)
    caches=[index.night_predictions,index.motion_observers,geometry.exact_fit_cache] if index.work_reuse else []
    before=Counter()
    for cache in caches:before.update(cache.counts)
    key,status,counts,rows=_recover_seed(seed,geometry,index,ephemeris,args,deadline)
    after=Counter()
    for cache in caches:after.update(cache.counts)
    counts.update(after-before)
    return key,status,counts,rows


def _recover_seed(seed, geometry, index, ephemeris, args, deadline):
    _, _, rank, mode_index, row, mode = seed[:6]
    attempt = seed[6] if len(seed) > 6 else 1
    key = f'{rank}:{mode_index}'
    counts, accepted = Counter(), []
    seed_deadline = min(deadline, time.monotonic() + min(
        getattr(args, 'seed_seconds', 10.) * 2**(attempt-1), getattr(args, 'max_seed_seconds', 60.)))
    fit_seconds = getattr(args, 'fit_seconds', 2.) * 2**(attempt-1)
    incomplete = False

    def fit(indices, initial, epoch):
        nonlocal incomplete
        check_deadline(seed_deadline)
        if hasattr(geometry, 'prepare'):
            geometry.prepare(indices, seed_deadline)
        from .reuse import fit_key
        options=dict(sigma=args.sigma_arcsec,max_nfev=35*2**min(attempt-1,2),fast_start=getattr(args,'fast_fit',False),
            pilot_gate_arcsec=3. if getattr(args,'fit_cascade',False) and set(indices)!=set(row['tracklet_indices']) else None)
        cache=geometry.exact_fit_cache if index.work_reuse else None
        cache_key=fit_key(indices,initial,epoch,options)
        try:
            saved=cache.get(cache_key) if cache else None
            if saved:
                if saved['kind']=='deferred':raise FitDeferred(saved['rms'],saved['evaluations'])
                return saved['fit']
            fitted=fit_nbody(geometry,indices,initial,epoch,
                deadline=min(seed_deadline,time.monotonic()+fit_seconds),**options)
            # None is a completed input rejection; non-convergence and deadlines
            # raise below and remain retryable. This key is exact, including the
            # starting orbit and evaluation cap, not an association exclusion.
            if cache:cache.put(cache_key,dict(kind='fit' if fitted is not None else 'completed_rejection',fit=fitted))
            return fitted
        except FitDeferred as exc:
            if cache and not saved:cache.put(cache_key,dict(kind='deferred',rms=exc.rms_arcsec,evaluations=exc.evaluations))
            counts['deferred_full_fits'] += 1
            # Append the full restart inputs before continuing to another proposal.
            # Workers have separate files, avoiding shared writer contention.
            destination = Path(args.output)/'deferred-fits'
            destination.mkdir(exist_ok=True)
            record = dict(seed_key=key, indices=list(map(int, indices)), initial=np.asarray(initial).tolist(),
                          epoch=float(epoch), sigma=args.sigma_arcsec, pilot_rms_arcsec=exc.rms_arcsec,
                          pilot_evaluations=exc.evaluations, reason='pilot_gate', status='pending_full_fit')
            with (destination/f'{os.getpid()}.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
        except BudgetExpired:
            check_deadline(seed_deadline)
            counts['fit_budget_exhaustions'] += 1
            incomplete = True
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            counts['fit_numerical_failures'] += 1
            incomplete = True
        return None

    def keep(extension, kind):
        if len(extension['tracklet_indices']) < 3:
            return  # Pair bootstraps are hypotheses, never accepted chains.
        new_row = candidate_row(extension, geometry, rank, mode_index, args.sigma_arcsec)
        new_row['extension_kind'] = kind
        accepted.append(new_row)

    try:
        fitted = fit(row['tracklet_indices'], np.array(mode['state_au_au_per_day']), mode['epoch_mjd_tdb'])
        if fitted is None:
            return key, 'numerical_failure' if incomplete else 'seed_refit_rejected', counts, accepted
        if row.get('recovered_input'):
            keep(fitted, 'validated_prior_chain')
            counts['validated_prior_chains'] += 1
        windows = ([(0., 14.), (14., 60.), (60., 180.), (180., 730.), (730., 0.)]
                   if getattr(args, 'bridge_first', False) else
                   [(0., 120.), (120., 730.), (730., 0.)] if getattr(args, 'staged', False) else [(args.min_gap_days, args.max_gap_days)])
        rounds, branches = getattr(args, 'iterations', 1), getattr(args, 'branches', 4)
        queue = deque([(fitted, 0, 0)])
        seen = {tuple(fitted['tracklet_indices'])}
        while queue:
            base, depth, first_window = queue.popleft()
            counts['hypotheses_searched'] += 1
            children = []
            for window_index in range(first_window, len(windows)):
                minimum, maximum = windows[window_index]
                check_deadline(seed_deadline)
                counts['window_searches'] += 1
                try:
                    from .recovery_explore import retrieve
                    proposals, proposal_counts, proposal_hints, arc_proposals = retrieve(
                        index, base, ephemeris, args, minimum, maximum, seed_deadline)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    counts['propagation_numerical_failures'] += 1
                    incomplete = True
                    print(f'Seed {key}, window {minimum:g}–{maximum:g}: {type(exc).__name__}: {exc}', flush=True)
                    continue
                counts.update(proposal_counts)
                if proposal_counts.get('uncertainty_probe_numerical_failures'):
                    incomplete = True
                arc_proposals = arc_proposals if getattr(args, 'direct_arcs', False) else proposals
                groups, raw = proposed_arcs(arc_proposals, base['tracklet_indices'],
                    getattr(index, 'arc_lookup', {}), args.max_candidates)
                counts['arc_merge_proposals'] += raw
                counts['arc_merges_discarded_at_cap'] += max(0, raw-len(groups))
                if hasattr(geometry, 'prepare'):
                    geometry.prepare(set(proposals).union(*(set(g) for g in groups)), seed_deadline)
                if getattr(args, 'fast_retrieval', False):
                    batches = {}
                    for candidate in proposals:
                        _, trajectory = proposal_hints.get(candidate, (base, index.last_trajectory))
                        batches.setdefault(id(trajectory), (trajectory, set()))[1].add(candidate)
                    for group in groups:
                        _, trajectory = next((proposal_hints[i] for i in group if i in proposal_hints), (base, index.last_trajectory))
                        batches.setdefault(id(trajectory), (trajectory, set()))[1].update(group)
                    for trajectory, indices in batches.values():
                        try:
                            prepare_screening(trajectory, geometry, indices)
                        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                            trajectory.screen_predictions = {}
                            counts['screen_prefetch_fallbacks'] += 1
                current = base
                for candidate in proposals:
                    check_deadline(seed_deadline)
                    if candidate in current['tracklet_indices']:
                        continue
                    hint, trajectory = proposal_hints.get(candidate, (base, index.last_trajectory))
                    initial, innovation = screen_extension(hint, trajectory, geometry, candidate, args.sigma_arcsec)
                    if initial is None and hint is not base:
                        initial = np.array(hint['state_au_au_per_day'])
                        counts['nonlinear_proposal_refits'] += 1
                    if initial is None:
                        counts['innovation_rejections'] += 1
                        continue
                    counts['extension_fits'] += 1
                    indices = sorted(set(current['tracklet_indices']) | {candidate})
                    extension = fit(indices, initial, base['epoch_mjd_tdb'])
                    if extension is not None:
                        keep(extension, 'single_tracklet_joint_fit')
                        children.append(extension)
                        current = extension
                        counts['accepted_extensions'] += 1
                    elif getattr(args, 'staged', False) and current is not base:
                        # A conflicting greedy addition must not permanently hide an alternative.
                        counts['alternative_branch_fits'] += 1
                        alternative = fit(sorted(set(base['tracklet_indices']) | {candidate}), initial, base['epoch_mjd_tdb'])
                        if alternative is not None:
                            keep(alternative, 'alternative_joint_fit')
                            children.append(alternative)
                            counts['accepted_alternative_branches'] += 1
                    if extension is None and getattr(args, 'swap_limit', 0):
                        removable = sorted(set(base['tracklet_indices'])-set(row['tracklet_indices']))
                        counts['swap_options_deferred'] += max(0, len(removable)-args.swap_limit)
                        for removed in removable[:args.swap_limit]:
                            counts['swap_fits'] += 1
                            replacement = fit(sorted((set(base['tracklet_indices'])-{removed})|{candidate}), initial, base['epoch_mjd_tdb'])
                            if replacement is not None:
                                replacement['replaced_tracklet_indices'] = [removed]
                                keep(replacement, 'reversible_attachment_joint_fit')
                                children.append(replacement)
                                counts['accepted_swaps'] += 1
                for group in groups:
                    check_deadline(seed_deadline)
                    hint, trajectory = next((proposal_hints[i] for i in group if i in proposal_hints), (base, index.last_trajectory))
                    initial, innovation = screen_extension(hint, trajectory, geometry, group, args.sigma_arcsec)
                    if initial is None:
                        counts['arc_merge_innovation_rejections'] += 1
                        continue
                    counts['arc_merge_fits'] += 1
                    indices = sorted(set(base['tracklet_indices']) | set(group))
                    extension = fit(indices, initial, base['epoch_mjd_tdb'])
                    if extension is not None:
                        keep(extension, 'whole_arc_joint_fit')
                        children.append(extension)
                        counts['accepted_arc_merges'] += 1
                if children and getattr(args, 'bridge_first', False) and depth+1 < rounds:
                    # Retain the parent's later windows, but let improved children
                    # search first. Better geometry can unlock previously wide cones.
                    if window_index+1 < len(windows):
                        queue.append((base, depth, window_index+1))
                        counts['parent_window_continuations'] += 1
                    counts['bridge_first_advances'] += 1
                    break
            # Search each new observation set once per original orbit mode. Every
            # accepted child remains exported even when its continuation is capped.
            unique = {tuple(child['tracklet_indices']): child for child in children
                      if tuple(child['tracklet_indices']) not in seen}
            candidates = list(unique.values())
            maximal = [c for c in candidates if not any(set(c['tracklet_indices']) < set(d['tracklet_indices']) for d in candidates)]
            maximal.sort(key=lambda c: (-c['span_days'], -len(c['tracklet_indices']), c['chi2']))
            if depth+1 < rounds:
                counts['branches_discarded_at_cap'] += max(0, len(maximal)-branches)
                for child in maximal[:branches]:
                    seen.add(tuple(child['tracklet_indices']))
                    if getattr(args, 'bridge_first', False):
                        queue.appendleft((child, depth+1, 0))
                    else:
                        queue.append((child, depth+1, 0))
                    counts['requery_hypotheses'] += 1
            else:
                counts['branches_at_depth_limit'] += len(maximal)
        return key, ('partial_failure' if incomplete else 'searched_deferred_fits'
                     if counts['deferred_full_fits'] else 'searched'), counts, accepted
    except BudgetExpired:
        counts['seed_budget_exhaustions'] += 1
        return key, 'budget_expired' if time.monotonic() >= deadline else 'seed_budget_exhausted', counts, accepted
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        counts['seed_numerical_failures'] += 1
        print(f'Seed {key}: {type(exc).__name__}: {exc}', flush=True)
        return key, 'numerical_failure', counts, accepted


_WORKER = None


def recovery_geometry(args, catalogue, sites, ephemeris):
    if getattr(args, 'campaign_geometry', None):
        from .campaign_jpl_geometry import JPLAtlasGeometry
        return JPLAtlasGeometry(args.cache,args.observatories,args.campaign_geometry,ephemeris,args.ephemeris)
    return JPLGeometry(catalogue,sites,ephemeris)


def initialize_worker(args, endpoints, deadline, groups):
    global _WORKER
    os.environ['ITF_NBODY_BACKEND'] = args.backend
    catalogue = Catalogue(args.cache)
    sites = read_sites(args.observatories)
    ephemeris = Ephemeris(args.ephemeris, *endpoints)
    index = NightIndex(catalogue, sites, ephemeris, deadline)
    index.fast_retrieval = getattr(args, 'fast_retrieval', False)
    attach_arcs(index, groups, catalogue)
    _WORKER = (recovery_geometry(args, catalogue, sites, ephemeris), index, ephemeris, args, deadline)


def worker_seed(seed):
    return recover_seed(seed, *_WORKER)


def seed_results(seeds, geometry, index, ephemeris, args, endpoints, deadline):
    if args.workers == 1:
        for seed in seeds:
            check_deadline(deadline)
            yield recover_seed(seed, geometry, index, ephemeris, args, deadline)
        return
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    import multiprocessing
    iterator = iter(seeds)
    with ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker, initargs=(args, endpoints, deadline, index.arc_groups)) as pool:
        pending = set()
        def submit():
            if time.monotonic() >= deadline:
                return
            seed = next(iterator, None)
            if seed is not None:
                pending.add(pool.submit(worker_seed, seed))
        for _ in range(args.workers):
            submit()
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                yield future.result()
                submit()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True, help='Short-arc report supplying all eligible modes')
    parser.add_argument('--observatories', type=Path, required=True)
    parser.add_argument('--ephemeris', type=Path, required=True, help='Local JPL DE440/DE440s SPK')
    parser.add_argument('--output', type=Path, required=True, help='Resumable output directory')
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--campaign-geometry', type=Path, help='Campaign root with a complete exact-JPL shared atlas')
    parser.add_argument('--workers', type=int, default=1, help='Independent spawned seed workers')
    parser.add_argument('--fast-retrieval', action='store_true', help='Relax planetary proposal propagation only; preserve precise innovation screening and final fits')
    parser.add_argument('--fast-fit', action='store_true', help='Initialize long-span strict fits with a cheaper planetary pilot; never accept the pilot directly')
    parser.add_argument('--fit-cascade', action='store_true', help='Defer difficult long-span extension pilots to durable full-fit queues; requires --fast-fit')
    parser.add_argument('--uncertainty-samples', type=int, default=0, help='Up to 24 nonlinear covariance-axis retrieval probes when cones exceed the cap')
    parser.add_argument('--bridge-first', action='store_true', help='Grow through intermediate dates before distant windows; preserve parent continuations')
    parser.add_argument('--direct-arcs', action='store_true', help='Retrieve source-arc anchors before single-tracklet proposal caps')
    parser.add_argument('--swap-limit', type=int, default=0, help='Try replacing this many non-root attachments after a failed addition')
    parser.add_argument('--allow-weak-modes', action='store_true', help='Explore locally weak modes within the declared relative-range bound')
    parser.add_argument('--staged', action='store_true', help='Search 0–120, 120–730, then >730-day windows')
    parser.add_argument('--iterations', type=int, default=1, help='Growth generations including the root mode')
    parser.add_argument('--no-work-reuse',action='store_true',help='Reference lane without exact fit/observer/apparent-position reuse')
    parser.add_argument('--branches', type=int, default=4, help='Continued maximal children per parent')
    parser.add_argument('--seed-seconds', type=float, default=10., help='Initial per-mode fairness budget')
    parser.add_argument('--max-seed-seconds', type=float, default=60., help='Retry per-mode budget cap')
    parser.add_argument('--fit-seconds', type=float, default=2., help='Initial per-fit budget; grows on retry')
    parser.add_argument('--retrieval', choices=('position','motion'), default='position')
    parser.add_argument('--position-reserve', type=int, default=32, help='Protected positional prefix in motion retrieval')
    parser.add_argument('--link-arcs', action='store_true',
                        help='Also jointly fit whole source arcs as independent branches')
    parser.add_argument('--backend', choices=('scipy', 'compiled'), default='compiled',
                        help='Same adaptive DOP853 model; compiled avoids Python integration overhead')
    parser.add_argument('--max-seeds', type=int, default=0, help='Modes per invocation; zero uses remaining time')
    parser.add_argument('--max-relative-range-error', type=float, default=.05)
    parser.add_argument('--min-gap-days', type=float, default=365.)
    parser.add_argument('--max-gap-days', type=float, default=0.,
                        help='Maximum epoch separation; zero searches the full date range')
    parser.add_argument('--max-cone-deg', type=float, default=2.)
    parser.add_argument('--max-candidates', type=int, default=32)
    parser.add_argument('--sigma-arcsec', type=float, default=.3)
    args = parser.parse_args(argv)
    if args.fit_cascade and not args.fast_fit:
        parser.error('--fit-cascade requires --fast-fit')
    if not all(math.isfinite(x) and x > 0 for x in (args.seconds, args.max_relative_range_error,
           args.min_gap_days, args.max_cone_deg, args.sigma_arcsec, args.seed_seconds, args.max_seed_seconds, args.fit_seconds)) or args.max_candidates < 1 or args.max_seeds < 0 or args.workers < 1 or args.iterations < 1 or args.branches < 1 or args.position_reserve < 0 or args.swap_limit < 0 or not 0 <= args.uncertainty_samples <= 24:
        parser.error('Numeric bounds must be positive; max-seeds may be zero')
    if not math.isfinite(args.max_gap_days) or args.max_gap_days < 0 or (args.max_gap_days and args.max_gap_days < args.min_gap_days):
        parser.error('max-gap-days must be zero or at least min-gap-days')
    os.environ['ITF_NBODY_BACKEND'] = args.backend
    started = time.monotonic()
    deadline = started + args.seconds
    catalogue = Catalogue(args.cache)
    source, seeds, arc_groups = read_recovery_input(args.report,
        args.max_relative_range_error, args.allow_weak_modes, args.link_arcs)
    if source['input'] != catalogue.manifest:
        parser.error('Seed report and full ITF cache manifests differ')
    sites = read_sites(args.observatories)
    identity = dict(schema=2, report_sha256=sha256(args.report), source_sha256=catalogue.manifest['source_sha256'],
                    migration=json.loads((args.output/'migration.json').read_text()) if (args.output/'migration.json').exists() else None,
                    ephemeris_sha256=sha256(args.ephemeris), observatories_sha256=sha256(args.observatories),
                    implementation={name: sha256(Path(__file__).with_name(name)) for name in ('longarc.py','nbody.py','nbody_fast.py','night_query.py','recovery_explore.py','report_io.py','campaign_jpl_geometry.py','campaign_geometry_fast.py','reuse.py')},
                    geometry_atlas_identity=json.loads((args.campaign_geometry/'fit-geometry-jpl/identity.json').read_text()) if args.campaign_geometry else None,
                    minimum_gap_days=args.min_gap_days, maximum_gap_days=args.max_gap_days, max_cone_deg=args.max_cone_deg,
                    max_candidates=args.max_candidates, max_relative_range_error=args.max_relative_range_error,
                    sigma_arcsec=args.sigma_arcsec, force_interpolation_days=.5, backend=args.backend,
                    uncertainty_samples=args.uncertainty_samples, bridge_first=args.bridge_first, direct_arcs=args.direct_arcs, swap_limit=args.swap_limit,
                    allow_weak_modes=args.allow_weak_modes, link_arcs=args.link_arcs, staged=args.staged, iterations=args.iterations, branches=args.branches,
                    retrieval=args.retrieval, position_reserve=args.position_reserve,
                    fast_retrieval=args.fast_retrieval,
                    fast_fit=args.fast_fit,
                    fit_cascade=args.fit_cascade,
                    work_reuse=not args.no_work_reuse,
                    scipy_version=__import__('scipy').__version__, numba_version=__import__('numba').__version__)
    args.output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.output/'recovery.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('CREATE TABLE IF NOT EXISTS metadata (identity TEXT); '
                    'CREATE TABLE IF NOT EXISTS seeds (key TEXT PRIMARY KEY, status TEXT, counts TEXT, attempts INTEGER NOT NULL DEFAULT 1); '
                    'CREATE TABLE IF NOT EXISTS candidates (key TEXT PRIMARY KEY, body TEXT);')
    old = db.execute('SELECT identity FROM metadata').fetchone()
    encoded = json.dumps(identity, sort_keys=True)
    if old and old[0] != encoded:
        parser.error('Recovery configuration changed; use a new output directory')
    if not old:
        with db:
            db.execute('INSERT INTO metadata VALUES (?)', (encoded,))
    print(f'Indexing the full {len(catalogue.tracks):,}-tracklet catalogue; {len(seeds):,} eligible seed modes', flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        endpoints = Time([float(catalogue.tracks['t'].min())-66, float(catalogue.tracks['t'].max())+2],
                         format='mjd', scale='utc').tdb.mjd
    ephemeris = Ephemeris(args.ephemeris, *endpoints)
    index = NightIndex(catalogue, sites, ephemeris, deadline)
    index.fast_retrieval = args.fast_retrieval
    attach_arcs(index, arc_groups, catalogue)
    geometry = recovery_geometry(args, catalogue, sites, ephemeris)
    counts = Counter(index.counts)
    counts['eligible_seed_modes'] = len(seeds)
    for (saved,) in db.execute('SELECT counts FROM seeds'):
        counts.update(json.loads(saved))
    print(f'Full index ready: {index.counts["indexed_tracklets"]:,} tracklets across {len(index.utc):,} nights', flush=True)
    atomic_json(args.output/'progress.json', dict(status='running', phase='index_ready',
        counts=dict(counts), updated_utc=time.time()))
    status = 'bounded_search_complete'
    completed = {key for (key,) in db.execute("SELECT key FROM seeds WHERE status IN ('searched','seed_refit_rejected','searched_deferred_fits')")}
    attempts = dict(db.execute('SELECT key, attempts FROM seeds'))
    remaining = [seed+(attempts.get(f'{seed[2]}:{seed[3]}', 0)+1,) for seed in seeds if f'{seed[2]}:{seed[3]}' not in completed]
    remaining.sort(key=lambda seed: seed[6])
    limited = bool(args.max_seeds and len(remaining) > args.max_seeds)
    if args.max_seeds:
        remaining = remaining[:args.max_seeds]
    counts['worker_processes'] = args.workers
    last_export = 0.
    try:
        for key, seed_status, seed_counts, accepted in seed_results(
                remaining, geometry, index, ephemeris, args, endpoints, deadline):
            with db:
                for candidate in accepted:
                    chain = ','.join(map(str, candidate['tracklet_indices']))
                    previous = db.execute('SELECT body FROM candidates WHERE key=?', (chain,)).fetchone()
                    combined = merge_candidate_rows(json.loads(previous[0]) if previous else None, candidate)
                    db.execute('INSERT OR REPLACE INTO candidates VALUES (?,?)', (chain, json.dumps(combined)))
                previous_counts = db.execute('SELECT counts FROM seeds WHERE key=?', (key,)).fetchone()
                if previous_counts:
                    counts.subtract(json.loads(previous_counts[0]))
                db.execute('INSERT INTO seeds VALUES (?,?,?,1) ON CONFLICT(key) DO UPDATE SET '
                           'status=excluded.status, counts=excluded.counts, attempts=seeds.attempts+1',
                           (key, seed_status, json.dumps(seed_counts)))
                counts.update(seed_counts)
                if seed_status == 'budget_expired':
                    status = 'time_budget_exhausted'
            if time.monotonic()-last_export >= 30.:
                counts['report_exports'] += 1
                report = export(db, args.output, catalogue, identity, counts, 'running', time.monotonic()-started)
                last_export = time.monotonic()
                print(f'Arc seeds {report["counts"]["seeds_completed"]:,}/{len(seeds):,}; '
                      f'{report["counts"]["seeds_retryable"]:,} retryable; '
                      f'{len(report["candidates"]):,} chains; '
                      f'{max(0, deadline-time.monotonic()):.0f}s remaining', flush=True)
        if time.monotonic() >= deadline:
            status = 'time_budget_exhausted'
        elif limited:
            status = 'seed_limit_reached'
        elif db.execute("SELECT COUNT(*) FROM seeds WHERE status NOT IN ('searched','seed_refit_rejected','searched_deferred_fits')").fetchone()[0]:
            status = 'retryable_work_remaining'
        elif counts.get('deferred_full_fits'):
            status = 'deferred_full_fits_remaining'
    except BudgetExpired:
        status = 'time_budget_exhausted'
    except Exception:
        status = 'compute_error'
        raise
    finally:
        counts['report_exports'] += 1
        report = export(db, args.output, catalogue, identity, counts, status, time.monotonic()-started)
        write_coverage(args.output/'report.json', catalogue, sites)
        db.close()
    print(json.dumps(dict(status=report['status'], counts=report['counts']), indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
