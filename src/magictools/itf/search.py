"""Whole-catalogue sparse candidate search and incremental multi-night linkage."""
from collections import defaultdict, deque
from contextlib import closing
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
import heapq
import multiprocessing
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from .catalogue import Catalogue
from .fit import BudgetExpired, Geometry, check_deadline, fit_chain
from .kernels import ARCSEC, directions


@dataclass
class SearchConfig:
    workers: int = 4
    max_gap_days: int = 14
    position_gate_arcsec: float = 30.
    curvature_gate_arcsec_per_day2: float = 1.
    neighbors_per_day: int = 8
    max_edges: int = 300000
    max_seeds: int = 10000
    seed_neighbors: int = 5  # Zero considers every retained graph neighbor.
    broad_search: bool = False
    sparse_curved: bool = False  # Opt-in equivalent sparse-night benchmark acceleration.
    branches: int = 4
    max_parent_seeds: int = 16
    joint_extensions: bool = True
    sigma_arcsec: float = .3
    rms_limit_arcsec: float = .6
    prior_link_probability: float = .001
    seconds: float = 280.

    @classmethod
    def for_profile(cls, profile):
        if profile == 'fast':
            return cls()
        if profile == 'broad':
            return cls(max_gap_days=28, neighbors_per_day=16, max_edges=1000000,
                       max_seeds=70000, seed_neighbors=0, broad_search=True, seconds=3600.)
        raise ValueError(f'Unknown search profile: {profile}')

    def validate(self):
        if not isinstance(self.seed_neighbors, int) or self.seed_neighbors < 0:
            raise ValueError('seed_neighbors must be a nonnegative integer')
        for name in ('workers', 'max_gap_days', 'neighbors_per_day', 'max_edges', 'max_seeds',
                     'branches', 'max_parent_seeds'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('position_gate_arcsec', 'curvature_gate_arcsec_per_day2',
                     'sigma_arcsec', 'rms_limit_arcsec', 'seconds'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not 0 < self.prior_link_probability < 1:
            raise ValueError('prior_link_probability must be between 0 and 1')


def _candidate_edges_serial(tracks, config, deadline, progress, counts, partition=None):
    """Symmetric tangent extrapolation, indexed on the unit sphere (RA wrap safe).

    All tracklets are accounted for. Rate-less rows cannot generate seeds but stay
    in the cache. KD-tree neighbor limits and edge-pool cuts are counted explicitly.
    """
    usable = (tracks['count'] >= 2) & (tracks['span'] > 0) & (tracks['span'] <= .6)
    counts['tracklets_total'] = len(tracks)
    counts['seed_eligible_tracklets'] = int(usable.sum())
    counts['rate_unavailable_tracklets'] = int(np.sum(tracks['span'] <= 0))
    counts['long_tracklets_not_seeded'] = int(np.sum(tracks['span'] > .6))
    chosen = np.flatnonzero(usable)
    day = np.floor(tracks['t'][chosen]).astype(int)
    order = np.argsort(day, kind='stable')
    chosen, day = chosen[order], day[order]
    days, starts = np.unique(day, return_index=True)
    ends = np.r_[starts[1:], len(chosen)]
    groups = {int(d): chosen[a:b] for d, a, b in zip(days, starts, ends)}
    edge_parts = []
    retained = 0
    counts.update(day_pairs_scanned=0, neighbor_queries_at_cap=0, edges_proposed=0,
                  edge_pool_discarded=0, tracklets_scanned=0)
    last_update = time.monotonic()
    complete = True
    for day_index, (d, a) in enumerate(groups.items()):
        if partition is not None and day_index % partition[1] != partition[0]:
            continue
        try:
            check_deadline(deadline)
        except BudgetExpired:
            complete = False
            break
        for gap in range(1, config.max_gap_days+1):
            if d+gap not in groups:
                continue
            check_deadline(deadline)
            b = groups[d+gap]
            ref = d+.5+gap/2
            left = tracks['u'][a]+tracks['v'][a]*(ref-tracks['t'][a])[:, None]
            right = tracks['u'][b]+tracks['v'][b]*(ref-tracks['t'][b])[:, None]
            left /= np.linalg.norm(left, axis=1)[:, None]
            right /= np.linalg.norm(right, axis=1)[:, None]
            radius = (config.position_gate_arcsec+config.curvature_gate_arcsec_per_day2*gap*gap)/ARCSEC
            k = min(config.neighbors_per_day+1, len(b))
            distances, neighbors = cKDTree(right).query(left, k=list(range(1, k+1)),
                                                       distance_upper_bound=radius, workers=1)
            if k > config.neighbors_per_day:
                counts['neighbor_queries_at_cap'] += int(np.isfinite(distances[:, -1]).sum())
                distances, neighbors = distances[:, :-1], neighbors[:, :-1]
            rows, cols = np.where(np.isfinite(distances))
            if len(rows):
                ai, bi = a[rows], b[neighbors[rows, cols]]
                dt = tracks['t'][bi]-tracks['t'][ai]
                # Separate nights, no aliases of the same designation.
                keep = (dt >= .3) & (dt <= config.max_gap_days) & (tracks['id'][ai] != tracks['id'][bi])
                ai, bi, dt = ai[keep], bi[keep], dt[keep]
                # Score candidates by angular mismatch, accounting for rate uncertainty.
                rate_noise = config.sigma_arcsec*np.sqrt(12/tracks['count'][ai])/tracks['span'][ai]
                rate_noise += config.sigma_arcsec*np.sqrt(12/tracks['count'][bi])/tracks['span'][bi]
                scale = config.position_gate_arcsec+.5*rate_noise*dt
                mismatch = distances[rows[keep], cols[keep]]*ARCSEC
                dv = np.linalg.norm(tracks['v'][ai]-tracks['v'][bi], axis=1)*ARCSEC
                # Gaussian location negative log density: broad rate uncertainty
                # must not push weakly constrained two-point arcs to the front.
                score = .5*(mismatch/scale)**2+2*np.log(scale/config.position_gate_arcsec)
                score += .01*dv*dt/scale
                part = np.column_stack([ai, bi, score])
                counts['edges_proposed'] += len(part)
                edge_parts.append(part)
                retained += len(part)
            counts['day_pairs_scanned'] += 1
            if retained > 2*config.max_edges:
                combined = np.concatenate(edge_parts)
                take = np.argpartition(combined[:, 2], config.max_edges-1)[:config.max_edges]
                counts['edge_pool_discarded'] += retained-config.max_edges
                edge_parts, retained = [combined[take]], config.max_edges
        counts['tracklets_scanned'] += len(a)
        if time.monotonic()-last_update > 15:
            progress(f'Scanned {day_index+1:,}/{len(groups):,} nights; {counts["edges_proposed"]:,} candidate edges')
            last_update = time.monotonic()
    edges = np.concatenate(edge_parts) if edge_parts else np.empty((0, 3))
    order = np.argsort(edges[:, 2], kind='stable')
    counts['edge_pool_discarded'] += max(0, len(edges)-config.max_edges)
    edges = edges[order[:config.max_edges]]
    counts['edges_retained'] = len(edges)
    counts['full_catalogue_scan_complete'] = complete
    return edges


def _scan_worker(arguments):
    filename, settings, deadline, partition = arguments
    tracks = np.load(filename, mmap_mode='r', allow_pickle=False)
    counts = {}
    edges = _candidate_edges_serial(tracks, SearchConfig(**settings), deadline, lambda message: None,
                                    counts, partition)
    return edges, counts


def candidate_edges(tracks, config, deadline, progress, counts):
    """Partition day-pair work; each day pair still receives the same exact query."""
    filename = getattr(tracks, 'filename', None)
    if config.workers == 1 or filename is None:
        return _candidate_edges_serial(tracks, config, deadline, progress, counts)
    context = multiprocessing.get_context('fork' if sys.platform.startswith('linux') else 'spawn')
    jobs = [(str(filename), asdict(config), deadline, (i, config.workers)) for i in range(config.workers)]
    results = []
    with ProcessPoolExecutor(max_workers=config.workers, mp_context=context) as pool:
        futures = [pool.submit(_scan_worker, job) for job in jobs]
        remaining = set(futures)
        last_update = time.monotonic()
        while remaining:
            done, remaining = wait(remaining, timeout=1., return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())
            if time.monotonic()-last_update > 15:
                progress(f'Full catalogue scan: {len(results)}/{config.workers} day partitions complete')
                last_update = time.monotonic()
    additive = ('day_pairs_scanned', 'neighbor_queries_at_cap', 'edges_proposed',
                'edge_pool_discarded', 'tracklets_scanned')
    for key, value in results[0][1].items():
        if key not in additive:
            counts[key] = value
    for key in additive:
        counts[key] = sum(part[key] for _, part in results)
    counts['full_catalogue_scan_complete'] = all(part['full_catalogue_scan_complete'] for _, part in results)
    edges = np.concatenate([part for part, _ in results])
    order = np.lexsort((edges[:,1], edges[:,0], edges[:,2]))
    counts['edge_pool_discarded'] += max(0, len(edges)-config.max_edges)
    edges = edges[order[:config.max_edges]]
    counts['edges_retained'] = len(edges)
    return edges


def make_seeds(edges, tracks, config, counts, deadline=np.inf):
    adjacent = defaultdict(list)
    for a, b, score in edges:
        a, b = int(a), int(b)
        adjacent[a].append((float(score), b))
        adjacent[b].append((float(score), a))
    # Every retained edge contributes; bounded neighbor combinations avoid O(degree^2).
    seeds = {}
    considered = 0
    score_ceiling = np.inf
    times, identifiers = tracks['t'], tracks['id']
    for ordinal, (middle, links) in enumerate(adjacent.items()):
        if ordinal % 1024 == 0:
            check_deadline(deadline)
        links.sort()
        top = links[:config.seed_neighbors] if config.seed_neighbors else links
        tm, identifier_middle = times[middle], identifiers[middle]
        for i, (score_a, a) in enumerate(top):
            for score_b, b in top[i+1:]:
                score = score_a+score_b
                # Sorted links make this an exact ranking bound. Equal scores
                # are still visited so deterministic triple-key ties survive.
                if score > score_ceiling:
                    break
                considered += 1
                ta, tb = times[a], times[b]
                if abs(ta-tm) < .3 or abs(tb-tm) < .3 or abs(ta-tb) < .3:
                    continue
                if identifiers[a] == identifier_middle or identifiers[b] == identifier_middle or identifiers[a] == identifiers[b]:
                    continue
                triple = tuple(sorted((a, middle, b)))
                if triple not in seeds or score < seeds[triple]:
                    seeds[triple] = score
        # Keep memory proportional to configured budget even in a crowded catalogue.
        if len(seeds) > 2*config.max_seeds:
            seeds = dict(heapq.nsmallest(config.max_seeds, seeds.items(), key=lambda p: (p[1], p[0])))
            score_ceiling = max(seeds.values())
    ordered = heapq.nsmallest(config.max_seeds, seeds, key=lambda key: (seeds[key], key))
    counts['seed_combinations_considered'] = considered
    counts['seeds_retained'] = len(ordered)
    counts['seed_search_bounded'] = True
    return ordered, adjacent


def annotate_candidates(ranked, tracks):
    """Expose containment and overlap without discarding competing hypotheses."""
    membership = defaultdict(set)
    members = [set(row['tracklet_indices']) for row in ranked]
    for rank, indices in enumerate(members, 1):
        for i in indices:
            membership[i].add(rank)
    for rank, row in enumerate(ranked, 1):
        indices = members[rank-1]
        overlaps = set().union(*(membership[i] for i in indices))-{rank}
        contains = {j for j in overlaps if members[j-1] < indices}
        contained_by = {j for j in overlaps if indices < members[j-1]}
        row.update(rank=rank,
                   tracklets=[tracks[i]['id'].decode().strip() for i in row['tracklet_indices']],
                   stations=sorted({tracks[i]['site'].decode() for i in indices}),
                   overlapping_ranks=sorted(overlaps), contains_ranks=sorted(contains),
                   contained_by_ranks=sorted(contained_by),
                   partially_overlapping_ranks=sorted(overlaps-contains-contained_by),
                   maximal=not contained_by)
    return dict(maximal_candidates=sum(row['maximal'] for row in ranked),
                contained_candidates=sum(not row['maximal'] for row in ranked),
                candidates_with_partial_overlaps=sum(bool(row['partially_overlapping_ranks']) for row in ranked),
                distinct_candidate_tracklets=len(membership),
                accepted_chains_with_close_epochs=sum(bool(np.min(np.diff(np.sort(tracks['t'][list(indices)]))) < .3)
                                                     for indices in members),
                weak_best_orbits=sum(row['modes'][0]['local_uncertainty']['local_linearization_weak']
                                     for row in ranked))


_FIT_GEOMETRY = None


def _initialize_fit_worker(inherited, path, sites, indices, deadline):
    global _FIT_GEOMETRY
    if inherited is not None:
        # Linux fork shares the read-only prepared arrays through copy-on-write.
        _FIT_GEOMETRY = inherited
    else:
        # Spawn platforms reopen mmap arrays rather than pickle the catalogue.
        _FIT_GEOMETRY = Geometry(Catalogue(path), sites)
        try:
            _FIT_GEOMETRY.prepare(indices, deadline)
        except BudgetExpired:
            pass  # Submitted fits will report the expired deadline normally.


def _fit_worker(indices, settings):
    missing = [i for i in indices if i not in _FIT_GEOMETRY.prepared]
    if missing:
        _FIT_GEOMETRY.prepare(missing, settings['deadline'])
    return fit_chain(_FIT_GEOMETRY, indices, **settings)


def solve(catalogue, sites, config, progress=print, *, deadline=None, audit=None, batch_fitter=None, geometry=None):
    # The observer receives proposals/outcomes, never supplies labels or decisions.
    def emit(event, **fields):
        if audit is not None:
            audit(dict(event=event, **fields))
    config.validate()
    started = time.monotonic()
    deadline = started+config.seconds if deadline is None else deadline
    counts = dict(fits_attempted=0, fits_rejected=0, fits_observer_unavailable=0,
                  fits_numerically_failed=0, successful_extensions=0, triple_seeds_started=0,
                  triple_seeds_completed=0, fits_interrupted=0, fits_cancelled=0)
    rows, fitted, expanded, covered_extensions = {}, set(), set(), set()
    counts['joint_extension_proposals'] = 0
    if geometry is None:
        geometry = Geometry(catalogue, sites)
    elif geometry.catalogue.manifest != catalogue.manifest or set(geometry.sites) != set(sites):
        raise ValueError('Injected geometry uses a different catalogue or observatory table')
    # Existing atlas constructors verify the observatory-file hash. Compare
    # actual coordinates too so other injected implementations cannot bypass it.
    for code in sites:
        if not np.array_equal(np.asarray(geometry.sites[code].value), np.asarray(sites[code].value)):
            raise ValueError('Injected geometry observatory coordinates differ')
    timings = {}
    status = 'search_limits_reached'
    executor = None
    pending = {}
    try:
        protected_seeds = []
        edges = candidate_edges(catalogue.tracks, config, deadline, progress, counts)
        if config.broad_search:
            from .broad import augment_edges
            edges = augment_edges(edges, catalogue.tracks, sites, config, deadline, progress, counts, protected_seeds)
        timings['candidate_scan_seconds'] = time.monotonic()-started
        check_deadline(deadline)
        seeds, adjacent = make_seeds(edges, catalogue.tracks, config, counts, deadline)
        if protected_seeds:
            protected_keys = set(protected_seeds)
            seeds = protected_seeds+[seed for seed in seeds if seed not in protected_keys][:config.max_seeds-len(protected_seeds)]
            counts['seeds_retained'] = len(seeds)
            counts['protected_fast_seeds'] = len(protected_seeds)
        timings['seed_generation_seconds'] = time.monotonic()-started-timings['candidate_scan_seconds']
        if audit is not None:
            emit('seed_bank', seeds=[list(map(int, seed)) for seed in seeds])
        progress(f'Full scan complete: {len(edges):,} retained edges; {len(seeds):,} ranked triple seeds')
        prepare_start = time.monotonic()
        geometry.prepare(np.array(seeds, dtype=int).ravel(), deadline)
        timings['observer_preparation_seconds'] = time.monotonic()-prepare_start
        if batch_fitter is not None:
            counts['fit_worker_start_method'] = 'injected_batch'
        elif config.workers == 1:
            executor = ThreadPoolExecutor(max_workers=1)
            counts['fit_worker_start_method'] = 'thread'
        else:
            method = 'fork' if sys.platform.startswith('linux') else 'spawn'
            executor = ProcessPoolExecutor(max_workers=config.workers,
                mp_context=multiprocessing.get_context(method), initializer=_initialize_fit_worker,
                initargs=(geometry if method == 'fork' else None, str(catalogue.path), sites,
                          np.array(sorted(geometry.prepared), dtype=int), deadline))
            counts['fit_worker_start_method'] = method
        last_update = time.monotonic()

        def parallel_fits(jobs, triple_phase):
            nonlocal last_update
            jobs = iter(jobs)
            if batch_fitter is not None:
                accepted = []
                batch_size = getattr(batch_fitter, 'batch_size', 128)
                if not isinstance(batch_size, int) or not 1 <= batch_size <= 4096:
                    raise ValueError('Batch size must be an integer in [1, 4096]')
                issued = deque()
                def batches():
                    while True:
                        batch = []
                        while len(batch) < batch_size:
                            check_deadline(deadline)
                            try:
                                indices, extra = next(jobs)
                            except StopIteration:
                                break
                            key = tuple(sorted(map(int, indices)))
                            if key in fitted:
                                continue
                            fitted.add(key)
                            if triple_phase:
                                counts['triple_seeds_started'] += 1
                            if any(catalogue.tracks[i]['site'] not in sites for i in key):
                                counts['fits_observer_unavailable'] += 1
                                emit('fit', indices=key, outcome='observer_unavailable')
                                if triple_phase:
                                    counts['triple_seeds_completed'] += 1
                                continue
                            settings = dict(sigma=config.sigma_arcsec, rms_limit=config.rms_limit_arcsec,
                                            prior=config.prior_link_probability, deadline=deadline,
                                            extra_seeds=extra, broad_starts=config.broad_search)
                            batch.append((key, settings))
                            emit('submitted', indices=key)
                        if not batch:
                            return
                        issued.append(tuple(key for key, _ in batch))
                        yield batch
                def serial():
                    for batch in batches():
                        yield batch, batch_fitter(geometry, batch)
                iterator = (batch_fitter.iter_batches(geometry, batches())
                            if hasattr(batch_fitter, 'iter_batches') else serial())
                try:
                    with closing(iterator):
                        for batch, outcomes in iterator:
                            if not issued or tuple(key for key, _ in batch) != issued[0]:
                                raise ValueError('Batch fitter changed submission order or membership')
                            issued.popleft()
                            if len(outcomes) != len(batch):
                                raise ValueError('Batch fitter omitted outcomes')
                            for (key, _), result in zip(batch, outcomes):
                                counts['fits_attempted'] += 1
                                if triple_phase:
                                    counts['triple_seeds_completed'] += 1
                                emit('fit', indices=key, outcome='retained' if result else 'not_retained',
                                     rms_arcsec=result.get('rms_arcsec') if result else None,
                                     log_odds_bic=result.get('log_odds_bic') if result else None)
                                if result is None:
                                    counts['fits_rejected'] += 1
                                else:
                                    rows[key] = result
                                    accepted.append(result)
                                    if not triple_phase:
                                        counts['successful_extensions'] += 1
                    if issued:
                        raise ValueError('Batch fitter omitted submitted batches')
                    return accepted
                finally:
                    counts['batch_submitted_not_returned'] = counts.get('batch_submitted_not_returned', 0)+sum(map(len, issued))
            exhausted = False
            accepted = []
            while pending or not exhausted:
                check_deadline(deadline)
                while not exhausted and len(pending) < 2*config.workers:
                    try:
                        indices, extra = next(jobs)
                    except StopIteration:
                        exhausted = True
                        break
                    key = tuple(sorted(map(int, indices)))
                    if key in fitted:
                        continue
                    fitted.add(key)
                    if triple_phase:
                        counts['triple_seeds_started'] += 1
                    if any(catalogue.tracks[i]['site'] not in sites for i in key):
                        counts['fits_observer_unavailable'] += 1
                        emit('fit', indices=key, outcome='observer_unavailable')
                        if triple_phase:
                            counts['triple_seeds_completed'] += 1
                        continue
                    settings = dict(sigma=config.sigma_arcsec, rms_limit=config.rms_limit_arcsec,
                                    prior=config.prior_link_probability, deadline=deadline, extra_seeds=extra,
                                    broad_starts=config.broad_search)
                    if config.workers == 1:
                        future = executor.submit(fit_chain, geometry, key, **settings)
                    else:
                        future = executor.submit(_fit_worker, key, settings)
                    pending[future] = key
                    emit('submitted', indices=key)
                if not pending:
                    continue
                done, _ = wait(pending, timeout=max(.001, min(1., deadline-time.monotonic())),
                               return_when=FIRST_COMPLETED)
                for future in done:
                    key = pending.pop(future)
                    counts['fits_attempted'] += 1
                    try:
                        result = future.result()
                    except BudgetExpired:
                        counts['fits_interrupted'] += 1
                        emit('fit', indices=key, outcome='budget_exhausted')
                        raise
                    except (FloatingPointError, np.linalg.LinAlgError):
                        counts['fits_numerically_failed'] += 1
                        emit('fit', indices=key, outcome='numerical_failure')
                        result = None
                    else:
                        emit('fit', indices=key, outcome='retained' if result else 'not_retained',
                             rms_arcsec=result.get('rms_arcsec') if result else None,
                             log_odds_bic=result.get('log_odds_bic') if result else None)
                    if triple_phase:
                        counts['triple_seeds_completed'] += 1
                    if result is None:
                        counts['fits_rejected'] += 1
                    else:
                        rows[key] = result
                        accepted.append(result)
                        if not triple_phase:
                            counts['successful_extensions'] += 1
                if time.monotonic()-last_update > 15:
                    progress(f'Completed {counts["triple_seeds_completed"]:,}/{len(seeds):,} triple seeds; '
                             f'{len(rows):,} accepted chains; {max(0., deadline-time.monotonic()):.0f}s remaining')
                    last_update = time.monotonic()
            return accepted

        seed_start = time.monotonic()
        frontier = parallel_fits(((seed, ()) for seed in seeds), True)
        timings['triple_fitting_seconds'] = time.monotonic()-seed_start
        progress(f'All {len(seeds):,} triple seeds completed; growing accepted chains')
        extension_start = time.monotonic()
        while frontier:
            check_deadline(deadline)
            neighbor_ids = {other for row in frontier for i in row['tracklet_indices']
                            for _, other in adjacent[i] if other not in row['tracklet_indices']}
            geometry.prepare(neighbor_ids, deadline)
            jobs, joint_jobs, children, joint_children = {}, {}, {}, {}

            def enqueue(destination, key, extra):
                if key not in fitted:
                    starts = destination.setdefault(key, [])
                    starts.extend(extra[:max(0, config.max_parent_seeds-len(starts))])

            # Shared joint proposals receive starts from the strongest parents
            # first. Keep the full Gauss/range grid, with bounded extra starts.
            for current in sorted(frontier, key=lambda row: (-row['log_odds_bic'], row['tracklet_indices'])):
                check_deadline(deadline)
                key = tuple(current['tracklet_indices'])
                if key in expanded:
                    continue
                expanded.add(key)
                members = set(key)
                member_ids = set(catalogue.tracks['id'][list(members)])
                proposals = {other for i in members for _, other in adjacent[i] if other not in members}
                children[key] = []
                joint_children[key] = []
                states = [(np.array(mode['state_au_au_per_day']), mode['epoch_mjd_tdb'])
                          for mode in current['modes']]
                groups = [{} for _ in states]
                for other in sorted(proposals):
                    check_deadline(deadline)
                    identifier = catalogue.tracks[other]['id']
                    if identifier in member_ids:
                        continue
                    # The seed already spans three separated epochs. Same-night
                    # additions can supply real observations (including another
                    # station); proximity in time alone must not veto a refit.
                    if catalogue.tracks[other]['site'] not in sites:
                        continue
                    _, t, u, obs = geometry.track(int(other))
                    compatible = []
                    for mode_index, (state, epoch) in enumerate(states):
                        pred = directions(state, t-epoch, obs)
                        error = np.linalg.norm(pred-u, axis=1)*ARCSEC
                        if np.isfinite(error).all() and np.max(error) <= max(5., 10*config.rms_limit_arcsec):
                            compatible.append((state, epoch))
                            choice = (float(np.max(error)), other)
                            if identifier not in groups[mode_index] or choice < groups[mode_index][identifier]:
                                groups[mode_index][identifier] = choice
                    if compatible:
                        extended = tuple(sorted(members | {other}))
                        children[key].append(extended)
                        enqueue(jobs, extended, compatible)
                if config.joint_extensions:
                    for group, seed in zip(groups, states):
                        if len(group) > 1:
                            extended = tuple(sorted(members | {other for _, other in group.values()}))
                            joint_children[key].append(extended)
                            enqueue(joint_jobs, extended, [seed])
            counts['joint_extension_proposals'] += len(joint_jobs)
            parallel_fits(joint_jobs.items(), False)
            covered = set()
            for parent, alternatives in children.items():
                accepted_joint = [set(key) for key in joint_children[parent] if key in rows]
                for key in alternatives:
                    if any(set(key) <= joint for joint in accepted_joint):
                        covered.add(key)
                alternatives.extend(joint_children[parent])
            # A successful joint fit already explains these additions together.
            # Retain the parent and joint modes, without enumerating every subset.
            covered_extensions.update(covered-fitted)
            parallel_fits(((key, extra) for key, extra in jobs.items() if key not in covered), False)
            next_frontier = {}
            for alternatives in children.values():
                accepted = [rows[key] for key in set(alternatives) if key in rows]
                for row in sorted(accepted, key=lambda r: (-r['log_odds_bic'], r['tracklet_indices']))[:config.branches]:
                    next_frontier[tuple(row['tracklet_indices'])] = row
            frontier = list(next_frontier.values())
        timings['extension_seconds'] = time.monotonic()-extension_start
        status = 'bounded_search_complete'
    except BudgetExpired:
        status = 'time_budget_exhausted'
    finally:
        for future in pending:
            if future.cancel():
                counts['fits_cancelled'] += 1
                emit('fit', indices=pending[future], outcome='cancelled')
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        # Preserve completed work when the deadline coincides with a worker return.
        for future, key in pending.items():
            if future.cancelled():
                continue
            counts['fits_attempted'] += 1
            try:
                result = future.result()
            except BudgetExpired:
                counts['fits_interrupted'] += 1
                emit('fit', indices=key, outcome='budget_exhausted')
                continue
            except (FloatingPointError, np.linalg.LinAlgError):
                counts['fits_numerically_failed'] += 1
                emit('fit', indices=key, outcome='numerical_failure')
                continue
            emit('fit', indices=key, outcome='retained' if result else 'not_retained',
                 rms_arcsec=result.get('rms_arcsec') if result else None,
                 log_odds_bic=result.get('log_odds_bic') if result else None)
            if len(key) == 3:
                counts['triple_seeds_completed'] += 1
            if result is None:
                counts['fits_rejected'] += 1
            else:
                rows[key] = result
                if len(key) > 3:
                    counts['successful_extensions'] += 1
    timings['solve_seconds'] = time.monotonic()-started
    counts.setdefault('full_catalogue_scan_complete', False)
    counts['triple_seeds_remaining'] = counts.get('seeds_retained', 0)-counts['triple_seeds_completed']
    counts['individual_extension_subsets_covered'] = len(covered_extensions)
    ranked = sorted(rows.values(), key=lambda r: (-r['log_odds_bic'], r['tracklet_indices']))
    counts.update(annotate_candidates(ranked, catalogue.tracks))
    return dict(schema_version=1, status=status, full_objective_complete=False,
                exhaustive=False, probabilities_calibrated=False,
                probability_scope='Separate BIC link-versus-independent-tracklets comparisons; '
                                  'overlapping rows are alternatives, not independent discoveries. '
                                  'Mode weights are normalized peak likelihoods, not integrated posterior masses.',
                model='All-conic heliocentric two-body; topocentric builtin ephemerides; '
                      'UTC to TDB; iterated light time; independent Gaussian astrometry',
                limitations=['Finite angular/time gates and nearest-neighbor/seed/branch limits can miss links.',
                             'Joint growth skips individual subsets covered by an accepted longer chain; subset-only orbit alternatives can be missed.',
                             'No completeness guarantee for any orbital population.',
                             'No planetary perturbations, debiasing or survey-specific astrometric covariance.',
                             'BIC probabilities are prior-sensitive and not empirically calibrated.',
                             'No posterior integration or global exclusive catalogue assignment.',
                             'Unsupported observer records and rate-less tracklets are not seed-linked.'],
                config=asdict(config), input=catalogue.manifest, counts=counts,
                timings=timings, candidates=ranked)
