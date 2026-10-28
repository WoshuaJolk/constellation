"""Additional distance-conditioned proposals; approximate geometry, exact refits.

Project each short arc to a shared epoch and geocentric observer using a trial
range/range rate, terrestrial rotation, Earth reflex motion and solar acceleration.
This is a proposal model only: every accepted association still passes the full
topocentric, light-time-corrected orbit fit. It is not a completeness guarantee.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, replace
import multiprocessing
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

from astropy import units as au
from astropy.time import Time
import erfa
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial import cKDTree

from .fit import check_deadline
from .kernels import ARCSEC, MU


def approximate_observers(tracks, sites, deadline=np.inf):
    """Bounded-memory proposal geometry, including diurnal position and velocity.

    Daily Hermite interpolation uses builtin Earth ephemerides. Site rotation
    uses UT1 approximately UTC and mean precession, omitting polar motion and
    nutation. The fitter uses its existing higher-accuracy observer calculation.
    """
    times = tracks['t']
    days = np.arange(np.floor(times.min())-2, np.ceil(times.max())+3)
    utc = Time(days, format='mjd', scale='utc')
    tdb = utc.tdb
    earth, _ = erfa.epv00(tdb.jd1, tdb.jd2)
    spline = CubicHermiteSpline(days, earth['p'], earth['v'])
    precession = erfa.pmat06(utc.tt.jd1, utc.tt.jd2)
    positions = np.full((len(tracks), 3), np.nan)
    velocities = np.full_like(positions, np.nan)
    site_xyz = {code: np.array([site.x.to_value(au.au), site.y.to_value(au.au),
                                site.z.to_value(au.au)]) for code, site in sites.items()}
    omega = 2*np.pi*1.00273781191135448
    for start in range(0, len(tracks), 65536):
        check_deadline(deadline)
        end = min(start+65536, len(tracks))
        tt = times[start:end]
        local = np.array([site_xyz.get(code, (np.nan, np.nan, np.nan))
                          for code in tracks['site'][start:end]])
        utc = Time(tt, format='mjd', scale='utc')
        angle = erfa.gmst06(2400000.5, tt, utc.tt.jd1, utc.tt.jd2)
        c, s = np.cos(angle), np.sin(angle)
        rotated = np.column_stack((c*local[:, 0]-s*local[:, 1],
                                   s*local[:, 0]+c*local[:, 1], local[:, 2]))
        rotation = precession[(np.floor(tt)-days[0]).astype(int)]
        offset = np.einsum('nji,nj->ni', rotation, rotated)
        rate = np.column_stack((-omega*rotated[:, 1], omega*rotated[:, 0], np.zeros(len(tt))))
        positions[start:end] = spline(tt)+offset
        velocities[start:end] = spline(tt, 1)+np.einsum('nji,nj->ni', rotation, rate)
    return positions, velocities, days, earth['p'], earth['v']


def project(tracks, observers, velocities, ref, earth_ref, distance, radial):
    """Trial range is defined at ref; do not constrain orbital plane or energy."""
    dt = ref-tracks['t']
    rho = distance-radial*dt
    r = observers+rho[:, None]*tracks['u']
    v = velocities+rho[:, None]*tracks['v']+radial*tracks['u']
    norm = np.linalg.norm(r, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        acceleration = -MU*r/norm[:, None]**3
        predicted = r+dt[:, None]*v+.5*dt[:, None]**2*acceleration-earth_ref
        predicted /= np.linalg.norm(predicted, axis=1)[:, None]
    valid = (rho > .001) & (norm > .01) & np.isfinite(predicted).all(axis=1)
    return predicted, valid


def distance_hypotheses(max_gap):
    for distance in (.03, .1, .3, 1., 3., 10., 50., 250.):
        gap_limit = min(max_gap, 7 if distance < .3 else 14 if distance < 1 else max_gap)
        for radial in ((0., -.02, .02) if distance <= .3 else (0.,)):
            yield distance, radial, gap_limit


def _curved_scan(tracks, positions, velocities, days, earth, earth_velocity,
                 config, deadline, partition=(0, 1), progress=lambda _: None, hypotheses=None):
    hypotheses = list(distance_hypotheses(config.max_gap_days)) if hypotheses is None else hypotheses
    chosen = np.flatnonzero((tracks['count'] >= 2) & (tracks['span'] > 0) &
                            (tracks['span'] <= .6) & np.isfinite(positions).all(axis=1))
    night = np.floor(tracks['t'][chosen]).astype(int)
    order = np.argsort(night, kind='stable')
    chosen, night = chosen[order], night[order]
    nights, starts = np.unique(night, return_index=True)
    groups = {int(d): chosen[a:b] for d, a, b in zip(nights, starts, np.r_[starts[1:], len(chosen)])}
    ephemeris = CubicHermiteSpline(days, earth, earth_velocity)
    edges = np.empty((0, 3))
    parts, retained = [], 0
    counts = dict(curved_queries=0, curved_queries_at_cap=0, curved_edges_proposed=0,
                  curved_edge_pool_discarded=0, curved_day_pairs_scanned=0)
    for ordinal, (day, a) in enumerate(groups.items()):
        if ordinal % partition[1] != partition[0]:
            continue
        check_deadline(deadline)
        for distance, radial, gap_limit in hypotheses:
            for gap in range(1, gap_limit+1):
                if day+gap not in groups:
                    continue
                check_deadline(deadline)
                b = groups[day+gap]
                ref = day+.5+gap/2
                earth_ref = ephemeris(ref)
                left, good_a = project(tracks[a], positions[a], velocities[a], ref, earth_ref, distance, radial)
                right, good_b = project(tracks[b], positions[b], velocities[b], ref, earth_ref, distance, radial)
                aa, bb = a[good_a], b[good_b]
                if not len(aa) or not len(bb):
                    continue
                # Allow coarse trial-distance and curvature approximation error.
                # This wider gate affects proposals only, never fit acceptance.
                gate = config.position_gate_arcsec+10*config.curvature_gate_arcsec_per_day2*gap*gap
                k = min(config.neighbors_per_day+1, len(bb))
                mismatch, neighbor = cKDTree(right[good_b]).query(left[good_a], k=list(range(1, k+1)),
                                    distance_upper_bound=min(2., gate/ARCSEC), workers=1)
                counts['curved_queries'] += len(aa)
                if k > config.neighbors_per_day:
                    counts['curved_queries_at_cap'] += int(np.isfinite(mismatch[:, -1]).sum())
                    mismatch, neighbor = mismatch[:, :-1], neighbor[:, :-1]
                ii, jj = np.where(np.isfinite(mismatch))
                if len(ii):
                    ai, bi = aa[ii], bb[neighbor[ii, jj]]
                    dt = tracks['t'][bi]-tracks['t'][ai]
                    keep = (dt >= .3) & (dt <= config.max_gap_days) & (tracks['id'][ai] != tracks['id'][bi])
                    ai, bi, dt = ai[keep], bi[keep], dt[keep]
                    noise = config.sigma_arcsec*np.sqrt(12/tracks['count'][ai])/tracks['span'][ai]
                    noise += config.sigma_arcsec*np.sqrt(12/tracks['count'][bi])/tracks['span'][bi]
                    scale = config.position_gate_arcsec+.5*noise*dt
                    score = .5*(mismatch[ii[keep], jj[keep]]*ARCSEC/scale)**2+2*np.log(scale/config.position_gate_arcsec)
                    # Avoid giving many distance trials a free ranking advantage.
                    score += np.log(len(hypotheses))
                    parts.append(np.column_stack((ai, bi, score)))
                    retained += len(ai)
                    counts['curved_edges_proposed'] += len(ai)
                counts['curved_day_pairs_scanned'] += 1
        if retained > config.max_edges:
            combined = unique_edges(np.concatenate([edges]+parts))
            counts['curved_edge_pool_discarded'] += max(0, len(combined)-config.max_edges)
            edges = best_edges(combined, config.max_edges)
            parts, retained = [], 0
    combined = unique_edges(np.concatenate([edges]+parts))
    counts['curved_edge_pool_discarded'] += max(0, len(combined)-config.max_edges)
    return best_edges(combined, config.max_edges), counts


def unique_edges(edges):
    if not len(edges):
        return edges
    order = np.lexsort((edges[:, 2], edges[:, 1], edges[:, 0]))
    ordered = edges[order]
    return ordered[np.r_[True, np.any(ordered[1:, :2] != ordered[:-1, :2], axis=1)]]


def best_edges(edges, limit):
    return edges[np.lexsort((edges[:, 1], edges[:, 0], edges[:, 2]))[:limit]]


def merge_proposals(protected, linear, curved, limit, counts):
    """Protect fast-search coverage and reserve half the remaining room for novel curves."""
    protected_keys = set(map(tuple, protected[:, :2].astype(int)))
    linear_keys = set(map(tuple, linear[:, :2].astype(int))) | protected_keys
    linear_extra = linear[[tuple(map(int, row[:2])) not in protected_keys for row in linear]]
    curved_extra = curved[[tuple(map(int, row[:2])) not in linear_keys for row in curved]]
    available = max(0, limit-len(protected))
    curved_count = min(len(curved_extra), max(available//2, available-len(linear_extra)))
    linear_count = min(len(linear_extra), available-curved_count)
    counts.update(protected_fast_edges=len(protected),
                  curved_unique_edges_added_before_cap=len(curved_extra),
                  curved_merge_edges_discarded=len(curved_extra)-curved_count,
                  expanded_linear_edges_discarded=len(linear_extra)-linear_count,
                  curved_edges_retained=curved_count)
    return best_edges(np.concatenate([protected, best_edges(linear_extra, linear_count),
                                       best_edges(curved_extra, curved_count)]), limit)


def _worker(arguments):
    directory, settings, deadline, partition = arguments
    from .search import SearchConfig
    arrays = [np.load(Path(directory)/(name+'.npy'), mmap_mode='r')
              for name in ('tracks', 'positions', 'velocities', 'days', 'earth', 'earth_velocity')]
    config = SearchConfig(**settings)
    if config.sparse_curved:
        from .sparse_curved import sparse_scan
        return sparse_scan(*arrays, config, deadline, partition)
    return _curved_scan(*arrays, config, deadline, partition)


def augment_edges(original, tracks, sites, config, deadline, progress, counts, protected_seed_bank=None):
    started = time.monotonic()
    counts['curved_scan_complete'] = False
    counts['linear_scan_complete'] = counts.get('full_catalogue_scan_complete', True)
    counts['full_catalogue_scan_complete'] = False
    # The expanded linear lane can saturate the graph on real ITF data. Protect
    # the original fast lane, rather than reserving the entire expanded graph.
    from .search import candidate_edges, make_seeds
    protected_config = replace(config, max_gap_days=min(14, config.max_gap_days),
                               neighbors_per_day=min(8, config.neighbors_per_day),
                               max_edges=min(300000, config.max_edges))
    protected_counts = {}
    protected = candidate_edges(tracks, protected_config, deadline, progress, protected_counts)
    check_deadline(deadline)
    if protected_seed_bank is not None:
        bank, _ = make_seeds(protected, tracks,
                             replace(protected_config, max_seeds=min(10000, config.max_seeds), seed_neighbors=5),
                             {}, deadline)
        protected_seed_bank.extend(bank)
    progress('Preparing distance-conditioned proposals for the full catalogue')
    geometry = approximate_observers(tracks, sites, deadline)
    counts['curved_observers_unavailable'] = int(np.sum(~np.isfinite(geometry[0]).all(axis=1)))
    counts['curved_geometry_seconds'] = time.monotonic()-started
    if config.workers == 1:
        scanner = _curved_scan
        if config.sparse_curved:
            from .sparse_curved import sparse_scan
            scanner = sparse_scan
        results = [scanner(tracks, *geometry, config, deadline)]
    else:
        with TemporaryDirectory(prefix='solve-itf-curved-') as directory:
            for name, array in zip(('tracks', 'positions', 'velocities', 'days', 'earth', 'earth_velocity'),
                                   (tracks, *geometry)):
                np.save(Path(directory)/(name+'.npy'), array, allow_pickle=False)
            del geometry
            context = multiprocessing.get_context('fork' if sys.platform.startswith('linux') else 'spawn')
            with ProcessPoolExecutor(max_workers=config.workers, mp_context=context) as pool:
                pending = {pool.submit(_worker, (directory, asdict(config), deadline, (i, config.workers)))
                           for i in range(config.workers)}
                results = []
                while pending:
                    done, pending = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                    for future in done:
                        results.append(future.result())
                    progress(f'Distance-conditioned scan: {len(results)}/{config.workers} partitions complete; '
                             f'{time.monotonic()-started:.0f}s elapsed')
    for key in results[0][1]:
        counts[key] = sum(result[1][key] for result in results)
    curved = unique_edges(np.concatenate([result[0] for result in results]))
    result = merge_proposals(protected, original, curved, config.max_edges, counts)
    counts['edges_retained'] = len(result)
    counts['curved_scan_complete'] = True
    counts['full_catalogue_scan_complete'] = counts['linear_scan_complete']
    counts['curved_scan_seconds'] = time.monotonic()-started
    progress(f'Retained {counts["curved_edges_retained"]:,} novel curved edges and '
             f'{len(protected):,} protected fast edges; {len(result):,} total')
    return result
