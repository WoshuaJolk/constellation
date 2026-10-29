"""Bounded residual-bank benchmark. Not a full linker/search benchmark yet."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def subset(bank, lo, hi):
    select = (bank['case_index'] >= lo) & (bank['case_index'] < hi)
    return dict(states=np.ascontiguousarray(bank['states'][lo:hi]),
                case_index=np.ascontiguousarray(bank['case_index'][select]-lo),
                sigma=bank['sigma'], **{k: np.ascontiguousarray(bank[k][select]) for k in
                ('offsets', 'observers', 'observed', 'east', 'north')})


def prepare(a):
    import ijson
    from magictools.itf.catalogue import Catalogue
    from magictools.itf.fit import read_sites
    from magictools.itf.nbody import Ephemeris, JPLGeometry
    cat = Catalogue(a.cache)
    rows = []
    with a.report.open('rb') as f:
        for row in ijson.items(f, 'candidates.item', use_float=True):
            rows.append(row)
            if len(rows) == a.limit:
                break
    if not rows:
        raise ValueError('No candidates')
    ids = sorted({int(i) for r in rows for i in r['tracklet_indices']})
    times = np.concatenate([cat.obs[int(cat.tracks[i]['start']):int(cat.tracks[i]['start'])+int(cat.tracks[i]['count'])]['t'] for i in ids])
    modes = [(r.get('modes') or [r])[0] for r in rows]
    epochs = [m['epoch_mjd_tdb'] for m in modes]
    ephem = Ephemeris(a.ephemeris, min(float(times.min()), min(epochs))-100,
                      max(float(times.max()), max(epochs))+2)
    geometry = JPLGeometry(cat, read_sites(a.observatories), ephem)
    arrays = {k: [] for k in ('states', 'case_index', 'offsets', 'observers', 'observed', 'east', 'north')}
    receipts = []
    try:
        geometry.prepare(ids)
        for i, (row, mode) in enumerate(zip(rows, modes)):
            t, u, observer, labels = geometry.get(row['tracklet_indices'])
            east = np.column_stack((-u[:, 1], u[:, 0], np.zeros(len(u))))
            east /= np.linalg.norm(east, axis=1)[:, None]
            arrays['states'].append(mode['state_au_au_per_day'])
            for k, v in dict(case_index=np.full(len(t), i, dtype=np.int64),
                             offsets=t-mode['epoch_mjd_tdb'], observers=observer,
                             observed=u, east=east, north=np.cross(u, east)).items():
                arrays[k].append(v)
            receipts.append(dict(source_ordinal=i, tracklet_indices=row['tracklet_indices'],
                                 epoch_mjd_tdb=mode['epoch_mjd_tdb'], observations=len(t),
                                 supplied_state=mode['state_au_au_per_day']))
    finally:
        ephem.kernel.close()
    bank = {k: np.ascontiguousarray(v if k == 'states' else np.concatenate(v),
                                  dtype=np.int64 if k == 'case_index' else np.float64)
            for k, v in arrays.items()}
    bank['sigma'] = np.array(.3)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **bank)
    a.output.with_suffix('.json').write_text(json.dumps(dict(
        selection='First N report rows in original order, first saved mode, no outcome-based replacement',
        scope='Saved retained candidates only; not representative of rejected search starts. Two-body residual evaluation at these states is not a planetary fit replay.',
        coordinates='Heliocentric ICRF AU/AU-day; offsets days TDB; exact JPL observer positions',
        report=str(a.report), report_sha256=digest(a.report), catalogue_manifest_sha256=digest(a.cache/'manifest.json'),
        ephemeris_sha256=digest(a.ephemeris), bank_sha256=digest(a.output), cases=receipts), indent=2))
    print(json.dumps(dict(bank=str(a.output), cases=len(rows), observations=len(bank['offsets']))))


def compare(reference, actual, sigma):
    # Fail closed on different validity, not just on the intersection of successes.
    if not np.array_equal(reference[2], actual[2]):
        raise ValueError('CPU/GPU validity masks disagree')
    good = reference[2].astype(bool)
    if not good.any():
        raise ValueError('No valid observations for parity')
    res_error = float(np.max(np.abs(reference[0][good]-actual[0][good]))*sigma)
    # Dimensionless scaled Jacobian error, with a floor for near-zero entries.
    jac_error = float(np.max(np.abs(reference[1][good]-actual[1][good]) /
                            np.maximum(1., np.abs(reference[1][good]))))
    if res_error > .001 or jac_error > 1e-6:
        raise ValueError(f'Numerical parity failed: residual={res_error}, Jacobian={jac_error}')
    return dict(max_residual_error_arcsec=res_error, max_scaled_jacobian_error=jac_error)


def run(a):
    from magictools.itf.gpu import ResidentBank, cpu_reference, validate_bank
    from numba import cuda
    with np.load(a.bank, allow_pickle=False) as z:
        bank = dict(z)
    validate_bank(bank)
    if np.any(np.diff(bank['case_index']) < 0):
        raise ValueError('Benchmark requires observations grouped in candidate order')
    result = dict(scope='Residual/Jacobian stage only; no optimization, chaining, or scientific acceptance',
                  bank_sha256=digest(a.bank), python=platform.python_version(),
                  machine=platform.machine(), cpu_threads=a.cpu_threads, budget_seconds=a.seconds,
                  timing='Wall time; warm-up and GPU upload separately recorded. Timed phase includes synchronization and output transfer. No repeated candidates.',
                  statuses={}, measurements={})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=a.cpu_threads)

    def cpu(b):
        if a.cpu_threads == 1:
            return cpu_reference(b)
        boundaries = np.linspace(0, len(b['states']), min(a.cpu_threads, len(b['states']))+1, dtype=int)
        parts = [subset(b, int(lo), int(hi)) for lo, hi in zip(boundaries[:-1], boundaries[1:])]
        evaluated = list(pool.map(cpu_reference, parts))
        return tuple(np.concatenate([x[k] for x in evaluated]) for k in range(3))

    try:
        stamp = time.perf_counter()
        cpu(subset(bank, 0, min(len(bank['states']), a.batch_cases)))
        result['cpu_warmup_seconds'] = time.perf_counter()-stamp
        resident = None
        if a.backend in ('gpu', 'both'):
            if cuda.config.ENABLE_CUDASIM:
                raise RuntimeError('Simulator timings are forbidden')
            stamp = time.perf_counter()
            # Check entire input bank once before timed GPU evaluation.
            resident = ResidentBank(bank)
            result['gpu_upload_seconds'] = time.perf_counter()-stamp
            stamp = time.perf_counter()
            actual = resident.evaluate()
            result['gpu_first_evaluation_seconds'] = time.perf_counter()-stamp
            result['parity'] = compare(cpu(bank), actual, float(bank['sigma']))
            device = cuda.get_current_device()
            name = device.name
            result['gpu_name'] = name.decode() if isinstance(name, bytes) else str(name)
            result['gpu_compute_capability'] = list(device.compute_capability)
        backends = ('cpu', 'gpu') if a.backend == 'both' else (a.backend,)
        for backend in backends:
            started = time.perf_counter()
            outcomes = []
            for lo in range(0, len(bank['states']), a.batch_cases):
                if time.perf_counter()-started >= a.seconds:
                    break
                hi = min(lo+a.batch_cases, len(bank['states']))
                b = subset(bank, lo, hi)
                if backend == 'cpu':
                    output = cpu(b)
                else:
                    # Stage benchmark uses chunk uploads. Resident throughput is
                    # not substituted for end-to-end bounded-pass throughput.
                    output = ResidentBank(b).evaluate()
                for j in range(hi-lo):
                    ix = b['case_index'] == j
                    valid = bool(output[2][ix].all())
                    outcomes.append(dict(case=lo+j, status='evaluated' if valid else 'unresolved',
                        tangent_rms_arcsec=float(np.sqrt(np.mean(np.sum(output[0][ix]**2, axis=1)))*float(bank['sigma'])) if valid else None))
            seconds = time.perf_counter()-started
            result['measurements'][backend] = dict(wall_seconds=seconds, deadline_overshoot_seconds=max(0., seconds-a.seconds),
                completed_unique_cases=len(outcomes), total_cases=len(bank['states']),
                valid_cases=sum(x['status']=='evaluated' for x in outcomes),
                unresolved=sum(x['status']=='unresolved' for x in outcomes),
                uncomputed=len(bank['states'])-len(outcomes))
            a.output.with_name(a.output.stem+'-'+backend+'-cases.json').write_text(json.dumps(outcomes))
            result['statuses'][backend] = 'completed' if len(outcomes)==len(bank['states']) else 'time_bounded_partial'
    except Exception as exc:
        result['blocked'] = type(exc).__name__+': '+str(exc)
    finally:
        pool.shutdown()
        a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if 'blocked' in result:
        raise SystemExit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('report', 'cache', 'observatories', 'ephemeris', 'output'):
        prep.add_argument('--'+name, type=Path, required=True)
    prep.add_argument('--limit', type=int, default=1000)
    bench = sub.add_parser('run')
    bench.add_argument('--bank', type=Path, required=True)
    bench.add_argument('--output', type=Path, required=True)
    bench.add_argument('--backend', choices=['cpu', 'gpu', 'both'], default='both')
    bench.add_argument('--cpu-threads', type=int, default=1)
    bench.add_argument('--batch-cases', type=int, default=128)
    bench.add_argument('--seconds', type=float, default=30)
    a = p.parse_args()
    if a.command == 'prepare':
        if a.limit < 1:
            p.error('--limit must be positive')
        prepare(a)
    else:
        if min(a.cpu_threads, a.batch_cases, a.seconds) <= 0:
            p.error('threads, batch size and time must be positive')
        run(a)


if __name__ == '__main__':
    main()
