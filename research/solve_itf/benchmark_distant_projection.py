"""Paired, input-frozen projection microbenchmark; does not search or fit chains."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from numba import cuda
from distant_epoch_graph import projected
from gpu_distant_projection import ProjectionBank
from magictools.itf.kernels import ARCSEC, MU


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True,
        help='NPZ: t, u, du, observers, velocities, ref, earth, hypotheses (distance, fraction)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if cuda.config.ENABLE_CUDASIM:
        raise RuntimeError('Hardware timing requires a real GPU, not CUDA simulation')
    if args.output.exists():
        raise FileExistsError('Preserve existing benchmark receipts')
    data = np.load(args.input, allow_pickle=False)
    arrays = [data[k] for k in ('t','u','du','observers','velocities')]
    ref, earth, hypotheses = float(data['ref']), data['earth'], data['hypotheses']
    if not len(arrays[0]) or not len(hypotheses):
        raise ValueError('Benchmark requires nonempty inputs')
    design = dict(input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        repeats=3, absolute_parity_arcsec=1e-5,
        scope='Projection only; no query, graph, fitting, or recovery speed claim',
        policy='Alternating backend order; cold/compilation separate; all repeats reported')
    args.output.mkdir(parents=True)
    (args.output/'design.json').write_text(json.dumps(design, indent=2))
    bank = ProjectionBank(*arrays)

    def cpu():
        result = np.asarray([projected(*arrays, ref, earth, d, f*np.sqrt(2*MU/d))
                             for d,f in hypotheses])
        result[:, np.abs(ref-arrays[0])>365] = np.nan
        return result

    started = time.perf_counter()
    cpu(); bank.project(ref, earth, hypotheses)
    warmup = time.perf_counter()-started
    report = dict(design=design, warmup_seconds=warmup,
        resident_upload_seconds=bank.upload_seconds, input_bytes=bank.input_bytes, repeats=[])
    for rep in range(3):
        results, elapsed = {}, {}
        for backend in (('cpu','gpu') if rep%2==0 else ('gpu','cpu')):
            started = time.perf_counter()
            results[backend] = cpu() if backend=='cpu' else bank.project(ref,earth,hypotheses)[0]
            elapsed[backend] = time.perf_counter()-started
        good = np.isfinite(results['cpu']).all(axis=2)
        masks = np.array_equal(np.isfinite(results['cpu']), np.isfinite(results['gpu']))
        errors = np.linalg.norm(np.cross(results['cpu'][good],results['gpu'][good]),axis=1)*ARCSEC
        maximum = float(np.max(errors, initial=0))
        vectors = np.allclose(results['cpu'],results['gpu'],rtol=0,atol=1e-10,equal_nan=True)
        passed = bool(masks and vectors and maximum < design['absolute_parity_arcsec'])
        report['repeats'].append(dict(seconds=elapsed, finite_mask_equal=masks,
            vectors_equal_within_tolerance=bool(vectors), maximum_arcsec=maximum,
            parity_passed=passed, valid=int(good.sum())))
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        if not passed:
            raise RuntimeError('Projection parity failed; evidence preserved, benchmark stopped')


if __name__ == '__main__':
    main()
