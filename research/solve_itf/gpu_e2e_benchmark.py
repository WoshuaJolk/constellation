"""Reproduce historical fast-profile whole-catalogue workload; no production writes."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def child(backend, output, cache):
    # Inject only the existing research adapter. Search policy and 128-chain
    # orchestration are unchanged. No sustained-harness prefetch is attributed here.
    from magictools.itf import search, cli
    original = search.solve
    adapter = None
    if backend == 'gpu':
        from gpu_compact_adapter import CompactGPUBatch
        adapter = CompactGPUBatch()
    audit = []
    def solve(*args, **kwargs):
        result = original(*args, **kwargs, batch_fitter=adapter, audit=audit.append)
        return result
    search.solve = solve
    code = cli.main(['run', '--input', 'input/itf.txt.gz', '--cache', str(cache),
                     '--observatories', 'input/ObsCodes.html', '--output', str(output),
                     '--workers', '4', '--seconds', '280', '--max-seeds', '10000', '--top', '10'])
    seeds = next((x['seeds'] for x in audit if x['event'] == 'seed_bank'), [])
    write(output/'benchmark-extra.json', dict(backend=backend, seed_count=len(seeds),
        seed_sha256=hashlib.sha256(json.dumps(seeds,separators=(',',':')).encode()).hexdigest(),
        batch_size=128 if adapter else None, batch_receipts=adapter.receipts if adapter else []))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', choices=['cpu','gpu'])
    parser.add_argument('--output', type=Path, default=Path('e2e-results'))
    parser.add_argument('--cache', type=Path)
    args = parser.parse_args()
    if args.child:
        return child(args.child,args.output,args.cache)
    args.output.mkdir(exist_ok=False)
    report = dict(status='running', scope='Full historical fast-profile workload, 10000 seeds; current code on one host.',
        limitations=['Historical CPU hardware/software differ.',
                     'GPU uses current 128-chain compact adapter; sustained 1024-chain prefetch harness is not wired into search.',
                     'Cold means absent application/JIT/CUDA disk caches, not cleared OS page cache.',
                     'Transfer and package installation excluded from application wall time, included in cost.'], phases=[], outputs={})
    def save(): write(args.output/'report.json',report)
    def record(path):
        report['outputs'][str(path.relative_to(args.output))]=dict(path=str(path.relative_to(args.output)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    save()
    runs=[('cpu-cold','cpu','cpu-cache','cpu-jit','cpu-cuda'),('gpu-cold','gpu','gpu-cache','gpu-jit','gpu-cuda'),('gpu-warm','gpu','gpu-cache','gpu-jit','gpu-cuda')]
    for label,backend,cache,jit,cuda in runs:
        report['phase']=label;save()
        out=args.output/label
        env=dict(os.environ,NUMBA_CACHE_DIR=str((args.output/jit).resolve()),CUDA_CACHE_PATH=str((args.output/cuda).resolve()),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        command=[sys.executable,__file__,'--child',backend,'--output',str(out),'--cache',str(args.output/cache)]
        before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.perf_counter()
        with (args.output/(label+'.log')).open('x') as stream:
            try:
                done=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=400)
                code=done.returncode
            except subprocess.TimeoutExpired:
                code='subprocess_timeout'
        elapsed=time.perf_counter()-start;after=resource.getrusage(resource.RUSAGE_CHILDREN)
        row=dict(name=label,backend=backend,elapsed_seconds=elapsed,exit_code=code,
                 user_cpu_seconds=after.ru_utime-before.ru_utime,system_cpu_seconds=after.ru_stime-before.ru_stime,
                 initial_data_cache='present' if label=='gpu-warm' else 'absent',initial_jit_cache='present' if label=='gpu-warm' else 'absent')
        if (out/'report.json').exists():
            result=json.loads((out/'report.json').read_text()); row.update(status=result['status'],timings=result['timings'],counts=result['counts'],candidates=len(result['candidates']))
            compressed=args.output/(label+'-report.json.gz')
            with gzip.open(compressed,'wb') as stream: stream.write((out/'report.json').read_bytes())
            record(compressed)
        for name in ['benchmark-extra.json','candidates.txt','maximal_candidates.txt']:
            p=out/name
            if p.exists():record(p)
        record(args.output/(label+'.log'))
        report['phases'].append(row);save()
        if code != 0:
            report.update(status='failed',failure_phase=label);save();return 1
    report.update(status='complete',phase='complete');save()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
