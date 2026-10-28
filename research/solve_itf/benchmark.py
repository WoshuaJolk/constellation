"""Time a full-ITF run on a POSIX host (fresh data/JIT by default).

Example from the repository root:
  .venv/bin/python research/solve_itf/benchmark.py \
    --input .cache/solve_itf/itf.txt.gz \
    --observatories .cache/solve_itf/ObsCodes.html \
    --cache .cache/solve_itf/new-benchmark --output runs/new-benchmark

Download is separate. --reuse-cache explicitly permits existing data/JIT caches.
Nothing is deleted or overwritten.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'observatories', 'cache', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--profile', choices=('fast', 'broad'), default='fast')
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-seeds', type=int)
    parser.add_argument('--reuse-cache', action='store_true')
    args = parser.parse_args()
    jit_cache = args.cache.with_name(args.cache.name+'-jit')
    console = args.output.with_name(args.output.name+'-console.txt')
    checked = (args.output, console) if args.reuse_cache else (args.cache, jit_cache, args.output, console)
    for path in checked:
        if path.exists():
            parser.error(f'Fresh benchmark requires an absent path: {path}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'magictools.itf.cli', 'run',
               '--input', str(args.input), '--cache', str(args.cache),
               '--observatories', str(args.observatories), '--output', str(args.output),
               '--profile', args.profile, '--workers', str(args.workers), '--top', '10']
    for name in ('seconds', 'max_seeds'):
        if getattr(args, name) is not None:
            command.extend(['--'+name.replace('_', '-'), str(getattr(args, name))])
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', NUMBA_CACHE_DIR=str(jit_cache.resolve()))
    initial_data = 'present' if args.cache.exists() else 'absent'
    initial_jit = 'present' if jit_cache.exists() else 'absent'
    started = time.monotonic()
    with console.open('x') as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
    elapsed = time.monotonic()-started
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    measurements = dict(command=command, elapsed_seconds=elapsed,
                        peak_rss_kib=usage.ru_maxrss/(1024 if sys.platform == 'darwin' else 1),
                        rss_scope='per-process maximum, not simultaneous sum across workers',
                        user_cpu_seconds=usage.ru_utime, system_cpu_seconds=usage.ru_stime,
                        exit_code=completed.returncode, initial_data_cache=initial_data,
                        initial_numba_cache=initial_jit, numba_cache=str(jit_cache),
                        openblas_num_threads=1)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'resources.json').write_text(json.dumps(measurements, indent=2)+'\n')
    print(json.dumps(measurements, indent=2))
    return completed.returncode


if __name__ == '__main__':
    raise SystemExit(main())
