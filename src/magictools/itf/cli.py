"""The solve_itf command: download, compact, and search the complete ITF."""
import argparse
import json
import os
import platform
from pathlib import Path
import sys
import tempfile
import time
import urllib.request


def log(message):
    print(message, file=sys.stderr, flush=True)


def fetch(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    urls = {'itf.txt.gz': 'https://www.minorplanetcenter.net/iau/ITF/itf.txt.gz',
            'ObsCodes.html': 'https://www.minorplanetcenter.net/iau/lists/ObsCodes.html'}
    for name, url in urls.items():
        target = destination/name
        if target.exists():
            log(f'Keeping existing snapshot {target}')
            continue
        fd, temp = tempfile.mkstemp(dir=destination, prefix=name+'.')
        try:
            with os.fdopen(fd, 'wb') as output, urllib.request.urlopen(url, timeout=60) as response:
                while block := response.read(1024*1024):
                    output.write(block)
            os.replace(temp, target)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        log(f'Downloaded {target} ({target.stat().st_size:,} bytes)')


def format_rows(report, top=None, *, maximal_only=False):
    lines = ['Ranked candidate fits — scores do not establish association probabilities.',
             f'Status: {report["status"]}; full catalogue scan: '
             f'{report["counts"]["full_catalogue_scan_complete"]}',
             'View: maximal chains; contained subsets remain in report.json and candidates.txt.'
             if maximal_only else 'View: all accepted chains, including contained subsets.',
             'Shared tracklets indicate alternatives, not independent discoveries. '
             'Inclination ± is a local formal 1σ error; [weak orbit] flags poor local constraints.', '']
    rows = [row for row in report['candidates'] if not maximal_only or row['maximal']]
    if top is not None:
        rows = rows[:top]
    for row in rows:
        uncertainty = row['modes'][0]['local_uncertainty']
        inclination = f'{row["inclination_deg"]:.1f}'
        if uncertainty['inclination_sigma_deg'] is not None:
            inclination += f' ± {uncertainty["inclination_sigma_deg"]:.1f}'
        weak = ' [weak orbit]' if uncertainty['local_linearization_weak'] else ''
        lines.append(f'{row["rank"]}. '+ ' + '.join(row['tracklets'])+
                     f' ({", ".join(row["stations"])}), {row["span_days"]:.1f} d, '
                     f'{row["observations"]} obs, {row["rms_arcsec"]:.2f}" rms, '
                     f'i {inclination} deg — BIC score {row["log_odds_bic"]:.4g}{weak}')
    if not rows:
        lines.append('No accepted candidate chains within this search budget.')
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description='Whole-ITF bounded orbit-linkage search; see docs/solve-itf.md.')
    commands = parser.add_subparsers(dest='command', required=True)
    download = commands.add_parser('fetch', help='Download a complete current MPC snapshot and observatory table')
    download.add_argument('--directory', type=Path, default=Path('.cache/solve_itf'))
    prepare = commands.add_parser('build', help='Compact every supported observation in an MPC80 file')
    prepare.add_argument('--input', type=Path, required=True)
    prepare.add_argument('--cache', type=Path, required=True)
    unpack = commands.add_parser('restore', help='Expand a portable compact catalogue for searching')
    unpack.add_argument('--archive', type=Path, required=True)
    unpack.add_argument('--manifest', type=Path, required=True)
    unpack.add_argument('--cache', type=Path, required=True)
    run = commands.add_parser('run', help='Scan the full cache and fit/grow ranked candidate chains')
    run.add_argument('--profile', choices=('fast', 'broad'), default='fast',
                     help='broad: curved distance-conditioned proposals, 70k seeds, 28-day links, one-hour budget')
    run.add_argument('--input', type=Path, help='Build/verify cache first; ingestion counts toward --seconds')
    run.add_argument('--cache', type=Path, required=True)
    run.add_argument('--observatories', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True, help='New directory for report.json and candidates.txt')
    run.add_argument('--workers', type=int, default=4)
    run.add_argument('--seconds', type=float)
    run.add_argument('--max-gap-days', type=int)
    run.add_argument('--position-gate-arcsec', type=float, default=30.)
    run.add_argument('--curvature-gate', type=float, default=1.)
    run.add_argument('--neighbors-per-day', type=int)
    run.add_argument('--max-edges', type=int)
    run.add_argument('--max-seeds', type=int)
    run.add_argument('--seed-neighbors', type=int, help='Neighbors combined into triple seeds; 0 uses all')
    run.add_argument('--branches', type=int, default=4)
    run.add_argument('--max-parent-seeds', type=int, default=16)
    run.add_argument('--single-addition-growth', action='store_true',
                     help='Fit individual additions even when a joint extension fits; can enumerate many subsets')
    run.add_argument('--sigma-arcsec', type=float, default=.3)
    run.add_argument('--rms-limit-arcsec', type=float, default=.6)
    run.add_argument('--prior-link-probability', type=float, default=.001)
    run.add_argument('--top', type=int, default=100, help='Console rows; files retain all accepted candidates')
    run.add_argument('--include-subsets', action='store_true', help='Include contained chains in the console view')
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        if args.command == 'fetch':
            fetch(args.directory)
            return 0
        from .catalogue import Catalogue, build, restore
        if args.command == 'restore':
            restore(args.archive, args.manifest, args.cache)
            return 0
        if args.command == 'build':
            build(args.input, args.cache, progress=log)
            return 0
        from .fit import read_sites
        from .search import SearchConfig, solve
        defaults = SearchConfig.for_profile(args.profile)
        for name in ('seconds', 'max_gap_days', 'neighbors_per_day', 'max_edges', 'max_seeds', 'seed_neighbors'):
            if getattr(args, name) is None:
                setattr(args, name, getattr(defaults, name))
        config = SearchConfig(workers=args.workers, max_gap_days=args.max_gap_days, position_gate_arcsec=args.position_gate_arcsec,
                              curvature_gate_arcsec_per_day2=args.curvature_gate,
                              neighbors_per_day=args.neighbors_per_day, max_edges=args.max_edges,
                              max_seeds=args.max_seeds, branches=args.branches,
                              seed_neighbors=args.seed_neighbors, broad_search=defaults.broad_search,
                              max_parent_seeds=args.max_parent_seeds, joint_extensions=not args.single_addition_growth,
                              sigma_arcsec=args.sigma_arcsec, rms_limit_arcsec=args.rms_limit_arcsec,
                              prior_link_probability=args.prior_link_probability, seconds=args.seconds)
        config.validate()
        if args.top < 1:
            raise ValueError('--top must be positive')
        if args.output.exists():
            raise ValueError('Output already exists; choose a new --output directory')
        sites = read_sites(args.observatories)
        from .warmup import KernelWarmup
        with KernelWarmup(enabled=bool(args.input and not args.cache.exists() and config.workers > 1)) as warmup:
            catalogue = build(args.input, args.cache, progress=log) if args.input else Catalogue(args.cache)
            compiler = warmup.finish(started+args.seconds)
        if compiler['enabled'] and not compiler['completed']:
            log('Background compilation did not complete; fitting will use normal JIT if time remains')
        preparation = time.monotonic()-started
        report = solve(catalogue, sites, config, progress=log, deadline=started+args.seconds)
        report['compiler_warmup'] = compiler
        from .catalogue import sha256
        report['observatories_sha256'] = sha256(args.observatories)
        from importlib.metadata import version
        report['implementation'] = dict(
            python=platform.python_version(), platform=platform.platform(),
            packages={name: version(name) for name in ('numpy', 'scipy', 'astropy', 'numba')},
            source_sha256={path.name: sha256(path) for path in sorted(Path(__file__).parent.glob('*.py'))})
        report['timings']['preparation_seconds'] = preparation
        report['timings']['total_seconds_before_output'] = time.monotonic()-started
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        (args.output/'candidates.txt').write_text(format_rows(report))
        (args.output/'maximal_candidates.txt').write_text(format_rows(report, maximal_only=True))
        print(format_rows(report, args.top, maximal_only=not args.include_subsets), end='')
        log(f'Results: {args.output.resolve()}; elapsed {time.monotonic()-started:.1f}s')
        # Budget exhaustion is a valid partial result, explicitly identified in report.
        return 0
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f'solve_itf: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
