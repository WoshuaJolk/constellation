"""Package finished runs with portable paths and compressed complete fit ledgers."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def portable(value):
    if 'provenance' in value:
        for row in value['provenance']['inputs']:
            path = Path(row['path'])
            row['path'] = ('../../../src/magictools/'+path.name) if path.suffix == '.py' else path.name
    if 'archive' in value: value['archive'] = Path(value['archive']).name
    if 'enclosures' in value:
        path = Path(value['enclosures']); value['enclosures'] = path.parent.name+'/'+path.name
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for case in ['observed', 'injection', 'historical']: parser.add_argument('--'+case, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Do not publish partially completed or overwritten artifacts.
    for case in ['observed', 'injection', 'historical']:
        root = getattr(args, case)
        for name in [case, case+'-bounds']:
            if not (root/name/'complete.json').is_file(): raise ValueError('Incomplete run: '+str(root/name))
    args.output.mkdir(parents=True, exist_ok=False)
    for case in ['observed', 'injection', 'historical']:
        root = getattr(args, case)
        shutil.copy2(root/(case+'.json'), args.output/(case+'.json'))
        for name in [case, case+'-bounds']:
            target = args.output/name; target.mkdir()
            source = root/name
            dump(target/'complete.json', portable(json.loads((source/'complete.json').read_text())))
            for path in source.glob('*.npz'): shutil.copy2(path, target/path.name)
            if (source/'manifest.json').exists(): shutil.copy2(source/'manifest.json', target/'manifest.json')
            if (source/'observed_fits.json').exists():
                (target/'observed_fits.json.gz').write_bytes(gzip.compress((source/'observed_fits.json').read_bytes(), mtime=0))
            controls = [json.loads(p.read_text()) for p in sorted(source.glob('control_*.json'))]
            if controls:
                (target/'control_fits.json.gz').write_bytes(gzip.compress(json.dumps(controls, allow_nan=False).encode(), mtime=0))
    for name in ['counterparts.json', 'selection_ledger.json']:
        shutil.copy2(args.observed/name, args.output/name)
    for name in ['injection_truth.json', 'recovery.json']:
        shutil.copy2(args.injection/name, args.output/name)
    inventory = {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(args.output.rglob('*')) if p.is_file()}
    dump(args.output/'inventory.json', inventory)


if __name__ == '__main__': main()
