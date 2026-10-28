"""Combine seed and recovered modes without discarding alternatives on shared rows."""
import argparse
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import zlib
import numpy as np
from magictools.itf.catalogue import sha256
from magictools.itf.nbody import Ephemeris
from magictools.itf.report_io import metadata, candidates


def mode_key(mode):
    return (mode['epoch_mjd_tdb'],tuple(mode['state_au_au_per_day']))


def combine(base,sources):
    report=copy.deepcopy(base)
    lookup={tuple(row['tracklet_indices']):row for row in report['candidates']}
    added_modes=0
    for source,digest in sources:
        if source['input']!=base['input']:raise ValueError('Cannot mix ITF catalogue snapshots')
        for incoming in source['candidates']:
            key=tuple(incoming['tracklet_indices'])
            if key in lookup:
                target=lookup[key];known={mode_key(m) for m in target['modes']}
                for mode in incoming['modes']:
                    if mode_key(mode) not in known:
                        target['modes'].append(copy.deepcopy(mode));known.add(mode_key(mode));added_modes+=1
                target['maximal']=target['maximal'] or incoming['maximal']
                # Keep original row statistics/provenance; each mode has its own
                # fit and uncertainty. Do not silently replace full modes by scout.
                continue
            row=copy.deepcopy(incoming)
            row['origin_rank']=row['rank'];row['origin_report_sha256']=digest
            row['recovered_input']=len(row['tracklet_indices'])>=3
            row['rank']=len(report['candidates'])+1
            report['candidates'].append(row);lookup[key]=row
    report['additional_modes_on_shared_chains']=added_modes
    return report


def stream_combine(paths, output, ephemeris=None):
    """Merge on disk with memory proportional to one chain, not all reports.

    The scratch database is disposable. Only an atomically replaced final JSON
    is a committed product. Raw source reports and all distinct modes survive.
    """
    paths = list(map(Path, paths)); output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = metadata(paths[0]); provenance = []; added_modes = 0
    count = mode_count = 0
    ep = None
    with tempfile.TemporaryDirectory(prefix='recovery-merge-', dir=output.parent) as scratch:
        db = sqlite3.connect(Path(scratch)/'rows.sqlite')
        db.execute('PRAGMA cache_size=-16384')
        db.execute('CREATE TABLE rows (id INTEGER PRIMARY KEY, chain TEXT UNIQUE, body BLOB)')
        try:
            for source_index, path in enumerate(paths):
                if metadata(path)['input'] != header['input']:
                    raise ValueError('Cannot mix ITF catalogue snapshots')
                digest = sha256(path); incoming_count = 0
                for incoming in candidates(path):
                    incoming_count += 1
                    key = ','.join(map(str, incoming['tracklet_indices']))
                    previous = db.execute('SELECT id,body FROM rows WHERE chain=?', (key,)).fetchone()
                    if previous:
                        index, body = previous; row = json.loads(zlib.decompress(body))
                        known = {mode_key(mode) for mode in row['modes']}
                        for mode in incoming['modes']:
                            if mode_key(mode) not in known:
                                row['modes'].append(mode); known.add(mode_key(mode)); added_modes += 1
                        row['maximal'] = row['maximal'] or incoming['maximal']
                    else:
                        count += 1; index = count; row = incoming
                        if source_index:
                            row['origin_rank'] = row['rank']; row['origin_report_sha256'] = digest
                            row['recovered_input'] = len(row['tracklet_indices']) >= 3
                            row['rank'] = index
                    body = zlib.compress(json.dumps(row, allow_nan=False, separators=(',', ':')).encode(), 1)
                    db.execute('INSERT OR REPLACE INTO rows VALUES (?,?,?)', (index, key, body))
                    if incoming_count % 1000 == 0:
                        db.commit()
                db.commit()
                provenance.append(dict(path=str(path), sha256=digest, candidates=incoming_count))
            header.update(recovery_bank_sources=provenance, status='prepared_recovery_bank',
                          additional_modes_on_shared_chains=added_modes)
            header['limitations'] = header.get('limitations', []) + [
                'Maximal flags retained across source reports to preserve original alternatives.',
                'Shared chains retain distinct fitted states; only identical epoch/state modes are deduplicated.',
                'Mode weights retain source-fit values and are not a normalized merged posterior.']
            temporary = output.with_suffix(output.suffix+'.tmp')
            with temporary.open('w') as stream:
                stream.write(json.dumps(header, allow_nan=False, separators=(',', ':'))[:-1]+',"candidates":[')
                for ordinal, (body,) in enumerate(db.execute('SELECT body FROM rows ORDER BY id')):
                    row = json.loads(zlib.decompress(body))
                    missing = [m for m in row['modes'] if 'range_au' not in m]
                    if missing:
                        if ephemeris is None:
                            raise ValueError('Missing ranges require an ephemeris')
                        times = np.array([m['epoch_mjd_tdb'] for m in missing])
                        if ep is None:
                            ep = Ephemeris(ephemeris, times.min()-1, times.max()+1)
                        # earth() reads exact SPK positions; it does not use the
                        # force spline's narrow initialization interval.
                        for mode, earth in zip(missing, ep.earth(times)):
                            mode['range_au'] = float(np.linalg.norm(np.array(mode['state_au_au_per_day'])[:3]-earth))
                    if ordinal:
                        stream.write(',')
                    stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')))
                    mode_count += len(row['modes'])
                stream.write(']}\n')
            temporary.replace(output)
        finally:
            db.close()
            if ep is not None:
                ep.kernel.close()
    return dict(candidates=count, modes=mode_count, additional_modes_on_shared_chains=added_modes,
                source_reports=provenance)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--include',type=Path,action='append',default=[])
    p.add_argument('--ephemeris',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(stream_combine([a.base]+a.include, a.output, a.ephemeris), indent=2))

if __name__=='__main__':main()
