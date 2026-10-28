"""Stream durable candidate records into a reproducible report without a RAM bank."""
import json
from pathlib import Path
import sqlite3
import zlib
import numpy as np
from .catalogue import Catalogue,sha256
from .campaign_graph import save_json
from .campaign_fit import policy_counts


def export(args):
    directory=args.output;c=Catalogue(args.cache)
    path=directory/('fitting-'+args.start_policy+'.sqlite')
    if not path.exists():raise ValueError('No fit checkpoint for this policy')
    db=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    db.execute('BEGIN')
    seed_info=json.loads((directory/'seeds.json').read_text())
    saved=json.loads(db.execute('SELECT body FROM metadata').fetchone()[0])
    if saved['geometry']['input'] != c.manifest or saved['seed_sha256'] != seed_info['seed_sha256']:
        db.close()
        raise ValueError('Export catalogue or seed bank differs from fit checkpoint')
    counts=policy_counts(db,seed_info['supplied'],args.start_policy)
    counts['full_catalogue_scan_complete']=True
    counts['triple_seeds_completed']=counts['seeds_completed']
    counts['seeds_retained']=seed_info['supplied']
    target=directory/('report-'+args.start_policy+'.json')
    temporary=target.with_suffix('.json.tmp')
    header=dict(schema_version=1,status='seed_bank_complete' if counts['seeds_unattempted']==0 and counts['seeds_retryable']==0 else 'work_remaining',
        input=c.manifest,counts=counts,config=dict(start_policy=args.start_policy),
        full_objective_complete=False,exhaustive=False,probabilities_calibrated=False,
        provenance=dict(seed_sha256=seed_info['seed_sha256'],fit_checkpoint=path.name),
        limitations=['Bounded proposal graph and initialization search; not an exhaustive posterior.',
          'Scout and cascade policies retain a separate full-grid refinement queue.',
          'Initial seed fits use solar two-body dynamics; extended recovery uses planetary dynamics.',
          'Rank here is deterministic seed order, not a calibrated probability ranking.'])
    count=0;mode_count=0;hypotheses=0
    with temporary.open('w') as stream:
        encoded=json.dumps(header,allow_nan=False)
        stream.write(encoded[:-1]+',"candidates":[')
        for seed,body in db.execute('SELECT seed,body FROM candidates ORDER BY seed'):
            row=json.loads(zlib.decompress(body));ids=row['tracklet_indices']
            count+=1;mode_count+=len(row['modes'])
            row.update(rank=count,seed_index=seed,maximal=True,
                tracklets=[c.tracks[i]['id'].decode().strip() for i in ids],
                stations=sorted({c.tracks[i]['site'].decode() for i in ids}),
                hypothesis_only=len(ids)<3)
            hypotheses+=int(len(ids)<3)
            if count>1:stream.write(',')
            stream.write(json.dumps(row,allow_nan=False,separators=(',',':')))
        stream.write(']}\n')
    temporary.replace(target);db.close()
    summary=dict(report=str(target),candidates=count,orbit_modes=mode_count,pair_hypotheses=hypotheses,
                 counts=counts,bytes=target.stat().st_size,sha256=sha256(target))
    save_json(directory/('export-'+args.start_policy+'.json'),summary)
    print(json.dumps(summary),flush=True)
