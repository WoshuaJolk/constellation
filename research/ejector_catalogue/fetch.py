#!/usr/bin/env python3
"""Fetch the whole declared Gaia DR3 selection, partitioned by source_id.

No TOP clause. A result is accepted only with TAP QUERY_STATUS=OK, no OVERFLOW,
unique IDs, and all 192 nonoverlapping partitions present. Cache permits retries.
This covers the query, not all Galactic fast stars or all astrometric surveys.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

import numpy as np
from astropy.table import vstack
from magictools.ejector_gaia import read_gaia

HERE=Path(__file__).resolve().parent
ENDPOINT='https://gea.esac.esa.int/tap-server/tap/sync'
NPARTS=192
ID_LIMIT=12*2**59


def fetch(output, workers):
    output.mkdir(parents=True,exist_ok=True)
    cache=output/'partitions';cache.mkdir(exist_ok=True)
    query=(HERE/'data/gaia_query.adql').read_text().strip()
    if ' TOP ' in query.upper(): raise ValueError('TOP truncation is forbidden')
    started=datetime.now(timezone.utc).isoformat()
    def partition(index):
        low,high=index*ID_LIMIT//NPARTS,(index+1)*ID_LIMIT//NPARTS
        adql=query+f' AND source_id >= {low} AND source_id < {high}'
        path=cache/f'{index:03}.vot'
        last_error=None
        for attempt in range(3):
            try:
                if not path.exists():
                    params=dict(REQUEST='doQuery',LANG='ADQL',FORMAT='votable',MAXREC=2000000,QUERY=adql)
                    data=urllib.request.urlopen(ENDPOINT+'?'+urllib.parse.urlencode(params),timeout=90).read()
                    partial=path.with_suffix('.partial.vot');partial.write_bytes(data)
                    read_gaia(partial)
                    partial.replace(path)
                table=read_gaia(path)
                ids=np.asarray(table['source_id'],dtype=np.int64)
                if np.any((ids < low)|(ids >= high)) or len(set(ids)) != len(ids):
                    raise ValueError('Unexpected/duplicate source IDs in partition')
                return dict(partition=index,low_inclusive=low,high_exclusive=high,rows=len(table),
                            query_status='OK',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            adql_sha256=hashlib.sha256(adql.encode()).hexdigest())
            except Exception as error:
                last_error=error
                if attempt < 2: time.sleep(2*(attempt+1))
        raise RuntimeError(f'Partition {index} failed: {last_error}')
    records=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(partition,i) for i in range(NPARTS)]
        for future in as_completed(pending):
            record=future.result();records.append(record)
            print('partition',record['partition'],'rows',record['rows'],flush=True)
    tables=[read_gaia(cache/f'{i:03}.vot') for i in range(NPARTS)]
    combined=vstack(tables,metadata_conflicts='silent')
    combined.sort('source_id')
    if len(set(combined['source_id'])) != len(combined): raise ValueError('Duplicate IDs across partitions')
    destination=output/'gaia_fast.fits.gz'
    if destination.exists(): raise ValueError('Merged catalogue already exists')
    combined.write(destination)
    report=dict(endpoint=ENDPOINT,started_utc=started,completed_utc=datetime.now(timezone.utc).isoformat(),
                query=query,partitions=sorted(records,key=lambda r:r['partition']),rows=len(combined),
                merged_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                declared_query_complete=True,all_fast_stars_complete=False)
    (output/'download_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('complete selection',len(combined),'rows',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=HERE/'data')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if not 1<=args.workers<=8: p.error('workers must be 1--8')
    fetch(args.output,args.workers)
