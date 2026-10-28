"""Transactional batched seed fitting: fixed memory, no seven-million-row queue."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing
from pathlib import Path
import sqlite3
import time
import zlib
import numpy as np
from .catalogue import sha256
from .campaign_graph import save_json
from .campaign_geometry import AtlasGeometry,prepare_atlas
from .fit import fit_chain,BudgetExpired

_GEOMETRY=None
_SEEDS=None
_SETTINGS=None


def initialize(cache,observatories,directory,settings):
    global _GEOMETRY,_SEEDS,_SETTINGS
    if settings['start_policy']=='cascade':
        from .campaign_geometry_fast import FastAtlasGeometry
        _GEOMETRY=FastAtlasGeometry(cache,observatories,Path(directory)/'fit-geometry',Path(directory)/'observation-identities')
    else:
        _GEOMETRY=AtlasGeometry(cache,observatories,Path(directory)/'fit-geometry')
    _SEEDS=np.load(Path(directory)/'seeds.npy',mmap_mode='r');_SETTINGS=settings


def fit_batch(job):
    batch_id,start,end,deadline=job
    status=np.zeros(end-start,np.uint8)
    rows=[];failures=[];diagnostics=[];started=time.monotonic()
    for offset,seed in enumerate(_SEEDS[start:end]):
        if time.monotonic()>=deadline:break
        try:
            seed_deadline=min(deadline,time.monotonic()+_SETTINGS['seed_seconds'])
            if _SETTINGS['start_policy']=='cascade':
                from .cascade import fit_cascade
                result, diagnostic=fit_cascade(_GEOMETRY,seed,deadline=seed_deadline,**_SETTINGS['cascade'])
                diagnostics.append(dict(seed=start+offset,**diagnostic))
            else:
                result=fit_chain(_GEOMETRY,seed,deadline=seed_deadline,
                    broad_starts=True,sigma=.3,rms_limit=.6,start_policy=_SETTINGS['start_policy'])
            status[offset]=2 if result is not None else 1
            if result is not None:
                rows.append((start+offset,zlib.compress(json.dumps(result,allow_nan=False,separators=(',',':')).encode(),3)))
        except BudgetExpired:
            status[offset]=4
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            status[offset]=3;failures.append((start+offset,type(exc).__name__,str(exc)[:300]))
    result=(batch_id,start,end,status.tobytes(),rows,failures,time.monotonic()-started)
    return result+(zlib.compress(json.dumps(diagnostics,allow_nan=False,separators=(',',':')).encode(),3),) if _SETTINGS['start_policy']=='cascade' else result


def open_checkpoint(path,identity):
    db=sqlite3.connect(path);db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.executescript('CREATE TABLE IF NOT EXISTS metadata (body TEXT);'
        'CREATE TABLE IF NOT EXISTS batches (id INTEGER PRIMARY KEY,start INTEGER,end INTEGER,status BLOB,seconds REAL);'
        'CREATE TABLE IF NOT EXISTS candidates (seed INTEGER PRIMARY KEY,body BLOB);'
        'CREATE TABLE IF NOT EXISTS diagnostics (batch INTEGER PRIMARY KEY,body BLOB);'
        'CREATE TABLE IF NOT EXISTS failures (seed INTEGER,kind TEXT,message TEXT);')
    encoded=json.dumps(identity,sort_keys=True)
    saved=db.execute('SELECT body FROM metadata').fetchone()
    if saved and saved[0]!=encoded:
        db.close()
        raise ValueError('Fit implementation/settings/input changed; new checkpoint required')
    if not saved:
        with db:db.execute('INSERT INTO metadata VALUES (?)',(encoded,))
    return db


def commit_batch(db,result):
    batch,start,end,status,rows,failures,seconds=result[:7]
    previous=db.execute('SELECT status FROM batches WHERE id=?',(batch,)).fetchone()
    if previous:
        old=np.frombuffer(previous[0],np.uint8)
        new=np.frombuffer(status,np.uint8).copy()
        preserve=(old==1)|(old==2)|(new==0)
        new[preserve]=old[preserve];status=new.tobytes()
    with db:
        db.execute('INSERT OR REPLACE INTO batches VALUES (?,?,?,?,?)',(batch,start,end,status,seconds))
        db.executemany('INSERT OR REPLACE INTO candidates VALUES (?,?)',rows)
        db.executemany('INSERT INTO failures VALUES (?,?,?)',failures)
        if len(result)>7:
            # Replayed batches replace diagnostics only for seeds actually retried.
            previous=db.execute('SELECT body FROM diagnostics WHERE batch=?',(batch,)).fetchone()
            records={r['seed']:r for r in json.loads(zlib.decompress(previous[0]))} if previous else {}
            records.update({r['seed']:r for r in json.loads(zlib.decompress(result[7]))})
            body=zlib.compress(json.dumps(list(records.values()),allow_nan=False,separators=(',',':')).encode(),3)
            db.execute('INSERT OR REPLACE INTO diagnostics VALUES (?,?)',(batch,body))


def checkpoint_counts(db,total):
    counts=np.zeros(5,np.int64)
    for (status,) in db.execute('SELECT status FROM batches'):
        counts+=np.bincount(np.frombuffer(status,np.uint8),minlength=5)
    return dict(seed_target=total,seeds_attempted=int(counts[1:].sum()),
        seeds_completed=int(counts[1:3].sum()),seeds_without_accepted_fit=int(counts[1]),
        accepted_seed_fits=int(counts[2]),seeds_retryable=int(counts[3:].sum()),
        seeds_unattempted=int(total-counts[1:].sum()))


def policy_counts(db,total,policy):
    counts=checkpoint_counts(db,total)
    if policy=='cascade':
        counts.update(seeds_deferred_full_grid=counts['seeds_without_accepted_fit'],
                      seeds_pending_deeper_initializations=total)
    return counts


def run(args):
    directory=args.output
    if not (directory/'seeds.json').exists():raise ValueError('Commit seed bank first')
    prepare_atlas(args.cache,args.observatories,directory/'fit-geometry',lambda s:print(s,flush=True))
    seeds=np.load(directory/'seeds.npy',mmap_mode='r')
    settings=dict(seed_seconds=args.seed_seconds,start_policy=args.start_policy)
    if args.start_policy=='cascade':
        from .cascade import CAMPAIGN_SETTINGS
        from .campaign_geometry_fast import prepare_observation_ids
        prepare_observation_ids(args.cache,directory/'observation-identities')
        settings['cascade']=CAMPAIGN_SETTINGS
    identity=dict(seed_sha256=sha256(directory/'seeds.npy'),batch_size=args.batch_size,settings=settings,
        geometry=json.loads((directory/'fit-geometry'/'identity.json').read_text()),
        implementation={n:sha256(Path(__file__).with_name(n)) for n in
                        ('fit.py','derivatives.py','campaign_fit.py','campaign_geometry.py','cascade.py','harmonic.py','campaign_geometry_fast.py')})
    db=open_checkpoint(directory/('fitting-'+args.start_policy+'.sqlite'),identity)
    # Only terminal batches are skipped. Unfinished batches are replayable; unique
    # seed primary keys make retries idempotent, without losing accepted results.
    finished=set()
    for batch,status in db.execute('SELECT id,status FROM batches'):
        state=np.frombuffer(status,np.uint8)
        if np.all((state==1)|(state==2)):finished.add(batch)
    started=time.monotonic();deadline=started+args.seconds
    jobs=((k,start,min(start+args.batch_size,len(seeds)),deadline)
          for k,start in enumerate(range(0,len(seeds),args.batch_size)) if k not in finished)
    pending={};exhausted=False;last_print=0
    try:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize,initargs=(str(args.cache),str(args.observatories),str(directory),settings)) as pool:
            while pending or not exhausted:
                while len(pending)<2*args.workers and not exhausted and time.monotonic()<deadline:
                    job=next(jobs,None)
                    if job is None:exhausted=True;break
                    pending[pool.submit(fit_batch,job)]=job
                if time.monotonic()>=deadline:exhausted=True
                if not pending:break
                done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future);commit_batch(db,future.result())
                if time.monotonic()-last_print>15:
                    counts=policy_counts(db,len(seeds),args.start_policy)
                    save_json(directory/'campaign-progress.json',dict(phase='fitting',start_policy=args.start_policy,full_grid_refinement_pending=args.start_policy!='full',status='running',counts=counts,
                        updated_utc=time.time(),invocation_seconds=time.monotonic()-started))
                    print(json.dumps(counts),flush=True);last_print=time.monotonic()
    finally:
        counts=policy_counts(db,len(seeds),args.start_policy)
        save_json(directory/'campaign-progress.json',dict(phase='fitting',start_policy=args.start_policy,full_grid_refinement_pending=args.start_policy!='full',status='work_remaining' if counts['seeds_unattempted'] or counts['seeds_retryable'] else 'seed_bank_complete',
            counts=counts,updated_utc=time.time(),invocation_seconds=time.monotonic()-started))
        db.close()
    print(json.dumps(counts),flush=True)
