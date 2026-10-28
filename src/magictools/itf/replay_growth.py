"""Resume extension fits deferred by the broad recovery cascade, at full precision."""
import argparse,hashlib,json,os,sqlite3,time,warnings
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import multiprocessing
import numpy as np
from astropy.time import Time
from .catalogue import Catalogue,sha256
from .fit import BudgetExpired,read_sites
from .nbody import Ephemeris,fit_nbody
from .longarc import recovery_geometry,candidate_row,merge_candidate_rows,mark_maximal,atomic_json

_WORKER=None


def initialize(args,deadline):
    global _WORKER
    os.environ['ITF_NBODY_BACKEND']='compiled'
    cat=Catalogue(args.cache);sites=read_sites(args.observatories)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        endpoints=Time([cat.tracks['t'].min()-66,cat.tracks['t'].max()+2],format='mjd',scale='utc').tdb.mjd
    ep=Ephemeris(args.ephemeris,*endpoints)
    _WORKER=(recovery_geometry(args,cat,sites,ep),args,deadline)


def run_job(job):
    key,record,attempt=job;geometry,args,deadline=_WORKER
    limit=min(deadline,time.monotonic()+args.fit_seconds*min(attempt,4))
    try:
        fitted=fit_nbody(geometry,record['indices'],np.array(record['initial']),record['epoch'],
            sigma=record['sigma'],deadline=limit,max_nfev=140,fast_start=False)
        if fitted is None:return key,'rejected',None,None
        rank,mode=map(int,record['seed_key'].split(':'))
        row=candidate_row(fitted,geometry,rank,mode,record['sigma'])
        row['extension_kind']='deferred_full_precision_fit'
        return key,'accepted',row,None
    except (BudgetExpired,ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
        return key,'retryable',None,type(exc).__name__+': '+str(exc)


def import_jobs(db,directories):
    from .deferred_queue import import_incremental
    return import_incremental(db,directories)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','observatories','ephemeris','campaign-geometry','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source',type=Path,action='append',required=True)
    p.add_argument('--seconds',type=float,default=1800)
    p.add_argument('--fit-seconds',type=float,default=30)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--max-jobs',type=int,default=0)
    a=p.parse_args(argv)
    if min(a.seconds,a.fit_seconds,a.workers)<=0 or a.max_jobs<0:p.error('Invalid compute budget')
    started=time.monotonic();deadline=started+a.seconds
    a.output.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(a.output/'replay.sqlite');db.execute('PRAGMA journal_mode=WAL')
    db.executescript('CREATE TABLE IF NOT EXISTS jobs (key TEXT PRIMARY KEY,body TEXT,status TEXT,attempts INTEGER,error TEXT);'
        'CREATE TABLE IF NOT EXISTS candidates (key TEXT PRIMARY KEY,body TEXT); CREATE TABLE IF NOT EXISTS metadata (identity TEXT);')
    cat=Catalogue(a.cache)
    identity=dict(cache=cat.manifest,ephemeris_sha256=sha256(a.ephemeris),observatories_sha256=sha256(a.observatories),
        implementation={name:sha256(Path(__file__).with_name(name)) for name in ('replay_growth.py','nbody.py','nbody_fast.py','longarc.py','deferred_queue.py')})
    encoded=json.dumps(identity,sort_keys=True);old=db.execute('SELECT identity FROM metadata').fetchone()
    if old and old[0]!=encoded:p.error('Replay configuration changed; use a new output directory')
    if not old:
        with db:db.execute('INSERT INTO metadata VALUES (?)',(encoded,))
    imported=import_jobs(db,a.source);print(dict(imported=imported),flush=True)
    from .deferred_queue import pending_jobs,WarmBank
    from .report_io import candidates
    from collections import deque
    warm=WarmBank(db)
    # Re-ingestion is idempotent and only reads the modest accepted-chain bank.
    for source in a.source:
        report=source/'report.json'
        if report.exists():
            with db:
                for row in candidates(report):warm.add(row)
    db.execute('CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status,attempts,key)');db.commit()
    def export():
        statuses=dict(db.execute('SELECT status,COUNT(*) FROM jobs GROUP BY status'))
        rows=[json.loads(row[0]) for row in db.execute('SELECT body FROM candidates')]
        rows.sort(key=lambda row:(-row['span_days'],row['chi2']/row['observations']))
        mark_maximal(rows)
        for rank,row in enumerate(rows,1):row['rank']=rank
        pending=statuses.get('pending',0)+statuses.get('retryable',0)
        counts=dict(jobs=sum(statuses.values()),jobs_remaining=pending,statuses=statuses,candidate_chains=len(rows))
        report=dict(input=cat.manifest,identity=identity,candidates=rows,counts=counts,
            status='full_fit_queue_complete' if not pending else 'work_remaining',exhaustive=False,probabilities_calibrated=False)
        atomic_json(a.output/'report.json',report)
        atomic_json(a.output/'progress.json',dict(status=report['status'],counts=counts,elapsed_seconds=time.monotonic()-started,updated_utc=time.time()))
        return counts
    export()
    iterator=iter(pending_jobs(db,a.max_jobs));last_export=0.;supplements=deque();submitted=set();original_turn=True
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize,initargs=(a,deadline)) as pool:
        pending=set()
        def submit():
            nonlocal original_turn
            if time.monotonic()>=deadline:return
            if a.max_jobs and len(submitted)>=a.max_jobs:return
            # One supplemental slot alternates with one original slot. Original
            # starts stay pending/retryable even when a supplement succeeds.
            job=None
            if supplements and not original_turn:job=supplements.popleft()
            else:
                while job is None:
                    item=next(iterator,None)
                    if item is None:break
                    if item[0] not in submitted:job=(item[0],json.loads(item[1]),item[2]+1)
            original_turn=not original_turn
            if job is None and supplements:job=supplements.popleft()
            if job:
                submitted.add(job[0]);pending.add(pool.submit(run_job,job))
        for _ in range(a.workers):submit()
        while pending:
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                key,status,row,error=future.result()
                with db:
                    db.execute('UPDATE jobs SET status=?,attempts=attempts+1,error=? WHERE key=?',(status,error,key))
                    if row:
                        warm.add(row)
                        chain=','.join(map(str,row['tracklet_indices']))
                        prior=db.execute('SELECT body FROM candidates WHERE key=?',(chain,)).fetchone()
                        row=merge_candidate_rows(json.loads(prior[0]) if prior else None,row)
                        db.execute('INSERT OR REPLACE INTO candidates VALUES (?,?)',(chain,json.dumps(row)))
                    elif status in ('retryable','rejected'):
                        record=json.loads(db.execute('SELECT body FROM jobs WHERE key=?',(key,)).fetchone()[0])
                        if 'warm_parent' not in record:
                            suggested=warm.suggest(key,record)
                            if suggested:supplements.append((suggested[0],suggested[1],1))
                if time.monotonic()-last_export>=30:print(export(),flush=True);last_export=time.monotonic()
                submit()
    print(export(),flush=True);db.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
