"""Fixed supplementary CPU process baseline; no concurrent scientific workload."""
import argparse,gzip,json,multiprocessing as mp,time,threading,os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Match the main benchmark before importing NumPy.
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np
from magictools.itf.campaign_geometry import AtlasGeometry
from magictools.itf.catalogue import sha256
from gpu_deep_benchmark import cpu_case,seed_positions,monitor

_G=None
_SETTINGS=dict(sigma=.3,rms_limit=.6,max_nfev=80,start_policy='full')

def run(key):return cpu_case(_G,key,_SETTINGS)

def ready(_):return __import__("os").getpid()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    global _G
    before=time.perf_counter();_G=AtlasGeometry('catalogue','ObsCodes.html','atlas');q=np.load('seeds.npy',mmap_mode='r')
    for key in q[seed_positions(6999984,16)]:
        if np.all(_G.ready[key]):run(tuple(map(int,key)))
    setup=time.perf_counter()-before;path=a.output/'cpu-process.jsonl.gz';counts=Counter();within=Counter();cursor=0;budget=120.;receipts=[]
    with ProcessPoolExecutor(max_workers=20,mp_context=mp.get_context('fork')) as pool,gzip.open(path,'wt',compresslevel=1) as stream:
        list(pool.map(ready,range(20)))  # Fork before the monitoring thread exists.
        setup=time.perf_counter()-before
        start=time.perf_counter();stop=threading.Event();thread=threading.Thread(target=monitor,args=(a.output/'utilization.jsonl',['cpu_process'],stop),daemon=True);thread.start()
        while time.perf_counter()-start<budget:
            t=time.perf_counter();positions=seed_positions(cursor,1024);keys=q[positions];usable=np.all(_G.ready[keys]!=0,axis=1)
            computed=list(pool.map(run,[tuple(map(int,key)) for key in keys[usable]],chunksize=8));at=time.perf_counter()-start;eligible=at<=budget
            c=Counter(attempted=len(computed),unavailable=int((~usable).sum()),starts=sum(r[1] for r in computed),retained=sum(r[0] is not None for r in computed),numerical_errors=sum(r[2] is not None for r in computed),modes=sum(len(r[0]['modes']) for r in computed if r[0] is not None));c['not_retained']=c['attempted']-c['retained']-c['numerical_errors'];counts.update(c)
            if eligible:within.update(c)
            items=iter(computed)
            for position,key,good in zip(positions,keys,usable):
                row,starts,error=next(items) if good else (None,0,'Observer geometry unavailable')
                stream.write(json.dumps(dict(source_position=int(position),membership=list(map(int,key)),within_budget=eligible,row=row,error=error),separators=(',',':'))+'\n')
            stream.flush();cursor+=1024;receipts.append(dict(cursor=cursor,fit_complete_elapsed_seconds=at,seconds=time.perf_counter()-t,within_budget=eligible,counts=dict(c)))
            report=dict(status='running',setup_seconds=setup,budget_seconds=budget,elapsed_seconds=time.perf_counter()-start,counts=dict(counts),within_budget_counts=dict(within),receipts=receipts)
            tmp=a.output/'report.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(a.output/'report.json')
    stop.set();thread.join(timeout=5)
    report.update(status='completed',elapsed_seconds=time.perf_counter()-start,output={'path':path.name,'bytes':path.stat().st_size,'sha256':sha256(path)})
    (a.output/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
