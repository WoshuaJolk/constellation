"""Bounded whole-search comparison with independently simulated distant controls."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from magictools.itf.catalogue import Catalogue,unit_vectors
from magictools.itf.fit import read_sites
from magictools.itf.search import solve,SearchConfig
from magictools.itf.kernels import directions,ARCSEC
from magictools.itf.night_query import query_nights
from gpu_pipeline_adapter import GPUBatch,cpu_batch
from astropy.time import Time
import erfa


def grade(result,cat,truth):
    ids=[x.decode().strip() for x in cat.tracks['id']];owners={track:t['identity'] for t in truth for track in t['tracklet_ids']};rows=[]
    for t in truth:
        pure=[];mixed=[]
        for r in result['candidates']:
            names=[ids[i] for i in r['tracklet_indices']]
            if any(owners.get(n)==t['identity'] for n in names):
                (pure if all(owners.get(n)==t['identity'] for n in names) else mixed).append(r)
        rows.append(dict(identity=t['identity'],distance_au=t['distance_au'],inclination_deg=t['inclination_deg'],pure_chains=len(pure),mixed_chains=len(mixed),longest_pure_tracklets=max([len(r['tracklet_indices']) for r in pure],default=0),retained_modes=sum(len(r['modes']) for r in pure)))
    return rows


def future_diagnostic(result,cat,reserved):
    # All retained modes, fixed two known windows. No truth-based mode selection.
    # This is a predeclared nominal 5-arcsec diagnostic (production default is 6 arcsec), not uncertainty recovery.
    modes=[(r,m) for r in result['candidates'] for m in r['modes']]
    hits=[];failures=0;started=time.perf_counter()
    all_times=np.array([test['times_utc'] for test in reserved])
    unique,inverse=np.unique(all_times,return_inverse=True);inverse=inverse.reshape(all_times.shape)
    utc=Time(unique,format='mjd',scale='utc');t=utc.tdb;earth,_=erfa.epv00(t.jd1,t.jd2)
    observed_pool=[unit_vectors(np.array(test['ra_mas'])/3600000,np.array(test['dec_mas'])/3600000) for test in reserved]
    for row,mode in modes:
        bank=directions(np.array(mode['state_au_au_per_day']),t.mjd-mode['epoch_mjd_tdb'],earth['p'])
        for test,indices,observed in zip(reserved,inverse,observed_pool):
            predicted=bank[indices]
            if not np.isfinite(predicted).all():failures+=1;continue
            error=np.arctan2(np.linalg.norm(np.cross(predicted,observed),axis=1),np.sum(predicted*observed,axis=1))*ARCSEC
            if error.max()<=5: hits.append(dict(membership=row['tracklet_indices'],epoch=mode['epoch_mjd_tdb'],test_id=test['id'],rms_arcsec=float(np.sqrt(np.mean(error**2))),maximum_arcsec=float(error.max())))
    return dict(seconds=time.perf_counter()-started,modes=len(modes),test_tracklets=len(reserved),hits=hits,failed_predictions=failures,scope='CPU nominal prediction screen across all modes and fixed reserved windows; no unknown-object archival census, uncertainty propagation or planetary refitting. Not included in short-arc search speedup.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--catalogue',type=Path,required=True);p.add_argument('--sites',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    stamp=time.perf_counter();cat=Catalogue(a.catalogue);sites=read_sites(a.sites);load=time.perf_counter()-stamp
    truth=json.loads((a.catalogue/'truth.json').read_text());reserved=json.loads((a.catalogue/'reserved.json').read_text())
    config=SearchConfig(workers=1,max_seeds=256,max_edges=100000,max_gap_days=14,broad_search=True,seed_neighbors=0,seconds=120,branches=4)
    report=dict(status='running',catalogue=cat.manifest,load_seconds=load,runs={},outputs={},scope='Full bounded short-arc search, same 256-seed cap and broad Gauss/range starts; CPU20-thread batches versus GPU+CPU checked batches. Synthetic controls are not survey completeness.')
    def save():(a.output/'report.json').write_text(json.dumps(report,indent=2))
    def write(name,data):
        path=a.output/(name+'.json');path.write_text(json.dumps(data));report['outputs'][name]=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    try:
      # Numerical cold-start compile is recorded separately, not hidden in GPU-only timing.
      from magictools.itf.warmup import compile_kernels
      report['phase']='warmup';save();start=time.perf_counter();compile_kernels();report['cpu_warmup_seconds']=time.perf_counter()-start
      backends=['cpu'] if a.cpu_only else ['cpu','gpu']
      for backend in backends:
        report['runs'][backend]=[]
        for rep in range(2):
            report['phase']=f'{backend}-{rep}';save();audit=[];adapter=cpu_batch if backend=='cpu' else GPUBatch()
            start=time.perf_counter();result=solve(cat,sites,config,progress=lambda _:None,audit=audit.append,batch_fitter=adapter);elapsed=time.perf_counter()-start
            write(f'{backend}-{rep}',result);write(f'{backend}-{rep}-audit',audit)
            row=dict(seconds=elapsed,status=result['status'],timings=result['timings'],counts=result['counts'],control_recovery=grade(result,cat,truth),memberships=[r['tracklet_indices'] for r in result['candidates']])
            if backend=='gpu':row['batch_receipts']=adapter.receipts
            report['runs'][backend].append(row);save()
        # Separate later-observation diagnostic; same frozen selection and no refit.
        report['phase']=backend+'-future';save();later=future_diagnostic(result,cat,reserved);write(backend+'-future',later);report.setdefault('future',{})[backend]={k:v for k,v in later.items() if k!='hits'};save()
      if not a.cpu_only:
        cpu=set(map(tuple,report['runs']['cpu'][1]['memberships']));gpu=set(map(tuple,report['runs']['gpu'][1]['memberships']))
        report['comparison']=dict(shared=len(cpu&gpu),cpu_only=[list(x) for x in sorted(cpu-gpu)],gpu_only=[list(x) for x in sorted(gpu-cpu)])
      report['status']='completed';report['phase']='done';save()
    except Exception as e:report['status']='blocked';report['error']=type(e).__name__+': '+str(e);save();raise

if __name__=='__main__':main()
