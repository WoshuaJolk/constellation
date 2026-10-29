"""Sustained checkpoint benchmark on the original full catalogue and 7M seeds.

Outputs are benchmark evidence, never additions to a production candidate list.
CPU/GPU fitting use the same complete start policy. Planetary recovery is a
separate, explicitly CPU-only downstream stage on a frozen historical mode bank.
"""
import argparse,csv,gzip,hashlib,json,multiprocessing as mp,os,subprocess,threading,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from astropy.time import Time
from magictools.itf.catalogue import Catalogue,sha256
from magictools.itf.campaign_geometry import AtlasGeometry
from magictools.itf.fit import prepare_fit,fit_prepared,read_sites
from magictools.itf.warmup import compile_kernels
from magictools.itf.nbody import Ephemeris,JPLGeometry
from magictools.itf.longarc import NightIndex,attach_arcs,recover_seed
from gpu_pipeline_adapter import GPUBatch


def seed_positions(begin,count,total=7000000):
    if total!=7000000:raise ValueError('Frozen queue length differs')
    return (104729*np.arange(begin,begin+count,dtype=np.int64)+20261028)%total


def cpu_case(geometry,key,settings):
    starts=0
    try:
        plan=prepare_fit(geometry,key,broad_starts=True)
        if plan is None:return None,0,None
        starts=sum(bool(np.isfinite(state).all()) for state,_ in plan.seeds)
        return fit_prepared(plan,**settings),starts,None
    except (ValueError,np.linalg.LinAlgError,FloatingPointError) as e:
        return None,starts,type(e).__name__+': '+str(e)


_RECOVERY=None

def recovery_init(output):
    # Pool is forked BEFORE the main process creates a CUDA context. Shared
    # catalogue/index arrays are read-only; caches become process-local on write.
    global _RECOVERY
    geometry,index,ephemeris,args=_RECOVERY
    args.output=Path(output)/'longgap-private'/str(os.getpid());args.output.mkdir(parents=True,exist_ok=True)
    os.environ['ITF_NBODY_BACKEND']='compiled'


def recovery_ping(value):return os.getpid()


def recovery_case(record,deadline):
    geometry,index,ephemeris,args=_RECOVERY
    row=record['row'];mode=record['mode'];before=time.perf_counter()
    seed=(0.,-float(row.get('span_days',0)),record['source_rank'],record['source_mode'],row,mode)
    key,status,counts,rows=recover_seed(seed,geometry,index,ephemeris,args,deadline)
    return dict(digest=record['digest'],roles=record['roles'],key=key,status=status,
                seconds=time.perf_counter()-before,counts=dict(counts),rows=rows)


def cpu_ticks():
    values=np.array(list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9])),dtype=np.int64)
    return int(values.sum()),int(values[3]+values[4])


def monitor(path,phase,stop):
    before=cpu_ticks();stamp=time.monotonic()
    with path.open('w') as f:
        while not stop.wait(5):
            now=cpu_ticks();t=time.monotonic();total=now[0]-before[0];idle=now[1]-before[1]
            row=dict(unix=time.time(),phase=phase[0],interval_seconds=t-stamp,cpu_busy_percent=100*(1-idle/total) if total else None)
            try:
                line=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=3).strip()
                row['gpu_util_percent'],row['gpu_memory_util_percent'],row['gpu_memory_used_mib'],row['gpu_power_watts']=map(float,line.split(','))
            except Exception as e:row['gpu_monitor_error']=type(e).__name__
            f.write(json.dumps(row)+'\n');f.flush();before,stamp=now,t


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=300);p.add_argument('--longgap-seconds',type=float,default=180);a=p.parse_args()
    root=a.root;a.output.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',phase='verify_inputs',outputs={},fits={},scope='Original 7M checkpoint throughput, plus separate historical-parent planetary recovery against full catalogue. No fresh graph generation or production linking.',setup={})
    phase=['verify_inputs'];stop=threading.Event();thread=threading.Thread(target=monitor,args=(a.output/'utilization.jsonl',phase,stop),daemon=True);thread.start()
    def save():
        report['updated_unix']=time.time();report['phase']=phase[0]
        temp=a.output/'report.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(a.output/'report.json')
    def artifact(path):
        report['outputs'][path.name]=dict(path=path.name,bytes=path.stat().st_size,sha256=sha256(path));save()
    pool=None
    try:
        save();t=time.perf_counter();manifest=json.loads((root/'data-manifest.json').read_text())
        for name,record in manifest.items():
            if sha256(root/name)!=record['sha256']:raise ValueError('Input checksum mismatch: '+name)
        report['setup']['verify_all_input_seconds']=time.perf_counter()-t
        phase[0]='load_checkpoint';save();t=time.perf_counter()
        geometry=AtlasGeometry(root/'catalogue',root/'ObsCodes.html',root/'atlas');cat=geometry.catalogue
        queue=np.load(root/'seeds.npy',mmap_mode='r');assert queue.shape==(7000000,3)
        graph=[np.load(root/'graph'/n,mmap_mode='r') for n in ['extended-offsets.npy','extended-neighbors.npy','extended-scores.npy']]
        # Touch every reused cache page. This is loading, NOT rebuilding the graph.
        checksums={}
        for name,array in [('seeds',queue),('graph_offsets',graph[0]),('graph_neighbors',graph[1]),('graph_scores',graph[2]),('times',geometry.times),('units',geometry.unit),('observers',geometry.observers)]:
            checksums[name]=float(np.nansum(array))
        report['setup'].update(checkpoint_load_and_touch_seconds=time.perf_counter()-t,checkpoint_checksums=checksums,tracklets=len(cat.tracks),observations=len(cat.obs),seeds=len(queue),graph_directed_neighbors=len(graph[1]),observer_unavailable=int(np.sum(geometry.ready==0)))
        del graph
        # Full-catalogue planetary infrastructure is shared by a CPU-only stage.
        phase[0]='planetary_setup';save();t=time.perf_counter();sites=read_sites(root/'ObsCodes.html')
        endpoints=Time([cat.tracks['t'].min()-66,cat.tracks['t'].max()+2],format='mjd',scale='utc').tdb.mjd
        ep=Ephemeris(root/'de440s.bsp',*endpoints);index=NightIndex(cat,sites,ep)
        parents=json.loads((root/'longgap-parents.json').read_text());groups=[r['row']['tracklet_indices'] for r in parents]
        attach_arcs(index,groups,cat)
        args=SimpleNamespace(ephemeris=root/'de440s.bsp',backend='compiled',output=None,sigma_arcsec=.3,
             seed_seconds=30.,max_seed_seconds=30.,fit_seconds=5.,fast_fit=False,fit_cascade=False,
             staged=True,bridge_first=True,iterations=2,branches=4,min_gap_days=0.,max_gap_days=0.,
             max_cone_deg=2.,max_candidates=16,retrieval='position',position_reserve=0,
             uncertainty_samples=12,direct_arcs=True,swap_limit=0,fast_retrieval=False,no_work_reuse=False)
        global _RECOVERY
        _RECOVERY=(JPLGeometry(cat,sites,ep),index,ep,args)
        pool=ProcessPoolExecutor(max_workers=20,mp_context=mp.get_context('fork'),initializer=recovery_init,initargs=(str(a.output),))
        list(pool.map(recovery_ping,range(20)))
        report['setup'].update(planetary_index_pool_seconds=time.perf_counter()-t,night_index=index.counts,ephemeris_bounds=[ep.start,ep.end],longgap_parent_count=len(parents))
        phase[0]='warmup';save();t=time.perf_counter();compile_kernels()
        # Fixed final queue positions are disjoint from the two timed ranges.
        warmkeys=queue[seed_positions(6999984,16)]
        valid=[tuple(map(int,key)) for key in warmkeys if np.all(geometry.ready[key])]
        settings=dict(sigma=.3,rms_limit=.6,max_nfev=80,start_policy='full')
        with ThreadPoolExecutor(max_workers=20) as cpu:
            list(cpu.map(lambda key:cpu_case(geometry,key,settings),valid))
        warm_gpu=GPUBatch();warm_gpu(geometry,[(key,dict(settings,broad_starts=True)) for key in valid])
        report['setup']['warmup_seconds']=time.perf_counter()-t
        # AB/BA orders; second deterministic range starts at a distinct ordinal.
        for round_id,names in enumerate([['cpu','gpu'],['gpu','cpu']]):
            for backend in names:
                name=f'{backend}-{round_id}';phase[0]=name;save();adapter=GPUBatch() if backend=='gpu' else None
                start=time.perf_counter();cursor=1000000*round_id;end=cursor+1000000
                counters=Counter();within=Counter();batch_receipts=[];path=a.output/(name+'.jsonl.gz')
                with ThreadPoolExecutor(max_workers=20) as cpu,gzip.open(path,'wt',compresslevel=1) as stream:
                    while time.perf_counter()-start<a.seconds and cursor<end:
                        before=time.perf_counter();ordinals=np.arange(cursor,min(cursor+1024,end),dtype=np.int64);positions=seed_positions(cursor,len(ordinals));keys=queue[positions]
                        usable=np.all(geometry.ready[keys]!=0,axis=1);valid=[tuple(map(int,key)) for key in keys[usable]]
                        if backend=='cpu':
                            computed=list(cpu.map(lambda key:cpu_case(geometry,key,settings),valid))
                            rows=[r[0] for r in computed];starts=sum(r[1] for r in computed);errors=[r[2] for r in computed]
                        else:
                            rows=adapter(geometry,[(key,dict(settings,broad_starts=True)) for key in valid]);errors=[None]*len(rows)
                            starts=adapter.receipts[-1]['starts'] if valid else 0
                        at=time.perf_counter()-start;eligible=at<=a.seconds;batch=Counter(attempted=len(valid),unavailable=int((~usable).sum()),starts=starts,retained=sum(row is not None for row in rows),numerical_errors=sum(e is not None for e in errors),modes=sum(len(row['modes']) for row in rows if row is not None))
                        batch['not_retained']=batch['attempted']-batch['retained']-batch['numerical_errors'];counters.update(batch)
                        if eligible:within.update(batch)
                        iterator=iter(zip(rows,errors))
                        for ordinal,position,key,good in zip(ordinals,positions,keys,usable):
                            row,error=next(iterator) if good else (None,'Observer geometry unavailable')
                            status='unavailable' if not good else 'numerical_error' if error else 'retained' if row is not None else 'not_retained'
                            stream.write(json.dumps(dict(queue_ordinal=int(ordinal),source_position=int(position),membership=list(map(int,key)),status=status,error=error,within_budget=eligible,row=row),separators=(',',':'))+'\n')
                        stream.flush();cursor+=len(ordinals)
                        receipt=dict(begin_ordinal=int(ordinals[0]),count=len(ordinals),seconds=time.perf_counter()-before,fit_complete_elapsed_seconds=at,completion_elapsed_seconds=time.perf_counter()-start,within_budget=eligible,counts=dict(batch))
                        if backend=='gpu':receipt['gpu']=adapter.receipts[-1]
                        batch_receipts.append(receipt)
                        report['fits'][name]=dict(budget_seconds=a.seconds,elapsed_seconds=time.perf_counter()-start,counts=dict(counters),within_budget_counts=dict(within),batches=len(batch_receipts),queue_start=1000000*round_id,next_ordinal=cursor)
                        save()
                report['fits'][name]['elapsed_seconds']=time.perf_counter()-start;artifact(path)
                receipts=a.output/(name+'-batches.json');receipts.write_text(json.dumps(batch_receipts,indent=2));artifact(receipts)
        # Match only seed positions genuinely completed within BOTH time limits.
        phase[0]='paired_comparison';save();comparisons=[]
        for round_id in range(2):
            def read(name):
                with gzip.open(a.output/f'{name}-{round_id}.jsonl.gz','rt') as f:
                    return {r['source_position']:r for r in map(json.loads,f) if r['within_budget']}
            cpu,gpu=read('cpu'),read('gpu');common=sorted(cpu.keys()&gpu.keys());different=[];rms_differences=[];mode_counts=0;both=0
            for key in common:
                arow,brow=cpu[key],gpu[key]
                if arow['status']!=brow['status']:different.append(dict(source_position=key,cpu=arow['status'],gpu=brow['status']))
                if arow['row'] is not None and brow['row'] is not None:
                    both+=1;delta=abs(arow['row']['rms_arcsec']-brow['row']['rms_arcsec'])
                    if delta>.001:rms_differences.append(dict(source_position=key,cpu_rms=arow['row']['rms_arcsec'],gpu_rms=brow['row']['rms_arcsec']))
                    mode_counts+=len(arow['row']['modes'])!=len(brow['row']['modes'])
            comparisons.append(dict(round=round_id,common_within_budget=len(common),both_retained=both,status_differences=different,rms_differences_over_001_arcsec=rms_differences,mode_count_differences=mode_counts))
        file=a.output/'paired-comparison.json';file.write_text(json.dumps(comparisons,indent=2));artifact(file);report['comparisons']=[{k:(len(v) if isinstance(v,list) else v) for k,v in x.items()} for x in comparisons]
        phase[0]='planetary_longgap';save();begin=time.perf_counter();deadline=time.monotonic()+a.longgap_seconds;iterator=iter(parents);pending={};summaries=[]
        def submit():
            if time.monotonic()>=deadline:return
            item=next(iterator,None)
            if item is not None:pending[pool.submit(recovery_case,item,deadline)]=item['digest']
        for _ in range(20):submit()
        path=a.output/'longgap.jsonl.gz'
        with gzip.open(path,'wt',compresslevel=1) as stream:
            while pending:
                done,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                for future in done:
                    digest=pending.pop(future)
                    try:row=future.result()
                    except Exception as e:row=dict(digest=digest,status='worker_error',error=type(e).__name__+': '+str(e),rows=[],counts={})
                    stream.write(json.dumps(row,separators=(',',':'))+'\n');stream.flush();summaries.append({k:v for k,v in row.items() if k!='rows'});submit()
                counts=Counter();statuses=Counter()
                for r in summaries:counts.update(r['counts']);statuses[r['status']]+=1
                report['longgap']=dict(elapsed_seconds=time.perf_counter()-begin,budget_seconds=a.longgap_seconds,completed=len(summaries),total_frozen_parents=len(parents),pending=len(pending),not_launched=len(parents)-len(summaries)-len(pending),counts=dict(counts),statuses=dict(statuses),backend='CPU20 compiled planetary dynamics, shared downstream stage; no GPU acceleration claim');save()
        artifact(path)
        pool.shutdown(wait=True);pool=None
        import tarfile
        cache_archive=a.output/'longgap-caches.tar.gz'
        with tarfile.open(cache_archive,'w:gz',compresslevel=1) as tar:
            tar.add(a.output/'longgap-private',arcname='longgap-private')
        artifact(cache_archive)
        phase[0]='done';report['status']='completed_bounded_benchmark';save()
    except Exception as e:
        report['status']='blocked';report['error']=type(e).__name__+': '+str(e);save();raise
    finally:
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        stop.set();thread.join(timeout=5);artifact(a.output/'utilization.jsonl')

if __name__=='__main__':main()
