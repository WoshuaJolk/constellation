"""Fixed real-catalogue host-batching comparison; no production search updates."""
import argparse,gzip,hashlib,importlib.util,json,multiprocessing as mp,sys,threading,time
from collections import Counter,deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from magictools.itf.campaign_geometry import AtlasGeometry
from magictools.itf.campaign_geometry_fast import FastAtlasGeometry
from magictools.itf.catalogue import sha256
from gpu_deep_benchmark import seed_positions,monitor
from gpu_batch_host import prepare_bank
from gpu_compact_adapter import CompactGPUBatch
import gpu_deep_cpu_process_check as process_cpu

_G = None
_SETTINGS = dict(sigma=.3,rms_limit=.6,max_nfev=80,start_policy='full',broad_starts=True)

def prepare_task(keys):
    jobs=[(tuple(map(int,k)),dict(_SETTINGS)) for k in keys]
    t=time.perf_counter();value=prepare_bank(_G,jobs)
    return value,jobs,time.perf_counter()-t

def ready(_):return True

def load_reference(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',phase='setup',phases={},outputs={},setup={})
    def save():
        report['updated_unix']=time.time();tmp=a.output/'report.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(a.output/'report.json')
    def artifact(path):
        report['outputs'][path.name]=dict(path=path.name,bytes=path.stat().st_size,sha256=sha256(path));save()
    save();before=time.perf_counter()
    for name,record in json.loads(Path('data-manifest.json').read_text()).items():
        assert sha256(name)==record['sha256'],name
    for name,record in json.loads(Path('extra-manifest.json').read_text()).items():
        assert sha256(name)==record['sha256'],name
    global _G
    _G=FastAtlasGeometry('catalogue','ObsCodes.html','atlas','observation-identities')
    legacy_geometry=AtlasGeometry('catalogue','ObsCodes.html','atlas');q=np.load('seeds.npy',mmap_mode='r')
    old=load_reference('magictools.itf._host_frozen','research/solve_itf/gpu-host-receipts/reference_fit.py')
    adapter=load_reference('_host_frozen_adapter','research/solve_itf/gpu-host-receipts/reference_adapter.py')
    adapter.prepare_fit=old.prepare_fit;adapter.fit_prepared=old.fit_prepared
    process_cpu._G=_G
    warm=q[seed_positions(6999984,16)]
    prepare_task(warm)
    for key in warm:process_cpu.run(tuple(map(int,key)))
    # Fork all workers before starting CUDA or monitor threads. Idle fit workers
    # perform no work during GPU phases; preprocessing is capped at four banks.
    prep_pool=ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('fork'))
    fit_pool=ProcessPoolExecutor(max_workers=20,mp_context=mp.get_context('fork'))
    list(prep_pool.map(ready,range(4)));list(fit_pool.map(process_cpu.ready,range(20)))
    report['setup']['cpu_checkpoint_pool_seconds']=time.perf_counter()-before
    phase=['warmup'];stop=threading.Event();thread=threading.Thread(target=monitor,args=(a.output/'utilization.jsonl',phase,stop),daemon=True);thread.start()
    try:
        t=time.perf_counter();old_gpu=adapter.GPUBatch();new_gpu=CompactGPUBatch();jobs=[(tuple(map(int,k)),dict(_SETTINGS)) for k in warm]
        old_rows=old_gpu(legacy_geometry,jobs);new_rows=new_gpu(_G,jobs)
        assert old_rows==new_rows,'GPU frozen warmup output differs'
        report['setup']['gpu_warmup_seconds']=time.perf_counter()-t;save()
        schedule=[('legacy_gpu',120,0),('compact_gpu',120,0),('prefetch_gpu_0',180,0),('process_cpu',120,0),('prefetch_gpu_1',180,1000000)]
        for name,budget,begin in schedule:
            phase[0]=name;report['phase']=name;save();start=time.perf_counter();cursor=begin;scheduled=begin;pending=deque();receipts=[];counts=Counter();within=Counter()
            gpu=adapter.GPUBatch() if name=='legacy_gpu' else CompactGPUBatch()
            path=a.output/(name+'.jsonl.gz')
            def submit():
                nonlocal scheduled
                positions=seed_positions(scheduled,1024);pending.append((scheduled,positions,prep_pool.submit(prepare_task,q[positions])));scheduled+=1024
            if name.startswith('prefetch'):
                for _ in range(4):submit()
            with gzip.open(path,'wt',compresslevel=1) as stream:
                while time.perf_counter()-start<budget:
                    t=time.perf_counter();positions=seed_positions(cursor,1024);keys=q[positions];jobs=[(tuple(map(int,k)),dict(_SETTINGS)) for k in keys]
                    if not np.all(_G.ready[keys]):raise ValueError('Unavailable geometry in frozen timed prefix; preserve partial run')
                    errors=[None]*len(keys)
                    if name=='process_cpu':
                        values=list(fit_pool.map(process_cpu.run,[tuple(map(int,k)) for k in keys],chunksize=8));rows=[v[0] for v in values];starts=sum(v[1] for v in values);errors=[v[2] for v in values]
                    elif name.startswith('prefetch'):
                        ordinal,positions,future=pending.popleft();assert ordinal==cursor
                        packed,jobs,planning=future.result();rows=gpu.execute_prepared(packed,jobs,planning_seconds=planning);starts=gpu.receipts[-1]['starts']
                    else:
                        rows=gpu(legacy_geometry if name=='legacy_gpu' else _G,jobs);starts=gpu.receipts[-1]['starts']
                    at=time.perf_counter()-start;eligible=at<=budget
                    batch=Counter(attempted=len(rows),starts=starts,retained=sum(r is not None for r in rows),numerical_errors=sum(e is not None for e in errors),modes=sum(len(r['modes']) for r in rows if r is not None));batch['not_retained']=batch['attempted']-batch['retained']-batch['numerical_errors'];counts.update(batch)
                    if eligible:within.update(batch)
                    for position,key,row,error in zip(positions,keys,rows,errors):
                        stream.write(json.dumps(dict(source_position=int(position),membership=list(map(int,key)),within_budget=eligible,row=row,error=error),separators=(',',':'))+'\n')
                    stream.flush();receipt=dict(begin_ordinal=cursor,seconds=time.perf_counter()-t,fit_complete_elapsed_seconds=at,within_budget=eligible,counts=dict(batch))
                    if name!='process_cpu':receipt['gpu']=gpu.receipts[-1]
                    receipts.append(receipt);cursor+=1024
                    if name.startswith('prefetch') and time.perf_counter()-start<budget:submit()
                    report['phases'][name]=dict(budget_seconds=budget,elapsed_seconds=time.perf_counter()-start,counts=dict(counts),within_budget_counts=dict(within),next_ordinal=cursor,seed_begin=begin);save()
            cancelled=0;unused=0
            while pending:
                _,_,future=pending.popleft()
                if future.cancel():cancelled+=1
                else:future.result();unused+=1
            report['phases'][name].update(elapsed_seconds=time.perf_counter()-start,unused_prepared_batches=unused,cancelled_preparations=cancelled)
            artifact(path);record=a.output/(name+'-batches.json');record.write_text(json.dumps(receipts,indent=2));artifact(record)
        report['status']='completed';report['phase']='done';save()
    except Exception as exc:
        report.update(status='blocked',error=type(exc).__name__+': '+str(exc));save();raise
    finally:
        prep_pool.shutdown(wait=True,cancel_futures=True);fit_pool.shutdown(wait=True,cancel_futures=True);stop.set();thread.join(timeout=5);artifact(a.output/'utilization.jsonl')

if __name__=='__main__':main()
