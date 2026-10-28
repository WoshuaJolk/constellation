"""Fixed distinct-seed scaling, shared graph and unchanged per-chain fit policy."""
import argparse,gzip,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from gpu_integrated_benchmark import write


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def config(seeds,seconds=900):
    from magictools.itf.search import SearchConfig
    return SearchConfig(workers=20,max_gap_days=60,neighbors_per_day=32,max_edges=3000000,
        max_seeds=seeds,seed_neighbors=0,broad_search=False,seconds=seconds)


def geometry():
    from magictools.itf.campaign_geometry_fast import FastAtlasGeometry
    return FastAtlasGeometry('catalogue','ObsCodes.html','atlas','observation-identities')


def inventory(root):
    import numpy as np
    from magictools.itf.search import candidate_edges,make_seeds
    g=geometry();c=config(1000000);counts={};started=time.perf_counter();deadline=time.monotonic()+300
    edges=candidate_edges(g.catalogue.tracks,c,deadline,lambda x:print(x,flush=True),counts)
    scan=time.perf_counter()-started
    seeds,_=make_seeds(edges,g.catalogue.tracks,c,counts,deadline)
    keys=np.asarray(seeds,dtype=np.int64).reshape(-1,3)
    if len(np.unique(keys,axis=0))!=len(keys):raise ValueError('Duplicate seeds')
    np.save(root/'edges.npy',edges);np.save(root/'seeds.npy',keys)
    record=dict(counts=counts,config=vars(c),scan_seconds=scan,total_seconds=time.perf_counter()-started,
        available_seeds=len(keys),edges_sha256=digest(root/'edges.npy'),seeds_sha256=digest(root/'seeds.npy'),
        scope='One frozen 60-day graph; original fast fit starts/cuts; no broad augmentation; no replicated benchmark seeds')
    write(root/'inventory.json',record)
    return record


def child(args):
    import numpy as np
    from magictools.itf import search
    from magictools.itf.cli import format_rows
    started=time.perf_counter();root=args.output;out=root/args.name;out.mkdir()
    inv=json.loads((root/'inventory.json').read_text())
    for name in ['edges','seeds']:
        if digest(root/(name+'.npy'))!=inv[name+'_sha256']:raise ValueError('Frozen graph changed')
    edges=np.load(root/'edges.npy',mmap_mode='r');keys=np.load(root/'seeds.npy',mmap_mode='r')[:args.seeds]
    g=geometry();adapter=None;original_seeds=search.make_seeds
    # Reuse only the frozen proposal graph. Adjacent lists retain all original edges.
    def frozen_edges(tracks,c,deadline,progress,counts):
        counts.update(inv['counts']);return edges
    def frozen_seeds(edges,tracks,c,counts,deadline):
        from collections import defaultdict
        adjacent=defaultdict(list)
        for a,b,score in edges:
            adjacent[int(a)].append((float(score),int(b)));adjacent[int(b)].append((float(score),int(a)))
        for links in adjacent.values():links.sort()
        counts['seeds_retained']=len(keys)
        return [tuple(map(int,k)) for k in keys],adjacent
    search.candidate_edges=frozen_edges;search.make_seeds=frozen_seeds
    if args.backend=='stream':
        from gpu_stream_adapter import StreamGPUBatch
        adapter=StreamGPUBatch()
    elif args.backend=='prefetch':
        from gpu_prefetch_adapter import PrefetchGPUBatch
        adapter=PrefetchGPUBatch()
    setup=time.perf_counter()-started
    try:result=search.solve(g.catalogue,g.sites,config(len(keys)),geometry=g,batch_fitter=adapter)
    finally:
        if adapter is not None:adapter.close()
    # Save every candidate and mode; no favorable subset selected for comparison.
    with gzip.open(out/'report.json.gz','wt',compresslevel=1) as f:json.dump(result,f,allow_nan=False)
    elapsed=time.perf_counter()-started
    record=dict(name=args.name,backend=args.backend,requested_seeds=args.seeds,actual_seeds=len(keys),
        seed_sha256=hashlib.sha256(keys.tobytes()).hexdigest(),status=result['status'],counts=result['counts'],
        timings=result['timings'],application_seconds=elapsed,setup_seconds=setup,
        with_shared_graph_build_seconds=elapsed+inv['total_seconds'],candidates=len(result['candidates']),
        batch_receipts=adapter.receipts if adapter else [],pipeline=adapter.pipeline if adapter else {},
        timing_scope='includes graph loading, adjacent reconstruction, fit, growth and gzip output; graph discovery/build separately measured once')
    write(out/'summary.json',record)
    return 0 if result['status']=='bounded_search_complete' else 1


def parity(root):
    import numpy as np
    from gpu_stream_adapter import StreamGPUBatch
    from gpu_compact_adapter import CompactGPUBatch
    keys=np.load(root/'seeds.npy',mmap_mode='r')[:128]
    g=geometry();jobs=[(tuple(map(int,k)),dict(sigma=.3,rms_limit=.6,max_nfev=80)) for k in keys]
    batches=[jobs[i:i+32] for i in range(0,len(jobs),32)]
    started=time.perf_counter()
    with StreamGPUBatch(batch_size=32) as adapter:
        actual=list(adapter.iter_batches(g,iter(batches)));pipeline=adapter.pipeline.copy()
    reference=CompactGPUBatch(completion_interval=16)
    expected=[reference(g,batch) for batch in batches]
    equal=[rows==old for (_,rows),old in zip(actual,expected)]
    write(root/'stream-parity.json',dict(chains=len(keys),batches=len(batches),exact_rows=equal,
        pipeline=pipeline,seconds=time.perf_counter()-started,remaining_shared_directories=[]))
    return 0 if len(actual)==len(expected) and all(equal) else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('scaling-results'))
    p.add_argument('--parity',action='store_true');p.add_argument('--backend',choices=['cpu','prefetch','stream']);p.add_argument('--seeds',type=int);p.add_argument('--name');a=p.parse_args()
    if a.parity:return parity(a.output)
    if a.backend:return child(a)
    a.output.mkdir(exist_ok=False);root=a.output;report=dict(status='running',phase='verify-inputs',phases=[],outputs={});started=time.perf_counter()
    def save():write(root/'report.json',report)
    def record(path):report['outputs'][str(path.relative_to(root))]=dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=digest(path))
    save()
    for manifest in ['data-manifest.json','extra-manifest.json']:
        for name,v in json.loads(Path(manifest).read_text()).items():
            if digest(name)!=v['sha256']:raise ValueError('Input changed: '+name)
    report['phase']='freeze-distinct-pool';save();inv=inventory(root);record(root/'inventory.json');save()
    expected={'edges_sha256':'1b99c06602fdfe15bbcb38fe3465d662b73c5a2cbc12b318af4c60f92b5f0d58','seeds_sha256':'3da886aa22d431dc2088149ac3c728988b700cf81077bb06b2c264a31e09e10d'}
    if any(inv[k]!=v for k,v in expected.items()):
        report.update(status='blocked',failure_phase='recreated-pool-hash');save();return 1
    report['phase']='stream-parity';save()
    with (root/'stream-parity.log').open('w') as stream:
        code=subprocess.run([sys.executable,__file__,'--output',str(root),'--parity'],stdout=stream,stderr=subprocess.STDOUT,timeout=180).returncode
    for file in [root/'stream-parity.log',root/'stream-parity.json']:
        if file.exists():record(file)
    if code!=0:
        report.update(status='blocked',failure_phase='stream-parity');save();return 1
    # Insufficient pool never causes repetition or another search configuration.
    sizes=sorted(set(min(n,inv['available_seeds']) for n in [100000,300000,1000000]))
    report['sizes']=sizes;report['shortfall']=max(0,1000000-inv['available_seeds']);save()
    if not sizes or sizes==[0]:raise ValueError('No distinct seeds')
    for size in sizes:
        backends=['cpu','prefetch','stream'] if size==sizes[0] else ['cpu','stream']
        for backend in backends:
            name=f'{backend}-{size}';report['phase']=name;save()
            before=time.perf_counter()
            with (root/(name+'.log')).open('w') as stream:
                try:code=subprocess.run([sys.executable,__file__,'--output',str(root),'--backend',backend,'--seeds',str(size),'--name',name],stdout=stream,stderr=subprocess.STDOUT,timeout=1000).returncode
                except subprocess.TimeoutExpired:code='timeout'
            phase=dict(name=name,exit_code=code,process_seconds=time.perf_counter()-before)
            for path in [root/(name+'.log'),root/name/'summary.json',root/name/'report.json.gz']:
                if path.exists():record(path)
            if (root/name/'summary.json').exists():
                row=json.loads((root/name/'summary.json').read_text());phase.update({k:v for k,v in row.items() if k not in ['batch_receipts']})
            report['phases'].append(phase);save()
            if code!=0:
                report.update(status='blocked',failure_phase=name);save();return 1
    report.update(status='complete',phase='done',elapsed_seconds=time.perf_counter()-started);save();return 0


if __name__=='__main__':raise SystemExit(main())
