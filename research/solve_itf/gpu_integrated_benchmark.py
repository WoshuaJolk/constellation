"""Matched cached-catalogue full searches with ordered prefetch and indexed CUDA."""
import argparse,gzip,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path


def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def child(args):
    import numpy as np
    from magictools.itf.catalogue import Catalogue,sha256
    from magictools.itf.campaign_geometry_fast import FastAtlasGeometry
    from magictools.itf.search import SearchConfig,solve
    from magictools.itf.cli import format_rows
    started=time.perf_counter();args.output.mkdir(exist_ok=False)
    g=FastAtlasGeometry('catalogue','ObsCodes.html','atlas','observation-identities');cat=g.catalogue
    config=SearchConfig(workers=args.workers,max_seeds=args.seeds,seed_neighbors=5 if args.seeds==10000 else 0,seconds=480)
    audit=[];adapter=None
    if args.backend=='compact':
        from gpu_compact_adapter import CompactGPUBatch
        adapter=CompactGPUBatch()
    elif args.backend in ('prefetch','indexed'):
        from gpu_prefetch_adapter import PrefetchGPUBatch
        adapter=PrefetchGPUBatch(indexed=args.backend=='indexed',register_normal=args.register_normal)
    setup=time.perf_counter()-started
    try:
        result=solve(cat,g.sites,config,deadline=time.monotonic()+480,progress=lambda x:print(x,flush=True),audit=audit.append,batch_fitter=adapter,geometry=g)
    finally:
        if hasattr(adapter,'close'):adapter.close()
    result['timings']['application_setup_seconds']=setup
    result['timings']['before_output_seconds']=time.perf_counter()-started
    output_start=time.perf_counter()
    (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output/'candidates.txt').write_text(format_rows(result))
    (args.output/'maximal_candidates.txt').write_text(format_rows(result,maximal_only=True))
    output_seconds=time.perf_counter()-output_start
    seeds=next(x['seeds'] for x in audit if x['event']=='seed_bank')
    extra=dict(backend=args.backend,config=result['config'],seed_count=len(seeds),seed_sha256=hashlib.sha256(json.dumps(seeds,separators=(',',':')).encode()).hexdigest(),profile_keys=seeds[:1024],output_seconds=output_seconds,batch_receipts=adapter.receipts if adapter else [])
    if hasattr(adapter,'pipeline'):extra['pipeline']=adapter.pipeline
    if getattr(adapter,'resident_geometry',None) is not None:
        r=adapter.resident_geometry
        extra['resident_geometry']=dict(upload_bytes=r.upload_bytes,setup_seconds=r.setup_seconds,total_preparation_seconds=r.total_preparation_seconds,bindings=r.bindings,avoided_upload_bytes=r.avoided_upload_bytes,validation_seconds=r.validation_seconds)
    write(args.output/'extra.json',extra)
    return 0


def profile(args):
    import numpy as np
    from magictools.itf.campaign_geometry_fast import FastAtlasGeometry
    from magictools.itf.gpu_fit import ResidentWorkspace
    from magictools.itf.gpu_indexed import ResidentGeometry
    from gpu_batch_host import prepare_bank
    g=FastAtlasGeometry('catalogue','ObsCodes.html','atlas','observation-identities')
    source=json.loads((args.output/'prefetch-large'/'extra.json').read_text())
    jobs=[(tuple(k),dict(sigma=.3,rms_limit=.6,max_nfev=80,broad_starts=False)) for k in source['profile_keys']]
    start=time.perf_counter();bank,*_=prepare_bank(g,jobs)
    resident=ResidentGeometry.from_atlas(g);reports={};outputs={}
    for name,shared,interval in [('copied-poll4',None,4),('copied-poll16',None,16),('indexed-poll16',resident,16)]:
        w=ResidentWorkspace(bank,coalesced=True,resident_geometry=shared)
        w.solve(bank['states'],80,parallel_normal=True,parallel_copy=True,completion_interval=interval)
        out=w.solve(bank['states'],80,parallel_normal=True,parallel_copy=True,completion_interval=interval,profile_events=True)
        outputs[name]=out;reports[name]=out['profile']
    parity={}
    for name in ['copied-poll16','indexed-poll16']:
        parity[name]={k:bool(np.array_equal(outputs[name][k],outputs['copied-poll4'][k],equal_nan=True)) for k in ['states','success','cost','reasons','iterations','evaluations']}
    write(args.output/'kernel-profile.json',dict(chains=len(jobs),starts=len(bank['states']),seconds=time.perf_counter()-start,profiles=reports,parity=parity,warning='Separate replay with per-kernel events; not headline timing. No new hypotheses or cuts.'))
    return 0 if all(all(v.values()) for v in parity.values()) else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('integrated-results'));parser.add_argument('--backend',choices=['cpu','compact','prefetch','indexed']);parser.add_argument('--workers',type=int,default=20);parser.add_argument('--seeds',type=int,default=10000);parser.add_argument('--profile',action='store_true');parser.add_argument('--register-normal',action='store_true');parser.add_argument('--register-followup',action='store_true');args=parser.parse_args()
    if args.backend:return child(args)
    if args.profile:return profile(args)
    args.output.mkdir(exist_ok=False)
    report=dict(status='running',phase='verify-inputs',phases=[],outputs={},scope='Full catalogue scan, fixed seed budget, unchanged fast-profile fit/growth policy, reused hash-verified catalogue/atlas; no production update')
    def save():write(args.output/'report.json',report)
    def record(path):report['outputs'][str(path.relative_to(args.output))]=dict(path=str(path.relative_to(args.output)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    save();start=time.perf_counter()
    from magictools.itf.catalogue import sha256
    for manifest in ['data-manifest.json','extra-manifest.json']:
        for name,item in json.loads(Path(manifest).read_text()).items():
            if sha256(name)!=item['sha256']:raise ValueError('Input hash changed: '+name)
    report['input_verification_seconds']=time.perf_counter()-start;save()
    schedule=[('cpu4-10k','cpu',4,10000),('cpu20-10k','cpu',20,10000),('compact-10k','compact',20,10000),('prefetch-10k','prefetch',20,10000),('cpu20-large','cpu',20,100000),('prefetch-large','prefetch',20,100000),('indexed-10k','indexed',20,10000),('indexed-large','indexed',20,100000)]
    if args.register_followup:
        report['phase']='register-check';save()
        with (args.output/'register-check.log').open('x') as stream:
            code=subprocess.run([sys.executable,'research/solve_itf/gpu_register_check.py','--output',str(args.output)],stdout=stream,stderr=subprocess.STDOUT,timeout=180).returncode
        for file in [args.output/'register-check.log',args.output/'register-kernel.json',args.output/'disagreement-replay.json']:
            if file.exists():record(file)
        if code!=0:
            report.update(status='blocked',failure_phase='register-check',check_exit_code=code);save();return 1
        schedule=[('register-10k','prefetch',20,10000),('register-large','prefetch',20,100000)]
    for name,backend,workers,seeds in schedule:
        report['phase']=name;save();out=args.output/name
        command=[sys.executable,__file__,'--output',str(out),'--backend',backend,'--workers',str(workers),'--seeds',str(seeds)]
        if args.register_followup:command.append('--register-normal')
        before=resource.getrusage(resource.RUSAGE_CHILDREN);started=time.perf_counter()
        with (args.output/(name+'.log')).open('x') as stream:
            try:code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=550).returncode
            except subprocess.TimeoutExpired:code='subprocess_timeout'
        elapsed=time.perf_counter()-started;after=resource.getrusage(resource.RUSAGE_CHILDREN)
        row=dict(name=name,backend=backend,workers=workers,seed_budget=seeds,elapsed_seconds=elapsed,exit_code=code,user_cpu_seconds=after.ru_utime-before.ru_utime,system_cpu_seconds=after.ru_stime-before.ru_stime)
        if (out/'report.json').exists():
            result=json.loads((out/'report.json').read_text());row.update(status=result['status'],timings=result['timings'],counts=result['counts'],candidates=len(result['candidates']))
            gz=args.output/(name+'-report.json.gz')
            with gzip.open(gz,'wb',compresslevel=1) as f:f.write((out/'report.json').read_bytes())
            record(gz)
        for file in [args.output/(name+'.log'),out/'extra.json']:
            if file.exists():record(file)
        report['phases'].append(row);save()
        if code!=0 or row.get('status')!='bounded_search_complete':
            report.update(status='blocked',failure_phase=name);save();return 1
    if args.register_followup:
        report.update(status='complete',phase='done');save();return 0
    report['phase']='kernel-profile';save()
    with (args.output/'kernel-profile.log').open('x') as stream:
        code=subprocess.run([sys.executable,__file__,'--output',str(args.output),'--profile'],stdout=stream,stderr=subprocess.STDOUT,timeout=180).returncode
    for file in [args.output/'kernel-profile.log',args.output/'kernel-profile.json']:
        if file.exists():record(file)
    report.update(status='complete' if code==0 else 'blocked',phase='done' if code==0 else 'kernel-profile',profile_exit_code=code);save()
    return code


if __name__=='__main__':raise SystemExit(main())
