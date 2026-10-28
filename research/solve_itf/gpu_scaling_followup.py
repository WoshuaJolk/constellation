"""Prospectively bounded host-copy follow-up on the SAME frozen seed pool."""
import argparse,gzip,json,subprocess,sys,time,hashlib
from pathlib import Path
from gpu_integrated_benchmark import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('scaling-v2-results'))
    p.add_argument('--child',action='store_true');p.add_argument('--parity',action='store_true')
    p.add_argument('--seeds',type=int);p.add_argument('--seconds',type=int,default=240);a=p.parse_args()
    if a.child or a.parity:
        import gpu_scaling_benchmark as benchmark
        import gpu_stream_adapter as stream
        from gpu_stream_adapter_v2 import StreamGPUBatchV2
        stream.StreamGPUBatch=StreamGPUBatchV2
        if a.parity:return benchmark.parity(a.root)
        config=benchmark.config;benchmark.config=lambda n:config(n,a.seconds)
        return benchmark.child(argparse.Namespace(output=a.root,name=f'streamv2-{a.seeds}',backend='stream',seeds=a.seeds))
    contract=json.loads(Path('followup-contract.json').read_text());deadline=contract['original_deadline_unix']
    # These choices use only remaining time, never retained rows or prediction results.
    remaining=deadline-time.time()
    sizes=([1000000,100000] if remaining>=650 else [100000] if remaining>=300 else [])
    a.root.mkdir();report=dict(status='running',phase='prepare',sizes=sizes,outputs={},phases=[],
        remaining_seconds_at_selection=remaining,rule='>=650s:1M then100K if time; >=300s:100K; otherwise skip',prior_results_not_replaced=True)
    def save():write(a.root/'report.json',report)
    def record(path):report['outputs'][str(path.relative_to(a.root))]=dict(path=str(path.relative_to(a.root)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    save()
    if not sizes:report.update(status='skipped_budget',phase='done');save();return 0
    for name in ['edges.npy','seeds.npy','inventory.json']:(a.root/name).symlink_to((Path('scaling-results')/name).resolve())
    report['phase']='parity';save()
    with (a.root/'parity.log').open('w') as log:
        result=subprocess.run([sys.executable,__file__,'--root',str(a.root),'--parity'],stdout=log,stderr=subprocess.STDOUT,timeout=100)
    for file in [a.root/'parity.log',a.root/'stream-parity.json']:
        if file.exists():record(file)
    if result.returncode!=0:report.update(status='blocked',phase='parity');save();return 1
    for size in sizes:
        limit=min({100000:180,300000:360,1000000:720}[size],int(deadline-time.time())-90)
        if limit<120:
            report.setdefault('budget_skipped',[]).append(size);continue
        name=f'streamv2-{size}';report['phase']=name;save();start=time.perf_counter()
        with (a.root/(name+'.log')).open('w') as log:
            try:code=subprocess.run([sys.executable,__file__,'--root',str(a.root),'--child','--seeds',str(size),'--seconds',str(limit)],stdout=log,stderr=subprocess.STDOUT,timeout=limit+30).returncode
            except subprocess.TimeoutExpired:code='timeout'
        phase=dict(name=name,seeds=size,exit_code=code,process_seconds=time.perf_counter()-start)
        for file in [a.root/(name+'.log'),a.root/name/'summary.json',a.root/name/'report.json.gz']:
            if file.exists():record(file)
        if (a.root/name/'summary.json').exists():
            row=json.loads((a.root/name/'summary.json').read_text());phase.update({k:v for k,v in row.items() if k!='batch_receipts'})
        report['phases'].append(phase);save()
        if code!=0:report.update(status='blocked',phase=name);save();return 1
    report.update(status='complete' if not report.get('budget_skipped') else 'partial_budget',phase='done');save();return 0


if __name__=='__main__':raise SystemExit(main())
