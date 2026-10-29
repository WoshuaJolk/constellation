"""Fixed-cohort columnar Jacobian and per-observation accepted-copy experiments; not production."""
import argparse, hashlib, json, time, gc
from pathlib import Path
import numpy as np
from numba import cuda
from magictools.itf.gpu import ResidentBank
from magictools.itf.gpu_fit import ResidentWorkspace, solve_resident
from gpu_benchmark import subset, compare
from gpu_cold_benchmark import efficient_cpu_residuals
from gpu_fit_benchmark import original, summary, disagreement


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    b=dict(np.load(a.bank,allow_pickle=False));a.output.mkdir(parents=True,exist_ok=True)
    report=dict(bank_sha256=hashlib.sha256(a.bank.read_bytes()).hexdigest(),starts=len(b['states']),
                status='running', runs={}, outputs={}, iterations=80, repeats=3,
                variants=['parallel-baseline','columnar','parallel-copy','columnar-copy'],
                limitation='FP64 two-body local fits on fixed retained parents; no production acceptance or full pipeline claim')
    def save(): (a.output/'report.json').write_text(json.dumps(report,indent=2))
    def preserve(name,r):
        path=a.output/(name+'.npz');np.savez_compressed(path,**{k:r[k] for k in ('states','success','cost')},**{k:r[k] for k in ('reasons','iterations','evaluations') if k in r})
        report['outputs'][name]=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    try:
        report['phase']='warmup';save();stamp=time.perf_counter();small=subset(b,0,128)
        original(small,1,80)
        for columnar,copy in [(False,False),(True,False),(False,True),(True,True)]:
            w=ResidentWorkspace(small,coalesced=True,columnar_jacobian=columnar);w.solve(small['states'],80,parallel_normal=True,parallel_copy=copy)
        report['warmup_seconds']=time.perf_counter()-stamp
        report['phase']='CPU20';save();cpu=None;report['runs']['cpu-20']=[]
        for i in range(3):
            r=original(b,20,80);row=summary(r,b)
            if cpu is None:cpu=r;preserve('cpu-reference',r)
            else:row['repeat_comparison']=disagreement(cpu,r,b)
            report['runs']['cpu-20'].append(row);save()
        base=None
        for name in report['variants']:
            report['phase']=name;save();workspace=None
            workspace=ResidentWorkspace(b,coalesced=True,columnar_jacobian='columnar' in name)
            report['runs'][name]=[];first=None
            for rep in range(3):
                r=workspace.solve(b['states'],80,parallel_normal=True,parallel_copy='copy' in name)
                row=summary(r,b);row['cpu_comparison']=disagreement(cpu,r,b)
                if base is None:base=r
                row['baseline_gpu_comparison']=disagreement(base,r,b)
                if first is None:first=r;preserve(name,r)
                else:row['repeat_comparison']=disagreement(first,r,b)
                report['runs'][name].append(row);save()
            # Preserve failed checks rather than prematurely discarding timings or masks.
            report['phase']=name+'-same-state-check';save()
            final=dict(b,states=first['states']);ref=efficient_cpu_residuals(final);act=ResidentBank(final).evaluate()
            mask_path=a.output/(name+'-masks.npz');np.savez_compressed(mask_path,cpu=ref[2],gpu=act[2])
            report['outputs'][name+'-masks']=dict(path=mask_path.name,sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest())
            try:parity=dict(status='passed',**compare(ref,act,float(b['sigma'])))
            except ValueError as e:parity=dict(status='failed',reason=str(e),observation_mask_disagreements=int(np.count_nonzero(ref[2]!=act[2])),case_mask_disagreements=int(len(np.unique(b['case_index'][ref[2]!=act[2]]))))
            report.setdefault('parity',{})[name]=parity;save()
            del ref,act,final,r,first,workspace;gc.collect();cuda.current_context().deallocations.clear()
        report['phase']='profiles';save()
        report['profiles']={}
        for columnar,copy in [(False,False),(True,True)]:
            w=ResidentWorkspace(b,coalesced=True,columnar_jacobian=columnar)
            r=w.solve(b['states'],80,parallel_normal=True,parallel_copy=copy,profile_events=True)
            report['profiles'][f'columnar={columnar};copy={copy}']=r['profile'];save()
            del w,r;gc.collect();cuda.current_context().deallocations.clear()
        report['status']='completed_with_numerical_blockers' if any(x['status']=='failed' for x in report['parity'].values()) else 'completed'
        report['phase']='done';save()
    except Exception as e:
        report['status']='blocked';report['error']=type(e).__name__+': '+str(e);save();raise

if __name__=='__main__':main()
