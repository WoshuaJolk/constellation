#!/usr/bin/env python3
"""Frozen production-grid local-fit benchmark; no chain acceptance or new search."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from magictools.itf.batched_fit import CPUEvaluator
from magictools.itf.gpu import ResidentBank
from magictools.itf.gpu_fit import solve_resident
from gpu_benchmark import subset,compare
from gpu_fit_benchmark import original,summary,disagreement,instrumented_cpu


def efficient_cpu_residuals(bank):
    evaluator=CPUEvaluator(bank)
    r,j=evaluator(bank['states'],np.ones(len(bank['states']),np.uint8))
    r=r.reshape(-1,2);j=j.reshape(-1,2,6)
    valid=np.isfinite(r).all(axis=1)&np.isfinite(j).all(axis=(1,2))&~(r==1e12).all(axis=1)
    group=np.logical_and.reduceat(valid,evaluator.bounds[:-1])
    valid=np.repeat(group,np.diff(evaluator.bounds)).astype(np.uint8)
    return r,j,valid


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    b=dict(np.load(a.bank,allow_pickle=False));a.output.mkdir(parents=True,exist_ok=True)
    report={'scope':__doc__,'source_bank_sha256':hashlib.sha256(a.bank.read_bytes()).hexdigest(),
            'starts':len(b['states']),'unique_parent_chains':len(b['states'])//24,'iterations_limit':80,
            'repeats':3,'runs':{},'outputs':{},'status':'running',
            'cpu_driver':'Compiled original solve_lm scheduling blocks; no Python per-fit loop',
            'limitations':['Parents are retained candidates, not all attempted ITF combinations',
                           '24 genuine range/radial starts per parent; no Gauss or saved starts',
                           'Numerical convergence is not astrometric/physical acceptance',
                           'Different local outcomes may occur from GPU floating-point ordering']}
    def save():(a.output/'report.json').write_text(json.dumps(report,indent=2))
    def preserve(name,r):
        path=a.output/(name+'.npz');np.savez_compressed(path,**{k:r[k] for k in ('states','success','cost')},**{k:r[k] for k in ('reasons','iterations','evaluations') if k in r})
        report['outputs'][name]={'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    try:
        save();stamp=time.perf_counter();warm=subset(b,0,128)
        original(warm,1,80);solve_resident(warm,80)
        report['warmup_seconds']=time.perf_counter()-stamp;save()
        report['phase']='initial_same_state_parity';save()
        report['initial_same_state_parity']=compare(efficient_cpu_residuals(b),ResidentBank(b).evaluate(),float(b['sigma']));save()
        reference=None
        for threads in [4,20]:
            report['runs']['cpu-'+str(threads)]=[]
            for repetition in range(3):
                report['phase']=f'cpu-{threads}-repeat-{repetition}';save()
                r=original(b,threads,80);row=summary(r,b)
                if reference is None:reference=r;preserve('cpu-reference',r)
                else:
                    row['reference_comparison']=disagreement(reference,r,b)
                    if row['reference_comparison']['exact_state_matches']!=len(b['states']) or row['reference_comparison']['success_disagreements']:
                        raise RuntimeError('CPU repeat/threading differs')
                report['runs']['cpu-'+str(threads)].append(row);save()
        report['runs']['gpu-resident']=[]
        first=None
        for repetition in range(3):
            report['phase']=f'gpu-repeat-{repetition}';save()
            r=solve_resident(b,80);row=summary(r,b);row['reference_comparison']=disagreement(reference,r,b)
            if first is None:first=r;preserve('gpu-reference',r)
            else:row['first_gpu_repeat_comparison']=disagreement(first,r,b)
            report['runs']['gpu-resident'].append(row);save()
        report['phase']='final_same_state_parity';save()
        final=dict(b,states=r['states'])
        report['final_same_state_parity']=compare(efficient_cpu_residuals(final),ResidentBank(final).evaluate(),float(b['sigma']));save()
        report['phase']='separate_profiles';save()
        report['instrumented_original_cpu']=instrumented_cpu(b,80);save()
        r=solve_resident(b,80,profile_events=True)
        report['instrumented_gpu']=summary(r,b);report['instrumented_gpu']['first_gpu_comparison']=disagreement(first,r,b)
        report['status']='completed';report['phase']='done';save()
    except Exception as e:
        report['status']='blocked';report['blocker']=type(e).__name__+': '+str(e);save();raise
    print(json.dumps({'status':report['status'],'starts':report['starts'],'parents':report['unique_parent_chains']}))

if __name__=='__main__':main()
