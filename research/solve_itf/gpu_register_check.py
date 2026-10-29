"""Fixed arithmetic-preserving normal-matrix check and two discrepancy replays."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from magictools.itf.campaign_geometry_fast import FastAtlasGeometry
from magictools.itf.fit import fit_chain
from magictools.itf.gpu_fit import ResidentWorkspace
from gpu_batch_host import prepare_bank
from gpu_pipeline_adapter import GPUBatch
from gpu_compact_adapter import CompactGPUBatch


def same(a,b):
    return {k:bool(np.array_equal(a[k],b[k],equal_nan=True)) for k in ['states','success','cost','reasons','iterations','evaluations']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();a.output.mkdir(exist_ok=True)
    d=json.loads(Path('register-design.json').read_text());g=FastAtlasGeometry('catalogue','ObsCodes.html','atlas','observation-identities')
    settings=dict(sigma=.3,rms_limit=.6,max_nfev=80,broad_starts=False)
    jobs=[(tuple(k),settings) for k in d['disagreement_keys']]
    results={'cpu':[fit_chain(g,key,**settings) for key,_ in jobs],
             'legacy_gpu':GPUBatch()(g,jobs),
             'compact_gpu':CompactGPUBatch()(g,jobs),
             'register_gpu':CompactGPUBatch(completion_interval=16,register_normal=True)(g,jobs)}
    (a.output/'disagreement-replay.json').write_text(json.dumps({'keys':d['disagreement_keys'],'settings':settings,'results':results},indent=2)+'\n')
    jobs=[(tuple(k),settings) for k in d['profile_keys']];bank,*_=prepare_bank(g,jobs)
    w=ResidentWorkspace(bank,coalesced=True);reports={};outputs={}
    for register in [False,True]:
        name='register' if register else 'existing'
        w.solve(bank['states'],80,parallel_normal=True,parallel_copy=True,completion_interval=16,register_normal=register)
        times=[]
        for _ in range(3):
            start=time.perf_counter();out=w.solve(bank['states'],80,parallel_normal=True,parallel_copy=True,completion_interval=16,register_normal=register);times.append(time.perf_counter()-start)
        outputs[name]=out
        prof=w.solve(bank['states'],80,parallel_normal=True,parallel_copy=True,completion_interval=16,register_normal=register,profile_events=True)
        reports[name]={'times':times,'median_seconds':float(np.median(times)),'profile':prof['profile'],'state_sha256':hashlib.sha256(out['states'].tobytes()).hexdigest()}
    parity=same(outputs['existing'],outputs['register'])
    result={'chains':len(jobs),'starts':len(bank['states']),'parity':parity,'timings':reports,'warning':'Fixed1024-chain replay. Uninstrumented medians and separate event profiles. Not full-search speedup.'}
    (a.output/'register-kernel.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if all(parity.values()) and results['legacy_gpu']==results['register_gpu'] else 1


if __name__=='__main__':raise SystemExit(main())
