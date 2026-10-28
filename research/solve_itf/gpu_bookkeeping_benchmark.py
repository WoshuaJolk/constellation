"""Paired old/new whole-search benchmark; never a production linking run."""
import argparse,importlib.util,json,time,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from magictools.itf.catalogue import Catalogue
from magictools.itf.fit import read_sites
from magictools.itf.search import solve,SearchConfig
from magictools.itf.warmup import compile_kernels
from gpu_pipeline_adapter import cpu_batch,GPUBatch
from gpu_pipeline_benchmark import grade


def main():
    p=argparse.ArgumentParser();p.add_argument('--catalogue',required=True,type=Path);p.add_argument('--sites',required=True);p.add_argument('--output',required=True,type=Path);p.add_argument('--cpu-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('magictools.itf._frozen_bookkeeping',Path(__file__).parent/'gpu-bookkeeping-receipts/reference_fit.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    def old_batch(geometry,jobs):
        with ThreadPoolExecutor(max_workers=20) as pool:
            return list(pool.map(lambda job:ref.fit_chain(geometry,job[0],**job[1]),jobs))
    cat=Catalogue(a.catalogue);sites=read_sites(a.sites);truth=json.loads((a.catalogue/'truth.json').read_text())
    config=SearchConfig(workers=1,max_seeds=256,max_edges=100000,max_gap_days=14,broad_search=True,seed_neighbors=0,seconds=120,branches=4)
    report=dict(status='running',phase='warmup',runs={},outputs={},scope='Identical bounded 747-tracklet fixture; all broad starts, checks, mode uncertainties, extension and ranked outputs. CPU20-thread old reference versus revised CPU and revised hybridGPU. No large catalogue linking.')
    def save():(a.output/'report.json').write_text(json.dumps(report,indent=2))
    def write(name,data):
        path=a.output/(name+'.json');path.write_text(json.dumps(data));report['outputs'][name]=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    save();compile_kernels();results={}
    names=['old_cpu','new_cpu']+([] if a.cpu_only else ['new_gpu'])
    for rep in range(3):
        for name in names if rep%2==0 else names[::-1]:
            report['phase']=f'{name}-{rep}';save();audit=[]
            adapter=old_batch if name=='old_cpu' else cpu_batch if name=='new_cpu' else GPUBatch()
            t=time.perf_counter();result=solve(cat,sites,config,progress=lambda _:None,audit=audit.append,batch_fitter=adapter);seconds=time.perf_counter()-t
            write(f'{name}-{rep}',result);write(f'{name}-{rep}-audit',audit)
            row=dict(seconds=seconds,timings=result['timings'],counts=result['counts'],control_recovery=grade(result,cat,truth))
            if name=='new_gpu':row['batch_receipts']=adapter.receipts
            report['runs'].setdefault(name,[]).append(row);results[name]=result;save()
    arows,brows=results['old_cpu']['candidates'],results['new_cpu']['candidates']
    assert [r['tracklet_indices'] for r in arows]==[r['tracklet_indices'] for r in brows]
    uncertainty_max_relative=0.;modes=0
    for arow,brow in zip(arows,brows):
        for k in ['rms_arcsec','seeds_attempted','observations','log_bayes_factor_bic']:
            assert arow[k]==brow[k],k
        assert len(arow['modes'])==len(brow['modes'])
        for am,bm in zip(arow['modes'],brow['modes']):
            modes+=1
            for k in ['state_au_au_per_day','epoch_mjd_tdb','chi2','rms_arcsec','jacobian_rank','relative_mode_likelihood']:
                assert am[k]==bm[k],k
            np.testing.assert_allclose(am['range_au'],bm['range_au'],rtol=1e-14,atol=1e-12)
            for k,av in am['local_uncertainty'].items():
                bv=bm['local_uncertainty'][k]
                if isinstance(av,float):
                    np.testing.assert_allclose(av,bv,rtol=1e-8,atol=1e-10)
                    uncertainty_max_relative=max(uncertainty_max_relative,abs(av-bv)/max(abs(av),1e-30))
                else:assert av==bv,k
    report['old_new_cpu_parity']=dict(memberships=len(arows),modes=modes,states_residuals_ranks_and_weights_exact=True,uncertainty_max_relative_difference=uncertainty_max_relative)
    if 'new_gpu' in results:
        keys=lambda result:set(tuple(row['tracklet_indices']) for row in result['candidates'])
        cpu,gpu=keys(results['old_cpu']),keys(results['new_gpu'])
        report['gpu_memberships']=dict(shared=len(cpu&gpu),cpu_only=len(cpu-gpu),gpu_only=len(gpu-cpu))
    report['status']='completed';report['phase']='done';save();print(json.dumps({k:v for k,v in report.items() if k not in ['runs','outputs']},indent=2))

if __name__=='__main__':main()
