"""Plan the full stress domain using an explicit 4 by 4 range/radial-rate cover.

Each tile retains the original cadence, bandwidth and error target. Complete
patch results are restartable. No large field or interpolation matrix is built.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from flint import arb,ctx
from research.detection_limit.interval_orbits import upper_float
from research.kepler_transform.constructive_experiments import domain
from research.kepler_transform.local_hierarchy import LocalAtlasPlan
from research.kepler_transform.local_resolution_experiments import hashes
from research.kepler_transform.factorization import ARCSEC
from research.kepler_transform.reception_flow import CircularObserver
from research.kepler_transform.verify_constructive_campaign import verify_rectangular_cover


def provenance():
    source=hashes()
    source[str(Path(__file__).resolve().relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return source


def tiles(center,widths):
    patches=[{'center':list(center),'widths':list(widths)}]
    for axis in [0,1,0,1]:
        following=[]
        for patch in patches:
            for sign in [-1,1]:
                c,w=list(patch['center']),list(patch['widths'])
                half=arb(w[axis])/2
                exact=arb(c[axis])+sign*half
                c[axis]=float(exact.mid())
                w[axis]=upper_float(half+abs(exact-arb(c[axis])))
                following.append({'center':c,'widths':w})
        patches=following
    return patches


def worker(index,patch,times,cache_directory):
    ctx.prec=100
    begin=perf_counter()
    try:
        atlas=LocalAtlasPlan(CircularObserver(),patch['center'],patch['widths'],times,[1/ARCSEC]*2,
                             max_patches=8,neighborhood_ellipses=(1.05,2.,4.,8.,32.),
                             max_step=.125,cache_directory=cache_directory,
                             progress=lambda record: print(json.dumps({'tile':index,**record}),flush=True))
        return {'index':index,'status':'complete','atlas':atlas.record(),'seconds':perf_counter()-begin}
    except ValueError as error:
        return {'index':index,'status':'refused','reason':str(error),'seconds':perf_counter()-begin}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--cache-directory',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    args.directory.mkdir(parents=True,exist_ok=True)
    ctx.prec=100
    source=provenance()
    *_,geom,c,w=domain('stress')
    patches=tiles(c,w)
    verify_rectangular_cover(c,w,patches)
    begin=perf_counter()
    results={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={}
        for i,patch in enumerate(patches):
            path=args.directory/f'patch-{i}.json'
            if path.exists():
                saved=json.loads(path.read_text())
                if saved['source_sha256']!=source:
                    raise ValueError('Checkpoint source changed')
                if saved['result']['status']=='complete':
                    results[i]=saved['result']
                    continue
            pending[pool.submit(worker,i,patch,geom.times.tolist(),str(args.cache_directory))]=i
        for future in as_completed(pending):
            i=pending[future]
            result=future.result()
            results[i]=result
            if provenance()!=source:
                raise RuntimeError('Source changed during tiled campaign')
            (args.directory/f'patch-{i}.json').write_text(json.dumps({'source_sha256':source,'result':result},indent=2,allow_nan=False)+'\n')
            print(json.dumps({'finished_tile':i,'status':result['status'],'seconds':result['seconds']}),flush=True)
    if any(r['status']!='complete' for r in results.values()):
        raise ValueError('Some tiles refused; no complete-domain certificate returned')
    atlases=[results[i]['atlas'] for i in range(len(patches))]
    plans=[p for a in atlases for p in a['plans']]
    domains=[p for a in atlases for p in a['patch_domains']]
    verify_rectangular_cover(c,w,domains)
    sum_keys=['root_nodes','total_stored_field_nodes','total_transfer_entries',
              'total_transfer_multiply_add_pairs','total_geometry_state_evaluations_bound',
              'sum_intermediate_nodes','coefficient_storage_bytes']
    stats={key:sum(a['stats'][key] for a in atlases) for key in sum_keys}
    stats.update({'patches':len(domains),'failed_patch_attempts':sum(a['stats']['failed_patch_attempts'] for a in atlases),
                  'uniform_error_per_source_l1':max(a['stats']['uniform_error_per_source_l1'] for a in atlases),
                  'peak_field_or_intermediate_nodes':max(a['stats']['peak_field_or_intermediate_nodes'] for a in atlases),
                  'original_box_covered_with_exact_rationals':True,
                  'total_planning_seconds_including_refusals':sum(a['stats']['total_planning_seconds_including_refusals'] for a in atlases),
                  'total_planning_seconds_charging_cached_work':sum(a['stats']['total_planning_seconds_charging_cached_work'] for a in atlases),
                  'parallel_wall_seconds_this_invocation':perf_counter()-begin,
                  'grid_allocated':False,'floating_point_evaluation_certified':False,
                  'archive_cost_scaling_proved':False,'full_objective_complete':False})
    artifact={'source_sha256':source,'arb_precision_bits':100,'full_objective_complete':False,
              'construction_options':{'initial_tiles':patches,'max_step':.125,'workers':args.workers},
              'hierarchy':{'case':'stress','original_times':geom.times.tolist(),
                           'atlas':{'original_center':c.tolist(),'original_widths':w.tolist(),
                                    'patch_domains':domains,'stats':stats,'plans':plans,
                                    'refused_attempts':[r for a in atlases for r in a['refused_attempts']]}}}
    (args.directory/'results.json').write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'saved':str(args.directory),'stats':stats}),flush=True)

if __name__=='__main__':
    main()
