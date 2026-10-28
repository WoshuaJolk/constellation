"""Fixed local-catalogue query workload; not an orbit recovery experiment."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from magictools.itf.catalogue import Catalogue
from magictools.itf.night_query import query_nights
from magictools.itf.night_query_indexed import build_axis_orders,query_indexed_nights
from magictools.itf.kernels import ARCSEC


def main():
    p=argparse.ArgumentParser();p.add_argument('--catalogue',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    design=dict(seed=20261028,selection='Every finite local tracklet with >=2 observations and finite rate. Query 4096 evenly spaced original eligible catalogue ordinals, plus 4096 seed-fixed isotropic directions at those nights. Query both at 1 arcsec, 1 arcmin, 0.1, 1 and 10 degrees. No outcome selection.',scope='Exact nightly spatial query stage only. Uses motion-to-night-center approximation of existing NightIndex; no new predictions, orbit fits, uncertainty model, classification or catalogue linking.',baseline='Existing compiled x-slab query_nights',alternative='Smallest of three coordinate slabs, identical full 3D distance check and output ordering')
    (a.output/'design.json').write_text(json.dumps(design,indent=2))
    cat=Catalogue(a.catalogue);tr=cat.tracks
    selected=np.flatnonzero((tr['count']>=2)&np.isfinite(tr['u']).all(axis=1)&np.isfinite(tr['v']).all(axis=1)&np.isfinite(tr['t']))
    days=np.floor(tr['t'][selected]).astype(int);order=np.argsort(days,kind='stable');selected=selected[order];days=days[order]
    _,starts=np.unique(days,return_index=True);ends=np.r_[starts[1:],len(days)]
    units=tr['u'][selected]+tr['v'][selected]*(days+.5-tr['t'][selected])[:,None];units/=np.linalg.norm(units,axis=1)[:,None]
    original_ids=selected.copy()
    for lo,hi in zip(starts,ends):
        order=np.argsort(units[lo:hi,0],kind='stable');units[lo:hi]=units[lo:hi][order];original_ids[lo:hi]=original_ids[lo:hi][order]
    indices=np.argsort(original_ids)[np.linspace(0,len(original_ids)-1,4096,dtype=int)]
    nights=np.searchsorted(ends,indices,side='right');pred=units[indices].copy()
    rng=np.random.default_rng(design['seed']);random=rng.normal(size=pred.shape);random/=np.linalg.norm(random,axis=1)[:,None]
    nights=np.tile(nights,2);pred=np.concatenate([pred,random])
    t=time.perf_counter();orders=build_axis_orders(units,starts,ends);build=time.perf_counter()-t
    # Compile both before timing; report compilation separately, never count
    # reference JIT while presenting warmed alternative timing.
    t=time.perf_counter();query_nights(units,starts,ends,nights[:1],pred[:1],np.ones(1)*1e-4);query_indexed_nights(units,orders,starts,ends,nights[:1],pred[:1],np.ones(1)*1e-4);warm=time.perf_counter()-t
    report=dict(design=design,tracklets=len(units),nights=len(starts),additional_index_bytes=orders.nbytes,build_seconds=build,warmup_seconds=warm,runs=[])
    for angle in [1/ARCSEC,60/ARCSEC,np.deg2rad(.1),np.deg2rad(1.),np.deg2rad(10.)]:
        radii=np.full(len(pred),2*np.sin(angle/2));times={'baseline':[],'indexed':[]}
        for rep in range(3):
            outputs={}
            for name in (('baseline','indexed') if rep%2==0 else ('indexed','baseline')):
                t=time.perf_counter()
                result=query_nights(units,starts,ends,nights,pred,radii) if name=='baseline' else query_indexed_nights(units,orders,starts,ends,nights,pred,radii)
                times[name].append(time.perf_counter()-t);outputs[name]=result
            for x,y in zip(outputs['baseline'],outputs['indexed']):np.testing.assert_array_equal(x,y)
        report['runs'].append(dict(radius_arcsec=angle*ARCSEC,queries=len(pred),hits=len(outputs['baseline'][0]),seconds=times,speedup=float(np.median(times['baseline'])/np.median(times['indexed'])),exact_parity=True))
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
