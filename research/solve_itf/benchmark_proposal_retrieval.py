"""Bounded real-data stage benchmarks and fixed opened-control retrieval replay."""
import argparse, json, time, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
from numba import njit, prange, set_num_threads, cuda
from magictools.itf.kernels import MU, ARCSEC, directions
from magictools.itf.night_query import query_nights
from magictools.itf.fit import Geometry, read_sites
from magictools.itf.catalogue import sha256
from distant_epoch_graph import projected
from gpu_distant_projection import ProjectionBank
from gpu_night_query import NightQueryBank


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


@njit(parallel=True)
def cpu_projection_parallel(t,u,du,obs,vel,ref,earth,hypotheses):
    result=np.empty((len(hypotheses),len(t),3))
    for h in prange(len(hypotheses)):
        d,f=hypotheses[h];result[h]=projected(t,u,du,obs,vel,ref,earth,d,f*np.sqrt(2*MU/d))
        for i in range(len(t)):
            if abs(ref-t[i])>365:result[h,i,:]=np.nan
    return result


@njit(nogil=True)
def cpu_query_block(points,starts,ends,nights,predictions,radii):
    return query_nights(points,starts,ends,nights,predictions,radii)


def cpu_query_parallel(points,starts,ends,nights,predictions,radii):
    splits=np.linspace(0,len(nights),21,dtype=int)
    def run(pair):
        lo,hi=pair;r=cpu_query_block(points,starts,ends,nights[lo:hi],predictions[lo:hi],radii[lo:hi])
        return r[0],r[1]+lo,r[2]
    with ThreadPoolExecutor(max_workers=20) as pool:parts=list(pool.map(run,zip(splits[:-1],splits[1:])))
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sites',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    if cuda.config.ENABLE_CUDASIM:raise RuntimeError('Real GPU required')
    for name,rec in json.loads((a.inputs/'manifest.json').read_text()).items():
        if sha256(a.inputs/name)!=rec['sha256']:raise ValueError('Frozen input changed: '+name)
    write(a.output/'design.json',json.loads((a.inputs/'design.json').read_text()))
    set_num_threads(20);started=time.perf_counter();report=dict(projection=[],retrieval=[],science=None)
    def save():write(a.output/'report.json',report)
    try:
        data=np.load(a.inputs/'projection.npz');h=data['hypotheses'];ref=float(data['ref']);earth=data['earth']
        allarrays=[data[k] for k in ('t','u','du','observers','velocities')]
        for n in sorted(set(min(len(allarrays[0]),x) for x in (1024,16384,65536,len(allarrays[0])))):
            arrays=[np.ascontiguousarray(x[:n]) for x in allarrays];bank=ProjectionBank(*arrays)
            def cpu1():
                result=np.asarray([projected(*arrays,ref,earth,d,f*np.sqrt(2*MU/d)) for d,f in h])
                result[:,np.abs(ref-arrays[0])>365]=np.nan
                return result
            def cpu20():return cpu_projection_parallel(*arrays,ref,earth,h)
            def gpu():return bank.project(ref,earth,h)[0]
            stamp=time.perf_counter();cpu1();cpu20();gpu();warm=time.perf_counter()-stamp
            case=dict(tracklets=n,hypotheses=len(h),cold_warmup_seconds=warm,resident_upload_seconds=bank.upload_seconds,repeats=[])
            report['projection'].append(case)
            for rep in range(3):
                results={};times={}
                for name in (('cpu1','cpu20','gpu') if rep%2==0 else ('gpu','cpu20','cpu1')):
                    stamp=time.perf_counter();results[name]={'cpu1':cpu1,'cpu20':cpu20,'gpu':gpu}[name]();times[name]=time.perf_counter()-stamp
                reference=results['cpu1'];good=np.isfinite(reference).all(axis=2);checks={}
                for name in ('cpu20','gpu'):
                    actual=results[name];error=np.linalg.norm(np.cross(reference[good],actual[good]),axis=1)*ARCSEC
                    checks[name]=dict(finite_masks_equal=bool(np.array_equal(np.isfinite(reference),np.isfinite(actual))),
                        maximum_arcsec=float(np.max(error,initial=0)),vectors_equal=bool(np.allclose(reference,actual,atol=1e-10,rtol=0,equal_nan=True)))
                case['repeats'].append(dict(seconds=times,checks=checks,valid=int(good.sum())));save()
                if not all(c['finite_masks_equal'] and c['vectors_equal'] and c['maximum_arcsec']<1e-5 for c in checks.values()):
                    raise RuntimeError('Projection parity failed; no scientific exclusions permitted')
            del results,bank
        data=np.load(a.inputs/'retrieval.npz');points,starts,ends=[data[k] for k in ('points','starts','ends')]
        bank=NightQueryBank(points,starts,ends);report['retrieval_resident_upload_seconds']=bank.upload_seconds
        for n in sorted(set(min(len(data['nights']),x) for x in (8192,65536,262144))):
            nights,predictions,radii=[data[k][:n] for k in ('nights','predictions','radii')]
            def cpu1():return query_nights(points,starts,ends,nights,predictions,radii)
            def cpu20():return cpu_query_parallel(points,starts,ends,nights,predictions,radii)
            def gpu():return bank.query(nights,predictions,radii)[0]
            stamp=time.perf_counter();cpu1();cpu20();gpu();warm=time.perf_counter()-stamp
            case=dict(queries=n,points=len(points),cold_warmup_seconds=warm,repeats=[]);report['retrieval'].append(case)
            for rep in range(3):
                results={};times={}
                for name in (('cpu1','cpu20','gpu') if rep%2==0 else ('gpu','cpu20','cpu1')):
                    stamp=time.perf_counter();results[name]={'cpu1':cpu1,'cpu20':cpu20,'gpu':gpu}[name]();times[name]=time.perf_counter()-stamp
                checks={}
                for name in ('cpu20','gpu'):
                    checks[name]=bool(np.array_equal(results[name][0],results['cpu1'][0]) and np.array_equal(results[name][1],results['cpu1'][1]) and np.allclose(results[name][2],results['cpu1'][2],atol=1e-15,rtol=0))
                case['repeats'].append(dict(seconds=times,exact_hit_and_order_parity=checks,hits=len(results['cpu1'][0])));save()
                if not all(checks.values()):raise RuntimeError('Retrieval parity failed')
        # Same frozen later windows/modes, no identity passed to retrieval or refinement.
        queries=json.loads((a.inputs/'science-queries.json').read_text());valid=[q for q in queries if q['finite']]
        nights=np.array([q['night'] for q in valid],dtype='i8');pred=np.array([q['prediction'] for q in valid],dtype='f8').reshape(-1,3)
        radii=np.full(len(valid),2*np.sin(120/ARCSEC/2))
        stamp=time.perf_counter();cpu=query_nights(points,starts,ends,nights,pred,radii);cpu_seconds=time.perf_counter()-stamp
        stamp=time.perf_counter();gpu,receipt=bank.query(nights,pred,radii);gpu_seconds=time.perf_counter()-stamp
        for x,y in zip(cpu[:2],gpu[:2]):np.testing.assert_array_equal(x,y)
        np.testing.assert_allclose(cpu[2],gpu[2],atol=1e-15,rtol=0)
        np.savez_compressed(a.output/'science-retrieved.npz',point_indices=gpu[0],query_indices=gpu[1],distances=gpu[2])
        # The retrieved set is persisted before labels are read below.
        observed=np.load(a.inputs/'science-observations.npz');indices=observed['indices'];lookup={int(x):i for i,x in enumerate(indices)}
        geometry=Geometry(SimpleNamespace(tracks=observed['tracks'],obs=observed['observations']),read_sites(a.sites))
        tracklet_ids=data['tracklet_ids'];hits=[int(tracklet_ids[i]) for i in gpu[0]]
        geometry.prepare([lookup[i] for i in hits]);out=[]
        for qi,q in enumerate(valid):
            selected=[int(tracklet_ids[i]) for i in gpu[0][gpu[1]==qi]];details=[]
            for i in selected:
                _,tt,u,oo=geometry.prepared[lookup[i]];pr=directions(np.array(q['state']),tt-q['epoch'],oo)
                residual=np.arctan2(np.linalg.norm(np.cross(pr,u),axis=1),np.sum(pr*u,axis=1))*ARCSEC
                good=bool(np.isfinite(residual).all());details.append(dict(tracklet=i,observations=len(tt),
                    residuals_arcsec=[float(x) if np.isfinite(x) else None for x in residual],
                    rms_arcsec=float(np.sqrt(np.mean(residual**2))) if good else None,
                    within_fixed_30arcsec=bool(good and np.all(residual<=30)),status='predicted' if good else 'numerical_failure'))
            out.append(dict(case=q['case'],mode=q['mode'],gap_days=q['gap_days'],state_sha256=q['state_sha256'],
                retrieved_tracklets=len(selected),details=details))
        write(a.output/'science-refined.json',out)
        grading=json.loads((a.inputs/'grading.json').read_text());grades=[]
        for g in grading:
            row=dict(g);predictions=[x for x in out if x['case']==g['case']]
            row['modes']=[dict(mode=x['mode'],retrieved=x['retrieved_tracklets'],
                reserved_screened=any(d['tracklet']==g['reserved_tracklet'] for d in x['details']),
                reserved_refined=any(d['tracklet']==g['reserved_tracklet'] and d['within_fixed_30arcsec'] for d in x['details']),
                other_refined_hits=sum(d['tracklet']!=g['reserved_tracklet'] and d['within_fixed_30arcsec'] for d in x['details'])) for x in predictions]
            grades.append(row)
        report['science']=dict(cpu_query_seconds=cpu_seconds,gpu_query_seconds=gpu_seconds,receipt=receipt,cases=grades,
            finite_queries=len(valid),unresolved_predictions=len(queries)-len(valid),retrieval_sha256=sha256(a.output/'science-retrieved.npz'),
            scope='Previously opened controls; supplied later windows; full available ITF plus reserved controls. Fixed nominal screening, not calibrated uncertainty, uniqueness, or confirmation.')
        report['status']='completed';report['elapsed_seconds']=time.perf_counter()-started;save()
    except BaseException as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc),elapsed_seconds=time.perf_counter()-started);save()
        (a.output/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
