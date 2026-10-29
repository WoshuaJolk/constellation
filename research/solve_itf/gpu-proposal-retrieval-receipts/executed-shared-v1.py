"""Compare per-night nearest queries with one shared spherical index.

Same finite proposal radius, per-night top16, rates, and scoring. No search
outcomes drive input selection; capped raw neighborhoods stay unresolved.
"""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from magictools.itf.kernels import ARCSEC, MU
from magictools.itf.catalogue import sha256
from distant_epoch_graph import projected


def finish(a,b,distance,times,noise,identities):
    dt=times[b]-times[a];scale=60+.5*(noise[a]+noise[b])*dt+.002*dt*dt
    mismatch=distance*ARCSEC
    good=(dt>=.3)&(dt<=365)&(identities[a]!=identities[b])&(mismatch<=np.minimum(3600,scale))
    score=.5*(mismatch[good]/scale[good])**2+2*np.log(scale[good]/60)+np.log(48)
    out=np.column_stack((a[good],b[good],score))
    return out[np.lexsort((out[:,1],out[:,0]))] if len(out) else out


def nightly(p,t,noise,identities,radius=3600/ARCSEC):
    night=np.floor(t).astype('i8');days=np.unique(night);groups=[np.flatnonzero(night==d) for d in days]
    trees=[cKDTree(p[g]) for g in groups];aa=[];bb=[];dd=[];queries=0;calls=0
    for i,(day,a) in enumerate(zip(days,groups)):
        for j in range(i+1,len(days)):
            if days[j]-day>365:break
            b=groups[j];k=min(16,len(b));d,loc=trees[j].query(p[a],k=list(range(1,k+1)),distance_upper_bound=radius,workers=1)
            qi,ki=np.where(np.isfinite(d));aa.append(a[qi]);bb.append(b[loc[qi,ki]]);dd.append(d[qi,ki]);queries+=len(a);calls+=1
    if not aa:return np.empty((0,3)),dict(queries=queries,calls=calls)
    return finish(np.concatenate(aa),np.concatenate(bb),np.concatenate(dd),t,noise,identities),dict(queries=queries,calls=calls)


def shared(p,t,noise,identities,radius=3600/ARCSEC,max_raw_neighbors=8000000):
    tree=cKDTree(p);counts=tree.query_ball_point(p,radius,return_length=True,workers=1)
    raw=int(counts.sum())
    if raw>max_raw_neighbors:raise MemoryError(f'{raw} raw neighbors exceed {max_raw_neighbors}; no partial graph returned')
    neighborhoods=tree.query_ball_point(p,radius,return_sorted=False,workers=1)
    a=np.repeat(np.arange(len(p)),counts);b=np.concatenate(neighborhoods).astype('i8');night=np.floor(t).astype('i8')
    good=(night[b]>night[a])&(night[b]-night[a]<=365);a,b=a[good],b[good]
    delta=p[a]-p[b];squared=np.einsum('ij,ij->i',delta,delta)
    # Deterministic tie order; exact boundary ties can differ from cKDTree.query.
    order=np.lexsort((b,squared,night[b],a));a,b,squared=a[order],b[order],squared[order]
    first=np.r_[True,(a[1:]!=a[:-1])|(night[b[1:]]!=night[b[:-1]])] if len(a) else np.empty(0,bool)
    begin=np.maximum.accumulate(np.where(first,np.arange(len(a)),0))
    keep=np.arange(len(a))-begin<16;a,b,squared=a[keep],b[keep],squared[keep]
    return finish(a,b,np.sqrt(squared),t,noise,identities),dict(queries=2*len(p),calls=2,raw_neighbors=raw)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(exist_ok=False);design=dict(projection_sha256=sha256(a.inputs/'projection.npz'),metadata_sha256=sha256(a.inputs/'projection-metadata.npz'),
        sizes=[4096,16384],hypothesis_indices=[6,24,45],repeats=2,raw_neighbor_cap=8000000,
        scope='Proposal-stage implementation comparison only; identical frozen projection inputs. No production fitting. Shared top16 ties use original point index; any differing edges fail parity.')
    (a.output/'design.json').write_text(json.dumps(design,indent=2));data=np.load(a.inputs/'projection.npz');meta=np.load(a.inputs/'projection-metadata.npz');report=dict(design=design,runs=[])
    def save():(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    for size in design['sizes']:
        ix=np.linspace(0,len(data['t'])-1,min(size,len(data['t'])),dtype=int)
        t,u,du,obs,vel=[data[k][ix] for k in ('t','u','du','observers','velocities')]
        noise=.3*np.sqrt(12/meta['count'][ix])/meta['span'][ix];identity=meta['identity'][ix]
        for hi in design['hypothesis_indices']:
            d,f=data['hypotheses'][hi];p=projected(t,u,du,obs,vel,float(data['ref']),data['earth'],d,f*np.sqrt(2*MU/d))
            valid=np.isfinite(p).all(axis=1);args=(p[valid],t[valid],noise[valid],identity[valid])
            row=dict(size=size,hypothesis_index=hi,distance=float(d),radial_fraction=float(f),invalid=int((~valid).sum()),repeats=[]);report['runs'].append(row)
            for repeat in range(design['repeats']):
                result={};timings={};counts={}
                try:
                    for name in (('nightly','shared') if repeat==0 else ('shared','nightly')):
                        stamp=time.perf_counter();result[name],counts[name]={'nightly':nightly,'shared':shared}[name](*args);timings[name]=time.perf_counter()-stamp
                    parity=bool(np.array_equal(result['nightly'][:,:2],result['shared'][:,:2]) and np.allclose(result['nightly'][:,2],result['shared'][:,2],rtol=0,atol=1e-12))
                    row['repeats'].append(dict(seconds=timings,counts=counts,edges={k:len(v) for k,v in result.items()},parity=parity));save()
                    if not parity:
                        np.savez_compressed(a.output/f'mismatch-{size}-{hi}-{repeat}.npz',**result)
                        row['status']='parity_failed';save();break
                except MemoryError as exc:
                    row['status']='unresolved_memory_cap';row['error']=str(exc);save();break
            else:row['status']='completed'
            save()
    report['status']='completed';save()


if __name__=='__main__':main()
