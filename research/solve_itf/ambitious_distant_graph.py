"""Bounded, coverage-accounted GPU projection/shared-index distant proposals."""
import argparse,hashlib,json,multiprocessing as mp,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from magictools.itf.catalogue import Catalogue,sha256
from magictools.itf.fit import read_sites
from magictools.itf.broad import approximate_observers,unique_edges,best_edges
from magictools.itf.kernels import ARCSEC,MU
from distant_search import distant_eligible,write
from distant_epoch_graph import projected
from gpu_distant_projection import ProjectionBank
from benchmark_sparse_epoch_join import shared

DISTANCES=(10.,20.,30.,35.,40.,45.,50.,60.,80.,100.,120.,150.,200.,250.,300.,400.,500.,700.,1000.,1500.,2000.)
LONG_DISTANCES=(40.,60.,100.,150.,250.,400.,700.,1000.,2000.)
RADIAL=(-.9,-.5,0.,.5,.9)
LANES=('near','kuiper','distant','remote','long_outer','long_remote')
REFERENCES=dict(near=20.,kuiper=50.,distant=150.,remote=700.,long_outer=150.,long_remote=700.)
CTX=None


def lane_for(distance,long):
    if long:return 'long_outer' if distance<300 else 'long_remote'
    return 'near' if distance<30 else 'kuiper' if distance<80 else 'distant' if distance<300 else 'remote'


def cap_edges(edges,cap,mag,band,distance):
    """Half score, half large-H-proxy priority, retaining same-band photometry."""
    edges=unique_edges(edges)
    if len(edges)<=cap:return edges
    ordinary=best_edges(edges,cap//2);a,b=edges[:,:2].astype('i8').T
    good=np.isfinite(mag[a])&np.isfinite(mag[b])&(band[a]==band[b])&(band[a]!=b'')
    h=.5*(mag[a]+mag[b])-10*np.log10(distance);good &= h<=5
    # Rank within each measured band, then cycle bands; no band conversion.
    picks=[];groups=[]
    for bb in np.unique(band[a[good]]):
        ix=np.flatnonzero(good&(band[a]==bb));ix=ix[np.lexsort((b[ix],a[ix],edges[ix,2],h[ix]))];groups.append(ix[:cap//2])
    for i in range(max(map(len,groups),default=0)):
        for g in groups:
            if i<len(g):picks.append(int(g[i]))
            if len(picks)>=cap//2:break
        if len(picks)>=cap//2:break
    priority=edges[np.array(picks,dtype='i8')]
    chosen=unique_edges(np.concatenate([ordinary,priority]));needed=cap-len(chosen)
    if needed:
        # Fill from score ordering, then preserve priority against another global cut.
        existing={(int(a),int(b)) for a,b,_ in chosen}
        fill=[]
        for row in best_edges(edges,min(len(edges),cap+len(chosen))):
            if (int(row[0]),int(row[1])) not in existing:fill.append(row)
            if len(fill)>=needed:break
        if fill:chosen=np.concatenate([chosen,np.array(fill)])
    return chosen


def plan_frames(times):
    plans=[]
    for gap,step,distances in [(365.,365.,DISTANCES),(1095.,730.,LONG_DISTANCES)]:
        epochs=np.arange(np.floor(times.min()/step)*step,np.ceil(times.max()/step)*step+1,step)
        epochs=sorted(epochs,key=lambda x:hashlib.sha256(('ambitious-v1:'+float(x).hex()).encode()).digest())
        # Coarse-to-fine distance ordering spreads early prefixes over the grid.
        bits=max(1,(len(distances)-1).bit_length());order=sorted(range(len(distances)),key=lambda i:int(format(i,f'0{bits}b')[::-1],2))
        hypotheses=[(distances[i],f) for f in RADIAL for i in order]
        frames=[]
        for round_index,start in enumerate(range(0,len(hypotheses),12)):
            for ordinal,epoch in enumerate(epochs):
                h=[hypotheses[(j+ordinal)%len(hypotheses)] for j in range(start,min(start+12,len(hypotheses))) ]
                frames.append(dict(gap=gap,epoch=float(epoch),hypotheses=h,round=round_index,epoch_ordinal=ordinal))
        plans.append(frames)
    # Interleave year and multi-year work instead of deferring the latter.
    out=[]
    for i in range(max(map(len,plans))):
        for lane in plans:
            if i<len(lane):out.append(lane[i])
    for i,r in enumerate(out):r['frame']=i
    return out


def init(root):
    global CTX
    root=Path(root);CTX=(root,np.load(root/'tracks.npy',mmap_mode='r'),np.load(root/'magnitude.npy',mmap_mode='r'),np.load(root/'band.npy',mmap_mode='r'))


def ready(_):return True


def one(task):
    h,frame=task;root,tracks,mag,band=CTX;ids=np.load(root/'indices.npy',mmap_mode='r');p=np.load(root/'projection.npy',mmap_mode='r')[h]
    good=np.isfinite(p).all(axis=1);local=tracks[ids[good]];distance,fraction=frame['hypotheses'][h];started=time.perf_counter()
    lane=lane_for(distance,frame['gap']>365)
    try:
        edges,counts=shared(p[good],local['t'],.3*np.sqrt(12/local['count'])/local['span'],local['id'],max_raw_neighbors=32000000,max_gap_days=frame['gap'])
        if frame['gap']>365 and len(edges):
            a,b=edges[:,:2].astype('i8').T;edges=edges[local['t'][b]-local['t'][a]>365]
        if len(edges):edges[:,:2]=ids[good][edges[:,:2].astype('i8')]
        before=len(edges);edges=cap_edges(edges,100000,mag,band,distance)
        return edges,dict(frame=frame['frame'],hypothesis=h,lane=lane,distance=distance,radial_fraction=fraction,
            status='completed',invalid_projections=int((~good).sum()),proposed=before,retained=len(edges),cap_discarded=before-len(edges),counts=counts,seconds=time.perf_counter()-started)
    except MemoryError as exc:
        return np.empty((0,3)),dict(frame=frame['frame'],hypothesis=h,lane=lane,distance=distance,radial_fraction=fraction,status='unresolved_memory_cap',error=str(exc),seconds=time.perf_counter()-started)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seconds',type=float,default=1800);a=ap.parse_args();a.output.mkdir(exist_ok=True)
    started=time.monotonic();deadline=started+a.seconds;cat=Catalogue('catalogue');sites=read_sites('ObsCodes.html');tracks=cat.tracks
    mag=np.load('tracklet-band-magnitude.npy');band=np.load('tracklet-band.npy');assert len(mag)==len(band)==len(tracks)
    pos,vel,days,earth,ev=approximate_observers(tracks,sites);ep=CubicHermiteSpline(days,earth,ev)
    eligible=np.flatnonzero(distant_eligible(tracks)&np.isfinite(pos).all(axis=1)&np.isfinite(vel).all(axis=1));plans=plan_frames(tracks['t'][eligible]);write(a.output/'plan.json',plans)
    work=a.output/'working';work.mkdir(exist_ok=True)
    for name,array in [('tracks',tracks),('magnitude',mag),('band',band)]:np.save(work/(name+'.npy'),array)
    buckets={k:np.empty((0,3)) for k in LANES};counts=Counter();completed=0
    def save():
        for lane,edges in buckets.items():
            tmp=a.output/(lane+'.tmp.npy');np.save(tmp,edges);tmp.replace(a.output/(lane+'.npy'))
        write(a.output/'summary.json',dict(status='running',frames_completed=completed,frames_planned=len(plans),
            hypotheses_terminal=int(counts['terminal']),hypotheses_planned=sum(len(x['hypotheses']) for x in plans),counts=dict(counts),
            lanes={k:len(v) for k,v in buckets.items()},seconds=time.monotonic()-started,eligible_tracklets=len(eligible),
            catalogue_tracklets=len(tracks),complete=completed==len(plans) and counts['unresolved']==0))
    # Fork workers before creating any CUDA context; no context inheritance.
    with ProcessPoolExecutor(max_workers=20,mp_context=mp.get_context('fork'),initializer=init,initargs=(str(work),)) as pool,(a.output/'tasks.jsonl').open('w') as log:
        list(pool.map(ready,range(20)))
        for frame in plans:
            if time.monotonic()>=deadline:break
            ids=eligible[np.abs(tracks['t'][eligible]-frame['epoch'])<=frame['gap']]
            if len(ids)<3:
                counts['empty_frames']+=1;counts['terminal']+=len(frame['hypotheses']);completed+=1;continue
            tr=tracks[ids];h=np.array(frame['hypotheses']);bank=ProjectionBank(tr['t'].copy(),tr['u'].copy(),tr['v'].copy(),pos[ids],vel[ids])
            p,profile=bank.project(frame['epoch'],ep(frame['epoch']),h,max_days=frame['gap'],max_output_bytes=1024**3)
            sample=np.linspace(0,len(ids)-1,min(64,len(ids)),dtype=int);fallback=False
            for j,(d,f) in enumerate(h):
                ref=projected(tr['t'][sample],tr['u'][sample],tr['v'][sample],pos[ids[sample]],vel[ids[sample]],frame['epoch'],ep(frame['epoch']),d,f*np.sqrt(2*MU/d))
                good=np.isfinite(ref).all(axis=1);actual=p[j,sample]
                if not np.array_equal(np.isfinite(ref),np.isfinite(actual)) or np.max(np.linalg.norm(np.cross(ref[good],actual[good]),axis=1)*ARCSEC,initial=0)>1e-5:fallback=True
            if fallback:
                counts['cpu_projection_fallback_frames']+=1
                p=np.array([projected(tr['t'],tr['u'],tr['v'],pos[ids],vel[ids],frame['epoch'],ep(frame['epoch']),d,f*np.sqrt(2*MU/d)) for d,f in h])
            np.save(work/'indices.npy',ids);np.save(work/'projection.npy',p)
            pending={k:[] for k in LANES}
            for edges,receipt in pool.map(one,[(j,frame) for j in range(len(h))]):
                log.write(json.dumps(receipt)+'\n');log.flush();counts['terminal']+=1
                if receipt['status']!='completed':counts['unresolved']+=1
                else:counts['proposed']+=receipt['proposed'];counts['local_cap_discarded']+=receipt['cap_discarded']
                if len(edges):pending[receipt['lane']].append(edges)
            for lane,parts in pending.items():
                if parts:
                    merged=unique_edges(np.concatenate([buckets[lane]]+parts));before=len(merged)
                    buckets[lane]=cap_edges(merged,1200000,mag,band,REFERENCES[lane]);counts['lane_cap_discarded']+=before-len(buckets[lane])
            counts['propagated']+=len(ids)*len(h);completed+=1;save()
            del bank,p
        save()
    result=json.loads((a.output/'summary.json').read_text());result['status']='complete_bounded_grid' if result['complete'] else 'partial_grid';result['lane_hashes']={k:sha256(a.output/(k+'.npy')) for k in LANES};write(a.output/'summary.json',result)
    import shutil
    shutil.rmtree(work)


if __name__=='__main__':main()
