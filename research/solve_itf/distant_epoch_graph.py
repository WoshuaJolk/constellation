"""Cached common-epoch two-body proposals; no final-fit or physics cuts.

Each reference has a +/-365 day window; references are 180 days apart.
Every pair separated by <=365 days can therefore share a reference window.
Range/radial hypotheses are approximate attribution states, not fitted orbits.
"""
import json,time,multiprocessing as mp
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from numba import njit
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial import cKDTree
from magictools.itf.kernels import position,MU,ARCSEC
from magictools.itf.broad import approximate_observers,unique_edges,best_edges
from magictools.itf.catalogue import sha256
from distant_search import distant_eligible,write

DISTANCES=(10.,20.,30.,35.,40.,45.,50.,60.,80.,100.,150.,250.,400.,700.,1000.,2000.)
RADIAL_FRACTIONS=(-.5,0.,.5)

@njit(cache=True)
def projected(t,u,du,observers,velocities,ref,earth_ref,distance,radial):
    result=np.full((len(t),3),np.nan)
    for i in range(len(t)):
        dt=ref-t[i];rho=distance-radial*dt
        if rho<=.001:continue
        state=np.empty(6);state[:3]=observers[i]+rho*u[i];state[3:]=velocities[i]+rho*du[i]+radial*u[i]
        r=position(state,dt)-earth_ref;norm=np.sqrt(np.dot(r,r))
        if np.isfinite(norm) and norm>.001:result[i]=r/norm
    return result

def partition(arg):
    directory,part,parts,seconds=arg;directory=Path(directory)
    tracks,pos,vel,days,earth,ev=[np.load(directory/(n+'.npy'),mmap_mode='r') for n in ('tracks','positions','velocities','days','earth','earth_velocity')]
    ep=CubicHermiteSpline(days,earth,ev);eligible=np.flatnonzero(distant_eligible(tracks)&np.isfinite(pos).all(axis=1));times=tracks['t'];clock=time.monotonic();deadline=clock+seconds
    anchors=np.arange(np.floor(times[eligible].min()/180)*180,np.ceil(times[eligible].max()/180)*180+1,180)
    counts=Counter();edges=np.empty((0,3));pending=[];n=0;complete=True;cap=250000
    for ordinal,ref in enumerate(anchors):
        if ordinal%parts!=part:continue
        ids=eligible[np.abs(times[eligible]-ref)<=365];night=np.floor(times[ids]).astype(int);order=np.argsort(night,kind='stable');ids,night=ids[order],night[order]
        nights,starts=np.unique(night,return_index=True);groups=[(int(d),np.arange(a,b)) for d,a,b in zip(nights,starts,np.r_[starts[1:],len(ids)])]
        if len(groups)<2:continue
        tr=tracks[ids];noise=.3*np.sqrt(12/tr['count'])/tr['span']
        for distance in DISTANCES:
            for frac in RADIAL_FRACTIONS:
                if time.monotonic()>deadline:complete=False;break
                radial=frac*np.sqrt(2*MU/distance)
                p=projected(tr['t'],tr['u'],tr['v'],pos[ids],vel[ids],ref,ep(ref),distance,radial);good=np.isfinite(p).all(axis=1)
                counts['propagated_tracklets']+=len(ids);counts['invalid_projections']+=int((~good).sum())
                trees={d:(ix[good[ix]],cKDTree(p[ix[good[ix]]])) for d,ix in groups if good[ix].any()}
                for ai,(day,ix) in enumerate(groups):
                    if time.monotonic()>deadline:complete=False;break
                    a=ix[good[ix]]
                    if not len(a):continue
                    for other,_ in groups[ai+1:]:
                        gap=other-day
                        if gap>365:break
                        if other not in trees:continue
                        b,tree=trees[other];k=min(17,len(b))
                        ds,loc=tree.query(p[a],k=list(range(1,k+1)),distance_upper_bound=3600/ARCSEC,workers=1)
                        counts['queries']+=len(a);counts['day_hypothesis_pairs']+=1
                        if k>16:counts['queries_at_cap']+=int(np.isfinite(ds[:,-1]).sum());ds,loc=ds[:,:-1],loc[:,:-1]
                        ii,jj=np.where(np.isfinite(ds))
                        if not len(ii):continue
                        aa,bb=a[ii],b[loc[ii,jj]];dt=tr['t'][bb]-tr['t'][aa]
                        scale=60+.5*(noise[aa]+noise[bb])*dt+.002*dt*dt
                        mismatch=ds[ii,jj]*ARCSEC
                        keep=(dt>=.3)&(dt<=365)&(tr['id'][aa]!=tr['id'][bb])&(mismatch<=np.minimum(3600,scale))
                        score=.5*(mismatch[keep]/scale[keep])**2+2*np.log(scale[keep]/60)+np.log(len(DISTANCES)*len(RADIAL_FRACTIONS))
                        pending.append(np.column_stack((ids[aa[keep]],ids[bb[keep]],score)));n+=int(keep.sum());counts['proposed_edges']+=int(keep.sum())
                        if n>cap:
                            merged=unique_edges(np.concatenate([edges]+pending));counts['edge_cap_discarded']+=max(0,len(merged)-cap);edges=best_edges(merged,cap);pending=[];n=0
                    if not complete:break
                counts['anchor_hypotheses_completed']+=int(complete)
                if not complete:break
            if not complete:break
        counts['anchors_completed']+=int(complete)
        if not complete:break
    merged=unique_edges(np.concatenate([edges]+pending));counts['edge_cap_discarded']+=max(0,len(merged)-cap);edges=best_edges(merged,cap);np.save(directory/f'part-{part}.npy',edges)
    receipt=dict(part=part,complete=complete,counts=dict(counts),edges=len(edges),seconds=time.monotonic()-clock,sha256=sha256(directory/f'part-{part}.npy'))
    write(directory/f'part-{part}.json',receipt);return receipt

def graph(cat,sites,out,workers,seconds,**ignored):
    out.mkdir(exist_ok=True);start=time.monotonic();geom=approximate_observers(cat.tracks,sites)
    for name,a in zip(('tracks','positions','velocities','days','earth','earth_velocity'),(cat.tracks,*geom)):np.save(out/(name+'.npy'),a)
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:receipts=list(pool.map(partition,[(str(out),i,workers,seconds) for i in range(workers)]))
    edges=unique_edges(np.concatenate([np.load(out/f'part-{i}.npy') for i in range(workers)]));before=len(edges);edges=best_edges(edges,4000000);counts=Counter()
    for r in receipts:counts.update(r['counts'])
    counts.update(tracklets_total=len(cat.tracks),distant_rate_eligible=int(distant_eligible(cat.tracks).sum()),merge_cap_discarded=before-len(edges),edges_retained=len(edges))
    np.save(out/'edges.npy',edges);r=dict(complete=all(x['complete'] for x in receipts),counts=dict(counts),partitions=receipts,seconds=time.monotonic()-start,distances=DISTANCES,radial_fractions=RADIAL_FRACTIONS,max_gap_days=365,sha256=sha256(out/'edges.npy'))
    write(out/'summary.json',r)
    for name in ('tracks','positions','velocities','days','earth','earth_velocity'):(out/(name+'.npy')).unlink()
    return edges,r
