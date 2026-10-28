"""Equivalent distance-trial proposals for sparse control catalogues.

Compile the same hypotheses/projection/gates/nearest-neighbour budget, avoiding
hundreds of thousands of tiny NumPy operations and KD-tree constructions. Dense
nights retain the existing tree path. No orbital class or distance trial removed.
"""
import numpy as np
from numba.typed import List
from scipy.interpolate import CubicHermiteSpline
from .kernels import njit, MU, ARCSEC
from .fit import check_deadline
from .broad import _curved_scan as reference_scan, distance_hypotheses, unique_edges, best_edges


@njit(cache=True)
def projected(index, times, unit, rate, positions, velocities, ref, earth, distance, radial):
    dt=ref-times[index];rho=distance-radial*dt
    r=positions[index]+rho*unit[index]
    norm=np.sqrt(np.dot(r,r))
    if rho<=.001 or norm<=.01:return np.zeros(3),False
    v=velocities[index]+rho*rate[index]+radial*unit[index]
    p=r+dt*v-.5*dt*dt*MU*r/(norm**3)-earth
    norm=np.sqrt(np.dot(p,p))
    if norm==0 or not np.isfinite(norm):return np.zeros(3),False
    return p/norm,True


@njit(cache=True)
def day_scan(day, a, rights, earth, hypotheses, times, unit, rate, positions,
             velocities, counts, spans, names, position_gate, curvature, sigma, neighbors, max_gap):
    capacity=len(hypotheses)*len(rights)*len(a)*neighbors
    edges=np.empty((capacity,3));length=0
    counters=np.zeros(5,np.int64)  # queries, at-cap, raw proposals, day pairs, boundary ties
    for distance,radial,gap_limit in hypotheses:
        for gap in range(1,int(gap_limit)+1):
            b=rights[gap-1]
            if len(b)==0:continue
            ref=day+.5+gap/2
            left=np.empty((len(a),3));good_a=np.zeros(len(a),np.bool_)
            right=np.empty((len(b),3));good_b=np.zeros(len(b),np.bool_)
            for k,index in enumerate(a):left[k],good_a[k]=projected(index,times,unit,rate,positions,velocities,ref,earth[gap-1],distance,radial)
            for k,index in enumerate(b):right[k],good_b[k]=projected(index,times,unit,rate,positions,velocities,ref,earth[gap-1],distance,radial)
            if not np.any(good_a) or not np.any(good_b):continue
            gate=min(2.,(position_gate+10*curvature*gap*gap)/ARCSEC)
            counters[0]+=np.sum(good_a);counters[3]+=1
            for k,index in enumerate(a):
                if not good_a[k]:continue
                ds=np.full(len(b),np.inf)
                for j in range(len(b)):
                    if good_b[j]:
                        delta=right[j]-left[k];ds[j]=np.sqrt(np.dot(delta,delta))
                order=np.argsort(ds)
                if len(order)>neighbors and ds[order[neighbors]]<gate:
                    if abs(ds[order[neighbors]]-ds[order[neighbors-1]])<1e-14:
                        counters[4]+=1
                if len(order)>neighbors and ds[order[neighbors]]<gate:counters[1]+=1
                for j in order[:neighbors]:
                    if ds[j]>=gate:continue
                    other=b[j];dt=times[other]-times[index]
                    if dt<.3 or dt>max_gap or names[index]==names[other]:continue
                    noise=sigma*np.sqrt(12/counts[index])/spans[index]+sigma*np.sqrt(12/counts[other])/spans[other]
                    scale=position_gate+.5*noise*dt
                    score=.5*(ds[j]*ARCSEC/scale)**2+2*np.log(scale/position_gate)+np.log(len(hypotheses))
                    edges[length,0]=index;edges[length,1]=other;edges[length,2]=score;length+=1;counters[2]+=1
    return edges[:length],counters


def sparse_scan(tracks,positions,velocities,days,earth,earth_velocity,config,deadline,
                partition=(0,1),progress=lambda _:None,hypotheses=None,max_sparse_night=32):
    hypotheses=list(distance_hypotheses(config.max_gap_days)) if hypotheses is None else list(hypotheses)
    chosen=np.flatnonzero((tracks['count']>=2)&(tracks['span']>0)&(tracks['span']<=.6)&np.isfinite(positions).all(axis=1))
    night=np.floor(tracks['t'][chosen]).astype(int);order=np.argsort(night,kind='stable');chosen,night=chosen[order],night[order]
    nights,starts=np.unique(night,return_index=True)
    groups={int(d):chosen[a:b] for d,a,b in zip(nights,starts,np.r_[starts[1:],len(chosen)])}
    # Dense nights and ambiguous capped ties use the original tree ordering.
    if any(len(ids)>max_sparse_night for ids in groups.values()):
        return reference_scan(tracks,positions,velocities,days,earth,earth_velocity,config,deadline,partition,progress,hypotheses)
    ephemeris=CubicHermiteSpline(days,earth,earth_velocity)
    _,names=np.unique(tracks['id'],return_inverse=True)
    counters=np.zeros(5,np.int64);discarded=0;retained=0;parts=[];edges=np.empty((0,3))
    max_gap=max((int(h[2]) for h in hypotheses),default=0)
    if not max_gap:return edges,dict(curved_queries=0,curved_queries_at_cap=0,curved_edges_proposed=0,curved_edge_pool_discarded=0,curved_day_pairs_scanned=0)
    for ordinal,(day,a) in enumerate(groups.items()):
        if ordinal%partition[1]!=partition[0]:continue
        check_deadline(deadline)
        rights=List([groups.get(day+gap,np.empty(0,np.int64)) for gap in range(1,max_gap+1)])
        refs=day+.5+np.arange(1,max_gap+1)/2
        part,counts=day_scan(day,a,rights,ephemeris(refs),np.asarray(hypotheses,float),
            tracks['t'],tracks['u'],tracks['v'],positions,velocities,tracks['count'],tracks['span'],names,
            config.position_gate_arcsec,config.curvature_gate_arcsec_per_day2,config.sigma_arcsec,
            config.neighbors_per_day,config.max_gap_days)
        if counts[4]:
            return reference_scan(tracks,positions,velocities,days,earth,earth_velocity,config,deadline,partition,progress,hypotheses)
        counters+=counts
        if len(part):parts.append(part);retained+=len(part)
        if retained>config.max_edges:
            combined=unique_edges(np.concatenate([edges]+parts));discarded+=max(0,len(combined)-config.max_edges)
            edges=best_edges(combined,config.max_edges);parts=[];retained=0
    combined=unique_edges(np.concatenate([edges]+parts));discarded+=max(0,len(combined)-config.max_edges)
    return best_edges(combined,config.max_edges),dict(zip(
        ('curved_queries','curved_queries_at_cap','curved_edges_proposed','curved_day_pairs_scanned'),map(int,counters[:4])),
        curved_edge_pool_discarded=discarded)
