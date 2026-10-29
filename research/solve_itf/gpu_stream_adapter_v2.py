"""Follow-up removes duplicate transport seeds and per-start gather allocations.

Every finite ordered start and its epoch remain in the packed bank. Plan.seeds
is only preparation input; result assembly uses groups/epochs/bank states.
"""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from numba import njit
import gpu_stream_adapter as base
from gpu_batch_host import prepare_bank
from magictools.itf.derivatives import residual_jacobian, solve_lm


def _prepare(jobs):
    started=time.perf_counter()
    bank,plans,groups,epochs,bounds,limit=prepare_bank(base._GEOMETRY,jobs,compact_geometry=True)
    plans=[replace(p,seeds=()) if p is not None else None for p in plans]
    receipt=base.store_prepared((bank,plans,groups,epochs,bounds,limit),base._SHARED)
    return receipt,time.perf_counter()-started


@njit(cache=True,nogil=True)
def _check(lo,hi,seeds,states,bounds,offsets,observers,observed,east,north,
           sigma,limit,costs,flags,residuals,jacobians,fallback,index):
    for c in range(lo,hi):
        begin,end=bounds[c],bounds[c+1];first=index[begin]
        contiguous=True
        for p in range(begin,end):
            if index[p]!=first+p-begin:contiguous=False;break
        if contiguous:
            finish=first+end-begin
            o,u,e,n=observers[first:finish],observed[first:finish],east[first:finish],north[first:finish]
        else:
            ix=index[begin:end];o,u,e,n=observers[ix],observed[ix],east[ix],north[ix]
        r,j=residual_jacobian(states[c],offsets[begin:end],o,u,e,n,sigma)
        cost=np.dot(r,r)
        mismatch=(not np.isfinite(r).all() or not np.isfinite(j).all() or np.all(r==1e12)
                  or not np.isfinite(costs[c])
                  or abs(sigma*np.sqrt(cost/(end-begin))-sigma*np.sqrt(costs[c]/(end-begin)))>.001)
        if mismatch:
            x,r,j,ok=solve_lm(seeds[c],offsets[begin:end],o,u,e,n,sigma,limit)
            states[c]=x;flags[c]=ok;fallback[c]=True
        residuals[2*begin:2*end]=r;jacobians[2*begin:2*end]=j


def checked(bank,out,limit,threads=20):
    bounds=np.r_[0,np.cumsum(np.bincount(bank['case_index']))]
    n=len(bank['states']);no=len(bank['offsets']);states=out['states'].copy();flags=out['success'].copy()
    r=np.empty(2*no);j=np.empty((2*no,6));fallback=np.zeros(n,dtype=np.bool_)
    def run(pair):
        _check(*pair,bank['states'],states,bounds,*[bank[k] for k in ('offsets','observers','observed','east','north')],
               float(bank['sigma']),limit,out['cost'],flags,r,j,fallback,bank['geometry_index'])
    edges=np.linspace(0,n,min(n,threads*8)+1,dtype=int)
    with ThreadPoolExecutor(max_workers=threads) as pool:list(pool.map(run,zip(edges[:-1],edges[1:])))
    return states,flags,r,j,fallback


class StreamGPUBatchV2(base.StreamGPUBatch):
    prepare_task=staticmethod(_prepare)
    verify=staticmethod(checked)
