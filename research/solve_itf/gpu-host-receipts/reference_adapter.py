"""Research-only batched injection into unchanged fit_chain planning/selection.

CPU checks every returned GPU state. A mismatch causes replay from that exact
original seed on the CPU, with explicit counters. This is a hybrid benchmark,
not proof that GPU and CPU explore the same basins or that a chain is genuine.
"""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import time
import numpy as np
from numba import njit
from magictools.itf.fit import fit_chain, prepare_fit, fit_prepared, check_deadline
from magictools.itf.derivatives import solve_lm, residual_jacobian
from magictools.itf.gpu_fit import ResidentWorkspace


@njit(cache=True,nogil=True)
def _check_block(lo,hi,seeds,states,bounds,offsets,observers,observed,east,north,
                 sigma,limit,costs,flags,residuals,jacobians,fallback):
    for c in range(lo,hi):
        begin,end=bounds[c],bounds[c+1]
        r,j=residual_jacobian(states[c],offsets[begin:end],observers[begin:end],
                             observed[begin:end],east[begin:end],north[begin:end],sigma)
        cost=np.dot(r,r)
        mismatch=(not np.isfinite(r).all() or not np.isfinite(j).all() or np.all(r==1e12)
                  or not np.isfinite(costs[c])
                  or abs(sigma*np.sqrt(cost/(end-begin))-sigma*np.sqrt(costs[c]/(end-begin)))>.001)
        if mismatch:
            x,r,j,ok=solve_lm(seeds[c],offsets[begin:end],observers[begin:end],
                             observed[begin:end],east[begin:end],north[begin:end],sigma,limit)
            states[c]=x;flags[c]=ok;fallback[c]=True
        residuals[2*begin:2*end]=r;jacobians[2*begin:2*end]=j


def checked_cpu_replay(bank,out,limit,threads=20):
    """Compiled blocks replace per-start Python tasks; same checks and fallback."""
    bounds=np.r_[0,np.cumsum(np.bincount(bank['case_index']))]
    n=len(bank['states']);no=len(bank['offsets']);states=out['states'].copy();flags=out['success'].copy()
    r=np.empty(2*no);j=np.empty((2*no,6));fallback=np.zeros(n,dtype=np.bool_)
    def run(pair):
        _check_block(*pair,bank['states'],states,bounds,*[bank[k] for k in ('offsets','observers','observed','east','north')],float(bank['sigma']),limit,out['cost'],flags,r,j,fallback)
    edges=np.linspace(0,n,min(n,threads*8)+1,dtype=int)
    with ThreadPoolExecutor(max_workers=threads) as pool:list(pool.map(run,zip(edges[:-1],edges[1:])))
    return [(states[c],r[2*bounds[c]:2*bounds[c+1]],j[2*bounds[c]:2*bounds[c+1]],bool(flags[c])) for c in range(n)],fallback


def cpu_batch(geometry, jobs):
    with ThreadPoolExecutor(max_workers=20) as pool:
        return list(pool.map(lambda job: fit_chain(geometry,job[0],**job[1]),jobs))


class GPUBatch:
    def __init__(self, *, allow_simulator=False):
        self.receipts=[]
        self.allow_simulator=allow_simulator

    def __call__(self, geometry, jobs):
        started=time.perf_counter();calls=[];groups=[];plans=[]
        planning_keys = {'deadline', 'extra_seeds', 'broad_starts', 'start_policy', 'allow_two_tracklets'}
        for key, settings in jobs:
            if settings.get('jacobian_backend', 'compiled') != 'compiled':
                raise ValueError('GPU adapter requires compiled policy')
            before = len(calls)
            plan = prepare_fit(geometry, key, **{k:v for k,v in settings.items() if k in planning_keys})
            plans.append(plan)
            if plan is not None:
                for state, epoch in plan.seeds:
                    check_deadline(settings.get('deadline', np.inf))
                    if np.isfinite(state).all():
                        calls.append((state.copy(), plan.t-epoch, plan.observers, plan.u,
                                      plan.east, plan.north, settings.get('sigma', .3),
                                      settings.get('max_nfev', 80)))
            groups.append((before, len(calls)))
        if not calls:return [None]*len(jobs)
        sigma=float(calls[0][-2]);limit=int(calls[0][-1])
        assert all(x[-2]==sigma and x[-1]==limit for x in calls)
        b=dict(states=np.array([x[0] for x in calls]),case_index=np.repeat(np.arange(len(calls),dtype=np.int64),[len(x[1]) for x in calls]),sigma=np.array(sigma))
        for col,name in enumerate(('offsets','observers','observed','east','north'),1):b[name]=np.ascontiguousarray(np.concatenate([x[col] for x in calls]))
        planned=time.perf_counter()-started
        workspace=ResidentWorkspace(b,coalesced=True,allow_simulator=self.allow_simulator)
        out=workspace.solve(b['states'],limit,parallel_normal=True,parallel_copy=True)
        check_start=time.perf_counter()
        checked,fallback=checked_cpu_replay(b,out,limit)
        verified=time.perf_counter()-check_start
        def select(job):
            (key,settings),(lo,hi),plan=job
            if plan is None:
                return None
            index=lo;fixed_geometry=None
            def replay(*args):
                nonlocal index,fixed_geometry
                if index>=hi:raise ValueError('Replay generated additional starts')
                expected=calls[index]
                # Geometry is constructed once per chain inside fit_chain and
                # remains the same object throughout its seed loop. Check its
                # values on the first call, then its identity on later calls.
                if fixed_geometry is None:
                    for a,b in zip(args[2:6],expected[2:6]):
                        if not np.array_equal(a,b,equal_nan=True):raise ValueError('Fit geometry changed')
                    fixed_geometry=args[2:6]
                elif not all(a is b for a,b in zip(args[2:6],fixed_geometry)):
                    raise ValueError('Fit geometry object changed within replay')
                if (args[0].tobytes()!=expected[0].tobytes()
                    or args[1].tobytes()!=expected[1].tobytes()
                    or args[-2:]!=expected[-2:]):raise ValueError('Fit seed/epoch/policy changed')
                answer=checked[index];index+=1;return answer
            row=fit_prepared(plan,**{k:v for k,v in settings.items() if k not in {'extra_seeds','broad_starts','allow_two_tracklets'}},optimizer=replay)
            if index!=hi:raise ValueError('Replay omitted starts')
            return row
        with ThreadPoolExecutor(max_workers=20) as pool:rows=list(pool.map(select,zip(jobs,groups,plans)))
        self.receipts.append(dict(chains=len(jobs),starts=len(calls),planning_seconds=planned,
            gpu_setup_seconds=workspace.setup_seconds,gpu_solve_profile=out['profile'],
            cpu_verify_fallback_seconds=verified,cpu_fallback_starts=int(fallback.sum()),
            total_seconds=time.perf_counter()-started))
        return rows
