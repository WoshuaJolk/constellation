"""Research-only batched injection into unchanged fit_chain planning/selection.

CPU checks every returned GPU state. A mismatch causes replay from that exact
original seed on the CPU, with explicit counters. This is a hybrid benchmark,
not proof that GPU and CPU explore the same basins or that a chain is genuine.
"""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import time
import numpy as np
from magictools.itf.fit import fit_chain, check_deadline
from magictools.itf.derivatives import solve_lm, residual_jacobian
from magictools.itf.gpu_fit import ResidentWorkspace


def cpu_batch(geometry, jobs):
    with ThreadPoolExecutor(max_workers=20) as pool:
        return list(pool.map(lambda job: fit_chain(geometry,job[0],**job[1]),jobs))


class GPUBatch:
    def __init__(self, *, allow_simulator=False):
        self.receipts=[]
        self.allow_simulator=allow_simulator

    def __call__(self, geometry, jobs):
        started=time.perf_counter();calls=[];groups=[];plans=[]
        def collect(*args):
            calls.append(tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args))
            return args[0], np.full(2*len(args[1]),1e12),np.zeros((2*len(args[1]),6)),False
        for key,settings in jobs:
            before=len(calls);fit_chain(geometry,key,**settings,optimizer=collect)
            groups.append((before,len(calls)))
        if not calls:return [None]*len(jobs)
        sigma=float(calls[0][-2]);limit=int(calls[0][-1])
        assert all(x[-2]==sigma and x[-1]==limit for x in calls)
        b=dict(states=np.array([x[0] for x in calls]),case_index=np.repeat(np.arange(len(calls),dtype=np.int64),[len(x[1]) for x in calls]),sigma=np.array(sigma))
        for col,name in enumerate(('offsets','observers','observed','east','north'),1):b[name]=np.ascontiguousarray(np.concatenate([x[col] for x in calls]))
        planned=time.perf_counter()-started
        workspace=ResidentWorkspace(b,coalesced=True,allow_simulator=self.allow_simulator)
        out=workspace.solve(b['states'],limit,parallel_normal=True,parallel_copy=True)
        check_start=time.perf_counter()
        def verify(index):
            args=calls[index];x=out['states'][index];r,j=residual_jacobian(x,*args[1:-1]);cost=float(r@r)
            expected=float(out['cost'][index]);n=len(args[1])
            valid=np.isfinite(r).all() and np.isfinite(j).all() and not np.all(r==1e12)
            # Existing numerical comparison precision, never an acceptance cut.
            discrepancy=not valid or not np.isfinite(expected) or abs(sigma*np.sqrt(cost/n)-sigma*np.sqrt(expected/n))>.001
            if discrepancy:
                return solve_lm(*args), True
            return (x,r,j,bool(out['success'][index])),False
        with ThreadPoolExecutor(max_workers=20) as pool:checked=list(pool.map(verify,range(len(calls))))
        verified=time.perf_counter()-check_start
        rows=[]
        for (key,settings),(lo,hi) in zip(jobs,groups):
            index=lo
            def replay(*args):
                nonlocal index
                if index>=hi:raise ValueError('Replay generated additional starts')
                expected=calls[index]
                for a,b in zip(args,expected):
                    if not np.array_equal(a,b,equal_nan=True):raise ValueError('Fit plan changed between collection/replay')
                answer=checked[index][0];index+=1;return answer
            rows.append(fit_chain(geometry,key,**settings,optimizer=replay))
            if index!=hi:raise ValueError('Replay omitted starts')
        self.receipts.append(dict(chains=len(jobs),starts=len(calls),planning_seconds=planned,
            gpu_setup_seconds=workspace.setup_seconds,gpu_solve_profile=out['profile'],
            cpu_verify_fallback_seconds=verified,cpu_fallback_starts=sum(x[1] for x in checked),
            total_seconds=time.perf_counter()-started))
        return rows
