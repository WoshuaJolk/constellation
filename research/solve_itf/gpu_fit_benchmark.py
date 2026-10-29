#!/usr/bin/env python3
"""Complete single-start LM benchmark; saved-state refinement, not new discovery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from numba import njit
from magictools.itf import derivatives
from magictools.itf.batched_fit import CPUEvaluator, GPUEvaluator, solve_batch
from magictools.itf.gpu import ResidentBank, cpu_reference
from magictools.itf.gpu_fit import solve_resident
from gpu_benchmark import subset, compare


@njit(cache=True, nogil=True)
def original_block(low, high, seeds, bounds, offsets, observers, observed,
                   east, north, sigma, iterations, states, success, cost):
    # Keep the scheduling loop inside compiled code so threads do not re-enter
    # Python once per tiny fit. No arithmetic/convergence-policy changes.
    for c in range(low, high):
        lo, hi = bounds[c], bounds[c+1]
        x, r, j, ok = derivatives.solve_lm(seeds[c], offsets[lo:hi],
            observers[lo:hi], observed[lo:hi], east[lo:hi], north[lo:hi], sigma, iterations)
        states[c], success[c], cost[c] = x, ok, np.dot(r, r)


def original(bank, threads, iterations):
    bounds = np.r_[0, np.cumsum(np.bincount(bank['case_index']))]
    n = len(bank['states'])
    states, success, cost = np.empty((n, 6)), np.empty(n, bool), np.empty(n)
    def block(low, high):
        original_block(low, high, bank['states'], bounds, bank['offsets'], bank['observers'],
                       bank['observed'], bank['east'], bank['north'], float(bank['sigma']),
                       iterations, states, success, cost)
    stamp = time.perf_counter()
    if threads == 1:
        block(0, n)
    else:
        # Many ordered blocks mitigate unequal iteration counts without copying inputs.
        edges = np.linspace(0, n, min(n, threads*8)+1, dtype=int)
        with ThreadPoolExecutor(max_workers=threads) as pool:
            list(pool.map(lambda x: block(*x), zip(edges[:-1], edges[1:])))
    return dict(states=states, success=success, cost=cost,
                profile={'wall_seconds': time.perf_counter()-stamp})


def instrumented_cpu(bank, iterations):
    """Same source with process-CPU clock brackets; one thread, untimed diagnostic.

    Linux libc clock uses 1e6 ticks/s. Brackets include clock-call overhead;
    throughput comes from uninstrumented runs, never this diagnostic.
    """
    clock = ctypes.CDLL(None).clock
    clock.argtypes, clock.restype = [], ctypes.c_long
    src = inspect.getsource(derivatives.solve_lm.py_func)
    src = src[src.index('def solve_lm'):].replace('def solve_lm(', 'def profiled_lm(', 1)
    src = src.replace('    state = seed.copy()', '    begin_tick = clock()\n    model_ticks = 0\n    calls = 0\n    state = seed.copy()', 1)
    lines = []
    for line in src.splitlines():
        if '= residual_jacobian(' in line:
            indentation = line[:len(line)-len(line.lstrip())]
            lines += [indentation+'tick = clock()', line,
                      indentation+'model_ticks += clock()-tick', indentation+'calls += 1']
        elif line.strip() == 'return state, residual, jacobian, success':
            lines.append('    return state, residual, jacobian, success, model_ticks, clock()-begin_tick, calls')
        else:
            lines.append(line)
    namespace = dict(derivatives.solve_lm.py_func.__globals__, clock=clock)
    exec('\n'.join(lines), namespace)
    function = njit(nogil=True)(namespace['profiled_lm'])
    b = subset(bank, 0, min(1000, len(bank['states'])))
    bounds = np.r_[0, np.cumsum(np.bincount(b['case_index']))]
    args = lambda c: (b['states'][c], b['offsets'][bounds[c]:bounds[c+1]],
        b['observers'][bounds[c]:bounds[c+1]], b['observed'][bounds[c]:bounds[c+1]],
        b['east'][bounds[c]:bounds[c+1]], b['north'][bounds[c]:bounds[c+1]], float(b['sigma']), iterations)
    function(*args(0))  # JIT compilation excluded
    total = model = calls = agreements = 0
    for c in range(len(b['states'])):
        out = function(*args(c)); reference = derivatives.solve_lm(*args(c))
        agreements += int(np.array_equal(out[0], reference[0]) and out[3] == reference[3])
        model += out[4]; total += out[5]; calls += out[6]
    return {'cases':len(b['states']), 'exact_state_and_flag_agreements':agreements,
            'model_calls':calls,'residual_jacobian_cpu_seconds':model/1e6,
            'total_instrumented_cpu_seconds':total/1e6,
            'remaining_solver_cpu_seconds':(total-model)/1e6,
            'scope':'First 1000 fixed cases; single-thread process CPU ticks, bracket overhead included; not throughput timing'}


def summary(result, bank):
    rms = float(bank['sigma'])*np.sqrt(result['cost']/np.bincount(bank['case_index']))
    return {'cases':len(rms),'converged':int(result['success'].sum()),
            'nonconverged':int((~result['success']).sum()),
            'invalid_objective':int((~np.isfinite(result['cost']) | (result['cost'] >= 1e23)).sum()),
            'rms_arcsec_quantiles':np.quantile(rms[np.isfinite(rms)], [0,.5,.9,.99,1]).tolist(),
            'profile':result['profile']}


def disagreement(a, b, bank):
    nr = np.bincount(bank['case_index'])
    rms_a = float(bank['sigma'])*np.sqrt(a['cost']/nr)
    rms_b = float(bank['sigma'])*np.sqrt(b['cost']/nr)
    diff = np.abs(rms_a-rms_b)
    return {'success_disagreements':int(np.count_nonzero(a['success']!=b['success'])),
            'exact_state_matches':int(np.all(a['states']==b['states'],axis=1).sum()),
            'max_state_absolute_difference':float(np.max(np.abs(a['states']-b['states']))),
            'max_rms_difference_arcsec':float(np.max(diff)),
            'rms_difference_above_0_001_arcsec':int((diff>.001).sum()),
            'rms_difference_quantiles_arcsec':np.quantile(diff,[0,.5,.9,.99,1]).tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--iterations',type=int,default=80)
    p.add_argument('--cpu-only',action='store_true');p.add_argument('--limit',type=int)
    a=p.parse_args();b=dict(np.load(a.bank,allow_pickle=False))
    if a.limit:b=subset(b,0,min(a.limit,len(b['states'])))
    threads=sorted(set([1,min(4,os.cpu_count()),os.cpu_count()]))
    report={'scope':__doc__,'bank_sha256':hashlib.sha256(a.bank.read_bytes()).hexdigest(),
        'cases':len(b['states']),'observations':len(b['offsets']),'iterations_limit':a.iterations,
        'cpu_driver':'compiled per-block loop; original solve_lm unchanged',
        'repeats':a.repeats,'cpu_threads':threads,'runs':{},'outputs':{},
        'selection':'Existing frozen first-10000 retained-candidate bank; one saved start per case; no fresh starts/observations',
        'limitations':['Single-start saved-state refinement, not full seed grid or chaining',
                       'No scientific acceptance, purity or discovery claims',
                       'Hybrid GPU keeps LM on host; resident GPU keeps LM on device with unchanged thresholds and reported reduction-order effects']}
    a.output.mkdir(parents=True,exist_ok=True)
    def save(): (a.output/'report.json').write_text(json.dumps(report,indent=2))
    def preserve(name,r):
        path=a.output/(name+'.npz');np.savez_compressed(path,**{k:r[k] for k in ('states','success','cost')},
            **{k:r[k] for k in ('reasons','iterations','evaluations') if k in r})
        report['outputs'][name]={'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    try:
        warm=subset(b,0,min(128,len(b['states'])))
        stamp=time.perf_counter();original(warm,1,a.iterations);solve_batch(warm,CPUEvaluator(warm),a.iterations)
        if not a.cpu_only:
            report['initial_same_state_parity']=compare(cpu_reference(b),ResidentBank(b).evaluate(),float(b['sigma']))
            solve_batch(warm,GPUEvaluator(warm),a.iterations)
            solve_resident(warm,a.iterations)
        report['warmup_seconds']=time.perf_counter()-stamp;save()
        reference=None
        for count in threads:
            rows=[]
            for repetition in range(a.repeats):
                r=original(b,count,a.iterations);rows.append(summary(r,b))
                if reference is None:reference=r;preserve('cpu-reference',r)
                else:
                    delta=disagreement(reference,r,b);rows[-1]['reference_comparison']=delta
                    if delta['exact_state_matches'] != len(b['states']) or delta['success_disagreements']:
                        raise RuntimeError('Original CPU rerun/threading changed results')
                report['runs']['cpu-'+str(count)]=rows;save()
        r=solve_batch(b,CPUEvaluator(b),a.iterations);preserve('cpu-batched',r)
        report['cpu_state_machine_equivalence']=disagreement(reference,r,b)
        report['runs']['cpu-batched']=[summary(r,b)];save()
        delta=report['cpu_state_machine_equivalence']
        if delta['exact_state_matches'] != len(b['states']) or delta['success_disagreements']:
            raise RuntimeError('CPU state machine differs from original policy')
        report['instrumented_original_cpu']=instrumented_cpu(b,a.iterations);save()
        if not a.cpu_only:
            report['runs']['gpu-batched']=[]
            for repetition in range(a.repeats):
                stamp=time.perf_counter();e=GPUEvaluator(b);allocation=time.perf_counter()-stamp
                r=solve_batch(b,e,a.iterations);preserve('gpu-'+str(repetition),r)
                row=summary(r,b);row['allocation_upload_seconds']=allocation
                row['end_to_end_seconds']=allocation+r['profile']['wall_seconds']
                row['reference_comparison']=disagreement(reference,r,b)
                report['runs']['gpu-batched'].append(row);save()
            # Check arithmetic at resulting states separately from optimizer path divergence.
            final_bank=dict(b,states=r['states'])
            report['final_same_state_parity']=compare(cpu_reference(final_bank),ResidentBank(final_bank).evaluate(),float(b['sigma']))
            report['runs']['gpu-resident']=[]
            for repetition in range(a.repeats):
                r=solve_resident(b,a.iterations);preserve('resident-'+str(repetition),r)
                row=summary(r,b);row['reference_comparison']=disagreement(reference,r,b)
                report['runs']['gpu-resident'].append(row);save()
            final_bank=dict(b,states=r['states'])
            report['resident_final_same_state_parity']=compare(cpu_reference(final_bank),ResidentBank(final_bank).evaluate(),float(b['sigma']))
            r=solve_resident(b,a.iterations,profile_events=True)
            preserve('resident-instrumented',r)
            report['resident_kernel_profile']=summary(r,b)
            report['resident_kernel_profile']['reference_comparison']=disagreement(reference,r,b)
        report['status']='completed'
    except Exception as e:
        report['status']='blocked';report['blocker']=type(e).__name__+': '+str(e);save();raise
    save();print(json.dumps({'status':report['status'],'cases':report['cases'],'runs':list(report['runs'])}))

if __name__=='__main__':main()
