"""Experimental device-resident batched LM, with the existing solver policy.

Per-observation propagation plus per-chain reductions/6x6 solves stay on GPU.
No production acceptance, new starts, clipped data or adjusted convergence cuts.
Floating-point reduction order can differ from BLAS; report optimizer status
changes separately even when final astrometric residuals agree.
"""
import math
import time
import numpy as np
from numba import cuda, float64
from .gpu import ResidentBank, _evaluate


@cuda.jit(fastmath=False)
def _initialize(bounds, r, j, valid, cost, damping, increase, refresh, done,
                success, reasons, iterations, evaluations):
    c = cuda.grid(1)
    if c >= len(cost):
        return
    good = True
    for obs in range(bounds[c], bounds[c+1]):
        if valid[obs] == 0:
            good = False
    value = 0.
    for row in range(2*bounds[c], 2*bounds[c+1]):
        if not good:
            r[row] = 1e12
            for k in range(6):
                j[row, k] = 0.
        value += r[row]*r[row]
    cost[c], damping[c], increase[c] = value, 1e-3, 2.
    refresh[c], done[c], success[c] = 1, 0, 0
    reasons[c], iterations[c], evaluations[c] = 0, 0, 1


@cuda.jit(fastmath=False)
def _normal_parallel(bounds, r, j, cost, scales, gradient, normal, refresh, done):
    index = cuda.grid(1)
    c, k = index // 6, index % 6
    if c >= len(cost) or done[c] or not refresh[c]:
        return
    # Do not act on invalid initial residual sentinels.
    if not math.isfinite(cost[c]) or cost[c] >= 1e23:
        return
    value = 0.
    for row in range(2*bounds[c], 2*bounds[c+1]):
        value += j[row, k]*j[row, k]
    scales[c, k] = max(math.sqrt(value), 1e-12)


@cuda.jit(fastmath=False)
def _normal_accumulate(bounds, r, j, cost, scales, gradient, normal, refresh, done):
    index = cuda.grid(1)
    c, k = index // 6, index % 6
    if c >= len(cost) or done[c] or not refresh[c]:
        return
    if not math.isfinite(cost[c]) or cost[c] >= 1e23:
        return
    gradient[c, k] = 0.
    for l in range(k+1):
        normal[c, k, l] = 0.
    for row in range(2*bounds[c], 2*bounds[c+1]):
        value = j[row, k]/scales[c, k]
        gradient[c, k] += value*r[row]
        for l in range(k+1):
            normal[c, k, l] += value*j[row, l]/scales[c, l]
    for l in range(k):
        normal[c, l, k] = normal[c, k, l]


@cuda.jit
def _clear_refresh(refresh, done):
    c = cuda.grid(1)
    if c < len(refresh) and not done[c]:
        refresh[c] = 0


@cuda.jit(fastmath=False)
def _propose(states, bounds, r, j, cost, damping, refresh, scales, normal,
             gradient, trials, scaled_steps, steps, done, success, reasons,
             iterations, pending, iteration):
    c = cuda.grid(1)
    if c >= len(states):
        return
    pending[c] = 0
    if done[c]:
        return
    iterations[c] = iteration+1
    if not math.isfinite(cost[c]) or cost[c] >= 1e23:
        done[c], reasons[c] = 1, 2
        return
    if refresh[c]:
        for k in range(6):
            scales[c, k], gradient[c, k] = 0., 0.
            for l in range(6):
                normal[c, k, l] = 0.
        for row in range(2*bounds[c], 2*bounds[c+1]):
            for k in range(6):
                scales[c, k] += j[row, k]*j[row, k]
        for k in range(6):
            scales[c, k] = max(math.sqrt(scales[c, k]), 1e-12)
        for row in range(2*bounds[c], 2*bounds[c+1]):
            for k in range(6):
                value = j[row, k]/scales[c, k]
                gradient[c, k] += value*r[row]
                for l in range(k+1):
                    normal[c, k, l] += value*j[row, l]/scales[c, l]
        for k in range(6):
            for l in range(k):
                normal[c, l, k] = normal[c, k, l]
        refresh[c] = 0
    largest = 0.
    for k in range(6):
        if not math.isfinite(gradient[c, k]):
            largest = math.nan
            break
        largest = max(largest, abs(gradient[c, k]))
    if largest <= 1e-7*max(1., math.sqrt(cost[c])):
        success[c], done[c], reasons[c] = 1, 1, 1
        return
    lower = cuda.local.array((6, 6), float64)
    temporary = cuda.local.array(6, float64)
    for k in range(6):
        for l in range(k+1):
            value = normal[c, k, l]
            if k == l:
                value += damping[c]
            for m in range(l):
                value -= lower[k, m]*lower[l, m]
            if k == l:
                if value <= 0:
                    damping[c] = max(damping[c]*10, 1e-12)
                    return
                lower[k, l] = math.sqrt(value)
            else:
                lower[k, l] = value/lower[l, l]
    for k in range(6):
        value = -gradient[c, k]
        for l in range(k):
            value -= lower[k, l]*temporary[l]
        temporary[k] = value/lower[k, k]
    for k in range(5, -1, -1):
        value = temporary[k]
        for l in range(k+1, 6):
            value -= lower[l, k]*scaled_steps[c, l]
        scaled_steps[c, k] = value/lower[k, k]
    for k in range(6):
        steps[c, k] = scaled_steps[c, k]/scales[c, k]
        trials[c, k] = states[c, k]+steps[c, k]
    pending[c] = 1


@cuda.jit(fastmath=False)
def _accept(states, bounds, r, j, cost, damping, increase, refresh, normal,
            gradient, trials, scaled_steps, steps, done, success, reasons,
            pending, nr, nj, valid, evaluations, copy_inline):
    c = cuda.grid(1)
    if c >= len(states) or pending[c] == 0:
        return
    evaluations[c] += 1
    good = True
    for obs in range(bounds[c], bounds[c+1]):
        if valid[obs] == 0:
            good = False
    new_cost = 0.
    for row in range(2*bounds[c], 2*bounds[c+1]):
        value = nr[row] if good else 1e12
        new_cost += value*value
    linear, quadratic = 0., 0.
    for k in range(6):
        linear += gradient[c, k]*scaled_steps[c, k]
        value = 0.
        for l in range(6):
            value += normal[c, k, l]*scaled_steps[c, l]
        quadratic += scaled_steps[c, k]*value
    predicted = -2*linear-quadratic
    if new_cost < cost[c] and predicted > 0 and math.isfinite(new_cost):
        ratio = (cost[c]-new_cost)/predicted
        relative = (cost[c]-new_cost)/max(cost[c], 1e-30)
        step_norm, state_norm = 0., 0.
        for k in range(6):
            states[c, k] = trials[c, k]
            step_norm += steps[c, k]*steps[c, k]
            state_norm += states[c, k]*states[c, k]
        if copy_inline:
            for row in range(2*bounds[c], 2*bounds[c+1]):
                r[row] = nr[row]
                for k in range(6):
                    j[row, k] = nj[row, k]
        refresh[c] = 1
        small = math.sqrt(step_norm) <= 1e-9*(math.sqrt(state_norm)+1e-9)
        cost[c] = new_cost
        damping[c] *= max(1/3, 1-(2*ratio-1)**3)
        damping[c] = max(damping[c], 1e-15)
        increase[c] = 2.
        if (relative < 1e-7 and predicted/max(cost[c], 1e-30) < 1e-7) or small:
            success[c], done[c], reasons[c] = 1, 1, 1
    else:
        damping[c] = min(damping[c]*increase[c], 1e30)
        increase[c] = min(2*increase[c], 1e12)
        if damping[c] >= 1e25:
            largest = 0.
            for k in range(6):
                if not math.isfinite(gradient[c, k]):
                    largest = math.nan
                    break
                largest = max(largest, abs(gradient[c, k]))
            success[c] = 1 if largest < 1e-5 else 0
            done[c], reasons[c] = 1, 1 if success[c] else 3


@cuda.jit(fastmath=False)
def _copy_accepted(case_index, refresh, pending, r, j, nr, nj):
    obs = cuda.grid(1)
    if obs >= len(case_index):
        return
    c = case_index[obs]
    if refresh[c] and pending[c]:
        for coordinate in range(2):
            row = 2*obs+coordinate
            r[row] = nr[row]
            for k in range(6):
                j[row,k] = nj[row,k]


@cuda.jit
def _remaining(done, remaining):
    c = cuda.grid(1)
    if c < len(done) and not done[c]:
        cuda.atomic.add(remaining, 0, 1)


class ResidentWorkspace:
    """Validated observation bank and scratch storage reused across start batches.

    Observations, weights and dimensions are fixed at construction. Each solve
    requires an explicit fresh state bank; damping, flags and counters reset.
    Setup cost is reported separately, never hidden in throughput claims.
    A workspace is single-stream/non-reentrant and must not be shared by callers.
    """
    def __init__(self, bank, *, allow_simulator=False, coalesced=False, columnar_jacobian=False, resident_geometry=None):
        started = time.perf_counter()
        order = 'F' if coalesced else 'C'
        base = ResidentBank(bank, allow_simulator=allow_simulator, columnar_jacobian=columnar_jacobian, resident_geometry=resident_geometry)
        if np.any(np.diff(base.case_index) < 0):
            raise ValueError("LM requires observations grouped contiguously by case")
        n, nobs = len(bank['states']), len(bank['offsets'])
        bounds = cuda.to_device(np.r_[0,np.cumsum(np.bincount(bank['case_index']))].astype(np.int64))
        states = base.inputs[0]
        r, j = base.residual.reshape(2*nobs), base.jacobian.reshape(2*nobs, 6)
        trials = cuda.device_array((n,6),np.float64, order=order)
        trial_r = cuda.device_array((nobs,2),np.float64)
        trial_j = (cuda.device_array((2*nobs,6),np.float64, order='F').reshape(nobs,2,6)
                   if columnar_jacobian else cuda.device_array((nobs,2,6),np.float64))
        trial_valid = cuda.device_array(nobs,np.uint8)
        cost, damping, increase = [cuda.device_array(n,np.float64) for _ in range(3)]
        scales, gradient, scaled_steps, steps = [cuda.device_array((n,6),np.float64, order=order) for _ in range(4)]
        normal = cuda.device_array((n,6,6),np.float64, order=order)
        refresh, done, success, pending = [cuda.device_array(n,np.uint8) for _ in range(4)]
        reasons, iterations, evaluations = [cuda.device_array(n,np.int64) for _ in range(3)]
        remaining = cuda.to_device(np.array([n],np.int32))
        self.base = base
        self.n = n
        self.nobs = nobs
        self.bounds = bounds
        self.states = states
        self.r = r
        self.j = j
        self.trials = trials
        self.trial_r = trial_r
        self.trial_j = trial_j
        self.trial_valid = trial_valid
        self.cost = cost
        self.damping = damping
        self.increase = increase
        self.scales = scales
        self.gradient = gradient
        self.scaled_steps = scaled_steps
        self.steps = steps
        self.normal = normal
        self.refresh = refresh
        self.done = done
        self.success = success
        self.pending = pending
        self.reasons = reasons
        self.iterations = iterations
        self.evaluations = evaluations
        self.remaining = remaining

        cuda.synchronize()
        self.setup_seconds = time.perf_counter()-started
        self.coalesced = coalesced
        self.columnar_jacobian = columnar_jacobian

    def solve(self, initial_states, max_iterations=80, *, profile_events=False,
              parallel_normal=False, parallel_copy=False, completion_interval=4):
        return _solve_workspace(self, initial_states, max_iterations,
                                profile_events=profile_events,
                                parallel_normal=parallel_normal, parallel_copy=parallel_copy, completion_interval=completion_interval)


def solve_resident(bank, max_iterations=80, *, allow_simulator=False,
                   profile_events=False):
    """Compatibility wrapper: charge construction and solve for every call."""
    started = time.perf_counter()
    workspace = ResidentWorkspace(bank, allow_simulator=allow_simulator)
    result = workspace.solve(bank['states'], max_iterations,
                             profile_events=profile_events)
    profile = result['profile']
    profile['allocation_upload_seconds'] = workspace.setup_seconds + profile['state_upload_seconds']
    profile['wall_seconds'] = time.perf_counter()-started
    return result


def _solve_workspace(workspace, initial_states, max_iterations, *,
                     profile_events=False, parallel_normal=False, parallel_copy=False, completion_interval=4):
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 0:
        raise ValueError('max_iterations must be a nonnegative integer')
    if not isinstance(completion_interval, (int, np.integer)) or not 1 <= completion_interval <= 80:
        raise ValueError('completion_interval must be an integer in [1, 80]')
    initial_states = np.asarray(initial_states)
    if initial_states.dtype != np.float64 or initial_states.shape != (workspace.n, 6):
        raise ValueError('Updated state shape/dtype differs from frozen workspace')
    started = time.perf_counter()
    base = workspace.base
    n = workspace.n
    nobs = workspace.nobs
    bounds = workspace.bounds
    states = workspace.states
    r = workspace.r
    j = workspace.j
    trials = workspace.trials
    trial_r = workspace.trial_r
    trial_j = workspace.trial_j
    trial_valid = workspace.trial_valid
    cost = workspace.cost
    damping = workspace.damping
    increase = workspace.increase
    scales = workspace.scales
    gradient = workspace.gradient
    scaled_steps = workspace.scaled_steps
    steps = workspace.steps
    normal = workspace.normal
    refresh = workspace.refresh
    done = workspace.done
    success = workspace.success
    pending = workspace.pending
    reasons = workspace.reasons
    iterations = workspace.iterations
    evaluations = workspace.evaluations
    remaining = workspace.remaining
    states.copy_to_device(np.ascontiguousarray(initial_states))
    cuda.synchronize()
    state_upload_seconds = time.perf_counter()-started
    launch_profile = []
    def launch(kind, kernel, blocks, args):
        if profile_events:
            start, end = cuda.event(timing=True), cuda.event(timing=True)
            start.record();kernel[blocks,128](*args);end.record()
            launch_profile.append((kind,start,end))
        else:
            kernel[blocks,128](*args)
    grid, observation_grid = (n+127)//128, (nobs+127)//128
    loop_start = time.perf_counter()
    launch('propagation',base.kernel,observation_grid,(*base.inputs,base.sigma,base.active,base.residual,base.jacobian,base.valid))
    launch('initialization',_initialize,grid,(bounds,r,j,base.valid,cost,damping,increase,refresh,done,success,reasons,iterations,evaluations))
    checks = []
    for iteration in range(max_iterations):
        if parallel_normal:
            normal_args = (bounds,r,j,cost,scales,gradient,normal,refresh,done)
            launch('normal_scales',_normal_parallel,(6*n+127)//128,normal_args)
            launch('normal_accumulate',_normal_accumulate,(6*n+127)//128,normal_args)
            launch('normal_flags',_clear_refresh,grid,(refresh,done))
        launch('proposal',_propose,grid,(states,bounds,r,j,cost,damping,refresh,scales,normal,gradient,trials,
               scaled_steps,steps,done,success,reasons,iterations,pending,iteration))
        launch('propagation',base.kernel,observation_grid,(trials,*base.inputs[1:],base.sigma,pending,trial_r,trial_j,trial_valid))
        launch('accept',_accept,grid,(states,bounds,r,j,cost,damping,increase,refresh,normal,gradient,trials,
               scaled_steps,steps,done,success,reasons,pending,trial_r.reshape(2*nobs),trial_j.reshape(2*nobs,6),trial_valid,evaluations,not parallel_copy))
        if parallel_copy:
            launch('copy_accepted',_copy_accepted,observation_grid,(base.inputs[1],refresh,pending,r,j,trial_r.reshape(2*nobs),trial_j.reshape(2*nobs,6)))
        if (iteration+1)%completion_interval == 0 or iteration+1==max_iterations:
            remaining.copy_to_device(np.zeros(1,np.int32));_remaining[grid,128](done,remaining)
            count=int(remaining.copy_to_host()[0]);checks.append([iteration+1,count])
            if count==0:break
    cuda.synchronize()
    loop_seconds = time.perf_counter()-loop_start
    stamp = time.perf_counter()
    result = dict(states=states.copy_to_host(),success=success.copy_to_host().astype(bool),
                  cost=cost.copy_to_host(),reasons=reasons.copy_to_host(),
                  iterations=iterations.copy_to_host(),evaluations=evaluations.copy_to_host())
    unfinished = done.copy_to_host()==0;result['reasons'][unfinished]=4
    download_seconds = time.perf_counter()-stamp
    profile=dict(wall_seconds=time.perf_counter()-started,
                 state_upload_seconds=state_upload_seconds,
                 workspace_setup_seconds=workspace.setup_seconds,
                 workspace_reused=True, parallel_normal=parallel_normal,
                 coalesced=workspace.coalesced, columnar_jacobian=workspace.columnar_jacobian,
                 parallel_copy=parallel_copy,
                 device_iterations_wall_seconds=loop_seconds,final_download_seconds=download_seconds,
                 completion_interval=completion_interval, completion_checks=checks,per_iteration_full_array_transfers=0)
    if profile_events:
        times={}
        for name,start,end in launch_profile:
            times[name]=times.get(name,0.)+cuda.event_elapsed_time(start,end)/1000
        profile['instrumented_kernel_seconds']=times
        profile['timing_warning']='Per-kernel events add launch overhead; not headline timing'
    result['profile']=profile
    return result
