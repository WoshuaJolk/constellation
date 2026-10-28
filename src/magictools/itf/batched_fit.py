"""Experimental batched host LM state machine for CPU/GPU fit benchmarks.

Mirrors derivatives.solve_lm (80 iterations in the benchmark). It is not wired
into production fit_chain and does not apply scientific acceptance criteria.
"""
import math
import time
import numpy as np
from numba import njit
from .derivatives import residual_jacobian


@njit(cache=True, nogil=True)
def cpu_evaluate(states, bounds, offsets, observers, observed, east, north,
                 sigma, active, residual, jacobian):
    for c in range(len(states)):
        if active[c]:
            lo, hi = bounds[c], bounds[c+1]
            r, j = residual_jacobian(states[c], offsets[lo:hi], observers[lo:hi],
                                    observed[lo:hi], east[lo:hi], north[lo:hi], sigma)
            residual[2*lo:2*hi] = r
            jacobian[2*lo:2*hi] = j


@njit(cache=True, nogil=True)
def costs(bounds, residual):
    result = np.empty(len(bounds)-1)
    for c in range(len(result)):
        r = residual[2*bounds[c]:2*bounds[c+1]]
        result[c] = np.dot(r, r)
    return result


@njit(cache=True, nogil=True)
def propose(states, bounds, residual, jacobian, cost, damping, refresh,
            scales, normal, gradient, trials, step_scaled, steps, done, success,
            reasons, iterations, iteration):
    pending = np.zeros(len(states), np.uint8)
    for c in range(len(states)):
        if done[c]:
            continue
        iterations[c] = iteration+1
        if not math.isfinite(cost[c]) or cost[c] >= 1e23:
            done[c], reasons[c] = True, 2  # invalid numerical objective
            continue
        if refresh[c]:
            scales[c, :] = 0.
            normal[c, :, :] = 0.
            gradient[c, :] = 0.
            for row in range(2*bounds[c], 2*bounds[c+1]):
                for k in range(6):
                    scales[c, k] += jacobian[row, k]*jacobian[row, k]
            for k in range(6):
                scales[c, k] = max(math.sqrt(scales[c, k]), 1e-12)
            for row in range(2*bounds[c], 2*bounds[c+1]):
                for k in range(6):
                    value = jacobian[row, k]/scales[c, k]
                    gradient[c, k] += value*residual[row]
                    for l in range(k+1):
                        normal[c, k, l] += value*jacobian[row, l]/scales[c, l]
            for k in range(6):
                for l in range(k):
                    normal[c, l, k] = normal[c, k, l]
            refresh[c] = False
        if np.max(np.abs(gradient[c])) <= 1e-7*max(1., math.sqrt(cost[c])):
            success[c], done[c], reasons[c] = True, True, 1
            continue
        system = normal[c].copy()
        for k in range(6):
            system[k, k] += damping[c]
        lower = np.zeros((6, 6))
        positive = True
        for k in range(6):
            for l in range(k+1):
                value = system[k, l]
                for m in range(l):
                    value -= lower[k, m]*lower[l, m]
                if k == l:
                    if value <= 0:
                        positive = False
                        break
                    lower[k, l] = math.sqrt(value)
                else:
                    lower[k, l] = value/lower[l, l]
            if not positive:
                break
        if not positive:
            damping[c] = max(damping[c]*10, 1e-12)
            continue
        temporary = np.zeros(6)
        for k in range(6):
            value = -gradient[c, k]
            for l in range(k):
                value -= lower[k, l]*temporary[l]
            temporary[k] = value/lower[k, k]
        for k in range(5, -1, -1):
            value = temporary[k]
            for l in range(k+1, 6):
                value -= lower[l, k]*step_scaled[c, l]
            step_scaled[c, k] = value/lower[k, k]
        steps[c] = step_scaled[c]/scales[c]
        trials[c] = states[c]+steps[c]
        pending[c] = 1
    return pending


@njit(cache=True, nogil=True)
def accept(states, bounds, residual, jacobian, cost, damping, increase, refresh,
           normal, gradient, trials, step_scaled, steps, done, success, reasons,
           pending, new_residual, new_jacobian, evaluations):
    for c in range(len(states)):
        if not pending[c]:
            continue
        evaluations[c] += 1
        lo, hi = 2*bounds[c], 2*bounds[c+1]
        r = new_residual[lo:hi]
        new_cost = np.dot(r, r)
        predicted = -2*np.dot(gradient[c], step_scaled[c])-np.dot(step_scaled[c], normal[c]@step_scaled[c])
        if new_cost < cost[c] and predicted > 0 and math.isfinite(new_cost):
            ratio = (cost[c]-new_cost)/predicted
            relative = (cost[c]-new_cost)/max(cost[c], 1e-30)
            states[c] = trials[c]
            residual[lo:hi] = r
            jacobian[lo:hi] = new_jacobian[lo:hi]
            refresh[c] = True
            small = np.linalg.norm(steps[c]) <= 1e-9*(np.linalg.norm(states[c])+1e-9)
            cost[c] = new_cost
            damping[c] *= max(1/3, 1-(2*ratio-1)**3)
            damping[c] = max(damping[c], 1e-15)
            increase[c] = 2.
            if (relative < 1e-7 and predicted/max(cost[c], 1e-30) < 1e-7) or small:
                success[c], done[c], reasons[c] = True, True, 1
        else:
            damping[c] = min(damping[c]*increase[c], 1e30)
            increase[c] = min(2*increase[c], 1e12)
            if damping[c] >= 1e25:
                success[c] = np.max(np.abs(gradient[c])) < 1e-5
                done[c], reasons[c] = True, 1 if success[c] else 3


class CPUEvaluator:
    def __init__(self, bank):
        self.bank = bank
        self.bounds = np.r_[0, np.cumsum(np.bincount(bank['case_index']))].astype(np.int64)
        self.residual = np.empty(2*len(bank['offsets']))
        self.jacobian = np.empty((len(self.residual), 6))
        self.profile = {'cpu_residual_jacobian_seconds': 0.}

    def __call__(self, states, active):
        started = time.perf_counter()
        b = self.bank
        cpu_evaluate(states, self.bounds, b['offsets'], b['observers'], b['observed'],
                     b['east'], b['north'], float(b['sigma']), active, self.residual, self.jacobian)
        self.profile['cpu_residual_jacobian_seconds'] += time.perf_counter()-started
        return self.residual, self.jacobian


class GPUEvaluator:
    def __init__(self, bank):
        from .gpu import ResidentBank
        self.bank = ResidentBank(bank)
        self.bounds = np.r_[0, np.cumsum(np.bincount(bank['case_index']))].astype(np.int64)
        self.profile = self.bank.profile

    def __call__(self, states, active):
        r, j, valid = self.bank.evaluate(states=states, active=active)
        # Original CPU sentinel for invalid geometry. Only active rows consumed.
        bad = valid == 0
        r[bad], j[bad] = 1e12, 0.
        return r.reshape(-1), j.reshape(-1, 6)


def solve_batch(bank, evaluator, max_iterations=80):
    """Complete independent single-start fits; same policy, batched evaluations."""
    started = time.perf_counter()
    states = bank['states'].copy()
    n = len(states)
    bounds = evaluator.bounds
    initial = time.perf_counter()
    r, j = evaluator(states, np.ones(n, np.uint8))
    residual, jacobian = r.copy(), j.copy()
    cost = costs(bounds, residual)
    damping, increase = np.full(n, 1e-3), np.full(n, 2.)
    refresh = np.ones(n, np.bool_)
    done, success = np.zeros(n, np.bool_), np.zeros(n, np.bool_)
    reasons, iterations = np.zeros(n, np.int64), np.zeros(n, np.int64)
    evaluations = np.ones(n, np.int64)
    scales, gradient = np.zeros((n, 6)), np.zeros((n, 6))
    normal = np.zeros((n, 6, 6))
    trials, steps, step_scaled = states.copy(), np.zeros((n, 6)), np.zeros((n, 6))
    init_seconds = time.perf_counter()-initial
    proposal_seconds = update_seconds = evaluation_seconds = 0.
    active_history = []
    for iteration in range(max_iterations):
        stamp = time.perf_counter()
        pending = propose(states, bounds, residual, jacobian, cost, damping, refresh,
                          scales, normal, gradient, trials, step_scaled, steps,
                          done, success, reasons, iterations, iteration)
        proposal_seconds += time.perf_counter()-stamp
        active_history.append(int(pending.sum()))
        if pending.any():
            stamp = time.perf_counter()
            nr, nj = evaluator(trials, pending)
            evaluation_seconds += time.perf_counter()-stamp
            stamp = time.perf_counter()
            accept(states, bounds, residual, jacobian, cost, damping, increase, refresh,
                   normal, gradient, trials, step_scaled, steps, done, success, reasons,
                   pending, nr, nj, evaluations)
            update_seconds += time.perf_counter()-stamp
        if done.all():
            break
    reasons[~done] = 4  # unchanged iteration limit, not scientific rejection
    profile = dict(wall_seconds=time.perf_counter()-started, initialization_seconds=init_seconds,
                   proposal_seconds=proposal_seconds, update_seconds=update_seconds,
                   trial_evaluation_seconds=evaluation_seconds, active_history=active_history,
                   evaluation_detail=dict(evaluator.profile))
    return dict(states=states, residual=residual, jacobian=jacobian, success=success,
                cost=cost, reasons=reasons, iterations=iterations, evaluations=evaluations,
                profile=profile)
