"""Experimental cheap-start triage followed by exact all-observation correction.

No fit from this cascade is evidence that a seed is impossible. Unresolved seeds
and unsearched initializations remain deferred to the full-grid queue.
"""
import math
import numpy as np
from .fit import fit_chain, gauss_seeds, check_deadline
from .derivatives import solve_lm
from .kernels import ARCSEC


# Versioned campaign settings; these select initial guesses, never final acceptance.
CAMPAIGN_SETTINGS = dict(pilot_evaluations=12, promote_rms=300., ranges=(),
    harmonic_gate=100., compiled_gauss=True, protect_fast=True, rescue=True)


class PreparedSeed:
    def __init__(self, values):
        self.values = values

    def get(self, indices):
        return self.values


def fit_cascade(geometry, indices, *, pilot_evaluations=6, promote_rms=30.,
                ranges=(.03, 1., 40., 1000.), harmonic_gate=None, compiled_gauss=False,
                include_gauss=True, protect_fast=False, rescue=False, radial_speeds=(0.,), deadline=np.inf):
    values = geometry.get(indices)
    t, u, observers, labels = values
    if len(t) < 6 or len(np.unique(labels)) < len(indices):
        return None, dict(disposition='invalid_observation_set', pilots=0)
    ra = np.arctan2(u[:, 1], u[:, 0])
    east = np.column_stack([-np.sin(ra), np.cos(ra), np.zeros(len(t))])
    north = np.cross(u, east)
    middle = len(t)//2; epoch = float(t[middle]); dt = t[-1]-t[0]
    if not include_gauss:
        starts = []
    elif compiled_gauss:
        from .harmonic import gauss_starts
        starts = gauss_starts(t, u, observers)
    else:
        starts = gauss_seeds(t, u, observers)
    harmonic_count = 0
    if harmonic_gate is not None:
        from .harmonic import range_starts
        states, epochs, costs = range_starts(t, observers, u, east, north, harmonic_gate)
        starts.extend(zip(states, epochs)); harmonic_count = len(states)
    apparent_rate = float(np.linalg.norm(u[-1]-u[0])*ARCSEC/dt)
    ranges = list(ranges)
    if protect_fast and apparent_rate > 1000.:
        ranges += [r for r in (.01, .03, .1, .3) if r not in ranges]
    for distance in ranges:
        r = observers[middle]+distance*u[middle]
        v = (observers[-1]-observers[0]+distance*(u[-1]-u[0]))/dt
        for radial in radial_speeds:
            starts.append((np.r_[r, v+radial*u[middle]], epoch))
    promoted = []; best = np.inf; attempted = 0
    for state, seed_epoch in starts:
        check_deadline(deadline)
        if not np.isfinite(state).all():
            continue
        x, residual, jacobian, success = solve_lm(state, t-seed_epoch, observers, u, east, north, .3, pilot_evaluations)
        attempted += 1
        rms = .3*math.sqrt(float(residual@residual)/len(t))
        best = min(best, rms)
        if np.isfinite(rms) and rms <= promote_rms:
            promoted.append((x, seed_epoch))
    result = None
    if promoted:
        result = fit_chain(PreparedSeed(values), indices, extra_seeds=promoted,
            start_policy='provided', deadline=deadline)
        if result is not None:
            result.update(orbit_start_policy='cascade', full_grid_refinement_pending=True,
                cascade_pilot_evaluations=pilot_evaluations, cascade_promote_rms_arcsec=promote_rms)
    rescued = None
    dynamic_fallback = harmonic_gate is not None and harmonic_count == 0 and bool(promoted)
    if result is None and rescue and (dynamic_fallback or (promoted and best <= 3.5) or (harmonic_count and best >= 1e10)):
        # Near-threshold failures and invalid harmonic states merit additional
        # initializations; neither is evidence against the association.
        result, rescued = fit_cascade(PreparedSeed(values), indices,
            pilot_evaluations=pilot_evaluations, promote_rms=promote_rms,
            ranges=(.01, .03, .1, .3, 1.) if dynamic_fallback else (.03, 1., 40., 1000.),
            radial_speeds=(0., -.05, .05) if dynamic_fallback else (0.,),
            include_gauss=False, deadline=deadline)
    counts = dict(disposition='accepted_fit' if result else 'deferred_full_grid',
        pilots=attempted, harmonic_starts=harmonic_count,
        protected_fast_starts=len(ranges) if protect_fast and apparent_rate > 1000. else 0,
        promoted_starts=len(promoted), best_pilot_rms_arcsec=best if np.isfinite(best) else None)
    if rescued is not None:
        counts.update(rescue_pilots=rescued['pilots'], rescue_accepted=result is not None,
                      dynamic_fallback=dynamic_fallback)
    return result, counts
