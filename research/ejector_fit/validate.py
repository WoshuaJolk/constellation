#!/usr/bin/env python3
"""Independent numerical checks and a low-power Monte Carlo counterexample."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from magictools.ejector_fit import EncounterProblem
from magictools.ejector_continuous import orbit_segments, scan_segments
from magictools.galactic import KMS_TO_PC_MYR


def check_fit(spec, result):
    best = result['best_fit']
    if best is None: return None
    problem = EncounterProblem(spec)
    by_id = {r['id']: r for r in best['fitted_objects']}
    z = np.concatenate([np.linalg.solve(f, np.array(by_id[key]['mean'])-m)
                        for key, m, f in zip(problem.ids, problem.means, problem.factors)])
    ages = [e['lookback_myr'] for e in best['events']]
    states = problem.states(np.r_[z, ages])[0]
    checks = []
    for age, event, (members, radius, _) in zip(ages, best['events'], problem.events):
        initial = states[members]
        def rhs(t, y):
            s = y.reshape(-1, 6)
            return np.c_[s[:, 3:]*KMS_TO_PC_MYR, problem.potential.acceleration(s[:, :3])/KMS_TO_PC_MYR].ravel()
        result = solve_ivp(rhs, [0, -age], initial.ravel(), method='DOP853', rtol=1e-12, atol=1e-12)
        if not result.success: raise RuntimeError(result.message)
        final = result.y[:, -1].reshape(-1, 6)
        p, v = final[:, :3], final[:, 3:]
        spread = float(np.linalg.norm(p-p.mean(axis=0), axis=1).max())
        checks.append(dict(id=event['id'], lookback_myr=age, max_centroid_radius_pc=spread,
                           radius_limit_pc=radius, constraint_excess_pc=max(0., spread-radius),
                           rms_relative_speed_kms=float(np.sqrt(np.mean(np.sum((v-v.mean(axis=0))**2, axis=1)))),
                           max_pairwise_relative_speed_kms=float(np.linalg.norm(v[:, None]-v[None], axis=-1).max())))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    args = parser.parse_args()
    directory = args.campaign
    validated = {}
    for path in sorted((directory/'fits').glob('*.json')):
        result = json.loads(path.read_text())
        spec = json.loads((directory/'manifests'/path.name).read_text())
        checked = check_fit(spec, result)
        if checked is not None:
            if any(r['constraint_excess_pc'] > spec['fit']['constraint_tolerance_pc'] for r in checked):
                raise RuntimeError('Independent trajectory check failed: '+path.name)
            validated[path.stem] = checked
    spec = json.loads((directory/'manifests/hip1997_triple.json').read_text())
    problem = EncounterProblem(spec)
    rng = np.random.default_rng(20261019)
    latent = rng.normal(size=(16, 12))
    # An explicitly stated positive-parallax truncated Gaussian experiment;
    # this is not a reproduction of the original paper's conditional sampling.
    for j in range(3):
        bad = problem.means[j, 0]+latent[:, 4*j]*problem.factors[j, 0, 0] <= .1
        while bad.any():
            latent[bad, 4*j] = rng.normal(size=bad.sum())
            bad = problem.means[j, 0]+latent[:, 4*j]*problem.factors[j, 0, 0] <= .1
    states = problem.states(np.c_[latent, np.zeros(16)])
    times = np.linspace(0., 6., 61)
    diagnostics = {}
    rows, search = scan_segments(orbit_segments(states, times, problem.potential, .02, diagnostics),
                                 times, 1., 16, min_support=.25)
    output = dict(independent_integrator='scipy DOP853, rtol=atol=1e-12', fits=validated,
        monte_carlo_counterexample=dict(objects=problem.ids, samples=16, seed=20261019,
            sampling='Independent Gaussian parallax/PM/RV measurements with parallax truncated above 0.1 mas; fixed angles; not the original paper sampling or previous Gaia distance prior.',
            radius_pc=1., min_support=.25, lookback_myr=[0., 6.], recovered_groups=len(rows),
            known_feasible_fit='hip1997_triple', numerics=diagnostics, search=search,
            conclusion='A sampled-support nondetection does not exclude a compatible common-origin solution.'),
        full_objective_complete=False)
    (directory/'validation.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print('Independently validated', len(validated), 'fits; 16-draw historical search groups:', len(rows))


if __name__ == '__main__': main()
