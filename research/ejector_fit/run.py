#!/usr/bin/env python3
"""Reproduce historical benchmarks and a two-cluster search of published runaways.

No network needed: raw Gaia/SIMBAD replies, queries, literature RVs and
benchmark manifests are retained alongside this script. Every fit is saved.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.table import Table
from magictools.ejector_fit import run_fit
from magictools.ejector_gaia import read_gaia, number

HERE = Path(__file__).resolve().parent


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def catalogue():
    raw = read_gaia(HERE/'data/gaia_runaways.vot')
    supplement = read_gaia(HERE/'data/gaia_orion.vot')
    literature = json.loads((HERE/'data/literature_rvs.json').read_text())
    objects, ledger = [], []
    for rv in literature:
        hip = rv['hip']
        rows = raw[np.array(raw['hip'], int) == hip]
        if hip == 26241:
            rows = supplement[supplement['source_id'] == 3017190546641718144]
        entry = dict(hip=hip, included=False)
        ledger.append(entry)
        if len(rows) != 1:
            entry['reason'] = 'missing_or_ambiguous_Gaia_crossmatch'
            continue
        row = rows[0]
        entry.update(source_id=str(row['source_id']))
        ruwe = number(row, 'ruwe')
        if not np.isfinite(ruwe) or ruwe > 1.4:
            entry['reason'] = 'RUWE_missing_or_above_1.4'
            continue
        pars = ['parallax', 'pmra', 'pmdec']
        mean = [number(row, p) for p in pars]+[rv['rv']]
        err = np.array([number(row, p+'_error') for p in pars]+[rv['rv_error']])
        corr = np.eye(4)
        for i in range(3):
            for j in range(i+1, 3): corr[i, j] = corr[j, i] = number(row, pars[i]+'_'+pars[j]+'_corr')
        if not np.isfinite(np.r_[mean, err, corr.ravel()]).all() or mean[0] <= .1 or np.any(err <= 0) or np.linalg.eigvalsh(corr).min() <= 0:
            entry['reason'] = 'invalid_observation_likelihood'
            continue
        entry.update(included=True, ruwe=ruwe, radial_velocity_source='Bhat et al. 2022 Table A.1; not independently revalidated here')
        objects.append(dict(id=f'HIP_{hip}', ra_deg=float(row['ra']), dec_deg=float(row['dec']), mean=mean,
                            covariance=(corr*err[:, None]*err[None, :]).tolist(),
                            notes=f'Gaia DR3 {row["source_id"]}; RUWE {ruwe}; fixed angles. Bhat et al. 2022 RV. Binary/systemic RV validity requires individual follow-up.'))
    # Cluster mean kinematics and printed errors: Bhat et al. 2022 Table A.2.
    # The omitted cross-correlations and fixed angular centroid are explicit
    # approximations. The cluster has finite extent, not a point-like ejector.
    positions = Table.read(HERE/'data/cluster_positions.csv', format='ascii.csv')
    clusters = []
    for name, match, mean, err in [
        ('Collinder_69', 'Cl Collinder   69', [2.46, 1.19, -2.12, 29.1], [.005, .02, .02, 3.24]),
        ('ONC', 'NAME Orion Nebula Cluster', [2.48, 1.51, .5, 21.8], [.04, .11, .12, 6.6])]:
        row = positions[positions['main_id'] == match][0]
        clusters.append(dict(id=name, ra_deg=float(row['ra']), dec_deg=float(row['dec']), mean=mean,
                             covariance=np.diag(np.array(err)**2).tolist(),
                             notes='Bhat et al. 2022 Table A.2 diagonal covariance; fixed SIMBAD J2000 angular centroid used at J2016.0 (sub-milliparsec epoch displacement). No centroid/cluster-structure uncertainty fitted.'))
    return objects, clusters, ledger


def make_manifests(output):
    output.mkdir(parents=True, exist_ok=True)
    specs = {}
    for path in sorted((HERE/'data').glob('*_pair.json')) + sorted((HERE/'data').glob('*_triple.json')):
        specs[path.stem] = json.loads(path.read_text())
    base = copy.deepcopy(specs['gaia_pair'])
    stars, clusters, ledger = catalogue()
    for star in stars:
        for cluster in clusters:
            spec = copy.deepcopy(base)
            spec['objects'] = [star, cluster]
            # For two members, centroid radius R means separation <= 2R.
            separation = 7. if cluster['id'] == 'Collinder_69' else 5.
            spec['encounters'] = [dict(id='cluster_passage', members=[star['id'], cluster['id']],
                                       lookback_bounds_myr=[.1, 6.], radius_pc=separation/2)]
            spec['notes'] = 'All retained literature runaways tested against both supplied clusters. Targeted two-cluster experiment, no all-sky discovery calibration. Lookback 0.1--6 Myr. No stellar-age likelihood.'
            specs[star['id']+'_'+cluster['id']] = spec
    # Published candidates, jointly fitted with one shared source measurement.
    names = ['HIP_22061', 'HIP_29678', 'HIP_27204']
    repeated = copy.deepcopy(base)
    repeated['objects'] = [next(r for r in stars if r['id'] == name) for name in names]+[clusters[0]]
    repeated['encounters'] = [dict(id=name, members=[name, 'Collinder_69'], lookback_bounds_myr=[.1, 6.], radius_pc=3.5) for name in names]
    repeated['fit']['starts_myr'] = [[1., 1., 1.], [2., 2., 2.], [3., 3., 3.], [4., 4., 4.], [5., 5., 5.], [.5, 2., 4.]]
    repeated['notes'] = 'Joint repeated-launch compatibility test of three previously published candidate Collinder 69 origins. One shared cluster state, distinct times, <=7 pc star-cluster separation. Not a compact ejector discovery.'
    specs['repeated_Collinder69'] = repeated
    exact = copy.deepcopy(repeated)
    for event in exact['encounters']: event['radius_pc'] = 0.
    exact['notes'] = 'Exact cluster-center sensitivity test of the same published candidates. Failure to fit is not exclusion.'
    specs['repeated_Collinder69_exact'] = exact
    calibrated = copy.deepcopy(repeated)
    for star in calibrated['objects'][:-1]: star['mean'][0] += .017
    calibrated['notes'] += ' Sensitivity: +0.017 mas stellar parallax shift, cluster unchanged; not official source-dependent Gaia calibration.'
    specs['repeated_Collinder69_zpt'] = calibrated
    compact = copy.deepcopy(base)
    compact['objects'] = [next(r for r in stars if r['id'] == 'HIP_27204'), clusters[0]]
    compact['encounters'] = [dict(id='mu_Col_Collinder69_exact', members=['HIP_27204', 'Collinder_69'], lookback_bounds_myr=[.1, 6.], radius_pc=0.)]
    compact['fit']['starts_myr'] = [[1.], [2.], [3.], [4.], [5.]]
    compact['notes'] = 'Exact cluster-center sensitivity of a published candidate; constrained source point, no causal claim.'
    specs['mu_Col_Collinder69_exact'] = compact
    network = copy.deepcopy(base)
    network['objects'] = [next(r for r in stars if r['id'] == name) for name in ['HIP_24575', 'HIP_27204']]+clusters
    network['encounters'] = [
        dict(id='AE_Aur_ONC', members=['HIP_24575', 'ONC'], lookback_bounds_myr=[.1, 6.], radius_pc=2.5),
        dict(id='mu_Col_ONC', members=['HIP_27204', 'ONC'], lookback_bounds_myr=[.1, 6.], radius_pc=2.5),
        dict(id='mu_Col_Collinder69', members=['HIP_27204', 'Collinder_69'], lookback_bounds_myr=[.1, 6.], radius_pc=3.5)]
    network['fit']['starts_myr'] = [[1., 1., 1.], [2., 2., 2.], [3., 3., 3.], [4., 4., 4.], [2.5, 2.5, 3.5]]
    network['notes'] = 'Test whether one fitted mu Col trajectory can pass both proposed source clusters. A consistent multi-encounter solution would demonstrate that proximity alone does not identify its birthplace. Previously published candidate associations, not new discoveries.'
    specs['Orion_network'] = network
    paths = []
    for name, spec in specs.items():
        path = output/(name+'.json')
        dump(path, spec); paths.append(path)
    dump(output/'selection_ledger.json', ledger)
    return paths


def execute(task):
    path, output = task
    result = run_fit(path)
    result['manifest'] = path.name
    # Paths are portable in committed artifacts; hashes remain exact.
    for row in result['provenance']['inputs']: row['path'] = 'manifests/'+Path(row['path']).name
    dump(output/(path.stem+'.json'), result)
    best = result['best_fit']
    return dict(name=path.stem, status=result['status'], chi2=best['chi2'] if best else None,
                events=best['events'] if best else [], active_bounds=best['active_bound_parameter_indices'] if best else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    paths = make_manifests(args.output/'manifests')
    result_dir = args.output/'fits'; result_dir.mkdir(exist_ok=True)
    core_hash = hashlib.sha256((HERE.parents[1]/'src/magictools/ejector_fit.py').read_bytes()).hexdigest()
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for row in pool.map(execute, [(p, result_dir) for p in paths]):
            rows.append(row)
            print(row['name'], row['status'], row['chi2'], flush=True)
    dump(args.output/'campaign.json', dict(results=sorted(rows, key=lambda r: r['name']), full_objective_complete=False, breakthrough=False,
        core_sha256=core_hash,
        data_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((HERE/'data').iterdir()) if p.is_file()}))


if __name__ == '__main__': main()
