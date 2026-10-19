"""Run the declared catalogue searches; inputs/results are portable and hashed."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from astropy import units as u
from astropy.coordinates import CartesianDifferential, Galactocentric, SkyCoord
from astropy.table import Table
from astropy.io.votable import parse
import numpy as np
from scipy.integrate import solve_ivp

from magictools.ejector_gaia import number
from magictools.ejector_search import run_search
from magictools.galactic import GalacticPotential, KMS_TO_PC_MYR
from magictools.stellar_inputs import icrs_to_galactocentric

HERE = Path(__file__).resolve().parent
EPOCH = 'Gaia DR3 J2016.0'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def source_catalogue():
    path = HERE/'data/hunt_reffert_clusters.vot.gz'
    document = parse(path)
    table = document.get_first_table().to_table(use_names_over_ids=True)
    counts = [i.value for r in document.resources for i in r.infos if i.name == 'matches']
    if len(table) != 7167 or counts != ['7167']:
        raise ValueError('VizieR ASU response must contain all 7,167 published clusters')
    objects, excluded = [], []
    for row in table:
        name = str(row['Name'])
        values = [number(row, k) for k in ['RA_ICRS', 'DE_ICRS', 'dist50', 'pmRA', 'pmDE', 'RV']]
        radius = number(row, 'rtpc')
        if not np.isfinite(values).all() or values[2] <= 0 or not np.isfinite(radius) or radius <= 0:
            excluded.append(dict(id=name, reason='missing/invalid 6D nominal state or tidal radius',
                                 missing_rv=not np.isfinite(values[-1])))
            continue
        metadata = dict(catalogue='Hunt & Reffert 2023, VizieR J/A+A/673/A114 (retrieved 2026-10-19)',
                        aliases=str(row['AllNames']), type=str(row['Type']), members=int(row['N']),
                        rv_count=int(row['n_RV']), current_epoch=EPOCH)
        for key in ['e_RV', 's_RV', 'r50pc', 'rtpc', 'logAge16', 'logAge50', 'logAge84', 'CST']:
            v = number(row, key); metadata[key] = v if np.isfinite(v) else None
        objects.append(dict(id=name, icrs=values, match_radius_pc=radius, metadata=metadata))
    return dict(schema_version=1, epoch_label=EPOCH,
                description='All 7,167 rows of Hunt & Reffert cluster catalogue; nominal published distance median, proper motions, Gaia mean RV, published tidal radius. No quality cut on cluster RV; counts/errors retained.',
                coverage=dict(input_rows=len(table), included=len(objects), excluded=len(excluded),
                    source_catalogue='https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/673/A114',
                    complete_for_all_visible_sources=False,
                    limitations=['Open/moving/globular groups found in Gaia; not all visible objects or all clusters.',
                                 'No selection function, extinction correction, or luminosity-completeness limit.',
                                 'No missing RV is invented. Excluded rows remain in the ledger.',
                                 'Nominal trajectories omit cluster distance/kinematic, centroid and dynamical uncertainty.',
                                 'A tidal-radius overlap is an association lead, not a compact source or birth-site identification.']),
                objects=objects, excluded=excluded)


def base_spec():
    loader = importlib.util.spec_from_file_location('previous_ejector_campaign', HERE.parent/'ejector_fit/run.py')
    previous = importlib.util.module_from_spec(loader); loader.loader.exec_module(previous)
    stars, _, ledger = previous.catalogue()
    spec = json.loads((HERE.parent/'ejector_fit/data/gaia_pair.json').read_text())
    spec.pop('encounters'); spec.pop('notes', None)
    spec['objects'] = stars
    spec['epoch_label'] = EPOCH
    spec['catalogue_description'] = 'All 18 of the 23 Bhat et al. 2022 Table A.1 literature runaways passing the pre-existing Gaia DR3 crossmatch, RUWE<=1.4 and valid-likelihood cuts. Literature selection is not homogeneous; no all-sky or population-calibrated discovery claim.'
    spec['search'] = dict(group_size=3, radius_pc=1., lookback_bounds_myr=[.1, 6.], report_chi2_max=25., max_hypotheses=5000, workers=3)
    spec['fit']['starts_myr'] = [[.5], [2.5], [5.]]
    spec['fit']['max_iterations'] = 120
    spec['controls'] = dict(method='proper_motion_rotation', n_controls=19, seed=20261020)
    spec['archive_output'] = 'observed'
    spec['counterparts'] = 'counterparts.json'
    return spec, ledger


def independent_flow(states, duration, potential):
    def rhs(t, y):
        s = y.reshape(-1, 6)
        return np.c_[s[:, 3:]*KMS_TO_PC_MYR, potential.acceleration(s[:, :3])/KMS_TO_PC_MYR].ravel()
    solution = solve_ivp(rhs, [0., duration], np.asarray(states).ravel(), method='DOP853', rtol=1e-12, atol=1e-12)
    if not solution.success: raise RuntimeError(solution.message)
    return solution.y[:, -1].reshape(-1, 6)


def injection(spec, counterparts):
    rng = np.random.default_rng(26061983)
    cluster = next(r for r in counterparts['objects'] if r['id'] == 'Collinder_69')
    present = icrs_to_galactocentric(np.array([cluster['icrs']]), spec['frame'])
    potential = GalacticPotential(spec['potential'])
    age = 2.345
    past = independent_flow(present, -age, potential)[0]
    ejected = np.tile(past, (3, 1))
    ejected[:, 3:] += np.array([[100., 50., 20.], [-80., 110., -30.], [60., -90., 50.]])
    now = independent_flow(ejected, age, potential)
    f = spec['frame']
    frame = Galactocentric(galcen_distance=f['galcen_distance_pc']*u.pc, z_sun=f['z_sun_pc']*u.pc,
        galcen_v_sun=CartesianDifferential(np.array(f['galcen_v_sun_kms'])*u.km/u.s),
        galcen_coord=SkyCoord(ra=266.4051*u.deg, dec=-28.936175*u.deg), roll=0*u.deg)
    sky = SkyCoord(x=now[:, 0]*u.pc, y=now[:, 1]*u.pc, z=now[:, 2]*u.pc,
        v_x=now[:, 3]*u.km/u.s, v_y=now[:, 4]*u.km/u.s, v_z=now[:, 5]*u.km/u.s,
        representation_type='cartesian', differential_type='cartesian', frame=frame).icrs
    cov = np.diag(np.array([.03, .04, .04, 1.5])**2)
    objects = []
    for i, identifier in enumerate(['trial_8e04', 'trial_2a91', 'trial_70bb']):
        truth = np.array([1000/sky.distance[i].to_value(u.pc), sky.pm_ra_cosdec[i].to_value(u.mas/u.yr),
                          sky.pm_dec[i].to_value(u.mas/u.yr), sky.radial_velocity[i].to_value(u.km/u.s)])
        observed = rng.multivariate_normal(truth, cov)
        objects.append(dict(id=identifier, ra_deg=float(sky.ra[i].deg), dec_deg=float(sky.dec[i].deg),
                            mean=observed.tolist(), covariance=cov.tolist(), notes='Synthetic measured source; truth withheld from the search algorithm.'))
    result = deepcopy(spec)
    result['objects'] += objects
    rng.shuffle(result['objects'])
    result['archive_output'] = 'injection'
    result['controls']['n_controls'] = 0
    result['catalogue_description'] += ' Plus three independently forward-integrated synthetic ejecta with Gaussian observation noise; unknown membership in a shuffled 21-object catalogue. This is a recovery test, not observed stars.'
    truth = dict(star_ids=sorted(r['id'] for r in objects), lookback_myr=age, source='Collinder_69', origin_pc=past[:3].tolist(),
                 error_model='Independent synthetic Gaussian 4-observable errors, fixed angular positions',
                 injection_selection='Added after literature selection, so this tests conditional search recovery, not survey detection completeness.')
    return result, truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=['observed', 'injection', 'historical'], required=True)
    parser.add_argument('--controls', type=int)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    counterparts = source_catalogue()
    spec, ledger = base_spec()
    spec['search']['workers'] = args.workers
    truth = None
    if args.case == 'injection':
        spec, truth = injection(spec, counterparts)
        dump(args.output/'injection_truth.json', truth)
    elif args.case == 'historical':
        historical = json.loads((HERE.parent/'ejector_fit/data/hip1997_triple.json').read_text())
        spec['objects'] = historical['objects']
        spec['epoch_label'] = historical['epoch_label']
        spec['archive_output'] = 'historical'
        spec['catalogue_description'] = 'Previously known historical Orion triple, Hoogerwerf et al. 2001 Table 3; diagonal printed uncertainties, fixed angles. Positive benchmark, not a blinded discovery or updated astrometric claim.'
        # Retain the source catalogue at its real epoch; the benchmark does not
        # request counterpart matching, avoiding a fabricated angular conversion.
        spec.pop('counterparts')
    if args.controls is not None: spec['controls']['n_controls'] = args.controls
    dump(args.output/'counterparts.json', counterparts)
    dump(args.output/'selection_ledger.json', ledger)
    manifest = args.output/(args.case+'.json')
    dump(manifest, spec)
    result = run_search(manifest)
    if truth is not None:
        recovered = [c for c in result['candidates'] if sorted(c['star_ids']) == truth['star_ids']]
        dump(args.output/'recovery.json', dict(truth=truth, recovered=bool(recovered),
            candidate=recovered[0] if recovered else None, evaluated_groups=result['hypotheses_per_catalogue']))
    print(json.dumps(dict(case=args.case, hypotheses=result['hypotheses_per_catalogue'],
        catalogues=result['catalogues_searched'], converged=result['converged_groups'],
        candidates=result['reported_candidates'], minimum_chi2=result['minimum_chi2']), indent=2), flush=True)


if __name__ == '__main__': main()
