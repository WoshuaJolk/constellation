#!/usr/bin/env python3
"""Reproduce the Gaia/web-data campaign. Raw download is a separate explicit step.

Use --prepare-only to build audited posteriors; --scenario selects a sensitivity
run. Each scenario searches the entire usable selection, never a ranked TOP N.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from magictools.ejector import run_ejector
from magictools.ejector_gaia import run_prepare, read_gaia, gaia_joint_draws
from magictools.galactic import GalacticPotential, integrate_stars

HERE = Path(__file__).resolve().parent
FRAME = dict(galcen_distance_pc=8122., z_sun_pc=20.8, galcen_v_sun_kms=[12.9,245.6,7.78])
POTENTIAL = dict(components=[dict(type='plummer',mass_msun=5e9,scale_pc=500.),
                            dict(type='miyamoto_nagai',mass_msun=6.8e10,a_pc=3000.,b_pc=280.),
                            dict(type='log_halo',speed_kms=180.,core_pc=12000.)])
PRIORS = dict(distance_length_pc=1350., missing_rv_mean_kms=0., missing_rv_sigma_kms=300.,
              parallax_zero_point_mas=-.017, shared_zero_point_sigma_mas=.01)


def dump(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run(args):
    output = args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    scenario = args.scenario
    priors = PRIORS.copy(); potential = json.loads(json.dumps(POTENTIAL));frame = FRAME.copy()
    if scenario == 'broad_rv': priors['missing_rv_sigma_kms'] = 600.
    if scenario == 'long_distance': priors['distance_length_pc'] = 5000.
    if scenario == 'heavy_galaxy':
        for component in potential['components']:
            if 'mass_msun' in component: component['mass_msun'] *= 1.2
            if 'speed_kms' in component: component['speed_kms'] *= np.sqrt(1.2)
        frame['galcen_v_sun_kms'] = [12.9,255.6,7.78]
    source = (args.data or HERE/'data/gaia_fast.fits.gz').resolve()
    prep = output/'prepare.json'
    specification = dict(schema_version=1,input=str(source),output='catalogue.npz',n_samples=args.samples,
                         seed=20261019,frame=frame,priors=priors)
    if prep.exists() and json.loads(prep.read_text()) != specification:
        raise ValueError('Existing preparation differs; choose a new output directory')
    if not prep.exists(): dump(prep, specification)
    start=time.perf_counter()
    if not (output/'catalogue.npz').exists():
        audit=run_prepare(prep)
    else:
        audit=json.loads((output/'catalogue.audit.json').read_text())
    print('prepared',scenario,audit['included_rows'],'of',audit['input_rows'],'seconds',round(time.perf_counter()-start,1),flush=True)
    if args.prepare_only: return
    catalogue=output/'catalogue.npz'
    injection_ids=[]
    if scenario == 'measured_rv':
        with np.load(catalogue) as data:
            keep=np.array([row['radial_velocity']=='measurement' for row in audit['ledger'] if row['included']])
            np.savez_compressed(output/'measured.npz',states_pc_kms=data['states_pc_kms'][:,keep],
                                source_ids=data['source_ids'][keep],control_strata=data['control_strata'][keep])
        catalogue=output/'measured.npz'
    if scenario == 'injection':
        with np.load(catalogue) as data:
            states=data['states_pc_kms'].copy(); ids=data['source_ids'].copy(); strata=data['control_strata'].copy()
        # Three narrow but nonzero clouds, unknown to the search. Full real
        # background retained. These added stars are synthetic, explicitly labeled.
        birth=np.array([[-8000.,0,20,500,250,0],[-8000.,0,20,-450,350,100],[-8000.,0,20,100,-500,-100]])
        current=integrate_stars(birth,[0,2.345],GalacticPotential(potential),.002)[-1]
        rng=np.random.default_rng(117)
        injected=current[None]+rng.normal(size=(len(states),3,6))*np.array([.3,.3,.3,.1,.1,.1])
        injection_ids=['SYNTHETIC-injection-A','SYNTHETIC-injection-B','SYNTHETIC-injection-C']
        np.savez_compressed(output/'injected.npz',states_pc_kms=np.concatenate([states,injected],axis=1),
                            source_ids=np.r_[ids,injection_ids],control_strata=np.r_[strata,['injection']*3])
        catalogue=output/'injected.npz'
        source_now=integrate_stars(np.array([[-8000.,0,20,0,220,0]]),[0,2.345],GalacticPotential(potential),.002)[-1]
        np.savez_compressed(output/'injected-counterpart.npz',states_pc_kms=np.repeat(source_now[None],len(states),axis=0),source_ids=['SYNTHETIC-source'])
    manifest=output/'search.json'
    search_specification = dict(schema_version=2,epoch_label='Gaia DR3 J2016.0',catalogue=catalogue.name,potential=potential,
                          tracebacks_output='tracebacks',search=dict(lookback_max_myr=args.lookback,time_step_myr=args.step,
                          radius_pc=args.radius,max_step_myr=args.integration_step,n_controls=args.controls,min_support=.25,seed=20261019,
                          max_pair_checks=100000000,max_group_evaluations=1000000))
    if scenario == 'injection': search_specification['counterparts'] = 'injected-counterpart.npz'
    if args.control_method == 'tangent_rotation':
        search_specification['search']['control_method'] = args.control_method
        search_specification['search']['observer_state_pc_kms'] = [-float(np.sqrt(frame['galcen_distance_pc']**2-frame['z_sun_pc']**2)),0.,frame['z_sun_pc'],*frame['galcen_v_sun_kms']]
    if manifest.exists() and json.loads(manifest.read_text()) != search_specification:
        raise ValueError('Existing search differs; choose a new output directory')
    if not manifest.exists(): dump(manifest, search_specification)
    report_path=output/'result.json'
    if not report_path.exists():
        result=run_ejector(manifest);dump(report_path,result)
    else: result=json.loads(report_path.read_text())
    ledger = {row['source_id']:row for row in audit['ledger']}
    for row in result['candidates']:
        row['members_with_rv_prior'] = [key for key in row['star_ids'] if ledger.get(key, {}).get('radial_velocity') == 'explicit_prior']
    elapsed=time.perf_counter()-start
    summary=dict(scenario=scenario,priors=priors,frame=frame,potential=potential,input_rows=audit['input_rows'],included_rows=audit['included_rows'],excluded_rows=audit['excluded_rows'],
                 rv_measurement_rows=sum(r.get('radial_velocity')=='measurement' for r in audit['ledger']),
                 rv_prior_rows=sum(r.get('radial_velocity')=='explicit_prior' for r in audit['ledger']),
                 search_summary=result['search_summary'],settings=result['settings'],numerics=result['numerics'],
                 candidates=result['candidates'],controls=result['controls'],elapsed_seconds=elapsed,
                 raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),full_objective_complete=False)
    if injection_ids:
        recovered=[r for r in result['candidates'] if set(r['star_ids'])==set(injection_ids)]
        summary['injection_validation']=dict(injected_ids=injection_ids,age_myr=2.345,recovered=bool(recovered),
                                            recovered_age_myr=recovered[0]['lookback_myr'] if recovered else None)
    dump(output/'summary.json',summary)
    print('done',scenario,'groups',len(result['candidates']),'seconds',round(elapsed,1),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--scenario',choices=['baseline','broad_rv','long_distance','heavy_galaxy','measured_rv','injection'],default='baseline')
    parser.add_argument('--samples',type=int,default=16)
    parser.add_argument('--controls',type=int,default=0)
    parser.add_argument('--control-method',choices=['velocity_permutation','tangent_rotation'],default='velocity_permutation')
    parser.add_argument('--radius',type=float,default=10.)
    parser.add_argument('--step',type=float,default=.1)
    parser.add_argument('--integration-step',type=float,default=.05)
    parser.add_argument('--lookback',type=float,default=10.)
    parser.add_argument('--prepare-only',action='store_true')
    run(parser.parse_args())
