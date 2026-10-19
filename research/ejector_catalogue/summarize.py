#!/usr/bin/env python3
"""Package completed experiment summaries and an explicitly coarse path overview."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def summarize(workspace, destination):
    destination.mkdir(parents=True,exist_ok=True)
    names=['final-baseline','final-measured','final-broad-rv','final-long-distance','heavy-galaxy',
           'measured-radial-control','quality-refinement','full-injection','web-field-final']
    campaign={}
    for name in names:
        folder=workspace/name
        if not (folder/'tracebacks/complete.json').exists():
            raise ValueError(f'{name} has no completed traceback/search marker')
        result=json.loads((folder/'summary.json').read_text())
        with gzip.open(destination/f'{name}.json.gz','wt') as stream:
            json.dump(result,stream,indent=2,allow_nan=False)
        campaign[name]={k:v for k,v in result.items() if k not in {'candidates','controls'}}
        campaign[name]['control_maxima']=result['controls']['max_search_scores']
        campaign[name]['control_method']=result['controls'].get('method','velocity_permutation')
        campaign[name]['groups_with_nonzero_lookback']=sum(r['lookback_myr']>0 for r in result['candidates'])
        campaign[name]['groups_involving_rv_prior']=sum(bool(r['members_with_rv_prior']) for r in result['candidates'])
        campaign[name]['top_candidates']=result['candidates'][:3]
    baseline=workspace/'final-baseline'
    archive=baseline/'tracebacks'
    paths=np.load(archive/'states.npy',mmap_mode='r')
    times=np.load(archive/'lookback_myr.npy')
    # This overview is not used by the search: pointwise quantiles at 1 Myr
    # spacing are not individual physical orbits and cannot recover brief events.
    chosen=np.arange(0,len(times),10)
    quantiles=np.stack([np.quantile(paths[i,...,:3],[.16,.5,.84],axis=0) for i in chosen])
    np.savez_compressed(destination/'traceback-overview.npz',lookback_myr=times[chosen],
                        source_ids=np.load(archive/'source_ids.npy'),percentiles=[.16,.5,.84],
                        position_percentiles_pc=quantiles.astype(np.float32))
    with gzip.open(destination/'baseline-input-audit.json.gz','wt') as stream:
        stream.write((baseline/'catalogue.audit.json').read_text())
    (destination/'quality-vetting.json').write_text((workspace/'quality-refinement/vetting.json').read_text())
    campaign['overview']=dict(file='traceback-overview.npz',stars=paths.shape[2],times=len(chosen),
        description='Pointwise coordinate percentiles from 16 draws at 1 Myr spacing, float32; not physical orbits or a search input. Full float64 joint-draw archive is reproducible with run.py.',
        sha256=hashlib.sha256((destination/'traceback-overview.npz').read_bytes()).hexdigest())
    for name in ['quality-pair-control','quality-1pc']:
        folder=workspace/name
        result=json.loads((folder/'tracebacks/complete.json').read_text())
        with gzip.open(destination/f'{name}.json.gz','wt') as stream:json.dump(result,stream,indent=2)
        campaign[name]=dict(targeted_not_catalogue_wide=True,search_summary=result['search_summary'],
                            candidates=result['candidates'],controls=result['controls'],numerics=result['numerics'])
    campaign['one_pc_rescans']={}
    for name in ['baseline','measured','broad-rv','long-distance','heavy-galaxy','injection']:
        source=workspace/f'{name}-1pc.json'
        result=json.loads(source.read_text())
        (destination/source.name).write_text(source.read_text())
        campaign['one_pc_rescans'][name]={key:result[key] for key in ['parent_catalogue_size','searched_member_union','radius_pc','numerical_diagnostics_within_tenth_of_new_radius']}
        campaign['one_pc_rescans'][name]['qualifying_groups']=len(result['candidates'])
    (destination/'closest_pair_encounters.json').write_text((workspace/'closest_pair_encounters.json').read_text())
    code=Path(__file__).resolve().parents[2]/'src/magictools'
    campaign['implementation_sha256']={name:hashlib.sha256((code/name).read_bytes()).hexdigest() for name in ['ejector.py','ejector_continuous.py','ejector_gaia.py','galactic.py']}
    campaign['full_objective_complete']=False
    campaign['breakthrough_claimed']=False
    (destination/'campaign.json').write_text(json.dumps(campaign,indent=2,allow_nan=False)+'\n')
    return campaign


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('workspace',type=Path);p.add_argument('destination',type=Path)
    a=p.parse_args();summarize(a.workspace,a.destination)
