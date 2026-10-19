#!/usr/bin/env python3
"""Reproduce tighter-radius tests, close-pair distances and grouped controls."""
import argparse
import json
from pathlib import Path
import numpy as np
from magictools.ejector import run_ejector
from research.ejector_catalogue.tighten_radius import tighten
from research.ejector_catalogue.pair_control import run as pair_control


def followups(workspace):
    workspace=Path(workspace).resolve()
    for name,parent in [('baseline','final-baseline'),('measured','measured-radial-control'),
                        ('broad-rv','final-broad-rv'),('long-distance','final-long-distance'),
                        ('heavy-galaxy','heavy-galaxy'),('injection','full-injection')]:
        output=workspace/f'{name}-1pc.json'
        if not output.exists():tighten(workspace/parent/'tracebacks',1.,output)
    source=workspace/'quality-refinement'
    strict=workspace/'quality-1pc'
    if not strict.exists():
        strict.mkdir()
        (strict/'catalogue.npz').hardlink_to(source/'catalogue.npz')
        spec=json.loads((source/'search.json').read_text())
        spec['search']['radius_pc']=1.;spec['search']['n_controls']=0
        (strict/'search.json').write_text(json.dumps(spec,indent=2)+'\n')
        result=run_ejector(strict/'search.json')
        (strict/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    if not (strict/'tracebacks/complete.json').exists():raise ValueError('Incomplete tighter-radius refinement')
    grouped=workspace/'quality-pair-control'
    if not grouped.exists():pair_control(source,grouped)
    if not (grouped/'tracebacks/complete.json').exists():raise ValueError('Incomplete grouped control')
    archive=source/'tracebacks'
    x=np.load(archive/'states.npy',mmap_mode='r');ids=np.load(archive/'source_ids.npy').tolist();times=np.load(archive/'lookback_myr.npy')
    a,b=[ids.index(key) for key in ['2364812051194680448','2364812055490127488']]
    reports=[]
    for key in ['6395326738923518976','4979220221218050432']:
        index=ids.index(key)
        difference=x[:,:,index,:3]-(x[:,:,a,:3]+x[:,:,b,:3])/2
        start= difference[:-1]; change=np.diff(difference,axis=0)
        u=np.clip(-np.sum(start*change,axis=-1)/np.sum(change*change,axis=-1),0,1)
        separation=np.linalg.norm(start+u[...,None]*change,axis=-1)
        closest=separation.min(axis=0);indices=separation.argmin(axis=0)
        ages=times[indices]+u[indices,np.arange(len(indices))]*np.diff(times)[indices]
        reports.append(dict(third_star=key,closest_to_pair_center_percentiles_pc=np.quantile(closest,[.16,.5,.84]).tolist(),
                            closest_age_percentiles_myr=np.quantile(ages,[.16,.5,.84]).tolist(),fraction_within_1pc=float(np.mean(closest<1))))
    (workspace/'closest_pair_encounters.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('workspace',type=Path);a=p.parse_args();followups(a.workspace)
