#!/usr/bin/env python3
"""Sensitivity of targeted controls to keeping the G 266-60/61 pair together."""
import json
from pathlib import Path
import numpy as np
from magictools.ejector import run_ejector


def run(source,destination):
    source,destination=Path(source),Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    with np.load(source/'catalogue.npz') as data:
        ids=data['source_ids'];groups=ids.copy()
        groups[np.isin(ids,['2364812051194680448','2364812055490127488'])]='G266-pair'
        np.savez_compressed(destination/'catalogue.npz',states_pc_kms=data['states_pc_kms'],source_ids=ids,
                            control_strata=data['control_strata'],rotation_groups=groups)
    spec=json.loads((source/'search.json').read_text())
    spec['catalogue']='catalogue.npz'
    (destination/'search.json').write_text(json.dumps(spec,indent=2)+'\n')
    report=run_ejector(destination/'search.json')
    (destination/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    for row in report['candidates']:
        print(row['star_ids'],row['support'],row['search_wide_control_rank_fraction'],flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
    a=p.parse_args();run(a.source,a.destination)
