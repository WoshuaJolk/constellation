#!/usr/bin/env python3
"""Complete tighter-radius rescan of a completed schema-2 search's saved paths.

A group satisfying r<=R at a common time/draw also satisfies R. Therefore every
qualifying tighter-radius group must be present among the complete parent R
search's groups. Restricting to their member union cannot lose a qualifying
group. Sample count, support, group size, paths and full time range stay fixed.
This does not transfer the parent's control rank fractions to the new radius.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from magictools.ejector_continuous import scan_segments
from magictools.io import _provenance


def tighten(archive,radius,output):
    archive=Path(archive).resolve();output=Path(output)
    if output.exists():raise ValueError('Output already exists')
    parent=json.loads((archive/'complete.json').read_text())
    settings=parent['settings']
    if parent['schema_version'] != 2 or parent['tool'] != 'find_ejector' or not 0 < radius <= settings['radius_pc']:
        raise ValueError('Require completed schema-2 search and a positive radius <= parent radius')
    if parent['search_summary']['qualifying_groups'] != len(parent['candidates']):
        raise ValueError('Parent candidate list is truncated')
    x=np.load(archive/'states.npy',mmap_mode='r');ids=np.load(archive/'source_ids.npy').tolist();times=np.load(archive/'lookback_myr.npy')
    members=set(key for row in parent['candidates'] for key in row['star_ids'])
    indices=[i for i,key in enumerate(ids) if key in members]
    if len(indices) >= settings['group_size']:
        rows,counts=scan_segments(((x[i][:,indices],x[i+1][:,indices]) for i in range(len(times)-1)),times,
            radius,len(x[0]),group_size=settings['group_size'],min_support=settings['min_support'],
            max_group_evaluations=settings['max_group_evaluations'],max_pair_checks=settings['max_pair_checks'])
        for row in rows:row['star_ids']=[ids[indices[i]] for i in row.pop('indices')]
    else:rows,counts=[],dict(group_segment_evaluations=0,broad_phase_pair_checks=0,segments=len(times)-1)
    result=dict(tool='find_ejector_tighter_radius',parent_catalogue_size=len(ids),searched_member_union=len(indices),
                parent_radius_pc=settings['radius_pc'],radius_pc=radius,candidates=rows,search_counts=counts,
                completeness='Full parent catalogue, unchanged sampled piecewise-linear paths/support/group size/time range; tighter events are a subset of parent events',
                significance=None,parent_numerics=parent['numerics'],
                numerical_diagnostics_within_tenth_of_new_radius=all(v<=.1*radius for v in parent['numerics'].values()),
                provenance=_provenance([archive/'complete.json',archive/'source_ids.npy',archive/'lookback_myr.npy',archive/'states.npy']))
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('parent',len(ids),'candidate member union',len(indices),'radius',radius,'groups',len(rows),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('radius',type=float);p.add_argument('output',type=Path)
    a=p.parse_args();tighten(a.archive,a.radius,a.output)
