"""Replay stage certificates, coordinate identities, tree error and all costs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from flint import arb,ctx

from ..local_resolution import grid_plan
from ..local_hierarchy import transfer_cost
from ..verify_local_resolution import replay_proof
from ..reception_flow import CircularObserver
from .resolution import stage_phase,PERMUTATION


def replay_grid(grid,proof,observer,max_step=.03125):
    order=tuple(grid['permutation'])
    if proof['start']!=proof['stop'] and order!=PERMUTATION:
        raise ValueError('A nonidentity transfer uses the wrong stage order')
    for axis,(choice,saved) in enumerate(zip(grid['axes'],grid['certificates'])):
        current=stage_phase(proof,observer,axis,saved['ellipse'],grid['time_range'],
                            grid['frequency_bounds'],max_step=max_step)
        for key in ['kind','selected_parent_axes','phase_derivative_bound','hybrid_halfwidth',
                    'child_real_halfwidths','child_imaginary_halfwidths','child_coordinate_derivative_majorant']:
            if current.get(key)!=saved.get(key):
                raise ValueError(f'Stage certificate changed: axis {axis}, {key}')
        if (choice['phase_derivative_bound']<current['phase_derivative_bound']
                or choice['hybrid_halfwidth']<current['hybrid_halfwidth']
                or choice['neighborhood_ellipse']!=current['ellipse']):
            raise ValueError('Grid uses an understated stage bound')
        n,p=choice['cells'],choice['order']
        if (type(n) is not int or n<1 or type(p) is not int or not 1<=p<=64
                or choice['axis_nodes']!=n*p):
            raise ValueError('Invalid cell/order count')
        r,r0=arb(choice['ellipse']),arb(current['ellipse'])
        if not (choice['ellipse']<=current['ellipse'] or (
                arb(n)*((r0+1/r0)/2-1)>=(r+1/r)/2-1 and arb(n)*(r0-1/r0)>=r-1/r)):
            raise ValueError('Cell ellipse leaves the certified field neighborhood')
    current=grid_plan(grid['axes'],grid['tolerance'],permutation=order)
    for key in ['nodes','cells','orders','shape','error_bound','input_error_amplification']:
        if current[key]!=grid[key]:
            raise ValueError(f'Grid count/error changed: {key}')
    return current


def replay_tree(record,observer):
    times,K=record['times'],record['frequency_bounds']
    options=record['certificate_options']
    step=record['local_options']['max_step']
    identity=replay_proof(record['root_identity_certificate'],observer,options)
    root=record['root']
    if root['center']!=identity['parent_center'] or root['widths']!=identity['parent_widths']:
        raise ValueError('Root identity certificate has the wrong box')
    counts={'grids':0,'nodes':0,'entries':0,'geometry_state_evaluations_bound':0,'sum_intermediate_nodes':0}
    def visit(node,proof):
        grid=replay_grid(node['grid'],proof,observer,step)
        counts['grids']+=1;counts['nodes']+=grid['nodes']
        if (node['grid']['time_range']!=[times[node['first']],times[node['last']-1]]
                or node['grid']['frequency_bounds']!=K):
            raise ValueError('Grid does not cover its exposure interval/bandwidth')
        errors=[]
        if not node['children']:
            if node['last']-node['first']!=1 or node['epoch']!=times[node['first']]:
                raise ValueError('Leaf is not an instantaneous exposure')
            for i in (0,1,2,5):
                if grid['cells'][i]!=1 or grid['orders'][i]!=1:
                    raise ValueError('Instantaneous leaf unexpectedly resolves a constant direction')
            errors=[arb(0)]
        else:
            if len(node['children'])!=2 or len(node['edges'])!=2:
                raise ValueError('Incomplete tree')
            left,right=node['children']
            if (left['first']!=node['first'] or left['last']!=right['first'] or right['last']!=node['last']):
                raise ValueError('Exposure tree omits or overlaps frames')
            for child,edge in zip(node['children'],node['edges']):
                proof=replay_proof(edge['certificate'],observer,options)
                if (proof['parent_center']!=node['center'] or proof['parent_widths']!=node['widths']
                        or proof['child_center']!=child['center'] or proof['child_widths']!=child['widths']
                        or proof['start']!=node['epoch'] or proof['stop']!=child['epoch']
                        or tuple(edge['permutation'])!=PERMUTATION):
                    raise ValueError('Transfer does not match its certified geometry/order')
                err=visit(child,proof)
                cg=child['grid']
                errors.append(arb(cg['error_bound'])+arb(cg['input_error_amplification'])*err)
                cost=transfer_cost(grid['shape'],cg['shape'],cg['orders'],permutation=PERMUTATION)
                if cost!=edge['cost']:
                    raise ValueError('Intermediate grid count mismatch')
                counts['entries']+=cost['stored_entries']
                counts['geometry_state_evaluations_bound']+=cost['rebase_state_evaluations_upper_bound']
                counts['sum_intermediate_nodes']+=sum(cost['intermediate_nodes'])
        error=max(errors)
        if not error<=arb(node['nodal_error_bound_per_source_l1'])<=arb(node['nodal_budget']):
            raise ValueError('Subtree error budget failed')
        return arb(node['nodal_error_bound_per_source_l1'])
    error=visit(root,identity)
    stats=record['stats']
    bound=arb(root['grid']['error_bound'])+arb(root['grid']['input_error_amplification'])*error
    if not bound<=arb(stats['uniform_error_per_source_l1'])<=arb(stats['requested_tolerance']):
        raise ValueError('Continuous field error budget failed')
    checks={'nodes':'total_stored_field_nodes','entries':'total_transfer_entries',
            'geometry_state_evaluations_bound':'total_geometry_state_evaluations_bound',
            'sum_intermediate_nodes':'sum_intermediate_nodes'}
    for key,stat in checks.items():
        if counts[key]!=stats[stat]:
            raise ValueError(f'Tree total is inconsistent: {stat}')
    return counts


def verify(path):
    begin=perf_counter()
    root=Path(__file__).resolve().parents[3]
    artifact=json.loads(path.read_text())
    if artifact['full_objective_complete']:
        raise ValueError('Full objective is not certified by this construction')
    for name,digest in artifact['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Source changed: {name}')
    ctx.prec=artifact['arb_precision_bits']
    observer=CircularObserver()
    if 'local' in artifact:
        item=artifact['local'];proof=replay_proof(item['certificate'],observer)
        grid=replay_grid(item['grid'],proof,observer)
        counts={'grids':1,'nodes':grid['nodes']}
    else:
        counts=replay_tree(artifact['hierarchy' if 'hierarchy' in artifact else 'numerical']['plan'],observer)
    return {**counts,'source_files_hashed':len(artifact['source_sha256']),
            'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'seconds':perf_counter()-begin,'full_objective_complete':False,
            'floating_point_evaluation_certified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('file',type=Path)
    path=parser.parse_args().file
    report=verify(path)
    path.with_name('verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
