"""Replay completed stress tiles, then verify their exact combined cover/cost."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from research.kepler_transform.verify_local_resolution import verify
from research.kepler_transform.verify_constructive_campaign import verify_rectangular_cover


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    started=time.perf_counter()
    verified={}
    verifier_hash=digest(ROOT/'research/kepler_transform/verify_local_resolution.py')
    while True:
        for i in range(16):
            checkpoint=args.directory/f'patch-{i}.json'
            if not checkpoint.exists() or i in verified:
                continue
            saved=json.loads(checkpoint.read_text())
            if saved['result']['status']!='complete':
                raise ValueError(f'Tile {i} did not complete; no combined certificate')
            wrapper={'source_sha256':saved['source_sha256'],'arb_precision_bits':100,
                     'full_objective_complete':False,
                     'hierarchy':{'case':f'stress_tile_{i}','atlas':saved['result']['atlas']}}
            folder=args.directory/f'verification-tile-{i}'
            folder.mkdir(exist_ok=True)
            artifact=folder/'results.json'
            artifact.write_text(json.dumps(wrapper,indent=2,allow_nan=False)+'\n')
            input_hash=digest(artifact)
            report_path=folder/'verification.json'
            prior=json.loads(report_path.read_text()) if report_path.exists() else {}
            if prior.get('input_sha256')==input_hash and prior.get('verifier_source_sha256')==verifier_hash:
                for name,value in saved['source_sha256'].items():
                    if digest(ROOT/name)!=value:
                        raise ValueError('Source changed after tile replay')
                report=prior
            else:
                report=verify(artifact)
                report.update({'input_sha256':input_hash,'verifier_source_sha256':verifier_hash})
                report_path.write_text(json.dumps(report,indent=2)+'\n')
            verified[i]=(saved,report)
            print(json.dumps({'tile_verified':i,**report}),flush=True)
        final_path=args.directory/'results.json'
        if len(verified)==16 and final_path.exists():
            break
        if not args.watch:
            raise ValueError('Not all 16 complete tile checkpoints are available')
        time.sleep(10)
    final=json.loads(final_path.read_text())
    atlas=final['hierarchy']['atlas']
    source=final['source_sha256']
    plans=[]
    for i in range(16):
        saved,_=verified[i]
        if saved['source_sha256']!=source:
            raise ValueError('Tile provenance does not match combined artifact')
        plans.extend(saved['result']['atlas']['plans'])
    if plans!=atlas['plans']:
        raise ValueError('Combined artifact does not contain the replayed trees')
    domains=[{'center':p['root']['center'],'widths':p['root']['widths']} for p in plans]
    if domains!=atlas['patch_domains']:
        raise ValueError('A declared patch lacks a replayed tree')
    verify_rectangular_cover(atlas['original_center'],atlas['original_widths'],domains)
    expected={k:sum(p['stats'][k] for p in plans) for k in ['root_nodes','total_stored_field_nodes',
        'total_transfer_entries','total_transfer_multiply_add_pairs','total_geometry_state_evaluations_bound',
        'sum_intermediate_nodes','coefficient_storage_bytes']}
    for key,value in expected.items():
        if value!=atlas['stats'][key]:
            raise ValueError(f'Combined count mismatch: {key}')
    error=max(p['stats']['uniform_error_per_source_l1'] for p in plans)
    if error!=atlas['stats']['uniform_error_per_source_l1'] or error>1e-8:
        raise ValueError('Combined continuous-field error budget failed')
    report={'input_sha256':digest(final_path),'verifier_source_sha256':verifier_hash,
            'aggregation_verifier_sha256':digest(Path(__file__)),
            'grids':sum(r['grids'] for _,r in verified.values()),'atlas_covers':17,
            'retained_final_patches':len(domains),'entries':expected['total_transfer_entries'],
            'source_files_hashed':len(source),
            'tile_replay_seconds':sum(r['seconds'] for _,r in verified.values()),
            'wall_seconds_including_waits':time.perf_counter()-started,
            'full_objective_complete':False,'floating_point_evaluation_certified':False}
    (args.directory/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'combined_verification':report}),flush=True)

if __name__=='__main__':
    main()
