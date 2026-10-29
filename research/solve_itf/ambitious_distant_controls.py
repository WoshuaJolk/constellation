"""Refit fixed opened real control memberships with the additional-start adapter."""
import json,gzip
from pathlib import Path
from magictools.itf.catalogue import Catalogue
from magictools.itf.campaign_geometry import prepare_atlas
from magictools.itf.campaign_geometry_fast import prepare_observation_ids,FastAtlasGeometry
from ambitious_distant_adapter import DistantGPUAdapter
from distant_search import write

def main():
    out=Path('distant-results/controls');out.mkdir(exist_ok=True)
    prepare_atlas('control-catalogue','ObsCodes.html',out/'atlas');prepare_observation_ids('control-catalogue',out/'identities')
    g=FastAtlasGeometry('control-catalogue','ObsCodes.html',out/'atlas',out/'identities')
    cases=json.loads(Path('control-orbit-recovery.json').read_text())['cases'];selected=[r for r in cases if r['status']=='recovered']
    jobs=[(tuple(r['later']['membership']),dict(sigma=.3,rms_limit=.6,max_nfev=80)) for r in selected]
    with DistantGPUAdapter() as adapter:
        outputs=list(adapter.iter_batches(g,[jobs]));rows=outputs[0][1];profile=dict(receipts=adapter.receipts,pipeline=adapter.pipeline)
    results=[]
    for case,row in zip(selected,rows):
        results.append(dict(identity=case['identity'],name=case['name'],membership=list(jobs[len(results)][0]),row=row))
    ok=any(r['name'] in ('Varuna','2008 KV42') and r['row'] is not None and any(m['eccentricity']<1 and 25<=m['range_au']<=100 for m in r['row']['modes']) for r in results)
    write(out/'adapter-check.json',dict(engineering_gate_passed=ok,results=results,profile=profile,interpretation='Opened real fixed memberships; added-start backend check, not discovery completeness, new withheld validation or confirmation. Previous nominal prediction receipts remain separate.'))
    if not ok:raise RuntimeError('Additional-start adapter failed real distant control gate')
if __name__=='__main__':main()
