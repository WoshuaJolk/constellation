"""Longer known-object development seasons, with unchanged fit acceptance."""
import argparse,json
from pathlib import Path
import distant_search as original
from distant_epoch_graph import graph

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    original.graph=graph
    solve=original.solve_graph
    original.solve_graph=lambda *args,**kwargs:solve(*args,**dict(kwargs,max_gap_days=365))
    original.controls(a.output)
    rows=json.load(__import__('gzip').open(a.output/'candidates.json.gz','rt'))['candidates']
    truth=json.loads(Path('control-truth.json').read_text());owners=truth['owners']
    predictions=json.loads((a.output/'later-predictions.json').read_text())['results'];results=[]
    for case in truth['selection']:
        pure=[r for r in rows if all(owners[i]==case['identity'] for i in r['tracklet_indices'])]
        if not pure:results.append(dict(identity=case['identity'],name=case['name'],status='not_recovered'));continue
        r=min(pure,key=lambda r:(-r['span_days'],-len(r['tracklet_indices']),r['rms_arcsec'],r['tracklet_indices']))
        mi,m=min(enumerate(r['modes']),key=lambda x:(x[1]['rms_arcsec'],x[0]))
        pp=next((x for x in predictions if x.get('identity')==case['identity'] and x.get('membership')==r['tracklet_indices'] and x.get('mode')==mi),None)
        results.append(dict(identity=case['identity'],name=case['name'],status='recovered',tracklets=len(r['tracklet_indices']),span_days=r['span_days'],rms_arcsec=m['rms_arcsec'],range_au=m['range_au'],eccentricity=m['eccentricity'],inclination_deg=m['inclination_deg'],later=pp,selected_membership=r['tracklet_indices'],selected_mode=mi))
    # Prospective engineering sanity gate only. No final fit or production cut.
    gate=any(r['identity'] in ('20000','q8219') and r['status']=='recovered' and r['span_days']>=60 and 25<=r['range_au']<=100 and r['eccentricity']<1 and r['later'] and r['later'].get('rms_arcsec') is not None and r['later']['rms_arcsec']<=30 for r in results)
    original.write(a.output/'orbit-recovery.json',dict(cases=results,engineering_gate_passed=gate,rule='At least Varuna or KV42: pure >=60day chain, nominal bound orbit at25–100AU, first reserved later-tracklet RMS<=30arcsec; representative chosen only from fitted data. Engineering prerequisite for this launch, not association acceptance/calibrated coverage.',limitations='Opened development controls with longer input seasons than prior run; not an isolated algorithm comparison. Every later prediction retained, no best-after-outcome mode selection.'))

if __name__=='__main__':main()
