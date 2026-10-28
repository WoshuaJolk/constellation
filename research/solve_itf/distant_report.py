"""Read-only classification, provenance and cautious size context for saved fits."""
import argparse,csv,gzip,hashlib,json,math,sqlite3,zlib
from pathlib import Path
from collections import Counter
import numpy as np
from distant_followup import lane_tags
from distant_search import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--extra-recovery',type=Path,action='append',default=[]);p.add_argument('--source-root',type=Path,default=Path('/workspace/constellation'));a=p.parse_args()
    cache=a.source_root/'.cache/solve_itf/iteration5-cold-cache';tracks=np.load(cache/'tracklets.npy',mmap_mode='r');photo=np.load(a.source_root/'.cache/solve_itf/photometry-v1/photometry.npy',mmap_mode='r')
    known=a.source_root/'runs/skyknown-bulk/20261023-import/known.sqlite';db=sqlite3.connect('file:'+str(known)+'?mode=ro',uri=True)
    metadata=json.loads(db.execute("select body from metadata where key='summary'").fetchone()[0]);assert metadata['catalogue']['source_sha256']==json.loads((cache/'manifest.json').read_text())['source_sha256'];assert len(photo)==json.loads((cache/'manifest.json').read_text())['counts']['stored_observations']
    annotations={};records=[];follow=a.run/'distant-results/followup';source=follow/'all-candidates.json.gz'
    with gzip.open(source,'rt') as f:data=json.load(f)
    baseline_keys={tuple(sorted(r['tracklet_indices'])) for r in data['candidates'] if 'general-candidates.json.gz' in r['search_sources']}
    recoveries={};recovered_rows=[]
    with gzip.open(follow/'recovery.jsonl.gz','rt') as f:
        for line in f:
            v=json.loads(line);recoveries[v['digest']]=dict(status=v['status'],rows=len(v['rows']),counts=v['counts']);recovered_rows.extend(v['rows'])
    for path in a.extra_recovery:
        with gzip.open(path,'rt') as f:
            for line in f:
                v=json.loads(line);recoveries[v['digest']]=dict(status=v['status'],rows=len(v['rows']),counts=v['counts']);recovered_rows.extend(v['rows'])
    with gzip.open(a.run/'planetary-candidates.json.gz','wt',compresslevel=1) as f:json.dump({'candidates':recovered_rows},f,allow_nan=False)
    def annotation(i):
        if i in annotations:return annotations[i]
        all_matches=set();states=[]
        for state,body in db.execute('select state,body from tracks where track_index=?',(i,)):
            row=json.loads(zlib.decompress(body)) if body is not None else {'status':('complete_no_match','matched','incomplete')[state]};states.append(row['status']);all_matches.update(row.get('matched_object_ids',[]))
        v={'matched':sorted(all_matches),'status':'matched' if all_matches else 'complete_no_match' if states and all(s=='complete_no_match' for s in states) else 'incomplete_or_unmapped'};annotations[i]=v;return v
    def describe(row):
        modes=row['modes'];mi,best=min(enumerate(modes),key=lambda x:(x[1]['rms_arcsec'],x[0]));ids=row['tracklet_indices'];times=tracks['t'][ids]
        mags=[];bands=Counter();band_values={}
        for i in ids:
            t=tracks[i];v=photo[t['start']:t['start']+t['count']];ok=np.isfinite(v['magnitude'])&(v['conflict']==0)
            bands.update(x.decode() for x in v['band'][ok]);mags.extend(v['magnitude'][ok&(v['band']==b'V')].tolist())
            for band in np.unique(v['band'][ok]):band_values.setdefault(band.decode(),[]).extend(v['magnitude'][ok&(v['band']==band)].tolist())
        med=float(np.median(mags)) if mags else None;r=float(np.linalg.norm(np.array(best['state_au_au_per_day'])[:3]));delta=best.get('range_au');H=med-5*math.log10(r*delta) if med is not None and delta and r>0 and delta>0 else None
        ann=[annotation(i) for i in ids];common=set(ann[0]['matched'])
        for v in ann[1:]:common &=set(v['matched'])
        state='common_returned_match' if common else 'partial_or_conflicting_matches' if any(v['matched'] for v in ann) else 'complete_no_match' if all(v['status']=='complete_no_match' for v in ann) else 'incomplete_or_unmapped'
        digest=hashlib.sha256(json.dumps([ids,best['epoch_mjd_tdb'],best['state_au_au_per_day']],separators=(',',':')).encode()).hexdigest()
        rec=dict(membership_key='-'.join(map(str,ids)),tracklets=' + '.join(tracks['id'][i].decode().strip() for i in ids),stations=','.join(sorted(set(tracks['site'][i].decode() for i in ids))),tracklet_count=len(ids),observations=row.get('observations'),arc_days=float(row['span_days']),rms_arcsec=best['rms_arcsec'],best_mode_index=mi,best_mode_tags=','.join(lane_tags(best)),all_mode_tags=','.join(sorted({tag for m in modes for tag in lane_tags(m)})),modes=len(modes),min_fitted_range_au=min(m.get('range_au',float('inf')) for m in modes),max_fitted_range_au=max(m.get('range_au',0) for m in modes),best_heliocentric_radius_au=r,best_geocentric_range_au=delta,local_range_sigma_au=best.get('local_uncertainty',{}).get('range_sigma_au'),local_linearization_weak=best.get('local_uncertainty',{}).get('local_linearization_weak'),local_energy_sign_resolved_3sigma=best.get('local_uncertainty',{}).get('energy_sign_resolved_3sigma'),eccentricity=best['eccentricity'],inclination_deg=best['inclination_deg'],perihelion_au=best['perihelion_au'],semimajor_axis_au=best['semimajor_axis_au'],V_count=len(mags),median_V=med,approx_H_ignoring_phase=H,conditional_diameter_km_p025=1329/.5*10**(-.2*H) if H is not None else None,conditional_diameter_km_p004=1329/.2*10**(-.2*H) if H is not None else None,photometry_bands=json.dumps(dict(bands),sort_keys=True),median_magnitudes_by_band=json.dumps({k:float(np.median(v)) for k,v in band_values.items()},sort_keys=True),known_status=state,known_common_ids=','.join(sorted(common)),best_mode_followup_status=recoveries.get(digest,{}).get('status','not_selected_as_this_mode'),search_sources=','.join(row['search_sources']),novel_membership=tuple(sorted(ids)) not in baseline_keys)
        return rec
    records=[describe(row) for row in data['candidates']]
    planetary_groups={}
    for row in recovered_rows:
        key=tuple(row['tracklet_indices'])
        if key not in planetary_groups:
            planetary_groups[key]=dict(row,search_sources=['planetary_followup']);planetary_groups[key]['modes']=list(row['modes'])
        else:planetary_groups[key]['modes'].extend(row['modes'])
    planetary_records=[describe(row) for row in planetary_groups.values()]
    with (a.run/'planetary-candidates.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(planetary_records[0]) if planetary_records else ['membership_key']);w.writeheader();w.writerows(planetary_records)
    with (a.run/'candidates.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]) if records else ['membership_key']);w.writeheader();w.writerows(records)
    # Follow-up lists are browsing aids, not acceptance labels. All rows stay in CSV.
    distant=[r for r in records if r['best_geocentric_range_au']>=30 and r['eccentricity']<1]
    distant.sort(key=lambda r:(-r['arc_days'],-r['tracklet_count'],r['rms_arcsec'],r['membership_key']))
    ambiguous_distant=[r for r in records if 'distant' in r['all_mode_tags'] and (r['best_geocentric_range_au']<30 or r['eccentricity']>=1)]
    ambiguous_distant.sort(key=lambda r:(-r['arc_days'],-r['tracklet_count'],r['rms_arcsec'],r['membership_key']))
    summary=dict(planetary_unique_memberships=len(planetary_records),planetary_best_bound_distant=sum(r['best_geocentric_range_au']>=30 and r['eccentricity']<1 for r in planetary_records),planetary_distant_examples=sorted([r for r in planetary_records if r['best_geocentric_range_au']>=30 and r['eccentricity']<1],key=lambda r:(-r['arc_days'],-r['tracklet_count'],r['rms_arcsec']))[:20],ambiguous_distant_examples=ambiguous_distant[:20],total_memberships=len(records),novel_memberships=sum(r['novel_membership'] for r in records),any_distant_mode=sum('distant' in r['all_mode_tags'] for r in records),best_bound_distant=len(distant),known_counts=dict(Counter(r['known_status'] for r in records)),new_best_bound_distant=sum(r['novel_membership'] for r in distant),planetary_output_rows=len(recovered_rows),distant_browsing_examples=distant[:20],warnings=['All classifications are properties of fitted modes, not confirmed objects.','Displayed range extrema cover saved modes, not a confidence interval or exhaustive orbit family.','Known annotations reuse an older pinned export; no-match is not discovery, older astrometric sensitivity remains.','Sizes are conditional on nominal orbit, median reported V, no phase correction and assumed albedo .04/.25; no dwarf-planet classification.','A hyperbolic osculating fit is not interstellar evidence.','The general lane is reused prior work; novel_membership means absent from that million-seed result, not from all historical campaigns.'])
    write(a.run/'candidate-summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ['warnings','distant_browsing_examples','ambiguous_distant_examples','planetary_distant_examples']},indent=2))

if __name__=='__main__':main()
