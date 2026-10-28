"""Fixed follow-up priorities for large-object prospects; all alternatives saved."""
import hashlib,json
from pathlib import Path
import numpy as np
import distant_followup as follow

def select(records):
    mag=np.load('tracklet-magnitude.npy');prior=set(json.loads(Path('prior-followup-digests.json').read_text()))
    selected={};stats={}
    for lane,limit in [('bright_bound_distant',256),('bound_distant',128),('retrograde',64),('hyperbolic_mode',64),('NEO_mode',64),('other',64)]:
        choices=[]
        for row in records:
            if lane in ('bright_bound_distant','bound_distant'):
                modes=[(i,m) for i,m in enumerate(row['modes']) if m.get('range_au',0)>=30 and m.get('eccentricity',2)<1]
            else:
                modes=[(i,m) for i,m in enumerate(row['modes']) if lane=='other' or lane in follow.lane_tags(m)]
            if not modes:continue
            mi,m=min(modes,key=lambda x:(x[1]['rms_arcsec'],x[0]));key=hashlib.sha256(json.dumps([row['tracklet_indices'],m['epoch_mjd_tdb'],m['state_au_au_per_day']],separators=(',',':')).encode()).hexdigest()
            if key in prior:continue
            vals=mag[row['tracklet_indices']];vals=vals[np.isfinite(vals)]
            radius=float(np.linalg.norm(m['state_au_au_per_day'][:3]));distance=m.get('range_au',0)
            H=float(np.median(vals)-5*np.log10(radius*distance)) if len(vals)>=2 and radius>0 and distance>0 else float('inf')
            if lane=='bright_bound_distant' and not np.isfinite(H):continue
            order=((H,) if lane=='bright_bound_distant' else ())+(-row.get('span_days',0),-len(row['tracklet_indices']),m['rms_arcsec'],key)
            choices.append((order,key,row,mi,m))
        stats[lane]=dict(available=len(choices),quota=limit)
        for _,key,row,mi,m in sorted(choices,key=lambda x:x[0])[:limit]:
            if key not in selected:selected[key]=dict(digest=key,row=row,mode=m,source_mode=mi,roles=[])
            selected[key]['roles'].append(lane)
    return list(selected.values()),stats

if __name__=='__main__':
    follow.select=select
    follow.main()
