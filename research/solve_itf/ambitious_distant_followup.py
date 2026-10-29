"""Diverse multi-mode, same-band size-proxy priority; finite planetary recovery."""
import hashlib,json
from pathlib import Path
import numpy as np
import distant_followup as follow

def select(records):
    mag=np.load('tracklet-band-magnitude.npy');band=np.load('tracklet-band.npy');prior=set(json.loads(Path('prior-followup-digests.json').read_text()))
    # Connected overlap families prevent hundreds of variants exhausting priority.
    parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for row in records:
        ids=row['tracklet_indices']
        for x in ids[1:]:parent[find(x)]=find(ids[0])
    pools={k:[] for k in ('large_proxy','kuiper','outer','remote','retrograde','hyperbolic_mode','NEO_mode','general')}
    for row in records:
        ids=row['tracklet_indices'];family=find(ids[0])
        for mi,m in enumerate(row['modes']):
            d=m.get('range_au',0);r=float(np.linalg.norm(m['state_au_au_per_day'][:3]));e=m.get('eccentricity',2)
            key=hashlib.sha256(json.dumps([ids,m['epoch_mjd_tdb'],m['state_au_au_per_day']],separators=(',',':')).encode()).hexdigest()
            if key in prior:continue
            hs=[]
            for b in np.unique(band[ids]):
                vals=mag[ids][(band[ids]==b)&np.isfinite(mag[ids])]
                if b!=b'' and len(vals)>=2 and r>0 and d>0:hs.append((float(np.median(vals)-5*np.log10(r*d)),b.decode()))
            H,measured_band=min(hs,default=(float('inf'),''))
            lanes=['general']+[x for x in follow.lane_tags(m) if x in pools]
            if d>=30 and e<1:lanes.append('kuiper' if d<80 else 'outer' if d<300 else 'remote')
            if d>=30 and e<1 and H<=5:lanes.append('large_proxy')
            for lane in set(lanes):
                rank=((H,) if lane=='large_proxy' else ())+(-row.get('span_days',0),-len(ids),m['rms_arcsec'],key)
                pools[lane].append((rank,key,family,row,mi,m,H,measured_band))
    quotas=dict(large_proxy=192,kuiper=64,outer=96,remote=96,retrograde=32,hyperbolic_mode=32,NEO_mode=32,general=32)
    chosen={};stats={};queues={}
    for lane,limit in quotas.items():
        family_counts={};queue=[]
        for item in sorted(pools[lane],key=lambda x:x[0]):
            family=item[2]
            if family_counts.get(family,0)>=4:continue
            family_counts[family]=family_counts.get(family,0)+1;queue.append(item)
            if len(queue)>=limit:break
        queues[lane]=queue;stats[lane]=dict(available_modes=len(pools[lane]),quota=limit,selected=len(queue),max_per_overlap_family=4)
    # Round-robin lanes; if stage deadline intervenes no lane is placed at the tail.
    for i in range(max(map(len,queues.values()),default=0)):
        for lane,queue in queues.items():
            if i>=len(queue):continue
            _,key,family,row,mi,m,H,b=queue[i]
            if key not in chosen:chosen[key]=dict(digest=key,row=row,mode=m,source_mode=mi,roles=[],family=family,absolute_magnitude_proxy=H if np.isfinite(H) else None,measured_band=b)
            chosen[key]['roles'].append(lane)
    return list(chosen.values()),stats

if __name__=='__main__':
    follow.select=select;follow.main()
