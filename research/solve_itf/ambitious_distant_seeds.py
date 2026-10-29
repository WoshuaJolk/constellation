"""Freeze a balanced, finite proposal queue before any production fitting."""
import json,time,gc
from pathlib import Path
import numpy as np
from magictools.itf.catalogue import Catalogue,sha256
from magictools.itf.broad import unique_edges
from ranked_seed_stream import make_ranked_seeds
from ambitious_distant_graph import LANES,REFERENCES
from distant_search import write

def prioritize(keys,mag,band,distance):
    a=np.asarray(keys,dtype=np.int64).reshape(-1,3)
    if not len(a):return a
    m=mag[a];b=band[a];same=(b[:,0]==b[:,1])&(b[:,1]==b[:,2])&(b[:,0]!=b'')
    h=np.mean(m,axis=1)-10*np.log10(distance)
    bright=same&np.isfinite(m).all(axis=1)&(h<=5)
    # Fixed one bright-priority slot followed by two ordinary score slots.
    priority=np.flatnonzero(bright);ordinary=np.flatnonzero(~bright)
    result=[];i=j=0
    while i<len(priority) or j<len(ordinary):
        if i<len(priority):result.append(priority[i]);i+=1
        result.extend(ordinary[j:j+2]);j+=2
    return a[np.asarray(result,dtype=np.int64)]

def main():
    root=Path('distant-results/production');root.mkdir(exist_ok=True);graph=Path('distant-results/graph');start=time.monotonic()
    cat=Catalogue('catalogue');mag=np.load('tracklet-band-magnitude.npy');band=np.load('tracklet-band.npy');lanes=[];stats={};parts=[]
    # All earlier proposal queues are excluded, including their unfinished suffix.
    # That exclusion is explicit; the new queue seeks new combinations, not a complete census.
    prior=np.load('prior-seeds.npy',mmap_mode='r');seen={tuple(map(int,x)) for x in prior}
    for lane in LANES:
        edges=np.load(graph/(lane+'.npy'));parts.append(edges)
        keys,adj,counts=make_ranked_seeds(edges,cat.tracks,4000000);del adj
        fresh=[k for k in keys if k not in seen];del keys
        lanes.append(prioritize(fresh,mag,band,REFERENCES[lane]));stats[lane]=dict(counts,new_after_prior_exclusion=len(fresh));del fresh;gc.collect()
        write(root/'seed-progress.json',dict(status='ranking',lanes=stats,seconds=time.monotonic()-start))
    chosen=[];origins=[];positions=[0]*len(lanes)
    while len(chosen)<20000000:
        advanced=False
        for j,rows in enumerate(lanes):
            while positions[j]<len(rows):
                key=tuple(map(int,rows[positions[j]]));positions[j]+=1
                if key in seen:continue
                seen.add(key);chosen.append(key);origins.append(j);advanced=True;break
            if len(chosen)>=20000000:break
        if not advanced:break
    np.save(root/'seeds.npy',np.asarray(chosen,dtype=np.int64).reshape(-1,3));np.save(root/'seed-lanes.npy',np.asarray(origins,dtype=np.uint8))
    edges=unique_edges(np.concatenate(parts));np.save(root/'edges.npy',edges)
    write(root/'seed-freeze.json',dict(sha256=sha256(root/'seeds.npy'),graph_sha256=sha256(root/'edges.npy'),prior_sha256=sha256('prior-seeds.npy'),prior_count=len(prior),
        requested_cap=20000000,selected=len(chosen),lanes=stats,selected_per_lane={lane:origins.count(i) for i,lane in enumerate(LANES)},seconds=time.monotonic()-start,
        counts={'seeds_retained':len(chosen)},policy='Up to 4M ranked per lane; same-band H proxy <=5 gets one slot per two ordinary; round-robin six lanes; exclude every prior supplied seed, including uncomputed prior suffix; no fit outcomes used.'))

if __name__=='__main__':main()
