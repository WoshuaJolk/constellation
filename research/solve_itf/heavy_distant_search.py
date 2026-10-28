"""Prospective, bounded expansion; brightness prioritizes work, never acceptance."""
import argparse, gzip, hashlib, json, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from distant_search import write
from distant_epoch_graph import graph
from magictools.itf.catalogue import sha256
from magictools.itf.search import SearchConfig, make_seeds


def select_new_seeds(keys, prior, times, magnitudes, limit):
    """Disjoint quotas, then deterministic round robin; no fit outcomes used."""
    prior = {tuple(map(int, k)) for k in prior}
    available = [k for k in keys if tuple(k) not in prior]
    bright, long, ordinary = [], [], []
    bright_quota = limit * 3 // 14  # 1.5M at 7M
    long_quota = limit * 3 // 14
    selected = set()
    for k in available:
        if len(bright) < bright_quota and sum(np.isfinite(magnitudes[i]) and magnitudes[i] <= 22.5 for i in k) >= 2:
            bright.append(k); selected.add(k)
    for k in available:
        if k not in selected and len(long) < long_quota and max(times[i] for i in k)-min(times[i] for i in k) > 120:
            long.append(k); selected.add(k)
    for k in available:
        if len(selected) >= limit: break
        if k not in selected: ordinary.append(k); selected.add(k)
    # Interleave rather than leave an entire lane until a possible time limit.
    ordered = []
    for i in range(max(map(len, (ordinary, bright, long)), default=0)):
        for lane in (ordinary, bright, long):
            if i < len(lane): ordered.append(lane[i])
    return ordered, dict(available_new=len(available), excluded_prior=len(keys)-len(available),
                        bright=len(bright), long_gap=len(long), ordinary=len(ordinary), selected=len(ordered))


def main():
    from gpu_scaling_benchmark import geometry
    from gpu_stream_adapter_v2 import StreamGPUBatchV2
    from magictools.itf import search, broad
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True)
    g=geometry();started=time.monotonic()
    options=dict(max_gap_days=365,partition_edge_cap=400000,global_edge_cap=4000000)
    edges,receipt=graph(g.catalogue,g.sites,a.output/'graph',20,480,options=options)
    cfg=SearchConfig(workers=20,max_gap_days=365,max_edges=max(1,len(edges)),max_seeds=8000000,seed_neighbors=0,broad_search=True,seconds=3000)
    counts={};keys,adj=make_seeds(edges,g.catalogue.tracks,cfg,counts,time.monotonic()+300)
    prior=np.load('prior-distant-seeds.npy');mag=np.load('tracklet-magnitude.npy')
    assert len(mag)==len(g.catalogue.tracks)
    keys,selection=select_new_seeds(keys,prior,g.catalogue.tracks['t'],mag,7000000)
    counts['seeds_retained']=len(keys);cfg.max_seeds=len(keys)
    np.save(a.output/'seeds.npy',np.asarray(keys,dtype=np.int64).reshape(-1,3))
    write(a.output/'seed-freeze.json',dict(counts=counts,selection=selection,sha256=sha256(a.output/'seeds.npy'),config=asdict(cfg),proposal_options=options,
        magnitude_caveat='Median of available nonconflicting measurements, bands unconverted. Priority only, not physical diameter or acceptance.',prior_seeds_sha256=sha256('prior-distant-seeds.npy')))
    if not keys: raise RuntimeError('No additional seeds; stop without repeating prior fits')
    def fixed_edges(tracks,c,deadline,progress,counters):
        counters.update(edges_retained=len(edges),full_catalogue_scan_complete=receipt['complete']);return edges
    def fixed_seeds(e,tracks,c,counters,deadline):counters.update(counts);return keys,adj
    search.candidate_edges=fixed_edges;search.make_seeds=fixed_seeds;broad.augment_edges=lambda original,*args,**kw:original
    with StreamGPUBatchV2() as adapter:
        result=search.solve(g.catalogue,g.sites,cfg,geometry=g,batch_fitter=adapter)
        write(a.output/'batch-profile.json',dict(receipts=adapter.receipts,pipeline=adapter.pipeline))
    with gzip.open(a.output/'candidates.json.gz','wt',compresslevel=1) as f:json.dump(result,f,allow_nan=False)
    write(a.output/'fit-summary.json',{k:v for k,v in result.items() if k!='candidates'}|{'candidates':len(result['candidates']),'total_seconds':time.monotonic()-started})

if __name__=='__main__':main()
