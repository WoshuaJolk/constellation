"""Resume fitting from hash-verified graph and prospectively frozen seed queue."""
import argparse,json,gzip,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from gpu_scaling_benchmark import geometry
from ambitious_distant_adapter import DistantGPUAdapter
from distant_search import write
from magictools.itf.catalogue import sha256
from magictools.itf import search,broad

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True)
    g=geometry();edges=np.load(a.output/'edges.npy');seeds=np.load(a.output/'seeds.npy');freeze=json.loads((a.output/'seed-freeze.json').read_text());graph=json.loads(Path('distant-results/graph/summary.json').read_text())
    assert sha256(a.output/'edges.npy')==freeze['graph_sha256']
    assert sha256(a.output/'seeds.npy')==freeze['sha256']
    keys=[tuple(map(int,k)) for k in seeds];adj=defaultdict(list)
    for x,y,score in edges:
        adj[int(x)].append((float(score),int(y)));adj[int(y)].append((float(score),int(x)))
    for links in adj.values():links.sort()
    remaining=json.loads(Path('absolute-deadline.json').read_text())['deadline']-time.time()
    seconds=min(8700.,remaining-2700.)
    if seconds<60:raise RuntimeError('Insufficient lifetime for fit plus follow-up/evidence reserves')
    cfg=search.SearchConfig(workers=20,max_gap_days=1095,max_edges=len(edges),max_seeds=len(keys),seed_neighbors=0,broad_search=True,seconds=seconds)
    write(a.output/'fit-admission.json',dict(actual_fit_seconds=seconds,seed_freeze_sha256=sha256(a.output/'seed-freeze.json'),followup_and_return_reserve_seconds=2700))
    (a.output/'graph').mkdir(exist_ok=True);write(a.output/'graph/summary.json',graph)
    def fixed_edges(tracks,c,deadline,progress,counters):
        counters.update(edges_retained=len(edges),full_catalogue_scan_complete=graph['complete']);return edges
    def fixed_seeds(e,tracks,c,counters,deadline):
        counters.update(freeze['counts']);counters['seeds_retained']=len(keys);return keys,adj
    search.candidate_edges=fixed_edges;search.make_seeds=fixed_seeds;broad.augment_edges=lambda original,*args,**kwargs:original
    with DistantGPUAdapter() as adapter:
        result=search.solve(g.catalogue,g.sites,cfg,geometry=g,batch_fitter=adapter)
        write(a.output/'batch-profile.json',dict(receipts=adapter.receipts,pipeline=adapter.pipeline))
    with gzip.open(a.output/'candidates.json.gz','wt',compresslevel=1) as f:json.dump(result,f,allow_nan=False)
    write(a.output/'fit-summary.json',{k:v for k,v in result.items() if k!='candidates'}|{'candidates':len(result['candidates'])})

if __name__=='__main__':main()
