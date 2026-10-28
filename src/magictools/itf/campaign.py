"""Durable, bounded-memory seven-million-seed campaign on the current CPU host."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import fcntl
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .catalogue import Catalogue, sha256
from .fit import read_sites
from .search import SearchConfig
from .campaign_graph import (save_array, save_json, graph_partition, csr_graph,
                             enumerate_triples, distinct_triples)


def log(message):
    print(message,flush=True)


def identity(directory, name, value):
    path = directory/(name+'-identity.json')
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'{name} inputs/code/settings changed; use a new campaign directory')
    if not path.exists():save_json(path,value)


def graph(args):
    from .broad import approximate_observers, distance_hypotheses
    c=Catalogue(args.cache); sites=read_sites(args.observatories)
    directory=args.output/'graph'; directory.mkdir(exist_ok=True)
    config=replace(SearchConfig.for_profile('broad'),workers=1,neighbors_per_day=args.neighbors,
                   max_edges=(args.edges+args.partitions-1)//args.partitions)
    # Keep existing trials; add close and very distant paths without a conic gate.
    hypotheses=list(distance_hypotheses(config.max_gap_days))
    hypotheses += [(d,v,g) for d,v,g in [(0.01,0.,3),(0.01,-.02,3),(0.01,.02,3),
        (3.,-.01,28),(3.,.01,28),(50.,-.01,28),(50.,.01,28),(1000.,0.,28)]]
    provenance=dict(input=c.manifest,observatories_sha256=sha256(args.observatories),
        config=asdict(config),partitions=args.partitions,hypotheses=[list(h) for h in hypotheses],
        implementation={n:sha256(Path(__file__).with_name(n)) for n in
                        ('campaign_graph.py','broad.py','search.py')})
    identity(directory,'graph',provenance)
    names=('positions','velocities','days','earth','earth_velocity')
    if not (directory/'geometry.json').exists():
        log('Preparing full-catalogue proposal geometry')
        arrays=approximate_observers(c.tracks,sites)
        for name,array in zip(names,arrays):save_array(directory/(name+'.npy'),array)
        save_json(directory/'geometry.json',dict(complete=True,unavailable=int(np.sum(~np.isfinite(arrays[0]).all(axis=1)))))
        del arrays
    jobs=[(str(directory),str(args.cache),asdict(config),(i,args.partitions),hypotheses)
          for i in range(args.partitions) if not (directory/f'part-{i:03d}.json').exists()]
    complete=args.partitions-len(jobs)
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(graph_partition,j):j for j in jobs}
        for future in as_completed(futures):
            result=future.result();complete+=1
            log(f'Graph partitions {complete}/{args.partitions}; part {result["partition"][0]}: {result["edges"]:,} edges')
            save_json(args.output/'campaign-progress.json',dict(phase='graph',partitions_completed=complete,
                partitions_total=args.partitions,seed_target=args.seeds,updated_utc=time.time(),
                invocation_seconds=time.monotonic()-started))
    if not (directory/'csr.json').exists():
        arrays=[np.load(directory/f'part-{i:03d}.npy',mmap_mode='r') for i in range(args.partitions)]
        edges=np.concatenate(arrays)
        offsets,neighbors,scores=csr_graph(edges,len(c.tracks))
        for key,array in [('offsets',offsets),('neighbors',neighbors),('scores',scores)]:
            save_array(directory/(key+'.npy'),array)
        save_json(directory/'csr.json',dict(edges=len(edges),bytes=offsets.nbytes+neighbors.nbytes+scores.nbytes,
            reached_tracklets=int(np.sum(np.diff(offsets)>0)),complete=True))
        log(f'Compact graph ready: {len(edges):,} edges, {int(np.sum(np.diff(offsets)>0)):,} tracklets reached')


def seeds(args):
    c=Catalogue(args.cache); directory=args.output/'graph'
    if not (directory/'csr.json').exists():raise ValueError('Complete the graph stage first')
    prefix='extended-' if (directory/'extended-csr.json').exists() else ''
    identity(args.output,'seeds',dict(graph_sha256=sha256(directory/'graph-identity.json'),
        csr_sha256=sha256(directory/(prefix+'csr.json')),
        target=args.seeds,per_anchor=args.per_anchor,visits=args.visits,
        implementation=sha256(Path(__file__).with_name('campaign_graph.py'))))
    if (args.output/'seeds.json').exists():
        log('Seed bank already committed');return
    offsets,neighbors,scores=[np.load(directory/(prefix+n+'.npy'),mmap_mode='r') for n in ('offsets','neighbors','scores')]
    # Codes preserve designation equality without carrying Python strings into Numba.
    _,codes=np.unique(c.tracks['id'],return_inverse=True)
    chunks=args.output/'seed-chunks';chunks.mkdir(exist_ok=True)
    counts=dict(visited=0,capped_anchors=0)
    for start in range(0,len(c.tracks),32768):
        key=chunks/f'{start:08d}'
        if key.with_suffix('.json').exists():continue
        t,s,o,visited,capped=enumerate_triples(offsets,neighbors,scores,c.tracks['t'],codes,
            start,min(len(c.tracks),start+32768),args.per_anchor,args.visits)
        save_array(str(key)+'-triples.npy',t);save_array(str(key)+'-scores.npy',s);save_array(str(key)+'-ordinals.npy',o)
        save_json(key.with_suffix('.json'),dict(visited=int(visited),capped_anchors=int(capped),triples=len(t)))
        log(f'Enumerated anchors {min(len(c.tracks),start+32768):,}/{len(c.tracks):,}; chunk {len(t):,} triples')
    files=sorted(chunks.glob('*-triples.npy'))
    t=np.concatenate([np.load(p) for p in files])
    s=np.concatenate([np.load(str(p).replace('-triples','-scores')) for p in files])
    o=np.concatenate([np.load(str(p).replace('-triples','-ordinals')) for p in files])
    raw=len(t);t,s,o=distinct_triples(t,s,o)
    # First proposal from each anchor competes before its second proposal, etc.
    # Stable hashing of exact IDs breaks ties without chronological preference.
    hashed=(t[:,0].astype(np.uint64)*np.uint64(73856093)) ^ (t[:,1].astype(np.uint64)*np.uint64(19349663)) ^ (t[:,2].astype(np.uint64)*np.uint64(83492791))
    order=np.lexsort((hashed,s,o))[:args.seeds]
    save_array(args.output/'seeds.npy',t[order])
    for path in chunks.glob('*.json'):
        part=json.loads(path.read_text())
        for k in counts:counts[k]+=part[k]
    product=dict(target=args.seeds,supplied=len(order),unique_available=len(t),raw_wedges=raw,
        shortfall=max(0,args.seeds-len(order)),bytes=int(t[order].nbytes),counts=counts,
        ordering='per-anchor proposal ordinal, summed edge score, deterministic ID hash',
        exhaustive=False,seed_sha256=sha256(args.output/'seeds.npy'))
    save_json(args.output/'seeds.json',product);log(json.dumps(product))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['graph','seeds','fit','geometry','parallax','pairs','export'])
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--observatories',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--edges',type=int,default=10000000)
    parser.add_argument('--partitions',type=int,default=32)
    parser.add_argument('--neighbors',type=int,default=32)
    parser.add_argument('--seeds',type=int,default=7000000)
    parser.add_argument('--pair-seeds',type=int,default=10000,help='Supplemental pair limit; zero retains every eligible pair')
    parser.add_argument('--per-anchor',type=int,default=32)
    parser.add_argument('--visits',type=int,default=4096)
    parser.add_argument('--start-policy',choices=['full','scout','cascade'],default='full')
    parser.add_argument('--seconds',type=float,default=86400.)
    parser.add_argument('--seed-seconds',type=float,default=2.)
    parser.add_argument('--batch-size',type=int,default=128)
    args=parser.parse_args(argv)
    if not np.isfinite([args.seconds,args.seed_seconds]).all() or min(args.seconds,args.seed_seconds)<=0 or args.batch_size<1:
        parser.error('Fit budgets must be positive and finite')
    if any(getattr(args,k)<1 for k in ('workers','edges','partitions','neighbors','seeds','per_anchor','visits')):
        parser.error('All numeric options must be positive')
    if args.pair_seeds < 0:parser.error('Pair seed budget must be nonnegative')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=='fit':
            from .campaign_fit import run
            run(args)
        elif args.stage=='export':
            from .campaign_export import export
            export(args)
        elif args.stage=='pairs':
            from .campaign_pairs import run
            run(args)
        elif args.stage=='parallax':
            from .parallax_pairs import run
            run(args)
        elif args.stage=='geometry':
            from .campaign_geometry import prepare_atlas
            prepare_atlas(args.cache,args.observatories,args.output/'fit-geometry',log)
        else:
            globals()[args.stage](args)
    return 0
