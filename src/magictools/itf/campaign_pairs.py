"""Supplemental pair bootstrap bank; never count a pair as an accepted chain."""
import copy
import json
from pathlib import Path
import numpy as np
from .catalogue import Catalogue,sha256
from .campaign_graph import save_array,save_json
from .broad import unique_edges,best_edges


def prepare(args):
    root=args.output;directory=root/'pairs';directory.mkdir(exist_ok=True)
    requested=getattr(args,'pair_seeds',10000)
    if (directory/'seeds.json').exists():
        saved=json.loads((directory/'seeds.json').read_text())
        if saved.get('requested_pair_budget',10000)!=requested:raise ValueError('Pair budget changed; use a separate campaign')
        return directory
    c=Catalogue(args.cache);graph=root/'graph'
    offsets=np.load(graph/'offsets.npy',mmap_mode='r');degree=np.diff(offsets)
    parts=[]
    parallax=graph/'parallax.npy'
    if parallax.exists():parts.append(np.load(parallax))
    for path in sorted(graph.glob('part-[0-9][0-9][0-9].npy')):
        edges=np.load(path,mmap_mode='r');a,b=edges[:,0].astype(int),edges[:,1].astype(int)
        keep=(degree[a]<=1)&(degree[b]<=1)&(c.tracks['count'][a]+c.tracks['count'][b]>=6)
        if np.any(keep):parts.append(edges[keep])
    edges=unique_edges(np.concatenate(parts)) if parts else np.empty((0,3))
    # The supplementary bank is separately counted, outside the 7m triple target.
    target=requested or len(edges)
    pairs=best_edges(edges,target)[:,:2].astype(np.int32) if target else np.empty((0,2),dtype=np.int32)
    save_array(directory/'seeds.npy',pairs)
    save_json(directory/'seeds.json',dict(target=target,requested_pair_budget=requested,supplied=len(pairs),unique_available=len(edges),
        deferred_pairs=max(0,len(edges)-len(pairs)),
        hypothesis_only=True,seed_sha256=sha256(directory/'seeds.npy'),
        source_graph_sha256=sha256(graph/'graph-identity.json')))
    atlas=directory/'fit-geometry'
    if not atlas.exists():atlas.symlink_to((root/'fit-geometry').resolve(),target_is_directory=True)
    return directory


def run(args):
    from .campaign_fit import run as fit
    from .campaign_export import export
    local=copy.copy(args);local.output=prepare(args)
    fit(local);export(local)
