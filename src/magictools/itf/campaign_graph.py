"""Compact, resumable full-ITF proposal graph and diverse triple enumeration."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from .kernels import njit
from .fit import check_deadline


def save_array(path, array):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False)+'\n')
    temporary.replace(path)


def csr_graph(edges, size):
    """Undirected CSR, each adjacency ordered by score then neighbor ID."""
    n = len(edges)
    source = np.empty(2*n, np.int32)
    target = np.empty(2*n, np.int32)
    score = np.empty(2*n, np.float64)
    source[:n], source[n:] = edges[:, 0], edges[:, 1]
    target[:n], target[n:] = edges[:, 1], edges[:, 0]
    score[:n] = score[n:] = edges[:, 2]
    order = np.lexsort((target, score, source))
    target, score = target[order], score[order]
    offsets = np.r_[0, np.cumsum(np.bincount(source, minlength=size), dtype=np.int64)]
    return offsets, target, score


@njit(cache=True)
def _heap_push(values, left, right, length, value, i, j):
    k = length
    while k > 0:
        parent = (k-1)//2
        if values[parent] < value or (values[parent] == value and
                (left[parent] < i or (left[parent] == i and right[parent] <= j))):
            break
        values[k], left[k], right[k] = values[parent], left[parent], right[parent]
        k = parent
    values[k], left[k], right[k] = value, i, j
    return length+1


@njit(cache=True)
def _heap_pop(values, left, right, length):
    value, i, j = values[0], left[0], right[0]
    length -= 1
    v, a, b = values[length], left[length], right[length]
    k = 0
    while 2*k+1 < length:
        child = 2*k+1
        if child+1 < length and (values[child+1], left[child+1], right[child+1]) < (values[child], left[child], right[child]):
            child += 1
        if (v, a, b) <= (values[child], left[child], right[child]):
            break
        values[k], left[k], right[k] = values[child], left[child], right[child]
        k = child
    values[k], left[k], right[k] = v, a, b
    return value, i, j, length


@njit(cache=True)
def enumerate_triples(offsets, neighbors, scores, times, identity, start, end, per_anchor, max_visits):
    """Exact lowest-score eligible wedges per anchor; bounded visits are reported.

    All pair rows in the score-sorted adjacency are merged lazily. No quadratic
    materialization. Output includes a per-anchor ordinal for diversity selection.
    """
    triples = np.empty(((end-start)*per_anchor, 3), np.int32)
    weights = np.empty((end-start)*per_anchor, np.float64)
    ordinal = np.empty((end-start)*per_anchor, np.int32)
    count, visited, capped = 0, 0, 0
    for middle in range(start, end):
        lo, hi = offsets[middle], offsets[middle+1]
        degree = hi-lo
        if degree < 2:
            continue
        values = np.empty(degree, np.float64)
        left, right = np.empty(degree, np.int32), np.empty(degree, np.int32)
        length = 0
        for i in range(degree-1):
            length = _heap_push(values,left,right,length,scores[lo+i]+scores[lo+i+1],i,i+1)
        accepted, examined = 0, 0
        while length and accepted < per_anchor and examined < max_visits:
            value,i,j,length = _heap_pop(values,left,right,length)
            examined += 1
            a,b = neighbors[lo+i],neighbors[lo+j]
            if j+1 < degree:
                length = _heap_push(values,left,right,length,scores[lo+i]+scores[lo+j+1],i,j+1)
            if abs(times[a]-times[middle]) < .3 or abs(times[b]-times[middle]) < .3 or abs(times[a]-times[b]) < .3:
                continue
            if identity[a] == identity[middle] or identity[b] == identity[middle] or identity[a] == identity[b]:
                continue
            x,y,z = a,middle,b
            if x > y:x,y = y,x
            if y > z:y,z = z,y
            if x > y:x,y = y,x
            triples[count,0],triples[count,1],triples[count,2] = x,y,z
            weights[count],ordinal[count] = value,accepted
            count += 1
            accepted += 1
        visited += examined
        if length and (accepted == per_anchor or examined == max_visits):
            capped += 1
    return triples[:count],weights[:count],ordinal[:count],visited,capped


def distinct_triples(triples, scores, ordinals):
    """Keep best exploration ordinal, then score, for every exact observation set."""
    order = np.lexsort((scores,ordinals,triples[:,2],triples[:,1],triples[:,0]))
    t = triples[order]
    keep = np.r_[True,np.any(t[1:] != t[:-1],axis=1)] if len(t) else np.zeros(0,bool)
    take = order[keep]
    return triples[take],scores[take],ordinals[take]


def graph_partition(job):
    """Each deterministic day partition commits independently; restart skips it."""
    directory, cache, config_dict, partition, hypotheses = job
    from .catalogue import Catalogue
    from .search import SearchConfig, _candidate_edges_serial
    from .broad import _curved_scan, unique_edges, best_edges
    directory = Path(directory)
    name = f'part-{partition[0]:03d}'
    marker = directory/(name+'.json')
    if marker.exists():
        return json.loads(marker.read_text())
    tracks = Catalogue(cache).tracks
    config = SearchConfig(**config_dict)
    started = time.monotonic()
    linear_counts = {}
    linear = _candidate_edges_serial(tracks,config,np.inf,lambda _:None,linear_counts,partition)
    arrays = [np.load(directory/(key+'.npy'),mmap_mode='r') for key in
              ('positions','velocities','days','earth','earth_velocity')]
    curved,curved_counts = _curved_scan(tracks,*arrays,config,np.inf,partition,hypotheses=hypotheses)
    # Integer edge keys replace millions of Python tuple/set allocations.
    base = np.uint64(len(tracks))
    keys = linear[:,0].astype(np.uint64)*base+linear[:,1].astype(np.uint64)
    ck = curved[:,0].astype(np.uint64)*base+curved[:,1].astype(np.uint64)
    novel = curved[~np.isin(ck,keys)]
    budget = config.max_edges
    nc = min(len(novel),max(budget//2,budget-len(linear)))
    nl = min(len(linear),budget-nc)
    edges = np.concatenate((best_edges(linear,nl),best_edges(novel,nc)))
    save_array(directory/(name+'.npy'),edges)
    result = dict(partition=list(partition),seconds=time.monotonic()-started,edges=len(edges),
        linear=linear_counts,curved=curved_counts,linear_deferred=len(linear)-nl,curved_deferred=len(novel)-nc)
    # Retain deferred proposed pools, so later expansion doesn't rescan the sky.
    save_array(directory/(name+'-linear.npy'),linear)
    save_array(directory/(name+'-curved.npy'),curved)
    save_json(marker,result)
    return result
