"""Exact graph-score top-K triples via merged sorted pair streams.

This changes enumeration, not ranking, tracklet eligibility or fit settings.
Duplicate triples enter at their minimum edge-pair score. Heap keys include the
canonical triple so score ties follow the original lexicographic policy.
"""
import heapq
import numpy as np
from numba import njit
from collections import defaultdict

@njit(cache=True)
def enumerate_keys(middles,starts,neighbors,scores,times,identifiers,limit):
    heap=[(0.,0,0,0,0,0,0)];heap.pop()
    for m in range(len(middles)):
        lo,hi=starts[m],starts[m+1]
        if hi-lo<2:continue
        a,b=neighbors[lo],neighbors[lo+1];mid=middles[m]
        key=sorted([a,mid,b]);heap.append((scores[lo]+scores[lo+1],key[0],key[1],key[2],m,0,1))
    heapq.heapify(heap);seen=set();out=np.empty((limit,3),np.int64);n=0;visited=0
    while heap and n<limit:
        score,k0,k1,k2,m,i,j=heapq.heappop(heap);visited+=1
        lo,hi=starts[m],starts[m+1];mid=middles[m];a,b=neighbors[lo+i],neighbors[lo+j]
        if j+1<hi-lo:
            bb=neighbors[lo+j+1];key=sorted([a,mid,bb]);heapq.heappush(heap,(scores[lo+i]+scores[lo+j+1],key[0],key[1],key[2],m,i,j+1))
        if j==i+1 and i+2<hi-lo:
            aa,bb=neighbors[lo+i+1],neighbors[lo+i+2];key=sorted([aa,mid,bb]);heapq.heappush(heap,(scores[lo+i+1]+scores[lo+i+2],key[0],key[1],key[2],m,i+1,i+2))
        if abs(times[a]-times[mid])<.3 or abs(times[b]-times[mid])<.3 or abs(times[a]-times[b])<.3:continue
        if identifiers[a]==identifiers[mid] or identifiers[b]==identifiers[mid] or identifiers[a]==identifiers[b]:continue
        key=(k0,k1,k2)
        if key in seen:continue
        seen.add(key);out[n,0]=k0;out[n,1]=k1;out[n,2]=k2;n+=1
    return out[:n],visited,len(heap)==0

def make_ranked_seeds(edges,tracks,limit):
    a=edges[:,0].astype(np.int64);b=edges[:,1].astype(np.int64)
    middle=np.r_[a,b];neighbor=np.r_[b,a];score=np.r_[edges[:,2],edges[:,2]]
    order=np.lexsort((neighbor,score,middle));middle=middle[order];neighbor=neighbor[order];score=score[order]
    mids,starts=np.unique(middle,return_index=True);starts=np.r_[starts,len(middle)].astype(np.int64)
    _,ids=np.unique(tracks['id'],return_inverse=True)
    keys,visited,exhausted=enumerate_keys(mids,starts,neighbor,score,tracks['t'],ids,limit)
    adjacent={int(m):[(float(s),int(i)) for s,i in zip(score[lo:hi],neighbor[lo:hi])] for m,lo,hi in zip(mids,starts[:-1],starts[1:])}
    return [tuple(map(int,k)) for k in keys],adjacent,{'seed_stream_pairs_visited':visited,'seed_stream_exhausted':exhausted,'seeds_retained':len(keys),'seed_search_bounded':True}
