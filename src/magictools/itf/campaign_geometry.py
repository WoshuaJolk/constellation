"""Exact fit geometry prepared once and shared read-only by CPU workers."""
import json
import fcntl
from pathlib import Path
import time
import numpy as np
from .catalogue import Catalogue, sha256
from .fit import Geometry, read_sites
from .campaign_graph import save_json


def prepare_atlas(cache, observatories, directory, progress=print):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'geometry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _prepare_atlas(cache,observatories,directory,progress)


def _prepare_atlas(cache, observatories, directory, progress):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    c=Catalogue(cache);sites=read_sites(observatories)
    identity=dict(input=c.manifest,observatories_sha256=sha256(observatories),
                  schema=1,kind='Exact existing Geometry.prepare, stored by source observation offset')
    meta=directory/'identity.json'
    if meta.exists() and json.loads(meta.read_text())!=identity:raise ValueError('Geometry atlas input differs')
    save_json(meta,identity)
    if (directory/'complete.json').exists():return
    count=len(c.obs)
    arrays=[]
    for name,shape,dtype in [('times',(count,),np.float64),('unit',(count,3),np.float64),
                              ('observers',(count,3),np.float64),('ready',(len(c.tracks),),np.uint8)]:
        path=directory/(name+'.npy')
        arrays.append(np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=dtype,shape=shape))
    times,unit,observers,ready=arrays
    g=Geometry(c,sites);started=time.monotonic();done=int(np.count_nonzero(ready))
    for code in np.unique(c.tracks['site']):
        if code not in sites:continue
        indices=np.flatnonzero((c.tracks['site']==code)&(ready==0))
        for start in range(0,len(indices),8192):
            batch=indices[start:start+8192]
            g.prepare(batch)
            for i in batch:
                track=c.tracks[i];lo=int(track['start']);hi=lo+int(track['count'])
                _,tt,uu,oo=g.prepared[int(i)]
                times[lo:hi],unit[lo:hi],observers[lo:hi]=tt,uu,oo
            for array in arrays[:3]:array.flush()
            ready[batch]=1;ready.flush();g.prepared.clear()
            done+=len(batch)
            progress(f'Exact geometry {done:,}/{len(c.tracks):,} tracklets; {time.monotonic()-started:.0f}s')
    save_json(directory/'complete.json',dict(prepared=int(np.count_nonzero(ready)),
        unavailable=int(np.count_nonzero(ready==0)),bytes=sum(a.nbytes for a in arrays)))


class AtlasGeometry(Geometry):
    def __init__(self,cache,observatories,directory):
        super().__init__(Catalogue(cache),read_sites(observatories))
        directory=Path(directory)
        if not (directory/'complete.json').exists():raise ValueError('Geometry atlas is not complete')
        saved=json.loads((directory/'identity.json').read_text())
        if saved['input']!=self.catalogue.manifest or saved['observatories_sha256']!=sha256(observatories):
            raise ValueError('Geometry atlas input differs')
        self.times,self.unit,self.observers,self.ready=[np.load(directory/(n+'.npy'),mmap_mode='r')
            for n in ('times','unit','observers','ready')]

    def prepare(self,indices,deadline=np.inf):
        return  # Atlas is immutable and complete; get/track checks missing sites.

    def track(self,index):
        if not self.ready[index]:raise ValueError('Observer geometry unavailable')
        track=self.catalogue.tracks[index];lo=int(track['start']);hi=lo+int(track['count'])
        return self.catalogue.obs[lo:hi],self.times[lo:hi],self.unit[lo:hi],self.observers[lo:hi]
