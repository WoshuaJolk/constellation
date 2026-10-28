"""Shared exact-JPL observer atlas, reusing previously prepared station offsets."""
import fcntl
import json
from pathlib import Path
import time
import warnings
import erfa
from astropy.time import Time
from astropy.utils import iers
import numpy as np
from .catalogue import Catalogue, sha256
from .campaign_graph import save_json
from .campaign_geometry_fast import FastAtlasGeometry


class ExactEarth:
    def __init__(self,path):
        from jplephem.spk import SPK
        self.kernel=SPK.open(str(path))

    def earth(self,t):
        jd=np.asarray(t)+2400000.5
        p=self.kernel[0,3].compute(jd)+self.kernel[3,399].compute(jd)-self.kernel[0,10].compute(jd)
        return np.moveaxis(p/149597870.7,0,-1)


def prepare_jpl_atlas(cache,observatories,campaign,ephemeris_path,progress=print):
    campaign=Path(campaign); source=campaign/'fit-geometry';target=campaign/'fit-geometry-jpl'
    target.mkdir(exist_ok=True)
    with (target/'geometry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        c=Catalogue(cache); original=json.loads((source/'identity.json').read_text())
        if original['input']!=c.manifest or original['observatories_sha256']!=sha256(observatories):
            raise ValueError('Source observer atlas differs')
        if not (source/'complete.json').exists():raise ValueError('Complete source atlas first')
        identity=dict(input=c.manifest,observatories_sha256=sha256(observatories),schema=1,
            kind='Exact JPL Earth plus source atlas station offsets',source_identity=original,
            ephemeris_sha256=sha256(ephemeris_path),implementation_sha256=sha256(Path(__file__)))
        meta=target/'identity.json'
        if meta.exists() and json.loads(meta.read_text())!=identity:raise ValueError('JPL geometry identity changed')
        save_json(meta,identity)
        if (target/'complete.json').exists():return json.loads((target/'complete.json').read_text())
        for name in ('times','unit','ready'):
            p=target/(name+'.npy')
            if not p.exists():p.symlink_to((source/p.name).resolve())
        old=np.load(source/'observers.npy',mmap_mode='r');times=np.load(source/'times.npy',mmap_mode='r')
        outpath=target/'observers.npy'
        observers=np.lib.format.open_memmap(outpath,mode='r+' if outpath.exists() else 'w+',dtype=np.float64,shape=old.shape)
        checkpoints=target/'chunks.json'; done=set(json.loads(checkpoints.read_text())) if checkpoints.exists() else set()
        exact=ExactEarth(ephemeris_path);started=time.monotonic()
        try:
            for lo in range(0,len(c.obs),65536):
                if lo in done:continue
                hi=min(lo+65536,len(c.obs))
                with iers.conf.set_temp('auto_download',False),warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    tdb=Time(c.obs['t'][lo:hi],format='mjd',scale='utc').tdb
                    original_earth,_=erfa.epv00(tdb.jd1,tdb.jd2)
                # Subtract the identical ERFA Earth used by Geometry.prepare.
                # Station rotation, UTC->TDB, and direction reduction are unchanged.
                observers[lo:hi]=(old[lo:hi]-original_earth['p'])+exact.earth(times[lo:hi])
                observers.flush();done.add(lo);save_json(checkpoints,sorted(done))
                progress(f'JPL observer atlas {hi:,}/{len(c.obs):,}; {time.monotonic()-started:.1f}s')
        finally:exact.kernel.close()
        result=dict(observations=len(c.obs),additional_bytes=int(observers.nbytes),invocation_seconds=time.monotonic()-started,
            prepared=json.loads((source/'complete.json').read_text())['prepared'])
        save_json(target/'complete.json',result);return result


class JPLAtlasGeometry(FastAtlasGeometry):
    def __init__(self,cache,observatories,campaign,ephemeris,ephemeris_path):
        campaign=Path(campaign)
        super().__init__(cache,observatories,campaign/'fit-geometry-jpl',campaign/'observation-identities')
        saved=json.loads((campaign/'fit-geometry-jpl/identity.json').read_text())
        if saved['ephemeris_sha256']!=sha256(ephemeris_path):raise ValueError('JPL atlas uses a different ephemeris')
        self.ephemeris=ephemeris
