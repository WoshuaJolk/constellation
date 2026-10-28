"""Bounded exact-input reuse. No state rounding or orbit-family merging."""
from collections import OrderedDict,Counter
from copy import deepcopy
import hashlib,json,sqlite3,zlib
import numpy as np


class ObserverCache:
    def __init__(self,limit=100000):
        self.values=OrderedDict();self.limit=limit;self.counts=Counter()

    def earth(self,ephemeris,times):
        times=np.asarray(times);unique,inverse=np.unique(times,return_inverse=True)
        result=np.empty((len(unique),3));missing=[]
        for i,t in enumerate(unique):
            value=self.values.get(float(t))
            if value is None:missing.append(i)
            else:result[i]=value;self.values.move_to_end(float(t))
        self.counts['observer_cache_hits']+=len(unique)-len(missing)
        self.counts['observer_cache_misses']+=len(missing)
        if missing:
            values=ephemeris.earth(unique[missing]);result[missing]=values
            for i,value in zip(missing,values):self.values[float(unique[i])]=value.copy()
            while len(self.values)>self.limit:self.values.popitem(last=False)
        return result[inverse]


class IndexedObserverCache:
    """Vectorized station-independent motion observers indexed by tracklet.

    At most 256 MiB; arrays allocate pages on first use. No Python per-point LRU
    work in crowded cones. Changed input times invalidate both stencil points.
    """
    def __init__(self,size,max_bytes=256*1024**2):
        self.enabled=size*57<=max_bytes;self.counts=Counter()
        if self.enabled:
            self.ready=np.zeros(size,dtype=bool);self.times=np.full(size,np.nan);self.values=np.empty((size,2,3));self.step=None

    def pair(self,ephemeris,indices,times,step):
        indices=np.asarray(indices);times=np.asarray(times)
        if not self.enabled:
            both=ephemeris.earth(np.r_[times+step,times-step]);return both[:len(times)],both[len(times):]
        if self.step!=step:self.ready[:]=False;self.step=step
        missing=~self.ready[indices] | (self.times[indices]!=times)
        self.counts['observer_cache_hits']+=int((~missing).sum())*2;self.counts['observer_cache_misses']+=int(missing.sum())*2
        if missing.any():
            ids=indices[missing];t=times[missing];both=ephemeris.earth(np.r_[t+step,t-step])
            self.values[ids,0]=both[:len(t)];self.values[ids,1]=both[len(t):];self.times[ids]=t;self.ready[ids]=True
        return self.values[indices,0],self.values[indices,1]


class NightPredictionCache:
    def __init__(self,nights,limit_bytes=32*1024**2):
        self.nights=nights;self.limit=max(1,limit_bytes//max(1,nights*22*8));self.values=OrderedDict();self.counts=Counter()

    def apparent(self,key,trajectory,nights,times,observers):
        data=self.values.pop(key,None)
        if data is None:data=np.full((self.nights,22),np.nan)
        self.values[key]=data
        while len(self.values)>self.limit:self.values.popitem(last=False)
        missing=np.isnan(data[nights,0]);self.counts['night_prediction_hits']+=int((~missing).sum());self.counts['night_prediction_misses']+=int(missing.sum())
        if missing.any():
            # Failed/partial predictions are never entered into the cache.
            from .nbody import TrajectoryDomainError
            try:p,d,j=trajectory.apparent(times[missing],observers[missing])
            except TrajectoryDomainError as exc:
                invalid=np.zeros(len(nights),dtype=bool);invalid[np.flatnonzero(missing)]=exc.invalid
                raise TrajectoryDomainError(invalid) from exc
            data[nights[missing]]=np.column_stack((p,d,j.reshape(-1,18)))
        selected=data[nights]
        return selected[:,:3].copy(),selected[:,3].copy(),selected[:,4:].reshape(-1,3,6).copy()


def fit_key(indices,initial,epoch,options):
    # JSON preserves full Python float values. Sorting memberships is safe only
    # because Geometry.get canonicalizes observation order/deduplication.
    value=[sorted(set(map(int,indices))),np.asarray(initial,dtype=float).tolist(),float(epoch),options]
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


class ExactFitCache:
    """Per-worker LRU backed by a shared WAL store of completed fit outcomes."""
    def __init__(self,path=None,identity=None,limit=2048):
        self.values=OrderedDict();self.limit=limit;self.counts=Counter();self.db=None
        if path is not None:
            self.db=sqlite3.connect(path,timeout=30)
            self.db.execute('PRAGMA journal_mode=WAL')
            # This is a rebuildable performance cache; authoritative scientific
            # results and deferred restart inputs retain their separate stores.
            self.db.execute('PRAGMA synchronous=NORMAL')
            self.db.executescript('CREATE TABLE IF NOT EXISTS identity(id INTEGER PRIMARY KEY CHECK(id=1),value TEXT); CREATE TABLE IF NOT EXISTS fits(key TEXT PRIMARY KEY,body BLOB);')
            encoded=json.dumps(identity,sort_keys=True);old=self.db.execute('SELECT value FROM identity').fetchone()
            if old and old[0]!=encoded:raise ValueError('Fit reuse identity differs')
            if not old:
                with self.db:self.db.execute('INSERT OR IGNORE INTO identity VALUES (1,?)',(encoded,))
                if self.db.execute('SELECT value FROM identity').fetchone()[0]!=encoded:raise ValueError('Concurrent fit reuse identity differs')

    def get(self,key):
        value=self.values.pop(key,None)
        if value is None and self.db is not None:
            saved=self.db.execute('SELECT body FROM fits WHERE key=?',(key,)).fetchone()
            if saved:value=json.loads(zlib.decompress(saved[0]))
        if value is None:self.counts['exact_fit_cache_misses']+=1;return None
        self.counts['exact_fit_cache_hits']+=1;self._remember(key,value)
        return deepcopy(value)

    def _remember(self,key,value):
        self.values[key]=value
        while len(self.values)>self.limit:self.values.popitem(last=False)

    def put(self,key,value):
        self._remember(key,deepcopy(value))
        if self.db is not None:
            body=zlib.compress(json.dumps(value,allow_nan=False,separators=(',',':')).encode(),3)
            with self.db:self.db.execute('INSERT OR IGNORE INTO fits VALUES (?,?)',(key,body))
