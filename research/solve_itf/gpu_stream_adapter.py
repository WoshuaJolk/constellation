"""Bounded shared-memory preparation, serial GPU ownership, overlapped CPU finish.

No changes to seed, convergence, residual, duplicate or ranking policies. CUDA
buffers are reused only after the preceding solve has downloaded owned outputs.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import io
import multiprocessing as mp
import mmap
import pickle
import shutil
import tempfile
import time
import numpy as np
from numba import cuda
from gpu_batch_host import prepare_bank, finish_group
from gpu_pipeline_adapter import checked_cpu_arrays
from gpu_prefetch_adapter import PrefetchGPUBatch, ordered_prepared
from magictools.itf.gpu_fit import ResidentWorkspace

_GEOMETRY = None
_SHARED = None


def _initialize(geometry, directory):
    global _GEOMETRY, _SHARED
    _GEOMETRY, _SHARED = geometry, directory


def _ready(_): return True


def store_prepared(value, directory):
    """Local trusted-process transport; array bytes are mapped, not pickled in IPC."""
    path = Path(tempfile.mktemp(prefix='batch-', suffix='.bin', dir=directory))
    seen = {}
    try:
        with path.open('xb') as data:
            class Writer(pickle.Pickler):
                def persistent_id(self, obj):
                    if not isinstance(obj, np.ndarray): return None
                    if obj.dtype.hasobject: raise ValueError('Object arrays are not a numerical bank')
                    if id(obj) not in seen:
                        a = np.array(obj,copy=True,order="C")
                        padding=(-data.tell())%64
                        if padding:data.write(b'\0'*padding)
                        offset = data.tell(); data.write(a.tobytes())
                        seen[id(obj)] = (offset, a.shape, a.dtype.str)
                    return seen[id(obj)]
            meta = io.BytesIO(); Writer(meta, protocol=5).dump(value)
        return dict(path=str(path), metadata=meta.getvalue(), bytes=path.stat().st_size)
    except BaseException:
        path.unlink(missing_ok=True); raise


def load_prepared(receipt):
    path = Path(receipt['path'])
    size = path.stat().st_size
    if size != receipt['bytes']: raise ValueError('Shared bank size changed')
    seen = {}
    with path.open('rb') as stream:
        mapping=mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) if size else b''
    class Reader(pickle.Unpickler):
        def persistent_load(self, item):
            offset, shape, dtype = item; dt = np.dtype(dtype)
            if dt.hasobject or offset < 0 or offset + int(np.prod(shape))*dt.itemsize > size:
                raise ValueError('Invalid shared array bounds')
            if item not in seen:
                seen[item] = (np.ndarray(shape=shape,dtype=dt,buffer=mapping,offset=offset)
                              if np.prod(shape) else np.empty(shape,dtype=dt))
            return seen[item]
    return Reader(io.BytesIO(receipt['metadata'])).load()


def _prepare(jobs):
    started=time.perf_counter()
    value=prepare_bank(_GEOMETRY,jobs,compact_geometry=True)
    receipt=store_prepared(value,_SHARED)
    return receipt,time.perf_counter()-started


class DeviceBufferPool:
    """One sequential construction lane; views never survive into another GPU solve."""
    def __init__(self):
        self.buffers=[]; self.cursor=0; self.allocations=0; self.reuses=0
    def begin(self): self.cursor=0
    def empty(self,shape,dtype=np.float64,order='C'):
        shape=(shape,) if isinstance(shape,int) else tuple(shape)
        count=int(np.prod(shape));dtype=np.dtype(dtype);i=self.cursor;self.cursor+=1
        if i==len(self.buffers): self.buffers.append(None)
        slot=self.buffers[i]
        if slot is None or slot.dtype != dtype or slot.size < count:
            capacity=max(count,int(slot.size*1.25) if slot is not None else count)
            slot=cuda.device_array(capacity,dtype);self.buffers[i]=slot;self.allocations+=1
        else:self.reuses+=1
        # Both C and F layouts are backed by contiguous one-dimensional storage.
        if order=='F' and len(shape)>1:
            if cuda.config.ENABLE_CUDASIM:
                return slot[:count].reshape(tuple(reversed(shape))).transpose(tuple(reversed(range(len(shape)))))
            from numba.cuda.cudadrv.devicearray import DeviceNDArray
            strides=[];stride=dtype.itemsize
            for length in shape:
                strides.append(stride);stride*=length
            return DeviceNDArray(shape=shape,strides=tuple(strides),dtype=dtype,gpu_data=slot.gpu_data)
        return slot[:count].reshape(shape)
    def upload(self,value):
        a=np.ascontiguousarray(value);out=self.empty(a.shape,a.dtype);out.copy_to_device(a);return out
    @property
    def bytes(self):return sum(x.nbytes for x in self.buffers if x is not None)


class StreamGPUBatch(PrefetchGPUBatch):
    prepare_task = staticmethod(_prepare)
    verify = staticmethod(checked_cpu_arrays)
    def __init__(self,*,shared_root='/dev/shm',finish_threads=16,**kwargs):
        if kwargs.get('indexed'):raise ValueError('Streaming uses compact per-bank indexed vectors')
        super().__init__(**kwargs)
        self.shared_root=shared_root;self.shared_dir=None;self.finish_threads=finish_threads
        self.finish_pool=None;self.selection_pool=None;self.buffers=DeviceBufferPool()
        self.pipeline.update(shared_bytes=0,shared_metadata_bytes=0,overlapped_batches=0,
                             device_allocations=0,device_reuses=0,device_capacity_bytes=0)

    def start(self,geometry):
        if self.pool is not None:
            if geometry is not self.geometry:raise ValueError('Geometry changed')
            return
        from magictools.itf.campaign_geometry import AtlasGeometry
        from numba.cuda.cudadrv import driver
        if not isinstance(geometry,AtlasGeometry):raise ValueError('Immutable atlas required')
        if driver.driver.is_initialized:raise RuntimeError('Start preparation before CUDA')
        self.geometry=geometry
        self.shared_dir=tempfile.mkdtemp(prefix='itf-stream-',dir=self.shared_root)
        self.pool=ProcessPoolExecutor(max_workers=self.preparation_workers,mp_context=mp.get_context('fork'),
            initializer=_initialize,initargs=(geometry,self.shared_dir))
        list(self.pool.map(_ready,range(self.preparation_workers)))
        self.finish_pool=ThreadPoolExecutor(max_workers=1)
        self.selection_pool=ThreadPoolExecutor(max_workers=self.finish_threads)

    def _finish(self,prepared,jobs,out,receipt,planning,solve_receipt):
        try:
            bank,plans,groups,epochs,bounds,limit=prepared
            stamp=time.perf_counter()
            states,flags,residuals,jacobians,fallback=self.verify(bank,out,limit,threads=self.finish_threads)
            verified=time.perf_counter()-stamp;stamp=time.perf_counter()
            def finish(item):
                plan,(_,settings),group=item
                return finish_group(plan,settings,group,epochs,bounds,states,flags,residuals,jacobians)
            rows=list(self.selection_pool.map(finish,zip(plans,jobs,groups)))
            return rows,dict(chains=len(jobs),starts=len(bank['states']),planning_seconds=planning,
                **solve_receipt,cpu_verify_fallback_seconds=verified,cpu_fallback_starts=int(fallback.sum()),
                selection_seconds=time.perf_counter()-stamp)
        finally:Path(receipt['path']).unlink(missing_ok=True)

    def iter_batches(self,geometry,batches):
        self.start(geometry);pending=None
        try:
            with closing(ordered_prepared(batches,self.pool,self.prepare_task,self.depth,self.pipeline)) as source:
                for batch,(receipt,planned) in source:
                    prepared=load_prepared(receipt);bank,_,_,_,_,limit=prepared
                    self.pipeline['shared_bytes']+=receipt['bytes']
                    self.pipeline['shared_metadata_bytes']+=len(receipt['metadata'])
                    if len(bank['states']):
                        self.buffers.begin()
                        workspace=ResidentWorkspace(bank,coalesced=True,allow_simulator=self.allow_simulator,allocator=self.buffers)
                        out=workspace.solve(bank['states'],limit,parallel_normal=True,parallel_copy=True,
                            completion_interval=self.completion_interval,register_normal=self.register_normal)
                        solve_receipt=dict(gpu_setup_seconds=workspace.setup_seconds,gpu_solve_profile=out['profile'])
                    else:out=None
                    # CPU assembly for batch N-1 overlaps GPU N. Bound backlog to one.
                    if pending is not None:
                        prior,future=pending;rows,record=future.result();self.receipts.append(record)
                        self.pipeline['overlapped_batches']+=1
                        pending=None
                        yield prior,rows
                    if out is None:
                        Path(receipt['path']).unlink(missing_ok=True)
                        yield batch,[None]*len(batch)
                    else:
                        pending=(batch,self.finish_pool.submit(self._finish,prepared,batch,out,receipt,planned,solve_receipt))
                if pending is not None:
                    prior,future=pending;rows,record=future.result();self.receipts.append(record)
                    pending=None;yield prior,rows
        finally:
            if pending is not None:
                # Drain before shared files/device memory can be reclaimed.
                pending[1].result()
            self.pipeline.update(device_allocations=self.buffers.allocations,device_reuses=self.buffers.reuses,
                device_capacity_bytes=self.buffers.bytes)

    def close(self):
        super().close()
        if self.finish_pool is not None:self.finish_pool.shutdown(wait=True);self.finish_pool=None
        if self.selection_pool is not None:self.selection_pool.shutdown(wait=True);self.selection_pool=None
        if self.shared_dir is not None:shutil.rmtree(self.shared_dir);self.shared_dir=None
