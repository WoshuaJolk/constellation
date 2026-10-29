"""Bounded, ordered preparation overlap for an immutable full-search atlas."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
from numba import cuda
from magictools.itf.campaign_geometry import AtlasGeometry
from gpu_batch_host import prepare_bank
from gpu_compact_adapter import CompactGPUBatch

_PREP_GEOMETRY = None


def _initialize(geometry):
    global _PREP_GEOMETRY
    _PREP_GEOMETRY = geometry


def _ready(_):
    return True


def _prepare(jobs):
    started = time.perf_counter()
    return prepare_bank(_PREP_GEOMETRY, jobs), time.perf_counter()-started


def ordered_prepared(batches, pool, prepare, depth, receipt):
    """Consume only a bounded prefix; yield original order and drain on exit.

    GPU/result errors never cause selection of a different preparation. Pending
    work is cancelled where possible, otherwise drained and charged to wall time.
    """
    pending = deque(); batches = iter(batches)
    def submit():
        try: batch = next(batches)
        except StopIteration: return False
        pending.append((batch, pool.submit(prepare, batch)))
        receipt['submitted_batches'] += 1
        receipt['peak_pending_batches'] = max(receipt['peak_pending_batches'], len(pending))
        return True
    try:
        for _ in range(depth):
            if not submit(): break
        while pending:
            batch, future = pending.popleft()
            value = future.result()
            # Keep the next preparation running while the caller executes GPU.
            submit()
            receipt['returned_batches'] += 1
            yield batch, value
    finally:
        started = time.perf_counter()
        for _, future in pending:
            if future.cancel(): receipt['cancelled_batches'] += 1
            else:
                try: future.result()
                except Exception as exc: receipt['drain_errors'].append(type(exc).__name__+': '+str(exc))
                receipt['unused_prepared_batches'] += 1
        receipt['drain_seconds'] += time.perf_counter()-started


class PrefetchGPUBatch(CompactGPUBatch):
    def __init__(self, *, batch_size=1024, preparation_workers=4, depth=3, indexed=False, **kwargs):
        kwargs.setdefault('completion_interval',16)
        super().__init__(**kwargs)
        if not 1 <= batch_size <= 4096 or not 1 <= preparation_workers <= 4 or not 1 <= depth <= 4:
            raise ValueError('Invalid bounded preparation configuration')
        self.batch_size, self.preparation_workers, self.depth = batch_size, preparation_workers, depth
        self.indexed = indexed
        self.pool = None
        self.geometry = None
        self.pipeline = dict(submitted_batches=0,returned_batches=0,cancelled_batches=0,
                             unused_prepared_batches=0,peak_pending_batches=0,drain_seconds=0.,drain_errors=[])

    def start(self, geometry):
        if self.pool is not None:
            if geometry is not self.geometry: raise ValueError('Prefetch geometry changed')
            return
        if not isinstance(geometry, AtlasGeometry):
            raise ValueError('Process prefetch requires immutable, validated atlas geometry')
        from numba.cuda.cudadrv import driver
        if driver.driver.is_initialized:
            raise RuntimeError('Start CPU preparation workers before initializing CUDA')
        self.geometry = geometry
        self.pool = ProcessPoolExecutor(max_workers=self.preparation_workers,
            mp_context=mp.get_context('fork'), initializer=_initialize, initargs=(geometry,))
        list(self.pool.map(_ready, range(self.preparation_workers)))
        if self.indexed:
            from magictools.itf.gpu_indexed import ResidentGeometry
            self.resident_geometry = ResidentGeometry.from_atlas(geometry, allow_simulator=self.allow_simulator)

    def iter_batches(self, geometry, batches):
        from contextlib import closing
        self.start(geometry)
        with closing(ordered_prepared(batches,self.pool,_prepare,self.depth,self.pipeline)) as iterator:
            for batch,(prepared,seconds) in iterator:
                yield batch,self.execute_prepared(prepared,batch,planning_seconds=seconds)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True,cancel_futures=True)
            self.pool = None

    def __enter__(self): return self
    def __exit__(self,*error): self.close()
