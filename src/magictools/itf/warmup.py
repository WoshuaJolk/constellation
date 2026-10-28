"""Compile fitting kernels while the main process ingests a fresh catalogue."""
import multiprocessing
import time

import numpy as np

from .derivatives import solve_lm, state_geometry
from .kernels import directions


def compile_kernels():
    # Match the contiguous float64 signatures used by real fits. This prepares
    # code only: no synthetic observations enter the catalogue or candidate bank.
    state = np.array([2.4, .7, .3, -.004, .009, .002])
    offsets = np.array([-3., -2., -1., 0., 1., 2.])
    observers = np.tile(np.array([1., .1, 0.]), (len(offsets), 1))
    unit = directions(state, offsets, observers)
    ra = np.arctan2(unit[:, 1], unit[:, 0])
    east = np.column_stack([-np.sin(ra), np.cos(ra), np.zeros(len(ra))])
    north = np.cross(unit, east)
    solve_lm(state, offsets, observers, unit, east, north, .3, 80)
    state_geometry(state, 0.)


class KernelWarmup:
    """Use a separate compiler process: Numba's thread compilation lock is global.

    Only used during fresh ingestion with more than one requested worker. The
    child stops before the parallel scan starts, so worker counts do not stack.
    Compilation is an optimization; a failed warmup falls back to normal JIT.
    """
    def __init__(self, enabled):
        self.process = None
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            self.process = multiprocessing.get_context('spawn').Process(target=compile_kernels)
            self.process.start()
        return self

    def finish(self, deadline):
        started = time.monotonic()
        if self.process is not None:
            self.process.join(timeout=max(0., deadline-time.monotonic()))
        completed = self.process is not None and self.process.exitcode == 0
        return dict(enabled=self.enabled, completed=completed,
                    exit_code=self.process.exitcode if self.process is not None else None,
                    wait_seconds=time.monotonic()-started)

    def __exit__(self, *error):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join()
            self.process.close()
