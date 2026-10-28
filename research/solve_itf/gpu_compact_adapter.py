"""Research adapter with compiled host packing and ordered result selection."""
from concurrent.futures import ThreadPoolExecutor
import time
from magictools.itf.gpu_fit import ResidentWorkspace
from gpu_pipeline_adapter import checked_cpu_arrays
from gpu_batch_host import prepare_bank, finish_group


class CompactGPUBatch:
    def __init__(self, *, allow_simulator=False):
        self.receipts = []
        self.allow_simulator = allow_simulator

    def __call__(self, geometry, jobs):
        before = time.perf_counter()
        prepared = prepare_bank(geometry, jobs)
        planned = time.perf_counter()-before
        return self.execute_prepared(prepared, jobs, planning_seconds=planned)

    def execute_prepared(self, prepared, jobs, *, planning_seconds):
        """Consume immutable packed inputs, optionally prepared by CPU workers.

        Caller timing must include queueing/IPC/drain; planning time can overlap
        this execution and must not be added to wall time as if serial.
        """
        before = time.perf_counter()
        bank, plans, groups, epochs, bounds, limit = prepared
        if not len(bank['states']):
            return [None]*len(jobs)
        workspace = ResidentWorkspace(bank, coalesced=True, allow_simulator=self.allow_simulator)
        out = workspace.solve(bank['states'], limit, parallel_normal=True, parallel_copy=True)
        t = time.perf_counter()
        states, flags, residuals, jacobians, fallback = checked_cpu_arrays(bank, out, limit)
        verified = time.perf_counter()-t
        t = time.perf_counter()
        def finish(item):
            plan, (_, settings), group = item
            return finish_group(plan, settings, group, epochs, bounds, states, flags, residuals, jacobians)
        with ThreadPoolExecutor(max_workers=20) as pool:
            rows = list(pool.map(finish, zip(plans, jobs, groups)))
        self.receipts.append(dict(chains=len(jobs), starts=len(bank['states']), planning_seconds=planning_seconds,
            gpu_setup_seconds=workspace.setup_seconds, gpu_solve_profile=out['profile'],
            cpu_verify_fallback_seconds=verified, cpu_fallback_starts=int(fallback.sum()),
            selection_seconds=time.perf_counter()-t, execution_seconds=time.perf_counter()-before,
            serial_planning_plus_execution_seconds=planning_seconds+time.perf_counter()-before))
        return rows
