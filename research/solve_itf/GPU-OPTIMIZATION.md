# GPU local fitting: resident-buffer and kernel optimization

Later whole-search and distant-control results: [GPU-PIPELINE-AND-DISTANT.md](GPU-PIPELINE-AND-DISTANT.md).

Measured October 28, 2026. Follow-up to [the complete-fit benchmark](GPU-FIT-BENCHMARK.md).
This is an experimental two-body optimizer, not a deployed linker or an orbit verifier.

## Result

The same **240,000 range/radial starts on 10,000 distinct existing chains** now
run in **0.2774 seconds with their observation bank resident**. The unchanged
compiled CPU solver takes **3.4978 seconds on 20 threads** on the same machine:
**12.61× steady-state speedup**. Three timed repetitions were run per variant.

The first call also pays **0.5326 seconds** to validate, allocate, and upload its
bank. Setup plus the first solve was **0.8099 seconds**, or **4.32×** versus the
CPU median. Do not describe 12.61× as one-shot or full-search acceleration.
Reusing the same observations benefits subsequent start batches; repeating a
benchmark does not itself demonstrate useful additional search work.

| Implementation | Median seconds for 240,000 starts | Setup included? |
|---|---:|---|
| Compiled original CPU, 20 threads | 3.49784 | Existing host geometry; solver timing |
| Previous GPU, fresh workspace (first follow-up worker) | 0.97100 | Yes |
| Reused buffers, original kernels (first follow-up worker) | 0.43909 | No |
| Reused buffers, coalesced solver arrays (first follow-up worker) | 0.39866 | No |
| Parallel normal equations (second follow-up worker baseline) | 0.32914 | No |
| Also columnar Jacobian storage | 0.32920 | No |
| Also parallel accepted-residual/Jacobian copies | **0.27742** | No |
| Both columnar storage and parallel copies | 0.27783 | No |

Medians are wall-clock timings without per-kernel timing events. Initial JIT
compilation/warm-up is excluded from throughput, but included in billed machine
lifetime. CPU scheduling remains inside the compiled loop, with BLAS threads
fixed at one. CPU reference arrays are identical to the preceding benchmark.

## What changed

- `ResidentWorkspace` retains validated observations and scratch arrays between
  explicitly supplied start batches. Every solve resets damping, activity,
  convergence flags, counters, and initial residuals. We test repeated reuse
  after contaminating scratch state with a different solve.
- Optional column-major storage for small solver arrays improves neighboring
  threads' memory access.
- Six threads per chain compute parameter scales and normal-equation rows.
  Each element retains its original ordered observation sum; no tree reduction,
  new stopping criterion, precision reduction, or fast math is introduced.
- Accepted residuals/Jacobians are copied in parallel per observation, rather
  than serially by the thread managing an entire chain.
- Columnar Jacobian storage was tested but provided no useful gain. It remains
  opt-in and is not used in the recommended measured configuration.

The first follow-up improved the proposal/normal-equation component from about
187 ms to 75 ms in the separate instrumented profiles. The second targeted
accepted-fit copies. Full event profiles are retained as diagnostics; they add
launch overhead and are not substituted for uninstrumented throughput.

The measured fast configuration is:

```python
workspace = ResidentWorkspace(bank, coalesced=True)
result = workspace.solve(initial_states, 80,
                         parallel_normal=True, parallel_copy=True)
```

This API is experimental, single-stream, and non-reentrant. Bank observations,
weights, and dimensions are fixed at construction. It is not imported by the
production `fit_chain` path. The compatibility `solve_resident` wrapper still
charges construction each time and keeps the original kernel choices.

## Numerical evidence and remaining blocker

All new GPU variants and repetitions preserve **all 240,000 final states,
objective costs, and optimizer flags exactly** relative to the previous GPU.
The saved result archives have the identical SHA256
`56cdaf0f050905ad0ee84a703d28c7c9313a7f71cbd1348207d40c29a8b3c1fb`.
No starts, observations, weights, iterations, or validity thresholds were removed.

This preserves, rather than fixes, the previous CPU/GPU differences:

- 68 convergence-flag differences and 106 final RMS differences over 0.001″.
- Final same-state comparison fails on **28 starts / 291 observation entries**.
  Unlike the preceding run, both CPU and GPU validity masks are now saved.
- All 28 have poor GPU RMS (previously measured 111–366,327″). That does not
  authorize suppressing the failures or asserting full numerical compatibility.

A separate CPU-only trace of those frozen states identifies exhaustion of the
**100-iteration universal-Kepler root solve** in all 28 cases. The instrumented
propagator exactly matches the original output on all 441 traced calls. The
trace uses the original scalar light-time arithmetic. Its earlier vector-dot
version is preserved separately and superseded. This locates the CPU failure;
it does not establish whether CPU or GPU predictions are physically correct,
justify increasing the iteration cap, or validate every optimizer branch.

The reports therefore say `completed_with_numerical_blockers`. No production
backend, acceptance rule, calibrated probability, or discovery claim was deployed.
Hardware tests: **18 passed, one simulator-only test skipped** in the final wave.
Local simulator checks covered repeated workspace use, invalid inputs, exhausted
iterations, layout changes, and parallel-copy behavior.

## What reaching 100× would require

Against this optimized 20-thread baseline, 100× would mean about **35 ms** for
240,000 starts—another **7.9×** beyond the best measured steady-state result.
That is not established by these measurements. Removing the remaining one-time
setup cannot accelerate an already-resident solve by that factor.

Useful next targets are compact shared observation geometry across the 24 seed
hypotheses, active-work compaction, and static ordering of similar hypotheses
so neighboring threads spend less time waiting on different propagation paths.
These are not implemented or claimed speedups here. The old cold cohort occupied
19.2 million iteration slots for 7.65 million actual candidate iterations: even
perfect compaction would reduce scheduled slots by at most about 2.51×, not
necessarily runtime by that amount. Propagation work and numerical compatibility
still need attention. Broader search and final planetary verification remain
outside this benchmark.

## Reproduction, evidence, cost and cleanup

```bash
python research/solve_itf/gpu_reuse_benchmark.py --bank cold-bank.npz --output reuse-results
python research/solve_itf/gpu_columnar_benchmark.py --bank cold-bank.npz --output layout-results
```

The bank is unchanged, SHA256
`9bcec2d7483d045fa6f3ea577ecb5013957b8230b08a7f67615babb492d00765`.
Two designs were recorded before their respective hardware runs. Each retained
its three repetitions, numerical checks, fixed 80-iteration limit, $1.50 ceiling,
and 20-minute resource deadline. No post-result threshold tuning occurred.

`gpu-optimization-receipts/` contains designs, code hashes, raw reports, hardware,
test output, returned-file hashes, failure trace, and deletion receipts. Full
arrays and frozen source payloads remain under `/tmp/itf-gpu-reuse-20261028/` and
`/tmp/itf-gpu-columnar-20261028/`. All reported outputs were returned and hash
verified before either worker was deleted.

This round cost **$0.64477 estimated compute**; the combined prototype,
connection-repair, fitting and optimization sequence is **$3.44177 / $5**.
These are pretax time × rate estimates, not invoices. **Zero Droplets remained**
at the final provider check. No scientific snapshots were deleted.
