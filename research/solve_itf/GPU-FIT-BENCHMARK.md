# Complete-fit GPU development benchmark, 2026-10-28

The experiment runs complete **single-start** `solve_lm` optimizations, not the
entire linker, multi-start selection or scientific verification. It preserves
observations, sigma, two-body/light-time equations, 80-iteration limit, damping
and convergence constants. No production backend or acceptance rule changed.
All inputs come from the already-opened saved-candidate development catalogue.

## Architecture and CPU fairness

The first integration batched GPU residual/Jacobian work but returned derivative
arrays for CPU optimizer updates every iteration. The resident implementation
keeps states, residuals, derivatives, normal equations, six-variable Cholesky
solves and accept/reject updates on the GPU. It uploads inputs once, checks a
four-byte unfinished count every four iterations, and downloads final results.
Finished cases skip numerical work. There is no full-array iteration transfer.

CPU comparisons use the original compiled `derivatives.solve_lm`. Its scheduling
loop is also compiled, with independent blocks dispatched over 1, 4 or 20 CPU
threads. The initial Python-per-fit wrapper impeded CPU scaling; its preliminary
apparent GPU win is superseded by this stronger CPU baseline. Its receipts are
retained under `hybrid-initial-*`; the corrected comparison is `resident-warm-*`.
The batched CPU state machine reproduces all 10,000 original final states and
convergence flags exactly. GPU reduction/transcendental rounding may differ.

## Saved-state refinement: an easy workload

10,000 saved states, 121,813 observation entries across the chains; no new chains
or observations. Shared detections can occur in more than one chain: this count
is not a claim of 121,813 distinct physical measurements.
One supplied start per case; 90% of CPU fits finish within two iterations.
Three uninstrumented repetitions; medians below, compilation/warm-up excluded.
GPU headline times include per-bank allocation, upload and final download.

| Backend | Seconds for 10,000 fits |
| --- | ---: |
| Compiled CPU, 1 thread | 0.230461 |
| Compiled CPU, 4 threads | 0.061003 |
| Compiled CPU, 20 threads | **0.019137** |
| GPU residuals + CPU optimizer | 0.116634 |
| Device-resident GPU optimizer | 0.033780 |

The resident GPU is **3.45x faster than the hybrid GPU**, but the strongest tested
CPU remains **1.77x faster end to end** on this small, easy workload. Resident
GPU setup/upload takes about 20 ms; device iterations about 13 ms; final output
about 0.46 ms. The separate instrumented device run attributes about 8.35 ms to
proposal/normal equations/Cholesky, 3.07 ms to acceptance and 2.09 ms to
propagation. CUDA-event profiling adds overhead and is excluded from timing
medians. Normal-equation work and amortizing setup are the next GPU targets.

Independent instrumentation of the original CPU solver's first 1,000 cases
measures 23.085 ms in prediction/derivatives out of 29.369 ms process CPU time,
about 79%. Clock-call overhead is included; throughput uses uninstrumented runs.
This is a profile of this workload, not proof of whole-campaign bottlenecks.

CPU: 9,999 converged / 1 nonconverged. Both GPU implementations: 9,998 / 2.
Three flags disagree in both directions; they remain visible and block claiming
an interchangeable production replacement. The maximum final RMS difference is
4.12e-10 arcsec. Same-state prediction and derivative parity passes before and
after fitting. Close residuals do not erase stopping-status discrepancies or
establish that any association is real. No numerical threshold was changed.

These complete-fit timings should not be divided by the earlier isolated-stage
numbers: the original stage harness repeatedly grouped observations with masks
and built per-case summaries inside its timed loop. This benchmark uses indexed
compiled scheduling and prepares summaries outside the timed fit region.

## Harder workload: production range/radial starts

The next registered comparison expands every one of those 10,000 parents with
the existing `fit_chain` grid: distance 0.1, 0.5, 1, 2, 4, 10, 40, 150 AU, each
with radial velocity 0, -0.01, +0.01 AU/day. This gives **240,000 real starting
guesses**, not 240,000 unique chains. The 10,000 parent membership sets were checked distinct. Observation geometry is reused unchanged;
epochs shift to the parent midpoint exactly as in the original start generator.
A regression test captures `fit_chain` starts and checks exact seed/geometry
parity. No Gauss, saved-mode or extended-range starts are added. No outcome
filtering or replacement occurs. Parent selection is still biased toward
previously retained chains, so this is not a full 7M search benchmark.

All six CPU timing repetitions and all three GPU timing repetitions completed.
The post-fit same-state numerical check then failed; the frozen report retains
`status: blocked`, not a successful compatibility claim.

| Backend | Median seconds for 240,000 starts |
| --- | ---: |
| Compiled CPU, 4 threads | 17.126070 |
| Compiled CPU, 20 threads | 3.499441 |
| Device-resident H100 | 0.962064 |

This is **3.64x raw fitting throughput versus 20 CPU threads**, or **17.8x versus
4 threads**, with per-bank GPU setup and transfers included. These are measured
computational timings, not an end-to-end discovery speedup or a passed numerical
compatibility result. Three repeats are bit-identical within each backend.
GPU setup/upload takes about 0.523 s, iterations 0.437 s and final output 0.00236 s.

CPU: 220,217 converged / 19,783 nonconverged starts; GPU: 220,239 / 19,761.
There are **68 convergence-flag disagreements**, and **106 starts** with final
RMS differing by more than 0.001 arcsec (159 in their union, spanning 91 parents).
Most starts agree closely, but the largest RMS difference is 419.64 arcsec and
some fitted state components differ substantially. Retain all alternatives;
these cannot be summarized as identical solutions.

Initial same-state parity passed (maximum residual difference 5.48e-8 arcsec,
scaled derivative difference 9.33e-9). Final same-state validity masks disagreed.
A CPU-only audit of **all 240,000 final GPU states** found **28** for which the
original CPU predictor returns its invalid sentinel. A diagnostic CPU trace
reproduces all 28 exactly and locates the failure at nonfinite/too-small
propagated range during the light-time loop. It does not establish which
internal universal-variable failure occurred or whether CPU/GPU should both
fail there. The failed hardware check did not save its individual GPU masks;
28 is the CPU replay count, not a newly asserted exact hardware mask-difference
count. No new GPU run or tolerance change was used to suppress the failure.

All 28 already have poor GPU RMS: **111.07–366,327.19 arcsec**, none below the
existing 0.6-arcsec baseline. That is context, not permission to exclude them
from the benchmark. They remain unresolved numerical compatibility cases.
Post-check cold CPU/CUDA-event profiles were not run because the fixed check
stopped the script. The warm-workload profiles above remain separate evidence.

## Disposition and next work

No production filter or linker backend was deployed. The next engineering work
is to resolve or explicitly route predictor-invalid results through verified CPU
checking, then test production multi-start orchestration. Reusing observation
and scratch buffers across successive start batches would amortize the measured
GPU setup cost; parallel reductions for each chain could improve the remaining
normal-equation bottleneck. Neither gain is claimed as measured here.

All result/state archives were returned and hash-verified before deleting each
worker. A fresh provider check found **zero Droplets** and no test SSH keys.
This full-fit/optimization sequence cost an estimated **$1.19**, with **$2.80**
combined including the prior prototype and connection repairs, against the
unchanged $5 cap (pretax compute estimates, not invoices).

`gpu-fit-receipts/` preserves each frozen design, source/array hashes, hardware,
test outputs, raw reports, failure diagnostics and resource dispositions. Local
full state archives and the 240,000-row paired table are under
`/tmp/itf-gpu-cold-grid-20261028/`; table hash is in
`cold-paired-table-receipt.json`. Earlier warm outputs remain in
`/tmp/itf-gpu-full-fit-20261028/` and `/tmp/itf-gpu-resident-20261028/`.
Hardware tests: 11 passed, simulator-only guard skipped; all 12 tests passed
locally with the explicit simulator. Compilation and warm-up are excluded from
throughput but remain charged in the compute ledger.

## Reproduce

Dependencies and image are recorded in hardware/setup receipts. `numba-cuda`
provides the CUDA backend. Host BLAS threads are fixed at one to avoid nested
oversubscription. The GPU kernels use FP64 without fast math.

```bash
NUMBA_ENABLE_CUDASIM=1 python -m pytest tests/test_itf_gpu.py tests/test_itf_batched_fit.py tests/test_itf_gpu_cold_bank.py
python research/solve_itf/gpu_fit_benchmark.py --bank saved-bank.npz --output warm-results
python research/solve_itf/gpu_cold_bank.py --input saved-bank.npz --output cold-bank.npz
python research/solve_itf/gpu_cold_benchmark.py --bank cold-bank.npz --output cold-results
```

The frozen saved bank SHA256 is
`2c32db53b6a4a8f197e97007e1827910f5b006a4911dbbf30d28e93e5dcf6b68`.
The derived cold bank SHA256 is
`9bcec2d7483d045fa6f3ea577ecb5013957b8230b08a7f67615babb492d00765`.
No calibrated probabilities, discoveries or public certification follow from
these computational development benchmarks.
