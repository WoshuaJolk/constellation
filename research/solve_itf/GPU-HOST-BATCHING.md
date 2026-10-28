# GPU host batching and prefetch — 2026-10-28

The GPU fitting path now sustains **2,258.49 seed chains/sec**, compared with **657.07/sec** for the updated 20-process CPU baseline on the same H100 host: **3.44× faster**. Compared with the frozen GPU adapter on that host, the gain is **3.63×**. The requested **100× is not achieved**. This measures the full seed-fitting stage, including host preparation and output handling, not an entire discovery/verification campaign.

## Changes

- Generate the existing range/radial hypotheses with array operations, preserving their order and arithmetic. Gauss roots, fitted observations, iteration limits and all acceptance cuts are unchanged.
- Reuse the existing `FastAtlasGeometry` observation-identity cache. This avoids rebuilding structured observation keys for every chain and preserves duplicate handling and time ordering. It is existing infrastructure, not a newly invented scientific method.
- Pack all ordered starts in compiled code. CPU verification retains arrays instead of creating a Python result tuple for every start.
- Apply the existing candidate checks and ordered duplicate replacement in compiled code; construct Python dictionaries only for retained modes. Output assembly is shared with the CPU fitter.
- Prepare at most four batches in forked CPU workers while the GPU solves the previous batch. Inputs are consumed in queue order. Pool startup, IPC, result writing and unused prefetch drain are accounted for; producer work is not hidden outside the timed comparison.

The default production search backend remains CPU. The compact adapter and prefetch harness are research paths; no production catalogue or website was changed. A direct small-chain geometry-deduplication prototype was retired in favor of the existing observation-identity implementation.

## Frozen real-data experiment

The original 2,548,307-tracklet, 8,934,160-observation catalogue and 7M seed checkpoint were reused. The source order remained `(104729*i + 20261028) % 7000000`, with 1,024 chains per batch. All backends use Gauss plus 44 range/radial starts, full broad initialization, 80 iterations, sigma 0.3″, RMS limit 0.6″ and the existing per-tracklet/physical cuts. No timed-out starts or failed fits are silently promoted.

| Phase | Timed seconds | Chain evaluations inside budget | Chains/sec | Retained chains |
| --- | ---: | ---: | ---: | ---: |
| Frozen GPU adapter | 120 | 74,752 | 622.93 | 2,587 |
| Compact GPU, serial preparation | 120 | 204,800 | 1,706.67 | 7,019 |
| Prefetched GPU, first range | 180 | 406,528 | 2,258.49 | 13,971 |
| Updated CPU, 20 processes | 120 | 78,848 | 657.07 | 2,729 |
| Prefetched GPU, second range | 180 | 406,528 | 2,258.49 | 13,921 |

The second prefetched range begins at ordinal 1,000,000. Altogether the comparison contains **813,056 distinct seed positions**, **1,171,456 chain evaluations** including repetitions, and **51,996,441 optimizer starts**. This covers 11.61% of the original queue, not all 7M seeds. Overrun batches and unused prepared batches remain recorded separately. The repeated phases' equality reflects the discrete batch counts inside their time limits, not a claim of perfectly identical timing.

No numerical exceptions occurred in the timed prefixes. Nonconverged starts and nonretained chains remain part of the measured workload. Retention is an unchanged fitting disposition, not confirmed physical identity or calibrated probability.

## Numerical preservation

- Local preparation tests match every bank array byte-for-byte on a fixed 1,024-chain real-data batch.
- Frozen GPU versus compact GPU: **74,752 shared seeds**, no retention changes, no best-RMS differences, no mode-count changes and no differences in complete retained output rows.
- Frozen GPU versus prefetched GPU: the same **74,752 shared seeds**, again no output differences.
- Compact versus prefetched GPU: **204,800 shared seeds**, no output differences.
- Updated CPU versus prefetched GPU: **78,848 shared seeds**, no retention changes; maximum best-RMS difference **6.82e-10″**. Nine chains differ in alternate-mode counts, and retained rows can differ in floating-point/derived fields. The pre-existing CPU/GPU branch-equivalence limitation remains open.

The hardware suite passed **64 tests with 1 simulator-only skip**. Local tests cover frozen seed preparation, exact packing, invalid-input handling, ordered duplicate replacement/ties and complete output assembly. Simulator comparison passed. Final input guards additionally reject a non-float64 seed bank, epochs wider than float64, and fractional iteration limits instead of silently converting them. These guards were added after hardware timing and tested locally; the exact measured source is preserved as `gpu-host-receipts/executed-gpu_batch_host.py`. No additional hardware timing is attributed to those guards.

## Utilization and remaining bottlenecks

Sampled GPU utilization increased from **7.46%** with the frozen adapter to **19.0%** with compact batching and **25.2–29.0%** with prefetch. CPU utilization was approximately **17–19%** during prefetch and **89.85%** during the process-CPU baseline. Sampled peak GPU memory remained below 750 MiB. Intervals can straddle phase boundaries; these measurements are not exact occupancy counters.

For each 180-second prefetched phase, measured GPU-solve wall time was approximately 74 seconds and CPU verification/fallback approximately 23 seconds. Cumulative producer planning time was 106–108 seconds and overlaps GPU execution; it must not be added as though sequential. Process transfer, deserialization, output construction and scheduling still matter. This implementation transfers packed NumPy banks through process IPC; it does not yet use shared pinned buffers or a fully resident catalogue.

The next directed optimization is to remove those transfers and keep compact input geometry resident, then profile and optimize the actual propagation/LM kernels. Faster host preparation alone cannot establish a 100× whole-search gain. A claim of 100× against the frozen 622.93/sec CPU reference would require **62,293 chains/sec**; the observed rate is 2,258.49/sec. Further scientific-work reductions would require a separate recall comparison, not merely relabelling a cheaper search as equivalent.

At this rate the configured full-start fitting policy alone projects to roughly **52 minutes for 7M seeds**. That is a linear projection, not a completed run or a whole-project ETA. Graph construction, long-gap recovery and verification were not retimed here. Their previous measurements and limitations remain in `GPU-DEEP-BENCHMARK.md`.

## Cost, failure and reproduction

Total estimated compute was **$1.71775**, including **$0.41072** for the first attempt, which stopped before timing because its deployment archive omitted the frozen reference file required by two tests. The two failures and 62 passes from that attempt are preserved; they are not rewritten as a successful preflight. The corrected attempt cost **$1.30703**. Both workers were deleted after verified evidence returns. A fresh provider check found zero droplets; historical snapshots were untouched.

The retry retained the original shutdown deadline and the total **$3** task cap. Timing and code contracts, both attempts' hardware/test/deletion receipts, hashes, local experiments and machine-readable results are in `gpu-host-receipts/`. Full raw fit outputs and utilization traces remain under `/tmp/itf-gpu-host-batch-retry-20261028`; the first attempt and local diagnostics remain under `/tmp/itf-gpu-host-batch-20261028`. Large raw data are not added to Git.

To reproduce with the manifest-listed catalogue, atlas, observation identities and seed checkpoint already staged:

```sh
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
export NUMBA_CACHE_DIR="$PWD/.numba-cache"
python research/solve_itf/gpu_host_benchmark.py --output fit-results
python research/solve_itf/gpu_host_summary.py fit-results
```

The harness expects Linux fork workers, a CUDA-capable GPU, `data-manifest.json` and `extra-manifest.json`, and the recorded reference sources. It does not provision cloud resources or install a cost guard. The recorded experiment used an external deadline guard. [`summary.json`](gpu-host-receipts/summary.json) contains the complete comparison; [`accounting.json`](gpu-host-receipts/accounting.json) contains cost and preservation details.
