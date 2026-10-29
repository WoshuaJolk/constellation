# Sustained full-catalogue checkpoint benchmark — 2026-10-28

The current hybrid GPU fitter does **not** outperform the stronger CPU implementation on this workload. It processes 565 seed chains/second, compared with 352 using 20 CPU threads and 623 using 20 CPU processes. The process check used the same fitter, initialization policy and scientific cuts. It was prospectively added before the first GPU pass and ran once for two minutes; it is not a second AB/BA experiment.

The benchmark is complete. The H100 worker was deleted after checksum-verified returns. Estimated incremental compute cost, including provisioning, installation, tests, transfer and idle intervals, was **$2.1691**, within the additional $5 cap. Provider verification found **zero remaining droplets**. Historical snapshots were untouched. This is estimated compute usage, not a final invoice or a claim that historical storage has no cost.

## Workload and comparison

We loaded the original **2,548,307 tracklets / 8,934,160 observations**, the original **7,000,000-seed queue**, its observer geometry, and its 20,038,466 directed graph entries. Seeds were visited in the frozen permutation `(104729*i + 20261028) % 7000000`, without selecting on prior fit success.

Every seed used the full Gauss plus 44 range/radial starts, broad initialization, 80 solver iterations, assumed 0.3″ observational sigma, the existing 0.6″ RMS limit and all other existing fit cuts. Range hypotheses span 0.01–1000 AU. This is a full-start benchmark, **not** the original campaign's cheaper scout/cascade initialization workload. It does not measure the completeness of the original candidate graph or the plausibility of each fitted orbit.

The primary comparison used CPU→GPU, then GPU→CPU, with five minutes per phase, 1,024-chain batches and separate deterministic queue ranges. Only batches whose fitting finished within the time limit receive throughput credit. Serialization contributes to elapsed time; final batch drain and all overrun results remain recorded separately.

| Backend | Timed seconds | Chain evaluations | Optimizer starts | Chains/sec | Retained chains |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU, 20 threads | 600 | 210,944 | 9,362,841 | 351.57 | 7,205 |
| Hybrid H100 | 600 | 338,944 | 15,044,525 | 564.91 | 11,626 |
| CPU, 20 processes; supplementary check | 120 | 74,752 | 3,318,012 | 622.93 | 2,587 |

The two GPU passes measured 566.61 and 563.20 chains/sec; CPU-thread passes measured 354.99 and 348.16. The GPU/thread ratio is **1.607×**. GPU/process throughput is **0.907×**: no GPU advantage was observed against the stronger CPU baseline. All backends ran sequentially on the same H100 host, so this is not a separate CPU-machine price comparison.

Across the timed passes there were **624,640 chain evaluations and 27,725,378 optimizer starts**, including repeated matched work. The main passes covered **338,944 distinct seed positions: 4.842% of the 7M queue**, not all seven million. The supplementary prefix was already covered. No numerical exceptions or unavailable-geometry cases occurred in these processed prefixes; this does not mean every optimizer start converged. All nonretained chains and overrun work are saved.

At these sustained rates, fitting all seven million with this particular full-start policy would take approximately 5.53 hours with threads, 3.44 hours with the hybrid GPU, or 3.12 hours with processes. These are fitting-only linear projections, not measured full runs, historical campaign timings, or estimates for complete discovery and verification.

## Numerical agreement

All **210,944 shared CPU/GPU seeds** agree on retain/nonretain status. No shared retained chain differs in best-fit RMS by more than 0.001″. In the first range the largest best-fit difference was 6.82e-10″. However, **28 chains have different numbers of alternate modes** (13 and 15 by round). Best-fit agreement is not complete orbital-branch equivalence. These discrepancies remain open; this benchmark does not approve replacing production fitting.

The process/thread CPU comparison covers 74,752 shared seeds with identical retention, best-fit RMS and mode counts. GPU same-state verification triggered 27,319 CPU fallback starts across its timed batches; their cost is included. Hardware regression tests passed **39 tests, with 1 skipped**. Four focused local tests passed, covering queue permutation, instrumented fitter equivalence, process equivalence, and disagreement/overtime accounting.

## Where the time goes

GPU batches spent 585.11 seconds inside the adapter:

| Component | Seconds | Adapter time |
| --- | ---: | ---: |
| Preparing starts and packing observations | 329.23 | 56.27% |
| Selecting modes and other result assembly | 158.13 | 27.03% |
| GPU solve | 47.82 | 8.17% |
| CPU state verification and fallback | 28.00 | 4.79% |
| GPU workspace setup | 21.93 | 3.75% |

Host CPU utilization averaged 52.39% for threaded fitting, 89.52% for process fitting and 6.88% during hybrid fitting. GPU utilization averaged 6.37%; sampled peak GPU memory was 747 MiB. These are periodic utilization samples, not exact occupancy measurements; intervals can straddle stage changes.

**A 100× pipeline improvement is not demonstrated.** Making the GPU solve instantaneous would remove only the measured 8.17% of adapter time. The next implementation target is batched start generation, shared observation packing, persistent workspaces and compiled result assembly. More kernel optimization alone cannot deliver the requested improvement in this architecture. The stronger CPU process implementation must remain a baseline.

## Planetary long-gap recovery

This was a separate, CPU-only downstream workload, not recovery driven by the newly fitted GPU outputs. We deterministically selected 541 exact historical mode/context records: the 512 lowest hashes overall plus 32 distant and 32 retrograde modes, deduplicated. Roles overlap. These historical modes are hypotheses, not verified objects.

The retrieval index contained **2,545,457 tracklets across 10,246 nights**; 2,850 catalogue tracklets were unindexed. DE440s planetary propagation, original joint fitting, five gap windows through the full date range, two growth rounds, four branches, 16 proposals/window and 12 uncertainty-axis probes were retained. Per-parent and individual-fit caps were 30 and 5 seconds. The frozen 541-parent queue exhausted in **46.31 seconds**, inside its 180-second allowance:

- 467 parent modes finished their configured search without a recorded partial failure.
- 73 finished with partial numerical failures; one exhausted its parent time budget. These remain unresolved, not rejected objects.
- 7,824,622 night queries and 119,050 raw proposals were recorded. These are repeated mode/night operations, not distinct observing nights or independent observations.
- 3,103 single-tracklet, 192 alternative-branch and 1,003 whole-arc fit proposals were attempted. There were 681 fit numerical-failure events; that is not a count of 681 distinct false chains.
- Output contains **358 distinct extension memberships**, up to **15 tracklets** and **10,558 days** of arc. These are benchmark-retained hypotheses, not confirmed discoveries or calibrated associations; they were not added to production.

All 32 distant and all 32 retrograde modes finished their configured searches, but neither subset produced an extension. Distant input ranges span 30.79–1164.31 AU. Their 4,951 raw proposals reduced to 1,092 innovation rejections and **zero extension fits**; retrograde modes reached eight extension fits with no retained extension. This identifies a retrieval/screening question that faster fitting does not answer. It does not show that any rejected proposal was a real association.

There were substantial existing search caps: 103,079 raw proposals discarded at the candidate cap, 10,333 uncertainty proposals deferred, and uncertainty-gated nights. Therefore neither the fast runtime nor zero distant outputs establishes exhaustive recovery. The local covariance model remains uncalibrated. The test also says nothing about whether a dwarf planet is bright enough to appear in ITF.

## Setup, preservation and reproduction

The compressed input archive was 792,157,194 bytes (1,257,620,407 uncompressed bytes), taking 37.71 seconds to package and approximately two minutes to transfer. Remote checksum verification took 1.04 seconds; loading/touching the reused checkpoints took 0.45 seconds; planetary index/pool setup took 2.12 seconds; main numerical warmup took 6.99 seconds. Fresh graph construction and observer-geometry generation were **not** benchmarked. Main timed phases including drain took 1,206.71 seconds; the supplementary process pass took 120.38 seconds plus 0.81 seconds setup.

Inputs and exact executed code hashes are in [`gpu-deep-receipts/data-manifest.json`](gpu-deep-receipts/data-manifest.json) and [`code-manifest.json`](gpu-deep-receipts/code-manifest.json). The original queue SHA-256 is `884f1f3c36850bbe469419205aac8fcae3930d4c65735c9f79140327a3f25a86`. The supplemental process amendment, prelaunch fork/BLAS corrections, hardware versions, test logs, output hash receipts and deletion receipt are preserved alongside them.

With the manifest-listed files staged under the same relative names and the recorded dependencies installed:

```sh
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
export NUMBA_CACHE_DIR="$PWD/.numba-cache"
python research/solve_itf/gpu_deep_benchmark.py --root . --output fit-results
# Run sequentially, only after the original job finishes:
python research/solve_itf/gpu_deep_cpu_process_check.py --output process-results
python research/solve_itf/gpu_deep_summary.py fit-results
```

The benchmark uses Linux fork workers and an NVIDIA CUDA environment. Restore the paths from the manifest first; these commands do not download the historical dataset or provision cloud machines. They do not install a spending guard; the recorded run used an external 40-minute lifetime guard and a $5 cap.

Full returned phase outputs, individual fits, utilization traces and planetary fit caches remain in `/tmp/itf-deep-benchmark-20261028`, verified against `returned-evidence.json` and `cpu-process-returned.json`. Compact reports are committed here; the large source catalogue and raw output archives are not added to Git. [`summary.json`](gpu-deep-receipts/summary.json) is the machine-readable result. Original production data, paused discovery work and the website were unchanged.
