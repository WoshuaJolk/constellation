# Integrated GPU search optimizations — 2026-10-28

Ordered preparation overlap and larger batches now work inside the full linker. On a matched cached-input search of **69,836 triple seeds**, the prefetched GPU completed scanning, fitting, growth and output in **38.82 seconds**, versus **49.39 seconds for 20 CPU processes**: **1.27× faster**. The requested 100× has not been achieved. At 10K seeds the strong CPU control remains faster.

## Implemented changes

1. `solve` accepts an optional validated geometry object and an ordered batch iterator. Existing CPU defaults and the synchronous batch callback remain supported. The iterator checks submission order, exact membership and result counts. Unreturned prefetched work is explicitly counted; budget exhaustion does not turn it into a fitted rejection.
2. `PrefetchGPUBatch` prepares 1,024-chain batches in four CPU processes while the GPU executes prior batches. At most three pending preparations plus the consumed batch are retained. Results are consumed in order, and shutdown drains/cancels pending work. Preparation processes start before CUDA; this research adapter requires Linux fork and an immutable atlas.
3. Existing catalogue, exact observation-identity and observer-atlas caches are reused. Their manifests and file hashes are checked. No fresh catalogue/atlas construction is included in the cached-input headline timings.
4. Optional resident GPU geometry uploads fixed observation vectors once and sends exact indices for each fitting bank. The indexed kernel is generated from the existing kernel body by changing only observation-vector addressing. Every addressed vector is checked bit-for-bit against the actual CPU fitting bank. This option was slower here and remains opt-in.
5. Prefetched execution checks completion every 16 iterations instead of every four; completed states remain inactive, and the maximum 80 iterations and convergence rules do not change. Separate exact checks preserve states, costs, convergence flags and iteration/evaluation counts.
6. A separately tested optional normal-matrix kernel keeps each running sum in a register, preserving observation and arithmetic order. It improved a fixed kernel/solver replay but has not established a further whole-search gain. It also remains opt-in.

No observations, start grids, weights, physical cuts, RMS limits, mode-deduplication rules or growth tests were removed. The default production backend remains CPU. No production catalogue, dashboard or discovery campaign was updated.

## Fixed full-search comparisons

All runs used the same original 2,548,307-tracklet catalogue and existing hash-verified atlas. Each phase used a fresh process with the same prepared data/compiler caches. GPU initialization, preparation-worker startup/drain, graph scan, fitting, extension, result assembly and output are included in application elapsed time. Transfer, package installation, numerical preflight and input verification are separately recorded and charged in the cost ledger.

The smaller workload uses the original fast-profile 10K seed cap and five neighbors per anchor. The larger workload prospectively uses a 100K cap and all retained neighbors, yielding **69,836 available seeds**. It was not expanded after seeing that shortfall. Both backends within each comparison use identical seed hashes, the 14-day proposal graph, Gauss plus 24 range/radial starts, 80 iterations, sigma 0.3″, RMS limit 0.6″ and unchanged growth settings. The full catalogue scan retained 134,165 graph edges. All phases completed within their 480-second application limits.

| Backend | 10,000 seeds | 69,836 seeds |
| --- | ---: | ---: |
| CPU, four processes | 43.18 s | Not run |
| CPU, 20 processes | **11.53 s** | 49.39 s |
| Existing compact GPU, synchronous 128-chain batches | 25.00 s | Not run |
| Prefetched GPU, 1,024-chain batches | 18.34 s | **38.82 s** |
| Prefetched GPU with resident indexed vectors | 19.89 s | 40.84 s |

Prefetch is 1.36× faster than the synchronous GPU at 10K. It is slower than CPU20 at that size, and 1.27× faster than CPU20 on the larger workload. These are single fixed-order passes on one H100/20-vCPU host, not statistical estimates. The CPU-only machine price comparison remains unmeasured. The cold 184-second historical result and cold 87-second newer result must not be compared directly with these cached-input timings.

The small runs completed 10,282 fits including extensions and retained 4,343 rows. The larger runs completed 70,133 fits including extensions and retained 12,919 rows. Equal counts do not imply identical memberships, as the checks below show. Retained rows are candidate associations, not confirmed objects or calibrated probabilities.

## Numerical and membership checks

- At 10K, CPU4 and CPU20 have identical retained memberships, best RMS and alternate-mode counts. CPU20 and GPU retain the same 4,343 memberships, with best-RMS differences at most 1.86e-9″ and 23 alternate-mode-count differences.
- At 69,836 seeds, CPU and GPU share 12,918 retained memberships. CPU alone retains `0bb0fb + 0dd9d5 + 0dec5a` (705, 0.4250″); GPU alone retains `H488326 + H488478 + H488859` (T09, 0.4426″). Shared best RMS differences remain at most 1.86e-9″; 51 alternate-mode counts differ.
- A fixed replay of those two cases reproduces the same disagreement in the older GPU adapter, compact GPU and register-kernel GPU. Thus these two differences were not introduced by prefetch, indexed inputs or register accumulation. They remain a CPU/GPU orbital-branch-equivalence limitation, not truth labels.
- Compact versus prefetched GPU at 10K, and prefetched versus indexed GPU at both sizes, have identical memberships, best RMS and mode counts.
- The register follow-up preserves complete retained output rows at both sizes: all 4,343 and 12,919 rows compare equal, including alternate modes.
- Separate fixed-bank checks compare all states, costs, success flags, reason codes, iterations and evaluations before CPU verification/fallback. Indexed addressing, polling intervals and register accumulation pass exact equality on their recorded banks.

The first hardware preflight passed **36 tests, one skipped**. The follow-up passed **36 tests, one skipped**. Local integration tests passed **47 with four GPU skips**; the initial full simulator suite passed **24**, followed by **16** focused simulator checks after adding register accumulation. Tests cover ordered/bounded preparation, cancellation/drain, expired work accounting, rejected cache identities, exact observation mapping, and CUDA output parity. These are numerical/software checks, not association validation.

## What the profiles show

For the larger prefetched run, GPU solve calls totaled about **10.87 seconds**, output selection/assembly about **10.68 seconds**, CPU verification/fallback about **2.94 seconds**, and cumulative producer preparation about **18.78 seconds**. Preparation overlaps execution and is not an additive wall-time component. Eleven starts required the existing CPU fallback. Process IPC, Python result construction and startup remain material costs.

The resident-vector variant uploaded **857,679,360 bytes** once. It avoided approximately **999,429,640 bytes** of repeated vector upload in the larger run, before subtracting that initial upload, but preparation and exact-validation overhead outweighed the benefit. No default was changed based on byte counts alone.

The fixed 1,024-chain/25,786-start profile identified normal-matrix accumulation as the largest kernel cost. In a second bounded check on another H100 instance, keeping sums in registers reduced that instrumented kernel from **34.14 ms to 15.59 ms (2.19×)**. Three uninstrumented solver replays had median times **69.10 ms versus 59.50 ms (1.16×)**. Per-kernel events add overhead and are not headline timing.

Register-kernel full-search runs took **21.25 s** at 10K and **40.40 s** at 69,836 seeds. Those runs occurred on the second instance; there was no same-instance full-search control. They do not establish further end-to-end improvement over the first instance's 18.34/38.82 s. The option stays disabled by default. A contemplated extra same-instance control was not launched: automatic evidence return and worker deletion had already completed. The PID ownership check prevented any process termination.

## Cost, artifacts and use

Estimated incremental compute for both workers was **$1.22974**, within the original **$3** total and original **35-minute worker deadline**. The follow-up did not reset either allowance. Both workers were deleted after hash-verified evidence returns; the final provider check found zero droplets. Existing snapshots were untouched; their storage charges are separate.

Designs, code hashes, hardware/tests, summaries, kernel profiles, discrepancy replays and billing/deletion receipts are in `gpu-integrated-receipts/`. Large full reports and batch receipts remain under `/tmp/itf-integrated-prefetch-20261028` and `/tmp/itf-integrated-register-20261028`. The first experiment's executed source versions are preserved separately from the later optional register kernel.

Research use, after constructing the validated immutable atlas and observation identities:

```python
from magictools.itf.search import SearchConfig, solve
from magictools.itf.campaign_geometry_fast import FastAtlasGeometry
from gpu_prefetch_adapter import PrefetchGPUBatch

g = FastAtlasGeometry(cache, observatories, atlas, observation_id_cache)
with PrefetchGPUBatch(batch_size=1024, preparation_workers=4, depth=3) as fitter:
    report = solve(g.catalogue, g.sites, SearchConfig(workers=20),
                   geometry=g, batch_fitter=fitter)
```

The research module directory must be on `PYTHONPATH`. Start this in a fresh process before initializing CUDA. `indexed=True` and `register_normal=True` are explicit experimental options, not default recommendations. `gpu_integrated_benchmark.py` reproduces the measured comparisons with the manifest-listed input files; it does not provision resources or impose a billing guard.

The measured next bottleneck is host preparation/result handling as well as solver work. Further kernel gains alone cannot be assumed to multiply into whole-search gains, and these experiments do not demonstrate a 100× linker or long-gap/planetary recovery speedup.
