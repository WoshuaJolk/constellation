# GPU streaming and million-seed benchmark — 2026-10-28

The revised streaming GPU adapter completed **1,000,000 distinct triple seeds, chain growth and output in 312.37 seconds**. The 20-process CPU control completed 961,914 seeds before its 900-second search limit and did not run growth. Its 902.80-second application time is **partial**, so this experiment does not establish an exact full-search CPU/GPU speedup at one million seeds.

At the matched, fully completed 100K workload, the revised adapter took **65.20 seconds**, versus **74.33 seconds for the existing prefetched GPU** and **111.52 seconds for CPU20**: 1.14× faster than the previous GPU and 1.71× faster than CPU. A 100× whole-linker gain has not been demonstrated.

## Timing and scope

| Distinct triple seeds | CPU20 | Existing prefetch GPU | Streaming v1 | Streaming v2 |
| --- | ---: | ---: | ---: | ---: |
| 100,000 | 111.523 s | 74.329 s | 81.646 s | **65.201 s** |
| 300,000 | 302.546 s | Not run | **177.815 s** | Not run |
| 1,000,000 | 902.797 s — **partial** | Not run | Not run after CPU limit | **312.370 s — complete** |

These are application times with a **shared, prebuilt proposal graph**. They include loading and verifying that graph, rebuilding adjacency lists, preparing geometry, fitting, growth, GPU initialization, CPU verification/fallback, result assembly and compressed output. Whole-subprocess times are also preserved in the original reports. The successful attempt's graph scan and seed construction took another **62.629 seconds**. Adding that cost makes the million-seed GPU search **375.00 seconds (6m15s)**. At 100K, adding it to both backends reduces the CPU/GPU comparison from 1.71× to 1.36×.

Catalogue ingestion, observation-identity and observer-atlas construction were already completed before this experiment. Their cached files were hash-verified; these timings are not fresh raw-data ingestion benchmarks. Transfer, installation, tests, failed preflights and idle time are charged in the compute ledger.

Each phase ran in a fresh process on the same H100/20-vCPU host; CPU and GPU searches were sequential. These are single fixed-order engineering measurements, not repeated statistical estimates. CPU-only machine pricing was not measured. The prior 14-day/69,836-seed timings are a different workload and are not a matched baseline for this table.

## Fixed workload and preserved rules

The existing catalogue contains 2,548,307 tracklets. A prospectively fixed 60-day graph, 32 neighbors per day and three-million-edge cap scanned the full catalogue, proposed 13,054,614 edges and retained 3,000,000. It supplied one million **distinct** triples. Nested 100K, 300K and 1M prefixes were frozen without using fitting outcomes. The retry reproduced both exact edge and seed SHA-256 hashes. This is a bounded graph, not every possible association: 10,054,614 proposed edges were outside its retained cap, and other finite proposal limits remain.

Both backends use the existing solar two-body fitter, Gauss plus the 24 range/radial starts, maximum 80 iterations, sigma 0.3 arcseconds, RMS limit 0.6 arcseconds, and unchanged growth, mode-selection and deduplication rules. No observations, weights, starts or checks were removed to improve runtime. No new observations were fetched. No production catalogue, dashboard or certification mission was updated.

The million-seed v2 run completed **1,000,991 chain fits including extensions**, involving **24,413,054 optimizer starts**. There were 650 successful extensions and 12,831 retained candidate rows. Those rows are not confirmed objects or calibrated probabilities. All million seeds returned; none was interrupted or left unreturned. The CPU control left 38,086 seeds unfinished and did not perform growth; that unfinished work is not a fitted rejection.

## Implementation and optimization loop

`StreamGPUBatch` adds three changes to the existing research adapter:

- Observation vectors are stored once per chain's detections, with exact indices shared across its starting states. Offset/state/observation order and FP64 arithmetic remain unchanged.
- Prepared arrays cross process boundaries through one aligned, read-only shared-memory mapping per batch. Preparation is bounded to three pending batches, with at most one active GPU batch and one preceding CPU-finish batch. This replaces large pickled array payloads.
- CUDA buffers are reused through explicit shape/stride views, and CPU verification/result assembly overlaps the next GPU solve. GPU ownership remains serial. Downloaded outputs are owned CPU arrays before device buffers can be reused.

The first streaming version was **slower** than existing prefetch at 100K (81.65 vs74.33 seconds). Its profile exposed extra gather allocations during CPU verification and duplicate seed lists in transport. That result remains preserved.

`StreamGPUBatchV2` addresses those measured costs. It uses contiguous observation views only when every index proves that the slice is exact; other mappings retain general gathers. It omits the redundant `plan.seeds` transport list, while every finite ordered start and epoch remains in the packed bank. These transport-only plans are for result assembly, not reusable inputs to a new fit.

At the matched 100K workload, CPU verification/fallback time fell from **25.36 seconds in v1 to 9.64 seconds in v2**; application time fell from 81.65 to 65.20 seconds. The original prefetched adapter's verification was 7.59 seconds, so the revised total improvement is from the combined pipeline, not a claim that every component beats its old counterpart.

Across the million-seed run, the buffer pool made 65 allocations and reused 32,374 slots, with approximately 129.5 MB of allocated device-buffer capacity. Aggregate preparation time was 310.91 CPU-seconds, GPU solve calls 144.98 wall-seconds, CPU verification 85.33 seconds and selection 78.78 seconds. **These overlap and must not be added as wall time.** CPU checking, result processing and GPU launch/solve work remain material bottlenecks. All existing CPU checks remain; 25,753 starts used the existing CPU fallback.

Observed v2 full-search throughput rose from about 1,534 seeds/s at 100K to 3,201 seeds/s at 1M. That supports amortizing overhead with larger workloads, but the prefixes also contain different cases. It does not establish unlimited scaling, complete orbital-branch coverage, long-gap recovery quality or object brightness/detectability.

## Numerical comparisons and limits

- All **6,311 complete retained rows** at 100K compare exactly between old prefetch, streaming v1 and v2, including modes and other row fields.
- CPU100K retained 6,310 rows, all shared with GPU; GPU retained one additional membership. Shared best-RMS differences are at most1.86e-9 arcseconds, and 37 alternate-mode counts differ. At 300K, CPU/GPU share 9,034 rows, GPU has one additional membership, and 77 alternate-mode counts differ.
- Full-row CPU equality is not expected: the extra membership changes ranking/normalized display fields as well as the recorded mode differences. These differences are not truth labels. CPU/GPU orbital-branch equivalence remains unresolved; default production execution stays CPU.
- Each streaming variant passed a separate 128-chain/four-batch exact retained-row comparison with the old compact GPU before timing. Focused tests also cover states/costs/convergence parity on fixed banks, changed buffer shapes, compact index validation, readonly inputs, CPU fallback, transport seed omission, ordering, overlap and cleanup.
- There is **no complete CPU or old-GPU million-seed result** in this experiment. Do not infer full million-case equivalence or an exact CPU speedup from the partial control.

## Failures, tests and resource rules

The first worker's hardware preflight exposed CUDA's unsupported non-2D transpose; explicit Fortran stride views replaced it. A later preflight exposed one mmap per array exhausting file descriptors; one mapping per batch repaired it. Both failures and source hashes remain preserved. No timed search ran on that worker. A local mapping-lifetime test initially retained its own last loop view; its corrected assertion and failed log are retained too.

The successful initial hardware suite passed **42 tests with one skip**. Local integration passed 53 with 16 hardware skips, and focused simulator checks passed 6. The v2 follow-up passed 8 focused tests on hardware,8 in the simulator, and an additional readonly-input parity check. Both 128-chain hardware gates passed. These are numerical/software checks, not association validation.

The original combined ceiling was **$5**, with an absolute worker deadline of 2026-10-28 22:22:00 UTC (18:22 EDT). Neither was reset. After preserving the initial results, the v2 follow-up used a prospectively recorded remaining-time rule to prioritize 1M and then 100K; it did not choose cases based on their outcomes. It ran on the same second worker, not an additional machine. The original schedule stopped at the CPU million-seed limit; the separately registered follow-up completed both selected sizes.

Estimated combined compute cost was **$3.651286**. Worker 50 cost $0.544621; worker 51 cost $3.106665. Both were deleted after verified evidence returns, before the original deadline. A fresh provider check found **zero droplets**. Existing snapshots were untouched and storage charges are separate.

## Artifacts and use

Small immutable designs, failures, tests, source manifests, numerical comparisons, timings, returned-file hashes and deletion/accounting receipts are in `gpu-stream-receipts/`. `results.csv` contains every timed phase, including the partial CPU run. Large full reports and frozen graph/seed arrays are retained under `/tmp/itf-stream-scaling-retry-20261028/`; the failed attempt remains under `/tmp/itf-stream-scaling-20261028/`. The executed v1 adapter is preserved separately from the final v2 injection boundary.

Research use requires the research module directory on `PYTHONPATH`, an immutable validated atlas, Linux fork, and a fresh process before CUDA initialization:

```python
from magictools.itf.search import SearchConfig, solve
from magictools.itf.campaign_geometry_fast import FastAtlasGeometry
from gpu_stream_adapter_v2 import StreamGPUBatchV2

g = FastAtlasGeometry(cache, observatories, atlas, observation_id_cache)
with StreamGPUBatchV2(batch_size=1024, preparation_workers=4, depth=3) as fitter:
    report = solve(g.catalogue, g.sites, SearchConfig(workers=20),
                   geometry=g, batch_fitter=fitter)
```

The benchmark scripts reproduce the fixed development comparisons; they do not provision machines or enforce cloud billing limits. Resident whole-catalogue GPU geometry and the optional register-normal kernel were not enabled in this experiment. No GPU path was made the production default.
