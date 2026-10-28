# ITF linker tools and recorded performance

The runnable package is `magictools.itf`; start with [the linker guide](../../docs/solve-itf.md).
`benchmark.py` measures a fresh-cache end-to-end invocation and saves wall time,
resource usage and solver output. `make_recovery_bank.py` merges compatible
saved reports without dropping alternative modes.

## Historical full-catalogue benchmark (2026-10-21)

Source: [original summary receipt](benchmark-2026-10-21.json). This measures an
earlier optimized revision, not a new benchmark of this source release.
Four workers; 2,548,307 tracklets (2,545,389 seed-eligible); 10,000 retained
triple seeds, 10,282 total fits, 4,343 candidate chains. Cold numerical and Numba
caches; input download excluded. Candidate counts are not confirmed objects.

| Stage | Seconds |
| --- | ---: |
| Input/cache preparation and startup | 45.429 |
| Candidate scan | 12.754 |
| Seed generation | 2.787 |
| Observer preparation | 6.956 |
| Triple fitting | 110.379 |
| Chain extension | 3.660 |
| Whole process, including output/overhead | 184.004 |

Small unitemized solver and output costs mean the stage rows need not sum exactly
to wall time. User CPU was 541.731 s; system CPU 5.826 s. Peak RSS was 1.52 GiB
for the largest individual process, not the simultaneous sum across workers.
This is a cloud-host result, not a laptop guarantee or the seven-million-seed
campaign. Later, broader graphs, more starts, planetary dynamics and deeper
extension change the workload; do not extrapolate this total as their runtime.

For a new measurement, run `benchmark.py --help`. Reports expose `timings`,
`counts`, input hashes, configuration and implementation hashes. Record seed
limits, workers, dynamics, hardware, cold/warm caches and completion counters
alongside timings before comparing speedups.

## Historical seven-million-seed production pass

The [7M production receipt](seven-million-production-receipt.json) records the
completed first cascade pass on four workers: **7,000,000 seeds**, **240,102
accepted seed fits**, **276,359 modes**, zero outstanding first-pass retries.
The 6,836 final checkpoint batches sum to **12,710.543 batch-seconds (3.531 hours)**.
These durations overlap across workers: this is NOT elapsed wall time or CPU time,
and superseded failed-attempt durations are not retained in that sum. A clean
whole-campaign wall-time benchmark was not located; it is explicitly null in the
receipt, not inferred from filesystem timestamps or this sum.

The original scout was replaced by cascade initialization; do not call this
seven million full-grid fits. All seven million retained deeper initialization
work, including the 6,759,898 seeds without an accepted cascade fit. Arc growth,
mode refinement and deferred replay are separate stages. The receipt includes
settings, seed selection, original code hashes and extraction SQL. No production
run was restarted to package these records, and the 184-second earlier benchmark
must not be substituted for 7M campaign performance.
