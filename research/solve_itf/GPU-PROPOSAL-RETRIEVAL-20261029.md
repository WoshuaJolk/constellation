# Proposal acceleration and long-gap retrieval replay

The combined proposal path is **7.70x faster** on one complete frozen real-data
epoch window, using 20 CPU workers in both paths. Smaller shared-index comparisons
reach28–143x. These are proposal-stage gains, not whole-linker or discovery gains.
The GPU cone-query prototype loses to20 CPU threads and remains optional.

One H10080 GB/20-vCPU worker was created, used, hash-collected and deleted.
Estimated rate-times-lifetime compute: **$1.10736**, below the $5/one-hour ceiling.
The final provider check reports zero droplets. No production catalogue fitting,
new observations, or additional discovery claims were introduced in this pass.

## Fixed workloads and results

The inputs were selected before fitting or prediction outcomes were computed in
this experiment. The proposal bank contains 50,497 eligible actual ITF tracklets
within 365 days of UTC MJD 56880 and all 48 existing distance/radial hypotheses
(10–2,000 AU). This is a two-year common-epoch window, with pair gaps capped at 365 days.
The spatial-query bank has 2,545,467 eligible points across 10,247 nights, including
nine explicitly inserted reserved controls, and 262,144 deterministic distinct
source-ordinal queries. Half use their own night and half a later available night.
These large query timings test geometry, not physically propagated orbit recovery.

Warm medians include output downloads; initial resident uploads and compilation
are reported separately in the raw receipts. No simulator timings are included.

| Stage/workload | One CPU core |20 CPU threads/workers | GPU or hybrid | Comparison |
|---|---:|---:|---:|---|
| Projection,50,497x48 |1.912 s |0.139 s |0.0564 s |33.9x vs one core;2.46x vs20 |
| Night cone retrieval,262,144 queries |0.0478 s |0.00812 s |0.0173 s |GPU is2.13x slower than20 threads |
| Full epoch proposals,all 48 hypotheses |not measured |15.878 s old path |2.061 s hybrid |7.70x stage improvement |

Projection and cone retrieval each have three alternating-order repeats at all
recorded scales. Full-window proposals have two alternating-order repeats.
The hybrid uses resident GPU projection followed by a shared CPU spatial index.
Its two wall times were 2.0671/2.0559 s; the old CPU path took15.8186/15.9367 s.
Initial GPU upload/context cost in this last process was 0.366 s, outside warm timing.
Tree construction, candidate selection, worker coordination and saving/hashing
per-hypothesis edge arrays are included in the full-window stage timings.

The shared index queries spatial neighborhoods once across nights, then groups
by night and applies the same top 16/rate/time/identity/scoring predicates. It
avoids hundreds of thousands of empty night-tree calls. On4,096-row windows at
30, 80, 2000 AU it took 0.0123–0.0125 s versus 1.75–1.76 s (140.7–143.2x). On 16,384 rows,
it improved 28.4–35.0x. These smaller results do not extrapolate to the entire sky.

All 96 full-window comparisons (48 hypotheses × 2 repeats) have identical edge
memberships and score differences <=1e-8. Both return 4,966,048 hypothesis-specific
edges per repeat; these include overlapping proposals, not that many distinct
chains or objects. Projection finite masks match, and angular differences pass
the predeclared 0.00001 arcsecond check. All cone-query hits and their order match.

**Do not attribute 7.70x to GPU kernels alone.** The shared-index change removes
most of the repeated work. A 20-worker CPU-projection-plus-shared-index ablation
was not run on this worker. The separately paired projection result establishes
the GPU component's gain. No cost advantage over a separately priced CPU-only
machine or100x complete-search speedup is inferred.

## Long-gap development controls

All nine previously opened controls remain visible. Six have a previously
selected fitted representative; three (Eros,Vesta,Toro) have none. Every saved
mode of those representatives was used; in this dataset each has one. No refits,
new source retrieval, or favorable later-tracklet substitutions occurred.

The full available ITF point bank was augmented with the nine actual reserved
tracklets, which were absent as equivalent whole tracklets. Their known later
windows were supplied. Query inputs contain anonymous indices and frozen states;
retrieval and actual-time/station refinement were saved before reading grading
labels. The positions and earlier nominal prediction results were already opened
in previous development work, so this is not independent validation.

A fixed 120 arcsecond nominal screening cone is followed by the previously used
engineering 30 arcsecond per-detection diagnostic. This is not an uncertainty
confidence region or a newly deployed acceptance rule. CPU/GPU retrieval agrees:

| Control | Later tracklet screened and passes fixed refinement? |
|---|---|
| Varuna |Yes |
|2008KV42 |Yes |
| Chiron |Yes |
| Pholus |No |
|111309 |No |
|514107 |No |
| Eros,Vesta,Toro |No fitted representative; not tested |

There are three retrieved tracklets total, all the reserved counterparts, and
zero additional refined hits in this incomplete pool. That does not prove
uniqueness. Existing final nominal residuals and poorly extrapolating positives
remain limitations. The six-query CPU call takes about 17 microseconds versus 3 ms
for GPU: small follow-ups should remain CPU work. This implements/tests the
retrieval primitive, not full uncertainty propagation or unrestricted multi-year
archival corroboration. Those remain integration and scientific-validation work.

## New distant-object priorities and correctness repairs

`distant_brightness_priority.py` provides an optional distance-conditioned
photometric ordering. Magnitude 25 at 150 AU can have a brighter absolute-magnitude
proxy than magnitude 23 at 40 AU. Bands stay separate, missing inputs retain an
unsized lane, and diameters are returned only for explicit V photometry under
stated albedo/zero-phase assumptions. No production selection or physical size
claim changed. The earlier deterministic coverage planner remains available to
spread bounded work across epochs and hypotheses; full scheduling/quota integration
is still pending.

After the hardware run, an adversarial exact-radius test exposed a boundary
semantic difference: `query` excludes the exact upper bound, while
`query_ball_point` includes it. The final shared-index code restores the strict
boundary and delegates tied 16th/17th neighbor groups to the original nightly
tree. Original executed code, failed-boundary observation and amendment receipt
are preserved. A separate local replay of all 48 full-window outputs matches the
frozen hardware CPU edge memberships and scores. Hardware timings above belong
to the preserved pre-repair implementation; the corrected version was not
re-timed on a newly rented GPU. Ten final local tests pass; five GPU tests passed
on the actual H100 before the boundary repair.

The initial local input attempt encountered an old schema 1 cache and stopped;
the error is preserved. Inputs were then built from the existing schema 2 cache,
not rebuilt or reacquired. An initial utilization sampler stopped on a missing
report `phase` field; its error was saved and a separate repaired sampler ran.
Five-second sampling misses millisecond GPU bursts, and its inherited phase tag
is not a reliable stage label. No sustained GPU utilization claim is made.

## Reproduction and evidence

Code: `prepare_proposal_retrieval_benchmark.py`, `gpu_distant_projection.py`,
`gpu_night_query.py`, `benchmark_proposal_retrieval.py`,
`benchmark_sparse_epoch_join.py`, `benchmark_epoch_pipeline.py`, and
`distant_brightness_priority.py`. The changed query implementations are research
options; the existing production linker remains unchanged.

```bash
python research/solve_itf/prepare_proposal_retrieval_benchmark.py \
  --catalogue /path/to/schema 2/catalogue --sites /path/to/ObsCodes.html \
  --controls /path/to/itf-heavy-distant-20261028 --output /path/to/inputs
python research/solve_itf/benchmark_proposal_retrieval.py \
  --inputs /path/to/inputs --sites /path/to/ObsCodes.html --output /path/to/results
```

Full local evidence: `/tmp/itf-gpu-proposal-retrieval-20261029/`. This includes
frozen inputs, source hashes, designs, original failures, phase contracts, every
timing, actual control residuals, hardware metadata, all per-hypothesis edge
arrays in a 231 MB hash-verified archive, corrected-code replay and deletion receipt.
The metadata supplement required by the sparse/full-epoch benchmarks is frozen
separately; it preserves count, span and original identifier in the projection
bank's exact catalogue-index order. Compact receipts and original executed join
code are in `gpu-proposal-retrieval-receipts/`.

The next production integration should combine the faster proposal path with
explicit distance/time coverage, separate photometric priorities, and a protected
corroboration budget. It must preserve all finite-cap and uncertainty limitations.
No new dwarf-planet lead was sought or claimed in this benchmark.
