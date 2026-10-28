# Whole-search GPU comparison and distant-object development controls

October 28, 2026. **100× end-to-end acceleration was not demonstrated.**
The previous 12.6× resident-kernel result does not transfer to this small complete
linking workload: after optimization, warmed CPU and hybrid GPU searches are
approximately tied. This experiment locates the remaining cost and checks that
slow, distant examples survive the search. It is not a production catalogue run.

## Fixed workload and comparison

The fixture contains **675 actual local ITF background tracklets and 72 synthetic
tracklets**, representing twelve objects with six observing nights each. It uses
all available background rows in UTC MJD [60000,60010), below the prospectively
chosen 2,000-row cap. No cases were replaced after seeing fit or retrieval results.

The independent synthetic generator integrates two-body motion with SciPy
DOP853 (rtol 2e-12, atol 1e-14), applies observer reflex and iterated light time,
and adds fixed-seed 0.1″ Gaussian positional noise. Radii are 8, 40, 80, 150, 300,
and 1,000 AU; each has a low-eccentricity 15° and an eccentric 120° case. Twelve
controls are too few to establish population completeness. Geocentric station
500, simple noise and a sparse background are favorable simplifications.

Both paths run the existing linear and curved candidate graph, seed ranking,
observer preparation, full Gauss/range/radial starts, original mode-selection
and residual/physical cuts, iterative chain extension, and ranked candidate
output. The fixed search cap is 256 triple seeds; the run attempts 275 chain
memberships and 12,703 optimizer starts, including extensions. The cap remains
visible: the graph considered 775 triple combinations. This is not the full
2.5-million-tracklet/7-million-seed campaign.

CPU fitting uses twenty threads, with compiled solver calls. The experimental
GPU adapter batches starts, evaluates every final GPU state on the CPU, and
replays an original CPU fit from the same seed if numerical validity or objective
agreement fails the existing 0.001″ comparison. CPU verification/replay time is
included. Zero starts needed fallback in this particular fixture. The 28
incompatible states in the earlier cold-bank benchmark remain separate failures.

The adapter injects only an optimizer into `fit_chain`; the same function still
plans starts and performs selection. A research-only `batch_fitter` hook in
`search.solve` keeps the original graph/extension implementation. Defaults and
CLI behavior are unchanged. This is not an enabled production GPU backend.

## Stage timings

The table uses the second run on the final H100/20-vCPU worker, after warm-up.
There is only one warmed end-to-end timing per backend per worker; these small
differences are not evidence of a statistically meaningful speed advantage.

| Stage | CPU, seconds | Hybrid GPU, seconds |
|---|---:|---:|
| Linear + curved candidate scan | 0.2046 | 0.2057 |
| Triple-seed generation | 0.0024 | 0.0025 |
| Observer preparation | 0.0427 | 0.0429 |
| Triple fitting + verification + mode selection | 2.9949 | 2.9494 |
| Chain extension | 0.3617 | 0.4219 |
| Solve, including remaining ranking/overhead | 3.6064 | 3.6225 |
| Outer search wall time | **3.6145** | **3.6307** |

First-run CPU/GPU outer times were 4.038/10.487 seconds. The GPU first run
includes CUDA compilation/context overhead. Raw runs are retained; no favorable
median is manufactured by mixing cold and warmed runs. Input archive generation,
source download and JSON export are outside these remote search timings.
Serialization of each saved ~1.18 MB report was separately timed locally at
roughly 0.016–0.021 seconds (buffered writes, no fsync); that is not a GPU speedup.

The first adapter took 4.2968 seconds warmed versus 3.8666 seconds on that
worker's CPU. We then replaced per-start Python verification tasks with compiled
CPU blocks and removed repeated geometry comparisons during replay, preserving
seed/offset/policy checks. CPU verification across its three batches fell from
0.7748 to **0.0427 seconds** across the two recorded workers, about 18×. Small host
variation is visible in the corresponding CPU baselines. Total GPU-pipeline time
fell to 3.6307 seconds; the CPU baseline remained competitive.

GPU solves themselves account for only **0.1773 seconds** in the final warmed
run. Allocation/transfer, planning, CPU verification and especially mode replay/
bookkeeping dominate. Making just those kernels infinitely fast would improve
this measured search by only about **1.05×**. A claim of 100× for the complete
search therefore requires substantial changes outside those kernels as well.

## Recovery and output differences

Both CPU and GPU recovered **all 12 injected objects as six-tracklet chains**.
They retained exactly the same **259 chain memberships**, including the same
12 maximal candidates; no mixed-identity chain was retained in these controls.
Those development labels are used for grading, not by the linker.

The complete mode sets are not identical: CPU retained **944** modes, GPU **943**.
Small optimizer differences can change which modes are merged or retained even
when all chain memberships agree. We do not claim orbital completeness,
bit-identical full-search output, calibrated reliability, or actual discoveries.
The tiny control count and favorable field selection cannot establish those.

## Long-gap check: a separate limitation

We reserved one complete four-observation tracklet per object at +365 days and
another at +730 days. They never entered fitting or initialization. Every retained
mode was tested against the same 24 reserved tracklets, at actual timestamps,
using the predeclared nominal 5″ diagnostic limit. The current short-arc
extension default is max(5″, 10 × 0.6″) = 6″, so this later-window diagnostic
is not an exact replay of that production gate. There was no refitting,
threshold adjustment or favorable alternative-window search.

Both paths found **the same 4 of 24 later tracklets**, producing 31 mode/tracklet
hits. Three identities supplied these hits: the 300-AU low-inclination object and
both 1,000-AU objects. Only the 1,000-AU retrograde case supplied a two-year hit.
CPU had 624 and GPU 636 nonfinite mode/tracklet prediction evaluations; these are
repeated evaluations, not counts of false objects. These raw failures are saved.

This is an easier, fixed-window nominal-prediction diagnostic, not a full
identity-blind archival census. Its future pool contains the 24 controls, not a
representative real sky. The short-arc recoveries and the poor later recovery
must remain separate: a good short-arc fit does not pin a long-term ephemeris.
Uncertainty propagation and alternative-mode retrieval are the next substantive
long-gap tasks. The existing planetary long-arc fitter was not GPU-translated or
benchmarked here; no claim is made for that later stage.

## An optional distant proposal bank

`magictools.itf.distant.distant_seeds` supplies additional starts over 64 log-spaced
geocentric distances from 30–2,000 AU. For each distance it computes the interval
of radial adjustments giving a bound heliocentric velocity, accounting for
observer motion, and samples nine interior points. It does not replace the
ordinary Gauss/all-conic starts or impose a new final-fit energy cut.

This is only a chord-motion approximation, not an exact admissible region.
No bound proposal at a grid distance does **not** exclude the distance or object.
Astrometric errors, cadence and coarse grid spacing can omit valid solutions.
The generator produced 9–108 starts per distant control, and none for the two
8-AU controls outside its configured distance lane. Exact per-distance counts
and unsupported distances are preserved. Unit tests check the bound-energy
construction, observer-velocity offset, invalid inputs and nonmutation.
**These extra starts were not fitted or added to the frozen CPU/GPU comparison.**

Useful next experiments are a denser/adaptive distance grid, comparison of
multiple retained modes' later predictions, and GPU batches large enough to
amortize planning/selection overhead. Each needs recovery and failure accounting,
not merely a kernel timing. No 100× gain or improved long-gap recovery is inferred
from generating the extra seeds.

A distant-size prize also needs photometry and survey coverage. For illustration,
a 1,000-km body with geometric albedo 0.1 has H≈3.12; the simple opposition relation
m≈H+5log10(rΔ), with r≈Δ, gives m≈19.1 at 40 AU, 24.9 at 150 AU and 33.1 at 1,000 AU.
These rough assumptions are not a detectability model. Recovering an injected
1,000-AU position does not show that such a dwarf planet is detectable in ITF.
The compact orbit fixture carries no magnitudes; original photometry would need
to be joined before making size-based candidate priorities.

## Reproduce and audit

The 82,952-byte `gpu-pipeline-fixture/fixture.npz` preserves the exact background
and synthetic observation/tracklet arrays, with separate truth, reserved data,
manifest and hashes. It is a development fixture, not sealed evaluation material.

```bash
python research/solve_itf/gpu_deep_fixture.py \
  --archive research/solve_itf/gpu-pipeline-fixture/fixture.npz \
  --output /tmp/itf-distant-fixture
python research/solve_itf/gpu_pipeline_benchmark.py \
  --catalogue /tmp/itf-distant-fixture --sites path/to/ObsCodes.html \
  --output /tmp/itf-pipeline-results
```

Frozen designs, source hashes, raw reports, tests and deletion receipts are in
`gpu-pipeline-receipts/`. Full case reports/audits and original source payloads
are under `/tmp/itf-gpu-pipeline-20261028/` and
`/tmp/itf-gpu-pipeline-fast-20261028/`. The first adapter source is preserved too.
Local regression: 32 original linker tests passed; new batch/verification/distant
checks passed. Final hardware suite: **23 passed, one simulator-only test skipped**.
The two workers cost an estimated **$0.63366**; the combined GPU prototype,
connection and benchmark sequence is **$4.07543 / $5** (pretax rate × lifetime,
not an invoice). Zero Droplets remained at the final provider check.
Both workers were deleted after hash-verified returns. No old scientific snapshots
or production evidence were deleted, and no old search/certification campaign was
restarted.

The subsequent [bookkeeping optimization](GPU-BOOKKEEPING.md) reduces the warm
hybrid controlled pipeline to 1.436 s (median of two warm repeats), with unchanged
CPU states and memberships. The further 100× target remains unmet and new
catalogue linking remains paused. It also preserves a negative exact spatial-index
benchmark on 2.55 million local tracklets; that index is not enabled.
