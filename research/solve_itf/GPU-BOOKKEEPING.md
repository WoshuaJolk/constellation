# CPU bookkeeping optimization and exact retrieval experiment

2026-10-28. **The requested further 100× end-to-end speedup was not achieved. New catalogue linking remains paused.** This iteration improves the controlled hybrid pipeline by about 2.5× over the previous run. It does not change the search scope or scientific cuts, claim distant-object detectability, or establish linkage reliability.

## Same-machine comparison

One H100 80 GB / 20-vCPU worker, $4.41/hour. The original 747-tracklet fixture, 256 ranked-seed cap, broad Gauss/range/radial starts, 80 optimizer iterations, residual checks, mode selection, formal uncertainty summaries, iterative graph extension and ranked candidate construction were preserved. All 275 chain fits and 12,703 starts remain. No production catalogue linking was run.

Three interleaved runs were made for each backend, reversing order in the second round. The first repetition is reported separately because it includes additional initialization, especially CUDA compilation. The warm summary is the median of the remaining **two** repetitions, not a high-confidence performance estimate.

| Backend | First repetition | Warm repetitions | Warm median |
|---|---:|---:|---:|
| Original CPU bookkeeping, 20 fit threads | 4.302 s | 3.864 / 3.934 s | 3.899 s |
| Revised CPU bookkeeping, 20 fit threads | 1.558 s | 1.521 / 1.600 s | 1.560 s |
| Revised hybrid GPU, including CPU checks | 8.434 s | 1.492 / 1.380 s | 1.436 s |

The revised CPU path is **2.50×** faster than the original CPU path on this worker. The revised hybrid path is **2.71×** faster than that original CPU baseline and **1.09×** faster than the revised CPU path. Relative to the previous 3.631 s warm hybrid result on another identically sized worker, the improvement is **2.53×**; that latter comparison includes host/run variation.

The requested 100× improvement over 3.631 s would require about **0.0363 s**. The revised hybrid median is still approximately **40× above that target**. Stage-only speedups do not close that gap.

Final-repetition stage timings, included within the outer runtimes above:

| Stage | Original CPU | Revised CPU | Revised hybrid |
|---|---:|---:|---:|
| Candidate graph scan | 0.207 s | 0.206 s | 0.202 s |
| Seed generation | 0.0025 s | 0.0025 s | 0.0026 s |
| Observer preparation | 0.043 s | 0.043 s | 0.042 s |
| Triple fitting, selection, uncertainty | 3.281 s | 1.162 s | 0.911 s |
| Extension | 0.392 s | 0.180 s | 0.215 s |

The three GPU solves together take 0.178 s. Planning takes 0.277 s, GPU workspace setup 0.036 s, and CPU state verification 0.046 s. Mode selection and Python orchestration still consume substantial time. Even the graph scan alone exceeds the target. A further 100× end-to-end gain has not been justified by these measurements.

## Changes

- Prepare each chain's observation geometry and exact ordered starts once. The GPU adapter no longer calls the full fitter with dummy failed results just to collect starts, then prepares the chain again to replay them.
- Compile the small-array orbital-element, residual/per-tracklet, canonical-state and duplicate-comparison operations. Preserve float64 arithmetic and disable fast-math.
- Compile the original column-scaled SVD uncertainty calculation. Do not replace it with inverted normal equations or alter the assumed astrometric noise.
- Calculate Jacobian rank only for modes that remain in the final result, rather than intermediate modes later replaced by a better duplicate.
- Preserve CPU verification of every returned GPU state and exact-seed CPU fallback on a failed verification. This fixture triggered zero fallback starts; earlier cold-bank failures remain unresolved and are not erased by this result.

A post-hardware metadata guard binds a prepared plan to its original start policy: a scout plan cannot be relabelled `full`. This guard was tested locally, including GPU simulation; it does not change calculations on the valid benchmark path. The hardware payload hashes preserve the exact version that ran.

## Output checks and numerical limits

On the H100 host, original and revised CPU paths retain the same **259 memberships and 944 modes**, with exact agreement in fitted states, epochs, chi-squared, RMS, rank and relative mode weights. Every simulated object still reaches its six-tracklet chain. GPU retains the same 259 memberships and 943 modes as previously; equal memberships are not proof that CPU and GPU explore identical orbital basins.

Derived range and uncertainty arithmetic are not bit-identical. Local tests found maximum range change 1.82e-12 AU, causing an initially overly strict bitwise range assertion to fail. That failure and its separate reconciliation are preserved. The hardware comparison prospectively used 1e-14 relative / 1e-12 AU absolute tolerance for derived range, and 1e-8 relative / 1e-10 absolute for uncertainty summaries. These are engineering comparison tolerances, **not changed fit acceptance cuts**. The largest hardware uncertainty relative difference was 8.85e-10. Rank/weak-linearization flags agreed. This is observed fixture parity, not a guarantee for every ill-conditioned input or boundary case.

The local machine uses different NumPy/Numba/Astropy versions and retains 946 modes with either CPU implementation. That environment difference is not counted as an optimization effect. Local old/new paired timings were 8.61/10.47/11.73 s versus 3.76/4.24/5.03 s; they should not be substituted for the H100 host comparison.

## Long-gap spatial retrieval: negative optimization result

A separate exact-query experiment indexed **2,545,458 existing local tracklets across 10,246 nights**. It uses the same motion-to-night-center representation as the current night index. A deterministic set of 4,096 evenly spaced eligible original catalogue ordinals plus 4,096 fixed-seed isotropic directions was queried at each of five cone radii. The design was saved before queries. This tests spatial-query machinery, not uncertainty validity, recovery recall, actual-time prediction or full archival linking.

The proposed index stores all three coordinate orderings, chooses the narrowest slab, then applies the unchanged spherical chord-distance test and restores original output order. It produced exactly the same points, ordering and distances as the existing compiled single-axis query in every test, but was **slower**:

| Cone radius | Queries | Returned point/query matches | Existing/new speed ratio |
|---|---:|---:|---:|
| 1 arcsec | 8,192 | 4,362 | 0.182× |
| 1 arcmin | 8,192 | 4,889 | 0.148× |
| 0.1 degree | 8,192 | 11,008 | 0.210× |
| 1 degree | 8,192 | 260,132 | 0.374× |
| 10 degrees | 8,192 | 3,384,169 | 0.447× |

The additional index costs 61.1 MB and 0.471 s to build. It remains an **unwired optional experiment**; the production `NightIndex` still uses the previous query. The slower result is preserved rather than presented as a speedup or used to change cone widths.

## Tests, cost and next bottleneck

- Local existing fitter, batch adapter and index tests: 35 passed, one CUDA-only test skipped; three additional bookkeeping/parity tests passed.
- H100 suite: 24 passed, one simulator-only test skipped. Its small test grids emit expected low-occupancy warnings.
- Focused final policy-guard and simulator tests are stored with the receipts.
- Worker 43 was deleted after output hashes were verified; provider returned zero remaining Droplets. This iteration cost an estimated **$0.31636** in compute, bringing GPU development to **$4.39179 / $5**. These are rate-times-lifetime estimates, not an invoice. No snapshots or storage volumes were created.

The remaining architectural opportunity is to remove Python planning/selection and repeated host/device representation changes, then measure graph construction and planetary uncertainty propagation on representative workloads. This is work still required, not a forecast of a 100× result. Simply making the current GPU solve faster cannot deliver the requested whole-pipeline improvement. Long-gap uncertainty retrieval still needs an end-to-end benchmark with the original physical and numerical checks; this spatial-index experiment does not supply one.

## Reproduce

Restore the compact fixture as described in `GPU-PIPELINE-AND-DISTANT.md`, then:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python research/solve_itf/gpu_bookkeeping_benchmark.py \
  --catalogue /tmp/itf-distant-fixture --sites /path/to/ObsCodes.html \
  --output /tmp/bookkeeping-results
```

Add `--cpu-only` without CUDA. Original baseline source is pinned in `gpu-bookkeeping-receipts/reference_fit.py`. The local real-catalogue query experiment is:

```sh
PYTHONPATH=src python research/solve_itf/night_index_benchmark.py \
  --catalogue /path/to/existing/compact --output /tmp/night-index-results
```

`gpu-bookkeeping-receipts/` contains the designs, hardware/toolchain metadata, raw timing and parity reports, preserved failed range assertion, source/payload hashes, tests and verified deletion receipt. Full returned fit/audit outputs remain in `/tmp/itf-gpu-bookkeeping-20261028/fit-results`; raw local experiment outputs are in `/tmp/itf-speedup-20261028/`. No cloud machine is needed to read them.
