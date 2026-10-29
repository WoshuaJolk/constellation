# Distant-priority ITF search — 28 October 2026

This is a bounded development search for distant-object leads, with other conics and scientific opportunities retained. It is not a discovery claim, a dwarf-planet classification, or an association reliability estimate. The primary target determines where new proposal work goes; it does not discard nearby, retrograde, or hyperbolic fitted modes.

## Design and execution

The starting catalogue has 2,548,307 tracklets. The dedicated lane uses existing measured rates and real positions: at least two observations, a positive duration no longer than 0.6 day, and angular speed at most 600 arcseconds/day. This leaves 623,221 seed-eligible tracklets. Other tracklets are still present in the full recovery catalogue and the existing general lane.

A distance-conditioned midpoint projection evaluates trial ranges of 10, 20, 30, 40, 60, 100, 150, 300 and 1,000 AU. The proposal model uses zero trial radial velocity, terrestrial rotation, Earth reflex motion and a solar-acceleration approximation. The fitter has its full existing broad radial/range/Gauss starting grid; the proposal approximation is not final dynamics. Pair gaps are at most 120 days, with a fixed angular aperture of `30 + 0.1*gap_days²` arcseconds and 16 neighbors per day/distance. Triples can span more than 120 days because two retained pair edges can meet at a middle tracklet.

The full fixed scan completed in 82.744 seconds. It made 547,045,965 queries across 8,500 eligible source nights and retained 644,477 distinct connections. There were 1,685 queries at the neighbor cap. Neither the per-partition nor global edge-pool cap discarded edges. The finite angular/rate/distance/neighbor restrictions still limit recall.

Seed construction considered 3,579,205 combinations and froze the first 1,000,000 under the existing ranking. Their median span is 112.64 days; 965,181 span more than 60 days. Unselected combinations are not fitted rejections. The general-search lane reuses the prior completed million-seed result with its source hash; those candidates are not counted as newly found by this run.

The opt-in streaming GPU fitter uses the existing all-conic solar two-body model, 80-iteration limit, assumed 0.3-arcsecond astrometric sigma, full-fit and per-tracklet RMS limit of 0.6 arcseconds, and maximum single residual limit of 1.2 arcseconds. The prior perihelion and numerical cuts remain. Every GPU result retains the existing CPU residual/Jacobian check and fallback. Initializations, convergence failures and finite search caps prevent any claim of exhaustive orbit-family coverage.

The new software passed 46 hardware tests with one skip. A separate local proposal regression matched the existing projection implementation exactly. The local test-only addition after packaging does not change the executed scientific code manifest.

## Real-object recovery check

Before the full scan, existing observations of nine catalogue identities were anonymized and mixed with 675 real background tracklets. For each identity the first available season with at least three eligible tracklets within 120 days was selected, taking up to six. The first complete tracklet at least 180 days beyond the fitted season was reserved. Catalogue identities are development labels, not independent adjudication. Neither labels nor reserved positions were supplied to proposal construction, fitting or initialization.

There are 726 tracklets in the control catalogue. All 27 reserved observations are absent from the input, both as exact station/time/position keys and under a same-station 1-ms/1-mas near-duplicate check. This is a small availability-selected experiment, not representative survey completeness. Old controls also inherit the logged limitations of builtin ephemerides and historical Earth-orientation data.

| Identity | Supplied tracklets | Longest recovered pure chain | Arc |
| --- | ---: | ---: | ---: |
| Varuna | 6 | 6 | 12.23 d |
| 2008 KV42, retrograde TNO | 6 | 6 | 64.98 d |
| Chiron | 6 | 6 | 82.51 d |
| Pholus | 6 | 5 | 9.24 d |
| 111309 | 6 | 5 | 54.57 d |
| 514107, retrograde co-orbital | 6 | 4 | 4.91 d |
| Eros | 6 | 0 | — |
| Vesta | 3 | 0 | — |
| Toro | 6 | 0 | — |

There were no mixed-control retained chains in this small background, which does not establish production purity. The prospective minimum gate—recover at least one pure chain for Varuna or KV42—passed. This gate establishes useful proposal/linking behavior, not correct orbital recovery.

### Tracklet recovery is not orbit recovery

The later-observation diagnostic predicts every pure retained mode without refitting. A descriptive representative is selected by longest chain, then longest fitted arc, input-fit RMS and membership, with the minimum input-fit-RMS mode. Reserved residuals do not select the representative.

| Identity | Representative input RMS | Reserved gap | Reserved nominal RMS |
| --- | ---: | ---: | ---: |
| Varuna | 0.067″ | 319 d | 8,455″ |
| KV42 | 0.209″ | 235 d | 8.29″ |
| Chiron | 0.197″ | 290 d | 15.43″ |
| Pholus | 0.328″ | 321 d | 7,136″ |
| 111309 | 0.445″ | 311 d | 246″ |
| 514107 | 0.178″ | 268 d | 703″ |

Varuna's full six-tracklet solutions are hyperbolic, with weak local linearization and unresolved energy sign, despite the correct membership. This is a direct warning against treating a low RMS, fitted distance, or hyperbolic mode as the physical answer. KV42's and Chiron's longer arcs extrapolate much better, but these nominal predictions do not establish uncertainty coverage or reliable archival recovery. All alternatives and all prediction outcomes remain saved.

## Follow-up and interpretation

The registered follow-up gives 128 slots to distant modes and 32 each to retrograde, hyperbolic, NEO and general candidates. Exact duplicate selections across lanes are merged. It selects one lowest-input-RMS mode per chain/lane, then orders by arc length, tracklet count, input RMS and hash. This finite selection can miss a physically useful alternative; all original modes remain in the output.

Follow-up uses existing Newtonian Sun/planet/Moon dynamics with DE440s, an uncertainty-guided full-catalogue night index, two growth rounds, 12 uncertainty probes, 16 proposals per window, and a two-degree maximum cone. Per-parent and per-fit budgets are 60 and 10 seconds, with 900 seconds for the cohort. Numerical failures, exhausted budgets, and untested branches remain unresolved. Passing a refit is not confirmation. The current error model and local uncertainty are not calibrated.

Known-object annotations reuse the pinned SkyKnown export with exact catalogue-index provenance. A returned common match is useful attribution evidence; no-match does not establish novelty. Snapshot coverage, older astrometric sensitivity and incomplete/unmapped rows remain explicit.

Reported size context, where V-band data exist, is conditional on the nominal fitted distance, median reported V magnitude, zero phase correction and assumed albedos of 0.04 and 0.25. Other bands are retained without conversion. Neither a large conditional diameter nor any orbital class establishes a dwarf planet. Hyperbolic fits are not interstellar identifications.

## Evidence and resource limits

The prospective design, source/code hashes, control selection, failures, phase receipts, complete candidate/mode outputs, selected follow-ups, photometry and known annotations are preserved under `/tmp/itf-distant-search-20261028`. No production dashboard or certification filter is changed.

One H100/20-vCPU worker is authorized, at $4.41/hour, with a 90-minute absolute lifetime and $20 total incremental ceiling. The lifecycle controller returns and verifies new evidence before deletion; an absolute-deadline guard preserves evidence before resource cleanup. The completed worker lifetime cost an estimated **$1.2853**. All 78 returned files were SHA-256 verified before deleting the worker. A later provider check confirmed **zero droplets**. Snapshot holdings from previous campaigns were not changed.


## Completed results

All **1,000,000 frozen distant seeds** finished. Including extension work, the solver performed 1,000,447 fits and retained 6,460 chain rows. Production took 593.44 seconds including proposal construction; the solver took 489.65 seconds. These times reuse cached catalogue geometry and exclude initial raw-data ingestion and worker setup. Combined with the prior general result, there are 17,949 distinct memberships; 5,118 are absent from that prior million-seed report. All conics, nearby objects, retrograde modes and ambiguous alternatives are preserved.

The original planetary follow-up selected 161 distinct modes, with 146 searched and 15 partial failures. It returned 18 row versions, all hyperbolic. Its longest-arc ordering had crowded out every best-mode bound distant candidate. This limitation was discovered and recorded, then a separate fixed amendment selected **all 28** such candidates before inspecting further outcomes. The same dynamics, cuts and per-parent budgets ran locally in 48.98 seconds, producing 39 row versions: 20 parents searched, six partial failures and two numerical failures. No additional cloud resources were used.

Together the planetary follow-ups preserve 57 row versions / **47 distinct memberships**. **21 have a bound distant solution**, falling into **five connected shared-tracklet families**. Families group competing or nested associations; they are not five confirmed objects. Full-catalogue retrieval was finite: proposal caps, local uncertainty and numerical failures limit completeness. In the first follow-up, 218,031 raw proposals and 1,082,466 arc-merge proposals were discarded at configured caps; these are not rejected astronomical associations.

| Family representative | Tracklets | Arc | Nominal range | RMS | Exact membership in earlier frozen inventory? |
| --- | ---: | ---: | ---: | ---: | --- |
| H250540 + H311994 + H365156 + H366550 + H459783 + H489530 + LA5424 | 7 | 1,831.05 d | 39.36 AU | 0.151″ | No |
| H222383 + H435252 + H484585 + H484592 + H484609 + H485634 + H492491 | 7 | 1,212.76 d | 36.76 AU | 0.232″ | Yes |
| H239723 [661001, 661002] + H397290 + H476685 + H476687 + H487533 | 6 | 419.88 d | 32.76 AU | 0.179″ | No |
| H269351 + H273962 + H273967 + H308587 | 4 | 354.08 d | 41.03 AU | 0.179″ | No |
| H227873 + H321307 + yy0000b + yy0000c | 4 | 83.97 d | 34.83 AU | 0.173″ | No |

H239723 has two distinct catalogue tracklets on different nights (MJD 57636 and 57638), not duplicated detections; immutable indices disambiguate its repeated display designation. That family's known-object coverage is incomplete. The other four representatives have complete-no-match annotations in the pinned export, with all its earlier limitations.

For all five representatives, a frozen-state planetary prediction replay accounts for every detection and exactly reproduces saved angular RMS. Detailed dossiers retain observations, states, alternatives and signed residuals. This confirms numerical accounting, not association correctness. The five-year lead has 27 observations, fitted a = 40.85 AU, e = 0.071 and i = 25.63°. Its available photometry is roughly magnitude 23–24 in differing bands, with no V-band measurement; it does not establish a dwarf-planet size or brightness. It warrants follow-up as a distant candidate.

The exact-membership comparison uses all **312,133 frozen fit versions**, reconstructing catalogue indices from verified annotations and checking membership lengths. Across the initial-plus-planetary union, 9,324 of 17,985 memberships are absent from that inventory. Among the 47 planetary memberships, 35 are absent. Absence is neither an independent new object count nor a claim that no other campaign or the MPC has the association.

## Reporting corrections and reproducibility

The injected frozen seed queue originally left two raw status counters inconsistent: negative remaining seeds and an incomplete-scan flag despite all seeds and partitions returning. `accounting-correction.json` reconciles these against immutable counts and receipts. Executed source and original outputs remain unchanged; the shipped correction only propagates frozen queue metadata. A separate table correction uses the full observation span rather than mean-tracklet-time span and tests exact membership against the general baseline for planetary rows too. No fits or thresholds changed.

The sibling `distant-search-receipts/` directory contains the prospective design, executed-source hashes, controls, outcome summaries, resource deletion receipt, corrected candidate table, planetary states and five-family dossiers. Full numerical return hashes identify the retained raw outputs under `/tmp/itf-distant-search-20261028`. The large catalogue and physics kernels remain external dependencies. Reproduce using `distant_search.py`, `distant_followup.py` and the read-only `distant_report.py`; inspect their `--help` and the committed designs before allocating a worker. This first bounded search is complete; no automated successor or production rule was deployed.
