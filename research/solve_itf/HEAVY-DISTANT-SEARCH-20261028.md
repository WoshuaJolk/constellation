# Expanded distant-object search — 28 October 2026

This development run prioritizes potentially larger distant bodies while preserving nearby, retrograde, hyperbolic and ambiguous candidates. It is not a discovery or size-certification procedure. The first pass's five-year, nominal 39 AU candidate is logged separately: its rough 80–200 km diameter estimate treats r=23.6 as a visible-light proxy and assumes 4–25% albedo. Its association, distance, band conversion and physical size remain uncertain.

## What changes

The unlaunched seed-count-only proposal is superseded. New proposals propagate each trial state with the existing checked solar two-body universal-variable solver, rather than constant acceleration. Common reference epochs are 180 days apart, with observations within ±365 days, so a pair separated by at most 365 days can share an epoch window. Projections are cached within each epoch and hypothesis instead of integrated again for each night pair. Approximate observer geometry, assumed instantaneous tracklet motion, coarse range and radial grids and the two-body model remain limitations; this is not the final planetary fitter.

The 16 trial geocentric distances are 10, 20, 30, 35, 40, 45, 50, 60, 80, 100, 150, 250, 400, 700, 1,000 and 2,000 AU. Radial velocities are −0.5, 0 and +0.5 times the escape speed at the trial distance. These are proposal hypotheses, not bound-orbit acceptance restrictions. The broad all-conic fitting starts and every numerical/residual cut remain unchanged.

All 2,548,307 catalogue tracklets remain available. New proposals use the previous slow-tracklet eligibility conditions. Finite night-neighbor, aperture and edge caps remain explicit. Each of 20 proposal partitions has a 480-second limit; partial partition coverage must be reported. The maximum search aperture is one degree, with a smaller rate-error/gap-dependent aperture where applicable. It is a proposal envelope, not a calibrated error model.

Up to eight million ranked seed proposals are generated. The previous distant run's million seeds are removed exactly; up to seven million additional seeds are fitted. Up to 1.5 million slots prioritize at least two tracklets with median reported magnitude ≤22.5, and 1.5 million prioritize spans above 120 days. Remaining/unfilled slots use the normal graph ranking. Deterministic round-robin scheduling mixes these groups. Magnitude bands are not converted, missing photometry remains eligible for the ordinary group, and no fit outcome selects a seed.

The previous 17,985 memberships and their alternative modes are preserved. The new run must not count those as new discoveries.

## Recovery checks before production

Nine already-opened known-object development histories supply their first eligible 365-day season, with up to twelve timestamp-quantile tracklets. The first later complete tracklet at least 180 days beyond the fitted set is reserved. These longer seasons differ from the earlier six-tracklet control; this is not an isolated algorithm A/B comparison. Catalogue labels are development assumptions. All 28 reserved detections were checked for same-station near-duplicate input overlap at 1 ms/1 mas: none was found.

Search is anonymous. Every pure recovered mode is saved and predicted at its reserved observations without refitting. The descriptive representative uses fitted span, number of tracklets, input RMS and deterministic membership/mode tie breaks, never reserved residuals.

A prospective engineering gate requires at least one Varuna or 2008 KV42 representative with ≥60 days of fitted arc, a nominal bound solution at 25–100 AU and reserved nominal RMS ≤30 arcseconds. This is a modest prerequisite for spending on this run, not an association-acceptance rule, calibrated coverage estimate or proof of completeness. Failure stops production; it must not be relabeled as a pass or repaired by choosing a favorable reserved set.

Seven local tests pass, including agreement with independent DOP853 integrations over one year at 30, 100 and 1,000 AU, nonzero radial motion, epoch-window coverage, exclusion of prior seeds and deterministic quota selection. Hardware tests and real-control outcomes are separate prerequisites.

## Planetary corroboration and size context

The fixed follow-up budgets are 256 nominally bright-for-distance bound distant modes, 128 bound distant modes ordered by measured arc, and 64 each for retrograde, hyperbolic, NEO and general modes. Exact duplicate states are merged and the 189 previously attempted follow-up states are excluded. One input-RMS-best mode per membership/lane is selected; all other saved modes remain available.

Brightness-for-distance uses available unconverted photometry and nominal heliocentric/geocentric distances. It ranks hypotheses only. A weak distance can imply a spurious large size, so outputs must retain orbital ambiguities and photometric caveats. No minimum physical diameter is certified by this ranking.

Existing Sun/planet/Moon propagation with DE440s searches the full catalogue for corroborating tracklets and jointly refits proposals. Two growth rounds, 12 local uncertainty probes, two-degree retrieval cones, finite proposal caps, 60 seconds per parent, ten seconds per fit and a 900-second cohort cap remain. Numerical failures, incomplete searches and untested alternatives are unresolved. This does not provide calibrated identity probabilities or eliminate all false linkages.

## Resource and provenance contract

One H100/20-vCPU worker at $4.41/hour, maximum $20 incremental authorization and an absolute 90-minute lifetime. The actual deadline is recorded once in the launch receipt. Controls precede production. A single-writer watcher collects and SHA-256 verifies outputs before deleting the worker. The deadline guard stops only this run and preserves unreturned evidence by snapshot if necessary. All prior evidence remains intact. No recurring reporting automation is created.

Design, source/input hashes, control selection, launch intents, runtime receipts and outputs live under `/tmp/itf-heavy-distant-20261028`. Read the final result/resource receipts before asserting completion. Implementation: `distant_epoch_graph.py`, `heavy_distant_search.py`, `heavy_distant_controls.py`, and `heavy_distant_followup.py`. Production dashboard and certification rules are not changed.


## First execution observations

Hardware preflight: **51 passed, one skipped**. The engineering gate passed and production started automatically. Varuna recovered eleven tracklets over 331.49 days, with nominal range 43.08 AU and eccentricity 0.055; the first reserved tracklet, 299 days later, has nominal RMS 2.621 arcseconds. KV42 recovered nine tracklets over 327.27 days, range 31.19 AU, eccentricity 0.498, and reserved RMS 7.246 arcseconds. Chiron's reserved RMS is 6.034 arcseconds. Pholus, 111309 and 514107 still predict poorly (7,135.8, 246.4 and 860.3 arcseconds); Eros, Vesta and Toro were not recovered. These failures are preserved. Varuna is the only large-TNO control in this small existing set; broad dwarf-planet completeness has not been measured.

This supersedes neither the earlier run's failures nor its receipts. Longer control input seasons and a different proposal algorithm changed together, so the improvement is not an isolated algorithm speed/accuracy comparison. The full search and follow-up remain pending until final receipts say otherwise.
