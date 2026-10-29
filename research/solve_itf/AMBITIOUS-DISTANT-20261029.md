# Ambitious distant search — 2026-10-29

This is an authorized finite discovery attempt, not the old certification mission. The user requested the strongest overnight dwarf-planet-oriented search. The assistant prospectively bounded it to **one H100, $20 incremental compute, and four hours including setup/return**. Do not extend/reset that clock. The original `launch.json` supplies the immutable deadline. Worker 56 is the sole new machine. Credentials remain in the protected fleet helper; never print them.

## Frozen science

`ambitious-design.json`, `freeze-receipt.json`, and both mission manifests preserve the inputs/code. This considers the 2,548,307-tracklet catalogue for eligibility. It prioritizes distant and potentially intrinsically bright modes while preserving ordinary/retrograde/hyperbolic/NEO lanes. These are fit classifications, not confirmed objects.

- Common-epoch GPU proposals: 21 ranges from 10–2,000 AU for year gaps; nine ranges from 40–2,000 AU for gaps up to three years; five radial fractions. Hashed epoch order/interleaved windows prevent chronological starvation. Finite caps and incomplete grid coverage remain explicit.
- Shared spatial index replaces many empty per-night queries. Prior hardware benchmark measured **7.70×** for the full 50,497-tracklet proposal pipeline, not a 100× whole-linker gain. New grid has not been benchmarked end-to-end against the old search.
- Same-band photometry priority: no band conversion; no measured diameter claim. 2,405,877 tracklets have a usable selected band.
- Up to 20M additional triples across six distance/gap lanes. Exclude the entire prior 8M seed queues, **including their previously unfinished suffix**, to prioritize new combinations. This is not completion of all prior work.
- Added bound distant initial guesses preserve ordinary all-conic starts. Existing fit settings and CPU residual verification stay unchanged.
- Protected planetary follow-up: multi-mode selection, maximum four modes per overlapping family per lane, more uncertainty probes/branches, DE440s planetary dynamics, all failures retained. This is finite retrieval, not complete orbital uncertainty.
- Prior baseline contains 22,168 memberships, including previous planetary follow-up. New membership does not imply a new object.

## Execution and protection

`run-search.py` advances controls → proposals → fixed seeds → fitting → planetary follow-up. The selected follow-up reserve is protected before fitting. The 12 local non-GPU tests and four GPU-simulator parity tests passed. Hardware tests and the fixed opened-real-control adapter check gate production. Opened controls are not independent scientific validation.

`controller-pids.json` records single local controller PID/start ticks. `guard.py` enforces the original deadline. `watch.py` collects after terminal science. `collect.py` inventories and verifies returned evidence before `cleanup.py` deletes the droplet; deadline return failure uses a preservation snapshot. `finalize.py` annotates returned results locally and writes `completion.json`, `RESULTS.md`, CSVs and shared-tracklet families. No independent certification is produced. Do not restart a named job after launch intent without reconciliation.

Compact progress: `progress.json`, `latest-status.json`, `launch.log`, `watch.log`. Returned results live under `distant-results/`. Scientific completion and resource disposition require `completion.json` and `deleted.json`, respectively. Until then, report running/partial rather than a successful complete search. GPU/CPU utilization is sampled into `utilization.jsonl`; phase-specific CPU work and brief kernels must not be presented as sustained full GPU utilization.
