# GPU priorities after the expanded distant search

The next goal is more useful distance/time coverage and corroborated families per
dollar. More fitted triples alone is insufficient. This assessment uses the
completed heavy-run receipts; it does not restart that closed run or change any
candidate acceptance rule.

## What actually limited the last run

The 6,204 recorded batches cover 6,352,896 chains and 280,935,075 starting states.
GPU solve wall times sum to 1,804.84 seconds; preparation to 2,103.38 seconds;
CPU verification/fallback to 1,428.57 seconds; selection to 564.66 seconds.
These phases overlap. They must not be added to obtain elapsed time or used as
GPU utilization measurements. The production phase took about 2,462 seconds.
There were 5,666,914 CPU fallback starts, about 2.02% of all starts. Preserve the
verification requirement while investigating why these starts needed replay.
Transport mapped 167.30 GB cumulatively; that is neither peak RAM nor a claim
that all these bytes crossed PCIe. A faster fitting kernel alone leaves major
host work and discovery coverage limits untouched.

The proposal stage issued 13.69 billion night-specific queries and retained a
capped four-million-edge graph. All twenty partitions stopped on time. The code
visits epochs chronologically and distances in ascending order, so capped runs
can systematically underserve late epochs and farther hypotheses. Existing
receipts do not identify every distance/epoch reached; exact coverage cannot be
reconstructed from the aggregate counters. A 2,000-AU grid endpoint is not proof
that the catalogue was searched completely at that distance.

Finally, triple fitting exhausted its time allowance before ordinary chain
growth ran. A separately budgeted planetary follow-up examined 147 selected
parents. Any next search needs protected resources for corroboration, including
alternative modes, rather than allowing triple generation to consume everything.

## Implemented groundwork, not deployed production changes

1. `distant_coverage_plan.py` creates unique, deterministic epoch/range/radial
   tasks. Hashed epoch order and interleaved hypotheses spread early work over
   the grid; the full task set equals the original Cartesian product. Equal task
   counts do not mean equal CPU cost or equal scientific completeness. Completion
   receipts, time slicing and per-lane edge quotas still need integration.
2. `gpu_distant_projection.py` uploads tracklet geometry once and evaluates a
   bank of range/radial hypotheses in FP64, one CUDA thread per tracklet and
   hypothesis. It reuses the existing universal-variable device propagator.
   This is the same approximate common-epoch proposal model, not planetary final
   fitting or an uncertainty distribution. Memory bounds require explicit tiling.
3. Seven local tests passed, including the existing heavy-search checks and CUDA
   simulator parity across all sixteen proposal distances, three radial fractions,
   two reference epochs and the 365-day window boundaries. Finite masks match;
   angular differences must stay below 0.00001 arcsecond. These synthetic numerical
   checks are not real-hardware throughput or population recovery measurements.
4. `benchmark_distant_projection.py` is ready for a frozen real-catalogue input
   bank. It records input hashes before evaluation, alternating CPU/GPU order,
   three complete repeats, transfer costs and parity failures. It explicitly
   refuses simulator timing. No hardware speedup is claimed yet.

The old production path remains unchanged. No new cloud instance was created;
the provider check found zero droplets. Incremental paid compute in this pass: $0.

## Recommended implementation order

First integrate explicit coverage accounting and bounded task chunks, including
per-distance/time progress and clipping counters. Use per-lane proposal quotas
so a dense region or one hypothesis cannot erase every other lane at a global
edge cap. This is a versioned search-policy change, not a speedup-parity claim.

Then benchmark resident GPU projection plus a spatially indexed, batched neighbor
stage on frozen catalogue windows. Avoid a dense all-tracklet-by-all-tracklet
matrix. Projection alone may be a small fraction of the proposal bottleneck;
measure query/build/transfer time together before spending on a full search.
Keep CPU checks and distinguish finite-cap tie/order changes from exact parity.

Next reduce host preparation/selection and investigate fallback mechanisms without
dropping difficult starts or relaxing precision. Record device-active time and
host-wait time directly; current wall profiles cannot establish SM utilization.
Measure distinct seeds and recovered families at equal cost, not copied workloads.

Reserve a separate, immutable budget for long-gap retrieval and planetary fits.
Propagate multiple retained modes and conditional uncertainty over available
earlier/later observations. Track empirical recovery and confusing alternatives;
no local uncertainty model or low residual alone certifies a chain.

For the dwarf-planet priority, replace apparent-brightness-only scheduling with
distance-conditioned brightness lanes while retaining ordinary, unusual and
missing-photometry lanes. Our previous priority used magnitude <=22.5 in mixed,
unconverted bands. A hypothetical 1,000-km body with albedo 0.1 has H about 3.12:
at opposition, m approximately H+10 log10(r) gives 19.14 at 40 AU, 23.12 at 100 AU,
24.89 at 150 AU and 27.89 at 300 AU. These are illustrative V-band, r≈Delta
calculations, not survey detectability or size measurements. A magnitude-25 object
at 150 AU could warrant more attention than a magnitude-23 object at 40 AU.
Uncertain distances, band offsets, phase and albedo must remain explicit.

Before a new full search, compare CPU and GPU at fixed wall time and dollars on
the same proposal domain. Include real known-object recovery where available,
longer-gap reserved observations, and synthetic cadence/brightness controls
reported separately. Do not promise a dwarf planet or 100x end-to-end speedup.
The specific opportunity is to search more of the right parameter space and
leave enough compute to test the resulting candidates.

Receipts: `/tmp/itf-gpu-next-20261029/` (profile source hash, phase summaries,
tests and provider check). The completed run and all earlier evidence are intact.
