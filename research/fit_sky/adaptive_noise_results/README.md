# Adaptive three-image validation

`joint-cover.json` retains the original three-image source/background model and
complete prior domain. The 120-second construction produced 664 leaves, with
1,324 support proofs constructed including discarded parent cells. Its
deterministic integral brackets remain unresolved; it supplies a much tighter
pointwise proposal for statistical integration.

The first one-million-draw sampling attempt was interrupted for runtime before
any statistical result was inspected. The complete cover and successful replay
were retained. `interrupted-pilot.json` records that attempt. The implementation
was changed to batch by catalogue structure, preserving per-cell bounds and
importance factors.

`validation-contract.json` fixes the fresh four-million-draw validation.
`validation/` contains the executed driver and sources, statistical result,
independent quadrature at orders 20 and 32, and predictive interval. The result
meets the original evidence and posterior accuracy targets under the stated
model/numerical assumptions.

To rerun against the retained cover in a new directory, from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python \
  research/fit_sky/adaptive_noise_results/run_validation.py \
  --output /tmp/new-adaptive-fit-sky-validation
```

To construct a new partition, use `magictools-fit-sky` with
`--method noise-floor`. Soft construction budgets may produce different leaf
counts on another machine. Version-two support bounds additionally retain
signed responses and positive flux lower bounds; the retained version-one
proofs remain independently replayable.

The full FitSky objective remains unmet.
