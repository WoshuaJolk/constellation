# Second full-objective attempt: construction, continuous coverage, inverse

The target remains [the full transform](objective.md). A discovery is not an
acceptance requirement. Neither is mathematical novelty required to provide a
useful implementation; novelty is required only for a novelty claim.

This attempt changes the algorithm, rather than hiding the previous dense
construction in preprocessing:

1. Build an interpolative butterfly using selected kernel entries. No full
   orbit-by-image-frequency matrix may be used during construction. Record
   every kernel entry, every propagated orbit/time pair, transfer storage,
   construction time and application time. Compare against direct summation
   and a competent NUFFT image evaluator, where applicable.
2. Use exact orientation-dependent light time, a moving observer and all
   conic energy classes. Test resolved image frequencies and near-observer
   cases, not just smooth low-frequency fields. A sampled interpolation
   residual is a diagnostic, not a continuous-domain certificate.
3. Attempt continuous six-dimensional score enclosures using validated
   dynamics and explicit Fourier-field derivative bounds. Preserve the entire
   root domain under subdivision, including unresolved cells. Budget exhaustion
   fails the requested tolerance; it must never mean "nothing there".
4. Implement the algebraic adjoint and a clearly identified regularized
   least-squares inverse. Compare with an independent dense solution on small
   cases; report nullspaces and conditioning. Do not claim recovery of
   information already lost by time summation.
5. Report complete one-shot costs and failed cases. Finite-bank speedups,
   completed tiny domains and sampled accuracy alone do not complete the full
   objective. Ship research results even if the full objective fails.

The construction uses established interpolative butterfly and numerical
linear-algebra ideas. It is an experiment, not a new theorem. The original
research's requirements concerning image models, arbitrary continuous orbital
coverage, meaningful scaling and inverse semantics remain in force.
