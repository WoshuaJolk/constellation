# Global mathematical attempt

The catalogue objective and `objective.md` remain the acceptance criteria.
The observer-chart campaign did not solve the global transform. This attempt
prioritizes a derivation and explicit complexity/error accounting.

The operator is

```
F(q) = sum_m d_m exp(i phi(q, s_m)),
s = (time, spatial_frequency_x, spatial_frequency_y), q in a six-dimensional box.
```

For Kepler, `phi = k dot apparent_position(q,t)`. Reception observer motion and
retarded emission time remain in the position calculation. A smooth common
tangent projection is used for the mathematical source-time interpolation;
this is an explicit input-model requirement, not an interpolation of arbitrary
per-exposure WCS headers. Real detector likelihood preparation remains separate.

Try a source-interpolating, phase-demodulated hierarchy over the entire six-
dimensional output box. It may generate a continuous representation before
any output query is supplied. Derive:

1. A uniform kernel error bound, including every interpolation level.
2. Counts for all block pairs, phase construction, storage, application and
   evaluation. A local low-rank claim alone is insufficient.
3. The distinction between a bandwidth-complete domain and sparse queries.
4. Explicit analytic hypotheses for retarded Kepler; sampled smoothness cannot
   establish a uniform complex-domain bound.
5. The adjoint and the additional conditioning requirements of an inverse.

Primary prior art includes the Candès–Demanet–Ying butterfly construction and
Hu et al.'s generalized Radon transform. An implementation or extension of
those ideas is not automatically a novel mathematical result.

Use the previous near, distant and stress orbit boxes without narrowing them.
Low-frequency controls may verify an algorithm, but cannot pass the physical
bandwidth objective. Keep failures at physical bandwidth. Do not automatically
substitute a dense transform or shrink a domain after a resource rejection.

Also attack two tempting shortcuts: a constant-degree perturbation of a linear
phase, and an unconditioned inverse. Derive counterexamples where possible.
The campaign additionally examines exact dynamical regularization and a
fixed-size nonlinear lift to a Fourier phase; their obstructions are preserved
in `regularization_attempt.md` and `finite_features_obstruction.md`.

Success requires the original scientific input/output and a useful global
algorithm, not merely a conditional theorem, a control, or an observed speedup.
