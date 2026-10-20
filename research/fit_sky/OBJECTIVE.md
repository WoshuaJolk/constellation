# FitSky full-objective research contract

## Authoritative catalogue rubric

- **In:** every photon ever recorded, with its time and instrument.
- **Out:** the posterior over one catalogue of everything: every star with
  parallax, motion and variability, every galaxy, every solar system body with
  an orbit.
- **Ships:** the generative model of every instrument and the inference engine
  that fits it jointly.
- **Why:** nothing is detected; everything is inferred, and every other tool
  here becomes a query against the posterior.

Fixed before implementation and campaign results. The objective is the catalogue's
joint posterior over the time-dependent sky from heterogeneous exposures. A full
solution must infer unknown source counts, identities, continuous parameters,
moving bodies, stellar motion/parallax/variability, extended emission, and shared
instrument uncertainty at useful scale. A narrow working example is evidence,
not completion. No mathematical novelty is assumed.

## What the output means

The input specifies observations, instrument likelihoods, a proper generative
prior, and the scientific queries. The output describes uncertainty over
catalogues, not a single supposedly unique true catalogue. Nonidentifiability
must survive inference. Source labels are exchangeable; permutations must not
be counted as distinct physical solutions. Each observation enters once.

An approximation must state which posterior quantities it controls, with error
bounds that include omitted source counts, continuous parameter coverage,
numerical evaluation, and shared nuisance parameters. Bounds conditional on a
supplied instrument model do not validate that model against nature.

## Completion gates

1. **Joint model:** competing source counts/types and continuous source parameters;
   moving and stationary/variable/extended sources; multiple exposures and
   instrument responses; explicit correlated/shared nuisance uncertainty.
2. **Complete accounting:** no supplied shortlist treated as the entire sky,
   no unknown modes discarded because an optimizer or sampler missed them,
   no source-count truncation without a bound, and no replacement of marginal
   probability with maximum likelihood. A proper prior defines the domain.
3. **Quantified accuracy:** executable posterior queries with independent,
   checkable error accounting. Initial target: posterior-probability intervals
   of width <= 0.01 and a relative evidence bracket <= 0.01. Confidence-based
   statements must be distinguished from deterministic arithmetic enclosures.
4. **Useful computation:** broad, physically resolved continuous domains,
   source confusion, time-dependent identities, real heterogeneous images,
   and scaling in sources/exposures/parameter extent. Count construction,
   inference, validation, memory, and failures. Tiny-domain success alone fails.
5. **Independent validation:** exact enumeration and analytic integrals where
   possible, independently generated pixels, permutation and prior tests,
   adversarial blends/aliases/calibration shifts, and held-out real observations.
6. **Full objective:** the components must operate together. Finite candidate
   inference, local continuous integration, a new API, a small sky patch, or a
   complete but unaffordable brute-force algorithm is not full completion.

The contract does not require reconstructing inaccessible observations, resolving
unidentifiable objects, an exact image inverse, or a uniform efficient algorithm
for arbitrary adversarial statistical models. It does require demonstrated useful
scope rather than silently choosing easier inputs after failures.

## Routes to pursue

**A. Interaction structure.** Derive joint Gaussian source statistics, retain
cross-source Gram terms, exactly sum finite catalogue alternatives using sparse
interaction structure, and bound the effect of any omitted interactions.
Marginalize shared nuisance parameters outside any conditional factorization.
Finite dictionaries are controls and computational experiments, not completion.

**B. Continuous catalogue integration.** Integrate over unknown source count,
exchangeable source classes, continuous parameters and instrument nuisance with
global evidence/query bounds. Compare interval subdivision with higher-order
validated quadrature. Account for source-count tails using a likelihood envelope.
Keep all unresolved probability mass on budget exhaustion.

**C. Physical experiment.** Use continuous forward models for pixel-integrated
stars, motion/parallax, variable flux, extended sources and Kepler trajectories.
Test mixed scenes and source ambiguity, then actual archived images. Keep Kepler
code read-only; its current approximations are not automatically posterior
certificates.

## Predeclared failure tests

- Confused two-source versus one-source scenes, exchangeable source labels.
- A shared astrometric offset that creates apparent motion if fitted incorrectly.
- An orbit or source mode outside a plausible initial candidate list.
- Broad position, velocity and flux domains at fixed original widths.
- Increasing unknown source count and shared nuisance dimension.
- A nonzero unenumerated Poisson source-count tail.
- Deliberate inference budget exhaustion and damaged saved certificates.
- Real-image model mismatch and disagreement between held-out predictions and
  the supplied likelihood.

A failed gate is reported as a failed gate. Completing bounded mathematical
subproblems is worthwhile, but the result must not be relabeled a full solution.
