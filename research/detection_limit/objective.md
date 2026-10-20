# Full-objective `prove_floor` research contract

The catalogue asks for the information-theoretic moving-source detection limit
for supplied exposures, PSFs, noise models and a source model, together with the
detector that attains it. The user explicitly prefers failed attempts at this
objective to presenting a restricted implementation as its solution.

## Acceptance criteria, fixed before experiments

1. State the statistical experiment and decision objective. Prior-average power
   and minimum power over an orbit family are different objectives; report both
   separately. Fix a false-alarm probability and a requested optimality gap.
2. Account for the continuous six-dimensional orbital domain, including the
   intervals between numerical templates, bound and unbound trajectories,
   observer displacement, light time, pixel-integrated PSFs, and any supplied
   nuisance uncertainty. An uncertified finite template bank is insufficient.
3. Give an upper bound applying to **every** admissible detector, and the power
   of an executable detector below that bound. Close their difference to the
   requested tolerance (initial target: 0.01 absolute power). Report arithmetic,
   discretization, calibration and statistical uncertainty separately. A
   high-confidence numerical statement is not an unconditional proof.
4. Demonstrate useful computation for physically resolved orbit families, not
   only a family essentially equivalent to one known track. Count construction,
   evaluation, memory and calibration. Include difficult close-observer and
   broad-family cases; report a budget failure as a failure.
5. Recover independently solvable cases, test against independent orbital
   integration and direct pixel likelihoods, and actively look for incorrect
   guarantees. Do not infer real-survey noise validity from simulated data.
6. Establish a substantive advance over prior work, including independent
   specialist scrutiny before claiming a breakthrough. Standard duality,
   likelihood ratios, total-variation bounds and confidence intervals are not
   new discoveries.

The first supported experiment will use specified Gaussian pixel noise, with
explicitly bounded mean-model uncertainty. This is an attempt at the supplied-
model objective, not a claim to solve arbitrary unknown noise. Failure to extend
to other supplied models, meaningful nuisance classes, or useful orbital scale
must remain visible in the final assessment.

## Three routes

**A — continuous mixture certificates.** Construct a finite approximation with
a justified total-variation error for the entire continuous source distribution.
Use the detection problem's primal/dual bounds to certify both a detector and
its distance from optimality. Refine the approximation instead of assuming
sampled orbits cover the family.

**B — least-favorable distributions.** Search for mixture weights that make the
family hardest to detect. Pair their dual upper bound with simultaneous lower
bounds on the detector's power throughout every orbit cell. Optimization alone
is not a certificate, and prior-average success is not minimax success.

**C — geometry and distinguishability.** Derive orbit-template covering errors
and compare a constructive geometric detector with information bounds. Test
whether these avoid the integration and six-dimensional covering costs in A/B.
Include a known symmetric continuous experiment as an independent exact control.

## Prior-art leads

- Neyman and Pearson (1933): optimal tests for specified distributions.
- Cvitanic and Karatzas, *Generalized Neyman–Pearson lemma via convex duality*:
  https://www.math.columbia.edu/~ik/neypea.pdf (the displayed dual bound already
  appears as equation 3.4; minimax duality is established prior work).
- Goldenshluger, Juditsky and Nemirovski (2015), *Hypothesis testing by convex
  optimization*: https://arxiv.org/abs/1311.6765.
- Fauß, Zoubir and Poor (2021), *Minimax Robust Detection: Classic Results and
  Recent Advances*: https://arxiv.org/abs/2105.09836.
- Zackay and Ofek (2017), *How to COADD Images. I*:
  https://authors.library.caltech.edu/records/c57s0-9y491.

A prior-art search does not itself establish novelty. None is claimed here.
