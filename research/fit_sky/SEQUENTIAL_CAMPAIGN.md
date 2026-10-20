# Sequential mixed-sky campaign

This is a follow-up to the original full-objective contract, not a replacement.
The 10–30-exposure milestone is only met if the original probability-width
and evidence-relative-gap targets are met on the mixed scene. Approximate
particle inference alone cannot meet those targets.

## Fixed problem

Keep the original `synthetic("broad")` source classes, parameter domains,
Poisson mean 2, shared `b_dx` and `b_log_gain`, truth, and first three images
unchanged. The enumeration `max_count=3` is not a prior cutoff.

Append 27 independently generated 3-by-3 exposures. Their times are
`linspace(-2, 2, 27)`, instruments alternate a/b, PSF widths cycle through
0.55, 0.65, 0.75, and noise sigma remains 0.6. Parallax factors are sin(time)
and cos(time). Use the independent physical renderer and noise seed 20261103.
Evaluate the prefixes of 3, 10, and 30 exposures. No source detection list or
truth initialization is supplied to inference.

## Implementation and validation

Implement sequential Monte Carlo with tempered exposure assimilation,
trans-dimensional birth/death and class moves, and continuous parameter moves.
All instrument parameters remain shared across exposures. Each move may
revise earlier conclusions. Poisson counts are unbounded; resource exhaustion
must fail explicitly rather than silently cap counts.

Accumulate Gaussian residual statistics in pixel blocks. Shared Gaussian
linear nuisance must be marginalized jointly; multiplying independently
marginalized exposure likelihoods would be incorrect. Preserve log likelihoods
to avoid underflow with many images.

Before the fixed campaign:

- Compare the block likelihood with independent rendering and dense covariance.
- Check forward/reverse birth/death ratios, prior invariance, source permutation,
  and count support beyond the old enumeration cutoff.
- Compare small posterior problems and evidence against analytic/quadrature
  controls, including shared calibration.
- Check checkpoint/resume, changed-input rejection, and exposure accounting.

Pilot budgets and implementation tuning are allowed and are recorded separately.
The fixed validation is four independent populations of 2,048 particles with
20 rejuvenation sweeps per tempering step, conditional ESS target 0.8, and
seeds 20261110–20261113. Record all runs, stages, acceptance rates, evidence
estimates, query estimates and population disagreement. Do not choose a
successful seed, drop a failed run, or interpret ESS as an accuracy certificate.
The full 30-image sequence is run once per seed, retaining the 3/10/30 prefixes.

Report approximate particle results separately from existing global confidence
bounds. Unless valid bounds meet the fixed accuracy targets, the milestone
remains unresolved, even if independent populations agree.

## Longer-term architecture

The implementation should admit new observations and checkpoints without fixing
catalogue identities. A 100/1,000-exposure likelihood throughput measurement can
test batching but is not a posterior-inference or scalability success. Arbitrary
instruments, orbital populations, and astronomical scale remain separate gates.

## Method provenance

Sequential Monte Carlo and reversible-jump proposals are established methods,
not mathematical novelty:

- Del Moral, Doucet and Jasra (2006),
  [Sequential Monte Carlo samplers](https://doi.org/10.1111/j.1467-9868.2006.00553.x).
- Green (1995),
  [Reversible jump Markov chain Monte Carlo computation and Bayesian model determination](https://doi.org/10.1093/biomet/82.4.711).
