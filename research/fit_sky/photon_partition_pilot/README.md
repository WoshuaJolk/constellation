# Photon partitions and an unbounded source population

This exploratory control investigates a different route to complete population
accounting. It does **not** solve the original broad Gaussian-image problem.
The executable has fixed spatial responses and either fixed or uniform flux.
It is limited to 12 observed photons because its exact partition calculation
is exponential. A 30-exposure, five-photon scalar control is not the mixed-sky
milestone.

Established Poisson multi-Bernoulli mixture results motivate this route:
[Granstrom, Fatemi and Svensson](https://arxiv.org/abs/1605.06311) give a conjugate
extended-object model, and
[Fatemi et al.](https://arxiv.org/abs/1811.03154) derive a batch map posterior
under Poisson object and measurement models. These works also confront the
combinatorial data-association problem. Their existence does not establish
tractable, certified astronomical inference.

## Specialization to photon observations

Condition on shared instrument parameters u. Let source states theta form a
Poisson process with finite intensity measure nu(dtheta), total mass lambda.
Theta can in principle include type, position, trajectory, variability and flux.
For each recorded photon j, let r_j(theta,u) be its source intensity and
R(theta,u) the expected source photon count across the complete observed
domain, including pixels where no photons arrived. Let b_j be known background
photon intensity and B its expected total over that domain.

For every subset S of observed photons, define

    kappa(S) = integral exp(-R(theta,u))
                        product_{j in S} r_j(theta,u) nu(dtheta).

In particular, kappa(empty) is the mean number of sources producing no recorded
photons. Expanding the Poisson likelihood and applying the Poisson-process
factorial moment identity gives

    Z(u) = exp(-B-lambda+kappa(empty)) *
           sum_partitions product_blocks [
               kappa(block) + background contribution for singleton blocks
           ].

For binned counts, divide by the product of pixel-count factorials. The
partition labels represent individual observed photons, including repeated
pixel positions; they are not a supplied source detection list.

All possible source counts have already been summed. Conditional on a photon
partition, each source-associated block has a continuous source posterior
proportional to its integrand. The remaining source population is an independent
Poisson process with intensity `exp(-R) nu`. Sources outside observational
sensitivity therefore retain posterior uncertainty instead of disappearing.

Introduce one factor t for each source-associated block, and none for background
singletons. If P(t) is the resulting partition polynomial, the posterior
generating function for the **total** number of sources is

    exp(kappa(empty)*(t-1)) * P(t)/P(1).

The exponential retains arbitrarily large source counts exactly. For a source
property A, form P_notA using only source moments over the complement of A.
Then

    Pr(no source has A | data,u)
      = exp(-kappa_A(empty)) * P_notA(1)/P(1).

This provides symmetric catalogue queries without enumerating K-source
parameter spaces.

## What the pilot checks

For a fixed spatial response a_j and uniform source flux F, each moment reduces
to a product of responses times

    E[exp(-A F) F^k], where A = sum of responses over all observed pixels.

Integration by parts evaluates these moments in Arb arithmetic. An anchored
subset recurrence sums every partition once and retains the polynomial in the
number of emitting sources.

The controls compare:

1. Two source classes against independent direct source-count enumeration,
   with an explicit Poisson tail bound on that reference.
2. Thirty scalar exposures with continuous uniform flux against independent
   count enumeration and Irwin-Hall quadrature for sums of uniform fluxes.
3. A completely insensitive observation against the unchanged Poisson source
   prior: zero observed photons does not imply zero sources.

Run from the repository root:

```sh
PYTHONPATH=. python research/fit_sky/photon_partition_pilot/driver.py NEW_OUTPUT
```

## Remaining barriers

The moments must be integrated over the actual continuous source prior with
quantified error. The partition function has exponentially many interactions;
the pilot is O(3^N) in observed photons and cannot handle ordinary bright images.
Useful inference needs a controlled compression or decomposition of these
associations, rather than selecting a few favorable partitions.

Shared instrument uncertainty must remain outside the complete conditional
expression: integrate `p(u) Z(u)` and its query numerators together. Integrating
each source moment independently over the same nuisance would incorrectly
erase its correlations. This outer integral is not implemented in the pilot.

The formula assumes a Poisson photon likelihood. Gaussian read noise,
calibrated difference images, saturation and other instrument effects need
their actual likelihoods or additional latent variables. The existing broad
Gaussian case cannot be converted by rounding, clipping or treating measured
pixel values as photon events.

This route removes a source-count sum mathematically. It does not establish a
full solution, mathematical novelty, or astronomical scaling.
