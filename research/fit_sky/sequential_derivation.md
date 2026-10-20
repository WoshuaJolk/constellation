# Sequential catalogue inference

This implements established sequential Monte Carlo and reversible-jump ideas,
not a new completeness theorem. A finite particle population can lose modes.
Neither ESS, accepted proposals, nor agreement of independent runs certifies
the resulting posterior. Reports always keep `accuracy_target_met=false` and
trivial `[0,1]` certified event bounds.

## Target and exposure bridges

For a catalogue X and shared nonlinear instrument parameters u, write

    pi_e(X,u) ∝ p(X,u) L_e(X,u),

where L_e is the normalized joint likelihood of the first e exposures after
integrating shared Gaussian linear nuisance. Assimilation from a to b uses

    pi_beta ∝ p(X,u) L_a(X,u)^(1-beta) L_b(X,u)^beta.

The incremental particle log weight is
`(beta_new-beta_old) * (log L_b-log L_a)`. When calibration is marginalized,
`L_b/L_a` is generally NOT the likelihood of just the new images with their
calibration marginalized independently.

Conditional ESS selects a temperature increment. Systematic resampling is
followed by fixed-mixture Metropolis moves targeting this full tempered
density. The log evidence estimate accumulates log average incremental
weights. Adaptive temperatures and correlated particles preclude interpreting
the output as an IID sample mean with the existing empirical-Bernstein bounds.
The implementation makes no finite-run unbiasedness or confidence claim.

## Variable source count and exchangeability

Use ordered latent slots conditional on K, with target prior

    p(K) product_i pi(class_i) product_j Uniform(unit_ij).

Symmetric catalogue queries integrate labels out automatically. A second
factor of K! would be an error. All source types share a unit-coordinate array;
coordinates absent from a particular type are auxiliary independent uniforms.
Mapping a unit coordinate into a class-specific parameter interval accounts
for its uniform prior exactly.

A birth inserts an independent class/prior-coordinate draw at a uniformly
selected one of K+1 positions. A death selects a uniformly random existing
position. Birth/death attempt probabilities remain 1/2 even at K=0, where a
death is a self-transition. Insertion and deletion probabilities cancel in the
Metropolis ratio, as do the source-prior factors. The remaining prior ratio is

    p(K+1)/p(K),

which is lambda/(K+1) for the Poisson model. There is no additional factorial.
The reverse death ratio is its reciprocal at the matching states.

A separate 5% prior-refresh component redraws the entire source catalogue,
leaving instrument parameters unchanged. Its prior and proposal factors cancel.
This connects finite count priors containing zero-probability intermediate
counts and provides full-support proposals independently of the current
population. Full-support proposals do not guarantee that a finite run explores
all important modes.

Class moves propose from the class prior and retain common unit coordinates.
This is a dimension-preserving transition in the auxiliary representation.
Reflected Gaussian unit-coordinate moves are symmetric. The mixture also
contains complete source refreshes. Scales and mixture weights are fixed;
they are not estimated from the current particles.

Poisson samples and proposals have no hard maximum K. Storage grows as needed.
`max_count` from the original interval enumeration is deliberately not used as
a count prior cutoff here. Memory or stage exhaustion is a failure, not an
excuse to silently exclude larger counts.

## Joint Gaussian nuisance in bounded rendering blocks

Let the base pixel covariance be diagonal D, and let b have prior N(mu,S).
For the model y=m(X,u)+B b+epsilon, define

    r = D^(-1/2) (y-m-B mu),
    T = D^(-1/2) B chol(S),
    G = I + T' T,
    h = r' T.

The marginalized squared residual is

    r'r - h G^(-1) h',

and `log det(C) = log det(D) + log det(G)`.
Accumulate `r'r`, h, and T'T over pixel blocks and evaluate these expressions
at each requested exposure prefix. Thus all exposures constrain the same
latent calibration. No independent per-exposure background prior is invented.

The renderer uses at most `particles * pixel_block` pixels at a time, plus
catalogue state and low-rank statistics. It does retain the input images and
geometry arrays. Arbitrary dense pixel covariance is explicitly rejected by
this route. Shared Gaussian linear nuisance and diagonal Gaussian or Poisson
noise are supported.

All likelihoods remain in log space; many exposures do not turn a valid
likelihood into an accidental floating-point zero. Nonfinite or stalled
tempering fails explicitly.

## Checkpoints and updates

A completed assimilation checkpoint stores every particle, normalized weights,
RNG state, settings, input manifest, evidence history and exposure count.
It uses an NPZ archive loaded with `allow_pickle=False`. Writes replace a
temporary file atomically. An interrupted tempering bridge cannot be saved or
reported as a completed posterior.

Resume permits the same observations or an appended sequence. Existing pixels,
times, masks, source priors, instrument priors and shared-linear-design rows
must match. Duplicated exposure IDs and attempting to assimilate an already
processed prefix are errors. A manifest with additional instrument nuisance
dimensions currently requires a fresh fit; existing priors may not silently
change during resume.

The method revisits all assimilated images when it proposes a catalogue change.
Consequently, assimilating E images one at a time can require quadratic total
work in E. Bounded rendering memory and incremental checkpoints do not by
themselves establish survey-scale inference.

## Gaussian conditional flux follow-up

With all other source parameters held fixed, a single source flux f enters
every Gaussian image linearly. At an exposure prefix, its log likelihood is
`constant + g f - h f²/2`, where `h=a' C^-1 a` and
`g=a' C^-1 (y-m_without_this_source)`. The same block statistics above retain
the other sources and the shared covariance in these expressions.

For the old/new bridge, combine h and g with weights 1-beta and beta. With a
uniform flux prior on [lo,hi], the conditional is a normal with mean g/h and
variance 1/h truncated to that interval. `--gibbs-flux` draws one randomly
selected source's flux this way every third rejuvenation sweep. It is an
additional conditional-invariant update, not a change to the source prior.
It can improve mixing but supplies no missing-mode guarantee.

Zero Poisson likelihoods are handled separately: an initial beta=0+ support
restriction includes the estimated surviving prior mass in the evidence.
If all particles have zero likelihood, the run fails explicitly.

## Controls and provenance

Tests compare normalized likelihoods against an independently rendered dense
multivariate Gaussian, including nontrivial masks and shared covariance.
Analytic/quadrature controls check source presence, flux events, evidence and
shared calibration. Other tests cover source/exposure permutations, count
support beyond the old enumeration cutoff, prior-invariant transitions,
checkpoint reproducibility and explicit budget failures.

- Del Moral, Doucet and Jasra (2006),
  [Sequential Monte Carlo samplers](https://doi.org/10.1111/j.1467-9868.2006.00553.x).
- Green (1995),
  [Reversible jump Markov chain Monte Carlo computation and Bayesian model determination](https://doi.org/10.1093/biomet/82.4.711).
