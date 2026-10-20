# Joint catalogue inference: derivation and exact limits

These constructions use established Gaussian algebra, graphical-model variable
elimination, Bayesian marginalization, interval arithmetic and Gaussian
quadrature. No novelty theorem is claimed. Their combination is an implemented
research attempt at the fixed objective in `OBJECTIVE.md`.

## The probability space

A catalogue has unknown count K, class counts, and continuous parameters for
each source. Each source contributes to all exposures in which its forward
model puts light on usable pixels. The source count is either an explicitly
finite categorical prior or Poisson with a retained computational tail.
Within each class, parameters have proper uniform priors on supplied boxes.
Fixed parameters are point masses. These priors are substantive assumptions.

For class probabilities q_t, a class-count vector (k_1,...,k_T), sum k_t=K,
has prior mass

    p(K) K! product_t q_t^k_t / k_t!.

The implementation integrates all labeled coordinates within each such vector.
This produces the correct expectation of permutation-invariant catalogue
queries under the IID prior. It does not multiply by K! a second time or impose
an arbitrary ordering that changes the prior.

An unrepresented Poisson count tail is never renormalized away. With a
likelihood scaled to be <=1, its evidence is in [0, prior_tail]. If represented
evidence is at least Z_lower, its posterior probability is at most

    prior_tail / (prior_tail + Z_lower).

This bound can be extremely loose. A small prior tail need not be a small
posterior tail when the represented likelihood is small.

## Interactions and the finite graph route

For fixed templates p_a and fluxes F_a, let s_a in {0,1} indicate existence.
Whiten pixels and templates using the *full supplied covariance*. Then

    log L(s) - log L(0)
      = sum_a s_a F_a p_a^T y
        - 1/2 sum_ab s_a s_b F_a F_b p_a^T p_b.

The off-diagonal terms make source hypotheses compete for shared light.
Discarding them turns a joint inference into independent source scores.

Bernoulli existence priors produce unary log odds and pair interactions. Log
sum-product variable elimination computes the partition function and conditioned
partition functions for posterior queries. Complexity is exponential in induced
interaction width, not automatically in the number of dictionary entries.
The implementation chooses small elimination buckets and fails before a factor
exceeds its table budget. It never interprets the failure as a posterior result.

For omitted edges with total absolute coupling epsilon, every configuration
has |log L - log L_approx| <= epsilon. Evidence is within exp(+-epsilon);
an event with approximate probability p has log odds within +-2 epsilon.
These inequalities concern interaction omission. The graph implementation is
float64 and does not certify its own rounding errors. Its dictionary is finite
and supplied; it cannot certify absent continuous source configurations.

Shared nuisance is conditioned on before factorization. Its posterior weights
are proportional to prior_weight * full_conditional_evidence. Averaging the
separately normalized source probabilities with prior nuisance weights is wrong.

## Continuous image model

The implemented sources include:

- Point sources with continuous position, linear proper motion, parallax
  factors, log-linear time variability and a log-linear band response.
- Extended sources with axis-aligned Gaussian intrinsic widths in detector
  coordinates. This is not a general galaxy morphology model.
- Two-body Kepler states with observer displacement, light time and gnomonic
  projection, using the existing read-only validated orbital implementation.

The pixel response integrates a separable Gaussian PSF over each pixel; cropped
wings are not renormalized. Exposure times, masks, flux scales, affine geometry,
and instrument identifiers are explicit. Shared per-instrument shifts, gain,
PSF scale and backgrounds can have continuous box priors. This implementation
does not yet include finite-exposure trailing, general WCS distortion,
non-Gaussian spatial PSFs, arbitrary light curves, general galaxies, or an
instrument archive. Those are full-objective failures, not implicit support.

Gaussian observations support known arbitrary covariance. Poisson observations
retain individual integer pixel counts and positive intensity. No difference-
image Gaussian approximation is silently passed off as a photon-count model.
Noise parameters are supplied, not learned robustly from the same pixels.

## Exact shared Gaussian nuisance elimination

If y = model + B eta + e, eta~N(m,S), e~N(0,D), marginalization gives

    y ~ N(model + B m, D + B S B^T).

The normalization determinant is retained; a profile fit is not substituted for
this integral. For diagonal D, factor S=A A^T, define T=D^(-1/2) B A and
H=I+T^T T. The quadratic form is

    r^T r - (T^T r)^T H^(-1) (T^T r),
    r = D^(-1/2)(y-model-B m),

and log det(D+B S B^T)=log det D+log det H.
Only the low-dimensional H requires a dense Cholesky factorization. All
coefficients, positive-definiteness checks and arithmetic are enclosed by Arb.
Subtracting two norm bounds can be conservative; positive definiteness permits
intersecting the result with the nonnegative half-line.

This removes a dimension from computation only because the nuisance model is
linear Gaussian with a specified proper prior. It does not solve arbitrary
calibration uncertainty. Full covariance plus Gaussian nuisance is also
implemented, using a dense certified Cholesky factorization.

## Exact integration of one continuous flux

Conditional on all remaining source and instrument parameters, the Gaussian
likelihood in one flux F is

    L(F)=exp[-(C-2BF+AF²)/2].

A, B, C retain all other sources and any marginalized Gaussian nuisance.
For F uniform on [l,h] and A>0, the average likelihood is

    exp[-(C-B²/A)/2] sqrt(pi/(2A))
      [erf((Ah-B)/sqrt(2A))-erf((Al-B)/sqrt(2A))] / (h-l).

The hybrid route evaluates this identity in Arb and differentiates it for the
remaining quadrature coordinates. A nonpositive information enclosure refuses
the analytic shortcut. Subtracting nearly equal error functions can lose
precision or widen bounds; there is no silent Laplace approximation.

## Validated tensor quadrature

For each continuous cell, the direct interval forward likelihood gives a valid
range [l,u], so its uniform average belongs to that range. This alone converges
slowly in many dimensions.

The higher-order route applies the two-node Gauss-Legendre rule in each active
coordinate. On an interval of half-width h, its error for the *normalized*
integral is bounded by

    h^4 sup|f''''| / 270.

A fourth-order directional Taylor jet stores f''''/4!. Its derivative seed is
h, so the bound becomes 4/45 times the supremum of the fourth jet coefficient.
The zeroth jet coefficient spans the entire multidimensional box. Each
directional derivative bound therefore holds for every value of all other
coordinates, not just at sampled nodes.

Normalized integration and the positive-weight Gauss rule both have sup-norm
operator norm one. Telescoping their tensor product bounds the total error by
the sum of the per-coordinate errors. Gauss nodes, weights, model evaluations
and remainders all use Arb. Intersecting this enclosure with the direct range
can only improve it. For large dimension the 2^d node count is unaffordable,
and the implementation explicitly falls back to range bounds.

Kepler parameters currently have interval forward enclosures but no
fourth-derivative jets. A fixed Kepler geometry can still use flux/instrument
derivatives. Broad uncertain orbital cells use the range route.

## Global accounting and scientific queries

Each root has a known prior mass. Every binary split halves a normalized box,
so a leaf at depth d has exactly 2^-d of its root's prior measure. Enclosed
leaf integrals, summed with their masses, give evidence lower and upper bounds.
The remaining Poisson count tail contributes [0,tail]. Budget exhaustion leaves
all active leaves in the output.

For an event A, derive bounds N_l,N_u for its unnormalized mass and C_l,C_u
for its complement. Then

    N_l/(N_l+C_u) <= P(A|y) <= N_u/(N_u+C_l).

Count/class events are constant on a root. Continuous parameter-range queries
can straddle cells; such cells contribute zero to the lower bounds and their
upper mass to both possible events. They are refined without discarding them.
Queries are symmetric over sources, not about an arbitrary exchangeable label.

A complete certificate records every leaf and subdivision path. Verification
checks a full binary partition for every root, rejects duplicate or omitted
leaves, replays enclosures, and recomputes the reported evidence/probabilities.
This is a computational certificate conditional on the supplied model and
trusted arithmetic library, not a formal proof assistant derivation or a
certificate that the instrument model describes the observations.

## Catalogue draws

Posterior samples use a mixture of leaf **pointwise** likelihood envelopes.
Quadrature bounds on average likelihood must not be used as rejection-sampling
envelopes. The sampler recomputes range bounds, proposes uniformly inside a
selected cell, and performs an Arb acceptance comparison. Proposal probabilities
and the RNG are conventional floating-point computations. Draws condition on
represented source counts; the excluded posterior mass bound accompanies them.
Their finite Monte Carlo error is separate from the deterministic query bounds.

## What this mathematics has not established

Neither sparse finite graphs nor continuous integration proves an efficient
algorithm for the entire catalogue rubric. Ambiguous sources can form large
interactions; common calibration connects otherwise separate regions; high
dimension defeats simple box quadrature; count tails can dominate the evidence
bound. Global low rank is not assumed. The repository's earlier resolved-track
counterexample would already invalidate that shortcut.

The research question is whether a representation can exploit physical
locality while retaining broad continuous alternatives, source-count uncertainty
and shared calibration with a *useful* global posterior error bound. The
experiments measure how far these particular constructions reach.

## A stronger likelihood envelope

For a cell, retain the continuous flux box but take source-shape and background
reference templates at the non-flux midpoint. A constrained quadratic fit
proposes a vector a in pixel coordinates. For any such vector,

    min_F ||y-b-PF||_(C^-1)^2
      >= 2 a^T(y-b) - a^T C a - 2 max_F a^T P F.

The box support function on the right is explicit. The optimizer need not find
the optimum: its proposed a is checked with Arb, including all covariance and
template arithmetic. This gives a lower bound d on the reference residual norm.

Interval source responses bound each shape perturbation by e_i in the noise
metric and background perturbation by e_b. Nonnegative bounded fluxes give

    ||actual residual||_(C^-1)
       >= max(0, d - e_b - sum_i F_i,max e_i).

Exponentiating minus half its square produces a valid likelihood ceiling for
every source configuration in the cell. Intersect it with the original range
ceiling. This preserves the correlation of all pixels with a source's flux,
which independent pixel boxes lose. It still becomes loose on broad shape
domains; there is no claim of a universal efficient bound.

## Fixed-budget statistical integration

For cell prior masses m_i and pointwise likelihood ceilings u_i, propose cells
with probabilities proportional to m_i u_i, then parameters uniformly inside
the cell. All represented cells retain positive probability. The importance
weights have a known ceiling B; after scaling they lie in [0,1].

Estimate the evidence and each event/complement mass using independent fixed-
budget draws. The implementation applies the empirical Bernstein bound

    |mean - expectation|
      <= sqrt(2 sample_variance log(2/delta)/n)
         + 7 log(2/delta)/(3(n-1))

as two separate one-sided statements, allocating failure probability across
all expectations before validation. Convert mass intervals to posterior
intervals with the same event/complement formula above. Add any unrepresented
count tail to the upper masses. This is conditional Monte Carlo confidence,
not deterministic arithmetic certification. Source counts are sampled without
truncation when drawing directly from the complete Poisson prior.

The vectorized renderer, proposal probabilities, RNG and concentration
calculation use float64. Independent comparisons with Arb test the numerical
implementation, but do not turn those assumptions into formal enclosures.
A fixed-budget unresolved result is retained; favorable intermediate estimates
do not trigger early stopping.

## Held-out observations

Under the same source prior and a consistent proper shared-background model,

    p(y_held | y_train) = Z(y_train, y_held) / Z(y_train).

The held-out experiment bounds this ratio by two independently enclosed
marginal likelihoods. An extra per-exposure Gaussian background deviation
integrates out to recover the original training model. This propagates shared
background uncertainty instead of resetting it to a best fit. Wide predictive
bounds remain a failure to establish a useful prediction.

## Primary prior work

- Portillo et al., [probabilistic cataloging](https://arxiv.org/abs/1703.01303):
  catalogue ensembles, source-count uncertainty and blending.
- Feder et al., [multiband probabilistic cataloging](https://arxiv.org/abs/1907.04929):
  joint source inference across bands.
- Regier et al., [Celeste](https://arxiv.org/abs/1801.10277):
  approximate Bayesian catalogue inference at large scale.
- Feder et al., [PCAT-DE](https://arxiv.org/abs/2307.10385):
  joint point-source and diffuse emission modeling.
- Maurer and Pontil, [empirical Bernstein bounds](https://arxiv.org/abs/0907.3740):
  finite-sample concentration used by the statistical route.

This is not an exhaustive novelty review. The derivations above do not establish
that these methods or their combination are new.
