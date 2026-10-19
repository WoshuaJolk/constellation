# Detection-limit certificates: derivation and limits

These derivations apply established testing duality, change of measure,
total-variation continuity, and concentration inequalities. They are not claimed
as new theorems. A mathematical inequality, a probabilistic numerical
certificate, and a validated physical noise model are different things.

## 1. Define what is optimal

Let P0 be the specified null distribution of the complete data. For a source
parameter theta, let P_theta be the alternative. A randomized detector is a
measurable function phi(y) in [0,1], with false-alarm constraint E0 phi <= alpha.

For a specified population prior pi, the objective is

    beta_pi(alpha) = sup_phi E_{P_pi} phi,
    P_pi = integral P_theta pi(d theta).

For a specified family Theta, the different minimax objective is

    beta_min(alpha) = sup_phi inf_{theta in Theta} E_theta phi.

The prior, flux, noise law, masks and instrument response are part of the input.
There need not be a uniformly most powerful detector for every individual orbit.
A detection probability is not the probability that a candidate is real.

## 2. Primal/dual certificate

For any alternative Q with likelihood ratio L=dQ/dP0 and any lambda >= 0,

    E_Q phi = E0[L phi]
            <= lambda alpha + E0[(L-lambda)+] = D_Q(lambda).

Thus D is an upper bound for **every** level-alpha detector against Q. The
Neyman–Pearson test attains equality when it spends the full false-alarm budget,
including randomization at an atom L=lambda if necessary. This inequality is
classical; see equation 3.4 of Cvitanic and Karatzas, linked below.

For minimax detection, every probability mixture Q of physical alternatives
also yields an upper bound: worst-case power cannot exceed mixture-average
power. Finding least-favorable weights can tighten it, but even an imperfect
weight optimizer gives a valid upper bound. A **lower** bound requires checking
the detector throughout the family, not just its mixture average.

## 3. Alternative sampling avoids rare null events

Writing the same dual expectation under Q gives

    H = (1-lambda/L)+,              0 <= H <= 1,
    E0[(L-lambda)+] = E_Q H.

For a candidate rule psi that rejects when L>lambda and with probability rho
when L=lambda, define an auxiliary Bernoulli decision A with that probability:

    W = (lambda/L) A,              0 <= W <= 1,
    P0(psi rejects) = E_Q W / lambda.

This is ordinary importance sampling/change of measure. It does not require
observing a one-in-ten-million event under P0. It **does** require drawing Q and
evaluating its likelihood ratio correctly; it does not solve that integration
problem for an arbitrary continuous orbit population.

Independent training chooses lambda and rho. On fresh validation draws, bound
E_Q H and E_Q W from above. The implementation uses the one-sided empirical
Bernstein inequality of Maurer and Pontil (2009), Theorem 4:

    E X <= mean(X) + sqrt(2 s^2 log(2/delta)/n)
                        + 7 log(2/delta)/(3(n-1)),

where 0<=X<=1 and s^2 is the unbiased sample variance. Clopper–Pearson gives a
one-sided lower confidence bound on P_Q(A=1). We union-bound the two bounded-
expectation statements and all the necessary power statements. The events need
not be independent. Training must be independent of validation.

If a_U bounds the nominal false-alarm probability of psi, set

    gamma = min(1, alpha/a_U),       phi = gamma psi.

This supplies an executable randomized detector with certified false-alarm
control. Its power is at least gamma times the validated lower power of psi.
If this is worse than the data-independent alpha-probability coin, use that
coin. A certificate reports the achieved lower bound and universal dual upper
bound; their gap is observable, rather than inferred from optimizer convergence.

For discrete observations, ignoring boundary randomization can leave a
permanent gap even with unlimited simulation. The Poisson test exercises this.

There is a complementary failure mode: for an almost perfectly detectable
source, Q can rarely visit the region controlling the null rejection rate.
Then the importance estimate of size can be inefficient even though its
integrand is bounded. The bright-source probe records this failure.

For Gaussian mixtures the implementation also offers a standard independent
null order-statistic calibration. Among n null scores, take the kth smallest,
where k is the smallest index with

    P[Binomial(n,1-alpha) <= k-1] >= 1-delta.

Strict rejection above this threshold has size at most alpha with confidence
1-delta; atoms make it conservative. This extra event is included in the
certificate's union bound. It repairs the bright-source case at alpha=0.01.
It does not remove rare-tail cost in general: the method refuses a sample count
too small for the requested alpha/confidence. The two calibration choices have
different regimes of usefulness, not a universal cost-free calibration trick.

All confidence statements here are conditional on the specified distributions
and exact sampling/evaluation assumptions. NumPy RNG, float64 likelihood scores,
SVD, and SciPy beta quantiles are conventional numerical implementations, not a
formal enclosure of the complete statistical calculation. The reports say so.

## 4. Transfer a certificate to continuous source parameters

In the supported Gaussian experiment, whiten the data so

    P0 = N(0,I),                 P_theta = N(mu(theta),I).

Partition the entire source domain into nonempty cells C_i, of known prior
probability w_i. Find representatives m_i and justified radii r_i such that

    ||mu(theta)-m_i|| <= r_i     for every theta in C_i.

Representatives need not be physically attainable. For equal-covariance
Gaussians, the exact TV distance at mean separation r is

    TV = 2 Phi(r/2)-1 = erf(r/(2 sqrt(2))).

Call this e_i. The continuous mixture P_pi and finite approximation
Q=sum_i w_i N(m_i,I) then differ in TV by at most e=sum_i w_i e_i.
For every [0,1]-valued detector, the power difference is at most e. Therefore

    continuous optimum <= finite dual upper + e,
    power(gamma psi) >= gamma (finite power lower - e).

This is a certificate for the whole declared prior, including unsampled
parameters. A finite bank without a justified covering radius cannot use it.

For minimax detection, validate a power lower bound p_i at every representative:

    worst-case power(gamma psi) >= gamma min_i (p_i-e_i).

For the upper bound, choose any actual parameter from each nonempty cell.
A mixture on those physical parameters differs from the representative
mixture by at most sum_i v_i e_i, where v_i are the candidate least-favorable
weights. Thus finite dual upper + sum_i v_i e_i remains a valid minimax upper
bound even when the representatives themselves are not physical means.

This transfer can be very conservative. Individual distributions can differ
substantially while their prior mixtures are close. The implementation does not
pretend that a poor cellwise bound is a fundamental impossibility result.

## 5. Nuisance and model uncertainty

If the true null is within TV eta0 of the nominal P0, enforce nominal size
alpha-eta0 for the constructed detector. A test feasible under the true null
can have nominal size at most alpha+eta0, so use that larger size in the dual
upper bound. Add declared alternative TV uncertainty to the cell errors.

This guarantees robustness to the **stated** uncertainty sets. It does not
estimate their radius from tested pixels or demonstrate that real artifacts
belong to them. The generic recipe refuses eta0>=alpha: that is a limitation of
this sufficient condition, not proof that every robust detector is impossible.

Known correlated Gaussian noise can in principle be whitened before applying
the theory. The physical orbit constructor in this experiment implements only
independent noise with a specified per-exposure standard deviation. Unknown
covariance, source shot noise, fitted PSFs, finite exposure trailing and empirical
artifact distributions are not silently included.

## 6. Validated orbital image cells

The physical constructor takes a six-dimensional Cartesian state box. It uses
the same Cartesian two-body ODE across bound, parabolic and unbound energies:

    r' = v,                 v' = -mu r / |r|^3.

Arb ball arithmetic encloses floating input values and all subsequent arithmetic
and elementary functions. Each time step first finds a tube B satisfying

    initial_state + [0,h] f(B) is strictly inside B.

Together with exclusion of r=0, this establishes an enclosing Picard tube for
every initial state in the box. Taylor coefficients computed at the initial
state give the polynomial; order-(p+1) coefficients evaluated on the tube
bound its Lagrange remainder. The default order is 10. A failed enclosure is an
explicit error requiring subdivision, never permission to use an unchecked flow.

For the position series, let q=r dot r and g=q^(-3/2). The coefficient recurrence
for a general power g=q^a is

    g_0=q_0^a,
    g_n = sum_{j=1}^n (((a+1)j-n) q_j g_{n-j}) / (n q_0).

The ODE then gives r_{n+2}=-mu (r g)_n/((n+1)(n+2)). Velocity coefficients
follow by differentiation. Computing these recurrences on intervals encloses
the derivatives, including their dependence on the entire initial state box.

For each displaced observer, solve the light-time fixed point

    tau = -|r(t+tau)-observer(t)|/c.

The initial emission-time interval must map into itself. A speed bound below c
establishes uniqueness for each initial state. Interval intersections retain all
possible roots. No common light-time approximation is used across orientations.

The resulting positions pass through the supplied gnomonic WCS. Gaussian PSFs
are integrated over pixel boundaries with erf. A pixel integral is unimodal
in source position; its extrema over an interval occur at an endpoint or the
pixel center, which avoids excessive interval subtraction. Missing PSF wings
are not renormalized. Combining all pixel intervals gives a Euclidean mean ball.

Inputs use exact values of the specified numerical model: mu and c are decimal
constants in the code, observers are supplied vectors, and exposure times are
instantaneous. This validates calculations within that model, not its physical
adequacy for an arbitrary telescope.

## 7. Score compression and computational cost

For a finite Gaussian mixture, projection into the span of its mean vectors is
sufficient. An SVD reduces the simulated score dimension; the norm of discarded
mean components is added to each covering radius. The retained basis is saved
so the detector can act on the original whitened pixels. SVD rounding and
orthogonality are not formally enclosed in this research implementation.

With K cells, d retained dimensions and n validation draws, ordinary evaluation
costs O(n K d), plus SVD and physical cell construction. Minimax validation
requires power checks for all K cells, which can add another factor K without
further structure. Batching caps temporary memory but does not remove this work.
The confidence guarantee does not imply affordable full-archive computation.

### From power to a brightness limit

For a fixed Gaussian noise model and unit-flux mean family, optimal power is
nondecreasing in flux. If F1<=F2, applying the channel

    y -> (F1/F2) y + sqrt(1-(F1/F2)^2) z,      z~N(0,I),

to the F2 experiment preserves the null and produces the F1 experiment. Any
detector usable at F1 can therefore be simulated at F2. This establishes the
monotonicity needed to bracket the minimum flux attaining a target power.

At a trial flux, a universal upper power below target proves that flux
insufficient. An achieved lower power at or above target proves it sufficient.
If neither holds, the algorithm increases its validation budget or reports the
remaining interval; it does not treat an inconclusive test as a bisection sign.
The failure budget is divided across the maximum number of adaptive evaluations,
each with fresh randomness, so optional stopping/selection does not silently
invalidate the final bracket's stated simultaneous confidence.

## 8. Geometry: exact control and failed general shortcut

For alternatives on the circle ||mu||=r in R^2, the uniform-mixture LR is

    exp(-r^2/2) I0(r ||y||).

It increases with ||y||. Rejecting when ||y||^2>-2 log(alpha) is prior-optimal
and has equal power at every angle, hence is also minimax. Its power is a
noncentral chi-square survival probability. A positive Poisson series with
an explicit tail bound encloses it using Arb, independently of Monte Carlo.
This is an established symmetry argument, not a new orbital theorem.

For a convex set of Gaussian means, the nearest point to zero can give an
optimal separating matched filter. But actual orbit-template sets are generally
nonconvex. For means (+3,0) and (-3,0), their convex hull includes the null;
convex separation offers only power alpha although a two-sided test has high
power. The code uses a valid component-oracle upper bound rather than falsely
declaring convex-hull distance to be the original family's detection limit.

## Sources

- Cvitanic and Karatzas, [Generalized Neyman–Pearson lemma via convex duality](https://www.math.columbia.edu/~ik/neypea.pdf).
- Goldenshluger, Juditsky and Nemirovski, [Hypothesis testing by convex optimization](https://arxiv.org/abs/1311.6765).
- Fauß, Zoubir and Poor, [Minimax Robust Detection](https://arxiv.org/abs/2105.09836).
- Maurer and Pontil, [Empirical Bernstein Bounds and Sample Variance Penalization](https://arxiv.org/abs/0907.3740).
- Zackay and Ofek, [How to COADD Images. I](https://authors.library.caltech.edu/records/c57s0-9y491).

The Gaussian TV identity, confidence bounds, invariance example and numerical
Taylor enclosures also have established foundations. No novelty claim is made
for their combination here, and no specialist review has yet been obtained.
