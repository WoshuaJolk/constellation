# A third route: can an exact finite change of features turn Kepler into Fourier?

One possible full-objective shortcut would be an exact fixed-size phase lift

```
phi(q,s) = sum_{j=1}^r A_j(q) B_j(s) + a(q) + b(s),
```

with finite `r` independent of accuracy, bandwidth and how many trajectories
are sampled. A nonuniform Fourier transform in the lifted coordinates could
then exploit linearity in the orbit features. The functions `A_j,B_j` could
be nonlinear; only their number is fixed.

This exact proposal fails already for a regular one-parameter circular
Kepler family. That does **not** rule out approximate lifts, hierarchical
transforms, or other fast nonlinear algorithms.

## A regular counterexample, including light time

Take a stationary observer at the central origin and circular orbits in the
`x,y` plane, with radius `R` in a positive bounded interval. This idealized
observer is a special case of the unrestricted mathematical transform, not
the moving-observer configuration used in the implemented campaign.
The angular frequency is `omega=sqrt(mu/R^3)`.

Reference the initial orbit at emission time `-R/c`, with apparent direction
along the positive x axis at reception time zero. The observer distance is
constant, so reception light time is exactly `R/c`. At reception time `t`,
the emitted position is

```
(R cos(omega*t), R sin(omega*t), 0).
```

In the fixed tangent projection with normal x and east y, its coordinate is
`P(omega,t)=tan(omega*t)`. Restrict
`|t|<T<pi/(2*omega_max)`. The whole family is collision-free, analytic and
separated from the projection horizon. There is no singular-orbit loophole.

At a fixed nonzero spatial Fourier frequency, finite phase separation would
require a finite separation
`tan(omega*t)=sum_j A_j(omega) B_j(t)`, allowing two extra terms to absorb
the phase gauges `a,b`.

## Infinite separation rank

The Taylor series is

```
tan z = sum_{n>=0} a_n z^(2n+1),   a_n>0.
```

Positivity follows directly from `y'=1+y^2`: `a_0=1` and

```
(2n+1) a_n = sum_{i+j=n-1} a_i a_j,   n>=1.
```

Choose any `N` distinct positive frequencies `omega_1,...,omega_N`.
If `sum_j c_j tan(omega_j*t)=0` on the observation interval, the first `N`
odd derivatives at zero give

```
sum_j c_j omega_j^(2n+1) = 0,  n=0,...,N-1.
```

The coefficient matrix has determinant

```
(prod_j omega_j) * prod_{i<j}(omega_j^2-omega_i^2),
```

which is nonzero. Thus all `c_j=0`. There are arbitrarily many linearly
independent time functions, contradicting any finite separation rank.
Adding orbit-only or source-only phase gauges changes rank by at most two
and cannot repair the contradiction.

Equality of continuous real phases modulo `2*pi` does not evade the argument:
on a connected domain their difference must be a constant integer multiple
of `2*pi`, another removable finite-rank gauge.

## What this rules out, and what remains open

An exact, finite, globally shared set of phase features cannot turn even this
regular Kepler subfamily into a single fixed-dimensional Fourier phase.
The common-clock regularization shortcut fails for a different reason:
its source times depend on the orbit; see
[the regularization attempt](regularization_attempt.md).

This is **not** a computational lower bound for the general transform.
An approximate separation on the bounded interval may converge rapidly.
A Fourier, butterfly, wave-packet, rational, implicit, or variable-rank
construction is not disproved. The implemented
[global hierarchy](global_theory.md) explicitly uses growing orders and local
demodulation, so it is consistent with this obstruction.

This elementary argument is supplied to make this attempted route falsifiable,
not as a claim of mathematical novelty. It identifies precisely the assumption
that must be abandoned in a further Fourier-lift attempt.
