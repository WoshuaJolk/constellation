# Attempt: linearize the dynamics before transforming the images

This is a second mathematical route to the full objective. It uses an
established Sundman/Burdet–Heggie regularization identity; no novelty is claimed
for the dynamics. It explains exactly where a tempting global FFT shortcut
fails, while leaving more sophisticated lifted constructions open.

Let the heliocentric position be `r`, radius `R=|r|`, velocity `v`, energy
`E=|v|^2/2-mu/R`, angular momentum `h=r cross v`, and eccentricity vector
`e=(v cross h)/mu-r/R`. Introduce `s` by `dt/ds=R`.

Direct differentiation gives

```
r_s = R v
r_ss = -mu r/R + (r dot v) v.
```

The eccentricity identity gives
`(r dot v)v = (|v|^2-mu/R)r-mu e`. Therefore

```
r_ss = 2E r - mu e,
R_ss = 2E R + mu,
t_s  = R.
```

The vector dynamics and the scalar radius obey linear constant-coefficient
equations. Elliptic, parabolic and hyperbolic cases correspond to negative,
zero and positive `2E`; the formula is regular at `E=0`. This is a precise
linearization, not a short-time Taylor approximation.

Set `lambda=2E`, `z=-lambda*s^2`, and use the entire Stumpff functions
`C(z), S(z)`. With epoch-zero values:

```
r(s) = r0 + R0*v0*s
       + (lambda*r0-mu*e)*s^2*C(z)
       + lambda*R0*v0*s^3*S(z),

t(s) = R0*s + (r0 dot v0)*s^2*C(z)
       + (lambda*R0+mu)*s^3*S(z).
```

The implementation is checked against the independent universal-variable
propagator evaluated at these physical times, including both energy signs and
near-zero energy. Those checks validate the identity's implementation, not an
all-orbit transform.

## Why one shared exposure clock does not follow

The clock has derivatives at the reference epoch

```
t_s(0)   = R0
t_ss(0)  = r0 dot v0
t_sss(0) = 2E R0 + mu.
```

If a single physical-time-to-Sundman-time mapping worked across an open
six-dimensional orbit domain, its inverse `t(s)` would have these same
derivatives for all states. Varying initial radius already contradicts the
first equality. Even at fixed radius, varying radial velocity contradicts the
second; fixing those still leaves energy dependence in the third.

Thus a single FFT in the regularized clock cannot simply reuse the original
exposure times for all orbits. Per-orbit clock inversion/resampling restores
the coupling we were trying to eliminate. Fixing the clock invariants would
return to a restricted family.

This is an obstruction to that specific common-clock shortcut, not a proof
that harmonic regularization cannot contribute to a global transform.

## Exact reception-time change of variables

Let emission position be `r(s)`, emission time `T(s)`, reception observer
`O(t)`, and `n=(r(s)-O(t))/|r(s)-O(t)|`. Reception time obeys

```
t = T(s) + |r(s)-O(t)|/c.
```

Implicit differentiation gives

```
dt/ds = R(s) [1+n dot v(s)/c] / [1+n dot O_dot(t)/c].
```

For subluminal source/observer velocities and nonzero separation this is
positive. A continuous-time response integral transforms exactly as

```
integral f(t, projection(r_emit(t)-O(t))) dt
 =
integral f(t(s), projection(r(s)-O(t(s)))) (dt/ds) ds.
```

For discrete exposures, the time source is a sum of weighted Dirac masses.
The identity is interpreted distributionally; evaluation still requires the
orbit-dependent reception roots. A missing Jacobian or treating `T(s)` as a
common exposure clock would change the requested operator.

## A lifted Fourier route remains unresolved

For bound energy, `r(s)` is an affine combination of a constant, a sine and a
cosine, while `T(s)` is a linear drift plus sine/cosine terms. In a
heliocentric **space-time** Fourier representation, individual phases can
therefore be expressed with these harmonics.

However the observed images live on projected sky coordinates, not a free
three-dimensional heliocentric volume. A complete lift must count radial
representation, observer translation, retarded timing, projection,
energy-dependent frequencies and finite observation windows. The lifted
coefficients depend on the orbital invariants. We have not derived a fast
global contraction of all those variables or a stable inverse.

We retain the exact identity and the common-clock obstruction rather than
calling the linearized dynamics a solution of the image-likelihood problem.

Primary context:
[Saha, interpreting the KS transform](https://arxiv.org/abs/0803.4441);
[Breiter and Langner, arbitrary-vector KS transformation](https://arxiv.org/abs/1701.02147).
These are established regularization methods. The displayed vector identity
above follows directly from the eccentricity vector and does not require
implementing a four-dimensional KS coordinate map.
