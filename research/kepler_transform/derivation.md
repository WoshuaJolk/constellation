# Identities and construction cost

This derives the experimental operators, not a new theorem or completed search.
See [results and limitations](README.md).

## Linear sufficient statistics

For known Gaussian noise and background, orbit theta has

```
N(theta) = sum_j p_j(theta)^T W_j d_j
I(theta) = sum_j p_j(theta)^T W_j p_j(theta)
log L(F,theta) - log L(0) = F N(theta) - F^2 I(theta)/2.
```

The transforms concern scalar response sums such as N and I. Flux optimization,
N/sqrt(I), and detection occur afterward. The experiments use harmonic response
fields; constructing both statistics from realistic images with certified error
is still required.

## Three scalar parameters and a rotation

At a common epoch choose r0=(rho,0,0), v0=(vr,vt,0), with rho>0 and vt>=0.
Rotating both by R in SO(3) generates the orientation part of state space.
For nonradial motion these are all six dimensions; radial motion is a
degenerate limit with redundant roll. Energy may be negative, zero or positive.

The two-body equation is rotationally equivariant:

```
r(t;rho,vr,vt,R) = R r(t;rho,vr,vt,identity).
```

Write the planar canonical position as rho(t)(cos(phi(t)),sin(phi(t)),0).
`universal.py` evaluates this with an established universal-variable method.

## Scalar response identity

Suppose f_j(u)=sum_lm a[j,l,m] Y_lm(u), through degree L. Use active rotations
R=Rz(alpha) Ry(beta) Rz(gamma), and
D_lmn(R)=exp(-i m alpha) d_lmn(beta) exp(-i n gamma). Then

```
Y_lm(R u) = sum_n D_lmn(R)^* Y_ln(u)
Y_ln(pi/2,phi_j) = Y_ln(pi/2,0) exp(i n phi_j).
```

Substitution into S(R)=sum_j f_j(R u_j) gives

```
S(R) = sum_lmn C[l,m,n] D_lmn(R)^*
C[l,m,n] = Y_ln(pi/2,0) sum_j a[j,l,m] exp(i n phi_j).
```

The inner sums are type-1 NUFFTs. Nonuniform phases, including parabolic and
hyperbolic arcs, do not require periodic orbital motion. C represents the
continuous orientation response; the grid evaluates and validates it.

For a central observer, t_obs=t_emit+rho(t_emit)/c determines an emission time
independent of R. The identity applies at those phases. With a displaced
observer emission time depends on R, breaking this simple scalar construction.

## Directional extension

Let the central-observer emission time in frame j define rho_j, vr_j, vt_j,
and phi_j. Set Q_j=R Rz(phi_j) Ry(pi/2). Its z-axis is radial and its y-axis
is tangential. The state at that reference time is

```
r = rho_j Q_j zhat
v = vr_j Q_j zhat + vt_j Q_j yhat.
```

The full reception response, including additional light-time iteration and
the actual observer, is a scalar function F_j(Q) on SO(3). Its roll angle
retains velocity-direction information. Expand

```
F_j(Q) = sum_l,m,s b[j,l,m,s] D_lms(Q)^*.
```

Representation composition gives

```
D_lms(R Rz(phi_j) Ry(pi/2))^*
    = sum_n D_lmn(R)^* exp(i n phi_j) d_lns(pi/2),
```

and therefore

```
C[l,m,n] = sum_s d_lns(pi/2)
                    sum_j b[j,l,m,s] exp(i n phi_j).
```

This is the implemented, independently checked directional identity. It is
exact for the specified finite response expansions. It does not show that b
is cheap to construct or that truncation preserves arcsecond sources. The
scalar case has only s=0 with b[j,l,m,0]=a[j,l,m]sqrt((2l+1)/(4pi)).

The physical test constructs b by sampling exact retarded responses at every
local orientation and performing Wigner analysis. That fixes the coupling in
the low-degree test but incurs O(NT) geometry/response work. Fast accumulation
cannot cancel that construction cost.

## Adjoint and limited inverse

Euclidean complex adjoints pair positive-phase type-1 NUFFTs with negative-
phase type-2 NUFFTs and transpose the synthesis FFT normalization. Random
complex tests check <Ax,y>=<x,A*y>.

With alpha/gamma at M=2L+1 uniform nodes and cos(beta) at L+1 Gauss-Legendre
nodes with weights w_b, inverse Wigner synthesis is

```
C[l,m,n] = (2l+1)/(2 M^2)
           sum_b w_b d_lmn(beta_b) FFT_alpha,gamma(S_b)[m,n].
```

It recovers finite rotation coefficients, not original time-separated images.
Time accumulation can have a nullspace; the full-likelihood counterexample is
given in the README. An adjoint is not relabeled as an inverse.

## Honest complexity

For maximum degree L, T exposures, and spin limit S:

- Scalar time accumulation uses O(L^2) NUFFTs; directional accumulation uses
  O((2S+1)L^2), each with T points and O(L) modes. At fixed precision this is
  nominally O((2S+1)L^2(T+L log L)).
- This prototype stores Wigner d at every beta sample, using O(L^4) memory.
  Its low-degree factorial formula takes O(L^5) arithmetic to build the table.
  It deliberately rejects L>12. Stable recurrences and established spherical
  transform libraries would be required above this range.
- Its synthesis contracts O(L^4) terms before the FFTs. This is not a claim
  of an optimal SO(3) transform.
- The physical response construction samples O(T L^3) orientations: precisely
  the work the full objective needed to eliminate. It is explicitly counted.
- Every result still fixes (rho,vr,vt). A continuous radial-domain covering
  theorem and efficient representation are absent.

The identity removes the exposure loop from querying a previously constructed
rotation expansion. That is an intermediate result, not the complete algorithm.

## Near-observer bandwidth

Small v/c does not ensure a small error in PSF units. The angular correction
can scale as v_t |observer|/(c * observer_target_distance). A nearby target can
move through many PSF widths as its velocity direction changes.

For a circular angular displacement of radius a through a Gaussian response,
roll dependence resembles exp[-kappa(1-cos(psi))], with kappa proportional to
(a/sigma)^2. Its Fourier bandwidth grows with a/sigma. The exact geometric
tests measure this growth without using the local approximation to calculate
the response. Uniformly keeping only a few roll modes is not sufficient.

A successful continuation must construct a localized representation with
controlled error and total cost across the full domain. This work has not
established one.
