# Stage-aware local Kepler grids

The first [local-grid construction](../local_resolution.md) bounded every hybrid
coordinate using an all-components inverse majorant. This is sufficient but can
lose exact cancellations: a single exposure depends on two angular coordinates,
yet its transfer was assigned a six-dimensional child grid.

This variant changes hidden coordinates first:

```
(rho, eta, wy, wx, x, y).
```

It bounds the **actual field at each interpolation stage**, expressed in child
coordinates. It retains the same whole-real-domain inverse certificates,
original state boxes, source bandwidth, whole-tree error accounting and
intermediate-grid cost formulas. It does not assume arbitrary images have small
tensor ranks. The reduction at an exposure leaf is an exact coordinate identity.

This statement concerns instantaneous angular fields. A finite-duration,
trail-aware exposure model can also depend on rate; that additional leaf
dependence would require its own resolution proof.

## The derivative that the stage actually needs

Let `S` be the already replaced coordinates. Hybrid coordinates consist of
parent coordinates on `S` and child coordinates on its complement. Just before
interpolating coordinate `i`, write the child state as `y(z)`.

For unconverted child coordinates the identities are exact:

```
dy_i/dz_i = 1,
dy_j/dz_i = 0       for j not in S and j != i.
```

For the selected coordinates, the existing verified inverse majorant gives

```
|dy_j/dz_i| <= [(I+E)(I-E)^(-1)]_(j,i),       j in S,
```

where all these coordinates are normalized using the parent widths `W`.
Define a vector `gamma` using the exact zeros and one above, and the majorant
only on `S`. If `J_child(t)` bounds the child-to-observation variational flow,
the stage's phase derivative is bounded by

```
L_i = sup_t sum_(a=x,y) K_a W_a [J_child(t) gamma]_a.
```

This is not a numerical subtraction of two nearly equal derivative estimates.
The cancellation is imposed through exact coordinate identities before any
interval multiplication.

Changing hidden/rate coordinates first is essential to this argument. During a
hidden-coordinate stage, `gamma` has support only in hidden coordinates. During
a rate stage its support lies in hidden/rate coordinates. Therefore the exact
reception identities `theta'=angular_rate` give

```
L_hidden = O(B h_child^2),
L_rate   = O(B h_child),
L_angle  = O(B),
```

under controlled analytic/inverse constants. The physical span here is the
**child's own exposure interval**, not its separation from the parent epoch.
The geometry constants still depend on that separation and remain obligations
in any archive-scale complexity claim.

## Certify the image neighborhood as well as the derivative

A derivative bound alone does not prove that the child flow is analytic where
it is evaluated. The code constructs a containing child-state rectangle.

Let `cw_j` be the padded child halfwidth, `b_i=cw_i/W_i`, and `a,s` the real and
imaginary semiaxes of the certified input-coordinate ellipse. Unconverted
coordinates retain their exact prescribed values. For a selected coordinate,
start at the real hybrid whose current coordinate is clipped to its real box.
The real inverse certificate puts its child image inside `cw`. Integrating the
holomorphic derivative over the real overhang and then the imaginary segment
gives the sufficient halfwidths

```
real_j <= cw_j + W_j gamma_j b_i (a-1+s),
imag_j <=        W_j gamma_j b_i s.
```

Both paths lie inside the previously certified hybrid rectangle. The child
variational integration checks holomorphy on this entire containing rectangle
for every time in the child segment. If that check fails, the trial is refused.
There is no acceptance based solely on sampled trajectories.

Some stages need less machinery:

- Before any coordinate has been replaced, the field is the native child
  field. Direct complex child-flow bounds suffice; a complex coordinate
  inverse is unnecessary for its interpolation error.
- A zero-duration rebase is the identity. Native tensor interpolation commutes,
  so its error decomposition can use the baseline tensor ordering even when
  physical transfers elsewhere use the new order.
- At an instantaneous exposure, until an angular coordinate has been replaced,
  the field is independent of all hidden/rate coordinates. Their error is
  exactly zero with one node. Its first angular interpolation is also a native
  angular field; only the remaining angular stage needs a mixed inverse bound.

The real geometry certificate is still required for the transfer itself. An
analytic extension of a constant or native field is not being presented as a
new inverse-domain certificate.

## Executable components

`resolution.py` constructs the stage certificates and selects sufficient local
cells/orders. `OrderedHierarchyPlan` builds the complete tree and charges every
mixed intermediate grid in the new physical order. `materialize()` checks
budgets before allocating matrices. The executable graph retains the algebraic
adjoint and continuous root-field evaluation.

`verify.py` recomputes every stage's field neighborhood and phase bound, verifies
its exact-constant/native/mixed classification, checks the allowed ordering,
replays the coordinate certificates, and reproduces the tree's accumulated
error and integer costs. Tests deliberately change the stage order and turn an
instantaneous interval into a finite interval; the verifier refuses both.

```bash
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.ordered.experiments NEW_DIRECTORY --mode local --case near
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.ordered.experiments NEW_DIRECTORY --mode hierarchy --case near
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.ordered.experiments NEW_DIRECTORY --mode numerical
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.ordered.verify NEW_DIRECTORY/results.json
```

The full original near and distant domains use a single real chart cover in
these experiments. The baseline atlas construction remains available separately;
this variant's hierarchy driver does not silently discard states if a domain
needs further subdivision.

See [saved results](../local_resolution_results/README.md) for the comparisons.
Small child grids do **not** make a full parent orbit field equally small. Parent
output grids, mixed-grid matrices, construction, input preparation and queries
remain part of the cost. All analytic guarantees here concern exact-arithmetic
Fourier-field interpolation; floating-point guarantees and the full FKT
objective remain unresolved.
