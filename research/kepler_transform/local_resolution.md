# Locally sufficient grids and their actual total cost

This construction replaces a guessed uniform interpolation order with verified
coordinate-dependent cell counts and orders. It closes the **exact-arithmetic
Fourier-field interpolation budget through a complete exposure tree**. It does
not yet make that tree affordable at archive scale, certify floating-point
orbit evaluation, prepare arbitrary detector pixels, or supply the full FKT
inverse. The saved campaigns distinguish planned grids from allocated grids.

Implementation:

- `local_resolution.py`: one-coordinate analytic neighborhoods, all-time phase
  derivative bounds, cell/order optimization and interval-checked error bounds.
- `local_hierarchy.py`: sufficient incoming grids, whole-tree error allocation,
  exact integer storage/application counts, covered atlases and optional execution.
- `local_resolution_experiments.py`, `verify_local_resolution.py`: provenance,
  original-domain campaigns and independent replay of the retained certificates.

## 1. Give each coordinate its own neighborhood

The earlier common complex polyellipse forced all coordinates to share a narrow
analytic radius. A slowly varying hidden coordinate then inherited essentially
the same large interpolation order as an angular coordinate.

For the normalized exact reception flow

```
R(c + W x) = c_child + W (x + d(x)),
```

we now certify a separate neighborhood for each interpolation direction `i`.
Other hybrid coordinates range over their **entire real boxes**. Only `i` is
complex, with Bernstein parameter `r0_i`. Its implicit inverse can have imaginary
parts in all six physical state coordinates; the enclosure includes those parts.
This is not a check only along the real coordinate axis.

The real/imaginary rectangle self-map is the same whole-box contraction used in
[the constructive factorization](constructive_factorization.md). If `Er` and
`Ei` bound the real and imaginary parts of `Dd`, use

```
E = Er + Ei >= |Dd|,
S = (I-E)^(-1).
```

A floating-point proposal supplies positive weights; interval arithmetic verifies
the weighted norm is below one. Interval matrix inversion then bounds the
nonnegative Neumann sum. Every mixed inverse satisfies `|Dx/Dz| <= S`.
A refusal enlarges neither the claimed domain nor the claimed accuracy.

For a child time interval, interval Picard tubes and nonnegative variational
majorants enclose

```
J(t) >= |W^-1 D R(start,t) W|
```

for **every time** between the endpoints, including gaps without exposures.
Each step uses `exp(|A| |dt|)` with a bounded series remainder; the same
nonnegative majorant bounds partial steps. Forward and backward sweeps cover
both sides of the parent epoch. This work depends on the physical span and
integration step, not on the number of exposures in that span.

For angular frequencies `|kx| <= Kx`, `|ky| <= Ky`, a certified phase derivative is

```
L_i = sup_t sum_(a=x,y) K_a W_a [J(t) S]_(a,i).
```

The Fourier kernel is real-phase on real states, so with only `z_i` complex,
`|exp(i k.theta)| <= exp(L_i |Im z_i|)`. These are bounds for the declared
analytic observer/Kepler/light-time model, not arbitrary WCS or PSF functions.

## 2. Select sufficient local cells, not one order for the whole box

Let the hybrid halfwidth in parent-normalized coordinates be `b_i`. On each of
`n_i` equal cells, a local ellipse with parameter `r_i` has semiaxes

```
a_i = (r_i + 1/r_i)/2,       s_i = (r_i - 1/r_i)/2.
```

It fits inside the certified neighborhood if

```
n_i (a0_i-1) >= a_i-1,
n_i s0_i     >= s_i.
```

For `r_i <= r0_i`, inclusion follows directly by monotonicity. Handling equality
symbolically avoids gratuitously doubling a grid because interval division of
an expression by itself straddles one.

For `p_i` Lobatto nodes, use `Lambda_i = 1` for orders 1 and 2 and
`Lambda_i = sqrt(2 p_i-2)` otherwise. Set `u_i=L_i b_i/n_i`. A sufficient local
interpolation error, per unit input Fourier-coefficient l1 norm, is

```
delta_i <= 2 (1+Lambda_i) r_i^(-p_i)/(1-1/r_i)
           * exp(u_i s_i) * min(1, u_i a_i).
```

The second option in the minimum follows by subtracting the function value at
the real cell center. Interpolation preserves that constant. The derivative
bound controls the remainder throughout the ellipse, including its real
extent. Thus a genuinely constant direction needs just one node; a weak
coordinate is not assigned a large order solely because another coordinate
oscillates rapidly.

The optimizer enumerates local radii and orders 1 through 64, solves for a
sufficient integer cell count, and checks both inclusion and error with Arb.
It minimizes `n_i p_i` among the tested choices. It can require nested multiples
of preexisting cell counts. It is **not** a globally optimal mesh solver.
Initially conservative separate budgets are relaxed using actual selected
Lebesgue factors, without increasing any factor, and the full recurrence is
checked again. High bandwidth causes real cell subdivision, not only a change
in order. Cells are uniform along each axis of each atlas patch, with different
sizes across coordinates, patches and segments; there is no unimplemented
claim of arbitrary spatially adaptive unstructured cells.

## 3. Close the whole-tree budget without circular refinement requirements

An incoming transfer chooses the child's stored grid. Consequently its input
values already lie on the sufficient working grid. The child's own values are
constructed recursively from its children. No internal transfer resums the
original exposures, and this variant needs no additional prolongation stage.

For an edge from child `c` to parent `v`, let

```
A_c = product_i Lambda_(c,i),
d_c = six-stage interpolation error bound.
```

If the child's nodal error per unit coefficient norm is `e_c`, the parent's
nodal error obeys

```
e_v <= max_children (A_c e_c + d_c).
```

The maximum, rather than sum, is valid because child coefficient blocks are
disjoint: normalization uses their **total** l1 norm. Instantaneous leaf
samples are exact in this mathematical model, so `e_leaf=0`. Floating-point
leaf evaluations remain a separate obligation.

At the root, interpolate the resulting nodal field. With root interpolation
error `d_root` and amplification `A_root`, the continuous-domain error is

```
e_continuous <= d_root + A_root e_root.
```

Allocate half the requested budget to root interpolation, then split each
node's nodal budget between transfer error and amplified child error. All
returned bounds and allocations are outward/downward rounded as appropriate.
An atlas returns success only when every original state is covered by a
complete sufficient tree. The verifier checks its cover with exact rational
rectangle arithmetic. Queries can select one containing patch; overlapping
patches are not summed as independent physical likelihoods.

## 4. Count every intermediate grid

Write `m_(v,i)=n_(v,i) p_(v,i)` and `G_v=product_i m_(v,i)`.
During a child-to-parent cascade, after replacing coordinate subset `S_j`,

```
G_(edge,j) = product_(i in S_j) m_(parent,i)
            * product_(i not in S_j) m_(child,i).
```

The stored interpolation weights and multiply-add pairs per data application
are exactly

```
W_edge = sum_j p_(child,axis_j) G_(edge,j).
```

The program sums these integers over the entire graph, rather than multiplying
one favorable transfer's cost by an assumed number of levels. It also reports
all field nodes, peak intermediate nodes, coefficient bytes, query stencil
size and a bound on numerical rebase state evaluations using the solver's
iteration cap. Geometry evaluation itself has a precision-dependent cost;
that cost is not silently treated as a unit operation. Planning time includes
failed neighborhood and atlas attempts.

A Fourier leaf currently stores a dense kernel: with `M_t` coefficients, its
cost is `M_t G_leaf`. Source preparation and this input-dependent leaf cost are
separate from transfer counts. Materialization charges leaf and transfer entries
together before allocation. This is a remaining practical bottleneck, not an
input-plus-output implementation claim. Root queries cost `product_i p_i`
weights per query in the current tensor evaluator.

## 5. A sufficient resolution law, with explicit hypotheses

There is a useful mathematical consequence beyond the finite counts. It
requires *uniform* regularity constants; a few successful certificates do not
establish such constants for the public archive.

Assume a fixed bounded regular state/time domain has a finite atlas with:

1. common positive complex-neighborhood margins;
2. uniformly bounded normalized ODE derivatives on the flow tubes;
3. uniformly bounded `S` for the hybrid inverses (split long transports into
   short steps if necessary);
4. uniformly controlled chart widths and atlas multiplicity.

These are geometric/analytic conditions, not an assumed grid resolution law.
The executable certificates verify their finite-instance counterparts.

Use groups `A=(x,y)`, `V=(wx,wy)`, `H=(rho,eta)`. Because reception coordinates
have the **exact** identities `theta'=V`, the generator's `A,H` block vanishes.
Integrating the variational equations once and twice gives, on the stipulated
complex neighborhood,

```
D R(t)_(A,V) = O(|t|),
D R(t)_(A,H) = O(t^2),
D R(t)_(A,A) = O(1).
```

The constants include physical widths and the bounded variational growth; this
is not a truncated constant-acceleration orbit model. Intermediate hybrids
retain these powers. Indeed, writing `E=O(|d|)` for the rebase increment,
`E_(A,H)=O(d^2)`, and using

```
S = I + E + E^2 S,
(I+D(t)) S = I + D(t) + E + E^2 S + D(t) E S,
```

the angular-to-hidden block is `O(t^2+d^2+|td|)`. The angular-to-rate block is
`O(|t|+|d|)` and the angular-to-angular block is bounded. For an edge inside a
segment of span `h`, this gives

```
L_A <= C B,       L_V <= C B h,       L_H <= C B h^2.
```

Choose a fixed local ellipse radius `r>1`. The constructive choice

```
n_i >= max(1, geometric inclusion requirements, b_i L_i (r-1/r)/2)
```

bounds each modulus by `e`. A sufficient order is logarithmic in the reciprocal
local error budget, up to logarithmic order/stability factors. Thus, for `B>=1`,
a sufficient field size is bounded by

```
C p^6 B^2 (1+B h)^2 (1+B h^2)^2.
```

This derives the earlier candidate law from analytic conditions; it does not
assume that law as an extra premise. A child uses its *parent's* span because
of the incoming transfer. Intermediate mixed grids can use the grandparent's
span. Only a bounded number of nodes/edges charge any one grandparent, so this
changes a constant in the tree sum, not an exposure factor.

For a count-balanced tree of arbitrary cadence and depth `D`, disjoint time
intervals at each level and nonnegative polynomial coefficients give

```
sum G(h_v) <= (2N-1) G(0) + D [G(T)-G(0)].
```

With `G` denoting the displayed size envelope, the six-stage application and
weight-storage bounds are `O(p * sum G(h_v))`, times the controlled atlas
factor. The error recurrence makes the needed order grow with
`log(1/epsilon) + D log p`, rather than with the number of trajectories. The
mathematical construction permits increasing `p`; the executable TensorGrid
currently caps it at 64 and adds cells when necessary. **Do not infer the
unbounded asymptotic order statement for that fixed-cap implementation.**

The constants can be very large, especially over long time spans or near
singular geometry. This lemma supplies a constructive route and identifies
what must be controlled; it is not a practical whole-archive complexity result.
All actual saved counts come from interval-verified local rules, not this
asymptotic envelope or extrapolation from sampled errors.

## 6. What the counts tell us to improve next

The local sufficient grids can be orders of magnitude smaller than the old
uniform-order estimates. But multiplying six axis sizes, propagating child
error budgets and storing every intermediate matrix still gives very large
complete trees. The new accounting makes those costs visible.

The next concrete reductions are to exploit the guaranteed two-dimensional
structure of exposure leaves, evaluate/apply transfer weights in streamed
blocks, tighten the stability/error allocation across the tree, and replace
conservative variational bounds where they dominate cell size. Any resulting
compression must retain the same continuous-domain budget and include its
construction, application, memory and query costs.

A specific further bound-tightening candidate is to factor gravity before
interval evaluation. With positive inverse range `rho`, sky direction vector
`D`, observer position `O` and `P=D+rho O`, the real Kepler acceleration is

```
g = -mu rho^2 P / (P.P)^(3/2).
```

Evaluating this expression can avoid the repeated uncertain divisions in
`r=O+D/rho` followed by `-mu r/(r.r)^(3/2)`. A replacement complex enclosure
would need its own analytic-branch and variational-equation verification. This
is an unimplemented constructive lead; none of the saved certificates relies
on it or assumes the resulting constants will be small.
