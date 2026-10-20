# A constructive route to a hierarchical Kepler transform

This investigation addresses the archive-scale objective, not a larger
three-image benchmark. It supplies two executable improvements to the proposed
factorization and a cadence-independent counting argument. **The full FKT,
its practical archive cost, and its data-reconstruction inverse remain open.**

The new code proves the existence of the *coordinate inverses used inside a
transfer*. These are not inverses of the image-to-orbit likelihood transform.

## 1. The proposed full algorithm

For a bounded regular physical state domain:

1. Prepare each exposure's angular likelihood sufficient-statistic fields.
   Every input pixel must be charged, along with WCS, PSF and noise handling.
2. Organize exposures into a binary tree by sample count. Use actual physical
   time spans to choose resolution.
3. At leaves, represent the two angular coordinates. At longer segments,
   resolve angular position, angular rate, inverse range and radial rate as
   their influence on the likelihood becomes measurable.
4. Refine a child's existing polynomial representation to the resolution
   needed by a transfer's intermediate fields. Use separable refinement.
5. Rebase the child field to its parent through six one-dimensional
   interpolations. Construct certified coordinate charts and error orders
   before accepting the transfer. Subdivide a domain if necessary, retaining
   every patch.
6. Add the child fields and retain a reusable representation of the parent.

The intended saving comes from **the sum of the sizes of all segment fields**,
not from calling a fast orbit integrator for every orbit at every exposure.
Neither refinement nor an internal transfer re-reads the image stack.

This is related to established semi-Lagrangian transport. In particular,
[Purser and Leslie's cascade interpolation](https://doi.org/10.1175/1520-0493(1991)119%3C2492:AEIPFH%3E2.0.CO;2)
factors multidimensional interpolation into one-dimensional stages.
[Kormann's tensor-train Vlasov solver](https://arxiv.org/abs/1408.7006)
is another constructive lead for six-dimensional transport. Its compression
results do not establish small tensor ranks for arbitrary astronomical images,
so this construction does not assume such ranks. These techniques are prior
art, not claimed inventions of this project.

## 2. A whole-domain construction of every mixed inverse

Let `R` be the exact Kepler reception-coordinate flow from the parent epoch
to the child epoch, including the declared observer and light travel time.
Normalize the parent coordinates using positive diagonal widths `W`:

```
R(c + W z) = c_child + W (z + d(z)).
```

The child center need not be known exactly. Its verified error is included
in `eta >= |W^-1 (R(c)-c_child)|`.

On an enclosing real box `|z_i| <= a_i`, enclose

```
|Dd(z)| <= E                    componentwise
delta_i = eta_i + sum_j E_ij a_j
b_i = 1 + delta_i.
```

Suppose the checked inequalities are

```
b_i + delta_i < a_i,
||diag(a)^-1 E diag(a)||_infinity < 1.
```

For **any subset** of fixed parent coordinates `S`, and any target hybrid
with `|z_S| <= 1` and `|y_U| <= b_U`, the unknown coordinates solve

```
x_U = y_U - d_U(z_S, x_U).
```

The right side maps the entire unknown box into its interior and is a strict
contraction. Banach's theorem therefore supplies one unique solution in that
box for every target hybrid, including targets between grid nodes. There are
64 such coordinate subsets in six dimensions; one enclosure covers all of
them. This establishes more than sampled Jacobian nonsingularity.

If coordinate `j` is fixed on the parent side, its mapped value obeys
`|R_normalized,j| <= 1+delta_j = b_j`. Consequently each one-dimensional
interpolation looks up values inside the padded child field. Padding is
constructed from the certificate instead of selected heuristically.

This direct contraction is a simple verified inverse construction, related
to the broader interval verification methods reviewed by
[Rump](https://www.tuhh.de/ti3/rump/intlab/ActaNumerica2010.pdf).
The novelty claim is **none**; the contribution here is applying and
implementing a checkable construction for this Kepler factorization.

### Verified flow derivatives

`variational_flow.py` differentiates the reception ODE by first-order interval
automatic differentiation. A state tube is enclosed by Picard inclusion.
On each tube, it bounds the scaled generator `A(t)=W^-1 Df W`.

For a step of length `h`, the derivative satisfies the Peano--Baker series.
An interval matrix power encloses every time-ordered product of generators;
the argument does **not** assume generators at different times commute.
The remaining terms are bounded by

```
exp(x) x^(m+1)/(m+1)!,    x = |h| sup ||A(t)||_infinity.
```

Successive derivative increments are composed as
`D_new = D_local + D_old + D_local D_old`, avoiding cancellation in `J-I`.
The result is an enclosure over a whole initial box. Independent Cartesian
propagation and high-precision differences are regression checks; they are
not the basis of the enclosure.

For perspective, validated integration of variational equations is an
established capability of libraries such as
[CAPD](https://capd.sourceforge.net/capdDynSys/docs/html/a05232.html).
The implementation here is intentionally small and conservative.

### Complex neighborhoods and interpolation

Analytic inverse maps are needed to derive interpolation errors. The code
extends the preceding construction to complex rectangles containing the
Bernstein ellipses of every hybrid grid.

It preserves the real axis in its bounds. For `z=x+i y`,

```
d(x+i y) = d(x) + integral_0^1 Dd(x+i s y) i y ds.
```

Writing interval bounds for real and imaginary parts of `Dd` as `E_r,E_i`,
the real displacement is bounded by `delta_real + E_i |y|`; the imaginary
displacement is bounded by `E_r |y|`. The nonnegative block matrix

```
[ E_r  E_i ]
[ E_i  E_r ]
```

controls contraction in twelve real coordinates. A floating-point linear
solve proposes positive norm weights; interval arithmetic then verifies the
weighted norm is less than one. The proposal is not trusted as a proof.
Strict inclusion and contraction give a unique holomorphic inverse on an
open neighborhood of the tested hybrid ellipses.

Broad interval expressions can lose correlations. In that case the
derivative enclosure subdivides the entire inverse-range interval and unions
all resulting bounds. If a single transfer patch still cannot be certified,
`certify_atlas` bisects the parent state domain, with outward rounding. It
returns a certificate only after every patch has passed. Original states
and conic classes are never removed.

For a child Fourier angular field, interval propagation gives

```
|exp(i k.theta)| <= exp(sum |k_i| sup |Im theta_i|) = M.
```

Per unit source-coefficient l1 norm, a one-dimensional `p`-node Lobatto
interpolation has error at most

```
delta_p = (1+Lambda_p) * 2 M r^(-p)/(1-1/r).
```

The six cascade errors obey `e_j <= Lambda_j e_(j-1)+delta_j`.
`sufficient_uniform_order` finds an order satisfying a requested bound and
reports the implied grid size **without allocating it**. These conservative
orders can be much larger than observed numerical accuracy suggests.
They do not establish affordable physical-resolution construction.

This bound concerns the exact transfer at parent nodes, with exact analytic
child values. An existing uniform child-field error is multiplied by the
reported product of Lebesgue constants. Final parent-field interpolation,
floating-point geometry, WCS and detector approximations require additional
error accounting.

## 3. Refine before changing coordinates

A source grid that resolves the child field need not resolve every intermediate
hybrid field. For example, after replacing a child angular coordinate by a
parent angular coordinate, the remaining child angular-rate coordinate can
affect the field through the entire epoch separation. Its required resolution
can be much finer than the original short child's rate resolution.

The repair is constructive:

1. Refine the child's piecewise tensor polynomial to a sufficiently fine,
   compatible Cartesian grid.
2. Apply the one-dimensional cascade on that working grid.

`TensorProlongation` performs step 1 with six one-dimensional sparse matrices,
applied along tensor axes. The target cells are nested inside source cells,
and target polynomial orders are no lower. Consequently the represented
piecewise polynomial is unchanged in exact arithmetic. Duplicated cell
endpoints retain their correct one-sided traces even for discontinuous source
polynomials. There is no image or orbit evaluation in this operation.

For intermediate tensor sizes `G_j` and source orders `p_j`, its application
cost is `sum_j p_j G_j`, not `p^6 G`. Matrices are stored by axis instead of
materializing their six-dimensional Kronecker products. The subsequent
cascade also costs `sum_j p_j G_j` once its geometry has been constructed.
Geometry, proof, refinement and remapping costs are all reported.

The reproducible polynomial experiment isolates this issue. The child field
is represented **exactly** by two nodes per coordinate. Increasing only the
parent grid leaves a transfer error around `1.1e-3`. Refining the child
representation before mixing coordinates reduces the measured parent-node
error to about `9.5e-11` with five nodes per working coordinate. This is a
finite transfer regression, not a full-stack error or archive-speed result.

### Complete recurrence and retained-domain atlas

`CertifiedSegmentTransform` integrates the domain certificates and separable
refinement into the existing bottom-up field recurrence. Leaf exposure models,
data application, the algebraic adjoint and the finite nodal regularized
inverse retain their previous definitions. The interpolation orders are still
user-supplied research settings, not an automatically sufficient accuracy rule.

`CertifiedAtlasTransform` retries a failed whole graph on subdivided parent
boxes until every patch has a complete certified geometry graph. It charges
the sum of retained nodes and matrix entries and reports the time spent on
failed attempts. Queries are assigned deterministically to one covering patch.
Its adjoint uses the same assignment. Its finite inverse uses the concatenated
root-nodal outputs of all patches with an explicit Euclidean measure.

The saved campaign retains all three original six-coordinate domains and their
original cadences/durations, including the 32-exposure distant case. The verifier
checks the final atlas cover with exact rational rectangle arithmetic and
replays every retained real/complex transfer certificate. These facts certify
coverage and coordinate maps, **not sufficient likelihood resolution**.

## 4. Why a hierarchy can avoid the exposure-by-orbit product

The reception chart has exact identities

```
theta(q,0) = (x,y),       d theta/dt(q,0) = (wx,wy).
```

On a regular domain with bounded derivatives, dependence on angular position
is order one; additional dependence on angular rate starts at order `h`;
dependence on inverse range and radial rate starts at order `h^2`.
The last statement follows by integrating angular acceleration twice. It
does not approximate a Kepler orbit by a constant-acceleration path.

This motivates the dimensionless resolution model

```
G(h) = C B^2 (1+B h)^2 (1+B h^2)^2.
```

`C` must account for physical widths, derivative bounds, chart covers and
interpolation order. This model is a **candidate resolution law**, not yet
a verified sufficient grid rule for all hybrid fields in the implementation.
The refinement repair is necessary to make a parent-scale working-grid rule
plausible; it does not prove the rule.

Nevertheless, the following counting statement is unconditional **given**
that resolution law. It handles arbitrary sorted cadence, including large
gaps and clustered or repeated times.

Let the full time span be `T`, the number of exposures be `N`, and the
sample-count-balanced tree have `D <= ceil(log2 N)+1` levels. At any level the
segment time intervals have disjoint interiors, so

```
sum_segments_at_level h <= T,
sum_segments_at_level h^k <= T^k   for every integer k>=1.
```

Expand `G(h)=a_0+sum_(k>=1) a_k h^k`, with all coefficients nonnegative.
Summing over the tree gives

```
sum_nodes G(h_node)
    <= (2N-1) a_0 + D [G(T)-a_0].
```

Thus the modeled total is `O(N B^2 + G(T) log N)`. This does **not** require
physical spans to halve at every node. It repairs the earlier cadence gap
in the conditional cost argument.

If certified working grids and atlas sizes remain within a controlled factor
of these counts, and transfer order/precision costs are polylogarithmic,
the architecture would approach prepared-input-plus-output cost:

```
image preparation
+ (N B^2 + G(T) log N) * transfer/order/precision factors.
```

The output size `G(T)` may itself be enormous. It cannot be erased by calling
the output a compressed field. Affordable search of a more compact
representation would need its own coverage, query and complexity guarantees.
For a heterogeneous archive, `N B^2` is not automatically equal to the number
of recorded pixels: footprints, angular tiling and preparation must be charged.

## 5. A concrete frequency-aware cell rule to implement next

The complex inverse certificate suggests a computable local rule, rather than
choosing grid sizes by trial and error. This derivation is **not yet implemented
as the hierarchy's grid policy**.

Suppose a nonnegative matrix `E` bounds `|Dd|` on the complex enclosing domain
and a weighted norm proves its spectral radius is below one. Then

```
S = (I-E)^(-1) = I+E+E^2+...
```

is a componentwise nonnegative inverse bound. For any hybrid coordinates `z`,
the implicit equations imply `|dx| <= |dz|+E|dx|`, hence
`|dx/dz| <= S`. The normalized child map therefore satisfies

```
|d R_normalized / dz| <= (I+E) S = J_bound.
```

For a child angular Fourier field with frequency bounds `K_x,K_y`, define

```
L_i = K_x W_x J_bound[x,i] + K_y W_y J_bound[y,i].
```

Only coordinate `i` is complex in a one-dimensional interpolation estimate.
Since the map is real on the real axis, integrating its derivative along the
imaginary coordinate segment bounds the kernel modulus by
`exp(L_i |Im z_i|)`. A hybrid coordinate has normalized half-width at most the
padded child width `b_i`. With `n_i` cells and Bernstein parameter `r`, its
imaginary extent is `b_i (r-r^(-1))/(2 n_i)`. Thus a sufficient phase-based rule
is

```
n_i >= b_i L_i (r-r^(-1))/2,
```

which bounds the single-coordinate kernel modulus by `e`.

The local ellipse must also remain inside the certified hybrid neighborhood.
If that neighborhood was certified with parameter `r_0`, sufficient additional
conditions are

```
n_i >= [(r+r^(-1))/2 - 1] / [(r_0+r_0^(-1))/2 - 1],
n_i >= (r-r^(-1))/(r_0-r_0^(-1)).
```

For `r=r_0` these geometric lower bounds are just one. Round cell counts
upward to nested multiples of the child's cells, then select interpolation
orders from the bounded modulus and the full cascade/tree amplification.
All matrix inversion and rounding in this recipe need outward verification.

For a child containing multiple exposures, replace the child-angle rows by
uniform derivative bounds for the angular positions at *all* its exposure
times composed with the hybrid inverse. Such bounds can be obtained from the
same variational flow machinery; summing the data still uses the shared field.

This provides a constructive bridge from the present certificates to sufficient
grids. To obtain the desired scaling, the resulting bounds must preserve the
`h^0,h^1,h^2` structure instead of flattening it into one worst-case scalar.
The present single-cell order calculation does not establish that result.

## 6. What is solved here and what to construct next

Established by the new implementation:

- Whole-box derivatives of the declared exact reception flow.
- Real and complex domains on which all cascade coordinate inverses exist
  uniquely, including adaptive covers that retain the original state box.
- Automatic, justified child padding.
- A separable refinement repair for loss of resolution at intermediate stages.
- A computable sufficient interpolation order for angular Fourier transfers.
- A conditional total-node count valid at arbitrary cadence.

The next constructive tasks are:

1. Turn the `h^0,h^1,h^2` sensitivities into sufficient **local cell sizes for
   every refined hybrid field**, with computable constants. Frequency-aware
   local cells should replace the extremely conservative single-cell orders.
2. Drive the implemented complete hierarchy with these resolution rules and the certified atlases,
   charging atlas growth, every intermediate grid and all geometry work.
3. Add certified numerical evaluation of the Kepler/light-time maps. On
   regular branches, interval Newton and high-order evaluation are candidates;
   the current first-order tube integrator alone is not a proof of
   polylogarithmic high-precision cost.
4. Connect full pixel preparation and archive footprints with controlled
   WCS/PSF/noise error. A Fourier transfer certificate does not supply this.
5. Specify and implement the scientifically useful full transform inverse,
   separately from the coordinate inverses proved here.

These are a constructive research program and concrete implementations,
not an impossibility claim and not a completed FFT analogue.
