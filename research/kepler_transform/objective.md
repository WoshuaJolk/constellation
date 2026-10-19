# Full-objective research contract

Date: 2026-10-19. This is a research attempt, not a new production release.
The user explicitly prefers failed attempts at the full objective to declaring
a restricted implementation a breakthrough.

The [catalogue](https://magictools-theta.vercel.app/) asks for summed likelihood
over every Keplerian trajectory through exposures with astrometry and times,
with substantially less work than separately evaluating the trajectories. It
also asks for a transform and inverse that the discovery engine can call.

## What would count as success

1. Handle the full six-dimensional, noncollision two-body state space over an
   explicitly bounded observational domain, including elliptic, parabolic and
   hyperbolic states. Bounds on distance/time/resolution are necessary to make
   numerical accuracy meaningful; they must not quietly remove difficult
   orbit classes. Observer motion and light travel time are part of the target.
2. Produce a representation of the likelihood over that domain, with a stated
   coverage/error guarantee. A list of preselected orbits is not the solution.
   Writing N distinct output values itself costs at least O(N); a compressed
   representation is allowed but its evaluation and error must be specified.
3. Beat a competent direct image-likelihood evaluation in scaling, accounting
   for geometry, preprocessing, construction, memory and query costs. Repeated
   application alone does not excuse an O(NT) orbit-by-exposure construction.
4. Validate against independent dynamics and direct likelihood calculations,
   including adversarial cases. Numerical identities below use a 1e-8 relative
   error target; unmodeled angular errors are compared with a one-arcsecond
   Gaussian PSF. These are test settings, not permission to waive physical
   error control.
5. Establish what an inverse means and whether the data determine it. An
   adjoint is not an inverse. If information is lost, identify the nullspace
   and distinguish a generalized inverse from exact reconstruction.
6. Establish a substantive difference from existing methods. A literature
   search is necessary but cannot certify novelty by itself. Independent
   specialist scrutiny would be needed before a breakthrough claim.

All six remain the acceptance criteria. Conditional successes are intermediate
results. A failed experiment must not be relabeled as the full solution.

## Three approaches to test

**A. Global low-rank compression.** Factor the whitened orbit-template operator
so one image projection scores the entire domain. Test the necessary low-rank
condition on both tightly neighboring and separated physical orbits. Reject
as a uniform route if the rank needed to preserve arbitrary inputs grows with
the number of independently resolvable trajectories. This says nothing against
full-rank fast transforms such as the FFT.

**B. Hierarchical phase factorization.** Write the spatially Fourier-transformed
image likelihood with phase exp(i k dot x(t;state)). Test complementary spatial
and frequency blocks, as required by butterfly methods, across increasing
frequency, orbital spread, cadence and close observer approaches. Build a
checkable factorization experiment, include construction work, and do not infer
a uniform theorem from a few small block ranks.

**C. Orbital invariants plus rotational harmonics.** Separate three scalar
radial invariants from the three orientation angles of a Kepler orbit. Derive
and implement the all-orientation harmonic transform rather than enumerating
orientations. Test nonuniform anomaly sampling, all conic classes, the adjoint,
moving-observer geometry and light-time coupling. Continue past a central-
observer demonstration; failure of the extension is a failure of this route
to the full objective, not a completed tool.

## Initial prior-art leads

- Whidden et al., [Fast algorithms for slow moving asteroids](https://arxiv.org/abs/1901.02492): GPU maximum-likelihood trajectory search.
- Napier et al., [heliostack](https://arxiv.org/abs/2604.16731): nonlinear image stacking using radial invariants and sky reprojection; already demonstrates real-data discoveries.
- Hu et al., [A fast butterfly algorithm for generalized Radon transforms](https://www.ahay.org/RSF/book/tccs/butterfly/paper_html/paper.html): curved-path transforms and complementary low-rank structure.
- Wandelt and Gorski, [Fast Convolution on the Sphere](https://arxiv.org/abs/astro-ph/0008227): harmonic acceleration over rotations.

These precedents rule out claiming novelty merely for nonlinear stacking,
shared radial dynamics, low-rank blocks, or rotational FFTs.
