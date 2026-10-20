# Fixed numerical campaign

These experiments test the contract; they do not replace its full scope.
The target is 0.01 absolute query width and 0.01 relative evidence gap.
All synthetic noise uses seed 20261020. Bounds, failures and original domains
are retained. Independent quadrature/reference values are diagnostics, not
certificates. Time limits are checked between completed cells.

1. Scalar unknown-source Gaussian control: flux [1,3], counts 0/1 with equal
   prior. Compare range bounds, fourth-order Gauss bounds, and analytic-flux
   integration against independent SciPy integration.
2. Confused source counts: five pixels, PSF sigma 0.7 pixels, Gaussian noise
   sigma 0.7. Two true sources at x=-0.4,+0.4, flux 4 each. Inference positions
   [-1.2,1.2], fluxes [1,8], counts 0/1/2 with weights .2/.4/.4. Budget 2,048
   leaves/60 seconds per method; keep the full position domain.
3. Moving versus stationary: three 3x3 exposures at -1,0,1 days, Gaussian
   noise 0.6, PSF 0.65 pixels. True x=0.15, vx=0.35 pixels/day, flux 8.
   Both classes have x in [-1,1], flux [2,12]; mover vx in [-0.6,0.6].
   Counts 0/1, equal class weights. The third exposure has a shared camera
   offset with prior [-0.3,0.3]. Budget 2,048 leaves/60 seconds. Include the
   query vx in [0,0.6].
4. Broad mixed scene: three 3x3 exposures, x/y in [-2,2], flux [1,12],
   motion in [-1,1] pixels/day, parallax [0,0.5], stellar variability
   [-0.2,0.2]/day, galaxy sizes [0.3,1.5]. Three classes, Poisson count mean 2
   truncated computationally at 3 with retained tail bound. Shared camera
   offset [-0.3,0.3], gain exp([-0.1,0.1]). Budget 2,048 leaves/60 seconds.
   A failure is not repaired by shrinking domains or conditioning on truth.
5. Finite-graph controls: independent exact enumeration at 8 sources; sparse
   chains up to 80 sources; increasing dense conflicts until the declared
   factor-table budget fails. Keep omitted-interaction error bounds separate
   from floating-point arithmetic.
6. Real ZTF data: reuse (read-only) the three archived 9x9 cutouts from the
   Kepler campaign. The first two are inference inputs; the third is held out.
   Fit a proper shared Gaussian background prior centered at 170 counts with
   standard deviation 30, plus per-exposure background deviations of 10.
   These are declared demonstration priors, not calibrated population priors.
   Compare stationary and moving source models with x/y [-2,2], flux [0,100]
   reference counts, rates [-30,30] pixels/day, count 0/1. Budget 512 leaves/
   90 seconds. Retain the real mask, variance, flux scale and PSF width.
   Use the archived WCS tangent linearization; report instrument-model omissions.
   No new-object or survey-completeness claim is an acceptance condition.

Tests separately cover Poisson counts, correlated noise, shared nuisance,
multinomial source classes, source-label symmetry, fixed Kepler physics against
an independent ODE, source-count tails and tampered/unfinished certificates.

## Follow-up routes after the first campaign

The first campaign is retained in `pilot_results`, including the constant-jet
failure that interrupted it. The same domains are rerun with a portfolio of
range/Gauss/analytic-flux enclosures, improved cancellation handling, and a fixed
split-priority bug. Its accuracy target and domain widths are unchanged.

A separate fixed-budget statistical campaign uses 1,000,000 draws per case,
seed 20261021, and 99% simultaneous confidence *within each case*. It does not
claim 99% joint confidence across the campaign. Every deterministic cell gets
positive proposal mass using a recomputed pointwise likelihood envelope.
For the broad scene, draw directly from the unbounded Poisson prior instead;
its source count is not truncated at the deterministic campaign's cutoff.
Record envelope construction, sampling work, failures and numerical assumptions.
Do not stop sampling when favorable confidence intervals first appear.

The one-million-draw confused-source run missed the 1% relative evidence target
(gap 0.01212). A follow-up uses a fresh seed, 20261022, and a fixed four million
draws on the unchanged confused-source domain. Constant-likelihood cells,
including the empty catalogue, are now integrated analytically rather than
relying on their possibly tiny proposal probabilities. Repeat the real-data
statistical calculation at one million draws with this accounting improvement.
These follow-ups have separate per-run confidence budgets; the first results
remain retained.
