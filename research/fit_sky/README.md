# FitSky: a full-objective research attempt

**The full FitSky objective is not achieved.** The authoritative rubric remains
the posterior over stars, galaxies and solar-system bodies across every
instrument and time. This campaign implements several mathematical routes,
tests them against that objective, and preserves their failures. A functioning
bounded model is not being presented as the full solution.

The [contract](OBJECTIVE.md) was committed before implementation. The
[numerical campaign](CAMPAIGN.md) fixed the broad priors and accuracy targets
before measurements. The work is isolated from ongoing Kepler development:
the existing orbital modules and archived exposures are read-only dependencies.

## Implemented routes

1. **Joint interaction graphs:** finite source-existence inference with all
   source-source Gaussian cross terms, exact algebraic sum-product elimination,
   and explicit bounds for omitted interactions. Computation uses float64.
2. **Continuous catalogue integration:** unknown source counts/classes and
   continuous parameters, proper exchangeable priors, Poisson count-tail bounds,
   shared instrument nuisance, and replayable Arb evidence/query enclosures.
3. **Analytic and higher-order integration:** shared linear Gaussian nuisance
   elimination, analytic integration of one uniform flux, fourth-derivative
   tensor Gauss enclosures, and a portfolio that intersects valid bounds.
4. **Global statistical integration:** a proposal covering every represented
   continuous cell, validated pointwise likelihood envelopes, fixed-budget
   simultaneous confidence bounds, and an alternative that samples the entire
   unbounded Poisson count prior.
5. **Physical and observational tests:** pixel-integrated point/extended sources,
   motion, parallax, variability, Gaussian/Poisson noise, and Kepler states;
   public ZTF exposures with shared backgrounds and a held-out predictive
   likelihood calculation.

The [derivation](derivation.md) distinguishes established identities,
deterministic enclosures, statistical confidence, numerical assumptions and
physical-model omissions. No new mathematical theorem is claimed.

## What the experiments establish

The scalar independent control has posterior source probability
**0.607926040537572**. The analytic-flux route encloses it to floating report
precision. Plain interval integration needs 13 leaves at the declared 1%
evidence target; Gauss and analytic-flux routes need two. This is a correctness
control, not a broad astronomical result.

The finite graph route matches independent direct-pixel enumeration to roughly
2.5e-17 in the tested marginals. It handles an 80-source chain, implicitly
summing 2^80 finite catalogue alternatives, with small factors. A dense
24-source case exceeds the 65,536-entry factor budget and fails explicitly.
This demonstrates interaction-width dependence, not unrestricted sky inference.

The confused-source experiment retains x in [-1.2,1.2], flux in [1,8], and
unknown count 0/1/2. Independent high-order numerical integration gives
approximately:

| Catalogue count | Posterior probability |
|---|---:|
| 0 | 0.007385 |
| 1 | 0.533072 |
| 2 | 0.459543 |

Although the simulated truth contains two sources, the data and prior leave
substantial ambiguity. Returning that ambiguity is the desired behavior.
The one-million-draw statistical run encloses these reference probabilities
but narrowly misses the *relative evidence* target (gap 0.01212). The larger,
fresh-seed validation is retained in `confused_followup/`; its exact results are
in the accompanying machine-readable reports. With four million draws it
**meets both requested targets**: the relative evidence gap is **0.005878**,
and the 99% simultaneous model-conditional probability intervals are:

| Catalogue count | Posterior interval |
|---|---:|
| 0 | [0.007360, 0.007444] |
| 1 | [0.530785, 0.535210] |
| 2 | [0.457416, 0.461825] |

This is a completed bounded continuous inference problem with unknown source
count. Its numerical/confidence assumptions and finite 0/1/2 count prior remain
explicit; it does not complete the all-sky objective.

## Follow-up: two earlier inference failures resolved

The [fixed follow-up](NOISE_FLOOR_CAMPAIGN.md) introduced a stronger pointwise
likelihood ceiling. A convex template-mixture fit proposes a dual vector;
Arb independently bounds its response over the full continuous source domain.
This captures unavoidable correlated image residuals that the old bounds lost.
Each original parameter cell is retained. The optimizer and its grid do not
define a candidate shortlist or exclude unvisited source positions.

Fresh, fixed eight-million-draw validations met the original 1% evidence and
posterior targets on **both previously failed cases**:

| Original scene | Relative evidence gap | Selected 99% posterior interval |
|---|---:|---|
| Moving vs stationary with shared astrometric offset | 0.007331 | Moving: [0.977965, 0.978842] |
| Two real ZTF exposures | 0.008578 | Moving: [0.043673, 0.048649] |

Confidence is simultaneous within each report, conditional on the stated
numerical and physical model assumptions. Independent tensor quadrature with
analytic flux integration agrees: moving-class probabilities are approximately
0.978426587 and 0.046669968, respectively. The real-data posterior mostly favors
the empty catalogue under this particular source/background prior. It is not a
validated claim that the astronomical region contains no objects.

`noise_floor_results/` preserves the executed sources, every support
certificate, complete original-cover hashes, and fixed-budget results.
`noise_floor_diagnostics/` retains exploratory runs, independent quadrature,
and the broad-scene failure. The new methods are exposed in the CLI through
`--noise-floor both`; `--marginalize-flux` adds analytic flux averaging to the
statistical route.

These results resolve two concrete numerical failures, not the full objective.
They do not supply general instrument responses, all-survey ingestion,
astronomical-scale catalogue inference, or a successful broad mixed-scene fit.

## Original failures against the full objective

| Fixed experiment | Deterministic evidence relative gap | Outcome |
|---|---:|---|
| Confused source count/positions/fluxes | 7.82 | Unresolved within 60 seconds |
| Moving versus stationary, shared astrometric offset | 1.06e23 | Unresolved within 60 seconds |
| Broad mixed scene, count uncertainty and shared calibration | 1.82e52 | Unresolved within 60 seconds |
| Two real ZTF exposures | 2.65e35 | Unresolved within 90 seconds |

These gaps are upper-bound/conditioning failures, not measurements of physical
uncertainty or evidence that the sources cannot be inferred. A huge gap must not
be interpreted as an impossibility theorem.

The statistical route is substantially more useful on the confused-source
case, but its one-million-draw moving, broad and real cases retain [0,1]
probability bounds for the main events. Numerical point estimates can look
persuasive while the missing-mass/error bound remains vacuous. The output labels
those point estimates explicitly. Sampling the complete Poisson prior removes
count truncation, but does not resolve the variance and likelihood-envelope
problem.

The second real-data statistical run integrates known constant-likelihood
contributions, including the empty catalogue, analytically. It still fails the
accuracy target. This correction prevents a rare proposal cell from silently
disappearing from a *point estimate*; it does not manufacture a tight confidence
bound.

The held-out third ZTF image has a model-conditional log predictive density
enclosure of approximately **[-366.30, -147.96]**, far too wide to establish a
useful prediction. It uses a consistent shared/per-exposure Gaussian background
prior through a joint-evidence ratio. The held-out image was not used to set
the source prior.

Instrument support is also incomplete: Gaussian PSFs, affine WCS, simple
variability/galaxy models and specified noise are implemented; general
instrument responses, trailing, arbitrary morphology/light curves, empirical
artifacts, all-survey ingestion, and astronomical-scale joint inference are not.
The vectorized statistical route does not yet propagate Kepler states.

## What changed during the attempt

The pilot is retained in `pilot_results/`, with its executed source snapshot.
Its range/Gauss/analytic-flux attempts failed the confused-source target.
A constant-likelihood derivative error interrupted the next case.
Independent tests and subsequent investigation repaired:

- Constant functions in the derivative evaluator.
- A split-priority bug caused by taking the logarithm of a zero error.
- Excessive interval dependency in the analytic flux formula.
- Certificate aggregation from serialized outward bounds.
- Rejection-sampling envelopes: an integral bound is not a pointwise bound.
- Exact Gaussian normalization and shared-nuisance accounting.
- Rare constant-cell contributions to statistical point estimates.

The repaired campaign kept the original domains. It did not silently shrink
the source prior or remove ambiguous catalogues to obtain a passing result.

## Files and reproduction

Install research dependencies and run the focused checks:

```sh
pip install '.[test,research]'
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_fit_sky.py
magictools-fit-sky examples/fit-sky.json /tmp/new-fit-sky --verify
```

The [interface guide](../../docs/fit-sky.md) documents model inputs, outputs,
queries and optional posterior catalogue draws.

```sh
OPENBLAS_NUM_THREADS=1 python -m research.fit_sky.experiments /tmp/new-fit-sky-campaign --methods auto
OPENBLAS_NUM_THREADS=1 python -m research.fit_sky.statistical_experiments \
  /tmp/new-fit-sky-campaign /tmp/new-fit-sky-statistical
OPENBLAS_NUM_THREADS=1 python -m research.fit_sky.predictive_experiment \
  /tmp/new-fit-sky-campaign/real.json /tmp/new-fit-sky-predictive
```

`results/` holds the deterministic campaign and independent references.
`statistical_results/`, `confused_followup/`, and `real_followup/` retain the
separate fixed-budget statistical runs. `predictive_results/` holds the
three-exposure joint fit and held-out calculation. Source snapshots identify
the executed code; `validation.json` records final tests and wheel execution.
The original Kepler FITS files are referenced by content hashes without edits
or redundant copies.

The initial repository suite passed **340 tests**, with one skipped. After
incorporating the concurrent Kepler update from main (`f102865`), the combined
suite passed **367 tests**, with the same optional test skipped. Every Kepler
file matches that upstream commit; validated FitSky source hashes are unchanged.
The wheel
was installed outside the repository, and its actual command produced a replayed
certificate, posterior catalogue draws, and a statistical report. Every
installed FitSky source file was checked against the workspace source.

Soft wall-clock budgets and a shared development host affect leaf counts and
timings. Some validation jobs overlapped campaign runs. The tables are
measurements of these runs, not hardware-independent speed claims.
Confidence is 99% simultaneously within each statistical report under its
stated numerical/model assumptions, not 99% jointly across every experiment.

## Remaining mathematical problem

The important failure is **not lack of a posterior formula**. The joint
likelihood, source interactions, exchangeability and nuisance integrals can
already be specified. The unsolved part is constructing a sufficiently tight,
affordable representation of the posterior mass across broad, ambiguous source
families and shared instrument parameters.

Locality works for fixed sparse graphs. Analytic marginalization helps selected
directions. Continuous certificates become expensive or loose. Global sampling
can give accurate-looking numbers without a useful guarantee when its likelihood
ceiling is too conservative. These experiments do not establish a method that
closes all those gaps at once.

The full completion contract remains unmet.
