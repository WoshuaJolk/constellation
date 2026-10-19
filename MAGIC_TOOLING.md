# Magic tools (2026-10-19)

These tools require breakthroughs, but if possible, may radically change astronomical work. For humans and agents.

Each entry says what it ships with, what already exists, and the gap between them.

Feasibility: 10 = buildable now, 5 = possible with time or cooperation, 1 = probably impossible.
Impact: 10 = changes the field outright, 5 = settles a major open question, 1 = trivial.
Tags: software, data (ships a large precomputed product), math (needs an algorithm or proof that does not yet exist),
theory (a physical model with stated parameters).

Glossary. Tracklet: two or more detections of one moving object on one night. Arc: the time span of all detections of an
object. Orbit family: every orbit consistent with an arc within its errors. Corridor: the sky region the family sweeps over
a date range. ITF: the Minor Planet Center's file of tracklets never linked to any object. Injection depth: the magnitude
at which synthetic sources planted in a frame are recovered half the time. Shift-and-stack: adding frames shifted along
a candidate motion so a faint mover sums while the sky does not. Pinned orbit: one whose every opposition is predicted by
the others to under an arcsecond.

- `map_cover` (feasibility 5, impact 6) [software, data]
  In: an orbit or orbit family, a date range. Out: every public exposure that covers any member, with chip and gap
  geometry, the fraction of the family on-chip, and the expected S/N of the object in that exposure.
  Ships: a footprint and depth index of every exposure in every public survey archive, computed once with per-exposure
  seeing, PSF and injection depth.
  Exists: CADC SSOIS returns the exposures covering a single orbit, with no family, no chip geometry and no depth.
  Why: every non-detection becomes a quantified depth, and no search again misses a frame or trusts a header limit.

- `force_track` (6, 7) [software, data]
  In: an orbit family, a frame list from `map_cover`. Out: forced photometry at the prediction in every frame, including
  difference images below any survey's alert threshold, with S/N, PSF width and stamps.
  Ships: per-instrument reduction recipes, Gaia-tied WCS for every frame, template libraries, and the access routes to
  each archive's pixels.
  Exists: ZTF and ATLAS forced-photometry services take a fixed sky position, one survey each. Nothing takes an orbit, and nothing crosses surveys.
  Why: the archive answers "is it here?" for any hypothesis in minutes, which is the entire archival search loop.

- `weigh_stamp` (feasibility 5, 6) [software]
  In: a stamp, the frame's PSF and noise model, a candidate motion. Out: the likelihood ratio of "moving point source"
  against each named artefact model (star residual, cosmic-ray worm, galaxy blend, chip edge, spurious stack peak).
  Ships: forward models of every artefact class per instrument, validated on injections. No training set.
  Exists: real-bogus classifiers in ZTF and Pan-STARRS are trained networks; the Tractor forward-models static sources. No forward-model artefact likelihood for moving sources.
  Why: a candidate is accepted or rejected by physics, and the same stamp gets the same verdict from anyone.

- `judge_line` (feasibility 6, 6) [software]
  In: a candidate detection and the object's existing arc. Out: a dossier: reversed-rate stack, static check in other
  epochs, jackknife, chance-coincidence density, PSF width against nearby stars, within-night motion, refit with every
  earlier line kept, and the combined false-positive probability at stated thresholds.
  Ships: the thresholds, calibrated per instrument on injections, and the scripts that produce each test.
  Exists: each survey pipeline runs some of these checks internally. No standalone, instrument-calibrated battery that takes an arbitrary candidate.
  Why: the acceptance battery that now takes a day of hand work and a human arbiter runs in one call and is reproducible.

- `learn_null` (feasibility 6, 7) [software]
  In: an arc or a hypothesis, the depths from `map_cover`, the results from `force_track`. Out: the posterior over
  orbits, shrunk by every non-detection, and the single exposure that would shrink it most.
  Ships: a ranging engine over distance and range rate with full n-body propagation, and the depth integration.
  Exists: Brown and Batygin update the Planet Nine posterior by hand, one survey at a time; OpenOrb and find_orb do ranging. Nothing closes the loop with measured depths for an arbitrary object.
  Why: a search stops being "look where we can" and becomes "look where the remaining probability is", and a null is
  worth exactly its depth.

- `link_world` (feasibility 5, impact 7) [software, data]
  In: a set of tracklets, or nothing. Out: every self-consistent orbit among all tracklets ever recorded, bound and
  hyperbolic, 1 AU to the Oort cloud, including sub-threshold forced detections, each with a chance-link probability
  against a reversed-night control, kept current as new tracklets arrive.
  Ships: every ITF tracklet, every broker alert and every forced detection, indexed in heliocentric coordinates, plus the
  linker with its validation.
  Exists: HelioLinC, run once on the ITF, produced orbits for over 200,000 bound minor planets. No hyperbolic branch, no
  sub-threshold input, no continuous operation, no chance-link control.
  Why: the Isolated Tracklet File stops existing as a concept, and an interstellar object is linked the night its second
  tracklet lands.

- `dig_floor` (feasibility 4, impact 8) [software, data, math]
  In: a sky region, a distance range, a date range. Out: every object below the single-frame detection limit whose summed
  likelihood along some orbit across every exposure from every instrument exceeds a stated threshold, with injections
  proving the depth.
  Ships: the joint-likelihood engine over the whole archive, with per-instrument PSF and noise models so frames from
  different telescopes and years sum coherently, and, if it is found, `fold_kepler`.
  Exists: KBMOD does GPU digital tracking within one survey's stare, DEEP reached r 26 with it, and multi-year stacking
  within one survey has been done. Nothing sums across instruments and archives.
  Why: an order of magnitude smaller at 40 AU from data already taken, and Planet Nine found or excluded in the sky
  regions where no single survey goes deep enough.

- `map_complete` (feasibility 5, impact 8) [software, data]
  In: orbital elements, size, epoch. Out: the probability that such an object would already have been discovered by the
  union of all surveys.
  Ships: one survey simulator with the real pointings, depths and detection efficiencies of every survey, bound and
  hyperbolic populations alike.
  Exists: the OSSOS survey simulator does this for OSSOS and CFEPS; the Brown and Batygin synthetic population does it
  for Planet Nine in ZTF, DES and Pan-STARRS separately. No simulator spans all surveys or all populations.
  Why: most debates about the structure of the Kuiper belt are debates about this function, held without it.

- `inject_all` (feasibility 5, impact 6) [software, data]
  In: a hypothesised population. Out: the detection record it would have produced in the real archive, against the actual
  one, as a likelihood.
  Ships: `map_complete` applied frame by frame.
  Exists: done by hand for Planet Nine, one survey at a time, ruling out 78 percent of its parameter space. Not done for
  any other population, and not as a tool.
  Why: population parameters, Planet Nine priors and interstellar flux become inference against the whole history of
  observation rather than one survey's completeness.

- `fuse_astrom` (feasibility 5, impact 7) [software, data]
  In: a position, or nothing. Out: proper motion and parallax for every source to the 24th magnitude, from every imaging
  survey over 30-year baselines, with a published selection function.
  Ships: the fused catalogue itself.
  Exists: GPS1 (Gaia, Pan-STARRS, SDSS) and the DESI-SDSS catalogue fuse two or three surveys each over their own
  footprints; Rubin will fuse only its own epochs. No all-sky, all-survey product.
  Why: Gaia four magnitudes deeper. Every runaway and hypervelocity star, nearby brown dwarf and free-floating planet
  falls out of the catalogue instead of a targeted search.

- `find_ejector` (6, 6) [software, theory]
  In: `fuse_astrom`, radial velocities where they exist. Out: every point where the tracebacks of several fast stars
  converge in space and time, with what is visible there.
  Ships: the tracebacks for every fast star, precomputed.
  Exists: tracebacks for individual runaways since Hoogerwerf et al. 2001, and Gaia hypervelocity samples. No all-sky convergence search over every fast star.
  Why: a convergence with no visible source is a candidate intermediate-mass black hole or dark remnant.

- `map_glow` (3, 6) [theory, software]
  In: deep ecliptic frames. Out: the excess angular power after shifting at the rate of each distance shell, and the
  sub-threshold population mass it implies.
  Ships: the shift-and-power pipeline with its flat-field systematics model.
  Exists: nothing. Intensity mapping is a cosmology technique and has never been applied to the Kuiper belt.
  Why: the Kuiper belt below every survey's limit, measured without detecting one object.

- `weigh_tno` (2, 5) [theory, data]
  In: all TNO orbits with covariances. Out: masses for the largest bodies from their perturbations on neighbours.
  Ships: the perturbation inversion over decades of astrometry.
  Exists: asteroid masses from mutual encounters in the main belt. Never attempted beyond Neptune.
  Why: densities for dwarf planets from astrometry alone.

- `map_sedna` (6, 6) [theory, software]
  In: a formation model and its parameters. Out: where on the sky, in which rate band and how bright inner Oort cloud
  bodies should be near perihelion, and how many each survey should already contain.
  Ships: the formation models as runnable code with their parameters exposed, and the pointing and depth history of every survey to turn a sky density into an expected count.
  Exists: inner Oort cloud formation models (Brasser et al., Kaib and Quinn) and OSSOS-specific predictions in papers. No per-survey, per-rate-band prior as a product.
  Why: turns "is there another Sedna" into a count the archive either matches or does not.

- `read_belt` (5, 6) [theory, software]
  In: the resonant population census. Out: the migration history of Neptune that reproduces it.
  Ships: the migration simulations as a parameterised model, and the inverse solver that fits migration rate and smoothness to a census.
  Exists: Nesvorny's migration simulations compared with OSSOS by hand. No inverse solver.
  Why: the Kuiper belt as a recording of how the giant planets moved.

- `read_oort` (3, 5) [theory, data]
  In: long-period comet orbits, Gaia tracebacks of nearby stars. Out: the Sun's stellar-encounter history.
  Ships: long-period comet orbits with original aphelion directions corrected for planetary perturbations, Gaia tracebacks of every star within 10 pc of the Sun's path, and the matcher.
  Exists: Gaia-based stellar encounter lists (Bailer-Jones) and comet-shower models exist separately. The two records have never been matched.
  Why: which stars perturbed the Oort cloud and when.

- `twin_system` (3, 7) [theory, software]
  In: formation model parameters. Out: a synthetic solar system whose census matches the real one within survey errors.
  Ships: the formation simulation as a parameterised model, coupled to `map_complete` so its synthetic census is observed the way the real one was.
  Exists: Nesvorny et al. models reproduce parts of the census. No end-to-end match within survey errors.
  Why: population priors become predictions, and every search points where the model says the missing bodies are.

- `run_hunt` (feasibility 5, 7) [software]
  In: a target (an arc, a hypothesis, a region). Out: the full discovery loop run to completion by fixed rules:
  `map_cover`, `learn_null`, `force_track`, `weigh_stamp`, `judge_line`, refit, repeat, stopping only for a
  human verdict on a line.
  Ships: the rules and thresholds, composed from the tools above.
  Exists: nothing packaged. Every step is run by hand, one target at a time.
  Why: the three-week moonshot in a day, on hundreds of targets at once.

- `fold_kepler` (feasibility 2, impact 8) [math]
  In: a stack of exposures with astrometric solutions and times. Out: the summed likelihood over every Keplerian
  trajectory through the stack at once, at a cost well below one evaluation per trajectory.
  Ships: the transform and its inverse, as a library `dig_floor` can call.
  Exists: brute force over rates and angles (KBMOD), and the Radon and Hough transforms, which do this for straight lines only.
  Why: all-orbit stacking of the whole archive becomes affordable, and finding the transform would be a bigger result
  than any object it finds.

- `match_all` (feasibility 3, impact 7) [math]
  In: every detection ever made, with errors. Out: the globally optimal set of orbits linking them, with proven bounds on
  completeness and on the false-link rate.
  Ships: the algorithm and its proof, as a library `link_world` can call.
  Exists: HelioLinC links in O(N log N) by clustering at a reference epoch, with no guarantee of completeness or uniqueness.
  Why: linking stops being a heuristic that needs a reversed-night control and becomes a solved problem.

- `prove_floor` (feasibility 3, impact 7) [math]
  In: a set of exposures with PSFs, noise models and times, and a source model. Out: the information-theoretic detection
  limit for a moving source across that set, and the estimator that attains it.
  Ships: the theorem and the estimator.
  Exists: per-frame five-sigma limits and per-survey injection depths. No joint bound across exposures, instruments or years.
  Why: says exactly what is findable in the existing archive before anyone searches, and settles whether Planet Nine is
  already excluded by data in hand.

- `fit_sky` (feasibility 2, impact 10) [math, software]
  In: every photon ever recorded, with its time and instrument. Out: the posterior over one catalogue of everything:
  every star with parallax, motion and variability, every galaxy, every solar system body with an orbit.
  Ships: the generative model of every instrument and the inference engine that fits it jointly.
  Exists: Celeste fit a static generative model to SDSS pixels at petascale. No time, no motion, one survey.
  Why: nothing is detected; everything is inferred, and every other tool here becomes a query against the posterior.

- `invert_chaos` (feasibility 1, impact 8) [math, theory]
  In: the present orbits of everything, with covariances. Out: the posterior over the solar system's dynamical history,
  including bodies ejected and stars that passed.
  Ships: a method for inverse problems in chaotic systems, which does not exist.
  Exists: forward simulations only (Nesvorny, Kaib). Chaos at Lyapunov timescales erases the history that forward runs
  would need to match.
  Why: the solar system's past becomes a measurement rather than a family of plausible stories.
