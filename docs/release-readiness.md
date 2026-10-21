# Constellation 0.7.0 release acceptance

The main README promotes **16 tools**, each with an executable CLI acceptance
case and a stated baseline. Promotion means useful for the bounded task described
here. It does not mean mathematical novelty, universal accuracy, discovery
confirmation, or complete public-archive coverage.

The [machine-readable results](release-validation.json) record the measured
values and a hash of the Python sources and acceptance inputs. Reproduce them:

```sh
python -m pip install -e '.[test]'
CONSTELLATION_RELEASE_REPORT=release-artifacts/validation.json \
  python -m pytest tests/test_release_cli.py -q
```

The release workflow builds an sdist and wheel, checks metadata with Twine,
installs the wheel in a clean virtual environment, checks its dependencies, and
runs these tests through the installed `constellation` executable from temporary
directories outside the checkout. The workflow uploads distributions, JSON
measurements, and JUnit results. A missing acceptance result for a promoted tool
fails the release check. No package is automatically published to a registry.

## Demonstrated value

Values below are from deterministic controlled cases. Numerical timings are not
used as acceptance thresholds. Image tests generate truth independently with
Gaussian CDFs or midpoint trail integration; orbital propagation is compared to
an analytic solution. Other independent checks include analytic probabilities,
a known combinatorial optimum, and intentionally corrupted artifacts.

| Tool | Baseline | Measured improvement | Interpretation limit |
| --- | --- | --- | --- |
| `find_frames` | Concatenate archive responses; interpret an unavailable service as zero rows | Two matching rows become one native frame with both provenance records; HTTP 503 remains an explicit failed provider | Local HTTP ObsCore controls; service uptime and the completeness of 49 public indexes are not certified |
| `check_frames` | Trust saved responses and query success | Verifies two response checksums, then detects deliberate corruption | Integrity and coverage evidence, not proof that an object was detectable |
| `kepler-propagate` | Constant-velocity extrapolation for 100 days | Circular-orbit position error below 1e-10 AU versus 0.164 AU for straight-line motion | Two-body model; no planetary perturbation or universal astrometric-accuracy claim |
| `map_cover` | Treat every on-chip position as usable | Rejects all three on-chip predictions in a fully masked exposure; exports no invented non-detections | Supplied orbit samples, WCS, masks, and observer model |
| `force_track` | Single central-pixel flux estimate, even with the true background known | Mean flux uncertainty 9.21 versus 19.64 units, with recovery consistent with injected flux | Correct supplied PSF and independent Gaussian variance model |
| `dig_floor` | A 5-sigma threshold per exposure | True track ranks first at joint S/N 6.55; the strongest individual exposure is only 3.19 | Finite supplied bank, common flux model, stated Gaussian trial correction |
| `inject_tracks` | Infer recovery from nominal sensitivity | Zero recovery at zero injected flux and full recovery at bright flux across the declared placements; eight source images unchanged | Controlled placements; overlapping placements do not imply independent calibration trials |
| `judge_line` | Accept a high joint score | Flags a single-visit impostor with joint S/N 15.35 when leave-one-out support disappears | Diagnostic flags, not a calibrated real/bogus probability |
| `weigh_stamp` | Equal integrated flux for a trail and cosmic ray | Prefers the correct moving-source and spike models for the two independent pixel controls | Conditional on the tested model family and priors |
| `learn_null` | Keep uniform weights and choose the first follow-up | Updates weights to 0.1/0.9 and selects 0.325 nats of expected information instead of zero | Supplied detection efficiencies and discrete hypothesis family must be justified |
| `map_complete` | Assume two surveys detect independently | Correct shared-exposure union 0.5 instead of 0.75 | Exact finite model with conditionally independent physical exposures |
| `inject_all` | Exact enumeration refuses the 24-exposure experiment | Estimate 0.7046 with interval [0.6965, 0.7127], covering analytic probability 0.7080 | Monte Carlo error only; no inferred calibration or population uncertainty |
| `match_all` | Greedily take the highest-scoring candidate | Compatible assignment score 12 instead of 10, with a certificate | Supplied candidates and integer utilities; no candidate-generation completeness |
| `verify_match` | Trust the assignment report | Accepts the valid certificate and rejects an altered objective | Verification of the finite supplied optimization problem |
| `prove_floor` | Use a known-template oracle power bound | Tighter mixture power upper bound 0.0392 versus 0.0924, matching an analytic calculation | Prior-average finite Gaussian-family bound, not a universal survey limit |
| `ejector_fit` | Impose one launch time on two distinct launches | Recovers 1.3 and 3.1 Myr with near-zero measurement cost; one shared age has at least 0.9 Myr maximum error | Declared encounter hypotheses, covariance, frame, and dynamics; no ejection-discovery probability |

The broader existing test suite also checks malformed inputs, negative flux,
PSF/mask cases, covariance, selection rules, resume behavior, raw archive format
fixtures, and research controls. Published historical Orion inputs and frozen
JPL Horizons vectors have independent comparisons in the existing tests. Those
checks supplement these release cases; they do not widen their scientific scope.

## Simpler command contracts

- Most tools use `constellation TOOL INPUT.json [-o OUTPUT.json]`.
- `find_frames INPUT.json [-o DIRECTORY]` creates a unique run under `runs/`
  when `-o` is omitted. A resume must name the saved directory explicitly.
- `kepler-propagate INPUT.json [-o DIRECTORY]` is a direct shortcut; it returns
  the artifact location in its JSON envelope. `run kepler-propagate` still works.
- `check_frames DIRECTORY --checksums` checks a saved search. Exit code 4 means
  an integrity failure; search exit code 3 means incomplete checks, including a
  budget-limited fast pass.
- `list --ready` shows the promoted set. `doctor` checks its core dependencies;
  `doctor --all` also reports optional research dependencies. A plain installation
  includes schema validation; optional research packages are not required by the
  promoted tools. Invalid inputs and existing output paths are rejected.

## Not promoted

These tools remain callable and retain their tests and documentation. Being below
means they are experimental, specialist, or superseded interfaces **outside this
release's acceptance set**; it is not a claim that their implementations produce
no useful results. Moving them into the main README requires an applicable CLI
baseline case and closure of the stated practical gap.

| Tool | Promotion gap |
| --- | --- |
| `fold_kepler` | Repeated-bank benchmarks can be faster, but one CLI invocation pays compilation cost and can lose to direct evaluation. Persistent reusable plans and a measured representative CLI crossover are needed. |
| `fold-chart` | Research transform; broad numerical/scaling guarantees and a general archive-pixel workflow remain open. |
| `fit_sky` | Interfaces are unified, but sequential mode coverage and broad-scene accuracy remain experimental. Bounded and sequential engines retain different contracts. |
| `find_ejector` | Specialist population traceback needs justified posterior inputs and calibrated search controls; no general-purpose release acceptance case is claimed. |
| `ejector_prepare` | Specialist Gaia preparation relies on explicit distance and missing-RV priors; promotion needs a bounded, independently checked ingestion-to-fit workflow. |
| `ejector_search` | Existing bounded catalogue validation is useful, but a practical general-purpose release acceptance workflow remains to be specified. |
| `ejector_bound` | Fixed-domain interval certificates remain specialist, require optional arithmetic dependencies, and are outside the current release cases. |
| `ztf_search` | Superseded for general discovery by `find_frames`; retained for the older ZTF preparation format. |
| `ztf_fetch` | Useful older ZTF preparation path, but its estimated local noise model and separate input contract are not a general validated archive-to-pixel workflow. |

## Distribution

Version 0.7.0 is built as a Python wheel and source distribution. The npm project
is the informational website and remains private. Registry publication is a
separate action; installing an unrelated package named `constellation` is not
an installation of this project. The GitHub install command in the README is
the supported source-install path until a registry release exists.
