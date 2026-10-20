# Conditional-flux follow-up

The original four 30-exposure runs are retained unchanged. A separately tested
conditional Gaussian flux update now integrates the other sources and shared
covariance into a truncated-normal draw. This is a mixing improvement, not a
posterior accuracy certificate.

Before inspecting follow-up outputs, fix:

- The same 30 exposures, priors, truth, and queries as SEQUENTIAL_CAMPAIGN.md.
- Four runs with seeds 20261120, 20261121, 20261122, 20261123.
- 2,048 particles, 20 rejuvenation sweeps, conditional ESS 0.8.
- Gaussian conditional flux updates enabled; all other proposal settings unchanged.
- Retain all 3/10/30-prefix outputs, errors and execution snapshots.

The executable uses `--pilot --gibbs-flux` to distinguish these from the original
campaign, whose settings must not be relabeled after execution. All four runs
are retained regardless of agreement or disagreement. No error target is
passed using between-population scatter alone.

Additionally, run the original complete-prior bounded estimator on the
10- and 30-exposure inputs using 1,000,000 draws, confidence 0.99, analytic
one-flux averaging, seeds 20261130 and 20261131 respectively. These separate
reports provide the existing global error accounting, including unbounded
Poisson counts. Their confidence is per report, not joint across the campaign.
