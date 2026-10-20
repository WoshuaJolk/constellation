# Follow-up: continuous template support bounds

This follows the original failed campaigns without changing their data,
source-count priors, coordinate ranges, instrument nuisance, or accuracy target.
The full completion contract is unchanged.

The exploratory runs used one million draws: a global noise-floor proposal
with seed 20261023, then conditional cell floors with seed 20261024. Those
results motivated a separate fixed-budget validation below; they are not
silently discarded or called preregistered validation.

## Fixed validation

- Cases: the original moving-versus-stationary scene and the original two
  real ZTF exposures.
- Reuse the full partitions in `results/moving-auto.json` and `results/real.json`.
- Global floors: 128 support leaves, tolerance 0.0001, grid 9, 1,000 optimizer
  iterations. Conditional floors: 8 support leaves, tolerance 0.00001, grid 3,
  100 optimizer iterations.
- Replay every support certificate. No optimizer termination flag is a proof.
- Exactly 8,000,000 independent validation draws per case, seed 20261025.
- 99% simultaneous confidence within each report, not across the campaign.
- Required widths: <= 0.01 absolute posterior probability and <= 0.01 relative
  evidence gap. Keep the original motion-direction event in the moving case.
- Preserve failures, original prior coverage, construction/replay work,
  source snapshots, and the numerical assumptions of the statistical route.

The mathematical change is a pointwise likelihood ceiling, not a new likelihood
or prior. The conditional Gaussian inference model and its known physical
limitations remain the same. A passing real-image calculation validates
integration under that model; it does not establish that the model fits nature.
