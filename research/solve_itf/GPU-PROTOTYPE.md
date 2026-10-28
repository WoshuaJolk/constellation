# GPU linker prototype: stage 1

This branch adds an experimental CUDA translation of two-body propagation,
retarded topocentric residuals, and their analytic six-state Jacobians. It is
**not yet a GPU optimizer or a full GPU linker**. The production pipeline and
scientific acceptance rules are unchanged.

Historical timing motivates this target: the 10,000-seed CPU run recorded
110.379 seconds in triple fitting out of 184.004 seconds overall, while scanning
took 12.754 seconds. This is not a new profile of the 7M campaign. Its completed
cascade receipt records 12,710.543 summed worker-batch seconds, not wall time;
see `seven-million-production-receipt.json`.

## Implementation

`magictools.itf.gpu` recompiles the existing CPU universal-variable propagator
and analytic derivative Python bodies as CUDA device functions. It uses FP64,
no fast math, no finite-difference propagation bank, one thread per
state/observation pair, and a resident input/output bank. Elliptic, parabolic,
hyperbolic and retrograde geometries use the same equations. Arrays are ragged
by candidate, so variable observation counts do not require padded fake data.

The light-time and tangent-plane residual wrapper mirrors `residual_jacobian`.
Invalid observations invalidate their whole candidate, as on CPU; the raw
point-validity mask remains available. Invalid, unresolved and uncomputed work
must never become accepted chains. A simulator is allowed only explicitly for
tests and is forbidden by the timing harness. No CUDA device causes an explicit
block, never a disguised CPU fallback.

The finite-state kernel has **not been compiled or timed on actual GPU hardware**.
CUDA simulator success checks logic; it does not prove hardware compilation,
rounding parity, memory behavior, occupancy, or performance. Planetary final
fitting remains on the existing CPU path.

## Evidence available on 2026-10-28

- 20 focused new/existing tests passed with the explicit CUDA simulator,
  covering all-conic derivatives, cascade behavior and curved/sparse cases.
- A frozen real-data bank contains the first 1,000 existing report candidates,
  first mode per candidate, and 13,498 observations. Selection is deterministic;
  no later replacement or outcome-based sampling. These are already retained
  candidates, **not a representative bank of all attempted search starts**.
- Simulator replay of the first 16 candidates / 176 observations matches CPU
  residuals and Jacobians exactly. No speed claim follows.
- CPU stage pass: 1,000/1,000 numerically evaluated, no unresolved; 0.091413
  seconds on one thread, 0.274013 seconds on four threads in this small bank.
  Warm-up was separately 0.357083 / 0.243082 seconds. These are single
  measurements, not stable throughput estimates. Thread overhead dominates the
  small workload; do not use the slower threaded result to inflate a GPU gain.
- Hardware attempt stopped with `No CUDA device`. Runtime inspection found no
  injected DigitalOcean credentials; the old protected helper directory is
  absent. No Droplet was started and no additional cloud compute was spent.

Small JSON receipts are in `gpu-prototype-receipts/`. The real input bank and
per-case outputs remain in `/tmp/itf-gpu-prototype-results` on the preparation
workspace. The input bank SHA256 is
`1c37a51fe051decfa9a401f4e70185e8b15c11e25d7e2a27976076afe1917f72`.

### Subsequent provider attempt

DigitalOcean access was subsequently restored. The API successfully provisioned
an H100 at $4.41/hour, then one serial retry with explicit firewall allowances
and a more tolerant package-update bootstrap. Both appeared active at the API
but neither exposed a reachable HTTPS worker service. Proxy probes reported an
upstream connection failure; bootstrap logs could not be retrieved, so the root
cause remains unresolved. No payload or science job reached either machine.

Both Droplets and their temporary SSH keys were deleted. A fresh provider check
at 2026-10-28 05:10 UTC returned zero Droplets. Estimated pre-tax compute for the
two setup attempts was **$1.23**, against the self-imposed $5 ceiling. The original
45-minute aggregate deadline was not reset. There was no third paid retry.
See `gpu-prototype-receipts/provider-attempt.json`. A larger frozen bank of
10,000 real candidates / 121,813 observations is also prepared locally, but no
GPU timing or hardware-parity result exists for it. Working remote transport
must be established before another paid GPU attempt.

## Reproduce

Install the existing ITF/test extras and a CUDA toolkit/driver compatible with
the installed Numba CUDA backend on a real NVIDIA worker. Simulator checks:

```bash
NUMBA_ENABLE_CUDASIM=1 python -m pytest tests/test_itf_gpu.py
```

Prepare an identical input bank using existing files (no catalogue download):

```bash
python research/solve_itf/gpu_benchmark.py prepare \
  --report runs/solve-itf-eod-20261023/inputs/0.json \
  --cache .cache/solve_itf/iteration5-cold-cache \
  --observatories .cache/solve_itf/ObsCodes.html \
  --ephemeris .cache/solve_itf/ephemerides/de440s.bsp \
  --limit 1000 --output /tmp/itf-gpu-bank.npz

python research/solve_itf/gpu_benchmark.py run \
  --bank /tmp/itf-gpu-bank.npz --backend both --cpu-threads 4 \
  --seconds 30 --batch-cases 128 --output /tmp/itf-gpu-comparison.json
```

The harness refuses simulator timings. Hardware parity checks **every row**,
requires matching failure masks, residual disagreement <=0.001 arcsec and
Jacobian disagreement <=1e-6 scaled by max(1, abs(CPU entry)). Failure blocks the
GPU timing result. The preliminary benchmark evaluates a fixed bank once, with
the same order and batch size, recording unique work, unresolved and uncomputed
cases. It does not count repeated scoring as new search.

Warm-up and initial upload/verification are reported separately. The timed GPU
pass includes per-chunk input transfer, synchronization, host output and
per-candidate accounting. Timed CPU passes include their corresponding work.
The deadline is checked between batches; one in-flight batch can overrun, and
overshoot is reported. This is a cooperative phase budget, not a hard cloud
billing guard. The small bank completes before 30 seconds on CPU and cannot
establish equal-time search gains.

## Next gates for the requested CPU-versus-GPU search comparison

1. DigitalOcean provisioning is now verified, but remote worker transport is
   blocked. Diagnose that connection before another paid GPU attempt; recheck
   current region stock and price when ready, with a bounded budget and teardown.
2. Compile on the selected GPU and run full-bank parity plus pathological and
   invalid cases. Measure transfer, compilation, kernel, and output costs; sweep
   a few prospectively declared batch sizes without changing scientific cuts.
3. Integrate the kernel into batched optimizer iterations: maintain states and
   Jacobians on device, preserve convergence/failure semantics, compact active
   candidates. Compare the optimized CPU implementation, including its better
   threading setting, against GPU on the **same starts and fit policy**.
4. Freeze a representative production queue including failed/deferred starts,
   distant, retrograde, NEO and hyperbolic modes. Run identical-work and
   identical-time full-search comparisons. Report distinct memberships, arcs,
   nights, development-label recovery, the same verification outcomes, wall
   time and actual incremental cost. Do not use additional low-quality fits as
   evidence of scientific improvement.

The hardware and full-search gates remain open. This prototype creates no
calibrated probabilities, new acceptance threshold, or confirmed objects.
