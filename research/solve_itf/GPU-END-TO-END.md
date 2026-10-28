# Whole-linker measurement on the original workload — 2026-10-28

The current integrated GPU path completed the original fast-profile workload in **87.28 seconds with fresh application caches**, compared with **87.80 seconds for four CPU fitting processes on the same host**. This is effectively a tie in a single-pass benchmark, not a demonstrated end-to-end GPU advantage. With the catalogue and compiler disk caches reused, a new GPU process took **38.75 seconds**. No paired warm-cache CPU run was made.

The historical October 21 CPU run took **184.00 seconds**. The new cold GPU run is 2.11× faster than that historical measurement, but the current CPU control is also 2.10× faster. Hardware, software and CPU-side implementation changes cannot be credited to the GPU. The earlier rough 25× historical comparison combined fitting rates from different workloads; it is not an end-to-end result.

## Measured work and timing

Every run read the original input snapshot (SHA-256 `f3bb8bd234c1e9438f165f6bcaae1d893a4d6f852ed14bb3426ed634e037b624`), scanned all 2,548,307 tracklets, fitted 10,000 triple seeds, performed the configured chain extensions, and wrote the full candidate reports. Input download is excluded, as in the historical benchmark. Installation, transfer and tests are included in the cloud cost.

| Stage | Historical CPU | Current CPU, cold | Current GPU, cold | Current GPU, cached |
| --- | ---: | ---: | ---: | ---: |
| Catalogue preparation/startup inside CLI | 45.43 s | 32.30 s | 32.58 s | 0.25 s |
| Candidate scan | 12.75 s | 7.06 s | 7.21 s | 6.97 s |
| Seed generation | 2.79 s | 0.88 s | 0.90 s | 0.77 s |
| Observer preparation | 6.96 s | 5.85 s | 5.84 s | 5.86 s |
| Triple fitting | 110.38 s | 37.91 s | 35.67 s | 20.58 s |
| Extension | 3.66 s | 1.77 s | 1.44 s | 1.40 s |
| **Whole application, including output/process overhead** | **184.00 s** | **87.80 s** | **87.28 s** | **38.75 s** |

Unitemized overhead means stage rows do not sum exactly to the whole-process measurement. The new wrapper also records seed hashes and batch receipts within its timed subprocess. All three runs completed their bounded search, with 10,282 fits attempted, zero numerical failures/interrupted fits, 256 successful extensions and 4,343 retained candidate rows. These are candidate fits, not confirmed objects.

The current fast-profile defaults were used: four scan/CPU-fit workers, 10,000 seeds, 14-day proposal gaps, 0.3″ assumed errors, 0.6″ RMS limit, Gauss plus 24 range/radial hypotheses and 80 solver iterations, with the existing growth and numerical checks. The 280-second application cap was unchanged and no run exhausted it. Current CPU and GPU runs used the same ordered seed hash. The historical source was not reconstructed from its hashes; its report and candidate-text checksum were verified and its actual outputs compared separately.

## Which GPU implementation this measures

This measures the existing compact GPU adapter injected into the full search's **128-chain synchronous batches**. CPU verification and exact-seed fallback remain active; no cuts or starts were removed. Production defaults remain CPU. The full search does **not yet incorporate** the separate sustained benchmark's 1,024-chain prefetch orchestration or prebuilt observation atlas. Consequently, the previously measured 2,258 chains/sec must not be substituted for this full-pipeline measurement. The workloads also differ: this ranked 10K queue retains roughly 43% of fits, while the sustained broad-queue benchmark retained far fewer, changing output-assembly work.

Cold means absent catalogue, Numba and CUDA disk-cache directories for each backend, not a flushed operating-system page cache. Every phase uses a fresh process. The cached GPU phase reuses disk caches but starts a new CUDA context; it is not a persistent resident-GPU run. The host was an H100 80GB with 20 vCPUs at $4.41/hour. CPU fitting used four processes, while GPU verification/output used the adapter's existing thread settings. This is not a comparison with a separately priced CPU-only machine. Package versions and hardware are in the receipts.

## Output checks and limitations

- Current CPU and both GPU runs retain the **same 4,343 exact catalogue-index memberships**. CPU/GPU maximum best-RMS difference is **1.86e-9″**; **24 chains differ in alternate-mode counts**. Complete orbital-branch equivalence is not established.
- Cold/cached GPU runs have identical best RMS values and mode counts on all 4,343 memberships.
- Against the actual historical outputs, **4,341 memberships are shared**; two historical chains are absent and two other chains are present. Those same changes occur in the new CPU control, so they are not specific to GPU execution. The exact differences and up to 0.06115″ best-RMS change are retained in `historical-membership-comparison.json`.
- An initial analysis mistakenly keyed chains by display names, which collapse 18 catalogue rows. It was corrected to exact catalogue indices with uniqueness checks. Both preliminary analyses and the reconciliation are preserved; no fitting was repeated or altered. Two focused regression tests cover this distinction.
- These are single passes in fixed CPU-cold → GPU-cold → GPU-cached order, not a statistical speed comparison or a seven-million-seed run. The 0.6% cold end-to-end difference is not evidence of a meaningful performance advantage.

## Cost and preservation

Estimated total worker compute was **$0.58120**, including provisioning, transfer, installation, 14 passing numerical preflight tests, all three runs and return checks. The worker was deleted after hash-verifying all returned outputs. A fresh provider check found zero droplets; no historical snapshots were deleted. The $2 / 25-minute cap was respected.

Code, design, hashes, hardware/tests, timings, exact comparisons, deletion and final provider receipts are under `gpu-e2e-receipts/`. Full returned reports are in `/tmp/itf-e2e-benchmark-20261028/e2e-results`; large duplicate candidate payloads are not committed. No production catalogue or site was updated.

To reproduce on a prepared CUDA host with the exact input files at `input/itf.txt.gz` and `input/ObsCodes.html`:

```sh
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python research/solve_itf/gpu_e2e_benchmark.py --output e2e-results
python research/solve_itf/gpu_e2e_summary.py e2e-results
```

The wrapper does not provision a machine or enforce a cloud billing cap; the measured run used a separate guarded worker. The next performance task is integrating the already-tested larger/prefetched batches into full-search orchestration, followed by another paired end-to-end measurement. No speedup is claimed for that unimplemented integration.
