# Issue #18 V3.4 multi-GPU rerun

Base: `2830c114014a44f67f09e7def025828b30ae1c8a`. Executed source: `3a3cb24ddbaf153ea540714f099bad2686c1dedb`.
SAE controller pin: `5fb26304e4c85183aa215e6ad4db9ced928cab12`. Refs #18; PR #26.

All four successful runs used Python 3.13.16, torch 2.11.0+cu130, CUDA 13.0, driver 580.82.07; CPU threads=4, TF32 off. Exact Python builds are in the results files.

| Request / observed device | G1 | G2 dense oracle | G3-C | G4 gradients | G5 timing | Max criterion / diagnostic GiB | Cleanup assignments |
|---|---|---|---|---|---|---|---|
| T4 / Tesla T4 | PASS, 10 | PASS, 20 | PASS 88; 272 CONSTRUCTION_BLOCKED | PASS, 20 | 40 medians | 4.911 / 8.585 | confirmed, 0 |
| L4 / NVIDIA L4 | PASS, 10 | PASS, 20 | PASS 88; 272 CONSTRUCTION_BLOCKED | PASS, 20 | 40 medians | 4.911 / 8.585 | confirmed, 0 |
| A100 / NVIDIA A100-SXM4-40GB | PASS, 10 | PASS, 20 | PASS 88; 272 CONSTRUCTION_BLOCKED | PASS, 20 | 40 medians | 4.911 / 8.585 | confirmed, 0 |
| G4 / NVIDIA RTX PRO 6000 Blackwell Server Edition | PASS, 10 | PASS, 20 | PASS 88; 272 CONSTRUCTION_BLOCKED | PASS, 20 | 40 medians | 4.911 / 8.585 | confirmed, 0 |
| H100 / not observed | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | n/a | confirmed, 0 |

H100: **BLOCKED**, one allocation attempt, controller `ALLOCATION_FAILED`, `new` exit 1. The pinned controller omits raw provider stderr, so a more specific provider reason is unavailable. No retry or H100 execution is claimed. Readback: `usage PASS -> new FAIL(exit 1) -> stop PASS -> usage PASS(assignments=0)`, run `m2b-v34-h100-20261030`.

## Method and correctness

Grid: p={3,10,30,100}, B={1,16,256,4096,16384}, fixed seed `182000+p*10000+B`. All devices received identical CPU-generated inputs. The GO caps the batch at 16,384; the earlier 70% memory target is not claimed reached. Peaks are measured around the fp64 criterion forward and the separate gradient diagnostic, including resident inputs. No diagnostics exceed 16,384 matrices.

G1 runs the existing 10-test c-optimality suite on CUDA. G2 compares every fp64 score with independent CPU dense `torch.linalg.solve`, including B=1 and B=16,384; tolerance 1e-9. Across devices the maximum relative error is 9.499e-16. G4 compares CUDA autograd with dense-solve analytic gradients, tolerance 1e-8; maximum error 2.606e-15.

G3-C uses exact Fraction Cayley rotations, strict rational geometric spectra, and exact q/kappa_c for its gates: 11/45 grid points, 88 matrices, three contrasts each, CPU/CUDA fp64/fp32. The remaining 272 matrices are CONSTRUCTION_BLOCKED, never approximated. No rejection below the A-bound and no B-bound exceedance occurred. Maximum err/(eps*kappa_c)=2.0; gate counts, per-group max/p99 and grouped disagreement seeds are in the result JSONs. Reported rational-derived numbers are converted to floats only for serialization, after exact gate comparisons.

G5 records synchronized medians of 5 after 3 warmups, CPU, GPU resident, and GPU including H2D/D2H transfer. Input generation is excluded. Transfer starts without a preloaded duplicate input. Timings describe only these measured configurations; no speedup or general support claim.

## Measured break-even intervals (including transfer)

Each cell is the first measured (B_prev, B] with GPU including transfer below CPU. The union contains only this fresh authoritative run per device; r2 partial results are discarded.

| p / dtype | T4 | L4 | A100 | G4 |
|---|---|---|---|---|
| 3 / float64 | (256, 4096] | (256, 4096] | (256, 4096] | (256, 4096] |
| 3 / float32 | (256, 4096] | (256, 4096] | (256, 4096] | (256, 4096] |
| 10 / float64 | (256, 4096] | (256, 4096] | (256, 4096] | (256, 4096] |
| 10 / float32 | (16, 256] | (256, 4096] | (256, 4096] | (256, 4096] |
| 30 / float64 | (16, 256] | (16, 256] | (16, 256] | (16, 256] |
| 30 / float32 | (16, 256] | (16, 256] | (16, 256] | (16, 256] |
| 100 / float64 | (1, 16] | (1, 16] | (1, 16] | (1, 16] |
| 100 / float32 | (1, 16] | (1, 16] | (1, 16] | (1, 16] |

## Execution and downloaded artifacts

For each device in T4 -> L4 -> A100 -> G4 -> H100 order: `wsl -e env TMPDIR=<owned-D-temp> PYTHONDONTWRITEBYTECODE=1 python3 <pinned>/colab_control.py --repo-path <exact-source-snapshot> --source-sha 3a3cb24ddbaf153ea540714f099bad2686c1dedb --cli /home/zhangru/.local/bin/colab --temp-root <owned-D-temp> --task-seconds 4800 --allocation-seconds 4200 --cleanup-seconds 180 --kernel-timeout 60 --work-timeout 3600 --run-id m2b-v34-<device>-20261030 --out <owned-evidence>/<DEVICE> --device <DEVICE> --driver tools/m2b_gpu_driver.py --artifacts tools/m2b_artifacts.json`.

Every device started after authenticated zero readback; every attempt ended with successful stop plus authenticated assignments=0. Each results JSON embeds the cleanup readback and observed identity. Downloaded external artifact sizes/SHA-256 below refer to the original remote `m2b-gpu.json`; committed result JSONs retain cited G2/G4/G5 data and G3-C gate/statistic summaries with grouped disagreement seeds, replace pytest output with its final result line, and add transport metadata. Redundant per-contrast G3-C rows are omitted.

- T4: 623012 bytes; SHA-256 `681a0783133da1189abca6c7291d185a86a9760abc99c336dfe2d64e1ea24f8c`.
- L4: 623037 bytes; SHA-256 `80598937f230040450ff3b84426d319d95c517975fbb006ba75525da1ec3ec12`.
- A100: 623067 bytes; SHA-256 `27011c5d5184dca6ad62e80dea6588efd182cea9cb93ff0575a447c24fdd1f72`.
- G4: 622979 bytes; SHA-256 `86c9ce5f5582c9fe2a645a29d93fa50557dfbd5b3edc5c5aecbb2a022a75eecb`.

r2 - leaked session had already been released; dangling local controller/exec remained - confirmed named session absent and authenticated assignments=0, interrupted only its task-owned local processes, discarded partial results, reran T4 from scratch.

PASS: four device suites and cleanup; H100 allocation BLOCKED with cleanup PASS. NOT_RUN: CI, full unrelated suite, support certification. src unchanged. Evidence: five files total (this summary plus four result JSONs). Owned D-drive checkout/snapshot and local controller download receipts remain for review; no remote sessions remain. No self-merge.
