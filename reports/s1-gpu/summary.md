# S1: GPU measurement

**measurement only; no speedup or GPU-support claim.**

**d_optimal_design end-to-end; k2_exchange on CUDA, start draws on CPU.** CPU float64 F and CPU generator; ten restarts, one thread, default search parameters. The timed capture boundary transfers F and IDs to CUDA on each restart and returns IDs to CPU; synchronization precedes timer stop. CPU oracle recomputation is outside the timed region.

Source `0e5d0ea87cb5eda8f7454f6bf665b1df1cbf9539`, base `2ff9ec8`, Issue #75. One untimed seed0 warm-up: 3.669648s. Environment: Python3.13.16, torch2.11.0+cu130, CUDA13.0, driver580.82.07. Observed **NVIDIA RTX PRO 6000 Blackwell Server Edition**, 97887MiB. Fixed candidate seed0, random2000 p10 n20; measured seeds0/1/2.

| Seed | GPU end-to-end s | Existing CPU G2 s | Peak GPU bytes | Objective / CPU difference | Floor / CPU error | Certificate | Same CPU ID set |
|---|---|---|---|---|---|---|---|
| 0 | 2.805329 | 8.046611 | 12339712 | 20.495607897444 / 0 | 0.843561018467 / 3.33e-16 | LOCAL_2_NUMERICAL | True |
| 1 | 2.761294 | 7.454587 | 12339712 | 20.495607897444 / -3.55e-15 | 0.843561018467 / 1.11e-16 | LOCAL_2_NUMERICAL | True |
| 2 | 3.079624 | 10.242654 | 12339712 | 20.495607897444 / -3.55e-15 | 0.843561018467 / 4.44e-16 | LOCAL_2_NUMERICAL | True |

All in-run objective/floor oracle checks pass at 1e-9. Floors are independently recomputed on CPU float64 with bench_b0.score_design from each returned ID set. IDs need not match the CPU path; here all three sets do. CPU comparison times reuse reports/beta/g2.json (source823749a), from a different machine/session; no CPU G2 rerun or generalized acceleration claim.

Command: `python -B tools/bench_g2.py s1-gpu cuda`, via the pinned SAE Colab controller and committed s1_gpu_driver.py. One amended G4 allocation, no fallback, new spending or plan upgrade; controller pin5fb26304e4c85183aa215e6ad4db9ced928cab12. Cleanup confirmed: stop PASS and authenticated assignments0 before/0 after. Downloaded artifact SHA256 and size (external bytes only) are in results.json.

First attempt was BLOCKED at the d_optimal_design device boundary (source98dd81f); this measurement follows the Architect amendment, moving only k2_exchange across CUDA and keeping restart starts/feasibility on CPU. No src change.
