# G1 crossover — BLOCKED / partial C3 CPU evidence
Issue #442. Measured source `fe35940ad57d7d6a7f2d5aa80315001b84f16af9`; pinned SAE `4129001fffcb336f5203e4b0d6765d7a0ea3150c`. Protocol/commands in runtime.json; raw downloaded artifact preserved inside results.json with its external size/SHA-256.
Observed NVIDIA A100-SXM4-40GB (40960 MiB), driver 580.82.07, torch 2.14.1+cu130, CUDA 13.0; CPU calls used all 12 threads. Local preflight PASS. H100 allocation failed (25.7823 s); A100 transport PASS (102.6364 s), scientific task incomplete.

| k | m | TorchOED CPU | TorchOED CUDA | PyDOE | objective / floor / peak GPU memory |
|---|---|---|---|---|---|
| 3, 5, 7 | 10³–10⁶ | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

Largest m reached: CPU / CUDA / PyDOE — none (no saved search designs). No measured cell establishes where GPU wins.

| C3 binomial nsim | CPU seconds | CPU powers (intercept, a, b) | nonconverged / separated | CUDA |
|---|---|---|---|---|
| 10000 | 0.431767538999992 | 0.9928, 0.8816, 0.8747 | 0 / 72 | ERROR |
| 100000 | 3.8576299780000056 | 0.99324, 0.88468, 0.87845 | 0 / 676 | ERROR |

C3 design, coefficients and alpha are the committed RD1b skpr.json attributes. Float64, generator seed 0 per call; CPU powers and both fit counts reproduce exactly locally (2 PASS, 0 FAIL). CPU/CUDA power agreement NOT_RUN; CUDA warmup and both timed calls failed validation, with full verbatim tracebacks in results.json.
The downloaded artifact ends before the first D row. Local reproduction identifies `dict(status='PASS', **child)` raising duplicate-keyword TypeError when successful bench_crossover output already includes status. This is an inferred remote stop cause; the remote parent traceback was not retained. Transport PASS is not science PASS.
Final driver removes the duplicate keyword and uses X.device explicitly for the GLM generator; these repairs were not GPU-tested. The measured source above predates them. No second successful session or rerun of completed CPU timings. No src/tests changes, tuning or speedup claim.
Required search objective rescoring and CPU/CUDA objective agreement NOT_RUN (zero saved search designs). C3 CPU exact reproduction PASS; C3 CPU/CUDA statistical agreement NOT_RUN. CI is linked in the PR handoff.
Cleanup PASS on all four attempts: stop/readback exit 0, assignments 0 → 0; final fresh usage 0 assignments, balance 149.19 CU. Displayed spend 0.00 CU (provider precision), below 25 CU. No L4/T4 fallback.
Routine repair attempt at source `5872652ff19dbd22defc598fed5d364c883d3b63`: preflight PASS, H100 ALLOCATION_FAILED (25.3255 s), A100 ALLOCATION_FAILED (57.1151 s). No second session was created; cleanup confirmed for both. Final driver is 56 lines.
Next: retry only in the next permitted night window (2026-11-22 00:30 Europe/Zurich, night 2 of at most 3), or Architect disposition. No further attempt tonight.
