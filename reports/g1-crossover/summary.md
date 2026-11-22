# G1 crossover — partial same-host D and C3 evidence

## First partial session (archived)
Issue #442. Measured source `fe35940ad57d7d6a7f2d5aa80315001b84f16af9`; pinned SAE `4129001fffcb336f5203e4b0d6765d7a0ea3150c`. Protocol/commands in runtime.json; raw downloaded artifact preserved inside results.json with its external size/SHA-256.
Observed NVIDIA A100-SXM4-40GB (40960 MiB), driver 580.82.07, torch 2.14.1+cu130, CUDA 13.0; CPU calls used all 12 threads. Local preflight PASS. H100 allocation failed (25.7823 s); A100 transport PASS (102.6364 s), scientific task incomplete.

| k | m | TorchOED CPU | TorchOED CUDA | PyDOE | objective / floor / peak GPU memory |
|---|---|---|---|---|---|
| 3, 5, 7 | 10³–10⁶ | NOT_CAPTURED | NOT_CAPTURED | NOT_RUN | NOT_RUN |

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
The original retry plan was superseded by the Architect 2026-11-21 00:19:38 UTC ruling; the authorized continuation is recorded below.

## Night 1 final session (Architect-authorized continuation)
Measured source `de10cf1277c3b14fd7ad12e407141889f9ec365c`; A100-SXM4-40GB, 40 GiB, driver 580.82.07, torch 2.14.1+cu130, CUDA 13.0; all 12 CPU threads. One timed call/cell after warm-up; float64, seed 0; synchronized CUDA stops, transfers included. Previous sessions above and their raw records are preserved.
Controller ended WORK_EXEC_FAILED (exec TIMEOUT); no final artifact was downloaded. This section uses only the last saved progress snapshot (file mtime 2026-11-21 03:19:09 UTC). The uncaptured tail until cleanup is NOT_CAPTURED, not an inferred failure or an unexecuted cell; NOT_CAPTURED means no result in this snapshot, without inferring whether the uncaptured tail attempted that cell. Raw snapshot bytes are preserved in results.json with external size/SHA-256.

| k | m | criterion | CPU seconds; objective; floor; peak GPU | CUDA seconds; objective; floor; peak GPU | fastest finished PyDOE (D only) |
|---|---|---|---|---|---|
| 3 | 1000 | D | 0.622947s; 20.009800061; 0.9722818445351031; 0 B | 1.74037s; 20.009800061; 0.9722818445351031; 60034560 B | fedorov ERROR, modified_fedorov ERROR, detmax ERROR |
| 5 | 1000 | D | 0.773627s; 52.3255678124; 0.9804275960797818; 0 B | 3.13702s; 52.3255678124; 0.9804275960797818; 134221312 B | fedorov ERROR, modified_fedorov ERROR, detmax ERROR |
| 5 | 1000 | A | 2.03314s; -2.51082312766; 0.9322530096215513; 0 B | ERROR | N/A |
| 5 | 1000 | I | 7.93212s; -0.330663993074; 0.9422175849514767; 0 B | ERROR | N/A |
| 7 | 1000 | D | 1.00402s; 102.240771662; 0.9712539566694275; 0 B | 1.72193s; 102.240771662; 0.9712539566694275; 190806528 B | fedorov ERROR, modified_fedorov ERROR, detmax ERROR |
| 3 | 10000 | D | 2.67166s; 21.2843521477; 0.9723462460858318; 0 B | 3.48706s; 21.2843521477; 0.9723462460858318; 393345024 B | fedorov ERROR, modified_fedorov ERROR, detmax ERROR |
| 5 | 10000 | D | 11.2823s; 56.8345659879; 0.9759982962346848; 0 B | 6.76831s; 56.8345659879; 0.9759982962346848; 582968320 B | fedorov TIMEOUT, modified_fedorov ERROR, detmax ERROR |
| 5 | 10000 | A | TIMEOUT | ERROR | N/A |
| 5 | 10000 | I | TIMEOUT | ERROR | N/A |
| 7 | 10000 | D | 6.21896s; 111.254981411; 0.9606112988011765; 0 B | 10.4344s; 111.254981411; 0.9606112988011765; 29826560 B | fedorov ERROR, modified_fedorov ERROR, detmax ERROR |
| 3 | 100000 | D | 143.181s; 21.7469493476; 0.973302014325836; 0 B | 29.1345s; 21.7469493476; 0.973302014325836; 566349824 B | fedorov TIMEOUT, modified_fedorov ERROR, detmax ERROR |
| 5 | 100000 | D | NOT_CAPTURED | NOT_CAPTURED | NOT_RUN: outside frozen PyDOE cells |
| 5 | 100000 | A | NOT_CAPTURED | NOT_CAPTURED | N/A |
| 5 | 100000 | I | NOT_CAPTURED | NOT_CAPTURED | N/A |
| 7 | 100000 | D | NOT_CAPTURED | NOT_CAPTURED | NOT_RUN: outside frozen PyDOE cells |
| 3 | 1000000 | D | NOT_CAPTURED | NOT_CAPTURED | NOT_RUN: outside frozen PyDOE cells |
| 5 | 1000000 | D | NOT_CAPTURED | NOT_CAPTURED | NOT_RUN: outside frozen PyDOE cells |
| 5 | 1000000 | A | NOT_CAPTURED | NOT_CAPTURED | N/A |
| 5 | 1000000 | I | NOT_CAPTURED | NOT_CAPTURED | N/A |
| 7 | 1000000 | D | NOT_CAPTURED | NOT_CAPTURED | NOT_RUN: outside frozen PyDOE cells |

| C3 nsim | CPU s | CUDA s | CPU/CUDA power vectors | nonconverged CPU/CUDA; separated CPU/CUDA | 4 combined MC SE |
|---|---|---|---|---|---|
| 10000 | 0.343351468 | 0.370266754 | [0.9928, 0.8816, 0.8747] / [0.9929, 0.8842000000000001, 0.8781] | 0/0; 72/71 | [True, True, True] |
| 100000 | 5.30053746 | 0.421676493 | [0.99324, 0.88468, 0.87845] / [0.9928100000000001, 0.88314, 0.8780100000000001] | 0/0; 676/719 | [True, True, True] |
Largest m reached, cpu: 100000.
Largest m reached, cuda: 100000.
Largest m reached, fedorov: none.
Largest m reached, modified_fedorov: none.
Largest m reached, detmax: none.
GPU search wins only in these measured pairs: D k=5 m=10000 (1.67x); D k=3 m=100000 (4.91x). C3 nsim=100000 CUDA/CPU timing is shown above; single-call timings are not a general speed guarantee.
Local checks: 21 PASS, 0 FAIL (saved search objectives at 1e-9 relative and C3 CPU powers/counts exact). Search pair and C3 statistical agreement is recorded separately, as-is, in results.json.
CUDA A/I failed on CPU/CUDA tensor-device mismatches inside pair-move scoring; PyDOE 1.6.0 cells failed because scipy.linalg.inv rejects assume_a, or reached the per-cell timeout. Cell timeouts remain recorded. No tuning, source workaround, extra allocation or re-timing; errors are not objective mismatches. Scientific DoD remains partial where methods fail. This session uses the accepted X.device workaround, so it does not independently test the #510 library fix.
All six allocation attempts have confirmed cleanup; final fresh usage 0 assignments, 133.84 CU; cumulative displayed spend 15.35 CU (25 CU cap). CI and final-SHA handoff linked on PR #509. Next: Architect disposition; no further night-1 allocation.

## Night 2 (2026-11-22 CET): allocation blocked
Driver `f97ed86b5eb46656cb32b7eb098a5187063d89ad`; library source pinned to `d869c1a394e03219eca40256c9c78f1f552bdc96` but not measured. Local PyDOE 1.6.0 / SciPy 1.18.1 import and `inspect.signature(scipy.linalg.inv)` confirm assume_a support. Driver compilation PASS.
Preflight PASS; H100 ALLOCATION_FAILED (23.9305 s), then A100 ALLOCATION_FAILED (53.8765 s). No session created and no science cells run or re-timed. New objective rescoring, CPU/CUDA agreement and #510 GPU confirmation NOT_RUN; historical evidence above is unchanged.
Both attempts: cleanup PASS, assignments 0; final fresh readback 133.84 CU, night-2 spend 0.00 CU; cumulative spend 15.35/25 CU from original 149.19 CU, 9.65 CU remain.
Work timeout 4500 s projects 7.88 CU from the observed night-1 burn rate; internal 3600 s budget leaves 900 s for final artifact transfer. Host monitor stops at cumulative 24.5 CU.
Next: final night 3, 2026-11-23 00:30-06:30 CET, preferably at or after 02:00, same frozen plan. No further night-2 allocation.
