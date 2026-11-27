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

## Final continuation attempt 1 (2026-11-22 CET): transport blocked
Scientist amendment 7076258204 permits any hour after #541/#524 and first available H100/A100/L4/T4. Driver `b34681c9acda08cf186f851873efd1e4b56ff85a` preserves source `d869c1a394e03219eca40256c9c78f1f552bdc96`, frozen cells and order, and raises the per-cell cap to 2700 s. Controller preflight and driver compilation PASS.
H100 and A100 ALLOCATION_FAILED; NVIDIA L4 allocated (23034 MiB, driver 580.82.07, torch 2.14.1+cu130, CUDA 13.0). Kernel/work exec returned success, then `colab download -s <session> /content/g1-crossover.json <output>/g1-crossover.json` failed (exit 1, DOWNLOAD_FAILED). No scientific artifact survived; remote output was discarded by the pinned controller. Cell completion is unknown. New local objective rescoring, CPU/CUDA comparison and #510 confirmation NOT_RUN; historical results remain unchanged. T4 NOT_RUN because L4 was the first allocated GPU.
Cleanup PASS on all three attempts; final fresh usage 0 assignments, 133.84 CU. Displayed new spend 0.00 CU; cumulative 15.35/25 CU; 9.65 CU remain. Work timeout 4500 s / internal budget 3600 s retains 900 s transfer margin, projected work spend 7.88 CU from night 1, host monitor stops at cumulative 24.5 CU.
What this tells us / next change to try: fallback allocation works; results transport failed. Architect recovery ruling is needed before retrying cells whose completion is unknown. Preserve a sanitized remote failure/traceback or persistent cell output in the next authorized attempt; no scientific or speed claim from this run.

## Final continuation attempt 2 (2026-11-22 CET): STARTED only
Recovery ruling 7078405964; driver `3a502232edb10ffe2e3ab4e9f77b98c3bb4ea6ea`; frozen library `d869c1a394e03219eca40256c9c78f1f552bdc96`, frozen cell order and 2700 s cap unchanged. H100, A100 and L4 allocation failed. Tesla T4 allocated (15360 MiB; driver 580.82.07; torch 2.14.1+cu130; CUDA 13.0).
Work exec returned 0; artifact download succeeded, containing only `{"status":"STARTED"}` (20 bytes). No completed cell; all three new scientific checks NOT_RUN. Sanitized log-tail exec returned 0 but output was a newline only. This readback cannot establish that no exception occurred: fd redirection may not capture notebook Python streams. Directory listing was conditional on download failure and was not needed. No same-session retry. Historical records remain unchanged; ruling 7078405964 supersedes attempt 1 completion uncertainty: none completed.
All four cleanup readbacks PASS; final 0 assignments, 133.84 CU, displayed spend 0.00 CU; cumulative 15.35/25 CU, 9.65 remain. Internal 3600 s budget / work timeout 4500 s / host 24.5 CU stop preserved.
What this tells us / next change to try: fallback T4 allocation and initial artifact transport work, but driver startup still fails before a completed cell. Use an explicit foreground Python subprocess wrapper that redirects stdout/stderr to the log and records its actual exit code, subject to Architect recovery ruling; no numerical or timing claim from attempt 2.


## Final continuation attempt 3 (2026-11-22): real subprocess wrapper
Ruling 7079058649; committed driver unchanged at e067247; frozen library d869c1a394e03219eca40256c9c78f1f552bdc96, cells/order/2700 s cap unchanged. Host kernel-globals preflight PASS: no NameError; real driver exit captured after a stubbed first network step.
Allocation/cleanup: H100: ALLOCATION_FAILED, A100: ALLOCATION_FAILED, L4: allocated.
Observed device: {"nvidia_smi": [{"name": "NVIDIA L4", "memory_total_mib": 23034, "driver_version": "580.82.07"}], "torch": {"status": "AVAILABLE", "version": "2.14.1+cu130", "cuda": "13.0", "name": "NVIDIA L4", "capability": [8, 9]}}.
Driver subprocess exit: 0; controller: None.
| k | m | method | seconds/status | logdet | returned floor |
|---|---|---|---|---|---|
| 3 | 1000 | fedorov | 8.79464928500002 | 19.99210978218451 |  |
| 3 | 1000 | modified_fedorov | 4.324637834000015 | 19.917102261222833 |  |
| 3 | 1000 | detmax | 7.198036080999998 | 19.99210978218451 |  |
| 5 | 1000 | fedorov | 7.572735434000009 | 49.62078438346606 |  |
| 5 | 1000 | modified_fedorov | 7.465130748000007 | 49.62078438346606 |  |
| 5 | 1000 | detmax | 7.724672249999998 | 49.62078438346606 |  |
| 7 | 1000 | fedorov | 19.37627909699998 | 95.77972507385135 |  |
| 7 | 1000 | modified_fedorov | 19.745858391000013 | 95.77972507385135 |  |
| 7 | 1000 | detmax | 19.663633465999965 | 95.77972507385135 |  |
| 3 | 10000 | fedorov | 180.43315084200003 | 21.27450044494405 |  |
| 3 | 10000 | modified_fedorov | 28.586297849999994 | 21.181697749824256 |  |
| 3 | 10000 | detmax | 138.136914331 | 21.274500444944053 |  |
| 5 | 10000 | fedorov | 1261.470988025 | 56.7516740899648 |  |
| 5 | 10000 | modified_fedorov | 171.18784787699997 | 56.48425485671348 |  |
| 5 | 10000 | detmax | 358.8465012060001 | 56.77632526255357 |  |
| 7 | 10000 | fedorov | 199.93528479700035 | 103.215613119547 |  |
| 7 | 10000 | modified_fedorov | 196.03564263999988 | 103.215613119547 |  |
| 7 | 10000 | detmax | 197.08180929599985 | 103.215613119547 |  |
| 5 | 100000 | cpu | TIMEOUT |  |  |
| 5 | 100000 | cuda | NOT_RUN |  |  |
| 7 | 100000 | cpu | NOT_RUN |  |  |
| 7 | 100000 | cuda | NOT_RUN |  |  |
| 3 | 1000000 | cuda | NOT_RUN |  |  |
| 3 | 1000000 | cpu | NOT_RUN |  |  |
Largest m reached, cpu: None.
Largest m reached, cuda: None.
Largest m reached, fedorov: 10000.
Largest m reached, modified_fedorov: 10000.
Largest m reached, detmax: 10000.
Local saved objective rescoring: 18 PASS, 0 FAIL at 1e-9 relative. Same-host CPU/CUDA agreement: "NOT_RUN".
C3 CUDA GLM (warmup excluded): [{"power": [0.9929, 0.8842000000000001, 0.8781], "nonconverged": 0, "separated": 71, "seconds": 0.4147639530000049, "threads": 12, "peak_gpu_bytes": 133902336, "status": "PASS", "k": "glm", "m": 10000, "method": "cuda", "criterion": "binomial", "warmup": false}]. Historical CPU statistical comparison: [{"reference": "Historical accepted A100-host CPU nsim=10000, seed 0; not a same-host timing comparison", "CPU": [0.9928, 0.8816, 0.8747], "CUDA": [0.9929, 0.8842000000000001, 0.8781], "MATCH": [true, true, true]}]. No cross-host timing ratio.
GPU wins only in measured same-host pairs: []. One GPU per timing table; no A100/L4/T4 timing conflation.
All 3 cleanup readbacks PASS; final 0 assignments, 131.95 CU; cumulative 17.24/25 CU, remaining 7.76. No same-session retry.
What this tells us / next change to try: the real subprocess wrapper reaches measured cells and confirms the prior kernel __file__ launch diagnosis. Missing/timed-out/error cells remain partial evidence. If the long reference cells consume the remaining budget before CPU/CUDA pairs, the next proposed change is to prioritize a paired TorchOED cell before long PyDOE calls, subject to a new Architect ruling. No order change or retry was made in this attempt.

## Attempt 4/6 (2026-11-26): prioritize TorchOED paired cells
Rulings 7080475080 / 7197505476 / 7203940085: any time and available GPU; frozen library d869c1a, driver ec7e0fc, six CUDA/CPU cells in the prescribed order. Prior 18 PyDOE reference cells and C3 GLM are cited from attempt 3 above, not rerun.
Allocation: H100: ALLOCATION_FAILED, A100: ALLOCATION_FAILED, L4: allocated. Observed device and versions: {"nvidia_smi": [{"name": "NVIDIA L4", "memory_total_mib": 23034, "driver_version": "580.82.07"}], "torch": {"status": "AVAILABLE", "version": "2.14.1+cu130", "cuda": "13.0", "name": "NVIDIA L4", "capability": [8, 9]}}.
| k | m | method | seconds / status | logdet | returned floor | peak GPU bytes |
|---|---|---|---|---|---|---|
| 5 | 100000 | cuda | 102.882662932 | 59.2955345215783 | 0.9763343515837629 | 592449536 |
| 5 | 100000 | cpu | 754.9892664230001 | 59.295534521578304 | 0.9763343515837629 | 0 |
| 7 | 100000 | cuda | 93.71315486100002 | 116.80849172968863 | 0.9557194734951894 | 213503488 |
| 7 | 100000 | cpu | 66.00600667999993 | 116.80849172968864 | 0.9557194734951894 | 0 |
| 3 | 1000000 | cuda | TIMEOUT |  |  |  |
| 3 | 1000000 | cpu | NOT_RUN |  |  |  |
Local saved-ID objective rescoring: 4 PASS, 0 FAIL at 1e-9 relative. Same-host CPU/CUDA agreement and speedup: [{"k": 5, "m": 100000, "CPU_status": "PASS", "CUDA_status": "PASS", "agreement": "MATCH", "speedup": 7.338352691376273, "CPU": 59.295534521578304, "CUDA": 59.2955345215783}, {"k": 7, "m": 100000, "CPU_status": "PASS", "CUDA_status": "PASS", "agreement": "MATCH", "speedup": 0.7043408876576964, "CPU": 116.80849172968864, "CUDA": 116.80849172968863}, {"k": 3, "m": 1000000, "CPU_status": "NOT_RUN", "CUDA_status": "TIMEOUT", "agreement": "NOT_RUN", "speedup": "NOT_RUN"}].
Largest m reached this attempt: cpu 100000, cuda 100000.
GPU wins only on completed measured pairs: [{"k": 5, "m": 100000, "CPU_status": "PASS", "CUDA_status": "PASS", "agreement": "MATCH", "speedup": 7.338352691376273, "CPU": 59.295534521578304, "CUDA": 59.2955345215783}]. No speedup inferred from TIMEOUT or cross-host data.
Cleanup: 3 readbacks PASS; final 0 assignments; 329.75 CU. This attempt 2.20 CU, cumulative 19.44/25, remaining 40.56 under latest 60 CU task cap; November known spend since Nov19 19.60 CU (full month NOT_VERIFIED: pre-Nov19 usage not reconciled). Frozen attempt stayed within its original cap. Balance top-up of 200 CU does not reset task spend. No same-session retry.
Next change to try: resume any unfinished cells in a later authorized attempt; preserve completed measurements and caps. No library or tuning change.

## G1-M: million-candidate continuation (#595) — BLOCKED
Library frozen at 89bab8d2c0d1393c6b391760a372bf19dee160b1; driver 655ac6fb832e7e031c8f17b7b7e3001e0ec1f9ea. Frozen eight-cell order, 7200 s per cell, 28800 s internal budget; seed 0, float64, one restart. No completed cell recovered.
Original host: T4 requested; kernel identity, driver, CUDA, torch and threads NOT_VERIFIED (no recovered science artifact). H100/A100/L4 allocation failed with cleanup PASS; T4 session history records creation at 00:14:27 UTC and termination at 00:51:01 UTC. The local controller no longer existed when this run resumed; only a STARTED checkpoint survived. Cleanup readback PASS: no active sessions and zero assignments, 329.25 CU.
Resume host: provider status reported A100; kernel identity, driver, CUDA, torch and threads NOT_RUN. H100 allocation failed; A100 kernel probe exec TIMEOUT after 90 s (KERNEL_PROBE_FAILED), before science. Both controller cleanup readbacks PASS; zero assignments, 329.08 CU. No further allocation this run.

### Original T4 (identity NOT_VERIFIED)
| Cell | k | m | method | status |
|---|---|---|---|---|
| 1 | 3 | 1000000 | cuda | NOT_RUN |
| 2 | 3 | 1000000 | cpu | NOT_RUN |
| 3 | 5 | 1000000 | cuda | NOT_RUN |
| 4 | 5 | 1000000 | cpu | NOT_RUN |
| 5 | 5 | 100000 | cuda | NOT_RUN |
| 6 | 5 | 100000 | cpu | NOT_RUN |
| 7 | 5 | 100000 | cuda | NOT_RUN |
| 8 | 5 | 100000 | cpu | NOT_RUN |

### Resume A100 (provider hardware; kernel NOT_VERIFIED)
| Cell | k | m | method | status |
|---|---|---|---|---|
| 1 | 3 | 1000000 | cuda | NOT_RUN |
| 2 | 3 | 1000000 | cpu | NOT_RUN |
| 3 | 5 | 1000000 | cuda | NOT_RUN |
| 4 | 5 | 1000000 | cpu | NOT_RUN |
| 5 | 5 | 100000 | cuda | NOT_RUN |
| 6 | 5 | 100000 | cpu | NOT_RUN |
| 7 | 5 | 100000 | cuda | NOT_RUN |
| 8 | 5 | 100000 | cpu | NOT_RUN |

No saved IDs or objective exist: dense rescoring 0 PASS, 0 FAIL, 8 NOT_RUN. CPU/CUDA agreement and speedup NOT_RUN. Largest m reached in this task: CPU NOT_RUN, CUDA NOT_RUN; historical completed same-host scale remains 100000.
Repeated k=5,m=100000 timing spread: CPU NOT_RUN, CUDA NOT_RUN (zero completed repeats; no min/max).
Cleanup: five controller readbacks PASS plus original T4 recovery readback PASS; final zero assignments. Task spend 0.67/60 CU, remaining 59.33 CU. November known since Nov19 20.27 CU; full calendar month NOT_VERIFIED (pre-Nov19 unreconciled). No top-up treated as a reset.
Next change to try: Architect review of the 90 s A100 kernel-start timeout, then resume the frozen cells from the first missing result on a later authorized run; preserve completed cells and separate hosts. No speedup claim from NOT_RUN.
