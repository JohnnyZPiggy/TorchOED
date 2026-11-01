# S1 GPU measurement: BLOCKED

**measurement only; no speedup or GPU-support claim.**

Source `98dd81f6dd9909e5c3e13cd6d0075b6562445f17`, base `2ff9ec8`, Issue #75. Pinned SAE controller `5fb26304e4c85183aa215e6ad4db9ced928cab12`, G4 requested once; no fallback or rerun. Observed **NVIDIA RTX PRO 6000 Blackwell Server Edition** (97887 MiB), remote torch 2.11.0+cu130 / CUDA13.0 / driver580.82.07. No new spending or plan change.

The seed0 warm-up raised `ValueError: ids and F must have the same device`: `d_optimal_design` passed CPU-generated IDs to `exact_information` with CUDA F. Full traceback is in results.json. No src fix is authorized. Warm-up did not complete; all three timed calls, objective/floor oracle checks, peak-memory results and ID comparisons are **NOT RUN**, not successful GPU measurements.

| Seed | Observed GPU | GPU wall / peak memory / correctness checks | Existing CPU G2 seconds |
|---|---|---|---|
| 0 | RTX PRO 6000 Blackwell Server Edition | NOT RUN (warm-up failed) | 8.046611 |
| 1 | same allocated device | NOT RUN | 7.454587 |
| 2 | same allocated device | NOT RUN | 10.242654 |

CPU values reuse `reports/beta/g2.json` (source823749a); no CPU G2 rerun. Command inside the pinned Colab adapter: `python -B tools/bench_g2.py s1-gpu cuda`, CPU float64 F, CPU generator, ten restarts, one thread. The intended timer includes CUDA transfer and ID return, synchronizing before stop. Since warm-up failed, no timing comparison is made.

Controller transport PASS means the failure artifact was downloaded, not scientific success: `scientific_status=BLOCKED`. Downloaded artifact SHA256/size are in results.json (external bytes only). Cleanup confirmed: stop PASS; authenticated usage readback **0 assignments before / 0 after**. No session remains from this task.

Tool-only change: optional device, CUDA warm-up/measurement and CPU oracle fields, plus committed transport adapter/manifest. No src/tests changed. Final-HEAD CPU suite and CI handoff are NOT RUN at this blocked stage, per the explicit stop-on-CUDA-error instruction. Next: Architect ruling or separate src-fix dispatch; no further run under this Issue.
