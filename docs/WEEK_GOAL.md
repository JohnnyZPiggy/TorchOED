# Week goal — 2026-10-31 → 2026-11-07 (Scientist-approved direction)

Every Engineer dispatch this week must name the goal (G1–G4) it serves. When G1–G4 are all met, the loop stops creating new work and reports to the Scientist. Baseline: `main@f8a7e99`, `reports/b1/summary.md`.

| Goal | Falsifiable target (CPU, one thread, `tools/bench_b0.py` protocol unless stated) | Baseline (B1) |
|---|---|---|
| **G1 Speed** | `d_optimal_design` median time ≤ 3× the fastest PyDOE 1.6.0 method that reaches equal-or-lower quality, on **grid4 n=30** and **random500 n=10/20**; quality (logdet) unchanged vs B1 within 1e-9 | grid4 n=30: 5.45 s vs 0.43 s; random500 n=20: 175.7 s vs 1.55 s |
| **G2 Scale** | random 2000 candidates in [−1,1]³, full quadratic (p=10), n=20, `restarts=10`: completes ≤ 120 s with default `max_elements`, returns a certified efficiency bound | not run |
| **G3 Certificates** | every TorchOED result and every PyDOE design in the benchmark reports a certified D-efficiency lower bound (Kiefer–Wolfowitz, B2 #47) | relative efficiency only |
| **G4 Beta report** | `reports/beta/summary.md` (≤ 1 page) consolidating G1–G3 vs PyDOE, plus 0.5.0 release prep (version, CHANGELOG); tag delegated to the Engineer | — |

**Stretch (only after G1–G4):** one measured GPU run of the k=2 scoring on the G2 case (observed device, end-to-end time incl. transfer); no speedup claim beyond the measurement.

**Out of scope this week:** new criteria (A/I/…), Modified Fedorov / DETMAX, k ≥ 3, PyPI, API redesign, Issue #13.
