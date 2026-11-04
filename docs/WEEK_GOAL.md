# Week 3 goal — 2026-11-05 → 2026-11-12 (VISION.md phase P1, last P1 week)

Week 2 (2026-11-01 → 11-08) is **complete**: W1 criterion-generic D/A/I exchange with certificates (#92, #108), W2 augmentation, constraints and categorical coding (#94, #98), W3 R AlgDesign probe and frozen held-out H1–H4 run once (#88, #102, `reports/w3-heldout/`), W4 typed API, docs site, macOS + minimum-torch CI, sdist and TestPyPI 0.5.0 (#96, #100, #104, #106, #107). Baseline now: `main@792c37e`, v0.5.0.

**Why this week:** the P1 exit evidence (VISION.md) is "held-out benchmark vs PyDOE + R AlgDesign: quality ≥, time ≤ 3× fastest comparable". Quality is met on H1–H4, but time is not: H3 D is 3.58× PyDOE detmax, and small dev cases have a fixed per-call overhead. The certified floors are also loose (median D floor 0.57 on the dev bench), so the "certified designs" signature is weak. This week closes P1 on those two numbers and ships 0.6.

Every Engineer dispatch must name the goal (W1–W4) it serves and the measured number it moves. Dev numbers come from `tools/bench_b0.py` only. Held-out cases are never used for tuning.

| Goal | Falsifiable target | Baseline (measured) |
|---|---|---|
| **W1 P1 time target** | On every `bench_b0` case where some PyDOE exchange method (fedorov / modified_fedorov / detmax) is feasible, TorchOED D median time ≤ 3× the fastest of them, same session and machine; D IDs, logdet and certificates unchanged vs v0.5.0 | 4/7 cases fail: grid2 n=7 16.5×, grid2 n=6 11.7×, grid3 n=10 4.4×, grid3 n=20 3.1× (`reports/w95-bench/head.json`); grid4/random500 ≤ 1.84× |
| **W2 Tighter certificates** | Gate extension first (independent review): an efficiency floor from the approximate-design dual bound (D: `logdet M* ≤ logdet M(w) + p·log(max_j d_j(w)/p)` for any PD weights `w`; A/I: the gate §4 bound evaluated at `M(w)`), returned as the max of it and the current floor. Targets: median D floor on the 7 feasible `bench_b0` cases ≥ 0.90; A and I floors on grid3 n=20 ≥ 0.75; the floor is never above the exact efficiency in an exhaustive `Fraction` oracle | D median 0.57 (0.29–0.86); A 0.588, I 0.511 on grid3 n=20 (`reports/w105/`). Architect float64 sandbox estimate with 2000 multiplicative iterations: D median 0.93 (0.86–0.99) |
| **W3 Held-out week 3** | 2–4 new cases H5–H7(H8) frozen in their Issue before any W1/W2 code merges (at least: one larger candidate set m ≥ 2000, one augmentation case with fixed runs, one constrained mixed-categorical I case). At week end, run H1–H8 once vs PyDOE 1.6.0 and R AlgDesign (Colab, VISION protocol), and report the P1 exit test per row as-is: quality ≥ best reference, time ≤ 3× fastest comparable (same machine only) | H1–H4: quality ≥ AlgDesign on 9/9 rows; time fails on H3 D (3.58× detmax); A/I rows predate the #108 two-swap |
| **W4 Release 0.6.0 readiness** | Version 0.6.0 with CHANGELOG (A/I, augmentation, constraints, categorical, new floors); TestPyPI 0.6.0 published by the existing workflow and a clean-venv install check PASS; a written real-PyPI checklist for the Scientist (the upload needs the Scientist's pypi.org trusted publisher or token) | 0.5.0 on TestPyPI (#107); real PyPI not set up |

**External oracles:** PyDOE locally; R/Julia only on Colab under the protocol in `docs/VISION.md`. Never a project dependency.

**Safeguards this week:** the W2 gate extension passes an independent fresh-context review (Fraction checks on small random cases) before merge. H5–H8 are frozen in the W3 Issue before any W1/W2 code merges and are never tuned against. W1 changes keep D results bit-identical and rerun `tools/bench_b0.py` as a same-session base/head pair.

**Out of scope this week:** P2 features (split-plot / hard-to-change factors, blocking, mixture, covariates, definitive screening, design diagnostics); nonlinear/Bayesian designs (P3); GPU work; k ≥ 3; Modified Fedorov/DETMAX implementations; Issue #13; any real-PyPI upload or tag/Release without the Scientist.

**Next:** week 4 (2026-11-12 →) is a **consolidation week**: no new features. Candidates: close stale Issues #2, #13 (historical) and #45 (Mentor, unanswered since 2026-10-31) with the Scientist's agreement, reduce the docs/ API files (M1–M3 API pages, 24 files now), and speed up the test suite.
