# Week 2 goal — 2026-11-01 → 2026-11-08 (VISION.md phase P1, first P1 week)

Previous week (2026-10-31 → 11-07, G1–G4 + stretch) is **complete**: see `reports/beta/summary.md`, `reports/s1-gpu/`. Baseline now: `main@1795c26`, v0.5.0.

Every Engineer dispatch must name the goal (W1–W4) it serves. When W1–W4 are all met, the loop stops creating new work and reports to the Scientist.

| Goal | Falsifiable target | Baseline |
|---|---|---|
| **W1 Generic criteria** | `optimal_design(F, n, criterion="D"\|"A"\|"I", ...)` runs the same 1-/2-swap exchange for each criterion from shared low-rank update identities (gate doc first), each returning a certified efficiency lower bound; independent `Fraction` exhaustive oracle on a small fixture for A and I; D results unchanged vs v0.5.0 | D only |
| **W2 Augmentation + constraints** | `fixed_ids=` keeps given runs and optimizes the rest (augmentation); candidate constraints (linear inequalities and disallowed level combinations) and categorical factors via a documented coding; exhaustive oracle on a small constrained fixture | none |
| **W3 Held-out external benchmark** | (a) S-size probe first: one R (and, if available, Julia) Colab runtime via the pinned SAE controller, AlgDesign installed, one published example reproduced, exact versions recorded; (b) new `tools/bench_heldout.py` on cases the loop has never optimized against (incl. constrained and categorical), vs PyDOE 1.6.0 and R AlgDesign run on Colab (dev-only oracles): report quality, certified floor and time as-is, run once at week end | PyDOE only, fixed cases |
| **W4 Public basics** | typed public API; docs site on GitHub Pages (quick start, criteria, augmentation, PyDOE comparison, API reference); CI adds macOS and the minimum supported torch; TestPyPI dry run (real PyPI upload needs the Scientist's token) | README only |

**External oracles:** R/Julia run only on Colab under the protocol in `docs/VISION.md`; never a project dependency.

**Safeguards this week:** every new gate document passes an independent fresh-context review before merge; W3 cases are frozen in the Issue before any W1/W2 code merges and are never tuned against.

**Out of scope this week:** split-plot, mixture, nonlinear/Bayesian (P2/P3), GPU work, k ≥ 3, Modified Fedorov/DETMAX, Issue #13.
