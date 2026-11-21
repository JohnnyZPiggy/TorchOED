# Week 16 goal — 2026-11-21 → 2026-11-28 (CONSOLIDATION: P2+ exit — parity audit and deferral table, documentation and repository health, 0.9.0 TestPyPI)

**Auto-advance (Architect, 2026-11-21).** Week 15 (P2+ part 3) goals were met on 2026-11-21:
- C1 split-split-plot and strip-plot: every cell filled. skpr 1.9.2 SS1 MATCH and the published `gen_design.Rd` nested example MATCH (#473).
- C2 minimum-aberration and blocked FF: FrF2 2.3-5 6/6 feasible cases MATCH, and BL2 is infeasible on both sides (#469).
- C3 GLM power by simulation: skpr 12/12 within 4 combined MC SE (#472). Published RD2b MATCH; RD1b NO MATCH as-is. A100 same-host timing (#509): nsim 10⁵ CPU 5.30 s vs CUDA 0.422 s. HO is open and is filled by H44 below.
- Published examples: 16 → 19 (target ≥ 19).

Held-out H40–H42 were recorded as-is (#512, PR #514, `reports/w15-heldout`):
- H40 MATCH: det = 2097152/441 = exhaustive maximum (4,304 feasible assignments, 128 maximizers).
- H41 MATCH: det = 2097152/315 = exhaustive maximum (5,952 feasible, 48 maximizers).
- H42 MATCH: WLP equal to FrF2 2.3-5 (R 4.5.2, run 44868066239).

Baseline: `main` at the START of this goal's commit task.

**Why this week:** week 16 is a consolidation week (VISION.md weekly cadence), so it adds no new features. The P2+ exit evidence in VISION.md has three parts:
- every parity-checklist row complete or Scientist-deferred;
- ≥ 15 published examples (19, met);
- one of the four signatures demonstrated on a published example.

Several checklist rows are still ◐ or ❌: orthogonal / covering arrays, hard-to-change covariates, non-box mixture regions, group-orthogonal supersaturated designs and the space-filling row, whose status text is stale. These need either evidence or a deferral the Scientist can decide on in one place. The documentation also lags weeks 13–15: 13 report directories are missing from `reports/README.md`, and `website/api.md` and `docs/ALGORITHM_COVERAGE.md` have not been checked since week 12. The last release was 0.8.0 (week 12). P2.5 (week 17) needs the Mentor landscape review of statistics packages first (VISION.md "Reuse before build"; matrix P2.5 row 1, dated W16). That review runs in parallel as a [Mentor] Issue; it is not a feature goal.

| Goal | Falsifiable target | Baseline |
|---|---|---|
| **G1 P2+ exit audit and deferral table** | (a) Every parity-checklist row whose status lags its P2/P2+ matrix row is corrected, with evidence links; the space-filling row at least. (b) One table "P2+ open items for Scientist decision" in `docs/CAPABILITY_MATRIX.md` lists every P2/P2+ cell or checklist row that is still ◐, ❌ or NR and not completed this week. Each entry gives the missing evidence, the reason and the week that would close it. It covers at least orthogonal / Taguchi / covering arrays, hard-to-change covariates, non-box mixture regions, group-orthogonal supersaturated designs, DSD Ext, mixture Con, A/I GPU timing, and incomplete block / crossover designs. (c) One line in the matrix names the P2+ signature evidence on a published example: a certificate or a measured GPU speed-up at equal quality, from merged reports only, or "not shown" stated as-is. Measured number: P2/P2+ cells and checklist rows that are neither complete nor listed, target **0**; stale checklist rows, target **0** | stale rows counted by the G1 task (at least the space-filling row); 5 open checklist rows; P2+ signature line absent |
| **G2 Documentation and repository health** | (a) `reports/README.md` indexes every directory under `reports/`, and the `g1-crossover` row is tidied (description in the Measurement column, status in the Status column). (b) `website/api.md` lists every name in `torchoed.__all__` and every public keyword merged through week 15. This is checked once by a throw-away script whose output goes in the handoff, not the repository. (c) `docs/ALGORITHM_COVERAGE.md` marks no merged capability PLANNED. (d) `docs/RELEASE_READINESS.md` reflects 0.9.0. (e) Test-suite wall time on `ubuntu-latest, 3.13` is recorded at the start and end of the week; any test change keeps every behaviour test and must not slow the suite. Measured number: unindexed report directories + exported names or keywords missing from the API page + stale PLANNED entries, target **0** | 13 unindexed report directories; API and coverage gaps counted by the G2 task |
| **G3 Release 0.9.0 on TestPyPI** | Steps: (1) one fully green CI run on `main`, including `test (macos-latest, …)`, recorded by run id; (2) version 0.9.0, with the CHANGELOG `Unreleased` section becoming `[0.9.0]`; (3) the `testpypi.yml` workflow publishes; (4) a clean-venv install from TestPyPI runs `examples/cpu_quickstart.py` and the industrial-guide snippets, and the outputs are recorded. The Engineer pushes the tag in an Engineer task. Measured number: release steps done, target **4/4** | 0/4 (0.8.0 on TestPyPI) |

**Held-out additions.** These are frozen here, before any week-16 task merges. They are never used for tests, tuning or dispatch choices. They run once at week end and are reported as-is. Every randomized TorchOED call uses `generator=torch.Generator().manual_seed(0)` and default settings otherwise.
- **H43 (blocked minimum aberration):** `minimum_aberration(10, 64, blocks=4)`.
  - Shared metric: `word_length_pattern` on the 10 factor columns. Also the list of main effects and two-factor interactions whose ±1 column is constant within every block, computed by one function for both designs from the design and its block labels.
  - Reference: R `FrF2::FrF2(nruns = 64, nfactors = 10, blocks = 4, randomize = FALSE)` through `r-oracle.yml` with `sessionInfo()`, exported as a CSV of the ±1 factor columns plus the block label.
  - Verdict: MATCH if the WLPs are equal and both lists are empty. TorchOED better if both lists are empty and TorchOED's WLP is lexicographically smaller. Otherwise NO MATCH, with the first differing word length or the confounded effects.
- **H44 (Poisson GLM power):** X is [1, A, B, C] on the 2³ grid on {−1, 1} in `full_factorial` order, listed twice (n = 16), frozen as a repo CSV before the reference dispatch. The call is `glm_power(X, beta=torch.tensor([0.5, 0.5, -0.25, 0.0], dtype=torch.float64), family="poisson", alpha=0.05, nsim=10000, generator=torch.Generator().manual_seed(0))`.
  - Shared metric: per-coefficient power with MC SE √(p(1 − p)/nsim).
  - Reference: R `skpr::eval_design_mc(design = <the frozen CSV columns A, B, C>, model = ~A + B + C, alpha = 0.05, nsim = 10000, glmfamily = "poisson", anticoef = c(0.5, 0.5, -0.25, 0))` through `r-oracle.yml` with `sessionInfo()`, using its parameter power rows.
  - Verdict: MATCH if all 4 coefficients are within 4 combined MC SE. Otherwise NO MATCH, with the worst z.
- **H45 (strip-plot A, unequal variance ratios):** candidates and F as in H40, so m = 16 and F = [1, A, B, C, AB, AC, BC]; n = 8 as a 2 × 4 grid. The call is `optimal_design(F, 8, criterion="A", strip=(2, 4), row_hard=<A level of each candidate>, col_hard=<B level of each candidate>, eta_row=2.0, eta_col=0.5, generator=torch.Generator().manual_seed(0))`, with runs in the documented row-major order.
  - Shared metric: tr((XᵀV⁻¹X)⁻¹) in exact `Fraction`, where V = I + 2·Z_rZ_rᵀ + ½·Z_cZ_cᵀ, on the design as returned.
  - Reference: dev-only exhaustive enumeration of the 2² · 2⁴ · 2⁸ = 16,384 level assignments (A per row, B per column, C per cell). Keep the assignments in which no (A, B, C) appears more than twice and whose information matrix is nonsingular.
  - Verdict: MATCH if TorchOED equals the exhaustive minimum. Otherwise NO MATCH, with Δ.

If a reference call fails or its versions are not recorded, its column is `NOT_RUN`, with the error verbatim.

**Timing rule (durable, Scientist 2026-11-07):** per restart unless a goal says otherwise.

**Safeguards:**
- No new public functions or keywords. Refactors keep every behaviour test and every D/A/I result unchanged. Any change to search code reruns `tools/bench_b0.py`, with no time regression and identical D results.
- Matrix and checklist statuses cite merged reports or PRs only. Deferral entries are proposals: a row counts as deferred only after a Scientist signal.
- External references follow the VISION.md oracle rules: R on `r-oracle.yml`, Colab only for GPU, one GPU task at a time.
- No tuning on held-out cases. Historical reports and evidence are preserved, never deleted.
- Releases and tag pushes are Engineer tasks. Real PyPI waits for the Scientist.

**Out of scope this week:**
- New capabilities: orthogonal / covering arrays, hard-to-change covariates, non-box mixture regions, new DSD sizes (deferral or scheduling is a Scientist decision via G1).
- Analysis and Studio (P2.5, week 17 onward). Adapters wait for the Mentor landscape review.
- P3. GPU timing of A/I (W23).
- Real PyPI and Issue #13.
