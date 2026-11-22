# Week 17 goal — 2026-11-22 → 2026-11-29 (P2.5 part 1: least-squares analysis, REML reuse check, P2+ completion items)

**Auto-advance (Architect, 2026-11-22).** Week 16 (consolidation, P2+ exit) goals were met on 2026-11-22:
- G1 P2+ exit audit: every P2/P2+ open item is listed and decided by the Scientist (21 items, #540, PR #542); undecided 0. P2+ signature on a published example shown (#541, PR #544): certified AlgDesign D/A/I floors; fully fixed A/I labels fixed (#545, PR #547).
- G2 documentation and repository health: unindexed report directories, API-page gaps and stale PLANNED entries 0 (#525, #528, #530, #534, #539). Suite wall time on `ubuntu-latest, 3.13`: start 388 passed in 71.64 s, end 392 passed in 55.05 s (different hosts; descriptive only, #546).
- G3 0.9.0 on TestPyPI: 4/4 (#518).
- W16-perf (CPU A/I at m = 10⁴ within 120 s) is not met: A 488 s, I 280 s (#537). It is not a week-17 goal; it stays open in the matrix Dev cell and is revisited with W23 GPU scale.

Held-out H43–H45 were recorded as-is (#524, PR #546, `reports/w16-heldout`):
- H43 MATCH: WLP equal to FrF2 2.3-5, no main effect or 2FI confounded with blocks on either side (R 4.5.2, run 45009364423).
- H44 MATCH: skpr 1.9.2 Poisson powers within 4 combined MC SE, worst z 1.74.
- H45 MATCH: exact A trace 13/4 equals the exhaustive minimum (960 feasible assignments, 48 minimizers).

Baseline: `main` at the START of this goal's commit task.

**Why this week:** P2.5 starts (VISION.md, weeks 17–19). The Mentor landscape review (§26, #517) recommends building the thin torch least-squares path ourselves (§26.6: one factorisation shared by all responses), and checking PyStatistics against lme4 + lmerTest on `nlme::Oats` before any REML adapter (§26.8: NEEDS CHECK). The Scientist also put two P2+ items in week 17 (2026-11-21, "Do now: W17"): orthogonal / Taguchi / covering arrays, and non-box mixture regions including ABCD.

| Goal | Falsifiable target | Baseline |
|---|---|---|
| **G1 Least-squares analysis in the library** | One public least-squares function (API fixed by its gate) on a design matrix and one or more responses (Y n × R, one shared factorisation). It returns coefficients, SEs, t tests, sequential per-term F tests (as R `anova(lm(...))`), lack of fit against pure error when replicates exist, and residual diagnostics (residuals, leverage, studentized residuals, Cook's distance). It matches R `lm` / `anova` to **1e-8** (relative for estimates, SEs, SS and F; absolute for p-values) on two published examples: `warpbreaks` (`breaks ~ wool + tension`, lack of fit against the 6 cells) and `rsm::ChemReact` (second-order model, lack of fit). R runs on `r-oracle.yml` with `sessionInfo()`. Measured number: published analysis examples matching R at 1e-8, target **2**; matrix row `least-squares fit` ✅ except Dev (CPU float64; CUDA runs, no timing claim) | 0 published analysis examples; row ❌ |
| **G2 REML for blocked and split-plot designs: reuse check first** | (a) Evidence: the PyStatistics CPU LMM (exact version pinned) on `nlme::Oats` (`yield ~ Variety * factor(nitro) + (1 | Block/Variety)`) and on one strip-plot data set frozen as a repo CSV, against `lme4::lmer` + `lmerTest` (Satterthwaite df) on `r-oracle.yml`. Compared: fixed effects, their SEs, variance components, REML log-likelihood and Satterthwaite df, at 1e-8 (Mentor §26.8). (b) Decision recorded in the matrix: **ADOPT** (both cases match; an optional extra adapter, never a core dependency, follows in W18) or **BUILD** (torch REML for blocks, whole plots and strip plots, gated by the same r-oracle check, W18). Measured number: Oats + strip-plot quantities matching lme4/lmerTest at 1e-8 out of those compared, recorded as-is; adapter row and REML row carry the decision | Mentor §26 NEEDS CHECK; rows ❌ |
| **G3 P2+ completion (Scientist 2026-11-21: "Do now: W17")** | (a) Orthogonal, Taguchi and covering arrays: one constructor family for strength-2 mixed-level orthogonal arrays (including the standard L-arrays) and strength-2/3 covering arrays, with an exact balance / coverage check. External: DoE.base `oa.design` catalogue arrays matched (run size, strength 2, equal GWLP A3) through `r-oracle.yml`. (b) Non-box mixture regions: extreme vertices and centroids of a mixture region with general linear constraints (ABCD / McLean–Anderson), checked against an exact `Fraction` vertex enumeration and against `mixexp::Xvert` with `conmx` at 1e-6 (its single precision, #289). Measured number: checklist rows `Orthogonal arrays …` and `Mixture …` complete, target **2 of 2** | OA row ❌; mixture row ◐ (box regions only) |

**Held-out additions.** These are frozen here, before any week-17 task merges. They are never used for tests, tuning or dispatch choices. They run once at week end and are reported as-is.
- **H46 (least-squares ANOVA, published data):** R `datasets::npk` (24 rows), exported to CSV by `r-oracle.yml` in the week-end dispatch. The model is `yield ~ block + N + P + K` with treatment contrasts in R's default column order.
  - TorchOED: the G1 function with default settings on that model matrix.
  - Shared metric: against R `lm` / `anova`, the max relative difference over coefficients, SEs, and the sequential ANOVA df, SS and F, and the max absolute difference over the t and F p-values.
  - Verdict: MATCH if both are ≤ 1e-8, otherwise NO MATCH with the worst quantity.
- **H47 (mixed-level orthogonal array):** one four-level and twelve two-level factors.
  - TorchOED: the G3(a) constructor with default settings, asked for the smallest strength-2 array.
  - Reference: R `DoE.base::oa.design(nlevels = c(4, rep(2, 12)), randomize = FALSE)` through `r-oracle.yml` with `sessionInfo()`.
  - Shared metric: run count, strength-2 balance of every column pair, and GWLP A3, computed by one function in `run.py` for both arrays.
  - Verdict: MATCH if both are strength 2 with equal run count and A3. TorchOED better if it is strength 2 with fewer runs, or equal runs and smaller A3. Otherwise NO MATCH.
- **H48 (non-box mixture vertices):** q = 4, x₁ + x₂ + x₃ + x₄ = 1, lower bounds (0.1, 0.1, 0.05, 0), upper bounds (0.6, 0.5, 0.4, 0.3), and 0.3 ≤ x₁ + x₂ ≤ 0.8.
  - TorchOED: the G3(b) vertex function with default settings.
  - Reference: dev-only exact `Fraction` enumeration of the region's vertices (all 3-subsets of the inequality constraints, solved together with the mixture equality, kept if feasible).
  - Verdict: MATCH if the vertex sets are equal to 1e-12, otherwise NO MATCH with the missing or extra vertices.

If a reference call fails or its versions are not recorded, its column is `NOT_RUN`, with the error verbatim.

**Timing rule (durable, Scientist 2026-11-07):** per restart unless a goal says otherwise.

**Safeguards:**
- Production numerical dependency stays torch only. R, statsmodels and PyStatistics are dev-only references this week; pyproject, `src/` and `tests/` never import them. An adapter extra is a W18 task and only after an ADOPT decision.
- Analysis code calls no design search; any change to search code reruns `tools/bench_b0.py`, with no time regression and identical D results.
- Every numerical claim has an independent oracle (dense `torch.linalg` closed form or exact `Fraction`) in its test. Analysis tables are float64; no FP32 path feeds a reported table.
- The library stays GUI-free; no Studio work this week (W19).
- External references follow the VISION.md oracle rules: R on `r-oracle.yml`, Colab only for GPU, one GPU task at a time.
- No tuning on held-out cases. Historical reports and evidence are preserved.

**Out of scope this week:**
- Studio (W19); prediction profiler, desirability and bootstrap / batched closed-loop fits (W18); the adapter extra itself (W18, after the G2 decision).
- General GLMM, survival analysis and GP surrogates (out of scope by decision).
- Deferred P2+ items (Scientist 2026-11-21): group-orthogonal supersaturated designs, hard-to-change covariates, sphere packing, MSA, batched sample-size search.
- P3, A/I GPU timing (W23), real PyPI and Issue #13.
