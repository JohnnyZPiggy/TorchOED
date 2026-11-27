# Week 18 goal — 2026-11-27 → 2026-12-03 (P2.5 part 2: torch REML, prediction profiler, desirability)

**Auto-advance (Architect, 2026-11-27).** Week 17 (P2.5 part 1) goals were met on 2026-11-27:
- G1 least-squares analysis: `fit_least_squares` (§123–§124, #554); R 4.5.2 `lm`/`anova` 2/2 published examples MATCH at 1e-8 in R's term order (warpbreaks; rsm::ChemReact, #562, #576).
- G2 REML reuse check: decision **BUILD** (#567, `reports/w17-reml-check`). PyStatistics 7.0.0 did not install (NOT_RUN, 0/81 compared). Recorded as-is: lme4 2.1-0 + lmerTest 3.2-1 against the exact balanced EMS solution reach max relative 1e-12 on fixed effects and log-likelihood, but only 3.6e-6–3.0e-5 on SEs, variance components and Satterthwaite df (optimizer tolerance): 26/81 R quantities within 1e-8 of exact.
- G3 P2+ completion: 2/2 checklist rows ✅. Orthogonal/Taguchi/covering arrays (§128–§134, #566, #570, #588; DoE.base 8/8 MATCH #573; N_H / best-known max 2.81, 70/99 cells ≤ 1.3, #596); non-box/ABCD mixtures (§126–§127, #559; Xvert conmx 2/2 MATCH at 1e-6, #572).

Held-out H46–H48 were recorded as-is (#587, PR #598, `reports/w17-heldout`, R 4.5.2, DoE.base 1.2-5, run 45924802098): **3/3 MATCH**.
- H46 MATCH: npk least squares vs R `lm`/`anova`, max relative 8.3e-15, max absolute p 2.7e-15.
- H47 MATCH: 4¹2¹² OA, 16 runs, strength 2, A3 = 34 on both sides.
- H48 MATCH: 13 vertices equal to the exact `Fraction` enumeration, missing 0, extra 0.

Baseline: `main` at the START of this goal's commit task.

**Why this week:** P2.5 continues (VISION.md, weeks 17–19). The G2 decision was BUILD, so REML for the fixed structures TorchOED designs (blocks, whole plots, strip plots) is implemented in torch. The W17 out-of-scope list put the prediction profiler and desirability in week 18. Studio is week 19.

**Reference rule for REML (measured in W17, #567).** R `lmer` meets 1e-8 against the exact solution only for fixed effects and the REML log-likelihood. So the primary REML oracle is the exact balanced EMS solution (`Fraction`). R lme4 + lmerTest is the external published reference: fixed effects and log-likelihood at 1e-8; variance components, SEs and Satterthwaite df compared at 1e-4 relative, with the R-vs-exact difference reported next to it. This is a measured limit of the reference, not a relaxed TorchOED target: TorchOED must match the exact solution at 1e-10.

| Goal | Falsifiable target | Baseline |
|---|---|---|
| **G1 REML fit for blocked, split-plot and strip-plot experiments (torch, BUILD #567)** | One public REML function (API fixed by its gate) for a fixed-effect model matrix X plus one to three random-intercept groupings (blocks; whole plots nested in blocks; the two strip factors crossed within blocks). It returns fixed effects, their SEs, variance components, the REML log-likelihood and Satterthwaite df per coefficient; CPU float64. On `nlme::Oats` and the frozen strip-plot CSV (`reports/w17-reml-check`) it matches the exact EMS solution at **1e-10** (relative) in every quantity, and R lme4 2.1-0 + lmerTest 3.2-1 per the reference rule above. Measured number: quantities matching the exact solution at 1e-10, target **81 / 81** (the #567 set); matrix rows `REML / mixed-model fit …` and `Mixed-model / REML analysis …` ✅ | 0 / 81 (no REML code); rows ◐ BUILD |
| **G2 Prediction profiler data** | One public function on a `fit_least_squares` result: the prediction at a point x₀ with its SE, confidence and prediction intervals (level given), and effect traces (prediction ± confidence interval along each factor on a grid, the others held at x₀). It matches R `predict(lm, newdata, interval = "confidence" / "prediction", se.fit = TRUE)` to **1e-8** relative on two published examples: `warpbreaks` (`breaks ~ wool + tension`, all 6 cells) and `rsm::ChemReact` (second-order model at the design centre and the `rsm` stationary point). Measured number: published prediction examples matching R at 1e-8, target **2**; matrix row `prediction profiler data …` ✅ | 0; row ❌ |
| **G3 Desirability and multi-response optimisation** | Derringer–Suich individual desirabilities (larger-is-better, smaller-is-better, target, with weights) and the overall geometric-mean desirability, plus a maximiser over a box region of the fitted models by torch autograd with multi-start (the "gradient-based optimisation of the fitted model" advantage). Checks: (a) desirability values equal R `desirability` 2.1 (Kuhn, pinned in `tools/r_packages.txt`) at **1e-10** on that package's published Myers–Montgomery conversion/activity example (vignette `desirability.Rnw` §4–§5: `dMax(80, 97)`, `dTarget(55, 57.5, 60)`, overall geometric mean) at the 5 points frozen in `reports/w18-r-refs` (#602, `r-oracle.yml` run 45936707367); (b) on the same example and its square region [−1.682, 1.682]³, the optimum's overall desirability is ≥ R's documented optimum (125-start Nelder–Mead, D = 0.942509403229837) − 1e-6, and ≥ the maximum over a dense grid − 1e-6. (Amended 2026-11-27: the package documentation has no tire-tread models, so the permitted published fallback of #602 is used.) Measured number: checks (a)+(b) passed, target **2 / 2**; matrix row `desirability functions …` ✅ | 0; row ❌ |

**Held-out additions.** These are frozen here, before any week-18 task merges. They are never used for tests, tuning or dispatch choices. They run once at week end and are reported as-is.
- **H49 (REML, published balanced split-plot):** R `lme4::cake` (270 rows), exported to CSV by `r-oracle.yml` in the week-end dispatch. Model `angle ~ recipe * temp + (1 | replicate) + (1 | replicate:recipe)`, where `temp` is `factor(temperature, ordered = FALSE)` with treatment contrasts in R's default column order; whole plot = `replicate:recipe`, block = `replicate`.
  - TorchOED: the G1 function with default settings on R's model matrix and those two groupings.
  - References: the exact balanced EMS solution (`Fraction`, as in `reports/w17-reml-check`); and `lmer` + lmerTest `summary()` from that dispatch with `sessionInfo()`.
  - Shared metric: max relative difference over fixed effects, SEs, the three variance components, the REML log-likelihood and the Satterthwaite df.
  - Verdict: MATCH if TorchOED vs exact ≤ 1e-10 in every quantity; otherwise NO MATCH with the worst quantity. If any exact EMS component estimate is ≤ 0, the exact column is NOT_RUN (boundary) and the verdict uses lme4 under the reference rule. The TorchOED-vs-lme4 differences are reported as-is under the reference rule.
- **H50 (prediction intervals, published data):** R `datasets::stackloss` (21 rows), exported to CSV in the same dispatch. Model `stack.loss ~ Air.Flow + Water.Temp + Acid.Conc.`; point x₀ = (Air.Flow 60, Water.Temp 21, Acid.Conc. 87), level 0.95.
  - TorchOED: the G2 function with default settings at x₀.
  - Reference: R `predict(fit, newdata, interval = "confidence", se.fit = TRUE)` and `interval = "prediction"`.
  - Shared metric: max relative difference over the fit, its SE and the four interval ends.
  - Verdict: MATCH if ≤ 1e-8, otherwise NO MATCH with the worst quantity.
- **H51 (desirability optimum, exact):** two fitted linear models on the box [−1, 1]²: ŷ₁ = 80 + 5x₁ + 3x₂ (larger is better, low 70, high 90, weight 1) and ŷ₂ = 10 − 2x₁ + 2x₂ (smaller is better, low 5, high 20, weight 1); overall D = (d₁ d₂)^{1/2}.
  - TorchOED: the G3 maximiser with default settings.
  - Reference: closed form. With x₁ = 1 optimal (both factors of d₁d₂ increase in x₁ on the box), d₁d₂ = (15 + 3x₂)(12 − 2x₂)/300 is maximised at x₂ = 1/2: optimum (1, 1/2), D* = √0.605.
  - Verdict: MATCH if |D − D*| ≤ 1e-8 and the returned point is within 1e-5 of (1, 1/2) in max norm; TorchOED better is impossible; otherwise NO MATCH with D and the point.

If a reference call fails or its versions are not recorded, its column is `NOT_RUN`, with the error verbatim.

**Timing rule (durable, Scientist 2026-11-07):** per restart unless a goal says otherwise.

**Safeguards:**
- Production numerical dependency stays torch only. R, statsmodels and the `desirability` package are dev-only references; pyproject, `src/` and `tests/` never import them. No adapter extra this week (BUILD decision #567).
- REML math is gated before code: identities marked PROVED/OBSERVED/CONJECTURE, independent math review, exact balanced EMS oracle in the tests. Reported tables are float64; no FP32 path feeds them.
- Analysis code calls no design search; any change to search code reruns `tools/bench_b0.py`, with no time regression and identical D results.
- The library stays GUI-free; no Studio work this week (W19). Profiler and desirability return data, not plots.
- External references follow the VISION.md oracle rules: R on `r-oracle.yml`, Colab only for GPU, one GPU task at a time.
- No tuning on held-out cases. Historical reports and evidence are preserved.

**Out of scope this week:**
- Studio (W19); bootstrap intervals and batched closed-loop fits (carried to W19 alongside Studio).
- Random slopes, unbalanced-data Kenward–Roger df, general GLMM, survival analysis and GP surrogates (out of scope by decision or beyond the designed structures).
- Deferred P2+ items (Scientist 2026-11-21): group-orthogonal supersaturated designs, hard-to-change covariates, sphere packing, MSA, batched sample-size search.
- P3, A/I GPU timing (W23), real PyPI and Issue #13.
