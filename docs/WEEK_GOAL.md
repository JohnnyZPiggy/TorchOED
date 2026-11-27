# Week 19 goal — 2026-11-27 → 2026-12-03 (P2.5 part 3: TorchOED Studio, bootstrap and batched fits)

**Auto-advance (Architect, 2026-11-27).** Week 18 (P2.5 part 2) goals were met on 2026-11-27:
- G1 torch REML: `fit_reml` (§136–§138, #607); exact balanced EMS solution **81 / 81** quantities at 1e-10; R lme4 2.1-0 + lmerTest 3.2-1 within the W18 reference rule.
- G2 prediction profiler data: `predict_least_squares` + `prediction_trace` (§139, #609); R `predict` **2 / 2** published examples MATCH at 1e-8 (`reports/w18-predict`).
- G3 desirability: `d_max`, `d_min`, `d_target`, `d_overall`, `maximize_desirability` (§140–§142, #611); **2 / 2** checks: 15 frozen R `desirability` 2.1 values, max abs 8.9e-15; default-start D = 0.9425094070843 ≥ R 0.942509403229837 and ≥ the 101³ grid maximum 0.9400058.

Held-out H49–H51 were recorded as-is (#610, PR #613, `reports/w18-heldout`, R 4.5.2, lme4 2.1-0, lmerTest 3.2-1, run 45985746679): **3/3 MATCH**.
- H49 MATCH: `lme4::cake` REML vs the exact EMS solution, max relative 2.96e-13; all variance components positive.
- H50 MATCH: `stackloss` prediction at x₀ vs R `predict`, max relative 2.1e-15.
- H51 MATCH: exact desirability optimum, |D − √0.605| = 2.2e-16, point error 5.1e-11.

Baseline: `main` at the START of this goal's commit task.

**Why this week:** P2.5 ends this week (VISION.md, weeks 17–19). The analysis functions are in the library; this week builds TorchOED Studio on top of them (VISION.md "Library first, light Studio") and closes the last P2.5 analysis row (bootstrap intervals and batched fits, carried from W18). Week 20 is a consolidation week.

**Studio rules (binding for G1–G2).**
- Studio is its own package in `studio/` (distribution `torchoed-studio`, import `torchoed_studio`), with its own `pyproject.toml`; it depends on `torchoed` and the Python standard library only. The core library never imports it; core `pyproject.toml` and `src/` are unchanged by Studio work.
- Server: a standard-library HTTP server with JSON endpoints, one per workflow step; every endpoint calls public `torchoed` functions only. Client: one static page (plain HTML/CSS/JS, no build step, no CDN), responsive, with a web app manifest and a service worker so it is installable as a PWA (VISION.md P6 groundwork).
- Tests live in `studio/tests/` and run in ordinary CI on one Linux job. Every Studio number in a test is compared with the direct public-library call on the same inputs (equal values, not re-tolerance) and, where a frozen R reference already exists in `reports/`, with that reference at its frozen tolerance.
- Published examples: **E1** `rsm::ChemReact` (2-factor central composite design in 2 blocks, 14 runs; data in `reports/w17-lm-published/ChemReact.csv`) and **E2** `nlme::Oats` (split-plot: 6 blocks, 3 varieties on whole plots, 4 nitrogen levels on subplots; data in `reports/w17-reml-check/Oats.csv`).

| Goal | Falsifiable target | Baseline |
|---|---|---|
| **G1 Studio design workflow** | Five steps through the Studio JSON API: (1) factors and model, (2) generate design (public constructors or search; for E1 and E2 the generated runs equal the published layout as a multiset of runs, block and whole-plot labels up to relabelling; E1 compared in natural units to the published 2-decimal rounding, |Δ| ≤ 0.005), (3) evaluate (`design_diagnostics`: correlation map, VIF, prediction variance/FDS, power), (4) compare designs (`compare_designs` of the generated design with one alternative of the same size), (5) export CSV (re-import is value-identical). Measured number: steps passing the automated workflow test on both examples with every value equal to the direct library call, target **10 / 10**; matrix row `Studio: factors/model → design → evaluate → compare → export` ✅ | 0 / 10; no `studio/` |
| **G2 Studio analysis workflow and demo** | Three steps through the Studio JSON API on E1 and E2: (1) import responses (CSV), (2) fit (`fit_least_squares` for E1 with the second-order model; `fit_reml` for E2 with blocks and whole plots), (3) profile (`predict_least_squares` / `prediction_trace` at the design centre for E1, and `maximize_desirability` for one response goal). Fit and profile values equal the direct library call and the frozen R references (`reports/w17-lm-published`, `reports/w18-predict` at 1e-8; `reports/w17-reml-check` at 1e-10 vs exact EMS). Plus a static demo of the Studio page on the docs site that replays recorded API responses for E1 and E2 (no server; live hosting needs a Scientist decision because no paid service is added). Measured number: analysis steps passing on both examples, target **6 / 6**, and the demo page reachable on the deployed site; matrix row `Studio: import responses → fit → profiler; hosted demo` ✅ (live hosting marked as the open item) | 0 / 6; no demo |
| **G3 Bootstrap intervals and batched fits** | One public residual-bootstrap function on a `fit_least_squares` fit (fixed design, Davison–Hinkley Algorithm 6.3: modified residuals rᵢ = eᵢ/√(1 − hᵢ), centred; y* = ŷ + r*; percentile intervals by R `quantile(type = 7)`), accepting either a torch generator or a given (n, B) index matrix, and fitting all B replicates in one batched call; plus batched least-squares fits over B designs of equal size, (B, n, p) → coefficients and SEs. Checks: (a) with index matrices frozen from one `r-oracle.yml` dispatch, replicate coefficients and interval ends equal R `lm` refits and `quantile(type = 7)` at **1e-10** on `datasets::stackloss` and `rsm::ChemReact`; (b) batched-over-designs coefficients and SEs equal a loop of `fit_least_squares` at **1e-12** and an exact `Fraction` oracle on a small case; (c) one GPU-track measurement, CPU vs CUDA, B ∈ {10⁴, 10⁵, 10⁶}, recorded as-is (the default device stays CPU unless CUDA is measured faster). Measured number: checks passed, target **3 / 3** ((c) counts only when measured); matrix row `bootstrap intervals and batched fits for closed-loop rounds` ✅ | 0 / 3; row ❌ |

**Held-out additions.** These are frozen here, before any week-19 task merges. They are never used for tests, tuning or dispatch choices. They run once at week end and are reported as-is.
- **H52 (Studio import → fit, published data):** R `datasets::ToothGrowth` (60 rows), exported to CSV by `r-oracle.yml` in the week-end dispatch. Through the Studio JSON API with default settings: import the CSV, fit `len ~ supp * factor(dose)` (treatment contrasts in R's default column order).
  - Reference: R `lm` + `anova` on the same model from that dispatch, with `sessionInfo()`.
  - Shared metric: max relative difference over coefficients, SEs, and the ANOVA F and p values.
  - Verdict: MATCH if ≤ 1e-8, otherwise NO MATCH with the worst quantity.
- **H53 (Studio profiler, published data):** R `datasets::PlantGrowth` (30 rows, `weight ~ group`), exported in the same dispatch. Through the Studio JSON API with default settings: import, fit, and profile at each of the three groups (level 0.95).
  - Reference: R `predict(fit, newdata, interval = "confidence", se.fit = TRUE)` and `interval = "prediction"`.
  - Shared metric: max relative difference over the fit, its SE and the four interval ends, all three groups.
  - Verdict: MATCH if ≤ 1e-8, otherwise NO MATCH with the worst quantity.
- **H54 (residual bootstrap, frozen indices):** R `datasets::cars` (50 rows), model `dist ~ speed`. The same dispatch writes the index matrix `set.seed(20261203); matrix(sample.int(50, 50 * 999, replace = TRUE), 50)` and computes in R the Algorithm 6.3 replicates (modified centred residuals as in G3, y* = ŷ + r*[idx], `lm` refit) and the 0.025 / 0.975 `quantile(type = 7)` interval for both coefficients.
  - TorchOED: the G3 bootstrap function with default settings and that index matrix.
  - Shared metric: max relative difference over the four interval ends.
  - Verdict: MATCH if ≤ 1e-10, otherwise NO MATCH with the worst quantity.

If a reference call fails or its versions are not recorded, its column is `NOT_RUN`, with the error verbatim.

**Timing rule (durable, Scientist 2026-11-07):** per restart unless a goal says otherwise.

**Safeguards:**
- Production numerical dependency stays torch only. Studio depends on `torchoed` and the standard library only and is never a dependency of `torchoed`. R stays a dev-only reference; pyproject, `src/`, `studio/` and `tests/` never import or call it.
- Studio calls public library functions only and adds no numerical code of its own; any number shown in Studio is the library's number.
- Bootstrap math is gated before code (identities marked PROVED/OBSERVED/CONJECTURE, independent math review). Reported tables are float64; no FP32 path feeds them.
- Analysis and Studio code call no new search code; any change to search code reruns `tools/bench_b0.py`, with no time regression and identical D results.
- No paid service: no paid hosting, no paid runners; live Studio hosting is a Scientist decision.
- External references follow the VISION.md oracle rules: R on `r-oracle.yml`, Colab only for GPU, one GPU task at a time.
- No tuning on held-out cases. Historical reports and evidence are preserved.

**Out of scope this week:**
- Live hosted Studio server (needs a Scientist decision on a free host), user accounts, persistent projects, plotting libraries, JMP-style data tables or scripting.
- Browser automation tests (consider in the W20 consolidation), mobile packaging (P6).
- Pairs (case) bootstrap, BCa intervals, bootstrap for REML fits.
- Random slopes, unbalanced-data Kenward–Roger df, general GLMM, survival analysis and GP surrogates (out of scope by decision or beyond the designed structures).
- Deferred P2+ items (Scientist 2026-11-21): group-orthogonal supersaturated designs, hard-to-change covariates, sphere packing, MSA, batched sample-size search.
- P3, A/I GPU timing (W23), real PyPI and Issue #13.
