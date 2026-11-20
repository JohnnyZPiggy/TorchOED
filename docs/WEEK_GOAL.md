# Week 15 goal — 2026-11-20 → 2026-11-27 (P2+ part 3: split-split-plot and strip-plot, minimum-aberration and blocked fractional factorials, GLM power by simulation)

**Auto-advance (Architect, 2026-11-20).** Week 14 (P2+ part 2) goals were met on 2026-11-20:
- B1 "If Possible" terms, Bayesian D / I: every cell filled; Ext NR (DuMouchel–Jones 1994 full text not accessible verbatim, #449); guide (#444).
- B2 model-robust D over a set of models: every cell filled; Ext NR (Li–Nachtsheim 2000 full text not accessible verbatim, #449); guide (#448).
- B3 Compare Designs and sample-size explorer: both parity rows ✅ (#450, #455); skpr 1.9.2 power NO MATCH at 1e-10 as-is (R-side Δ).
- Published examples 14 → 16 (target ≥ 16).

Held-out H36–H39 were recorded as-is (#452, PR #456, `reports/w14-heldout`):
- H36 MATCH (det(XᵀX + K) = 62208 = exhaustive maximum over 296,010 subsets).
- H37 MATCH (tr((XᵀX + K)⁻¹W) = 649/324 = exhaustive minimum).
- H38 MATCH ((det M₁, det M₂) = (24576, 524288) attains the exhaustive maximum over 12,870 subsets).
- H39 TorchOED recorded (MaxPro ψ 23.66830193247027, minimum distance 0.3132072073220248); R reference NOT_RUN (the one dispatch ran before `S.csv` was committed). Completing the R reference once from the committed sample is a W14 evidence carry-over; the TorchOED values are not recomputed.

Baseline: `main` at the START of this goal's commit task.

**Why this week:** VISION.md P2+ (weeks 13–15) still has open parity rows dated W15: split-split-plot and strip-plot, minimum aberration and blocked fractional factorials, power by simulation for GLMs, orthogonal and covering arrays, hard-to-change covariates. At most 3 feature goals fit a week; the three chosen are the rows with an R reference (skpr, FrF2) and, for C3, a TorchOED signature (batched GPU Monte Carlo). Orthogonal / covering arrays and hard-to-change covariates are listed for a Scientist decision in the week-16 consolidation (deferral or a later P2+ week).

| Goal | Falsifiable target | Baseline |
|---|---|---|
| **C1 Split-split-plot and strip-plot** | `optimal_design` D, A and I for (a) split-split-plot: whole plots of given sizes, subplots of given sizes nested in them, very-hard factors constant within whole plots and hard factors constant within subplots, GLS information Xᵀ(I + η_w Z_wZ_wᵀ + η_s Z_sZ_sᵀ)⁻¹X; (b) strip-plot: an r × c grid of runs, row factors constant along rows and column factors constant down columns, information Xᵀ(I + η_r Z_rZ_rᵀ + η_c Z_cZ_cᵀ)⁻¹X. Existing split-plot results stay bit-identical. Exact oracles: exhaustive optimum on small cases in `Fraction`. Cert: dual floor vs the OLS approximate optimum (conservative, as for split-plot), or a reasoned N/A. Diag: GLS SPV/VIF and power with stated denominator df. Ext: one skpr `gen_design` split-split-plot case through `r-oracle.yml`, re-scored by the shared GLS criterion. Parity row "Split-split-plot, strip-plot" ❌ → ✅ via one new matrix row. Measured number: filled cells of that row, target **all** | ❌ (split-plot has one stratum only) |
| **C2 Minimum-aberration and blocked fractional factorials** | (a) A minimum-aberration search for k two-level factors in 2^q runs returning generators whose word-length pattern is lexicographically minimal over the generator sets searched, exact by `word_length_pattern`; (b) blocked regular fractions with 2^b blocks from block generators, with main effects not aliased with blocks, and an alias report of block-confounded effects. Exact oracles: exhaustive enumeration of generator sets on small sizes. Ext: FrF2 at the recorded version (`FrF2(..., blocks=)` and catalogue MA designs) through `r-oracle.yml`, compared by WLP and block alias sets. Matrix: the FF row and the parity row ◐ → ✅ for MA and blocked FF. Measured number: FrF2 cases MATCH out of the frozen set, target **all** | generators from a fixed catalogue only; no blocks |
| **C3 Power by simulation for GLMs (logistic, Poisson)** | A public function returning per-coefficient (and effect-group) Wald power for a given design, coefficient vector β, family `binomial` or `poisson`, α and number of simulations, from batched torch IRLS fits over all simulated responses at once; separated / non-converged fits counted and reported. Exact oracle: for a tiny design, power by full enumeration of all response vectors in float64, MC estimate within 4 standard errors. Ext: skpr `eval_design_mc` at the recorded version on frozen designs through `r-oracle.yml`, agreement within 4 combined MC standard errors. Dev: CPU float64 and one measured same-host GPU timing (one Colab task, #374 policy). Parity row "Power by simulation for GLMs" ❌ → ✅ via one new matrix row. Measured number: filled cells of that row, target **all** | ❌ (normal-theory power only) |

**Published examples (P2+ exit, ≥ 15 in total, met).** Each C goal adds at least one published example (an Rd `\examples` of skpr or FrF2 at the recorded version, or a paper table frozen as repo data). Measured number: published examples MATCH or TorchOED better, **16 → ≥ 19** this week.

**Held-out additions.** These are frozen here, before any week-15 task merges. They are never used for tests, tuning or dispatch choices. They run once at week end and are reported as-is. Every TorchOED call uses `generator=torch.Generator().manual_seed(0)` and default settings otherwise.
- **H40 (split-split-plot D):** candidates the 2³ grid on {−1, 1} in `full_factorial` order listed twice (m = 16, IDs 0–15; distinct IDs, so each (A, B, C) appears at most twice in a design), factors (A, B, C); F columns [1, A, B, C, AB, AC, BC]; n = 8; whole plots of sizes [4, 4] with A constant within each; subplots of sizes [2, 2, 2, 2] nested in them with B constant within each; C free; η_w = η_s = 1.
  - Shared metric: det(Xᵀ(I + Z_wZ_wᵀ + Z_sZ_sᵀ)⁻¹X) in exact `Fraction`, on the design as returned (run order and plot membership).
  - Reference: exhaustive enumeration of the 2² · 2⁴ · 2⁸ = 16,384 level assignments (A per whole plot, B per subplot, C per run), keeping those in which no (A, B, C) appears more than twice, dev-only. Verdict: MATCH if equal to the exhaustive maximum.
- **H41 (strip-plot D):** candidates (listed twice, m = 16) and F as in H40; n = 8 as a 4 × 2 grid (4 rows, 2 columns); A constant along each row, B constant down each column, C free per cell; η_r = η_c = 1.
  - Shared metric: det(Xᵀ(I + Z_rZ_rᵀ + Z_cZ_cᵀ)⁻¹X) in exact `Fraction`, on the design as returned.
  - Reference: exhaustive enumeration of the 2⁴ · 2² · 2⁸ = 16,384 level assignments (A per row, B per column, C per cell), keeping those in which no (A, B, C) appears more than twice, dev-only. Verdict: MATCH if equal to the exhaustive maximum.
- **H42 (minimum aberration):** TorchOED's C2 minimum-aberration call for k = 12 factors in 32 runs.
  - Shared metric: `word_length_pattern` on the returned ±1 design.
  - Reference: R `FrF2::FrF2(nruns = 32, nfactors = 12, randomize = FALSE)` through `r-oracle.yml` with `sessionInfo()`, exported as a ±1 CSV and scored by the same `word_length_pattern`. Verdict: MATCH if the WLPs are equal; TorchOED better if TorchOED's WLP is lexicographically smaller; otherwise NO MATCH with the first differing word length.

If a reference call fails or its versions are not recorded, its column is `NOT_RUN`, with the error verbatim.

**Timing rule (durable, Scientist 2026-11-07):** per restart unless a goal says otherwise.

**Safeguards:**
- Existing `optimal_design` results without the new arguments stay bit-identical, existing diagnostics are unchanged and existing `fractional_factorial` outputs are unchanged. If shared search code is touched, `tools/bench_b0.py` reruns with no time regression and identical D results.
- Every identity used is in a gate document with PROVED/OBSERVED/CONJECTURE marks and an independent math review before it is STARTed.
- Published examples are cited by source (paper table or Rd line) and frozen as repo data. External references follow the VISION.md oracle rules (R on `r-oracle.yml`; Colab only for GPU, one GPU task at a time, #374 budget policy).
- No tuning on held-out cases. Historical reports and evidence are preserved.

**Out of scope this week:**
- Orthogonal / Taguchi / covering arrays and hard-to-change covariates (Scientist decision requested in week 16: defer or schedule).
- REML analysis of split-split-plot experiments (P2.5); GLM design optimisation (P3, Bayesian / local designs); exact (non-simulation) GLM power.
- Group-orthogonal supersaturated designs, non-box mixture regions, new DSD sizes, analysis and Studio (P2.5).
- Real PyPI and Issue #13.
