# Week 14 goal — 2026-11-20 → 2026-11-27 (P2+ part 2: "If Possible" terms with Bayesian D and I, model-robust designs over a set of models, Compare Designs and sample-size explorer)

**Auto-advance (Architect, 2026-11-20).** Week 13 (P2+ part 1, space-filling family) goals were met on 2026-11-20:
- F1 maximin / MaxPro LHS: every cell filled with evidence or reasoned N/A. Ext ◐ as-is: R `lhs` 1.3.0 `maximinLHS` TorchOED better; `MaxPro` 4.1-2 `MaxProLHD` 8 × 3 NO MATCH (ψ 12.246 vs 11.120) (#404).
- F2 uniform designs and diagnostics: every cell filled; DiceDesign 1.10 / MaxPro 4.1-2 diagnostics 16/16 agree ≤ 1e-12 (#404).
- F3 fast flexible filling (constrained): every cell filled except HO, which is frozen below as H39 (carried over by design).
- Same-host CPU vs NVIDIA L4 measurements for all three rows (#422, #428); published examples 8 → 14 (target ≥ 12).

Held-out H33–H35 were recorded as-is (#426, PR #430):
- H33 MATCH (minimum squared rank distance 10 = exhaustive maximum over 10! permutations).
- H34 MATCH or TorchOED better (MaxPro ψ 22.786 vs R `MaxProLHD` 23.575; R 4.5.2 / MaxPro 4.1-2 on `r-oracle.yml` run 44676636724).
- H35 MATCH (CD2² = 14095/2420208 = exhaustive minimum over 7! permutations).

Baseline: `main` at the START of this goal's commit task.

**Why this week:** VISION.md P2+ (weeks 13–15) names JMP "If Possible" terms (Bayesian D/I, DuMouchel–Jones) and robust model-uncertain designs; the parity row "If Possible terms, Bayesian D/I" is ❌ and dated W14. Compare Designs and the sample-size explorer (dated W15) are pulled forward because they reuse the existing diagnostics, which leaves week 15 for split-split-plot and strip-plot, minimum-aberration and blocked fractional factorials, orthogonal and covering arrays, and GLM power by simulation. The P2+ exit asks for ≥ 15 published examples (14 today).

| Goal | Falsifiable target | Baseline |
|---|---|---|
| **B1 "If Possible" terms: Bayesian D and I** | `optimal_design(F, n, criterion="D" or "I", if_possible=<column indices>, tau=1.0)`: maximise log det(XᵀX + K/τ²) (DuMouchel and Jones 1994) or minimise tr((XᵀX + K/τ²)⁻¹ W), W = FᵀF/m (Jones, Lin and Nachtsheim 2008), K = diagonal with 1 on the `if_possible` columns and 0 elsewhere. The potential columns are first scaled as DuMouchel–Jones prescribe (residual on the primary columns over the candidates, then scaled to max − min = 2); this is idempotent on columns already in that form. Requires m ≥ n ≥ number of primary columns, so n may be below the total column count. Exact oracles: exhaustive optimum on small cases. Cert: equivalence-theorem dual floor for the concave Bayesian criterion, or a reasoned N/A. Ext: one published example (DuMouchel–Jones 1994 or Jones–Lin–Nachtsheim 2008) frozen as repo data and re-scored by the shared criterion, else NR with reason. One new matrix row `"If Possible" terms: Bayesian D / I`. Measured number: filled cells of that row, target **all** | 0 (row does not exist; `optimal_design` has no `if_possible`) |
| **B2 Model-robust designs over a set of models** | `optimal_design(F, n, criterion="D", models=[<column lists>], model_weights=None)`: maximise Σ_m w_m (1/p_m) log det(F_mᵀF_m) over the models' column subsets (weighted geometric mean of D-efficiencies; Läuter 1974, Li and Nachtsheim 2000), equal weights by default. Exact oracles: exhaustive optimum on small cases. Cert: per-model D-efficiency floor from the existing dual bound, or a reasoned N/A. Ext: one published model-robust example (Li and Nachtsheim 2000) frozen as repo data, else NR with reason. One new matrix row `model-robust D over a set of models`. Measured number: filled cells of that row, target **all** | 0 |
| **B3 Compare Designs and sample-size explorer** | One public `compare_designs(F, designs, ...)` returning, for 2 or more designs on the same model, side-by-side D, A, I and G efficiencies relative to the first design, per-coefficient power and fraction-of-design-space quantiles, all from existing functions. Power over a list of run sizes in one batched call (sample-size explorer). External: R `skpr::eval_design` power on frozen designs at several n, through `r-oracle.yml`, agree ≤ 1e-10. Parity rows "Compare Designs" ❌ → ✅ and "Sample size explorer" ◐ → ✅, each with every applicable cell filled. Measured number: filled cells of the two rows, target **all** | Compare Designs ❌; power at fixed n only |

**Published examples (P2+ exit, ≥ 15 in total).** Each B goal adds at least one published example to the index (a paper table frozen as repo data, or an Rd `\examples` of `skpr` at the recorded version). Measured number: published examples MATCH or TorchOED better, **14 → ≥ 16** this week.

**Held-out additions.** These are frozen here, before any week-14 task merges. They are never used for tests, tuning or dispatch choices. They run once at week end and are reported as-is. Every TorchOED call uses `generator=torch.Generator().manual_seed(0)` and default settings otherwise.
- **H36 (Bayesian D, If Possible):** candidates the 3³ grid on {−1, 0, 1} in `full_factorial` order; F columns [1, x1, x2, x3, x1x2, x1x3, x2x3]; `if_possible` = columns 4, 5, 6; τ = 1; n = 6.
  - Shared metric: det(XᵀX + K) in exact `Fraction`.
  - Reference: exhaustive enumeration of all C(27, 6) = 296,010 subsets, dev-only, in the report. Verdict: MATCH if equal to the exhaustive maximum.
- **H37 (Bayesian I, If Possible):** candidates the 3³ grid as in H36; F columns [1, x1, x2, x3, 2(x1² − 2/3), 2(x2² − 2/3), 2(x3² − 2/3)] (already in DuMouchel–Jones form); `if_possible` = columns 4, 5, 6; τ = 1; n = 6; W = FᵀF/27.
  - Shared metric: tr((XᵀX + K)⁻¹ W) in exact `Fraction`.
  - Reference: exhaustive enumeration of all 296,010 subsets (subsets with singular XᵀX + K excluded). Verdict: MATCH if equal to the exhaustive minimum.
- **H38 (model-robust D):** candidates the 2⁴ grid on {−1, 1} in `full_factorial` order; F columns [1, x1, x2, x3, x4, x1x2, x3x4]; models [0, 1, 2, 3, 4] and [0, 1, 2, 3, 4, 5, 6], equal weights; n = 8.
  - Shared metric: (1/2)[(1/5) log det M₁ + (1/7) log det M₂], with det M₁ and det M₂ as exact integers.
  - Reference: exhaustive enumeration of all C(16, 8) = 12,870 subsets (subsets with a singular model excluded). Verdict: MATCH if (det M₁, det M₂) attains the exhaustive maximum of the shared metric.
- **H39 (fast flexible filling, constrained; W13 F3 carry-over):** TorchOED `fast_flexible_filling(15, 3, A=torch.tensor([[1., 1., 1.]]), b=torch.tensor([1.5]))` with default settings (seed 0, 2¹⁴ candidates).
  - Shared metric: MaxPro ψ and minimum distance from `space_filling_diagnostics` on the original coordinates, one function for both designs.
  - Reference: R `set.seed(20261127); support::sp(n = 15, p = 3, dist.samp = S, par.flg = FALSE)$sp` through `r-oracle.yml` with `sessionInfo()`, where S is the same feasible sample (scrambled Sobol 2¹⁴ × 3, seed 0, filtered by x1 + x2 + x3 ≤ 1.5), frozen as a repo CSV. Verdict: MATCH or TorchOED better if ψ_TorchOED ≤ ψ_R · (1 + 1e-9); otherwise NO MATCH with Δ.

If a reference call fails or its versions are not recorded, its column is `NOT_RUN`, with the error verbatim.

**Timing rule (durable, Scientist 2026-11-07):** per restart unless a goal says otherwise.

**Safeguards:**
- `if_possible=None` and `models=None` keep every existing `optimal_design` result bit-identical; existing diagnostics are unchanged. If shared search code is touched, `tools/bench_b0.py` reruns with no time regression and identical D results.
- Every identity used is in a gate document with PROVED/OBSERVED/CONJECTURE marks and an independent math review before it is STARTed.
- Published examples are cited by source (paper table or Rd line) and frozen as repo data. External references follow the VISION.md oracle rules (R on `r-oracle.yml`; Colab only for GPU, one GPU task at a time, #374 budget policy).
- No tuning on held-out cases. Historical reports and evidence are preserved.

**Out of scope this week:**
- Split-split-plot and strip-plot, minimum-aberration and blocked fractional factorials, orthogonal and covering arrays, GLM power by simulation, hard-to-change covariates (week 15).
- Bayesian designs for nonlinear models and minimax designs (P3); sphere packing beyond maximin, sliced LHS.
- Non-box mixture regions; new DSD sizes; analysis and Studio (P2.5).
- Real PyPI and Issue #13.
