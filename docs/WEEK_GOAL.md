# Week 13 goal — 2026-11-19 → 2026-11-26 (P2+ part 1: space-filling family — maximin and MaxPro LHS, uniform designs and space-filling diagnostics, fast flexible filling on constrained regions)

**Auto-advance (Architect, 2026-11-19).** Week 12 (consolidation, P2 exit) goals were met on 2026-11-19:
- E1 published-example matches: 8/13 (target ≥ 8) in `reports/w12-published`; 1 NO MATCH (alg-a), 4 NR to be re-run through `r-oracle.yml`.
- E2 matrix, parity checklist and documentation refresh: stale rows and doc debt 0; open P2 ◐ cells not listed 0 (4 listed: blocking and split-plot Dev → GPU program #374 G2; DSD Ext and mixture Con await Scientist deferral).
- E3 0.8.0 on TestPyPI: 4/4.

Held-out H30–H32 were recorded as-is (#368, PR #381):
- H30 MATCH (exhaustive `Fraction`, Δ = 0).
- H31 MATCH (FrF2 `pb(28)`, R 4.5.2 / FrF2 2.3-5 on `r-oracle.yml` run 44536661878).
- H32 NO MATCH: TorchOED's blocked `det(M_B)^(1/6)` 3.5728152202037835 vs AlgDesign `optBlock` 3.5725288795676673 (Δ = +2.86e-4, TorchOED higher).

Baseline: `main` at the START of this goal's commit task.

**Why this week:** VISION.md P2+ (weeks 13–15) asks for every remaining parity-checklist row, starting with the space-filling family. TorchOED today has only random LHS and Sobol (`lhs(mode="random")`, `sobol`). The parity row "Space filling: LHS (maximin), MaxPro, uniform, sphere packing, fast flexible filling (constrained)" is ◐ and dated W13. The P2+ exit also asks for ≥ 15 published JMP/R examples matched or beaten in total (8 today).

| Goal | Falsifiable target | Baseline |
|---|---|---|
| **F1 Maximin and MaxPro Latin hypercubes** | `lhs(..., mode="maximin")` (Morris–Mitchell φ_p) and `lhs(..., mode="maxpro")` (Joseph–Gul–Ba ψ) on midpoint levels, searched by batched within-column swaps (all C(n, 2) swaps of a column in one O(n³) evaluation from cached distances). Exact oracles: exhaustive optimum on small cases. External: R `lhs::maximinLHS` and `MaxPro::MaxProLHD` through `r-oracle.yml`, re-scored by one shared TorchOED metric. One new matrix row `space filling: maximin / MaxPro LHS` with every cell filled with evidence or reasoned N/A. Measured number: filled cells of that row, target **all** | 0 (row does not exist; `lhs` raises for any mode but `"random"`) |
| **F2 Uniform designs and space-filling diagnostics** | One public diagnostics function returning, for any design in [0, 1]^k: minimum pairwise distance, φ_p, MaxPro ψ and the centred L2 discrepancy CD2 (Hickernell). `lhs(..., mode="uniform")` minimises CD2 by the same swap search. External: R `DiceDesign::discrepancyCriteria` values on frozen designs agree to 1e-12. One matrix row `space filling: uniform designs and diagnostics`. Measured number: filled cells of that row, target **all** | 0 |
| **F3 Fast flexible filling on constrained regions** | A space-filling design on a region defined by the existing linear constraints / disallowed combinations: a large feasible sample (Sobol filtered by `feasible_candidates`), batched k-means clustering in torch, and one design point per cluster chosen by the MaxPro criterion (the JMP "Fast Flexible Filling" construction). External: R `support::sp` (support points, Mak and Joseph 2018) on the same frozen feasible sample, re-scored by the shared diagnostics of F2. One matrix row `space filling: fast flexible filling (constrained)`. Measured number: filled cells of that row, target **all** | 0 |

**Published examples (P2+ exit, ≥ 15 in total).** Each F goal adds at least one published example (Rd `\examples` of `lhs`, `MaxPro`, `DiceDesign` or `support` at the recorded version) to the index, and the 4 NR rows of `reports/w12-published` are re-run once through `r-oracle.yml`. Measured number: published examples MATCH or TorchOED better, **8 → ≥ 12** this week.

**Held-out additions.** These are frozen here, before any week-13 task merges. They are never used for tests, tuning or dispatch choices. They run once at week end and are reported as-is. Points are on midpoint levels `(r + 0.5)/n`, r = 0 … n − 1; the shared metrics first map each column to ranks, so they do not depend on a reference's scaling.
- **H33 (maximin LHS, n = 10, k = 2):** TorchOED `lhs(10, 2, mode="maximin", generator=torch.Generator().manual_seed(0))` with default settings.
  - Shared metric: the minimum pairwise squared distance on integer ranks (exact `int`), and the number of pairs attaining it.
  - Reference: exhaustive enumeration of all 10! = 3,628,800 second-column permutations (first column fixed to the identity), dev-only, in the report. Verdict: MATCH if the minimum distance equals the exhaustive maximum.
- **H34 (MaxPro LHS, n = 20, k = 3):** TorchOED `lhs(20, 3, mode="maxpro", generator=torch.Generator().manual_seed(0))` with default settings.
  - Shared metric: ψ = (C(n, 2)⁻¹ Σ_{i<j} 1/Π_l (x_il − x_jl)²)^{1/k} on midpoint levels, computed by one function for both designs, and the minimum pairwise distance.
  - Reference: R `set.seed(20261203); MaxPro::MaxProLHD(n = 20, p = 3)$Design` through `r-oracle.yml` with `sessionInfo()`, re-scored by the same function. Verdict: MATCH or TorchOED better if ψ_TorchOED ≤ ψ_R · (1 + 1e-9); otherwise NO MATCH with Δ.
- **H35 (uniform LHS, n = 7, k = 2):** TorchOED `lhs(7, 2, mode="uniform", generator=torch.Generator().manual_seed(0))` with default settings.
  - Shared metric: CD2² in exact `Fraction`.
  - Reference: exhaustive exact `Fraction` enumeration of CD2² over all 7! = 5,040 second-column permutations (first column fixed to the identity), dev-only, in the report. Verdict: MATCH if equal.

If a reference call fails or its versions are not recorded, its column is `NOT_RUN`, with the error verbatim.

**Timing rule (durable, Scientist 2026-11-07):** per restart unless a goal says otherwise.

**Safeguards:**
- `lhs(mode="random")` output stays bit-identical for every seed; `sobol` is unchanged. No change to `optimal_design` search; if shared search code is touched, `tools/bench_b0.py` reruns with no time regression and identical D results.
- Every identity used is in a gate document with PROVED/OBSERVED/CONJECTURE marks and an independent math review before it is STARTed.
- Published examples are cited by source line and frozen as repo data. External references follow the VISION.md oracle rules (R on `r-oracle.yml`; Colab only for GPU, one GPU task at a time, #374 budget policy).
- No tuning on held-out cases. Historical reports and evidence are preserved.

**Out of scope this week:**
- Sphere packing beyond maximin, sliced LHS (`SLHD`) and nested designs (backlog unless the Scientist asks).
- Bayesian "If Possible" terms (week 14); split-split-plot, strip-plot, minimum-aberration and blocked fractional factorials, orthogonal and covering arrays, GLM power and Compare Designs (week 15).
- Non-box mixture regions; new DSD sizes; analysis and Studio (P2.5); P3.
- Real PyPI and Issue #13.
