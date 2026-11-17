# Week 11 goal — 2026-11-17 → 2026-11-24 (VISION.md v4 phase P2 completion, part 3: mixtures, combined structures, covariates and compound criteria)

**Auto-advance (Architect, 2026-11-17).** Week 10 goals C1–C3 were met on 2026-11-17: C1 22/22 FrF2 catalogue entries (#251, #257), C2 10/10 constructor cases matched to rsm and FrF2 (#258, #261, #262), C3 26 k ≤ 30 supported (#248). The published-DSD reference is NOT_RUN (Colab `ALLOCATION_FAILED`, #266). Held-out H21–H23 were recorded as-is by #279 (`6c6f108`). H22 and H23 match rsm 2.10.6 with Δ ≤ 2e-15. H21 TorchOED is resolution IV with A₄ = 7, and its FrF2 reference is NOT_RUN (the R generator call errored). Baseline: `main` at the START of this goal's commit task.

**Why this week:** these are the last open P2 rows. JMP's Mixture Design and Custom Design platforms cover constrained mixtures, mixture-process designs, covariates, blocks combined with hard-to-change factors, and alias-optimal designs. TorchOED has only the simplex lattice and no row below is complete. Week 12 is consolidation, so P2 closes with this week's rows complete or Scientist-deferred.

Every Engineer dispatch names the goal (D1–D3) it serves and updates its matrix row in the same PR. New public functions and keywords are decided in a gate first, with an independent fresh-context review, and are exported from `torchoed` with an API entry, a guide snippet and a CHANGELOG line. Structured and compound criteria reuse the existing exchange engine and certificate machinery. Do not add a second search loop.

| Goal | Falsifiable target | Baseline |
|---|---|---|
| **D1 Mixtures, complete** | Gate first (independent review): simplex centroid; constrained mixture regions from lower and upper bounds (extreme vertices by the McLean–Anderson 1966 enumeration, plus edge and face centroids and the overall centroid); optimal mixture designs D/A/I with Scheffé linear, quadratic and special-cubic models through `optimal_design` on those candidates, composing with `fixed_ids` and the certificate; mixture-process variables (Scheffé terms crossed with a process model). Oracles: (a) every returned vertex satisfies the bounds and `Σx = 1` exactly, and the vertex set equals an exhaustive `Fraction` enumeration over all bound pairs on ≥ 3 regions, including one with an inactive bound; (b) centroid counts match the closed form `2^q − 1` for the simplex centroid; (c) the D/A/I certificate floor holds on the constrained candidates. Measured number: filled cells on the two mixture rows (simplex centroid … Scheffé models; mixture-process variables), target **18/18** ✅ / N/A / NR. External: ≥ 2 published mixture examples (Cornell 2002, or R mixexp `Xvert` / AlgDesign `optFederov` output) matched on one shared metric, or NR | 0/18 cells (Docs ◐) |
| **D2 Combined structures and covariates** | Gate first (independent review): `blocks=` and `whole_plots=` in one call, whole plots nested within blocks (fixed block effects, random whole plots with variance ratio `eta`), for D, A and I, with `fixed_ids` augmentation, a dual efficiency floor and structure-aware diagnostics; covariates as fixed per-run values that the design cannot change (JMP Covariates), with the search choosing the remaining factors. Oracles: (a) with one block the combined information equals the W9 split-plot information exactly, and with singleton whole plots and `eta = 0` it equals the W9 blocked information; (b) exhaustive enumeration on ≥ 2 tiny instances finds the same optimum value; (c) covariate columns are returned unchanged and in their given run order. Measured number: filled cells on the rows `blocks + split-plot together` and `covariates`, target **18/18** ✅ / N/A / NR. External: skpr `gen_design` with a split-plot design and blocks, or NR | 0/18 cells |
| **D3 Compound and alias-aware criteria; criterion and power gaps** | Gate first (independent review): compound criteria as weighted sums of efficiency-normalised D, A and I (log-efficiency form; the gate PROVES concavity in M); alias-aware design (Jones and Nachtsheim 2011: minimise `tr(AᵀA)` of the alias matrix subject to a D-efficiency floor, or the weighted form the gate chooses); G and V as criterion functions beside D, A, E, c and I; an effect-group F-test in `design_diagnostics` power. Oracles: (a) compound with weight 1 on one criterion reproduces that criterion's design value exactly; (b) the compound certificate floor (equivalence theorem for the weighted criterion) holds on ≥ 2 instances; (c) G and V equal dense evaluation over the candidate set at 1e-12; (d) the F-test power equals SciPy's noncentral F at 1e-10 (dev-only oracle) and reduces to the existing t-test power for a one-term group. Measured number: filled cells on the row `alias-aware and compound criteria` (target **9/9**), plus the criterion-functions Crit cell and the power Diag cell (target both ✅) | 0/9 cells; Crit ◐; Diag ◐ |

**Held-out additions (frozen here, before any D1 or D2 code merges; never used for tests, tuning or dispatch choices; run once at week end, reported as-is):**
- **H24 (constrained mixture vertices):** the D1 constructor's extreme vertices for q = 3 with `0.1 ≤ x₁ ≤ 0.5`, `0.1 ≤ x₂ ≤ 0.7`, `0.05 ≤ x₃ ≤ 0.6`. Record the vertex count and the vertices. Shared metric: the vertex count and the Hausdorff distance between the two vertex sets. Reference: R mixexp `Xvert(nfac=3, lc=c(0.1, 0.1, 0.05), uc=c(0.5, 0.7, 0.6), ndm=0)` with `sessionInfo()`, re-scored by the same function.
- **H25 (optimal constrained mixture):** the TorchOED D-optimal 10-run design for the quadratic Scheffé model `x₁, x₂, x₃, x₁x₂, x₁x₃, x₂x₃` on the 13 candidates of the H24 region: its 6 vertices, its 6 edge midpoints and the overall centroid (mean of the vertices), with default restarts. Shared metric: `D = (det(FᵀF/10))^(1/6)` and the certificate floor. Reference: R AlgDesign `optFederov(~ -1 + x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3, data=<the same 13 candidates frozen as a CSV>, nTrials=10, criterion="D")` with `sessionInfo()`, re-scored by the same function.
- **H26 (blocks with whole plots):** the TorchOED D design for factors A (hard to change), B and C, each at ±1, with the model intercept, A, B, C, AB, AC, BC, in 16 runs: 2 blocks, each of 4 whole plots of 2 runs, `eta = 1`. Shared metric: the D value of the gate's combined information matrix and its dual efficiency floor, computed by one function for both designs. Reference: skpr `gen_design` with `splitplotdesign` and `blocksizes`, if it accepts both, with `sessionInfo()`, re-scored by the same function.

If a reference call fails or its versions are not recorded, its column is `NOT_RUN`, with the error verbatim.

**Timing rule (durable, Scientist 2026-11-07):** per restart unless a goal says otherwise.

**Safeguards:**
- Every gate passes an independent fresh-context review before code.
- Published tables and examples are cited by source and edition and frozen as repo data. Do not transcribe from memory.
- No claim of JMP parity beyond reproduced references.
- External references follow the VISION.md oracle rules.
- No tuning on held-out cases.
- Search changes rerun `tools/bench_b0.py` with no time regression and unchanged D results.
- The matrix row of each goal is updated with evidence links in the same PR.
- At most 3 feature goals.

**Out of scope this week:**
- Mixture ABCD and linear-constraint (non-box) mixture regions, unless the D1 gate covers them at no extra cost.
- Split-split-plot, strip-plot, hard-to-change covariates, minimum-aberration and blocked fractional factorials (week 15).
- Space-filling designs (week 13), Bayesian "If Possible" terms (week 14).
- Analysis and Studio (P2.5), GPU work, P3.
- Real PyPI and Issue #13.
