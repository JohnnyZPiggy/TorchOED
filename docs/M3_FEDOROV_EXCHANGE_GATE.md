# M3-A Exact-ID Fedorov Exchange — Mathematics and Architecture Gate

**SAE V3.3 Chief Architect + Scientific Mentor · 2026-10-29.** Status: **ARCHITECTURE_ONLY / ENGINEER_NOT_AUTHORIZED / CPU_ORACLES_SPECIFIED_NOT_RUN**. Baseline main `eba94865cd0bb82fd15e6d606976bcee1094750f`; M0/M1/M2-A merged. E0 PR #3 remains independent and unmerged. No M3 Engineer Issue, PR, CLAIM or implementation is authorized by this document.

Read together with [Scientific Mentor Research §§11–14](SCIENTIFIC_MENTOR_RESEARCH.md), [M1/M2 Numerical Contract](M1_M2_NUMERICAL_CONTRACT.md), [M2 API](M2_API.md), and [Release Roadmap](RELEASE_ROADMAP.md). Reuse the frozen SAE V3.3 pin in `sae.yaml`.

## 1. First slice: one bounded 1-swap exact D search

Input **feature** matrix `F[m,p]` (not physical coordinates), exact selected `ids[n]` of dtype int64, `p>=1`, `m>=n>=p`. Initial selected design must be SPD under the documented M2-A numerical policy, with `M=F[ids].T @ F[ids]`. Distinct candidate IDs may have equal coordinates/features and still be different candidates. The default has **no replacement**: `ids` unique, and an incoming `j` must be outside the selected index set. Output carries selected IDs, initial/final D logdet, accepted swap history, termination status, validation counts and whether a *full neighborhood* was checked. Do not identify candidates through approximate `torch.isclose`.

**Explicit repeat policy:** `allow_repeats=False` for M3-A; `allow_repeats=True` is **DESIGN_ONLY / NOT IMPLEMENTED** for this slice. In a future with-replacement model, `ids` is a multiset of independent observation *events*; the same candidate ID can be selected repeatedly, and exchanging a selected event with any candidate ID (including one already present) is legal. A same-ID replacement is a no-op and excluded as a true improvement. Distinct event position, candidate ID, and feature equality remain separate concepts. Do not silently carry over no-replacement neighborhood or mark replication as physical measurement independence under correlated noise.

## 2. Exact exchange identity, positivity and ordering

At a full-rank state with outgoing feature row `a=F[ids[i]]` and incoming `b=F[j]`:
`M'=M-aa^T+bb^T`, and for `u=a^TM^{-1}a`, `v=b^TM^{-1}b`, `c=a^TM^{-1}b`,
`Γ=(1-u)(1+v)+c²=det(M')/det(M)`, `Δ_D=log Γ` **only if M' is SPD**. Evaluate `u,v,c` via triangular solves using the SPD Cholesky factor, never a dense explicit inverse. `Γ<=0`, nonfinite, or doubtful values are **invalid/inconclusive**, never clamp `Γ` or insert ridge. For `n=p`, dropping any row temporarily makes a rank-deficient matrix, yet the **complete exchange can remain SPD**: never score using an inverse of `M-aa^T`.

Select **best improvement** over the entire legal `n×(m-n)` neighborhood (optional bounded candidate tiles, exact semantics). With deterministic tie-breaking: maximize *independently reverified* `Δ_D`; if ties within declared deterministic score tolerance, choose smallest outgoing **position i**, then smallest incoming **ID j**. Preserve initial ID order except for replacing `ids[i]`; no implicit sorting. Avoid accepting an exchange merely because approximate `Γ>1`. A numerical candidate needs independent direct `M'`/D check; on uncertain cases, use a higher-confidence QR/SVD reference or report `NUMERICAL_INCONCLUSIVE` and do not claim strict local optimality.

## 3. Monotonicity, stopping and certificate scope

- A valid accepted swap must have a positive **verified** logdet difference above a documented scale-aware acceptance threshold; monotonic D history strictly increasing under the declared floating policy. An arbitrary fixed 1e-12 log ratio does not certify improvement for all condition numbers.
- Terminate with `LOCAL_1_SWAP_CERTIFIED` **only** if every legal neighbor was evaluated *and* verified no admissible strict improvement under the declared error policy. With tolerance/rounding uncertainty and no rigorous interval bound, label `LOCAL_1_SWAP_NUMERICAL` or `NUMERICAL_INCONCLUSIVE`, not mathematical exact optimality. If `max_iter` exhausted, label `MAX_ITER`; if no legal moves, label `NO_LEGAL_SWAP`. These are distinct.
- Strict mathematical local condition for full neighborhood: `max_{T∈N1(S)} det M(T)/det M(S) <= 1` (including identity yields equality 1). A locally optimal design need **not** be globally D-optimal. No early stop from only top-K candidate screens with a global certificate.
- Deterministic search should not consume global RNG. Later optional starts must use an explicit generator with recorded seed and consume it predictably. M3-A first slice accepts a provided full-rank initial `ids`; full-rank initialization, multistart, modified Fedorov, true DETMAX excursions and k-swaps are **separately gated**. This avoids conflating mathematical search with a complex initializer.

## 4. Tiled gain calculations and safety

With Cholesky `M=LL^T`, compute `R=L^{-1}A^T` and `T=L^{-1}B^T` for eligible incoming B rows; `u_i=||R_i||²`, `v_j=||T_j||²`, `c_ij=R_i^T T_j`; form `Γ_ij` in bounded tiles rather than an unlimited `n×m` allocation. Explicit checks for `F` finite and rank/conditioning, index constraints, budget, scalar/batched shape policy and intermediate memory. M3-A **single-case CPU float64** initially; float32 comparison and batched cases proposed as later separate gates. Reuse M2-A D-optimality to check *score semantics*, but reference quality must come from an independent literal Gram determinant or QR/unique exhaustive enumeration, not the exact same optimized gain function.

**Critical warning:** M2-A has a conservative Cholesky-pivot threshold; a mathematically SPD but very ill-conditioned neighbor can be rejected. This must yield clear `NUMERICAL_INCONCLUSIVE`/policy invalidity, **not** proof that the neighbor has zero determinant. Neither float64 tolerances nor a heuristic `Γ` error estimate constitute a rigorous universal certificate.

## 5. Proposed independent CPU oracle M3-F1–F12

These specify **future acceptance**; no TorchOED M3 code executed in this architecture phase.

| Gate | Fixture | Expected independent invariant |
|---|---|---|
| F1 | `F=[[1,0],[0,1],[2,0]]`, `ids=[0,1]` | raw det 1; swap outgoing i=0 with j=2 gives det 4, Δ=log 4 |
| F2 | `F=[[1,-1],[1,1],[1,0]]`, `ids=[0,1]`, n=p=2 | outgoing i=0, incoming j=2: old det 4, new det 1; Γ=1/4 even though intermediate deletion is singular |
| F3 | Selected `ids=[0,1]` and candidate j=2 with `F[2]==F[0]` | j=2 is legal by ID (although same coordinates); no-replacement must not reject by `isclose`; gain 0 |
| F4 | Repeated IDs `ids=[0,0]` | reject for M3-A; future with-replacement semantics marked NOT_IMPLEMENTED, not silently accepted |
| F5 | `F` rank deficient or `n<p` | explicit invalid-initial-design outcome; never invent full-rank initialization |
| F6 | `F=[[1,0],[0,1],[2,0]]`, `ids=[0,1]` | best move deterministic (i=0,j=2), resulting ids `[2,1]`; no further strict improvement |
| F7 | two exact equal-gain incoming candidate IDs with identical feature rows | choose smallest incoming ID, independent of tile size; outgoing-position tie selects smallest i |
| F8 | analytic 2×2 SPD fixture; enumerate **all legal swaps**, compare Γ to directly computed determinant ratios | exact or declared float64 tolerance, including non-improving moves; no self-referential oracle |
| F9 | `n=p` saturated full-rank design | complete-swap calculation works despite rank-deficient delete-only intermediate |
| F10 | simple finite design with max_iter=0/1 | distinct MAX_ITER vs full-scan no-improvement status; accepted D history monotone |
| F11 | a frozen 4-candidate, n=2 local-trap fixture from Mentor §13 | verify an exact 1-swap local optimum that is not globally optimal; compare all `binom(4,2)=6` designs and keep global vs local distinct |
| F12 | singular and near-singular Gram variants; compare direct QR/Gram checks | singular swaps rejected, uncertain near-singular labeled inconclusive; zero hidden ridge; no false exact optimality certificate |

Add independent small exhaustive oracle: for `m<=8` and feasible `n`, enumerate all `binom(m,n)` **distinct ID subsets**, compute their Gram determinants using exact rational arithmetic for integer/rational F, and compare final search result with (a) best full-neighborhood neighbor and (b) global optimum. Global equality may fail when a strict 1-swap trap exists; this is a *required negative control*, not algorithm failure. Prove stopping invariants on the frozen discrete neighborhood before performance tuning.

## 6. M3 engineering acceptance design / authority

Future Scientist may dispatch a single bounded Codex Engineer Issue after rechecking live Issue/PR and frozen SAE COLD/CPU WARM. Engineer would CLAIM in its own worktree/branch, implement only M3-A ID-based best-improvement 1-swap CPU algorithm, preserve 12 named F gates plus full-neighborhood exhaustive reference, actual Python/torch/pytest versions, exact logs and failures, 1 Draft PR, SHA256SUMS, install/wheel/outside-checkout smoke and `ENGINEER_HANDOFF`. No self-merge, GPU/Actions, M3-B, modified Fedorov, DETMAX, multi-GPU or claimed speedup.

**Architecture acceptance now:** `M3A_MATH_SCOPE_AND_CPU_ORACLES_FROZEN / M3_ENGINEER_NOT_DISPATCHED / ALL_M3_CODE_TESTS_NOT_RUN`. Separately authorize Engineer execution and later Scientist merge.
