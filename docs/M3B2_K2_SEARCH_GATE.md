# M3-B2 Mathematical Gate — Deterministic k=2 Exchange Search

**SAE V3.3 · Scientific Mentor + Chief Architect · 2026-10-30**

**Status: `M3B2_MATH_FROZEN / S_ORACLES_SPECIFIED / PRODUCTION_ENGINEER_NOT_DISPATCHED`.**

This gate is written against the following baseline:
- `main=cb944b0e7cf0a6cd1bfc99743a6e72bc69471ab4`.
- M3-B1 scoring is in Draft [PR #12](https://github.com/JohnnyZPiggy/TorchOED/pull/12) at HEAD `15f5970a…`. It is Architect-accepted but **not merged**.
- The scoring-kernel experiment M3-B2-P0 is [Issue #13](https://github.com/JohnnyZPiggy/TorchOED/issues/13) and is in progress.

This document creates no Issue, CLAIM, production code, GPU work or merge. It extends the [M3-B gate](M3B_ADVANCED_EXCHANGE_GATE.md) §7 step (ii) "bounded deterministic k=2 search". Notation follows that gate: `F[m,p]`, exact IDs, no replacement, fixed `n`, `m≥n≥p`, and `D(S)=logdet M(S)`.

Labels used below: **PROVED** means derived here and checked exactly. **OBSERVED** means seen in finite-precision or exhaustive tiny experiments, not a general theorem. **HYPOTHESIS** means a calibrated choice that must be confirmed by evidence.

## 1. Dispersion-matrix identities (PROVED)

For the current design `S` with `M=F_SᵀF_S=LLᵀ`, define `Z=L⁻¹Fᵀ` (shape `[p,m]`) and `G=ZᵀZ=F M⁻¹Fᵀ`. In other words, `G_ij=d(x_i,x_j)` and `G_jj` is the leverage/variance function.

Take outgoing IDs `O={a,b}⊂S` (positions `i1<i2`), incoming IDs `N={c,d}⊂Sᶜ`, `idx=(a,b,c,d)` and `J=diag(-1,-1,1,1)`.

1. **Gather form.** Since `M'=M+WJWᵀ` with `W=[F_Oᵀ,F_Nᵀ]`, the matrix determinant lemma gives `Γ₂=det M'/det M=det(I₄+J·G[idx,idx])`. Once G is available, each pair costs O(1), independent of p.
2. **Add-first Schur form.** `Γ₂=det(I₂+G_NN)·det(I₂−G_OO+G_ON(I₂+G_NN)⁻¹G_NO)`. The pivot `I₂+G_NN` is SPD for every design. This corresponds to the always-SPD intermediate state `n+2`.
3. **k=1 reduction.** For a single swap `i→j`: `Γ₁=(1+G_jj)(1−G_ii)+G_ij²`. This is the classical Fedorov Δ, so M3-A, B1 and B2 share one identity.
4. **Exact SPD equivalence.** `M'=F_TᵀF_T` is a Gram matrix, so `M'⪰0`. Hence **in exact arithmetic `Γ₂>0 ⇔ M'≻0`**.
   - This refines the M3-B gate §3 statement "ratio positive alone is insufficient". That statement remains true **for floating-point Γ̂**, which is never an SPD proof, but not in exact arithmetic.
5. **Delete-first Schur is prohibited.** The form `det(I−G_OO)·det(I+G_NN+G_NO(I−G_OO)⁻¹G_ON)` requires `I−G_OO` to be nonsingular. This fails whenever `n−2<p`: the n−2 remaining rows cannot span `R^p`.

**Exact evidence (warm start, `fractions.Fraction`, no torch).** The fixtures were the trap, B7 n6/m11, B8 saturated n=p=3, B9 duplicate rows, and a rank-deficient-trial fixture.
- Forms 1 and 2 equal the direct Gram ratio for **181/181** k=2 pairs.
- The k=1 Fedorov form is exact for **61/61** swaps.
- `Γ>0 ⇔ Sylvester-PD(M')` holds for **181/181** pairs, including 4 singular trials.

**float64 risk (OBSERVED).** Max relative error against exact ratios:

| design | cond₂ M | gather / add-first | delete-first |
|---|---|---|---|
| B7 n6 m11 p3 | 5.6e2 | 3.3e-16 / 4.3e-16 | 5.6e-16 |
| saturated n=p=3 | 1.7e2 | 2.5e-16 / 2.8e-16 | undefined 18/18 |
| monomial p=6 | 8.9e3 | 1.7e-12 / 1.7e-12 | undefined 588/588 |
| monomial p=8 | 5.1e5 | 7.4e-11 / 7.4e-11 | **1.8e-3**, undefined 1395/1620 |

The observed error is about `0.65·eps·cond₂M`. The M3-B1 tolerance `256·eps·p·max(…)·cond` keeps a margin of roughly 10³. This is a heuristic, not an interval bound.

## 2. Search algorithm: `k2_exchange` (VND, deterministic)

The public name must be explicit, for example `k2_exchange`. It must **not** be called `detmax`, and must not be confused with Modified Fedorov or a multi-level excursion.

```
S ← validated initial exact IDs (M2-A SPD policy, no ridge)
loop (bounded by max_iter accepted moves):
  Phase 1  (k=1, M3-A semantics): complete N₁ scan; accept the best verified strict improvement; repeat until none.
  Phase 2  (k=2): complete N₂ scan with a G-based scorer (§1.1 or §1.2, chosen from Issue #13 evidence).
           Candidate order: Γ̂ descending; exact ties and near-ties within the band Γ̂ ≥ Γ̂max − τ_band are broken
           lexicographically by (outgoing positions, incoming IDs), the M3-B1 row order.
           Verify up to verify_top candidates in that order by an independent full Gram + d_optimality:
             accept the first with direct gain ΔD > τ_acc; go to Phase 1.
           If no candidate has Γ̂ > 1+τ_band → stop LOCAL_2_NUMERICAL.
           If candidates exceed the threshold but all verify_top fail verification → stop NUMERICAL_INCONCLUSIVE.
  After every acceptance: recompute the Cholesky factor of M(S) from scratch (no rank updates in v1).
```

- **Acceptance uses the direct D difference, never `log Γ̂`.** Set `τ_acc` equal to the M3-B1 condition-aware tolerance evaluated at the trial. `τ_band` uses the same formula on the ratio scale.
- **Termination (PROVED).** Every accepted move strictly increases the directly verified D by more than `τ_acc>0`. There are finitely many n-subsets, and no subset can recur after a strict increase, so the loop terminates even without `max_iter`. `max_iter` remains a resource bound and has its own stop reason.
- **Locality semantics (PROVED).** `LOCAL_2_NUMERICAL` is reported only after complete N₁ **and** N₂ scans at the final state with no verified improvement. It means "no improving 1- or 2-swap within the numerical policy". It is **not** exact optimality and **not** global optimality.
- **Exact certificate.** `EXACT_LOCAL_2` is allowed only for tiny all-rational fixtures re-checked with exact arithmetic in tests. Production never emits it.
- **Dominance (PROVED).** Phase 1 is M3-A from the same start, so the `k2_exchange` endpoint D is at least the `fedorov_exchange` endpoint D (in exact arithmetic; numerically within tolerance).
- **When n<2 or q=m−n<2.** N₂ is empty. Stop with `NO_K2_NEIGHBORHOOD` after Phase 1. This is not an error.
- **Efficiency diagnostic (optional).** Because `Mbar=M/n`, we have `f_jᵀ Mbar⁻¹ f_j = n·G_jj`, and the gate §6 bound becomes `logdet M* − logdet M ≤ p·log(n·max_j G_jj / p)`, computed from diag(G) at no extra cost. Exact checks found 0 violations over 2033+ starts; it is tight at 2/6 trap starts. It is a bound, never an achieved-optimality claim.
- **Stop reasons:** `LOCAL_2_NUMERICAL`, `NUMERICAL_INCONCLUSIVE`, `MAX_ITER`, `NO_K2_NEIGHBORHOOD`, `NO_LEGAL_SWAP`.
- **Returned record:** history of `(k, outgoing positions, outgoing IDs, incoming IDs, Γ̂, direct ΔD, tolerance)`, counts of scanned pairs per phase, verification attempts, and `certificate ∈ {LOCAL_2_NUMERICAL, NOT_CERTIFIED}`.

## 3. Complexity and memory

- **Per accepted state:** one Cholesky `O(p³)`, `Z=L⁻¹Fᵀ` `O(mp²)`, and the G blocks. `G_SS[n,n]` and `G_SN[n,q]` cost `O((n²+nq)p)`. `G_NN[q,q]` costs `O(q²p)` and dominates memory.
- **Phase 2:** `C(n,2)·C(q,2)` gathers of a 4×4 matrix and their determinants. Verification costs `O(verify_top·(np²+p³))`.
- **Tiling:** tile `G_NN` by incoming-ID blocks under `max_elements`.
- **Scale example:** q=10⁴ in float64 needs about 800 MB for a full `G_NN`, so tiling is mandatory.
- **Scope of claims:** no wall-clock or GPU claims before measured evidence (Issue #13 is PROBE_ONLY).

## 4. Search oracles S1–S8 (NOT_RUN in production; exact references below)

The reference is an exact-rational VND using `Fraction` and `itertools`. It neither imports nor reuses TorchOED. Counts below are OBSERVED with that reference, over **all** starting n-subsets with `det>0`.

| Gate | Fixture | Falsifiable expectation |
|---|---|---|
| S1 | Four-point trap, start {0,1} | Phase 1 makes no move; Phase 2 accepts {0,1}→{2,3} with Γ=6561/2500; stop `LOCAL_2_NUMERICAL` at the global optimum (checked against all six 2-subsets) |
| S2 | `F=[[2,1,-2],[0,1,0],[-2,0,2],[-2,0,-2],[-1,1,-1],[-2,2,0],[1,0,-1],[1,-2,-1],[1,2,-2],[-1,2,-2]]`, n=4 | Over all **205** valid starts: 1-swap reaches the global optimum **111×**; 1-local but not 2-local endpoints **34**; VND reaches the global optimum **145×**; every VND endpoint is exactly 2-local; the remaining 60 are a documented **negative control** (2-local ≠ global), not a failure |
| S3 | Polynomial controls: integer11 quadratic n4, integer9 cubic n5, 3×3 grid interaction n5 | 1-swap is already global from all 330 / 126 / 117 starts; VND makes **zero** k=2 moves and must not degrade |
| S4 | Any start | Endpoint D ≥ `fedorov_exchange` endpoint D from the same start; verified D strictly increasing along the history |
| S5 | Injected verification failure (scorer stub returns an inflated Γ̂ for the top candidates) | Falls through to the next candidate, then `NUMERICAL_INCONCLUSIVE`; never accepts an unverified move |
| S6 | Ties: B9 duplicate feature rows with distinct IDs | Deterministic lexicographic choice; IDs not deduplicated by coordinates; identical histories across tile sizes |
| S7 | n=p saturated, n<2, q<2 | Add-first/gather scoring stays defined; `NO_K2_NEIGHBORHOOD` path covered; no deletion-first factorization |
| S8 | Efficiency bound | `opt/det M(S) ≤ (n·max G_jj/p)^p` at every start of S1–S3 (0 violations observed) and equality at 2/6 trap starts |

Aggregate random exhaustive evidence (OBSERVED, exact):
- **m10 n4 p3, integers ±2, 10 fixtures, 2033 starts:** 1-swap global 1939; traps 34; VND global 1973.
- **m8 n2 p2, 30 fixtures, 791 starts:** traps 38; VND global 791/791. This is trivially global because when n=k=2, N₂ covers every n-subset.
- **m9 n3 p3, 769 starts:** no traps.

Conclusion: 1-swap traps exist but are rare in tiny fixtures. k=2 escapes all of them, yet still leaves non-global 2-local endpoints.

## 5. Decisions and dependencies (Architect)

- **Scorer.** Choose between §1.1 gather4 and §1.2 add-first from the Issue #13 P2/P3/P6 evidence. If the error and ordering are equivalent, prefer the simpler gather4. Keep `score_two_swaps` (M3-B1) as the independent verification and test oracle; it is never the search hot path.
- **Production dispatch prerequisites.** All three are required:
  1. PR #12 merged (Scientist);
  2. Issue #13 Architect-accepted on CPU float64;
  3. this gate merged.
- **GPU.** GPU execution of the search remains outside M3-B2 v1 until CPU semantics are accepted. Device-agnostic code (`F.device`) is required so that a later probe needs no rewrite.
- **Out of scope for M3-B2:** Modified Fedorov (B3), multi-level DETMAX excursions (B5), k≥3, random restarts, rank-update factor maintenance and with-replacement designs. Each needs its own gate.

**Current gate:** `M3B2_MATH_FROZEN / S1–S8_SPECIFIED_NOT_RUN / ENGINEER_NOT_DISPATCHED`.
