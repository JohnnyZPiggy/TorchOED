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

### 1.3 Two-swap pruning bound (PROVED; added 2026-10-31 for B3)

For outgoing `O={a,b}` define, using only gather determinants of `I+J·G` (no inverse):
- `ρ(O)=det(I₂−G[O,O])` — the ratio for deleting a and b;
- `w_O(j)=det(I₃+diag(−1,−1,1)·G[(a,b,j),(a,b,j)])` for each `j∈Sᶜ` — the ratio for the 3-point exchange `{a,b}→{j}`.

**Theorem.** For every incoming pair `N={c,d}`: `ρ(O)·Γ₂(O→N) ≤ w_O(c)·w_O(d)`, and `ρ(O)≥0`, `w_O(j)≥0`.

*Proof.* `G[O,O]=F_O M⁻¹F_Oᵀ ⪯ I` because `F_OᵀF_O ⪯ M`, so `ρ≥0`; `w_O(j)` is a determinant ratio of the Gram matrix `M−f_af_aᵀ−f_bf_bᵀ+f_jf_jᵀ ⪰ 0`, so `w≥0`. If `ρ=0` the claim is `0 ≤ w_cw_d`. If `ρ>0`, `M_r=M−F_OᵀF_O ≻ 0` and `Γ₂=ρ·det(I₂+H)` with `H=F_N M_r⁻¹F_Nᵀ ⪰ 0`. By Woodbury, `H_jj=G_jj+G_Ojᵀ(I−G_OO)⁻¹G_Oj`, and the Schur form of the 3×3 determinant gives `w_O(j)=ρ(1+H_jj)`. Hadamard's inequality `det(I₂+H) ≤ (1+H_cc)(1+H_dd)` completes the proof; equality iff `H_cd=0`. ∎

**Use (search only, never as a score).** For each O with `ρ(O) ≥ ρ_min`, a pair N with `w_O(c)·w_O(d) ≤ (1+τ/2)·ρ(O)` has `Γ₂ ≤ 1+τ/2` and is not eligible, so it may be skipped without computing its 4×4 determinant. Sort `w_O` descending; the surviving pairs form a prefix-bounded set enumerable by a two-pointer scan. Every surviving pair is still scored by the exact gather4 determinant of §1 item 1, so eligibility, order and verification are unchanged. Cost: `O(n²q)` for all `w_O` (same order as Phase 1) plus `O(n² q log q)` sorting, instead of `C(n,2)·C(q,2)` 4×4 determinants.

**Exact evidence (Architect, Fraction, 2026-10-31).** 54 random integer fixtures (p=3; m=9,n=5 and m=8,n=3), 2490 two-swaps: identity `Γ₂=det(I₄+JG)` exact, `ρ·Γ₂ ≤ w_c·w_d` **0 violations**, 847 equalities, 115 outgoing pairs with `ρ=0` exactly.

**Floating-point guard (HYPOTHESIS).** Like delete-first (§1 item 5), the bound degenerates as `ρ→0`: when `n−2<p`, `ρ=0` exactly and float64 `ρ≈1e−17` makes the test meaningless. Apply the bound only when `ρ(O) ≥ ρ_min = 1e−2`; otherwise scan all pairs of that O exactly. In float64 at 1-local states this kept the following fraction of 4×4 determinants (OBSERVED, NumPy reference, 3 seeds each unless noted): random500 p=10 n=20 **2.6e−3**; grid3 p=10 n=20 1.0e−3–6.8e−3; grid4 p=15 n=30 2.7e−5–9.2e−5; saturated n=p (grid3 n=10, grid4 n=15, random500 n=10) **1.0** (no pruning, as the theorem predicts). Without the guard, the n=p=10 random500 case gave a spurious, noise-driven pruning pattern; the guard is required.

**Open.** No pruning bound is known here for `p<n<p+2`, that is `n=p+1` (there `ρ=0` too). That is a Mentor question, not an Engineer task. The saturated case `n=p` is covered by §1.4.

### 1.4 Saturated two-swap closed form and bound (PROVED; added 2026-10-31 for B5)

Let `n=p` and `M(S)≻0`. Then `F_S` is square and invertible, `M⁻¹=F_S⁻¹F_S⁻ᵀ`, and `G[S,S]=I_p` exactly. Moreover `G[S,j]=F_S⁻ᵀf_j` is the coordinate vector of `f_j` in the basis of the selected rows.

**Identity.** For outgoing positions `O={a,b}` and incoming `N={c,d}`: `Γ₂(O→N) = (G_ac·G_bd − G_ad·G_bc)²`.
*Proof.* Replacing rows a,b of `F_S` by `f_c,f_d` multiplies `det F_S` by the 2×2 minor of `F_S⁻ᵀ[f_c f_d]` on rows a,b (Cramer), and `Γ₂ = det(F_{S'})²/det(F_S)²`. Equivalently, with `G_OO=I`, `G_ab=0` the 4×4 block `I+J·G` is `[[0,−X],[Xᵀ,I+G_NN]]` with `X=G[O,N]`, whose determinant is `det(X)²`. ∎ The 1-swap ratio is likewise `G_aj²`.

**Bound.** With `u_O(j)=G_aj²+G_bj²`: `Γ₂(O→{c,d}) ≤ u_O(c)·u_O(d)` (Hadamard's inequality on the columns of `X`). ∎ This is exactly §1.3 with `ρ:=1` and `w_O:=u_O`. So the same sorted-prefix enumeration applies: skip pairs with `u_O(c)·u_O(d) ≤ 1+τ/2`, which are never eligible, and score all survivors exactly by the existing gather4 determinant.

**Exact evidence (Architect, Fraction, 2026-10-31).** 30 random integer fixtures (p∈{3,4}, m=p+5), every nonsingular saturated start, 5190 two-swaps: `G_SS=I` exact in all of them, identity **0 mismatches**, bound **0 violations**.

**Floating point (HYPOTHESIS).** The rule is selected structurally (`n==p`), never by testing `ρ`. In float64, `G_SS=I+E` with `|E|` at roundoff, and the τ/2 margin is meant to absorb it. Survivors are still scored by the 4×4 determinant and verified directly, so a roundoff error can only change *which* near-threshold pairs are skipped. An S9-style comparison against the public `score_two_swaps` is the required check.

**OBSERVED (NumPy float64 reference, random500-like pool p=10 n=10, 5 seeds, every Phase-2 state reached by 1-swap then best-2-swap descent).** Fraction of `C(n,2)·C(q,2)` pairs surviving: **2.2e−6 to 3.8e−4**, including states where an improving 2-swap exists (best Γ₂ up to 1.41, always among the survivors).

### 1.5 Closed-form survivor score (PROVED; added 2026-10-31 for B6)

Notation is that of §1.3, with `x=G[a,·]` and `y=G[b,·]` restricted to `Sᶜ` (already formed as `Gsn` rows). For incoming `c,d`, define
`v_O(c,d) = ρ·G_cd + (1−G_bb)·x_c x_d + (1−G_aa)·y_c y_d + G_ab·(x_c y_d + y_c x_d)`.

**Identity (ρ(O) > 0).** `Γ₂(O→{c,d}) = (w_O(c)·w_O(d) − v_O(c,d)²) / ρ(O)`.
*Proof.* From the §1.3 proof, `Γ₂ = ρ·det(I₂+H)`, `ρ(1+H_jj) = w_O(j)`, and by Woodbury `ρ·H_cd = ρ·G_cd + g_cᵀ adj(I−G_OO) g_d = v_O(c,d)`, with `g_j=(G_aj,G_bj)` and `adj(I−G_OO)=[[1−G_bb, G_ab],[G_ab, 1−G_aa]]`. Expanding the 2×2 determinant gives `ρ·det(I+H) = (w_c w_d − v²)/ρ`. ∎ The Hadamard bound of §1.3 is this identity with `v` dropped.

**Saturated form (n = p, §1.4).** With `ρ:=1`, `w:=u`, `v_sat(c,d) = G_ac G_ad + G_bc G_bd`: `Γ₂ = u_c u_d − v_sat²`, which equals `(G_ac G_bd − G_ad G_bc)²` by Lagrange's identity. ∎

**Cost.** Per survivor, `G_cd = Z_cᵀZ_d` costs O(p) and the rest is O(1), using values already formed for pruning. The gather4 path costs a 4×p gather, a 4×4 Gram matrix and a batched 4×4 LU.

**Exact evidence (Architect, Fraction, 2026-10-31).**
- General form: 20 random integer fixtures (m=10, p=3, n=5), 1880 two-swaps with ρ>0. The identity against `det M(S')/det M(S)` had **0 mismatches**.
- Saturated form: 30 fixtures (p∈{3,4}, m=p+5, n=p), 1170 two-swaps, **0 mismatches**.

**Floating point (OBSERVED, NumPy float64; HYPOTHESIS for torch).**
- Accuracy: on random500-like pools (p=10, n=20, 3 seeds, every 1-swap descent state, highest-bound pairs), the closed form agreed with the direct `det M(S')/det M(S)` to a max relative difference of **6.1e−14**. That is far below the `_tolerance` ratio band (≥ 128·ε·p ≈ 2.8e−13 before score/condition scaling). The subtraction `w_c w_d − v²` loses at most a factor `w_c w_d/(ρ·Γ₂)`. That factor is bounded on survivors, since `ρ ≥ ρ_min = 1e−2` on the guarded path.
- Speed: on 1e5 synthetic survivors (p=10), NumPy took 59.5 ms for gather4 and 16.8 ms for the closed form. This is not a torch measurement.

**Use.** The closed form may replace the gather4 determinant **only for pruned survivors** (guarded `ρ ≥ ρ_min` and saturated paths). Unpruned outgoing pairs keep gather4. Eligibility, `_keep_top` ordering and exact verification of the top 8 are unchanged. The required check is S9-style agreement with the public `score_two_swaps`.

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
| S9 | Pair bound §1.3 | With pruning enabled, the retained top-8 two-swap candidates at any state equal those computed from the unpruned `score_two_swaps` ratios; `ρ·Γ₂ ≤ w_c·w_d` on every S1–S3 pair |

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
