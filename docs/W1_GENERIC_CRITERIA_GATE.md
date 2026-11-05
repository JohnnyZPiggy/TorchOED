# W1 Gate — Criterion-generic (D, A, I) exchange and certificates

**Chief Architect · 2026-11-01 · Week 2 goal W1.** Status: `W1_MATH_FROZEN` — independent fresh-context review **PASS** (2026-11-01, recorded in the merging PR). Baseline `main=19e68fe`.

Extends the [M3-B2 gate](M3B2_K2_SEARCH_GATE.md) §1 (gather identities) to linear criteria. Notation as there: candidate features `F[m,p]`, exact IDs, no replacement, `m≥n≥p`, current design `S`, `M=F_SᵀF_S=LLᵀ`, `Z=L⁻¹Fᵀ`, `G=ZᵀZ=FM⁻¹Fᵀ`.
Labels: **PROVED** = derived here and checked exactly with `fractions.Fraction`; **OBSERVED** = finite evidence only; **HYPOTHESIS** = calibrated choice to be confirmed.

## 1. Criteria (definitions, decisions)

- **D**: maximize `logdet M` — unchanged; the v0.5.0 path is the reference and must give identical IDs/objectives.
- **L-criteria**: minimize `φ_B(M)=tr(B M⁻¹)` for a fixed symmetric PSD `B≠0` (`p×p`). Public scores keep the existing maximization sign: `score = −φ_B`.
  - **A**: `B=I_p`. Not invariant to feature scaling/coding; the design is optimal *for the given F*. Documented, not normalized.
  - **I** (decision): `B=W=FᵀF/m`, the uniform moment matrix of the candidate set, so `φ_W` is the average prediction variance over candidates (the AlgDesign `optFederov(criterion="I")` convention, needed for W3 comparison). A user-supplied region `B` is **Not required** this week.

## 2. Exchange identities (PROVED)

Let `Q = F M⁻¹ B M⁻¹ Fᵀ` (`m×m`, PSD). For a k-swap with outgoing IDs `O⊂S`, incoming `N⊂Sᶜ`, `|O|=|N|=k`, `idx=(O,N)`, `J=diag(−I_k, I_k)`, and `K = J + G[idx,idx]` (`2k×2k`):

1. **Determinant.** `det M'/det M = det(I+J·G[idx,idx]) = (−1)ᵏ det K` (matrix determinant lemma; `J⁻¹=J`). k=1: `Γ₁=(1−G_ii)(1+G_jj)+G_ij²`, k=2: `Γ₂=det K` — the existing D identities.
2. **SPD equivalence.** `M'` is a Gram matrix, so `M'⪰0` and `Γ_k>0 ⇔ M'≻0`. Hence `K` is invertible on every legal move; moves with `Γ̂≤0` are illegal (never ridged or clamped).
3. **L-update (Woodbury).** If `Γ_k>0`: `φ_B(M') = φ_B(M) − tr(K⁻¹ Q[idx,idx])`.
4. **k=1 closed form.** With `u=G_ii, v=G_jj, c=G_ij, α=Q_ii, β=Q_jj, γ=Q_ij`:
   `φ_B(M') = φ_B(M) + [(1+v)α − 2cγ − (1−u)β] / Γ₁`.
   Reductions: pure deletion gives `+α/(1−u)`, pure addition `−β/(1+v)` (Sherman–Morrison).
5. **Gather forms.** `Q = Zᵀ H Z` with `H=L⁻¹BL⁻ᵀ` (`p×p`). For **A**: `Q=(L⁻ᵀZ)ᵀ(L⁻ᵀZ)=FM⁻²Fᵀ`. For **I**: `H=ZZᵀ/m`, equivalently `Q = G·G/m` (the sum runs over all m candidates), and `φ_W = tr(G)/m`.

Preconditions throughout: current `M≻0`; the pure-deletion reduction needs `u<1` (L1 below has `u=1`, a legal swap whose delete-only intermediate is singular — never form it).

**Exact evidence.** `verify.py` (Architect sandbox, `Fraction`, no torch): 12 random integer fixtures (`p∈{2,3}`, `m∈{6,7}`, `n∈{p,p+1,p+2}`), B=I and B=W. Identity 1 + 4 against direct Gram determinant / `tr(B M'⁻¹)`: **1380/1380** single swaps; identity 3 for k=2: **1560/1560** legal pairs; `Γ>0 ⇔` Sylvester-PD: **1380/1380**.

**Complexity (target).** Per accepted move, the D path computes `Z` (`O(mp²)`) and gathers from `G`. L-criteria add `Q` at the same order: A one more triangular solve `O(mp²)`, I forms `H=ZZᵀ/m` (`O(mp²)`) then `Q=ZᵀHZ` restricted to the needed rows/columns (`diag(Q)`, `Q[S,:]`) — same order as A; do not form `G·G` (`O(m²p)`). Pair scoring stays O(1) per k=1 move and O(k³)=O(1) per k=2 pair. Target: A/I wall time ≤ 2× D on the same `bench_b0` case (HYPOTHESIS; measure, do not assume).

## 3. Search semantics

Same best-improvement 1-swap then 2-swap (VND) order, deterministic tie-breaks (smallest outgoing position, then smallest incoming ID), seeded restarts and feasibility rules as `d_optimal_design`. Acceptance: a move is accepted only if the **directly recomputed** `φ_B(M')` (Cholesky solve on the new `M'`) decreases by more than a scale-aware threshold `τ = 128·p·ε·max(1,|φ|)·max(1,cond)` (HYPOTHESIS, mirrors the D `_tolerance`). Certificate labels unchanged: `LOCAL_2_NUMERICAL` only after full N₁ and N₂ scans without verified improvement — never global optimality.

## 4. Certified efficiency lower bound (PROVED)

For an exact n-run design with SPD `M` and any PSD `B≠0`, let `Q_jj = f_jᵀM⁻¹BM⁻¹f_j` over the **whole candidate set**. Then

**L-eff ≥ tr(B M⁻¹) / (n · max_j Q_jj)**, where L-eff `= φ_B(M*)/φ_B(M)` relative to the optimal **approximate** design on the candidate set, `φ_B(M*)` meaning the infimum of `φ_B` over PD normalized moment matrices `M*=Σ_j w_j f_jf_jᵀ`, `Σw_j=1` (this also covers singular `B`) (hence also ≥ relative to the best exact n-run design). Analogue of the D floor `p/(n·max_j G_jj)`.

*Proof.* Use per-run information `M̄=M/n`; `φ_B` is convex and homogeneous of degree −1 on the PD cone, with gradient `−M̄⁻¹BM̄⁻¹`. For `t>0`, convexity at `tM̄`: `φ(M*) ≥ φ(tM̄) + tr(∇φ(tM̄)(M*−tM̄)) = 2φ(M̄)/t − E_{ξ*}[d(x)]/t²`, where `d(x)=fᵀM̄⁻¹BM̄⁻¹f ≤ d̄ = n²·max_j Q_jj`. Taking `t=d̄/φ(M̄)` gives `φ(M*) ≥ φ(M̄)²/d̄`; divide by `φ(M̄)=n·tr(BM⁻¹)`. The bound is ≤1 because the design's own points average `d` to `φ(M̄)`. ∎

**Exact evidence.** For every nonsingular subset of the 12 fixtures (556 designs, A and I; reviewer: 2792 further designs, an approximate optimum by multiplicative algorithm, and 861 rank-one-B designs — bound attained, never violated), the bound is ≤1 and ≤ the exact-design efficiency against the exhaustive optimum (min slack 0.0035). For I: `tr(BM⁻¹)=tr(G)/m`, `Q_jj=(G·G)_jj/m`.

Returned field name: `efficiency_lower_bound` (criterion-specific value), keeping `d_efficiency_lower_bound` for D backwards compatibility.

## 4b. Dual (approximate-design) efficiency floor — week 3 W2 (PROVED)

**Chief Architect · 2026-11-05 · Week 3 goal W2.** The §4 floor and the D floor `p/(n·max_j G_jj)` evaluate the dual bound at the design's own moment matrix `M̄=M/n`. Both proofs only use that the evaluation point is a PD normalized moment matrix, so any PD weights may be used instead. Notation: `w≥0`, `Σw_j=1`, `M_w=Σ_j w_j f_jf_jᵀ` PD; `ξ*` an optimal approximate design with moment matrix `M*` (§4 infimum convention).

**D (PROVED).** With `d_j(w)=f_jᵀM_w⁻¹f_j`, `d̄_w=max_j d_j(w)`:
`logdet M* ≤ logdet M_w + p·log(d̄_w/p)`.
*Proof.* `logdet` is concave on the PD cone with gradient `M⁻¹`. For `t>0`: `logdet M* ≤ logdet(tM_w) + tr((tM_w)⁻¹M*) − p = logdet M_w + p·log t + E_{ξ*}[d(x)]/t − p ≤ logdet M_w + p·log t + d̄_w/t − p`; `t=d̄_w/p` gives the claim. ∎
Hence the D-efficiency of the exact design against the approximate optimum (and so against the best exact n-run design) satisfies
**D-eff ≥ exp((logdet(M/n) − logdet M_w)/p) · p/d̄_w**. At `w` = the design's own uniform weights this is the current floor `p/(n·max_j G_jj)`.

**A/I (PROVED).** With `φ=φ_B`, `q_j(w)=f_jᵀM_w⁻¹BM_w⁻¹f_j`, `q̄_w=max_j q_j(w)`: the §4 proof with `M̄` replaced by `M_w` gives `φ(M*) ≥ φ(M_w)²/q̄_w`, so
**L-eff ≥ φ(M_w)² / (q̄_w · φ(M̄))**, `φ(M̄)=n·tr(BM⁻¹)`. At `w` = the design's own weights this is the §4 floor.

Both bounds are ≤ the true efficiency (≤ 1) for every `w` with `M_w` PD (zero weights allowed); the choice of `w` affects tightness only, never validity.

**Weights (decision).** One multiplicative run per `optimal_design` call (independent of the restarts and of the returned design: `w` depends only on `F` and the criterion), starting from uniform `w=1/m`:
- D: `w_j ← w_j·d_j(w)/p` (Titterington; keeps `Σw=1`).
- A/I: `w_j ← w_j·q_j(w)^{1/2} / Σ_k w_k q_k(w)^{1/2}`.
Evaluate the floor at every iterate (the `d`/`q` needed for the update give it for free) and keep the maximum. Stop when the weights are near-optimal — D: `p/d̄_w ≥ 1−10⁻⁴`; A/I: `φ(M_w)/q̄_w ≥ 1−10⁻⁴` (these are the §4-type floors of `w` itself, so the returned floor is then ≥ (1−10⁻⁴)× the design's true efficiency against `M*`) — or after **1000** iterations. Returned `efficiency_lower_bound` (and `d_efficiency_lower_bound` for D) = `max(current floor, dual floor)`; IDs, objectives, certificates and stop reasons are unchanged.

**Complexity.** Per iteration O(m·p²): form `M_w=Fᵀdiag(w)F`, one `p×p` Cholesky `M_w=LLᵀ`; D: `d_j=‖L⁻¹f_j‖²` (one `m×p` triangular solve); A: `q_j=‖M_w⁻¹f_j‖²` (two solves); I: precompute once `B=RᵀR` (`p×p`, from `FᵀF/m`) and use `q_j=‖R·M_w⁻¹f_j‖²` — **never** the §4 form `(G·G)_jj/m`, which needs `m×m` workspace. Total O(T·m·p²), T ≤ 1000, workspace O(mp + p²) within the existing `max_elements` budget. The weight run is defined for D, A and I only (B PD, so `q_j>0` for every `f_j≠0`); with a singular user `B` it could zero weights and is not used. Rows with `f_j=0` get `w_j→0`, harmless. In float64 non-support weights may underflow to exactly 0 over many iterations; validity is unaffected (any `w` with `M_w` PD gives a valid bound), but if `M_w` becomes numerically singular the run stops and keeps the best floor so far.

**OBSERVED (Architect sandbox, 2026-11-05, NumPy float64).** (a) 30 random integer fixtures `m=7,p=3,n=4`, every nonsingular subset, D/A/I: 3135 designs, `max(current, dual)` with 300 iterations ≤ exact efficiency against the exhaustive exact optimum in every case (0 violations, tolerance 1e−9). (b) `bench_b0` grid2 (3² grid, quadratic, p=6) n=7 design `[0,6,2,3,4,7,8]`: current floor 0.6122, dual floor 0.9448 after 10 iterations and 0.9454 at convergence. (c) Earlier Architect estimate on the 7 feasible `bench_b0` cases, 2000 iterations: D median 0.93 (0.86–0.99) (`WEEK_GOAL.md`).

**Independent review (2026-11-05): PASS** — fresh-context reviewer, exact `Fraction` checks on 60 random integer fixtures (m=6..8, p=2..3, n=p..p+2; D, A, I and rank-one B; uniform, random positive, sparse, multiplicative and design-own weights): 303,012 dual-inequality checks vs exact designs and 1,680 vs approximate optima, 292,192 (design, w) floor ≤ exact-efficiency checks, 25,408/25,408 exact reductions to the §4 / D current floors, W2-2 exact; 0 violations. Wording fixes applied (B PD for the weight run, I-criterion workspace, float underflow).

**HYPOTHESIS (to be measured by the W2 Engineer Issue).** Floor overhead ≤ 10 % of the `optimal_design` median time on every `bench_b0` case. If it is not, the Engineer reports the numbers and the Architect rules (W1 time has priority over W2 tightness; option: lower the iteration cap).

**Required oracles for the W2 Engineer Issue.**

| ID | Fixture | Expected |
|---|---|---|
| W2-1 | Exhaustive `Fraction` oracle, a frozen random integer fixture `m=7,p=3,n=4`, D, A and I | returned floor ≤ exact efficiency vs exhaustive optimum for every nonsingular subset; floor ≥ the §4 / D current floor |
| W2-2 | Closed form: 2-point line, `F=[[1,−1],[1,1]]`, `n=2`, D | `M*=I`, `w=(½,½)` is optimal: `d̄_w=2=p` → dual floor = `exp((logdet(M/2)−0)/2)·1 = 1` for `S={0,1}` |
| W2-1b | Same fixture | stopping rule: when `p/d̄_w ≥ 1−ε` (D) or `φ(M_w)/q̄_w ≥ 1−ε` (A/I) the floor is ≥ (1−ε)× the efficiency against the **approximate** optimum `M*` (not the best exact design) |
| W2-3 | `bench_b0` D cases | IDs/objectives/certificates identical to `main`; report median D floor on the 7 feasible cases (target ≥ 0.90) and the timing overhead |

## 5. Required oracles for the W1 Engineer Issue

| ID | Fixture | Expected |
|---|---|---|
| L1 | `F=[[1,0],[0,1],[2,0],[1,1]]`, `S=[0,1]`, swap position 0 → ID 2 | `Γ₁=4`; A: `tr M⁻¹` 2 → 5/4 (identity 4 equals direct) |
| L2 | Exhaustive `Fraction` oracle, one frozen random integer fixture with `m=7,p=3,n=4`, criteria A and I | every single and pair move's identity-4/3 value equals direct `tr(B M'⁻¹)`; search result's objective = its direct value; certified bound ≤ exact efficiency vs exhaustive optimum |
| L3 | `bench_b0` D cases | `criterion="D"` IDs and objectives identical to v0.5.0; time not regressed (rerun `tools/bench_b0.py`) |

L1 check: `M=I`, `a=(1,0)`, `b=(2,0)`; `u=1,v=4,c=2,α=1,β=4,γ=2`; `Γ₁=0·5+4=4`; `Δ=[(5)(1)−2·2·2−0]/4=−3/4`, `M'=diag(4,1)` → `tr M'⁻¹ = 1/4+1 = 5/4` ✓.

## 6. Out of scope

E- and c-criteria search, compound criteria, user-supplied I-region, k≥3, GPU, Modified Fedorov/DETMAX.

_Generated by [Claude Code](https://claude.ai/code)_
