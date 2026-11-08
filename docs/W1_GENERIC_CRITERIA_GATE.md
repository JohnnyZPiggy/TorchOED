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

## 2c. Closed-form two-swap L-gain — week 4 C1, answers [Mentor] #125 Q3 (PROVED identity; HYPOTHESIS float guards)

**Chief Architect · 2026-11-08.** This section gives an exact, O(1)-per-pair replacement for the batched 4×4 `det`/`solve` of identity 3 (k=2). It uses the **add-first** pivot, so it is valid for every `n ≥ p`, including `n = p` and `n = p+1` where `ρ(O) = det(I−G_OO) = 0`. The delete-first pivot is prohibited there (M3-B2 gate §1 item 5). It is not a pruning bound: every pair is still scored, at a lower constant cost. #125 Q1, Q2 and Q4 stay open.

**Notation.** As in §2: `O={a,b}⊂S`, `N={c,d}⊂Sᶜ`, `K = J + G[idx,idx] = [[A, X],[Xᵀ, P]]` with `A = −I₂ + G_OO`, `X = G_ON` (2×2, rows a,b, columns c,d), `P = I₂ + G_NN`. `Y = Q_ON`, `Q_OO`, `Q_NN` are the matching 2×2 blocks of `Q`.

**Identity (PROVED).** `P` is SPD for every design, because `G ⪰ 0`. Let
- `π = det P = (1+G_cc)(1+G_dd) − G_cd²` (> 0), `P⁻¹ = adj(P)/π`;
- `g_N = tr(P⁻¹Q_NN) = [(1+G_dd)Q_cc + (1+G_cc)Q_dd − 2G_cd Q_cd]/π` (the pure-addition gain of N; Sherman–Morrison/Woodbury);
- `E = X P⁻¹`, `T = A − E Xᵀ` (the Schur complement of `P` in `K`), `τ = det T`;
- `R = Q_OO − E Yᵀ − Y Eᵀ + E Q_NN Eᵀ`.

Then (i) `Γ₂ = det K = π·τ`, so the move is legal iff `τ > 0` (§2 item 2); (ii) if `τ ≠ 0`,
**`γ(O,N) = tr(K⁻¹Q[idx,idx]) = g_N + (T_bb R_aa + T_aa R_bb − 2 T_ab R_ab)/τ`**, and `φ_B(M') = φ_B(M) − γ`.

*Proof.* Block elimination of the invertible `P`: `det K = det P · det T`. When `T` is invertible, `K⁻¹ = diag(0, P⁻¹) + Wᵀ T⁻¹ W` with `W = [I₂, −X P⁻¹]` (2×4), which is the standard Schur-complement inverse. Then `tr(K⁻¹Q_idx) = tr(P⁻¹Q_NN) + tr(T⁻¹ W Q_idx Wᵀ)`. Expanding `W Q_idx Wᵀ` with `X P⁻¹ = E` gives `R`. `T` and `R` are symmetric, and `tr(T⁻¹R)` for 2×2 symmetric matrices is `(T_bb R_aa + T_aa R_bb − 2T_ab R_ab)/det T`. ∎ For `k=1` the same elimination reduces to §2 item 4.

**Cost.** Per state: `G` and `Q` on `S×S`, `S×Sᶜ` and `Sᶜ×Sᶜ`, which is `O((n+q)²p)` and tileable. `S×Sᶜ` is already formed for N₁. Per incoming pair: `π`, `P⁻¹`, `g_N` (O(1)). Per (O,N): 8 gathers and about 60 flops. There is no `p`-dependence and no batched LU. The `C(n,2)×C(q,2)` scan becomes elementwise tensor arithmetic over `(O, N)` tiles.

**Exact evidence (Architect sandbox, `Fraction`, 2026-11-08).** 24 random integer fixtures (p=3, m∈{7,8}, n∈{p, p+1, p+2}), A (`B=I`) and I (`B=FᵀF/m`): 1196 two-swaps, of which 188 outgoing pairs had `ρ(O)=0` exactly. `π·τ = det M'/det M` held in **1196/1196**. There were 1164 legal pairs, and on all of them the closed-form `γ` equals both the direct `φ_B(M) − tr(B M'⁻¹)` and the gather4 `tr(K⁻¹Q_idx)` (**0 mismatches**). The 32 pairs with `τ ≤ 0` all had `det M' = 0` exactly.

**Floating point (OBSERVED, NumPy float64; HYPOTHESIS for torch).** Pools: random500 (3 factors, quadratic, p=10), grid3 and grid4. n ∈ {10, 11, 20}, A and I, two 1-swap-local states each; all pairs, or a uniform sample of about 4·10⁶.
- For pairs with `max(γ̂_closed, γ̂_gather4) > −φ`, the largest absolute difference from gather4 was **5.6e−14**, at most 7.8e−4 of `τ_tol = 128·p·ε·max(1,|φ|)·max(1,cond M)` (§3). Against the direct recomputation on the top 50 pairs it was ≤ 1.2e−14.
- The differences grow with `|γ|` (up to 9e−3 at `|γ| ≈ 10⁷`). Those are nearly singular moves, far from eligible.
- **Legality.** `|π̂τ̂ − det K̂| ≤ 2.2e−15`. The legality verdicts `τ̂>0` and `det K̂>0` disagreed on up to 286 pairs per state. This happened only on grids with n ∈ {p, p+1}, at exactly-singular moves with `|Γ̂₂| ≤ 4.1e−16`.
- **Speed.** About 7× faster than batched `np.linalg.det` + `solve` on 4·10⁶ pairs (0.6 s vs 4.5 s, one process). This is not a torch measurement.

**Use (decision; keeps A/I results identical to `main`).** The closed form is a **screen** that only decides which pairs are skipped. Every pair that is not skipped is scored by gather4, which is the current `_linear_pair_scores` arithmetic. Its legality (`det K̂ > 0`) and gain come from gather4, and the existing rule (`gain > tolerance`, `key > cutoff`, maximum gain, ties → lexicographically smallest row) is applied to those gather4 values.
1. **Near-singular band (always rescored).** Every pair with `|π̂τ̂| ≤ η = 10⁻⁸`, or with a non-finite closed-form value, goes into the rescored set unconditionally. It is not part of the sort below. Its gather4 value enters `g_best` only if the pair is **eligible under the current rule**: legal, finite, `> tolerance` **and** `key > cutoff`.
2. **Skip rule.** Let `δ = τ_tol` of the current state and let `(g₀, r₀)` be the cutoff, if there is one. A non-band pair is skipped only if one of these holds:
   - (a) `γ̂_closed ≤ tolerance − δ/2`;
   - (b) `γ̂_closed > g₀ + δ/2` (strict; then its gather4 gain is `> g₀` and the pair lies before the cutoff);
   - (c) in descending `γ̂_closed` order, `γ̂_closed < g_best − δ`, where `g_best` is the best gather4 value found so far over pairs eligible under the current rule (`> tolerance` and `key > cutoff`), band-1 pairs included.
   The skip rule is re-applied for each new cutoff of the resume sequence.
3. **Why this is exact (under the HYPOTHESIS).**
   - (a) A pair with gather4 gain `> tolerance` has `γ̂_closed > tolerance − δ/2`, so (a) never skips it.
   - (b) A skipped pair has gather4 gain `> g₀`, so it lies before the cutoff.
   - (c) A skipped pair has gather4 gain `< g_best − δ/2`, so it can neither win nor tie. Every pair tied with the winner is rescored, so the tie-break is preserved.
   The winner and the whole resume sequence are therefore bit-identical to the current full gather4 scan.
4. **HYPOTHESIS (two premises, both checked by the Engineer evidence).**
   - (H1) For every non-band pair with `γ̂_gather4 > tolerance − δ` or `γ̂_closed > tolerance − δ/2`: `|γ̂_closed − γ̂_gather4| ≤ δ/2`. This is scoped on purpose. Outside it the gap can exceed δ by orders of magnitude: the reviewer found 9e−4 at `γ ≈ −2·10⁷` with `det K̂ ≈ 10⁻⁶`, because the error scales with `cond M'`, not `cond M`. Such pairs have `γ ≤ φ` far below eligibility.
   - (H2) Gather4 run on a subset batch gives bitwise the same per-pair values as on the full tiled batch (CPU float64). The Engineer's test must check this. GPU is out of scope.
   - Observed margins on H1: ≥ 90× on the `bench_b0`-type pools above; **6.5×** on the reviewer's random Gaussian fixtures, one third with near-duplicate rows.
   - In exact arithmetic `det K ≥ 0` always, since `M' ⪰ 0`, so the only illegal case is `τ = 0`. A negative `τ̂` is roundoff.
5. D, tolerances, certificates, tile capacity semantics (`max_elements`) and the public `score_two_swaps` are unchanged.

**Independent review (2026-11-08): PASS after two rounds.** A fresh-context reviewer checked the identity exactly with `Fraction`: 117 integer fixtures (p=2..3, m=6..8, n=p..p+2), 4,300 pairs × {A, I}, of which 3,188 had `ρ=0` and 114 had a singular `M'`. Every check passed: `det K = π·τ`, γ against both gather4 and the direct `tr(B M'⁻¹)`, and `P` SPD.
- Round 1 found that the premise as first stated was false outside the eligible range. It also found that legality was taken from the closed form, that the cutoff test was not strict, and that band-1 handling and the subset-batch bitwise assumption were not stated. All five were fixed, giving the scoped H1, gather4 legality, (b) strict, item 1 and H2.
- Round 2 found that a band-1 pair already before the cutoff could set `g_best`. That is fixed: `g_best` is taken over currently eligible pairs only. The reviewer stated that every other step is correct under H1+H2 with that fix.
- H1 re-run on the full scope, NumPy: 300 Gaussian fixtures, 105,138 in-scope pairs, maximum gap = 0.153·δ/2. H1 can only be validated offline, with full scans.

**Required checks for the Engineer Issue.** (a) The exhaustive `Fraction` identity test above on one frozen fixture with `n=p` (so `ρ=0` occurs). (b) The existing A/I `Fraction`/oracle tests pass unchanged. (c) H2: gather4 values of a rescored subset equal the full-scan values bitwise. (d) `tools/bench_b0.py`: A/I/D IDs and objectives identical to base on every case and seed, D time not regressed, new per-restart A:D and I:D reported as-is.

**What this does not do.** It is a constant-factor gain (≈7× in NumPy) on the N₂ scoring arithmetic only. It cannot by itself close the 39×–518× per-restart A:D gap of #126; that needs pruning (#125 Q1/Q2/Q4).

## 2d. Tangent prefix prune for the A/I two-swap scan — week 4 C1, partial answer to [Mentor] #125 Q1 (PROVED bound; HYPOTHESIS float guards)

**Chief Architect · 2026-11-08.** §2c scores every pair at O(1). This section skips pairs **without enumerating them**, using a separable upper bound on γ. It is a sound but weak bound: it removes most pairs only when the pool is large relative to `n` (random500, `n = 2p`). It does not close C1; #125 Q1/Q2/Q4 stay open.

**Notation.** As in §2 and §4: `φ = φ_B(M) = tr(BM⁻¹)`, `Q_jj = f_jᵀM⁻¹BM⁻¹f_j` (the diagonal of `Q` in §2 item 5, already formed for N₁), `s_O = Q_aa + Q_bb` for an outgoing pair `O={a,b}⊂S`, `s_N = Q_cc + Q_dd` for an incoming pair `N={c,d}⊂Sᶜ`, `Δ = s_N − s_O`, and `M' = M − F_OᵀF_O + F_NᵀF_N`.

**Bound (PROVED).** For every two-swap with `M' ≻ 0`: `φ + Δ > 0` and
**`γ(O,N) = φ − φ_B(M') ≤ u(Δ) := φΔ/(φ+Δ) ≤ Δ`.**

*Proof.* `tr(M⁻¹BM⁻¹M') = φ − s_O + s_N = φ + Δ`, which is `> 0` because `M⁻¹BM⁻¹ ≻ 0` (B PD for A and I) and `M' ≻ 0, M' ≠ 0`. `φ_B` is convex and homogeneous of degree −1 on the PD cone, with gradient `−X⁻¹BX⁻¹`. As in §4, for `t > 0`: `φ_B(M') ≥ φ_B(tM) + tr(∇φ_B(tM)(M' − tM)) = 2φ/t − (φ+Δ)/t²`. Taking `t = (φ+Δ)/φ` gives `φ_B(M') ≥ φ²/(φ+Δ)`, so `γ ≤ φ − φ²/(φ+Δ) = u(Δ)`. Finally `Δ − u(Δ) = Δ²/(φ+Δ) ≥ 0`. ∎
`u` is strictly increasing on `Δ > −φ` (`u' = φ²/(φ+Δ)² > 0`), and for `0 ≤ θ < φ`, `u(Δ) < θ ⇔ Δ < Δ*(θ) := φθ/(φ−θ)`.

**Prefix structure.** For a fixed outgoing pair `O`, only incoming pairs with `s_N ≥ s_O + Δ*(θ)` can reach a threshold `θ` (necessary, not sufficient). They form a prefix of the incoming pairs sorted by `s_N` in descending order. Its length is one `searchsorted`. The skipped incoming pairs are never enumerated, as in the D Hadamard prefix of `docs/M3B2_K2_SEARCH_GATE.md` §1.3.

**Use (decision; keeps A/I results identical to `main`).** This adds rule **(d)** to the §2c skip rule, with the same `δ`, `tolerance` (= δ in the code, `exchange._tolerance`), `g_best` (best gather4 value over pairs eligible under the current rule, band pairs included) and cutoff handling. Let `θ = min(max(tolerance, g_best) − δ, φ̂/2)`, recomputed whenever `g_best` changes (any order of `O` is allowed). The cap `φ̂/2` keeps `Δ*` well conditioned (about `dΔ*/dθ = φ²/(φ−θ)² ≤ 4` and `Δ* ≤ φ`, approximately since `φ̂ ≈ φ`; motivation only, not used in the proof); a lower `θ` only skips fewer pairs, so it is always sound.
- (d) If `θ ≥ 0`: skip every pair with `ŝ_N < ŝ_O + Δ̂*(θ) − δ`.
- (d) is applied **before** the §2c band test: a skipped pair is not enumerated, so it is neither band-rescored nor screened. Rules (a)–(c) and the band apply unchanged to the pairs (d) keeps.

*Why this is exact (under the HYPOTHESIS).* For a skipped pair, (H3) gives `Δ < Δ*(θ)`, hence (bound) `γ ≤ u(Δ) < θ`, hence (H4) `γ̂_gather4 < θ + δ/2 ≤ max(tolerance, g_best) − δ/2`. So in the base full scan the pair is either not eligible (`≤ tolerance`) or strictly below `g_best`, so it can neither win nor tie. This holds for every cutoff of the resume sequence, since `g_best` is always over currently eligible pairs. The winner and the whole resume sequence are bit-identical to the current scan. Pairs with singular `M'` have no exact bound but are illegal (`det K̂ ≤ 0` in gather4 is roundoff-only, §2c item 4); skipping them changes nothing unless gather4 would call them legal and eligible, which H4 excludes.

**HYPOTHESIS (two premises; checked by the Engineer evidence, never assumed).**
- (H3) For **every** pair that (d) skips, i.e. every pair with `x̂ := ŝ_N − ŝ_O − Δ̂*(θ) < −δ`, the exact `x := s_N − s_O − φθ/(φ−θ)` is `< 0`, with the **exact** `φ`, `s` and `θ` taken as the float number actually used (θ is only a threshold). If `θ ≥ φ` exactly, H3 is vacuous: then `γ ≤ u(Δ) < φ ≤ θ` directly, and the evidence reports this case separately instead of as an H3 failure. A sufficient check: `|x̂ − x| ≤ δ/2` over the whole skipped tail, not only near the threshold. Because `x` is a difference of sums of diagonal entries of the PSD `Q` plus a `θ ≤ φ̂/2` term, the expected error is a few `ε·(s_N + s_O + φ)`.
- (H4) `|γ̂_gather4 − γ| ≤ δ/2` for every legal pair with `γ̂_gather4 > tolerance − δ`. This premise is already implicit in the current scan, which trusts gather4 values against `tolerance`; it is stated here because (d) uses the exact bound.

**OBSERVED (Architect sandbox, NumPy float64, 2026-11-08; not torch).** Fraction of all pairs **kept** by (d) at the certifying threshold `θ = 0` (`Δ*(0) = 0`, i.e. keep iff `s_N > s_O`), two 1-swap-local states each, quadratic features (p=10 for 3 factors, p=15 for grid4):

| pool | n | A kept | I kept |
|---|---|---|---|
| random500 | 10 | 0.78–0.80 | 0.79–0.82 |
| random500 | 20 | **0.073–0.094** | **0.15–0.20** |
| grid3 | 10 / 20 | 0.88–0.94 | 0.81–0.85 |
| grid4 | 15 / 30 | 0.64–0.93 | 0.70–0.97 |

During an improving scan `θ > 0` grows with `g_best`, so more pairs are skipped; these numbers are the certifying-scan case. Expected effect: about 5×–14× less screen work on random500 n=20 (the worst A:D case, 194.5× after #138), negligible elsewhere. C1 stays **not met**.

**Exact evidence (Architect sandbox, `Fraction`, 2026-11-08).** 16 random integer fixtures (p=3, m∈{7,8}, n∈{p, p+1, p+2}), A (`B=I`) and I (`B=FᵀF/m`): 674 legal two-swaps; `φ+Δ > 0` and `γ ≤ u(Δ) ≤ Δ` held in **674/674** (no case of equality).

**Cost.** Per state: `s_N` for the `C(q,2)` incoming pairs and one descending sort, `O(q² log q)`; per outgoing pair one `searchsorted`. The scan then costs O(kept pairs) instead of O(all pairs).

**Independent review (2026-11-08): PASS after two rounds.** A fresh-context reviewer checked the bound exactly with `Fraction`: 60 integer fixtures (p=2..3, m=6..8, n=p..p+2), A and I, 2166 two-swaps with `M' ≻ 0` and 66 singular: `tr(M⁻¹BM⁻¹M') = φ+Δ`, `φ+Δ > 0`, `γ ≤ u(Δ) ≤ Δ` and `u(Δ) < θ ⇔ Δ < Δ*(θ)` (θ = kφ/10) all held. Round 1 found that H3 covered only pairs near the threshold, not the whole skipped tail, and that `Δ̂*` is ill-conditioned as `θ → φ`; fixed by H3 over the whole tail and the cap `θ ≤ φ̂/2`. Round 2 found that `x` must use the exact `φ` (not `φ̂`) and that `θ ≥ φ` must be handled; both fixed as stated. The reviewer confirmed every other step (ties, cutoff/resume, the cap never loosens the chain).

**Required checks for the Engineer Issue.** (a) An exhaustive `Fraction` test of `γ ≤ u(Δ)` on one frozen fixture, A and I, including `n = p`. (b) On one random CPU float64 fixture, A and I, with at least one cutoff state: the pruned path yields the same winner sequence as `tolerance=None` (full scan), `torch.equal`. (c) H3 (over the whole skipped tail, against a recomputation of `Q_jj` and `φ` in higher precision or `Fraction`) and H4 measured offline on full scans of the `bench_b0` random500 n=20 states and reported as margins. (d) `tools/bench_b0.py`: A/I/D IDs and objectives identical to base on every case and seed, D time not regressed, per-restart A:D, I:D and the kept fraction reported as-is.

## 2e. Removal-tangent prune per outgoing pair — week 4 C1, second partial answer to [Mentor] #125 Q1 (PROVED bound; HYPOTHESIS float guards)

**Chief Architect · 2026-11-08.** §2d expands the tangent at `M`. This section expands it at `M_r = M − F_OᵀF_O`, separately for each outgoing pair `O`. The bound is still separable in `c, d` for fixed `O`, so it admits the same sorted-prefix enumeration, and it is much tighter on `n = 2p` states. The bound needs only `B ⪰ 0`, `B ≠ 0`. It is undefined when `M_r` is singular (always at `n = p`); there §2d alone applies. It does not close C1.

**Notation.** As in §2d. For `O = {a,b} ⊂ S`: `G_OO`, `Q_OO` are the 2×2 blocks of `G`, `Q`; `ρ(O) = det(I − G_OO)`; for `j ∈ Sᶜ`, `g_j = G[O, j]`, `q_j = Q[O, j]` (2-vectors).

**Lemma 1 (PROVED).** `M_r ≻ 0 ⇔ ρ(O) > 0`. *Proof.* `M_r = M^{1/2}(I − P)M^{1/2}` with `P = M^{−1/2}F_OᵀF_OM^{−1/2}`, whose nonzero eigenvalues are those of `G_OO = F_OM⁻¹F_Oᵀ`. `G_OO` is a principal block of the projection `F_SM⁻¹F_Sᵀ`, so its eigenvalues lie in `[0,1]`; hence `I − G_OO ⪰ 0`, and `I − P ≻ 0 ⇔ I − G_OO ≻ 0 ⇔ ρ(O) > 0`. ∎

**Lemma 2 (PROVED, Woodbury).** If `ρ(O) > 0`, let `W = (I − G_OO)⁻¹`. Then `M_r⁻¹ = M⁻¹ + M⁻¹F_OᵀWF_OM⁻¹`, so
- `δ_O := φ_B(M_r) − φ = tr(W Q_OO) ≥ 0`, and `φ_r := φ_B(M_r) = φ + δ_O`;
- `t_O(j) := f_jᵀM_r⁻¹BM_r⁻¹f_j = Q_jj + 2 g_jᵀW q_j + g_jᵀ W Q_OO W g_j ≥ 0`.

*Proof.* `M_r⁻¹f_j = M⁻¹(f_j + F_OᵀWg_j)`; expand the quadratic form with `F_OM⁻¹BM⁻¹F_Oᵀ = Q_OO`, `F_OM⁻¹BM⁻¹f_j = q_j`. `δ_O` is the trace of the second Woodbury term against `B`. `t_O(j) ≥ 0` since `M_r⁻¹BM_r⁻¹ ⪰ 0`. ∎

**Bound (PROVED).** If `ρ(O) > 0`, for every incoming pair `N = {c,d}` with `M' = M_r + F_NᵀF_N ≻ 0` (always true here, since `M_r ≻ 0`), with `T = t_O(c) + t_O(d)`:
**`γ(O,N) = φ − φ_B(M') ≤ v_O(T) − δ_O`, `v_O(T) := φ_rT/(φ_r + T)`.**

*Proof.* `tr(M_r⁻¹BM_r⁻¹M') = φ_r + T > 0`. Convexity and degree −1 homogeneity of `φ_B` at `tM_r` (the §2d argument with `M_r` in place of `M`) give `φ_B(M') ≥ 2φ_r/t − (φ_r+T)/t²` for `t > 0`; `t = (φ_r+T)/φ_r` gives `φ_B(M') ≥ φ_r²/(φ_r+T)`. Hence `γ = (φ_r − φ_B(M')) − δ_O ≤ φ_r − φ_r²/(φ_r+T) − δ_O = v_O(T) − δ_O`. ∎

**Threshold.** `v_O` is increasing on `T ≥ 0` with `v_O < φ_r`. For a threshold `θ` let `η = min(θ + δ_O, φ_r/2)` and `T*_O = φ_rη/(φ_r − η)` (well defined, `≤ φ_r`). Then `T < T*_O ⇒ v_O(T) < η ⇒ γ < η − δ_O ≤ θ`. The cap `φ_r/2` only lowers the effective threshold (sound; bounds `dT*/dη ≤ 4`).

**Prefix structure.** For fixed `O`, sort the incoming candidates by `t̂_O` in descending order, `u_1 ≥ u_2 ≥ …`. The kept pairs `{i < k}` with `u_i + u_k ≥ thr` are, for each `i`, a contiguous run `k ∈ (i, L_i]` found by one `searchsorted`. Cost per `O`: `O(q)` for `t_O`, `O(q log q)` for the sort, `O(q log q)` for the run ends.

**Use (decision; keeps A/I results identical to `main`).** This adds rule **(e)** next to §2d rule (d), with the same `θ`, `δ`, `tolerance`, `g_best` and cutoff handling. `θ` may be recomputed at the start of each `O` or each tile; any order of `O` is allowed.
- (e) For each `O` with `ρ̂(O) ≥ 10⁻⁶` and `θ ≥ 0`: skip every incoming pair with `T̂ < T̂*_O − δ`, where `T̂*_O` uses `η̂ = min(θ + δ̂_O, φ̂_r/2)`. For `O` with `ρ̂(O) < 10⁻⁶`, (e) is not applied (only (d)).
- A pair is enumerated only if neither (d) nor (e) skips it. (e) may drive the enumeration and (d) be applied as a per-pair mask on the enumerated pairs, or the reverse. Skipped pairs are never band-rescored or screened; rules (a)–(c), the band, gather4, eligibility, cutoff/resume and tie-breaks are unchanged for kept pairs.

*Why this is exact (under the HYPOTHESIS).* For a pair skipped by (e), (H5) gives `T < T*_O`, so `γ < θ` (threshold paragraph), and then (H4) gives `γ̂_gather4 < θ + δ/2 ≤ max(tolerance, g_best) − δ/2`: the pair can neither be eligible-and-winning nor tie, exactly as in §2d. The winner and the whole resume sequence are bit-identical to the full scan.

**HYPOTHESIS (checked by the Engineer evidence, never assumed).**
- (H5) The exact `ρ(O) > 0` for every `O` where (e) is applied, and for every pair that (e) actually skips (the skip set the code uses, e.g. after `searchsorted`), the exact `x := T − T*_O` is `< 0`, where `T`, `δ_O`, `φ_r` (hence `η`, `T*_O`) are recomputed exactly (`Fraction`) or in float128 from the float `F` and the float `θ` actually used. A sufficient check: `|x̂ − x| ≤ δ/2` over the whole skipped tail. The guard `ρ̂ ≥ 10⁻⁶` is a float guard, not an accuracy proof (the error of `t̂_O` grows with `‖W‖`); the evidence must report `min ρ̂` and the exact (or float128) `min ρ` over the `O` where (e) was applied, and the H5 margin per state.
- (H4) as in §2d.

**OBSERVED (Architect sandbox, NumPy float64, 2026-11-08; not torch).** Fraction of all pairs kept at the certifying threshold `θ = 0`, two 1-swap-local states each (best-improvement 1-swap descent from random starts), quadratic features:

| pool | n | A: (d) / (e) / both | I: (d) / (e) / both |
|---|---|---|---|
| random500 | 20 | 0.073 / 0.036 / **0.0060** | 0.16–0.31 / 0.054–0.058 / **0.016–0.025** |
| grid4 | 30 | 0.66–0.73 / 0.13 / **0.12** | 0.71–0.75 / 0.095–0.099 / **0.089–0.094** |
| grid3 | 20 | 0.94 / 0.42 / 0.41 | 0.96 / 0.38 / 0.38 |
| any | `n = p` | (e) not applicable (`ρ = 0` for every `O`) | same |

The cap `φ_r/2` and the guard `ρ̂ ≥ 10⁻⁶` changed none of these fractions; `min ρ̂` over the `n = 2p` states was 0.023–0.071. Expected effect versus §2d alone: about 10–12× fewer enumerated pairs on random500 n=20 (A), 6–13× on random500 n=20 (I), about 6–8× on grid4 n=30. C1 stays **not met** until measured.

**Exact evidence (Architect sandbox, `Fraction`, 2026-11-08).** 16 random integer fixtures (p=3, m∈{7,8}, n∈{p+1,p+2}, entries in [−3,3]), A and I, every nonsingular `S`: Lemma 1 on every `O`; Lemma 2 (`δ_O`, `t_O(j)`) in 23,480/23,480 identity checks; the bound in **15,060/15,060** two-swaps with `M' ≻ 0` (no equality on these fixtures, min slack 0.0081·φ; equality is possible in general, e.g. when `F_NᵀF_N ∝ M_r`, which does not affect the strict implication below); and the capped threshold implication `T < T*_O ⇒ γ < θ` for `θ = kφ/10`, k = 0..4. 3,908 outgoing pairs had `ρ = 0` (rule not applied).

**Cost.** Per state `O(n² q)` for all `t_O(j)` (two 2-vectors and one 2×2 per `(O, j)`), `O(n² q log q)` for the per-`O` sorts and run ends, then O(kept pairs). Workspace `O(q)` per `O` plus the enumerated tile.

**Required checks for the Engineer Issue.** (a) An exhaustive `Fraction` test of Lemma 2 and the bound on one frozen fixture with `n = p+1` (some `O` with `ρ = 0`), A and I, computing `γ` directly from `tr(BM'⁻¹)`. (b) On one random CPU float64 fixture, A and I, with at least one cutoff state and at least one pair skipped by (e): same winner/resume sequence as `tolerance=None`, `torch.equal`. (c) H5 and H4 margins on full scans of the `bench_b0` random500 n=20 and grid4 n=30 1-local A and I states, plus `min ρ̂`. (d) `tools/bench_b0.py`: IDs and objectives identical to base, D unchanged and not slower, per-restart A:D, I:D and the kept fraction reported as-is.
**Independent review (2026-11-08): PASS.** A fresh-context reviewer checked §2e exactly with `Fraction` on 40 integer fixtures (p∈{2,3}, m=6..8, n∈{p+1,p+2}), A and I: Lemma 1 on 3,030 outgoing pairs (1,196 with `ρ = 0`); Lemma 2 in 4,124/4,124 `t_O(j)` cases and every `δ_O`; the bound in 3,204/3,204 two-swaps; the capped implication in 10,171/10,171 triggered cases over 23 thresholds (cap active in 9,587); float `t̂_O`, `φ̂_r` relative error ≤ 1.2e−13 on 600 Gaussian outgoing pairs. One unproved step found and fixed: H5 now also requires the exact `ρ(O) > 0` wherever (e) is applied (the float guard alone does not imply it). Wording fixes: equality is possible in general; H5 is measured on the skip set the code actually uses; `B ⪰ 0, B ≠ 0` suffices.

## 2f. Square-design coordinate prune at `n = p` — week 4 C1, third partial answer to [Mentor] #125 Q1 (PROVED bound; HYPOTHESIS float guards)

**Chief Architect · 2026-11-08.** §2e is undefined at `n = p` (`ρ(O) = 0` for every `O`), and §2d keeps about 80–97% of the pairs there. This section uses the fact that at `n = p` the design matrix `F_S` is square: a two-swap is a change of the 2 rows of a `p×p` coordinate matrix, and `φ_B(M')` splits into an outgoing-only term plus a term that depends on `N` only through one separable number. It applies **only when `n = p`** and `B ≻ 0` (A: `B = I`; I: `B = FᵀF/m`, PD because `F` has full column rank). It does not close C1 by itself.

**Notation.** As in §2d/§2e, with `n = |S| = p` and `F_S` invertible, so `M = F_SᵀF_S`, `G_SS = I` and `Q_SS = F_S^{−T}BF_S^{−1} ≻ 0`; in particular `φ = tr(Q_SS)`. For `O = {a,b} ⊂ S` let `R = S∖O`, `Q_OO` the 2×2 block, and for `j ∈ Sᶜ` let `s_O(j) = G_aj² + G_bj²`. Define, per `O`,
- `τ_O := tr(Q_RR − Q_RO Q_OO⁻¹ Q_OR)` (the trace of a Schur complement; `≥ 0`), equivalently `φ − τ_O = tr(Q_OO⁻¹ (Q_SS²)_OO)`;
- `κ_O := (tr Q_OO^{1/2})² = tr Q_OO + 2√(det Q_OO) > 0`.

**Bound (PROVED).** For `n = p`, every outgoing `O`, and every incoming `N = {c,d}` with `M' ≻ 0`, with `σ = s_O(c) + s_O(d)`:
**`φ_B(M') ≥ τ_O + κ_O/σ`, hence `γ(O,N) ≤ β_O(σ) := φ − τ_O − κ_O/σ`.**

*Proof.* For `j ∈ Sᶜ`, `g_j := G[S,j] = F_S^{−T}f_j`, i.e. `f_j = F_Sᵀg_j`. Replacing rows `a,b` of `F_S` by `f_c, f_d` gives `F' = KF_S`, where `K` is the identity with rows `a, b` replaced by `g_cᵀ, g_dᵀ`. With `W := G_ON` (2×2), the `O` rows of `K` are `[Wᵀ | G_RNᵀ]` (columns `O | R`), so `det K = ± det W`, `M' ≻ 0 ⇔ det W ≠ 0`, and `K⁻¹ = [[W^{−T}, −W^{−T}G_RNᵀ], [0, I]]`. Then `φ_B(M') = tr(BF_S⁻¹K⁻¹K^{−T}F_S^{−T}) = tr(K^{−T}Q_SSK⁻¹) = Σ_columns k of K⁻¹ kᵀQ_SSk`. The two `O` columns contribute `tr(W⁻¹Q_OOW^{−T}) = tr(Q_OO(WWᵀ)⁻¹)`. Each `R` column is `e_r + E_Ou_r` for some `u_r ∈ ℝ²` and contributes `Q_rr + 2u_rᵀQ_Or + u_rᵀQ_OOu_r ≥ Q_rr − Q_rOQ_OO⁻¹Q_Or` (minimum over `u`, `Q_OO ≻ 0`); summing over `r` gives `≥ τ_O`. For the `O` term, Cauchy–Schwarz on `tr(Q_OO^{1/2}) = tr(X^{1/2} · X^{−1/2}Q_OO^{1/2})` with `X = WWᵀ ≻ 0` gives `(tr Q_OO^{1/2})² ≤ tr X · tr(Q_OO X⁻¹)`, and `tr X = ‖W‖_F² = σ`. So `φ_B(M') ≥ τ_O + κ_O/σ`. The identity `φ − τ_O = tr Q_OO + tr(Q_OO⁻¹Q_ORQ_RO) = tr(Q_OO⁻¹(Q_SS²)_OO)` is direct. ∎

**Threshold.** `β_O` is strictly increasing in `σ > 0` and `β_O < φ − τ_O`. For `θ`: if `φ − τ_O ≤ θ`, then `γ < θ` for **every** `N` (no enumeration for this `O`); otherwise `σ < σ*_O := κ_O/(φ − τ_O − θ) ⇒ γ < θ`.

**Prefix structure.** As §2e with `s_O` in place of `t_O`: for fixed `O`, sort the incoming `ŝ_O(j)` descending; kept pairs `{i < k}` with `u_i + u_k ≥ σ̂*` form, for each `i`, one contiguous run found by `searchsorted`.

**Use (decision; keeps A/I results identical to `main`).** Rule **(f)**, with the same `θ`, `δ`, `tolerance`, `g_best` and cutoff handling as (d) (§2d). It applies only when `n = p`; there (e) never applies. For each `O` with `θ ≥ 0`:
- if `φ̂ − τ̂_O − θ + δ ≤ 0`: skip every incoming pair for this `O`;
- otherwise skip every pair with `σ̂ < κ̂_O/(φ̂ − τ̂_O − θ + δ)` (i.e. `β̂_O(σ̂) < θ − δ`), with `κ̂_O` from `√max(det Q̂_OO, 0)`.

A pair is enumerated only if neither (d) nor (f) skips it (either may drive the enumeration, the other be a mask). Skipped pairs are never band-rescored or screened; rules (a)–(c), the band, gather4, eligibility, cutoff/resume and tie-breaks are unchanged for kept pairs.

*Why this is exact (under the HYPOTHESIS).* For a pair skipped by (f), (H6) gives `β_O(σ) < θ` (or `φ − τ_O ≤ θ`), so `γ < θ` by the bound, and (H4) gives `γ̂_gather4 < θ + δ/2 ≤ max(tolerance, g_best) − δ/2`: it can neither be eligible-and-winning nor tie, exactly as in §2d. Pairs with singular `M'` are illegal and change nothing, as in §2d: H4, read with `γ = −∞` for singular `M'`, already forbids gather4 from accepting an exactly singular pair with `γ̂ > tolerance − δ`.

**HYPOTHESIS (checked by the Engineer evidence, never assumed).**
- (H6) For every pair that (f) actually skips (the skip set the code uses), the exact `β_O(σ) < θ` — or exact `φ − τ_O ≤ θ` for whole-`O` skips — with `φ`, `τ_O`, `κ_O`, `σ` recomputed in `Fraction` or float128 from the float `F` and the float `θ`. A sufficient check: `|β̂ − β| ≤ δ/2` over the whole skipped set. The evidence reports the minimum of `(θ − β)/δ` over the skipped set per state, and `min det Q̂_OO / (tr Q̂_OO)²` over the `O` used.
- (H4) as in §2d. Skipped pairs are never scored in the pruned run, so the H4 evidence comes from the unpruned scan or from rescoring the pairs (f) skips.

**OBSERVED (Architect sandbox, NumPy float64, 2026-11-08; not torch).** Fraction of all pairs kept at `θ = 0`, 1-swap-local states (best-improvement 1-swap descent from random starts), quadratic features, `n = p`:

| pool | n = p | A: (d) / (f) / both | I: (d) / (f) / both |
|---|---|---|---|
| random500 | 10 | 0.80 / 0.010–0.034 / **0.009–0.013** (reviewer: (f) 0.010–0.034) | 0.80–0.83 / 0.0019–0.0064 / **0.0015–0.0032** |
| grid3 | 10 | 0.88–0.90 / 0.050–0.15 / **0.050–0.15** | 0.81–0.91 / 0.026–0.11 / **0.026–0.10** |
| grid4 | 15 | 0.93–0.97 / 0.061–0.081 / **0.061–0.081** | 0.93–0.97 / 0.0032–0.023 / **0.0032–0.023** |

Expected effect at `n = p`: about 60×–500× fewer enumerated pairs than (d) alone on random500 n=10 (A:D 88×, I:D 134× on `reports/w141`), 6×–300× on the grids. C1 stays **not met** until measured.

**Exact evidence (Architect sandbox, `Fraction`, 2026-11-08).** 48 random integer fixtures (p∈{2,3}, m∈{6,7,8}, entries in [−3,3]), A and I, every nonsingular `S` with `n = p`: `φ − τ_O = tr(Q_OO⁻¹(Q_SS²)_OO)` in 2,437/2,437 outgoing pairs; the bound, checked exactly as `y := (φ_B(M') − τ_O)σ − tr Q_OO ≥ 0` and `y² ≥ 4 det Q_OO`, in **16,906/16,906** two-swaps with `M' ≻ 0` (1,814 singular `M'` skipped); the whole-`O` implication `φ − τ_O ≤ θ ⇒ γ < θ` in 39/39 triggered cases (`θ = kφ/10`, k = 0..4). NumPy float64: 12,828 further two-swaps, 0 violations.

**Cost.** Per state: `Q_SS²` in `O(p³)`; per `O` the 2×2 `Q_OO`, `τ_O`, `κ_O` in `O(1)` from `Q_SS²`; `s_O(j)` for all `j` in `O(q)`; per-`O` sort and run ends `O(q log q)`. Total `O(p² q log q)` plus O(kept pairs), same order as §2e. Workspace `O(q)` per `O`.

**Independent review (2026-11-08): PASS.** A fresh-context reviewer checked §2f exactly with `Fraction` on 218 integer fixtures (p∈{2,3,4}, m=p+3..p+5, entries in [−3,3], n = p), A and I: the `K⁻¹` trace equal to `φ_B(M')` exactly, `τ_O ≥ 0`, the identity, and the bound (squared form) in 4,938/4,938 nonsingular two-swaps; all 234 singular `M'` had `det W = 0` and conversely. Float64 on random500 n=10 (3 seeds × A, I, all 5.4M pairs per state): no `γ > β`, every positive-gain two-swap kept. The reviewer confirmed the δ direction, `θ ≥ 0`, the strict exactness chain, monotone `g_best`, contiguous runs and free-position restriction. Two wording fixes applied: singular pairs are covered by H4 with `γ = −∞`; H4 evidence comes from the unpruned scan or a rescore of the skipped pairs.

**Required checks for the Engineer Issue.** (a) An exhaustive `Fraction` test of the bound on one frozen integer fixture with `n = p`, A and I, computing `φ_B(M')` directly from `tr(BM'⁻¹)` (independent oracle, not the code's `τ`, `κ`). (b) On one random CPU float64 fixture with `n = p`, A and I, with at least one cutoff state and at least one pair skipped by (f) that (d) keeps: same winner/resume sequence as `tolerance=None`, `torch.equal`. (c) H6 and H4 margins on full scans of the `bench_b0` random500 n=10 and grid4 n=15 1-local A and I states. (d) `tools/bench_b0.py`: IDs and objectives identical to base, D unchanged and not slower, per-restart A:D, I:D and the kept fraction reported as-is.

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
