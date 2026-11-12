# P2 Gate — Blocking (N1) and week-6 held-out freeze

**Chief Architect · 2026-11-12 · Week 6 goal N1.** Status: `P2_N1_MATH_FROZEN` once the independent fresh-context review verdict is recorded in its dispatch Issue and this file is committed byte-identically. Baseline `main=c0f289d`.

Notation as in the [W1 gate](W1_GENERIC_CRITERIA_GATE.md): candidate rows `F[m,p]`, exact IDs, no replacement. Labels: **PROVED** = derived here (the Fraction counts are supporting evidence, not part of the proof); **OBSERVED** = finite evidence only; **CONJECTURE** = not established.

## 1. Fixed block effects

Model `y = Fθ + Zγ + ε` with `Z[n,g]` the 0/1 block-indicator matrix of a design with `g` blocks of sizes `n₁…n_g` (`Σ n_b = n`, block index `b`), `γ` free block effects, errors independent with common variance. `F` must **not** contain an intercept column (it is absorbed by the blocks; a constant column gives a singular `M_B`).

1. **Information (PROVED).** The information for `θ` after eliminating `γ` is `M_B = Fᵀ(I − Z(ZᵀZ)⁻¹Zᵀ)F = Σ_b Σ_{i∈b} (fᵢ − f̄_b)(fᵢ − f̄_b)ᵀ`, `f̄_b` the mean row of block `b`. Fraction check: equal in 40/40 random integer cases (`m=9, p=3`, blocks of 3 and 4).
2. **Rank and size (PROVED).** `rank M_B ≤ n − g`, so `n − g ≥ p` is necessary (not sufficient) for a nonsingular blocked design; `M_B = F_SᵀF_S − Σ_b n_b f̄_b f̄_bᵀ ≼ F_SᵀF_S`, with equality iff every column of `F_S` has mean zero in every block. A column constant within every block (e.g. an intercept) makes `M_B` singular.
3. **Within-block exchange is a rank-2 update (PROVED).** In block `b` (size `n_b`, mean `c`), replace design row `f_o` by an unused candidate `f_j`. With `u = f_j − c`, `v = f_o − c`, `d = u − v`: `ΔM_B = uuᵀ − vvᵀ − ddᵀ/n_b = W K Wᵀ`, `W = [u v]`, `K = [[1 − 1/n_b, 1/n_b], [1/n_b, −1 − 1/n_b]]`. (The cross terms `c dᵀ + d cᵀ` cancel exactly.) Hence `det M_B' / det M_B = det(I₂ + K G)`, `G = WᵀM_B⁻¹W = [[g_uu, g_uv], [g_uv, g_vv]]` with `g_uu = uᵀM_B⁻¹u`, `g_uv = uᵀM_B⁻¹v`, `g_vv = vᵀM_B⁻¹v`. Fraction check: `ΔM_B` and the ratio exact in 560/560 swaps.
4. **Between-block interchange is a rank-≤2 update (PROVED).** Swap design row `f_o` in block `r` (size `n_r`, mean `c_r`) with design row `f_q` in block `t` (size `n_t`, mean `c_t`). `Σ ffᵀ` is unchanged; with `d = f_q − f_o`, `e = c_r − c_t`: `ΔM_B = −(e dᵀ + d eᵀ) − (1/n_r + 1/n_t) d dᵀ = W K′ Wᵀ`, `W = [d e]`, `K′ = [[−(1/n_r + 1/n_t), −1], [−1, 0]]`, so `det M_B' / det M_B = det(I₂ + K′ WᵀM_B⁻¹W)`. Fraction check: `ΔM_B` and the ratio exact in 600/600 interchanges (`p = 4`); the reviewer's independent check agreed. Block sizes are preserved by both moves.
5. **2³ oracle (PROVED).** `F` = columns `A, B, C, AB, AC, BC` of the 2³ factorial (no intercept), `n = 8` (all runs), 2 blocks of 4. Every 4-run block on which `ABC` is constant has all six column means 0, so `M_B = FᵀF = 8I₆`, `det M_B = 8⁶ = 262144`, which by 2 is the maximum. Fraction enumeration of all 70 splits: the maximum is `8⁶`, attained exactly by the two `ABC`-constant splits (one split, two labelings). Uniqueness without enumeration: by 2, `M_B ≼ 8I₆`, so `det M_B = 8⁶` forces every eigenvalue to 8, i.e. `M_B = 8I₆`, i.e. all block means zero. This case needs the interchange move: with `m = n` no exchange with an unused candidate exists. 36/70 random splits are singular (see the start rule in §2).

## 2. API decision: `optimal_design(F, n, *, blocks=None, …)`

- `blocks`: `None` (unchanged behaviour, byte-identical results) or a list/tuple of positive Python ints with `sum(blocks) == n` (validated: trust boundary). Requires `m ≥ n` and `n − len(blocks) ≥ p` (necessary only, §1.2), else `ValueError`.
- D only: `blocks` with `criterion != "D"` or with `fixed_ids` raises `ValueError` this week.
- Start: per restart draw distinct IDs and a random partition into the given sizes; if `M_B` is not positive definite (Cholesky fails) redraw, exactly like the unblocked path: after `100·restarts` failed draws raise the existing `ValueError` ("No feasible start …"). No regularization or repair phase.
- Search: from the start, first-improvement local search over (i) within-block exchanges with unused candidates, scored by §1.3, and (ii) between-block interchanges of two design rows, scored by §1.4, accepting a move only if a dense `slogdet` of the new `M_B` confirms the improvement under the existing `_tolerance`; stop when no move improves or at `max_iter`. Restarts and `generator` as for the unblocked path; best restart by objective.
- Result: `ids` (int64, length `n`, the first `n₁` entries are block 1, each block sorted ascending), `blocks` (the sizes), `objective = logdet M_B`, `criterion="D"`, `stop_reason`, `certificate="LOCAL_BLOCK_EXCHANGE"` if no improving move remains else `"NOT_CERTIFIED"`, `efficiency_lower_bound=None` (no equivalence-theorem floor is claimed for blocked designs), `restart_objectives`, `best_restart`, `starts`.
- Complexity target per pass: `O(m p² + n m p)` for exchange scores (precompute `M_B⁻¹` and `M_B⁻¹` applied to the candidate rows and block means once per pass) and `O(n p² + n² p)` for interchange scores; no `m×m` matrices; workspace checked against `max_elements`.

## 3. Held-out additions (frozen 2026-11-12; never used for tests, tuning or dispatch choices; run once at week end, reported as-is)

Coding: `X = full_factorial(levels).double() − 1` (levels 3 → codes −1, 0, 1). Seed `generator=torch.Generator().manual_seed(20261112)`; all other arguments at defaults.

| Case | Definition | Record |
|---|---|---|
| **H11** | `F = polynomial_features(X, 2, include_bias=False)` on the 3³ grid (27 candidates, `p = 9`); `optimal_design(F, 18, blocks=[6, 6, 6])` | IDs per block, `logdet M_B`, stop reason, per-restart time. Reference: R AlgDesign `optBlock(~quad(A,B,C), withinData=<same 27 rows>, blocksizes=c(6,6,6))` on the already-probed Colab R runtime, versions recorded (`sessionInfo()`), its blocks re-scored by the shared TorchOED `M_B` (§1.1); record whether it repeats candidate rows. Not reproducible ⇒ `NOT_RUN`. |
| **H12** | `definitive_screening(13)` (order-14 conference matrix, 29 runs) | max abs of `CᵀC − 13I`; max abs inner product of each main-effect column with every other main-effect, square and two-factor-interaction column; the largest absolute correlation among second-order columns. |
| **H13** | `F = polynomial_features(X, 2)` on the 3⁴ grid (81 candidates, `p = 15`); `ids = optimal_design(F, 20)["ids"]` | `design_diagnostics(F, ids)`: max VIF, SPV quantiles (0, 0.25, 0.5, 0.75, 1) over the 81 candidates. |

## 4. Not in this gate

A/I blocked criteria, random block effects, split-plot, an intercept check (a user-supplied intercept column simply yields no feasible start), blocks with `fixed_ids`, an efficiency floor for blocked designs, orthogonal blocking via generators, `design_diagnostics` (N2) and `definitive_screening` (N3) mathematics.
