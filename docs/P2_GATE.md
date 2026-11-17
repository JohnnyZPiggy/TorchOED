# P2 Gate — Blocking (N1), design diagnostics (N2), definitive screening (N3) and week-6 held-out freeze

**Chief Architect · 2026-11-12 · Week 6 goal N1.** Status: `P2_N1_MATH_FROZEN` once the independent fresh-context review verdict is recorded in its dispatch Issue and this file is committed byte-identically. Baseline `main=c0f289d`. §5 (N3) added 2026-11-12: status `P2_N3_MATH_FROZEN` under the same rule. §6 (N2) added 2026-11-12: status `P2_N2_MATH_FROZEN` under the same rule.

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

A/I blocked criteria, random block effects, split-plot, an intercept check (a user-supplied intercept column simply yields no feasible start), blocks with `fixed_ids`, an efficiency floor for blocked designs, orthogonal blocking via generators, diagnostics for blocked designs, power, alias matrices (N2 is §6, N3 is §5).

## 5. Definitive screening designs (N3)

Labels as above. All arithmetic is exact (Python ints); the tensor is built only at the end.

1. **Paley conference matrix (PROVED, Paley 1933).** For an odd prime `q` let `χ` be the Legendre symbol mod `q` (`χ(0) = 0`; for `a ≢ 0`, `χ(a) = 1` if `a^((q−1)/2) ≡ 1 (mod q)` else `−1`), `Q[i,j] = χ(j − i)` (`0 ≤ i, j < q`), `ε = χ(−1)` (`+1` if `q ≡ 1 mod 4`, `−1` if `q ≡ 3 mod 4`), and `C = [[0, 1ᵀ], [ε·1, Q]]` of order `c = q + 1`. Then `Qᵀ = εQ`, `Q1 = 0` and `QQᵀ = qI − J` (classical: `Σ_x χ(x)χ(x + d) = −1` for `d ≢ 0`), hence `CᵀC = (c − 1)I`, `diag C = 0` and every off-diagonal entry is `±1`; `C` is symmetric (`q ≡ 1`) or skew (`q ≡ 3 mod 4`). Integer check: all eight `q ∈ {3, 5, 7, 11, 13, 17, 19, 23}`.
2. **Design (decision; Jones & Nachtsheim 2011).** `definitive_screening(k)` uses `c = k` if `k` is even and `k − 1 ∈ P`, `c = k + 1` if `k` is odd and `k ∈ P`, with `P = {3, 5, 7, 11, 13, 17, 19, 23}` (a fixed tuple, no primality code); `k` must be a Python `int` (validated: trust boundary) and every other `k` raises `ValueError`. Supported `k`: 3–8, 11–14, 17–20, 23, 24. With `C_k` the first `k` columns of `C` (odd `k` drops the last column), the design is `D = [C_k; −C_k; 0ᵀ]` in exactly this row order: `2c + 1` runs, levels `−1, 0, 1`.
3. **Structure (PROVED).** (i) `C_kᵀC_k = (c − 1)I_k`, the leading `k × k` principal submatrix of `CᵀC = (c − 1)I_c` (§5.1), so `DᵀD = 2(c − 1)I_k`. (ii) Every odd-degree monomial column sums to 0 over `D`, because the rows of `−C_k` negate those of `C_k` and the centre row is 0; in particular every column of `D` has mean 0 and each main-effect column is orthogonal to every square column `x_b²` and every two-factor-interaction column `x_b x_d` (including `b = a`). (iii) `C` has one zero per column, so every factor is at 0 in exactly 3 runs. Squares and two-factor interactions are correlated with each other (not orthogonal); no claim is made about them. Integer check of (i)–(iii): every supported `k`, both column counts.
4. **API.** `definitive_screening(k, *, dtype=torch.float64, device=None)` in `src/torchoed/factorial.py`, exported at the root; returns the `(2c + 1) × k` tensor of §5.2. Cost `O(c²)` Python-int work; no budget argument (largest output 49 × 24).

## 6. Design diagnostics (N2)

Labels as above. `S = F[ids]` (`n × p`, distinct IDs, no blocks), `M = SᵀS` nonsingular.

1. **Scaled prediction variance (definition).** `SPV(x) = n·f(x)ᵀM⁻¹f(x)` for a feature row `f(x)`; `region` rows `R` (`N × p`, already in feature space; default `R = F`).
2. **Design-average identity (PROVED).** `(1/n) Σ_{i∈S} SPV(fᵢ) = tr(M⁻¹ Σ_{i∈S} fᵢfᵢᵀ) = tr(I_p) = p` (the factor `n` in SPV cancels the `1/n`). Hence `max_{i∈S} SPV(fᵢ) ≥ p`, and `max_{x∈F} SPV ≥ p`. Fraction check: exact in 30/30 random cases (`m = 9, p = 3, n = 5`).
3. **D-efficiency floor (PROVED; Kiefer–Wolfowitz/Atwood).** For every probability measure `w` on the candidate rows with `M(w) = Σ_x w_x f(x)f(x)ᵀ`, AM–GM on the eigenvalues of `(M/n)^{−1/2} M(w) (M/n)^{−1/2}` gives `(det M(w) / det(M/n))^{1/p} ≤ tr((M/n)⁻¹M(w))/p = Σ_x w_x SPV(x)/p ≤ max_{x∈F} SPV(x)/p`. Every exact `n`-subset `T` gives the measure `M_T/n`, so `(det M / det M_T)^{1/p} ≥ p / max_{x∈F} SPV(x)`: the D-efficiency of `S` against the exact optimum is at least `p / max SPV`. This is exactly `exp(−efficiency_gap_bound/p)` of the [M3-C API](M3C_API.md) (`efficiency_gap_bound = p·log(max SPV/p)` at the returned design), and `optimal_design(..., criterion="D")` returns `efficiency_lower_bound` as the maximum of this value and the dual floors, so `efficiency_lower_bound ≥ p / max_{x∈F} SPV(x)` up to float64 rounding. The cross-check needs the maximum over all candidate rows, i.e. `region=None`; a user `region` not containing every row of `F` gives no floor. Fraction check: the floor held against the enumerated exact optimum in 30/30 random cases (reviewer: 300/300, attained with ratio 1 at the optimum).
4. **Correlation and VIF (decision + PROVED identity).** Columns of `S` whose design entries are all equal (exact float comparison; e.g. an intercept) are excluded; `J` = the remaining column indices of `F`, ascending, as a list of Python ints. A column constant only up to rounding stays in `J` and is reported as computed (decision: no tolerance). With `X` = the columns `J` of `S`, each centred by its design mean, and `Δ = diag(XᵀX)`: `correlation = Δ^{−1/2}XᵀXΔ^{−1/2}` and `vif = diag(correlation⁻¹)`. Identity: `vif_j = 1/(1 − R_j²)`, `R_j²` the coefficient of determination of the least-squares regression of column `j` on an intercept and the other columns of `J`, so VIFs are intercept-adjusted whether or not `F` contains an intercept. Fraction check: identity exact in 30/30 random cases (reviewer: 200/200, with, without and with an extra constant column). `vif` is computed as `cholesky_inverse(cholesky(correlation)).diagonal()`, so a `correlation` that is not numerically positive definite raises from `torch.linalg.cholesky`. Decision: this includes valid designs whose centred columns are collinear, e.g. full dummy coding without an intercept (`M` nonsingular, VIF undefined); the user drops one dummy. `|J| = 0` (e.g. intercept only) gives a `0 × 0` `correlation` and an empty `vif`; `|J| = 1` gives `vif = [1]`.
5. **2³ oracle (PROVED).** `F = [1, A, B, C]` on all 8 runs of the 2³ factorial (`p = 4`): `M = 8I₄`, `J = [1, 2, 3]`, `correlation = I₃`, `vif = 1`, and `SPV(x) = 8·‖f(x)‖²/8 = 4 = p` at every run. Fraction check: exact.
6. **API.** `design_diagnostics(F, ids, *, region=None)` in `src/torchoed/information.py`, exported at the root. `M` via `exact_information(F, ids)` (which validates `F` and `ids`); `M⁻¹` applications by Cholesky solve. `region` must be an `(N, p)` tensor with `F`'s dtype and device; it is not validated further (a mismatch raises from torch). Returns a dict: `columns` (list `J`), `correlation` (`|J| × |J|` tensor), `vif` (length-`|J|` tensor), `spv` (length-`N` tensor, region row order), `fds` (a separate length-`N` tensor: `spv` sorted ascending; entry `i` is at fraction of design space `(i + 1)/N`; no fraction tensor is returned). Cost `O(N p² + n p²)`; no `N × N` or `m × m` matrices.

## 7. Split-plot D (W7 S1)

**Chief Architect · 2026-11-13 · Week 7 goal S1.** Status: `P2_S1_MATH_FROZEN` once the independent fresh-context review verdict is recorded in its dispatch Issue and this section is committed byte-identically. Baseline `main=07783a6`. Labels as above.

Model `y = Fθ + Zγ + ε` for a design of `n` rows in `w` whole plots of sizes `n₁…n_w` (`Σ n_k = n`, plot index `k`, rows of plot `k` contiguous), `Z[n,w]` the 0/1 plot-indicator matrix, `γ ~ (0, ησ²I_w)` independent of `ε ~ (0, σ²I_n)`, known variance ratio `η > 0`. Hence `Var y = σ²V`, `V = I + ηZZᵀ`. `F` may contain an intercept. Each candidate row `i` carries an integer hard-to-change label `hᵢ` (equal labels = the same hard-to-change setting). A design is **feasible** iff all rows of each whole plot have the same label (the plot's label) and all IDs are distinct.

1. **GLS information (PROVED, Sherman–Morrison per plot).** `V` is block diagonal with blocks `I + η11ᵀ`, whose inverse is `I − c_k 11ᵀ`, `c_k = η/(1 + η n_k)`. Hence `M = FₛᵀV⁻¹Fₛ = Σ_k [F_kᵀF_k − c_k s_k s_kᵀ]`, `s_k = Σ_{i∈k} fᵢ`. Fraction check: equals dense `FₛᵀV⁻¹Fₛ` in 20/20 random cases (`p ≤ 4`, 2–3 plots of sizes 1–4, rational `η`).
2. **Rank (PROVED).** `V⁻¹` is positive definite, so `M` is positive definite iff `Fₛ` has full column rank `p`; this does not depend on `η`. Hence `n ≥ p` is necessary. A column that is a function of the label only (e.g. the hard-to-change main effect) together with an intercept is collinear when all plots share one label. Fraction check: `det M = 0 ⟺ det FₛᵀFₛ = 0` in 200/200 random cases.
3. **Small-η limit (PROVED).** `0 < c_k < η`, so `‖M − FₛᵀFₛ‖ ≤ η Σ_k ‖s_k‖²`; as `η → 0`, `logdet M → logdet FₛᵀFₛ` for full-rank `Fₛ`.
4. **Within-plot exchange is a rank-≤3 update (PROVED).** In plot `k` (size `n_k`, sum `s = s_k`, `c = c_k`) replace design row `f_o` by an unused candidate `f_j`; with `d = f_j − f_o`, `ΔM = f_jf_jᵀ − f_of_oᵀ − c(s dᵀ + d sᵀ + d dᵀ) = W K Wᵀ`, `W = [f_j f_o s]`, `K = [[1 − c, c, −c], [c, −1 − c, c], [−c, c, 0]]`. Hence `det M′ / det M = det(I₃ + K G)`, `G = WᵀM⁻¹W`. The identity holds for any rows (it does not use labels). Fraction check: `ΔM` and the ratio exact in 50/50 random exchanges.
5. **Moves (decision).** Every accepted move keeps the design feasible and the plot sizes fixed. (i) *Exchange:* replace a row of plot `k` by an unused candidate with the plot's label, scored by §7.4. (ii) *Relabel:* for plot `k` and a label `ℓ′` different from its label with at least `n_k` unused candidates, build a proposal by replacing the plot's rows one at a time, in the plot's current row order, each by the unused label-`ℓ′` candidate with the largest §7.4 ratio from the current proposal state (ties: smallest candidate ID); the intermediate states mix labels, which §7.4 allows. If a step's best ratio is `≤ 0` the proposal is discarded (so every intermediate state stays positive definite). The completed proposal is feasible. (ii) is a greedy heuristic: a feasible positive-definite relabel to `ℓ′` may exist when the greedy proposal is discarded or does not improve, so local optimality is only with respect to these two moves. No between-plot interchange (decision; not needed for the oracles).
6. **Tiny exhaustive oracle (PROVED by enumeration).** Candidates: `(h, x)` for `h ∈ {−1, 1}`, `x ∈ {−1, 0, 1}`, in this order, listed twice (`m = 12`, IDs 0–11); `F = [1, h, x, x², h·x]` (`p = 5`), labels `h`, whole plots `[3, 3]`, `η = 1`. Exact enumeration of all 420 feasible designs (unordered plot pairs): `max det M = 48`, attained by 64 designs, each with one plot at `h = −1` and one at `h = 1`, every plot containing `x = −1, 0, 1`. Designs with both plots at one label are singular (§7.2).

## 8. API decision: `optimal_design(F, n, *, whole_plots=None, hard=None, eta=1.0, …)`

- `whole_plots`: `None` (unchanged behaviour, byte-identical results) or a list/tuple of positive Python ints summing to `n`; `hard`: a 1-D integer tensor of length `m` on `F`'s device (the labels); both given or both `None`; `eta`: a finite positive Python `int` or `float`. These are validated (trust boundary). Requires `m ≥ n ≥ p`, CPU float64 `F`, `criterion == "D"`, no `fixed_ids`, no `blocks`, else `ValueError`.
- Validation detail: `bool` is not accepted for `whole_plots` entries or `eta` (`type(x) is int`, or `float` for `eta`).
- Start: `labels = hard.unique()` (ascending). Per draw, for plots `k = 1…w` in order: `ℓ = labels[torch.randint(len(labels), (1,), generator=generator)]`, then the plot's rows are `pool[torch.randperm(len(pool), generator=generator)[:n_k]]`, `pool` = the still-unused candidates with label `ℓ` in ascending ID. If some `pool` has fewer than `n_k` rows, or `M` (§7.1) fails the blocked start rule (Cholesky plus pivot floor), the whole design is redrawn; after `100·restarts` failed draws in total across all restarts (as in the blocked path) raise the existing "No feasible start …" `ValueError`.
- Search (mirrors `_block_exchange`): design positions keep their order (an exchange replaces a position in place; a relabel replaces the plot's positions in position order). Each iteration scans exchanges first: plots in order, positions in order, then the same-label unused candidates in ascending ID whose §7.4 log-ratio is `> 0`; the first one whose dense `slogdet` of the new `M` improves the objective by more than the existing `_tolerance` is accepted and the iteration ends. If no exchange is accepted, relabels are scanned: plots in order, `ℓ′` in ascending label order, skipping the plot's own label and any `ℓ′` with fewer than `n_k` unused candidates; each step takes `argmax` of the ratio over the label-`ℓ′` pool in ascending ID (the pool excludes candidates already used by the design or by earlier steps of the proposal) (first maximum on ties, exact float comparison; discard if the maximum is `≤ 0`), and the completed proposal is accepted under the same dense `slogdet` + `_tolerance` test. An iteration with no accepted move stops with `stop_reason="LOCAL_SPLIT_PLOT_EXCHANGE"`; `max_iter` bounds the number of iterations (accepted moves), and reaching it gives `stop_reason="MAX_ITER"`. Restarts and `generator` as for the blocked path; best restart by objective (first on ties).
- Result: `ids` (int64, length `n`, plot by plot, each plot sorted ascending), `whole_plots` (the sizes), `eta`, `objective = logdet M`, `criterion="D"`, `stop_reason`, `certificate="LOCAL_SPLIT_PLOT_EXCHANGE"` if the stop reason is that (local only with respect to §7.5) else `"NOT_CERTIFIED"`, `efficiency_lower_bound=None`, `restart_objectives`, `best_restart`, `starts`.
- Complexity target per iteration: `O(m p² + n m p)` for exchange scores (`M⁻¹` refreshed after each accepted move and applied once per iteration to the candidate rows and plot sums); each relabel proposal `O(n_k (p³ + m p²))`; no `m×m` or `n×n` matrices; workspace checked against `max_elements`.

## 9. Held-out additions W7 (frozen 2026-11-13; never used for tests, tuning or dispatch choices; run once at week end, reported as-is)

Coding: `X = full_factorial([3, 3, 3]).double() − 1` (codes −1, 0, 1). Seed `generator=torch.Generator().manual_seed(20261114)`; all other arguments at defaults.

| Case | Definition | Record |
|---|---|---|
| **H14** | Candidates `X4 = torch.cat([X] * 4)` (108 rows; each setting usable in up to 4 runs); `F = polynomial_features(X4, 2)` (`p = 10`), `hard = (X4[:, 0] + 1).long()` (factor A hard to change); `optimal_design(F, 24, whole_plots=[4]*6, hard=hard, eta=1.0)` | IDs and A-setting per plot, `logdet M` (§7.1), stop reason, per-restart time. Reference: R skpr `gen_design` for the same 27-point grid, model `~ (A + B + C)^2 + I(A^2) + I(B^2) + I(C^2)`, 24 runs, A hard-to-change in 6 whole plots of 4, variance ratio 1, on the already-probed Colab R runtime with versions recorded (`sessionInfo()`), its 24 runs re-scored by the shared §7.1 `M`. Not reproducible ⇒ `NOT_RUN`. |
| **H15** | `L = simplex_lattice(5, 4)` (70 points, proportions summing to 1 as in goal S2); `F` = `polynomial_features(L, 2, include_bias=False)` with the 5 square columns dropped (`p = 15`); `optimal_design(F, 15)` | `logdet` against the exact `{5, 2}`-lattice value `−20·log 4` (Scheffé quadratic `det X = 4^−10` on the `{5, 2}` lattice; Kiefer 1961: the uniform `{q, 2}` lattice is D-optimal among approximate designs, so with `n = p` this bounds every exact design), and whether the selected rows equal the `{5, 2}` lattice points. |
| **H16** | `F = polynomial_features(X, 2)` (27 candidates, `p = 10`); `ids = optimal_design(F, 16)["ids"]` | the S3 `power` entry of `design_diagnostics(F, ids, effect=δ, alpha=0.05)` with `δ` = 1 for every coefficient (σ = 1), as-is. |

## 10. Not in this section

A/I split-plot criteria, split-split-plot, unknown `η` (REML), Kenward–Roger degrees of freedom, between-plot interchange, split-plot with `blocks` or `fixed_ids`, an efficiency floor for split-plot designs, diagnostics or power for split-plot designs.

## 11. Power (W7 S3)

Labels as in §6: `S = F[ids]` (`n × p`, distinct IDs, no blocks), `M = SᵀS` nonsingular, `ν = n − p ≥ 1`, error variance `σ² = 1` (so `δ` is an effect in units of σ). Notation: `I_x(a, b)` is the regularized incomplete beta function, `μ = λ/2`.

1. **Definition (decision).** For coefficient `j`, the two-sided level-`α` t-test of `βⱼ = 0` rejects when `T² = β̂ⱼ² / ((M⁻¹)ⱼⱼ s²) > F_{1−α}(1, ν)`, `s²` the residual mean square on `ν` degrees of freedom. Under `βⱼ = δⱼ`, `T² ~ F′(1, ν, λⱼ)` with `λⱼ = δⱼ² / (M⁻¹)ⱼⱼ`, so `powerⱼ = P(F′(1, ν, λⱼ) > F_{1−α}(1, ν))`. `δⱼ = 0` gives `powerⱼ = α`.
2. **Beta form (PROVED).** If `Y ~ F′(1, ν, λ)` then `X = Y/(Y + ν)` is a Poisson(`μ`) mixture of `Beta(1/2 + k, ν/2)`, so `P(Y ≤ f) = Σ_{k≥0} e^{−μ} μᵏ/k! · I_x(1/2 + k, ν/2)` with `x = f/(f + ν)` (standard noncentral-F/Beta mixture). The map `f ↦ x` is increasing, so the critical point is `x* = ` the `1 − α` quantile of `Beta(1/2, ν/2)`, i.e. `I_{x*}(1/2, ν/2) = 1 − α`, and `F_{1−α}(1, ν) = ν x*/(1 − x*)`.
3. **Recurrences (PROVED; DLMF 8.17.20–21 form).** With `T(a, b) = x^a (1−x)^b Γ(a+b) / (Γ(a+1) Γ(b))` and `U(a, b) = x^a (1−x)^b Γ(a+b) / (Γ(a) Γ(b+1))`: `I_x(a+1, b) = I_x(a, b) − T(a, b)` and `I_x(a, b+1) = I_x(a, b) + U(a, b)`. Bases: `I_x(1/2, 1/2) = (2/π) asin √x` and `I_x(1/2, 1) = √x`. Hence `I_x(1/2, ν/2)` is the base for `b₀ = 1/2` (ν odd) or `b₀ = 1` (ν even) plus `Σ_{i=0}^{ν/2 − b₀ − 1} U(1/2, b₀ + i)`: a finite sum of `O(ν)` positive terms (this is A&S 26.7.3–26.7.4 for the Student t CDF). Each term is evaluated as `exp(lgamma(·) − lgamma(·) − lgamma(·) + a·log x + b·log1p(−x))`.
4. **Power sum (PROVED).** With `Q₀ = 1 − I_{x*}(1/2, ν/2)` and `Q_{k+1} = Q_k + T(1/2 + k, ν/2)` at `x = x*`, `1 − I_{x*}(1/2 + k, ν/2) = Q_k`, so `power = Σ_{k≥0} w_k Q_k`, `w_k = exp(xlogy(k, μ) − μ − lgamma(k+1))` (`xlogy` makes `μ = 0` exact: `w₀ = 1`, `w_k = 0` for `k ≥ 1`). All terms are nonnegative: no cancellation beyond `Q₀`.
5. **Truncation (PROVED).** Sum `k = 0…K` with `K = ⌈μ + 12√μ + 30⌉`. Since `0 ≤ Q_k ≤ 1`, the truncation error is at most the Poisson tail `P(N > K) ≤ exp(−t²/(2(μ + t/3)))`, `t = 12√μ + 30` (Bernstein's inequality for a Poisson variable); the exponent is at least 45 for every `μ ≥ 0` (its minimum, at `μ = 0`), so the error is below `3·10⁻²⁰`.
6. **Quantile (decision).** `x*` by 64 bisection steps on `[0, 1]` for `I_x(1/2, ν/2) = 1 − α` (`I_x` is continuous and increasing in `x`). No SciPy, no Newton step, no tolerance argument. Accuracy is limited by float64 near `x = 1` (small `α`: `1 − x*` falls toward the spacing of floats below 1, e.g. `1 − x* ≈ (πα/2)²` for `ν = 1`, and for `ν = 1` at `α = 10⁻⁹` bisection returns `x* = 1` and power 0) and by the `2⁻⁶⁴` bisection width near `x = 0` (α near 1). Decision: `alpha` is restricted to `0.001 ≤ alpha ≤ 0.5` (trust boundary, item 9), the range checked in item 7; no reflection or log-space variant.
7. **Architect check (OBSERVED, 2026-11-13, float64 sandbox, SciPy 1.18.1 as a dev-only reference).** Items 2–6 in plain float64 agree with `scipy.stats.ncf.sf(f.ppf(1−α, 1, ν), 1, ν, λ)` to `3.9·10⁻¹²` (max abs) on 420 cases with `λ ∈ [0, 10⁴]`, `ν ∈ [1, 200]`, `α ∈ [0.001, 0.5]` (a grid `λ ∈ {0, 10⁻⁸, 4, 100, 10³, 10⁴}` × `ν ∈ {1, 2, 3, 30, 200}` × `α ∈ {0.001, 0.01, 0.05, 0.5}` plus 300 random cases), and `ν x*/(1 − x*)` matches `scipy.stats.f.ppf` to relative `3.2·10⁻¹¹`. Independent reviewer (fresh context): `≤ 1.1·10⁻¹⁰` abs on 534 + 105 cases with `α ∈ [0.001, 0.5]`, `ν ≤ 10⁵`, `λ ≤ 10⁵` (error grows roughly with `ν`); outside the `α` range (e.g. `ν = 1, α = 10⁻⁷, λ = 100`: relative error `5.5·10⁻⁴`) accuracy degrades as item 6 says. Computed power can exceed 1 by rounding, by up to `5·10⁻¹¹` (reviewer, large `λ`); decision: returned as computed, no clamp.
8. **2ᵏ oracle (PROVED).** For `k ≥ 2` (so `ν = 2ᵏ − k − 1 ≥ 1`), `F = [1, x₁ … x_k]` on all `2ᵏ` runs of the coded `±1` full factorial: `M = 2ᵏ I`, `(M⁻¹)ⱼⱼ = 2⁻ᵏ`, so `λⱼ = 2ᵏ δⱼ²` and equal `δⱼ` give equal powers for all `k + 1` coefficients.
9. **API.** `design_diagnostics(F, ids, *, region=None, effect=None, alpha=0.05)`. `effect=None` (default) returns exactly the §6 dict. Otherwise the dict gains `power`, a length-`p` tensor in `F`'s dtype and device, column order of `F` (all `p` columns, including any intercept; unlike `columns`, nothing is excluded). `effect` is a Python float or a tensor broadcastable to `(p,)`; `(M⁻¹)ⱼⱼ = cholesky_inverse(L).diagonal()` from the Cholesky factor already computed. Trust boundary: `alpha` must be a Python float with `0.001 ≤ alpha ≤ 0.5` (item 6) and `n > p` (`ValueError` otherwise); nothing else is validated. Cost `O(p³ + p·(ν + K) + 64·ν)`.
10. **Not in this section.** Power for blocked or split-plot designs, joint (multi-degree-of-freedom) F-test power, one-sided tests, unknown-σ planning (σ is fixed at 1), Lenth / unreplicated-factorial power, sample-size search, Monte Carlo power, a public quantile or incomplete-beta function.

## 12. Blocked A and I (W9 B1)

**Chief Architect · 2026-11-16 · Week 9 goal B1.** Status: `P2_B1_MATH_FROZEN` once the independent fresh-context review verdict is recorded in its dispatch Issue and this section is committed byte-identically. Baseline `main=0f17161`. Labels as above; model, notation and moves as in §1 (fixed block effects, `F` without an intercept column, `M_B` of §1.1). This section supersedes the "D only" line of §2 and the "A/I blocked criteria" item of §4.

1. **Criteria (decision).** Minimize `φ_B(M_B) = tr(B M_B⁻¹)` with **A**: `B = I_p`; **I**: `B = W_c = (1/m) Σ_{j=1..m} (f_j − f̄)(f_j − f̄)ᵀ`, `f̄` the mean of all `m` candidate rows (the centered uniform candidate moment matrix). Public sign as in the W1 gate: `objective = −φ_B`. Rationale for centering: `φ_{W_c}` is the candidate-average variance (in units of `σ²`) of the estimated contrast `(f(x) − f̄)ᵀθ̂`, the part of a prediction that does not depend on the unknown block effect.
2. **Shift invariance (PROVED).** For any constant vector `a`, replacing every row `f` by `f + a` leaves `M_B` (block means shift by `a`) and `W_c` (the candidate mean shifts by `a`) unchanged, so both objectives and every search decision are invariant to recentering the features. (The uncentered `FᵀF/m` of the unblocked I criterion is not invariant here, which is why it is not used.) Fraction check: `M_B` and `W_c` equal before and after a random integer shift.
3. **Rank-2 gain (PROVED, Woodbury).** For the current `M_B ≻ 0` and symmetric `B`, both moves of §1.3–1.4 have the form `M_B′ = M_B + W K Wᵀ`, `W[p,2]`, `K` symmetric with `det K = −1` (within-block: `K = [[1 − 1/n_b, 1/n_b], [1/n_b, −1 − 1/n_b]]`, `W = [u v]`; interchange: `K′ = [[−(1/n_r + 1/n_t), −1], [−1, 0]]`, `W = [d e]`). With `G = WᵀM_B⁻¹W`, `H = WᵀM_B⁻¹ B M_B⁻¹W` and `N = K⁻¹ + G`, where `K⁻¹ = [[1 + 1/n_b, 1/n_b], [1/n_b, −1 + 1/n_b]]` resp. `K′⁻¹ = [[0, −1], [−1, 1/n_r + 1/n_t]]`:
   - `det M_B′ / det M_B = det(I₂ + K G) = −det N` (matrix determinant lemma, `det K = −1`);
   - if `det N ≠ 0`: `φ_B(M_B′) = φ_B(M_B) − γ`, `γ = tr(N⁻¹ H) = (N₂₂H₁₁ − 2N₁₂H₁₂ + N₁₁H₂₂) / det N`.
   Fraction check: in 300/300 random moves (`m = 9, p = 3`, blocks `[3,4]`, `[4,3]`, `[2,3,2]`, `B ∈ {I, W_c}`) the ratio and `φ_B(M_B) − γ` equal the direct `det M_B′ / det M_B` and `tr(B M_B′⁻¹)`.
4. **Positive definiteness from the ratio (PROVED).** `det K < 0`, so `WKWᵀ = aaᵀ − bbᵀ` for some vectors `a, b`. By Cauchy interlacing `M_B − bbᵀ` has at most one non-positive eigenvalue and adding `aaᵀ` does not decrease any eigenvalue, so `M_B′` has at most one non-positive eigenvalue; hence `M_B′ ≻ 0 ⟺ det(I₂ + KG) > 0`. A move is scored only if this ratio is `> 0` (then `det N < 0 ≠ 0`), and it improves iff `γ > 0`. Fraction check: leading-minor positive definiteness agrees with the ratio sign in 300/300 moves.
5. **2³ oracle (PROVED).** The §1.5 case (`F` = `A, B, C, AB, AC, BC`, `n = m = 8`, blocks `[4, 4]`): `W_c = FᵀF/8 = I₆` (orthogonal ±1 columns with mean 0). By §1.2 `M_B ≼ 8I₆`, so `M_B⁻¹ ≽ I₆/8` and `φ_A = φ_I = tr(M_B⁻¹) ≥ 6/8 = 3/4`, with equality iff `M_B = 8I₆`, i.e. exactly the `ABC`-confounded split. Optimal objective `−3/4` for both A and I; reachable only by interchanges.
6. **3² exhaustive oracle (PROVED by enumeration).** Candidates: the 3² grid `(x₁, x₂) ∈ {−1,0,1}²` in `full_factorial` order (`x₁` slowest; IDs 0–8), `F = [x₁, x₂, x₁², x₂², x₁x₂]` (`p = 5`, no intercept), `W_c = diag(2/3, 2/3, 2/9, 2/9, 4/9)`. Exact `Fraction` enumeration over all designs (distinct IDs, block contents unordered):
   - blocks `[3, 4]` (`n = 7`; 1260 designs, 236 singular): `min φ_A = 70/27`, `min φ_I = 208/243`, each attained by 4 designs (the same 4, which also maximize `det M_B = 108`);
   - blocks `[4, 4]` (`n = 8`; 315 designs with the two blocks unordered, 11 singular; 630 labeled, 22 singular): `min φ_A = 121/60` (8 designs), `min φ_I = 31/45` (12 designs).
7. **Complexity (target).** Per pass, as for blocked D (§2): one Cholesky of `M_B`; exchange scores `O(m p² + n m p)` (apply `M_B⁻¹` and `M_B⁻¹BM_B⁻¹` once to the candidate rows and block means, then `G`, `H` entries per (position, candidate) pair by inner products); interchange scores `O(n p² + n² p)`; no `m×m` matrices; workspace checked against `max_elements`.

## 13. API decision: `optimal_design(F, n, *, criterion="A"|"I", blocks=…)`

- `blocks` now accepts `criterion` `"D"`, `"A"` or `"I"`; `blocks` with `fixed_ids` still raises `ValueError` (W9 B3). Validation, requirements and the start rule are exactly §2 (the same draws for the same `generator`, whatever the criterion). CPU float64 only.
- Search: the §2 first-improvement order and moves, unchanged; for A/I a move is tried iff its §12.3 ratio is `> 0` and `γ > 0`, and accepted iff a dense recomputation (Cholesky of the new `M_B`, `tr(B M_B′⁻¹)` by `cholesky_solve`) improves `φ_B` by more than the existing `_tolerance` (called as in the unblocked A/I path). D: byte-identical to `main` (IDs, objectives, stop reasons).
- Result: as §2, with `criterion` the requested criterion and, for A/I, `objective = −φ_B(M_B)`; `certificate="LOCAL_BLOCK_EXCHANGE"` if no improving move remains else `"NOT_CERTIFIED"`; `efficiency_lower_bound=None` (W9 B3 decides any floor).

## 14. Held-out additions W9 (frozen 2026-11-16; never used for tests, tuning or dispatch choices; run once at week end, reported as-is)

Coding: `X = full_factorial([3, 3, 3]).double() − 1` (codes −1, 0, 1; column 0 slowest). Seed `generator=torch.Generator().manual_seed(20261117)`; all other arguments at defaults.

| Case | Definition | Record |
|---|---|---|
| **H19** | `F = polynomial_features(X, 2, include_bias=False)` (27 candidates, `p = 9`); `optimal_design(F, 16, criterion="I", blocks=[8, 8])` | IDs per block, `φ_{W_c}` (§12.1), stop reason, per-restart time. Reference: R AlgDesign `optBlock(~quad(A,B,C), withinData=<same 27 rows>, blocksizes=c(8,8))` only if its installed documentation lists an I criterion; otherwise `NOT_RUN` (reason recorded). If run: versions (`sessionInfo()`), its blocks re-scored by the shared TorchOED `φ_{W_c}` on `M_B`. |
| **H20** | `F = polynomial_features(X, 2)` (with intercept, `p = 10`), `hard = X[:, 0]` (column 0 hard to change), `optimal_design(F, 12, criterion="I", whole_plots=[3, 3, 3, 3], hard=hard, eta=1.0)` (needs W9 B2) | IDs per whole plot, the B2-gate GLS I objective, stop reason, per-restart time. Reference: R skpr 1.9.2, `wp <- gen_design(data.frame(A=c(-1,0,1)), ~A+I(A^2), trials=4, optimality="I")`, then `gen_design(expand.grid(B=c(-1,0,1), C=c(-1,0,1)), ~(A+B+C)^2+I(A^2)+I(B^2)+I(C^2), trials=12, splitplotdesign=wp, optimality="I")` (skpr default variance ratio 1), versions recorded; its design re-scored by the shared TorchOED GLS I metric. Not reproducible ⇒ `NOT_RUN`. |

## 15. Not in this section

`fixed_ids` with blocks, an efficiency floor or certificate for blocked designs (W9 B3), structure-aware diagnostics (W9 B4), split-plot A/I (W9 B2 gate), a user-supplied region matrix `B`, GPU or float32 blocked search, random block effects.

## 16. Split-plot A and I (W9 B2)

**Chief Architect · 2026-11-16 · Week 9 goal B2.** Status: `P2_B2_MATH_FROZEN` once the independent fresh-context review verdict is recorded in its dispatch Issue and this section is committed byte-identically. Labels, model, feasibility, `M` (§7.1), `c_k = η/(1 + ηn_k)` and moves (§7.5) as in §7; `F` may contain an intercept. This section supersedes the "A/I split-plot criteria" item of §10 and the `criterion == "D"` requirement of §8.

1. **Criteria (decision).** Minimize `φ_B(M) = tr(B M⁻¹)`, `M = FₛᵀV⁻¹Fₛ`, with **A**: `B = I_p`; **I**: `B = W = FᵀF/m`, the uncentered uniform moment matrix of all `m` candidate rows (the W1 convention). Public sign as in the W1 gate: `objective = −φ_B`. Rationale for not centering (unlike §12.1): the whole-plot effect is random with mean 0, so the mean response `f(x)ᵀθ` is estimable and its GLS estimator has variance `σ² f(x)ᵀM⁻¹f(x)`; `φ_W` is its candidate average in units of `σ²`.
2. **Within-plot exchange is a rank-2 update with `det K = −1` (PROVED).** In plot `k` (size `n_k`, current sum `s = s_k`, `c = c_k`) replace design row `f_o` by a candidate `f_j`, `d = f_j − f_o`. The plot's information is `F_kᵀ(I − c11ᵀ)F_k` and the row changes by `d`, so with `q = F_kᵀ(I − c11ᵀ)e_o = f_o − c s` and `e_oᵀ(I − c11ᵀ)e_o = 1 − c`: `ΔM = d qᵀ + q dᵀ + (1 − c) d dᵀ = W K Wᵀ`, `W = [d q]`, `K = [[1 − c, 1], [1, 0]]`, `det K = −1`, `K⁻¹ = [[0, 1], [1, −(1 − c)]]`. This is the same `ΔM` as §7.4 (whose 3×3 kernel is singular). The identity does not use labels.
3. **Ratio and gain (PROVED, as §12.3).** For `M ≻ 0` and symmetric `B`, with `G = WᵀM⁻¹W`, `H = WᵀM⁻¹BM⁻¹W`, `N = K⁻¹ + G`: `det M′ / det M = det(I₂ + KG) = −det N`; if `det N ≠ 0`, `φ_B(M′) = φ_B(M) − γ`, `γ = tr(N⁻¹H) = (N₂₂H₁₁ − 2N₁₂H₁₂ + N₁₁H₂₂) / det N`. By the §12.4 argument (`det K < 0`), `M′ ≻ 0 ⟺ det(I₂ + KG) > 0`. Fraction check (Architect): in 600/600 random exchanges (`p = 2…4`, plots `[3,3]`, `[2,3,2]`, `[4,2]`, `[3,2,2]`, rational `η`, `B ∈ {I, W}`) `ΔM = WKWᵀ`, the ratio equals `−det N` and the direct `det M′ / det M`, `φ_B(M) − γ` equals the direct `tr(B M′⁻¹)`, leading-minor positive definiteness of `M′` agrees with the ratio sign, and `M` of §7.1 equals the dense `FₛᵀV⁻¹Fₛ`.
4. **Relabel for A/I (decision).** As §7.5(ii), except that each step takes, among the unused label-`ℓ′` candidates whose §16.3 ratio from the current proposal state is `> 0`, the one with the largest `γ` (ties: smallest candidate ID); if no candidate has ratio `> 0` the proposal is discarded. By §16.3 every intermediate state is positive definite. Local optimality is only with respect to the two §7.5 moves, as for D.
5. **Plots of size 1 (PROVED).** If every `n_k = 1`, then `c_k = η/(1 + η)` and `M = FₛᵀFₛ/(1 + η)`, so `φ_B(M) = (1 + η)·tr(B(FₛᵀFₛ)⁻¹)`. Fraction check (Architect): 20/20 random cases.
6. **Tiny exhaustive oracles (PROVED by enumeration).** The §7.6 candidates (`(h, x)`, `h ∈ {−1, 1}`, `x ∈ {−1, 0, 1}`, listed twice, `m = 12`, IDs 0–11), `F = [1, h, x, x², h·x]` (`p = 5`), labels `h`, `W = FᵀF/m`. Exact `Fraction` enumeration (Architect) over all feasible designs (distinct IDs, plot contents unordered, plots of equal size unordered):
   - whole plots `[3, 3]`, `η = 1` (420 designs, 164 singular): `min φ_A = 35/12`, `min φ_I = 11/6`, each attained by 64 designs (the same 64 that attain `max det M = 48` of §7.6);
   - whole plots `[2, 2, 2]`, `η = 1/2` (1380 designs, 516 singular): `min φ_A = 71/24`, `min φ_I = 103/72`, each attained by the same 64 designs (one plot `{x = −1, x = 1}` at one label and two plots `{x = −1, x = 0}`, `{x = 0, x = 1}` at the other); `max det M = 96` is attained by 192 designs, so here the A- and I-optimal sets coincide and are a strict subset (64 of 192) of the D-optimal set.
7. **Complexity (target).** Per pass as §8: one Cholesky of `M`; exchange scores `O(m p² + n m p)` (apply `M⁻¹` and `M⁻¹BM⁻¹` once to the candidate rows and plot sums, then the `G`, `H` entries per (position, candidate) pair by inner products); each relabel proposal `O(n_k (p³ + m p²))`; no `m×m` or `n×n` matrices; workspace checked against `max_elements`.

## 17. API decision: `optimal_design(F, n, *, criterion="A"|"I", whole_plots=…, hard=…, eta=…)`

- `whole_plots` now accepts `criterion` `"D"`, `"A"` or `"I"`; `whole_plots` with `blocks` or `fixed_ids` still raises `ValueError` (W9 B3). Validation, requirements and the start rule are exactly §8 (the same draws for the same `generator`, whatever the criterion). CPU float64 only.
- Search: the §8 order (exchanges first, then relabels). For A/I an exchange is tried iff its §16.3 ratio is `> 0` and `γ > 0`; a relabel proposal is built by §16.4. A move is accepted iff a dense recomputation (§7.1 `M` of the new design, Cholesky, `tr(B M′⁻¹)` by `cholesky_solve`) improves `φ_B` by more than the existing `_tolerance` (called as in the unblocked A/I path). D: byte-identical to `main` (IDs, objectives, stop reasons).
- Result: as §8, with `criterion` the requested criterion and, for A/I, `objective = −φ_B(M)`; `certificate="LOCAL_SPLIT_PLOT_EXCHANGE"` if no improving move remains else `"NOT_CERTIFIED"`; `efficiency_lower_bound=None` (W9 B3 decides any floor).
- H20 (§14) clarification, frozen with this section: "the B2-gate GLS I objective" is `φ_W` of §16.1, and `hard = X[:, 0]` is passed as `X[:, 0].long()` (§8 requires an integer tensor; the labels −1, 0, 1 and the partition are unchanged).

## 18. Not in this section

`fixed_ids` with whole plots, an efficiency floor or certificate for split-plot designs (W9 B3), structure-aware diagnostics or power (W9 B4), whole plots together with `blocks`, a user-supplied region matrix `B`, unknown `η`, between-plot interchange, split-split-plot, GPU or float32 split-plot search.

## 19. Structure-aware diagnostics and power (W9 B4)

**Chief Architect · 2026-11-17 · Week 9 goal B4.** Status: `P2_B4_MATH_FROZEN` once the independent fresh-context review verdict is recorded in its dispatch Issue and this section is committed byte-identically. Labels as in §6 and §11 (`S = F[ids]`, `n × p`, distinct IDs; `region` rows `R`, `N × p`, default `R = F`; `J` and `Δ` as in §6.4: the columns of `S` that are not constant over the design rows, exact float comparison). Blocks as in §1 (fixed block effects, `g` blocks of sizes `n₁…n_g`, rows of block `b` contiguous in `ids`, `F` without an intercept, `M_B` of §1.1). Whole plots as in §7 (`w` plots of sizes `n₁…n_w`, rows of plot `k` contiguous in `ids`, `V = I + ηZZᵀ`, `c_k = η/(1 + ηn_k)`, `M = SᵀV⁻¹S` of §7.1). Unit error variance `σ² = 1` throughout. This section supersedes the "Power for blocked or split-plot designs" item of §11.10.

1. **Information (decision).** Blocks: `M_B`. Whole plots: the GLS `M` of §7.1. Each is factored once by Cholesky; a singular information raises from `torch.linalg.cholesky`, as in §6.
2. **SPV and FDS (decision; PROVED identities).**
   - Blocks: `SPV(x) = n·(f(x) − f̄_R)ᵀM_B⁻¹(f(x) − f̄_R)`, `f̄_R` the mean of the `N` region rows (the block effect is unknown, so only the contrast with the region mean is predicted; cf. §12.1). Identity: `(1/N) Σ_{x∈R} SPV(x) = n·tr(W_R M_B⁻¹)`, `W_R = (1/N) Σ_{x∈R}(f(x) − f̄_R)(f(x) − f̄_R)ᵀ`; for `region=None` this is `n·φ_I(M_B)` of §12.1.
   - Whole plots: `SPV(x) = n·f(x)ᵀM⁻¹f(x)` (uncentered, as §16.1). Identity: for `region=None`, `(1/m) Σ_{x∈F} SPV(x) = n·φ_W(M)` of §16.1.
   - `fds` is `spv` sorted ascending, as §6.6.
3. **Correlation and VIF (decision; PROVED identities).** Let `X = S[:, J]` and let `C` be the structure-adjusted cross-product matrix of `X`:
   - Blocks: `C = Xᵀ(I − Z(ZᵀZ)⁻¹Zᵀ)X = M_B[J, J]` (within-block centring).
   - Whole plots: `C = XᵀPX`, `P = V⁻¹ − V⁻¹1(1ᵀV⁻¹1)⁻¹1ᵀV⁻¹` (GLS centring). Closed form: `C = Σ_k [X_kᵀX_k − c_k t_k t_kᵀ] − u uᵀ/τ`, `t_k = Σ_{i∈k} xᵢ`, `u = Σ_k t_k/(1 + ηn_k)`, `τ = Σ_k n_k/(1 + ηn_k)`; no `n × n` matrix.
   With `Δ = diag(C)`: `correlation = Δ^{−1/2}CΔ^{−1/2}` and `vif = diag(correlation⁻¹)`, computed by Cholesky as §6.4; as there, a `C` that is not numerically positive definite raises (possible with `M ≻ 0`, e.g. whole plots with `F = [x, 1 − x]` and no intercept).
   - Blocks identity (PROVED): `vif_j = C_jj / RSS_j`, `RSS_j` the residual sum of squares of the least-squares regression of column `j` on the `g` block indicators and the other columns of `J` (`C_jj` is the within-block sum of squares, so `vif_j = 1/(1 − partial R_j²)`). Fraction check (Architect): 120/120 columns in 40 random cases (`p = 3`, blocks `[3,4]`, `[4,3]`, `[2,3,3]`).
   - Whole plots identity (PROVED): `vif_j = 1/(1 − R_j²)`, `R_j²` the GLS coefficient of determination (metric `V⁻¹`, total sum of squares about the GLS mean) of column `j` regressed on an intercept and the other columns of `J`; equivalently `C⁻¹ = (M̃⁻¹)[J, J]` for `M̃ = [1 X]ᵀV⁻¹[1 X]`. Fraction check (Architect): the closed form of `C` equals the dense `XᵀPX`, and both identities hold, in 40/40 random cases (`p = 3`, plots `[3,3]`, `[2,3,2]`, `[4,2]`, rational `η`).
   - At `η = 0`, `P = I − 11ᵀ/n` and `C` is the §6.4 `XᵀX` of centred columns (consistency check only; `η > 0` is required by §8).
4. **Blocked power (PROVED).** The model has `g + p` free parameters (block effects absorb the intercept) and the design matrix `[Z S]` has full column rank iff `M_B ≻ 0`, so the residual mean square has `ν = n − g − p` degrees of freedom and is independent of `θ̂ = M_B⁻¹ S̃ᵀy` (normal errors, `S̃` within-block centred). The two-sided level-`α` t-test of `θⱼ = 0` therefore has `powerⱼ = P(F′(1, ν, λⱼ) > F_{1−α}(1, ν))`, `λⱼ = δⱼ² / (M_B⁻¹)ⱼⱼ`, evaluated by §11.2–11.6. Requires `ν ≥ 1`.
5. **Split-plot power (decision: containment rule; PROVED exact in the orthogonal balanced case).** Coefficient `j` is **whole-plot** if column `j` of `S` is constant within every whole plot (exact float comparison; an intercept is whole-plot), else **subplot**. With `p_w` whole-plot and `p_s = p − p_w` subplot columns: `νⱼ = w − p_w` for whole-plot and `νⱼ = n − w − p_s` for subplot coefficients; `λⱼ = δⱼ² / (M⁻¹)ⱼⱼ` with the GLS `M`; `powerⱼ` by §11.2–11.6 with `νⱼ`. Requires every `νⱼ ≥ 1`.
   - **Exact case (PROVED).** If all plots have one size `k` and every subplot column sums to zero within every plot (`ZᵀS_s = 0`), then `V⁻¹S_s = S_s` and `V⁻¹S_w = S_w/(1 + ηk)`, so `M = diag(S_wᵀS_w/(1 + ηk), S_sᵀS_s)` (GLS = OLS). The whole-plot coefficients are the least-squares fit of the plot means on the plot rows of `S_w`, whose residual mean square on `w − p_w` degrees of freedom estimates `Var(ȳ_k) = (1 + ηk)/k`; the subplot coefficients are the fit of the within-plot deviations on `S_s`, residual mean square on `n − w − p_s` degrees of freedom estimating `1`. The strata are independent when `γ` and `ε` are jointly normal, so each stratum t-test has exactly the stated `F′(1, νⱼ, λⱼ)` law, with `Var θ̂ⱼ = (M⁻¹)ⱼⱼ`.
   - **Otherwise APPROXIMATE (decision).** Unequal plot sizes or subplot columns not orthogonal to the plots: the same `λⱼ` and `νⱼ` are used and the power is an approximation (no Kenward–Roger or Satterthwaite correction this week); the docstring says so and the result reports `df`. The classification depends on how the columns are written (e.g. `[1, h, x]` vs `[1, h + x, x]` give different `p_w`), which is another reason the rule is approximate outside the exact case. Exactness refers to the stratum (ANOVA) t-tests, not to REML-based tests.
   - Architect check (OBSERVED, float64 + NumPy/SciPy dev-only, 2·10⁵ simulated responses): the §19.6 design at `η = 3/2`, `α = 0.05`, `δ = (0, 0.9, 0.6, 0)`: simulated rejection rates `0.0496, 0.1212, 0.1731, 0.0495` vs exact `0.05, 0.1221, 0.1744, 0.05`.
6. **Oracles (PROVED).**
   - Blocked 2³: `F = [A, B, C]` on all 8 runs of the coded 2³ factorial, blocks `[4, 4]` confounded with `ABC` (`n = 8`, `g = 2`, `p = 3`): `M_B = 8I₃`, `correlation = I₃`, `vif = 1`, `SPV = 3` at every run (`f̄_R = 0`, `region=None`), `ν = 3`, `λⱼ = 8δⱼ²`.
   - Split-plot 2²: `F = [1, h, x, hx]`, whole plots `[2, 2, 2, 2]` at `h = −1, −1, 1, 1`, each plot `x = −1, 1` (`n = 8`, `w = 4`, `p_w = 2`, `p_s = 2`): exact case of item 5, `M = diag(8/(1 + 2η), 8/(1 + 2η), 8, 8)`, whole-plot `νⱼ = 2`, `λⱼ = 8δⱼ²/(1 + 2η)`, subplot `νⱼ = 2`, `λⱼ = 8δⱼ²`; `J = [1, 2, 3]`, `C = diag(8/(1 + 2η), 8, 8)`, `correlation = I₃`, `vif = 1`; `SPV = 2(1 + 2η) + 2` at every design run.
7. **Complexity (target).** `O(n p² + N p² + p³)` plus `O(p(ν + K) + 64ν)` per distinct `ν` for power; no `n × n`, `N × N` or `m × m` matrices.

## 20. API decision: `design_diagnostics(F, ids, *, region=None, effect=None, alpha=0.05, blocks=None, whole_plots=None, eta=1.0)`

- `blocks` and `whole_plots` are `None` (default: results byte-identical to `main`) or lists/tuples of positive Python ints summing to `len(ids)` (trust boundary, `ValueError` otherwise); giving both raises `ValueError`. `eta` is used only with `whole_plots` and is validated as in §8. The row order of `ids` defines the blocks / plots, as returned by `optimal_design`.
- With either structure the dict has the §6 keys (`columns`, `correlation`, `vif`, `spv`, `fds`) computed by §19.1–19.3, and with `effect` also `power` (§19.4–19.5) and `df`, a length-`p` int64 tensor of the `νⱼ` used, column order of `F`. Power trust boundary: `alpha` as §11.9; `ν ≥ 1` (blocks) or every `νⱼ ≥ 1` (whole plots), else `ValueError`.
- Without a structure argument nothing changes (no `df` key).

## 21. Not in this section

Kenward–Roger or Satterthwaite degrees of freedom, unknown `η`, random block effects, blocks together with whole plots, joint F-test power, structure-aware efficiency floors (W9 B3), GPU or float32 structured diagnostics.

## 22. Efficiency floors for blocked and split-plot designs (W9 B3, part 1)

**Chief Architect · 2026-11-17 · Week 9 goal B3.** Status: `P2_B3A_MATH_FROZEN` once the independent fresh-context review verdict is recorded in its dispatch Issue and this section is committed byte-identically. Labels as above. Blocked notation is §1 and §12 (`F` without an intercept, `M_B` of §1.1, `B ∈ {I_p, W_c}`). Split-plot notation is §7 and §16 (`M = FₛᵀV⁻¹Fₛ`, `B ∈ {I_p, FᵀF/m}`). The dual floor is that of the [W1 gate §4b](W1_GENERIC_CRITERIA_GATE.md): weights `w ≥ 0`, `Σ w_j = 1`, `M_w = Σ_j w_j g_j g_jᵀ` positive definite, on candidate vectors `g_j`. This section supersedes `efficiency_lower_bound=None` in §2, §8, §13 and §17 and the "efficiency floor" items of §4 and §10.

1. **Efficiency (definition).** Let `M(S)` be the structured information of a feasible design `S`: `M_B` (blocks) or the GLS `M` (whole plots), with the block or plot sizes fixed. Let `M°` be any feasible design with the same sizes. D-efficiency of `S` is `(det M(S) / sup det M°)^{1/p}`. For A and I it is `inf φ_B(M°) / φ_B(M(S))`, `φ_B(M) = tr(B M⁻¹)`. Both are at most 1.
2. **Domination (PROVED).**
   - **Blocks:** for any constant `a ∈ ℝᵖ` and every blocked design `S`, `M_B(S) ≼ Σ_{i∈S} (f_i − a)(f_i − a)ᵀ`. Proof: in block `b`, `Σ_{i∈b}(f_i − a)(f_i − a)ᵀ = Σ_{i∈b}(f_i − f̄_b)(f_i − f̄_b)ᵀ + n_b (f̄_b − a)(f̄_b − a)ᵀ`; sum over `b`.
   - **Whole plots:** `V = I + ηZZᵀ ≽ I`, so `V⁻¹ ≼ I` and `M(S) = FₛᵀV⁻¹Fₛ ≼ FₛᵀFₛ`.
   - **Combined:** with `g_j = f_j − f̄` (blocks; `f̄` the mean of all `m` candidate rows) or `g_j = f_j` (whole plots), `M(S) ≼ n·M̄_g(S)`. Here `M̄_g(S) = (1/n) Σ_{i∈S} g_i g_iᵀ` is the moment matrix of the uniform approximate design on `S` over the vectors `g_j`.
3. **Floor (PROVED).** Let `M*_g` be the optimal approximate moment matrix over the `g_j`: D-optimal for D, or the infimum of `φ_B` (W1 §4 convention) for A/I. By 2, for every feasible `S°` with the same sizes, `det M(S°) ≤ nᵖ det M*_g` and `φ_B(M(S°)) ≥ φ_B(M*_g)/n` (`φ_B` is antitone on positive definite matrices). The W1 §4b dual bounds `logdet M*_g ≤ logdet M_w + p log(d̄_w/p)` and `φ_B(M*_g) ≥ φ_B(M_w)²/q̄_w` hold for every admissible `w`. Here `d̄_w = max_j g_jᵀM_w⁻¹g_j` and `q̄_w = max_j g_jᵀM_w⁻¹BM_w⁻¹g_j`. Hence:
   - **D-eff ≥ exp((logdet(M(S)/n) − logdet M_w)/p) · p/d̄_w**;
   - **A/I-eff ≥ φ_B(M_w)² / (q̄_w · n·φ_B(M(S)))**.

   These are exactly the expressions that the existing `_dual_floor(G, M, n, criterion, current)` evaluates at its iterates (`G` the rows `g_j`, `M = M(S)`). Its `B` (`I_p` for A, `GᵀG/m` for I) equals §12.1's `W_c` for blocks (`G = F − f̄`) and §16.1's `FᵀF/m` for whole plots (`G = F`). Every iterate gives a valid floor (W1 §4b), so the returned maximum is valid.

   The floor compares with the **unstructured** approximate optimum over the `g_j`, so it is conservative: it cannot be 1 unless the structured design attains that optimum. With `fixed_ids` (B3 part 2) the feasible set shrinks, so the floor stays valid.
4. **Shift (decision).** For blocks the shift is `a = f̄` (the candidate mean); any constant is valid by 2. `f̄` is chosen because it makes the I-criterion `B` of `_dual_floor` equal `W_c`. Whole plots use no shift: the whole-plot domination of item 2 is proved only for `a = 0` (and `M` is not shift-invariant when `F` has an intercept).
5. **Positive definiteness (PROVED).**
   - Blocks: every `f_i − f̄_b` is a difference of two vectors in `span{g_j}` (`f̄_b − f̄` is an average of `g_j`), so `M_B ≻ 0` implies that the `g_j` span `ℝᵖ` and the uniform `M_w` is positive definite.
   - Whole plots: `M ≻ 0` implies `F` has full column rank (§7.2), so the same holds.
   - Hence the first iterate of `_dual_floor` is admissible for every returned design.
6. **2³ oracle (PROVED).** In the §1.5/§12.5 case (`F = [A, B, C, AB, AC, BC]`, `n = m = 8`, blocks `[4, 4]`), `f̄ = 0` and `GᵀG/8 = I₆`. Every `g_jᵀg_j = 6 = p`, so the uniform `w` is D-, A- and I-optimal (`d̄_w = p`; `q̄_w = 6 = φ(M_w)`). At the `ABC`-confounded split (`M_B = 8I₆`) the floor equals 1 for D, A and I. At every other nonsingular split it equals the true efficiency, because here the structured optimum attains the item-3 bounds (`det 8I₆ = 8⁶ det M*_g`, `φ(8I₆) = φ(M*_g)/8`) and the dual bound is exact at the uniform `w`, the first `_dual_floor` iterate, where the iteration stops (ratio 1).
7. **Exhaustive evidence (OBSERVED, Architect sandbox, NumPy float64, 2026-11-17).** The design-independent part of the floor was computed once by the `_dual_floor` iteration. Every nonsingular feasible design was enumerated. Results:

   | Case | Nonsingular designs | Floor at the optimum, D / A / I | Min slack, true efficiency − floor, D / A / I |
   |---|---|---|---|
   | §12.6, blocks `[3, 4]` | 1024 | 0.867 / 0.780 / 0.810 | 0.055 / 0.0059 / 0.0063 |
   | §12.6, blocks `[4, 4]` | 304 | 0.890 / 0.878 / 0.880 | 0.052 / 0.018 / 0.017 |
   | item 6 (2³), blocks `[4, 4]` | 17 | 1 / 1 / 1 | 0 / 0 / 0 |
   | §16.6, whole plots `[3, 3]`, `η = 1` | 256 | 0.567 / 0.622 / 0.451 | 0.30 / 0.21 / 0.29 |
   | §16.6, whole plots `[2, 2, 2]`, `η = 1/2` | 864 | 0.651 / 0.614 / 0.578 | 0.24 / 0.21 / 0.18 |

   The floor was never violated. The enumerated optima reproduce §12.6 and §16.6 (`max det M_B = 108` for `[3, 4]` and `240` for `[4, 4]`; `max det M = 48` and `96`). The split-plot floors are loose because `V⁻¹ ≼ I` discards the whole-plot variance. A tighter structured bound is an open question, not claimed here.
8. **Complexity.** One `_dual_floor` call per `optimal_design` call, on the returned design only: `O(T m p²)`, `T ≤ 1000`, workspace `O(mp + p²)`. No `m×m` or `n×n` matrices.

## 23. API decision: `efficiency_lower_bound` for `blocks=` and `whole_plots=`

- Blocked and split-plot results (D, A and I) return `efficiency_lower_bound = _dual_floor(G, M, n, criterion, 0.0)` as a Python float. `M` is the returned design's `M_B`, resp. GLS `M`, recomputed once after the restarts. `G = F − F.mean(0)` for blocks and `G = F` for whole plots.
- Nothing else changes, for blocked, split-plot and unblocked calls alike: IDs, objectives, stop reasons, certificates, `restart_objectives`, `best_restart`, `starts`. The value is returned as computed, with no clamp (as W1 §4b).

## 24. Not in this section

`fixed_ids` with `blocks` or `whole_plots` (W9 B3 part 2), a structure-aware approximate optimum or any tighter split-plot floor, between-plot interchange, blocks together with whole plots, GPU or float32.

## 25. Augmentation with fixed runs for blocked and split-plot designs (W9 B3, part 2)

**Chief Architect · 2026-11-17 · Week 9 goal B3.** Status: `P2_B3B_MATH_FROZEN` once the independent fresh-context review verdict is recorded in its dispatch Issue and this section is committed byte-identically. Labels as above. Blocks as in §1, §2, §12 and §13. Whole plots as in §7, §8, §16 and §17. Floor as in §22 and §23. `fixed_ids` is validated as in the unblocked path: a 1-D integer sequence of distinct candidate IDs in `[0, m)` with `n_f = len(fixed_ids) < n`. This section supersedes the "`fixed_ids` raises" lines of §2, §8, §13 and §17 and the `fixed_ids` item of §24.

1. **Fixed positions (decision).** Design positions are laid out block by block (plot by plot), as in the returned `ids` of §2 and §8: positions `n₁ + … + n_{b−1}` to `n₁ + … + n_b − 1` form block (plot) `b`. The fixed runs occupy positions `0 … n_f − 1` in the given order, so fixed run `i` lies in the block (plot) that contains position `i`. A position is **fixed** iff it is `< n_f`, else **free**. Hence the blocks (plots) before position `n_f` are fully fixed, at most one block (plot) is partly fixed (it holds position `n_f − 1` and position `n_f`), and the rest are fully free. Typical use: augment an existing blocked design by new blocks. Pass its returned `ids` as `fixed_ids` and its `blocks` followed by the new block sizes as `blocks`.
2. **Feasibility (decision; trust boundary).**
   - Whole plots: all fixed runs in one plot must have the same hard label, else `ValueError`. That label is the plot's label. Its free positions take only candidates with that label, and the plot is never relabelled.
   - Blocks: no condition beyond §2.
3. **Moves (decision; PROVED invariant).** Only free positions change:
   - an exchange (§1.3; §7.4 / §16.2) is scored and tried only at a free position;
   - a block interchange (§1.4) only between two free positions;
   - a relabel (§7.5(ii), §16.4) only for a plot with no fixed position.

   The scoring identities and the acceptance rules are unchanged, because they hold for any rows. The search replaces positions in place and never reorders them; per-block sorting happens only in the returned `ids` (§26). Every move leaves positions `0 … n_f − 1` unchanged. By induction from the start rule (item 4), every visited design, and hence the returned design, contains each fixed run in its specified block (plot), and every visited split-plot design is feasible (§7).
4. **Start (decision).** Without `fixed_ids`, nothing changes: the same draws, byte-identical results.
   - **Blocks.** Per draw: `ids = cat(fixed, pool[torch.randperm(len(pool), generator=generator)[:n − n_f]])`, where `pool` is the non-fixed IDs in ascending order (the unblocked augmentation draw); the drawn IDs fill the free positions in draw order, which gives the random partition of §2. The positive-definiteness and redraw rule is §2, unchanged.
   - **Whole plots.** Per draw, initialise `unused` as every ID except the fixed ones. Then go through plots `k = 1 … w` in order:
     - **Fully fixed:** no draw.
     - **Partly fixed** (`f_k ≥ 1` fixed positions): no label draw; `ℓ` is the fixed runs' label. The `n_k − f_k` free positions get `pool[torch.randperm(len(pool), generator=generator)[:n_k − f_k]]`, `pool` = the unused label-`ℓ` candidates in ascending ID.
     - **Fully free:** the §8 rule (`torch.randint` label, then `torch.randperm` of its pool).

     Drawn IDs are marked used. If a pool is too small, or the §8 start rule fails, the whole design is redrawn. The failure count and its error are as in §8.
5. **Floor (PROVED).** §22.3 bounds the structured optimum over **all** feasible designs with the given sizes. The designs that contain the fixed runs at their positions are a subset, so their optimum is no better. Hence the efficiency of the returned design relative to the best design containing the fixed runs is at least its efficiency relative to all feasible designs, which is at least the §23 floor. `efficiency_lower_bound` is computed exactly as in §23 on the returned design (the fixed runs included). The certificate strings are unchanged; local optimality now refers to the item-3 moves.
6. **Exhaustive oracles (PROVED by enumeration; Architect, exact `Fraction`, 2026-11-17).** Optima are over all designs that satisfy items 1–2 and have distinct IDs. Block (plot) contents are unordered, and equal-size fully free plots are unordered. `φ_B` and `W_c` / `W` are as in §12.1 and §16.1.
   - **3² blocks** (the §12.6 candidates, IDs 0–8, `F = [x₁, x₂, x₁², x₂², x₁x₂]`), blocks `[4, 4]`:
     - (B-a) `fixed_ids = [0, 8]` (block 1 partly fixed): 105 designs, 5 singular. `max det M_B = 204`, `min φ_A = 845/408`, `min φ_I = 109/153`, each attained by 4 designs. The unconstrained optimum is `240` (§12.6), so the fixed runs bind.
     - (B-b) `fixed_ids = [0, 2, 6, 8]` (block 1 fully fixed; augmentation by one new block): 5 designs, 1 singular. `max det M_B = 48`, `min φ_A = 53/12`, `min φ_I = 11/9`, each attained by 4 designs.
   - **Split-plot** (the §7.6 candidates, IDs 0–11, `F = [1, h, x, x², h·x]`, labels `h`):
     - (S-a) whole plots `[3, 3]`, `η = 1`, `fixed_ids = [0, 6, 1]` (plot 1 fully fixed at `h = −1`, `x = −1, −1, 0`): 21 designs, 13 singular. `max det M = 8`, `min φ_A = 21/4`, `min φ_I = 7/2`, each attained by 8 designs. The unconstrained optimum is `48`.
     - (S-b) whole plots `[2, 2, 2]`, `η = 1/2`, `fixed_ids = [3]` (plot 1 partly fixed at `h = 1`, `x = −1`): 690 designs, 258 singular. `max det M = 96` (112 designs), `min φ_A = 71/24` (32 designs), `min φ_I = 103/72` (32 designs). These values equal the §16.6 unconstrained optima (the counts differ because plot 1 is distinguished: 112 of 192 and 32 of 64): this case checks the partly-fixed label rule, not binding.
7. **Complexity (target).** Per pass as in §2 and §8, with the fixed positions masked out; no new workspace.

## 26. API decision: `optimal_design(F, n, *, fixed_ids=…, blocks=… | whole_plots=…, hard=…, eta=…)`

- `fixed_ids` is now accepted with `blocks` and with `whole_plots`, for criteria D, A and I. `blocks` with `whole_plots` still raises `ValueError`. Requirements, validation and CPU float64 are as in §2, §8, §13 and §17. In addition, `fixed_ids` is validated as in the unblocked path, and the whole-plot label rule of §25.2 applies.
- Result: the keys and layout of §2 / §8, unchanged. `ids` is block by block (plot by plot), with each block (plot) sorted ascending, so each fixed run appears inside its block (plot), not necessarily first. `efficiency_lower_bound` is computed as in §23. No new key.
- Without `fixed_ids`, blocked and split-plot results are byte-identical to `main`. The unblocked `fixed_ids` path is unchanged.

## 27. Not in this section

Other ways to assign fixed runs to blocks (e.g. a per-run block index), fixed runs with blocks together with whole plots, between-plot interchange, `next_batch` with structure, GPU or float32.

## 28. Regular fractional factorials, fold-over, word-length pattern and alias matrix (W10 C1, part 1)

Notation. A two-level design is an `N × k` matrix `X` with entries in `{−1, +1}`. For a nonempty column set `S ⊆ {0, …, k−1}`, the J-characteristic is `J_S = Σᵢ Π_{c∈S} x_{ic}` (an integer, `|J_S| ≤ N`). The generalized word-length pattern is `A_j = N⁻² Σ_{|S|=j} J_S²` for `j = 1 … k`. The resolution is the smallest `j` with `A_j > 0` (`∞` if none).

1. **Construction (decision).** `fractional_factorial(m, generators)` returns `N = 2^m` rows and `k = m + g` columns, `g = len(generators)`. Columns `0 … m−1` are `ff2n(m)` (same row order). Column `m + t` is `Π_{c∈generators[t]} x_{·c}`, the elementwise product of the base columns listed in generator `t`. Generators carry no sign. Length-1 and repeated generators are allowed; they create words of length 2 (and the WLP shows them).
2. **J-characteristics of a regular design (PROVED).** Identify base column `c` with the unit vector `e_c ∈ Z₂^m`, generator column `m + t` with `u_t = Σ_{c∈generators[t]} e_c`, and the rows of `ff2n(m)` with all `z ∈ {−1,+1}^m`. Every column of `X` is the character `χ_u(z) = Π_c z_c^{u_c}`, so `Π_{c∈S} x_{·c} = χ_{u(S)}` with `u(S) = Σ_{c∈S} u_c` (mod 2). Since `Σ_z χ_u(z) = 2^m` if `u = 0` and `0` otherwise, `J_S = N` if `u(S) = 0` and `J_S = 0` otherwise. The sets `S` with `u(S) = 0` are exactly the nonempty words of the defining contrast subgroup, the `2^g − 1` symmetric differences `⊕_{t∈T} ({m+t} ∪ generators[t])` over nonempty `T ⊆ {0, …, g−1}` (they are distinct and nonempty, because each contains `m + t` exactly for `t ∈ T`, and a set `S` with `u(S) = 0` determines `T` as its generator columns). Hence `A_j` is the number of words of length `j`, and the resolution is the shortest word length.
3. **Alias matrix (decision and PROVED).** For a model matrix `F₁ = F[ids]` (`n × p`, full column rank) and terms not in the model `F₂ = G[ids]` (`n × q`), the alias matrix is `A = (F₁ᵀF₁)⁻¹ F₁ᵀ F₂`, so that `E[β̂] = β₁ + A β₂` when the true mean is `F₁β₁ + F₂β₂`. For a regular design, let every column of `F₁` and `F₂` be a product `Π_{c∈T} x_{·c}` over a column set `T` (`T = ∅` is the intercept). Then each column is a character, so `F₁ᵀF₁ = N I` exactly when no two model terms `T, T'` have `T ⊕ T'` equal to a word (or empty), and then `A_{uv} = N⁻¹ Σᵢ χχ' ∈ {0, 1}`, with `A_{uv} = 1` exactly when `T_u ⊕ T_v` is a word of item 2 or empty (empty only if an `F₂` term repeats an `F₁` term). (Entries `−1` arise only for signed generators, which are not in this section.)
4. **Full fold-over (decision and PROVED).** `fold_over(X)` returns `[X; −X]` (`2N` rows, `X` first). For any `±1` design (regular or not), `J_S([X; −X]) = (1 + (−1)^{|S|}) J_S(X)`, so with `2N` rows `A_j` is unchanged for even `j` and `0` for odd `j`. In particular, if `A₂(X) = 0` (resolution ≥ III), the fold-over has `A₁ = A₂ = A₃ = 0`, i.e. resolution ≥ IV.
5. **Single-factor fold-over (decision and PROVED).** `fold_over(X, factor=c)` returns `[X; X']` where `X'` is `X` with column `c` negated. For any `±1` design, `J_S([X; X']) = 2 J_S(X)` if `c ∉ S` and `0` if `c ∈ S`. Hence with `2N` rows `A_j([X; X']) = N⁻² Σ_{|S|=j, c∉S} J_S(X)²`: for a regular design, the words containing `c` are removed and all other words are kept.
6. **Computation (decision; target complexity).** `word_length_pattern(X)` builds `P` (`N × 2^k`) by `P ← [P, P ⊙ x_{·c}]` for `c = 0 … k−1` from `P = 1`, so column `s` of `P` is `Π_{c : bit c of s set} x_{·c}`. Then `J = P.sum(0)`, and `A_j = (Σ_{popcount(s) = j} J_s²) / N²`. Time and memory `O(N 2^k)`; `budget((N, 2^k), max_elements)` is checked before building `P`, with the existing `budget` rule. In float64 every `J_s` and every partial sum `Σ J_s² ≤ 2^k N²` is an integer below `2^53` whenever `2^k N² ≤ 2^53`, so `N² A_j` is exact in that range and the returned `A_j` is that integer divided once by `N²` (exact whenever `N` is a power of 2).
7. **Worked cases (PROVED by enumeration; Architect, exact `Fraction`, 2026-11-17).** Base factors `0, 1, 2, …`:
   - `2^(7−4)`, `m = 3`, generators `[[0,1],[0,2],[1,2],[0,1,2]]`: WLP `(0, 0, 7, 7, 0, 0, 1)`, resolution III. Full fold-over: `(0, 0, 0, 7, 0, 0, 0)`. With `F₁ = [1, x₀ … x₆]`, `F₂` = the 21 two-factor interactions in `itertools.combinations` order: `F₁ᵀF₁ = 8I` and `A` has 21 entries equal to 1, all others 0.
   - `2^(5−2)`, `m = 3`, generators `[[0,1],[0,2]]`: WLP `(0, 0, 2, 1, 0)`. Full fold-over: `(0, 0, 0, 1, 0)`. `A` as above has 6 nonzero entries.
   - `2^(6−2)`, `m = 4`, generators `[[0,1,2],[1,2,3]]`: WLP `(0, 0, 0, 3, 0, 0)`, resolution IV; `A` as above is 0.
   - `2^(5−1)`, `m = 4`, generators `[[0,1,2,3]]`: WLP `(0, 0, 0, 0, 1)`, resolution V.
   - Single-factor fold-over on every `c` of the four designs: the WLP equals the word count with words containing `c` removed (item 5).

## 29. API decision: `fractional_factorial`, `fold_over`, `word_length_pattern`, `design_diagnostics(…, alias=…)`

- `fractional_factorial(m, generators, *, dtype=torch.float64, device=None, max_elements=1_000_000) -> Tensor`. `m` is a Python int ≥ 1 (`type(m) is int`, so `bool` is rejected, as in `_validation.integer`); `generators` is a nonempty list or tuple of nonempty lists or tuples of distinct Python ints in `range(m)`; otherwise `ValueError`. Budget `(2^m, m + g)` as in `ff2n`.
- `fold_over(X, factor=None) -> Tensor`. `X` is a 2-D floating tensor; `factor` is `None` or a Python int (`type is int`) in `range(k)`; otherwise `ValueError`. Entries are not checked (items 4–5 are stated for `±1` designs; negating other values is allowed). Same dtype and device as `X`.
- `word_length_pattern(X, *, max_elements=1_000_000) -> Tensor` of length `k` holding `A_1 … A_k` (float64, device of `X`). `X` is a 2-D floating tensor with every entry `−1` or `+1`, otherwise `ValueError`; budget `(N, 2^k)`.
- `design_diagnostics(F, ids, *, alias=None, …)`: `alias` is a 2-D tensor `G` with `G.shape[0] == F.shape[0]` and `F`'s dtype and device; the result gains key `"alias"`, the `p × q` matrix of item 3 computed as `cholesky_solve(F[ids]ᵀ G[ids], L)` with the existing unstructured Cholesky factor `L` of `exact_information(F, ids) = F[ids]ᵀF[ids]` (unweighted, unregularized). A wrong row count, dtype or device, or `alias` with `blocks` or `whole_plots`, raises `ValueError`. Without `alias`, results are unchanged.
- All three functions are exported from `torchoed`, with an API entry, a guide snippet and a CHANGELOG line.

## 30. Not in this section

Choosing generators by `(k, runs)` from a published table (W10 C1, part 2), signed generators, minimum-aberration search, blocked fractional factorials, alias matrices for blocked or split-plot designs, aliasing of three-level designs, GPU timing.

## 31. Conference matrices beyond prime Paley orders, and definitive screening for k ≤ 30 (W10 C3)

This section extends §5. Labels, row order and the design `D = [C_k; −C_k; 0ᵀ]` of §5.2 are unchanged. All arithmetic is exact (Python ints); the tensor is built only at the end.

1. **Finite fields (decision).** `GF(q)` for the odd prime powers `q ∈ {9, 25, 27}` is `F_p[x]/(f)` with the fixed monic polynomials `f = x² + 1` (`q = 9`, `p = 3`), `f = x² + 2` (`q = 25`, `p = 5`) and `f = x³ + 2x + 1` (`q = 27`, `p = 3`). Each has no root in `F_p` and degree 2 or 3, so it is irreducible and the quotient is a field (PROVED, elementary). Element `a ∈ {0, …, q − 1}` is the polynomial `Σ_t d_t xᵗ`, where `d_t` is the base-`p` digit `t` of `a`. Subtraction is digitwise mod `p`, and multiplication is polynomial multiplication reduced mod `f`. For a prime `q` (`r = 1`), this is ordinary arithmetic mod `q`.
2. **Paley conference matrix over GF(q) (PROVED, Paley 1933).** For every odd prime power `q`, let `χ(0) = 0`, `χ(a) = 1` if `a = b²` for some nonzero `b ∈ GF(q)`, and `χ(a) = −1` otherwise. Let `Q[i, j] = χ(a_j − a_i)` over the element order of item 1, `ε = χ(−1)`, and `C = [[0, 1ᵀ], [ε·1, Q]]`, of order `c = q + 1`. The proof of §5.1 uses only that `GF(q)` is a field of odd order: `χ` is multiplicative, half the nonzero elements are squares, `Σ_x χ(x)χ(x + d) = −1` for `d ≠ 0`, and `χ(−1) = +1` exactly when `q ≡ 1 (mod 4)`. Hence `CᵀC = (c − 1)I`, `diag C = 0`, every off-diagonal entry is `±1`, and `C` is symmetric if `q ≡ 1 (mod 4)` and skew if `q ≡ 3 (mod 4)`. For prime `q`, item 1 makes `Q[i, j] = χ((j − i) mod q)`, which is exactly §5.1, so the existing matrices are unchanged.
3. **Doubling a skew conference matrix (PROVED).** If `S` is a skew conference matrix of order `n` (`Sᵀ = −S`, `SSᵀ = (n − 1)I`, zero diagonal, off-diagonal `±1`), then `K = [[S, S + I], [S − I, −S]]` is a skew conference matrix of order `2n`. Proof: the diagonal of `K` is that of `S` and `−S`, which is 0. Every other entry is an off-diagonal entry of `±S`, or a diagonal entry of `S ± I`, which is `±1`. `Kᵀ = [[−S, −S − I], [−S + I, S]] = −K`. Using `S + Sᵀ = 0` and `S² = −SSᵀ = −(n − 1)I`, the blocks of `KKᵀ` are: (1,1) `SSᵀ + (S + I)(Sᵀ + I) = (n − 1)I + nI`; (1,2) `S(Sᵀ − I) + (S + I)(−Sᵀ) = SSᵀ − S + S² + S = 0`; (2,2) `(S − I)(Sᵀ − I) + SSᵀ = (2n − 1)I`. So `KKᵀ = (2n − 1)I`.
4. **Covered orders (decision).** `c ∈ {4, 6, 8, 10, 12, 14, 18, 20, 24, 26, 28, 30}` uses item 2 with `q = c − 1 ∈ {3, 5, 7, 9, 11, 13, 17, 19, 23, 25, 27, 29}`. `c = 16` uses item 3 with `S` the item-2 matrix for `q = 7`, which is skew because `7 ≡ 3 (mod 4)`. The table is fixed: there is no primality or prime-power code. **No order-22 conference matrix exists (CITED, Belevitch 1950):** a conference matrix of order `n ≡ 2 (mod 4)` requires `n − 1` to be a sum of two squares, and `21` is not.
5. **Design (decision; Jones & Nachtsheim 2011).** `definitive_screening(k)` uses `c = k` for even `k` and `c = k + 1` for odd `k`, provided `c` is covered (item 4). As in §5.2, `C_k` is the first `k` columns of `C`, and `D = [C_k; −C_k; 0ᵀ]` has `2c + 1` runs. The supported `k` are 3–20 and 23–30; new are 9, 10, 15, 16 and 25–30. `k ∈ {21, 22}` raises `ValueError` with a message that names the missing order-22 conference matrix. Every other `k` raises `ValueError` as before. For every `k` supported before this section, the output is identical. §5.3 (i)–(iii) holds unchanged for every supported `k`, because its proof uses only `CᵀC = (c − 1)I`, the zero diagonal and the `±1` off-diagonal entries (items 2–3).
6. **Verification (Architect, exact Python ints, 2026-11-17).** The Architect checked all thirteen covered orders for `CᵀC = (c − 1)I`, zero diagonal and `±1` off-diagonal entries. Symmetry holds for `c ≡ 2 (mod 4)` and skewness for `c ≡ 0 (mod 4)`, including `c = 16`. §5.3 (i)–(iii) holds for every supported `k`. The output equals the current prime-Paley code for all 16 previously supported `k` (§5.2: 3–8, 11–14, 17–20, 23, 24; `q = 29` was not in §5.2's tuple, so `k = 29, 30` are new). Every nonzero element of `GF(9)`, `GF(25)` and `GF(27)` has an inverse under item 1.
7. **Cost.** At most `O(c²)` Python-int operations plus `O(q²)` field multiplications to list the squares. The largest output is `61 × 30`, and there is no budget argument, as in §5.4.

## 32. API decision: `definitive_screening(k, *, dtype=torch.float64, device=None)`

Signature, dtype, device and row order are unchanged (§5.4). Only the set of supported `k` grows, to §31.5. The docstring and `website/api.md` say "Paley or doubled-Paley conference matrix" and list the supported `k`.

## 33. Not in this section

Conference matrices of order `> 30`, orders 22 and 34 (none exist), non-Paley constructions other than item 3, padding `k = 21, 22` onto order 24, DSD blocking or fake factors, augmented DSDs (Jones & Nachtsheim 2013), matching a published DSD (evidence lane, later), and GPU timing.

## 37. Plackett–Burman designs from Hadamard matrices, N ≡ 0 (mod 4), 4 ≤ N ≤ 48 (W10 C2, part 1)

This section reuses the conference matrices of §5 and §31 (items 1–3 and the item-1 field table). All arithmetic is exact (Python ints); the tensor is built only at the end.

1. **Skew-type Hadamard matrix (PROVED).** If `S` is a skew conference matrix of order `N` (`Sᵀ = −S`, `SSᵀ = (N − 1)I`), then `H = S + I` satisfies `HHᵀ = SSᵀ + S + Sᵀ + I = NI`, and its entries are `±1` (off-diagonal entries of `S` are `±1`, diagonal entries of `H` are 1). So `H` is a Hadamard matrix of order `N`.
2. **Paley II Hadamard matrix (PROVED).** If `C` is a symmetric conference matrix of order `n` (`Cᵀ = C`, `C² = (n − 1)I`), then `H = [[C + I, C − I], [C − I, −C − I]]` has entries `±1` and `HHᵀ = 2nI`. Proof: `C` commutes with `I`, so the diagonal blocks are `(C + I)² + (C − I)² = 2C² + 2I = 2nI`, and the off-diagonal block is `(C + I)(C − I) − (C − I)(C + I) = 0`.
3. **Covered orders (decision).** With `P(q)` the §31 item-2 Paley conference matrix of order `q + 1` and `K(·)` the §31 item-3 doubling:
   - `N ∈ {4, 8, 12, 20, 24, 28, 32, 44, 48}`: item 1 with `S = P(N − 1)`, where `N − 1 ∈ {3, 7, 11, 19, 23, 27, 31, 43, 47}` is a prime power `≡ 3 (mod 4)`, so `P(N − 1)` is skew (§5.1, §31.2). `q = 27` uses the §31.1 field; the others are prime and use arithmetic mod `q`.
   - `N ∈ {16, 40}`: item 1 with `S = K(P(N/2 − 1))`, where `N/2 − 1 ∈ {7, 19}`; `K` preserves skewness (§31.3). The rows and columns of `K` are ordered as in §31.3: the first block, then the second.
   - `N = 36`: item 2 with `C = P(17)`, which is symmetric because `17 ≡ 1 (mod 4)`.
   - The table is fixed: there is no primality or prime-power code. Every other `N` raises `ValueError`. This covers every `N ≡ 0 (mod 4)` with `4 ≤ N ≤ 48`.
4. **Design (decision).** Let `H` be the Hadamard matrix that item 3 selects (built by item 1 or item 2), rows and columns indexed `0 … N − 1`. Let `G = DH` with `D = diag(−H[r, 0])`, so column 0 of `G` is all `−1`. `plackett_burman(N)` returns the `N × (N − 1)` matrix whose rows are rows `1, 2, …, N − 1, 0` of `G` (row 0 last), with column 0 dropped and columns `1 … N − 1` of `G` kept in order. For item-1 orders `H[r, 0] = −1` for `r ≥ 1` and `H[0, ·] = 1ᵀ`, so the rows `1 … N − 1` are those of `H` and the last row is all `−1`. For `N = 36` (item 2) this does not hold: `H[r, 0] = +1` except at `r = 18`, so `D` flips 35 rows, and the last row of `X` is `−H[0, 1:]` (seventeen `−1`, then `+1`, then seventeen `−1`). Implementations must apply `D` as defined, not assume an all-`−1` last row.
5. **Properties (PROVED).** `GᵀG = HᵀDᵀDH = HᵀH = NI`, because `D² = I` and `HᵀH = NI` follows from `HHᵀ = NI` for square `H`. Dropping column 0 and permuting rows keeps the remaining columns orthogonal: `XᵀX = NI_{N−1}`. Orthogonality to the dropped all-`−1` column gives `1ᵀX = 0`, so every column is balanced (`N/2` runs at each level).
6. **Cyclic form (PROVED for prime `q = N − 1`).** For `N ∈ {4, 8, 12, 20, 24, 32, 44, 48}`, write `g_0 = 1` and `g_j = χ(j)` for `1 ≤ j < q`, with `χ` over `ℤ_q` and indices taken mod `q`. Then `X[i, j] = g_{(j − i) mod q}` for `0 ≤ i < q`, and the last row is all `−1`. This is the cyclic Plackett–Burman form (Plackett and Burman 1946) with generator `g`. Proof: for `r = i + 1 ≥ 1` and `j' = j + 1`, `H[r, j'] = Q[i, j] + [i = j] = χ(j − i)` for `i ≠ j`, and `1` for `i = j`. Whether this equals a published or FrF2 table row-for-row is an evidence question (§39), not claimed here. `N ∈ {16, 28, 36, 40}` are not cyclic.
7. **Verification (Architect, and independently a fresh-context reviewer, exact Python ints, 2026-11-17).** For every covered `N`, the Architect built `X` from the §31 helpers and checked `XᵀX = NI` exactly, `±1` entries and balanced columns. The Architect also checked the item-6 cyclic form for the eight prime `q`.
8. **Cost.** `O(N²)` Python-int operations plus the §31 field cost. The largest output is `48 × 47`, and there is no budget argument, as in §5.4.

## 38. API decision: `plackett_burman(runs, *, dtype=torch.float64, device=None)`

`plackett_burman(runs)` returns the §37.4 design, of shape `(runs, runs − 1)`, as a coded `±1` tensor of the given dtype on `device` (CPU by default), consistent with `ff2n` and `definitive_screening`. `runs` is validated as an integer, like `definitive_screening`'s `k`. It is exported from `torchoed`. Users who need fewer factors take the leading columns. There is no `factors=` keyword and no randomization.

## 39. Not in this section

`N > 48`. Orders `N ≢ 0 (mod 4)`. Matching the FrF2 `pb` or published Plackett–Burman tables row-for-row (a later test against the frozen `reports/w10-rref/frf2_pb.csv`, if #247 records it). Foldover of Plackett–Burman designs (use `fold_over`). Projectivity or alias-structure claims. Central composite and Box–Behnken designs (W10 C2, part 2). GPU timing.

## 40. Central composite and Box–Behnken designs (W10 C2, part 2)

All designs are in coded units. Everything except an irrational `α` is exact; the tensor is built at the end. The block model, `M_B` and the no-intercept rule are those of §1.

1. **CCD rows (decision).** For `k ≥ 2`, center counts `n_c, n_s ≥ 0` and `α > 0`, write `n_F = 2^k`. `central_composite` returns `n_F + n_c + 2k + n_s` rows in this order:
   - (a) cube: for `r = 0 … n_F − 1`, `x_i = 2·bit_i(r) − 1` for `i = 0 … k − 1`. Bit 0 is factor 1, so factor 1 varies fastest.
   - (b) `n_c` center rows.
   - (c) axial: for `i = 0 … k − 1`, the row `−α e_i`, then `+α e_i`.
   - (d) `n_s` center rows.
   The cube block is (a)+(b), of size `n_F + n_c`. The axial block is (c)+(d), of size `2k + n_s`. Blocked functions take `blocks=[n_F + n_c, 2k + n_s]`. The cube is always the full `2^k`.
2. **`α` choices (decision).**
   - `"rotatable"`: `α = n_F^{1/4}`.
   - `"orthogonal"`: `α = √(n_F(2k + n_s) / (2(n_F + n_c)))`.
   - `"faces"`: `α = 1`.
   - A positive Python `int` or `float` is used as given.
3. **CCD moments and rotatability (PROVED).** Sums are over all rows `u`.
   - Every moment of order ≤ 4 in which some exponent is odd is 0 (this includes `Σ x_i x_j` for `i ≠ j`). The full `2^k` cube is symmetric under each sign flip `x_i → −x_i`, and so is each axial pair. Center rows add 0.
   - `Σ_u x_{ui}² = n_F + 2α²` and `Σ_u x_{ui}⁴ = n_F + 2α⁴` for every `i`, so the pure second moments are equal for all `i`.
   - `Σ_u x_{ui}² x_{uj}² = n_F` for `i ≠ j`. Axial rows have one nonzero coordinate, so only the cube counts.
   - Every remaining moment of order ≤ 4 has an odd exponent, so it is 0 by the first point. Center rows change no moment sum, so they do not affect rotatability.
   - So the Box–Hunter (1957) second-order rotatability conditions reduce to `Σ x_i⁴ = 3 Σ x_i² x_j²`, that is `n_F + 2α⁴ = 3n_F`, that is `α⁴ = n_F`. This is the `"rotatable"` `α`.
4. **Orthogonal blocking (PROVED).** Let `f_u` be the row `u` of `F_q = polynomial_features(X, 2, include_bias=False)`, `f̄` the overall mean row, `f̄_b` the mean row of block `b` (size `n_b`), and `M_c = Σ_u (f_u − f̄)(f_u − f̄)ᵀ`.
   - Splitting `f_u − f̄ = (f_u − f̄_b) + (f̄_b − f̄)` inside each block, the cross terms sum to 0 because `Σ_{u∈b} (f_u − f̄_b) = 0`. So `M_c = M_B + Σ_b n_b (f̄_b − f̄)(f̄_b − f̄)ᵀ`, with `M_B` from §1.1.
   - The added sum is positive semidefinite, and its trace is `Σ_b n_b ‖f̄_b − f̄‖²`. So `M_B = M_c` if and only if `f̄_b = f̄` for every block.
   - For the item-1 CCD with its two blocks, the linear columns `x_i` and the products `x_i x_j` (`i ≠ j`) have block mean 0 in both blocks, hence overall mean 0. In the cube, each column and each product of two distinct columns of the full `2^k` sums to 0. In the axial block, each `x_i` sums to 0, and every product `x_i x_j` is 0.
   - The column `x_i²` has block mean `n_F / (n_F + n_c)` in the cube block and `2α² / (2k + n_s)` in the axial block.
   - So `M_B = M_c` if and only if `n_F / (n_F + n_c) = 2α² / (2k + n_s)`, that is `α² = n_F(2k + n_s) / (2(n_F + n_c))`. This is the `"orthogonal"` `α` (Box and Hunter 1957).
5. **Box–Behnken rows (decision).** For `k ∈ {3, 4, 5}` and `n_0 ≥ 0` center rows:
   - For every pair `i < j`, in lexicographic order, there are four rows with `x_i = 2·bit_0(r) − 1`, `x_j = 2·bit_1(r) − 1` for `r = 0 … 3`, and all other coordinates 0.
   - The pair rows are followed by `n_0` center rows, for `4·C(k, 2) + n_0` rows (12, 24 or 40 non-center rows).
   - The block design is the set of all pairs. That it agrees with the published Box–Behnken (1960) designs is an evidence claim, not proved here: see item 7.
   - Orders `k = 6, 7` use block designs that are not all pairs. They are left to a follow-up from frozen rsm output (§42).
6. **Box–Behnken moments (PROVED).**
   - Every entry is in `{−1, 0, 1}`. Every moment of order ≤ 4 in which some exponent is odd is 0: each pair's `2²` factorial is symmetric under each sign flip, and every other coordinate is 0.
   - Factor `i` lies in `k − 1` pairs and is nonzero in 4 rows of each. So `Σ x_i² = Σ x_i⁴ = 4(k − 1)`.
   - For `i ≠ j`, only the pair `{i, j}` has both coordinates nonzero, so `Σ x_i² x_j² = 4`.
   - So the rotatability condition `Σ x_i⁴ = 3 Σ x_i² x_j²` holds if and only if `4(k − 1) = 12`, that is `k = 4`. The design with `k = 4` is rotatable, and the designs with `k = 3` and `k = 5` are not, for every `n_0`.
7. **Verification (Architect, 2026-11-17, sandbox, Python ints, `Fraction` and float64).**
   - rsm 2.10.6 `ccd` cases A, B and C of `reports/w10-rref/rsm_ccd.csv` equal the items 1–2 rows row-for-row, block labels included, within 1e-12. The calls run were `ccd(2, n0=c(3,3), alpha="rotatable", randomize=FALSE)`, `ccd(4, n0=c(4,2), alpha="orthogonal", randomize=FALSE)` and `ccd(5, n0=c(2,2), alpha="faces", randomize=FALSE)`.
   - rsm `bbd(3, n0=3, block=FALSE, randomize=FALSE)` equals item 5 row-for-row. `bbd(5, n0=3, block=FALSE, randomize=FALSE)` equals it as a multiset of rows, because rsm lists the pairs in another order (OBSERVED).
   - The item-4 block-mean equality holds exactly, with `α²` as a `Fraction`, for `(k, n_c, n_s) ∈ {(2, 3, 3), (4, 4, 2), (5, 6, 1)}`.
   - No held-out input (H22 or H23) was evaluated.
   - An independent fresh-context reviewer rebuilt both families from this text and checked items 3, 4 and 6 exactly with `Fraction` in 13 CCD cases (orthogonal, off-`α` and rotatable-only) and in Box–Behnken designs with `k ∈ {3, 4, 5}`, `n_0 ≤ 3` (held-out inputs excluded). Verdict: PASS.
8. **Estimability (OBSERVED, not a guarantee).** Zero center counts are allowed but can make the second-order model singular. A Box–Behnken design with `n_0 = 0` has `Σ_i x_i² = 2` on every row, and a CCD with `n_c = n_s = 0` has a singular blocked `M_B`. These designs are returned as specified; nothing here claims they are estimable.
9. **Cost.** `O((n_F + 2k + n_c + n_s)·k)` output elements, checked with `max_elements` as in `ff2n`. The Box–Behnken output is at most `40 + n_0` rows.

## 41. API decision: `central_composite` and `box_behnken`

- `central_composite(k, *, alpha="orthogonal", center=(4, 4), dtype=torch.float64, device=None, max_elements=1_000_000)` returns the §40.1 rows with `center = (n_c, n_s)`.
  - `k` is a Python `int ≥ 2`, and `center` is a tuple of two Python `int`s ≥ 0.
  - `alpha` is `"rotatable"`, `"orthogonal"`, `"faces"`, or a positive Python `int` or `float` that is not a `bool`. Anything else raises `ValueError`.
- `box_behnken(k, *, center=3, dtype=torch.float64, device=None)` returns the §40.5 rows. It accepts `k ∈ {3, 4, 5}` and a Python `int` `center ≥ 0`. Any other value raises `ValueError`.
- Both functions return a coded tensor of the given dtype on `device` (CPU by default), are exported from `torchoed`, and do not randomize.
- No block labels are returned: the block sizes are those of §40.1.

## 42. Not in this section

A fractional cube portion. Spherical or inscribed CCDs. Box–Behnken designs for `k = 6, 7` (a follow-up tested against frozen rsm `bbd` output) and blocked Box–Behnken designs. Randomization. Optimal augmentation of these designs. GPU timing. Any claim about the held-out cases H22 and H23 beyond what the week-end run records.

## 34. Fractional factorials by `(k, runs)` from the FrF2 catalogue (W10 C1, part 2)

This section extends §28. The design, generator convention and word-length pattern are those of §28 items 1, 2 and 6.

1. **Source (decision; CITED).** The table is, for each size, the first catalogue entry `"<k>-<p>.1"` (`p = k − m`, `m = log₂ runs`) of `FrF2::catlg`. It is frozen in `reports/w10-rref/frf2_catlg.csv` by #257, together with the FrF2 version, `sessionInfo()` and the catalogue citation copied verbatim from the installed FrF2 documentation. The covered sizes are `runs ∈ {8, 16, 32, 64}` and `m < k ≤ min(11, runs − 1)`, restricted to the rows present in that file. Each row's generator numbers are distinct. Rows that fail the check of item 3 are left out of the TorchOED literal. TorchOED holds the `gen` column as a fixed Python literal of Yates numbers. There is no search and no file read at run time.
2. **Yates conversion (decision and PROVED).** A catalogue generator number `g` (`1 ≤ g < 2^m`, at least two bits set) names the column `Π_{c : bit c of g is set} x_{·c}` of the full factorial in base factors `0 … m−1`, with bit 0 = factor A = base column 0 (Yates order: 1 = A, 2 = B, 3 = AB, 4 = C, …). Generator `t` of the TorchOED design is `bits(g_t)`, the set bits of `g_t` in ascending order, so factor `m + t` is that product. This is §28 item 1 with `generators[t] = bits(g_t)`. By §28 item 2, `A_j` of the result is the number of length-`j` words of the defining contrast subgroup generated by the words `{m + t} ∪ bits(g_t)`, `t = 0 … p−1`. Because each generator has at least two bits, every generating word has length ≥ 3. A product of two or more generating words contains at least two generator columns, so it has length ≥ 2. Hence `A₁ = 0` always. Yates bit order only names columns. The row order of `ff2n` permutes rows and leaves every `J_S` unchanged, so it does not affect the WLP.
3. **Agreement with the catalogue WLP (OBSERVED).**
   - Let `j₀` be the word length of the first entry of the catalogue `WLP` vector. The installed FrF2 help, quoted in `reports/w10-rref/sessionInfo.txt` (#257), states `j₀ = 1`: "starting with words of length 1".
   - Let `L` be the number of entries stored for the row. In the frozen file, `L = 7` for every row, so the stored vector covers word lengths `1 … 7`. It is zero-padded when `k < 7` and truncated when `k > 7` (OBSERVED in the file, not cited).
   - `word_length_pattern` returns float `A_j`. Each must equal its rounded value exactly; that check is the integer test.
   - A row agrees if all three of these hold, compared as integers:
     - `A_j` equals the stored entry for every `j` with `j₀ ≤ j ≤ min(k, j₀ + L − 1)`;
     - every stored entry for a length `j > k` is 0;
     - `A_j = 0` for every `j < j₀`.
   - The catalogue does not store lengths `j > j₀ + L − 1`, that is `A₈ … A_k` when `k > 7`, so they are never checked against it.
   - Under this rule, all 22 frozen rows agree. Two counts confirm it independently: an Architect Python-integer defining-subgroup count (#260 review) and a fresh-context reviewer's J-characteristic count (OBSERVED).
   - `reports/w10-rref/run.py` printed 7/22 because it also compared the zero padding and the truncated lengths.
   - A row that does not agree is left out of the TorchOED literal, and its `(k, runs)` raises (§35).
4. **Worked cases (PROVED by enumeration; Architect, exact `Fraction`, 2026-11-17).** `m = 3`, `g = (3, 5, 6, 7)` gives the generators `[[0,1],[0,2],[1,2],[0,1,2]]` and WLP `(0, 0, 7, 7, 0, 0, 1)`, which is the first §28 item 7 case. `m = 4`, `g = (7, 14)` gives `[[0,1,2],[1,2,3]]` and WLP `(0, 0, 0, 3, 0, 0)`, which is the third §28 item 7 case.
5. **Cost.** A table lookup and `O(p·m)` bit operations. The design cost is that of §28.

## 35. API decision: `fractional_factorial_generators(k, runs) -> list[list[int]]`

- `k` and `runs` are Python ints (`type is int`). `(k, runs)` must be a covered size of §34.1: `runs ∈ {8, 16, 32, 64}` and `m < k ≤ min(11, runs − 1)` with the row in the literal. Any other pair raises `ValueError` naming the covered sizes. This includes `k ≤ m`, which is a full factorial: use `ff2n`. The function returns a fresh list of fresh lists, `bits(g_t)` for `t = 0 … p−1` (§34.2).
- Usage: `fractional_factorial(m, fractional_factorial_generators(k, runs))` with `m = runs.bit_length() − 1`. This is the C1 `(k, runs)` path.
- It is exported from `torchoed`, with an API entry, a guide snippet and a CHANGELOG line.

## 36. Not in this section

Catalogue entries other than `.1`, `k > 11`, 128 or more runs, blocked fractional factorials, minimum-aberration search, signed generators, choosing a design by resolution alone, clear-effects or estimation-capacity criteria, and GPU timing.

## 43. Box–Behnken designs for k = 6, 7 (W10 C2, part 3)

This extends §40.5–§40.6 and §41. All entries are exact integers in `{−1, 0, 1}`.

1. **Block designs (decision; OBSERVED source).** The three-factor blocks are taken from the rsm 2.10.6 output `bbd(k, n0=3, block=FALSE, randomize=FALSE)` frozen in `reports/w10-rref/rsm_bbd67.csv` (#257). Factors are 0-based, and the blocks are listed in rsm row order:
   - `k = 6`: `(0,1,3), (1,2,4), (2,3,5), (0,3,4), (1,4,5), (0,2,5)`.
   - `k = 7`: `(3,4,5), (0,5,6), (1,4,6), (0,1,3), (2,3,6), (0,2,4), (1,2,5)`.
   TorchOED holds them as a fixed Python literal. Nothing is read from the file at run time.
2. **Rows (decision).** For each block `(a, b, c)` with `a < b < c`, in the item-1 order, there are 8 rows for `r = 0 … 7`. In each, `x_a = 2·bit_0(r) − 1`, `x_b = 2·bit_1(r) − 1`, `x_c = 2·bit_2(r) − 1`, and all other coordinates are 0. Then come `n_0` center rows. That gives `8·6 + n_0 = 48 + n_0` rows for `k = 6` and `8·7 + n_0 = 56 + n_0` for `k = 7`.
3. **Agreement with rsm (OBSERVED; Architect, 2026-11-17, sandbox, Python ints).** With `n_0 = 3`, the item-2 rows equal `rsm_bbd67.csv` row-for-row: 51 × 6 and 59 × 7.
4. **Moments (PROVED).**
   - Rotatability here means the second-order moment conditions of §40.3. All moments of order ≤ 4 with an odd exponent are 0, and `Σ x_i⁴ = 3 Σ x_i² x_j²` for all `i ≠ j`.
   - Each block's `2³` factorial is invariant under flipping any single coordinate, and every coordinate outside the block is 0. So every moment of order ≤ 4 in which some exponent is odd is 0.
   - Each factor lies in exactly 3 blocks for both `k` (a count over the item-1 literals). So `Σ x_i² = Σ x_i⁴ = 24`.
   - For `i ≠ j`, `Σ x_i² x_j² = 8 λ_{ij}`, where `λ_{ij}` is the number of blocks that contain both `i` and `j`.
   - For `k = 7`, the blocks are the lines of a Fano plane, so every pair lies in exactly one block (`λ = 1`). Then `Σ x_i⁴ = 24 = 3 · 8 = 3 Σ x_i² x_j²`, and the design is rotatable for every `n_0`.
   - For `k = 6`, `λ_{ij} ∈ {1, 2}`: three pairs, `{0,3}`, `{1,4}` and `{2,5}`, lie in two blocks. Equality fails for those pairs, so the design is not rotatable for any `n_0`.
5. **Cost.** At most `56 + n_0` rows of `k ≤ 7` entries.

## 44. API amendment: `box_behnken`

- `box_behnken(k, *, center=3, dtype=torch.float64, device=None)` now accepts `k ∈ {3, 4, 5, 6, 7}`.
- For `k = 3, 4, 5` the output is unchanged (§40.5). For `k = 6, 7` it is the §43.2 rows.
- Any other `k` raises `ValueError`. Blocked Box–Behnken designs, randomization and `k > 7` stay out of scope.
