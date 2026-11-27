# API reference

All 42 names are exported from `torchoed`; tensor shapes, numerical guards and resource limits are detailed in the linked [source contracts](https://github.com/JohnnyZPiggy/TorchOED/tree/6c9d63d3b48b5afe6a3189364e19614d9897854d/docs). Search accepts unbatched CPU float64 features; constructors expose their explicit device/dtype choices. Signatures below preserve the source parameters, defaults and annotations.

## Candidate construction

`minimum_aberration(k: int, runs: int, *, blocks: int = 1, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None) -> dict`

Return the first lexicographically minimum word-length pattern among regular fractions, optionally restricted to blocks unconfounded with main effects and two-factor interactions (§108–§109). Covered runs: 8/16/32/64, k up to 7/15/14/12; blocked k ≤ 11, blocks 2/4/8 (8 only for runs ≤ 32). Returns design, generators, wlp, block, block_generators and block_confounded (zero-based effect tuples up to order 3); infeasible or uncovered sizes raise ValueError.

`full_factorial(levels: list[int] | tuple[int, ...], *, device: torch.device | str | None = None, max_elements: int = 1_000_000) -> torch.Tensor`

Return integer level-index rows, with the last factor varying fastest.

`ff2n(k: int, *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float64, max_elements: int = 1_000_000) -> torch.Tensor`

Return the signed corners of a k-factor two-level factorial.

`fractional_factorial(m: int, generators: list[list[int] | tuple[int, ...]] | tuple[list[int] | tuple[int, ...], ...], *, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None, max_elements: int = 1_000_000) -> torch.Tensor`

Return `ff2n(m)` with one unsigned product column per generator. Each generator lists distinct zero-based base columns.

`fractional_factorial_generators(k: int, runs: int) -> list[list[int]]`

Return fresh lists of zero-based generator columns from the first FrF2 catalogue entry for the requested size. Supports k = 4–7 with 8 runs, 5–11 with 16 runs, 6–11 with 32 runs, and 7–11 with 64 runs. Other pairs raise `ValueError`. Use `ff2n` for a full factorial.

`fold_over(X: torch.Tensor, factor: int | None = None) -> torch.Tensor`

Append a copy with every factor negated, or with only the selected zero-based factor negated. Preserve dtype and device.

`word_length_pattern(X: torch.Tensor, *, max_elements: int = 1_000_000) -> torch.Tensor`

Return the generalized word-length pattern in float64 on the input device. The input has one or more rows and only −1 and +1 entries. Resolution is the first nonzero word length, or infinity when every entry is zero. The product table uses `X.shape[0] * 2**X.shape[1]` elements.

`definitive_screening(k: int, *, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None) -> torch.Tensor`

Return the (2c + 1)-run three-level definitive screening design `[C_k; −C_k; 0]` from a Paley or doubled-Paley conference matrix of order c. Supports k = 3–20 and 23–30. No order-22 conference matrix exists for k = 21 or 22.

`plackett_burman(runs: int, *, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None) -> torch.Tensor`

Return a balanced, orthogonal two-level design of shape (runs, runs − 1) with entries ±1. Supports multiples of four from 4 to 48.

`central_composite(k: int, *, alpha: str | int | float = "orthogonal", center: tuple[int, int] = (4, 4), dtype: torch.dtype = torch.float64, device: torch.device | str | None = None, max_elements: int = 1_000_000) -> torch.Tensor`

Return the coded cube followed by axial rows, with the specified center counts in each block. Choose `alpha="orthogonal"`, `"rotatable"`, `"faces"` or a positive finite number.

`box_behnken(k: int, *, center: int = 3, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None) -> torch.Tensor`

Return coded Box–Behnken rows followed by center rows for 3 to 7 factors. Uses two-factor pairs for 3 to 5 factors and three-factor blocks for 6 or 7.

`simplex_centroid(q: int, *, dtype=torch.float64, device=None) -> Tensor`

One point per nonempty component subset, by subset size then lexicographic support; `q >= 2`.

`mixture_vertices(lower, upper, *, constraints=(), face_dims=(), dtype=torch.float64, device=None) -> Tensor`

Linearly constrained simplex vertices in exact lexicographic order, then vertex-mean centroids for each requested face dimension (1 through q−1). Bounds and constraint values are int, float, Fraction or str; floats use their shortest decimal. Each `constraints` item is `(a, lo, hi)`, imposing `lo <= a @ x <= hi`; `a` has q entries and either side may be `None`, but not both. Equalities and lower-dimensional regions are allowed; dimensions above the region's add no rows. Invalid inputs or empty regions raise `ValueError`. Exact enumeration is intended for textbook sizes (q ≤ 8, K ≤ 4); see gate §126–§127 and the [mixture guide](industrial.md#mixture-designs).

`simplex_lattice(q: int, m: int, *, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None, max_elements: int = 1_000_000) -> torch.Tensor`

`lhs(n_samples: int, n_factors: int, *, generator: torch.Generator | None = None, device: torch.device | str | None = None, dtype: torch.dtype = torch.float64, mode: str = "random", restarts: int = 10, max_iter: int | None = None, max_elements: int = 1_000_000) -> torch.Tensor`

Draw a random jittered Latin hypercube (default), or use `mode="maximin"` (Morris–Mitchell φ₁₅), `mode="maxpro"` (Joseph–Gul–Ba ψ), or `mode="uniform"` (centred L2 discrepancy CD2) for midpoint designs. Optimised modes evaluate every within-column swap in batches and take the best improving swap. `restarts=10` chooses the best of independent random starts; `max_iter=None` runs each to convergence. The supplied generator controls all starts; dtype and device are retained. These modes use the box only, with no fixed runs or dual certificate.

`space_filling_diagnostics(X: torch.Tensor, *, p: float = 15) -> dict[str, torch.Tensor]`

For a floating 2-D design with n ≥ 2 rows in [0, 1]^k, returns `min_distance`, Morris–Mitchell `phi_p`, Joseph–Gul–Ba `maxpro` and centred L2 discrepancy `cd2` as scalar tensors in the input dtype and device. Shared coordinates give infinite MaxPro. Dense cost is O(n² k); see P2 gate §86–87.

`fast_flexible_filling(n: int, k: int, *, A=None, b=None, disallowed=(), n_candidates=2**14, seed=0, max_iter=100, dtype=torch.float64, device=None) -> torch.Tensor`

Select n feasible scrambled-Sobol candidates on [0, 1]^k using batched Lloyd k-means and cluster-wise MaxPro exchange. Linear constraints `A x ≤ b` and `disallowed` use `feasible_candidates`. Candidate count must be a power of two. Raises `ValueError` for too few feasible candidates or an empty final cluster. Returns points in cluster order on the requested dtype/device; each loop stops on convergence or `max_iter`. No fixed runs, dual bound or GPU timing claim; see P2 gate §89–91.

`sobol(n: int, d: int, *, scramble: bool = False, seed: int | None = None, skip: int = 0, balance: str = "exact", dtype: torch.dtype = torch.float64, device: torch.device | str | None = None, max_elements: int = 1_000_000) -> torch.Tensor`

Draw exactly n points using native CPU Sobol generation followed by the requested device transfer.

## Features and constraints

`scheffe_features(X: torch.Tensor, model: str = "quadratic", *, max_elements: int = 1_000_000) -> torch.Tensor`

Return linear, quadratic or special-cubic Scheffé columns: components, then pair products, then triple products, each lexicographically ordered. No intercept or squares. Preserves dtype, device and autograd; accepts finite floating 2-D tensors with at least two columns. Rows need not sum to one. See [mixture guide](industrial.md#mixture-designs).

`mixture_process_features(S, P, *, max_elements=1_000_000) -> Tensor`

Kronecker crossing of nonempty floating 2-D features with matching dtype/device; process row varies fastest. Preserves dtype, device and autograd; checks the output element budget. See [mixture guide](industrial.md#mixture-designs).

`polynomial_features(C: torch.Tensor, degree: int = 2, include_bias: bool = True, *, max_elements: int = 1_000_000) -> torch.Tensor`

Build bias, linear, square and pair-product columns for degree one or two while preserving autograd.

`effects_coding(codes: torch.Tensor, levels: int) -> torch.Tensor`

Encode int64 categorical codes with unit vectors and an all-minus-one final level.

`feasible_candidates(C: torch.Tensor, *, A: torch.Tensor | None = None, b: torch.Tensor | None = None, disallowed: list[dict[int, float]] | tuple[dict[int, float], ...] = ()) -> torch.Tensor`

Return a boolean mask for linear inequalities and excluded matching level combinations.

`jacobian_features(model: torch.nn.Module, X: torch.Tensor, *, params: list[str] | tuple[str, ...] | None = None, max_elements: int = 1_000_000) -> torch.Tensor`

Build detached scalar-response Jacobian rows at the module's current parameters.
Call `model.eval()` first. `params=None` selects all trainable parameters in
`named_parameters()` order; explicit names select and order flattened columns.
The module is unchanged; output has the model dtype and shape `(m,q)`, checked
against `max_elements` before differentiation. Certificates apply only to the
linearized problem at these parameters over the supplied candidates.

For Michaelis–Menten with nominal `(V,K)=(1,1)`, the local two-point D design
on this grid is `{0.5,2}`:

```python
import torch
from torchoed import jacobian_features, optimal_design, next_batch

class MichaelisMenten(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))

    def forward(self, x):
        V, K = self.theta
        return V * x / (K + x)

model = MichaelisMenten().eval()
x = torch.linspace(0.01, 2, 200, dtype=torch.float64)
F = jacobian_features(model, x)
result = optimal_design(F, 2, criterion="D", generator=torch.Generator().manual_seed(0))
assert result["ids"].tolist() == [49, 199]
new_ids = next_batch(F, result["ids"], 1, generator=torch.Generator().manual_seed(0))
```

`next_batch(F: torch.Tensor, run_ids: list[int] | tuple[int, ...] | torch.Tensor, k: int, *, criterion: str = "D", restarts: int = 10, generator: torch.Generator | None = None, **kwargs: int) -> torch.Tensor` returns sorted new int64 IDs, forwarding kwargs to `optimal_design`.
Certificates describe the full augmented design conditional on `run_ids`, for the current linearization only.

## Information

`exact_information(F: torch.Tensor, ids: torch.Tensor, *, normalized: bool = False, allow_repeats: bool = False, max_elements: int = 1_000_000) -> torch.Tensor`

Form the chosen rows' Gram matrix, optionally normalized by the number of observation events.

`weighted_information(F: torch.Tensor, w: torch.Tensor, *, max_elements: int = 1_000_000) -> torch.Tensor`

Form information from nonnegative candidate weights summing to one.

`design_diagnostics(F: torch.Tensor, ids: torch.Tensor, *, region: torch.Tensor | None = None, effect: float | torch.Tensor | None = None, alpha: float = 0.05, blocks: list[int] | tuple[int, ...] | None = None, whole_plots: list[int] | tuple[int, ...] | None = None, eta: float = 1.0, subplots: list[int] | tuple[int, ...] | None = None, eta_sub: float = 1.0, strip: tuple[int, int] | None = None, eta_row: float = 1.0, eta_col: float = 1.0, covariates: torch.Tensor | None = None, alias: torch.Tensor | None = None, groups: list[list[int]] | None = None, centered: bool = True) -> dict[str, torch.Tensor | list[int]]` returns design-column correlations and VIFs, scaled prediction variances, and their sorted values for the fraction-of-design-space curve.

Pass `blocks` and `whole_plots` together when every block boundary is a plot boundary for `M_C` correlation/VIF, centered SPV/FDS and t/F power with approximate containment df `w - g - h` (whole plot) and `n - w - (p - h)` (subplot), requiring positive residual df for power (§74–§75).
With `centered=False` (bool, unstructured only), all columns use uncentred Gram correlations and VIFs; SPV/FDS, power and alias outputs are unchanged. The default `True` retains centred diagnostics, undefined on Scheffé designs.

With `effect`, `groups=[[j, ...], ...]` adds effect-group F-test `group_power` in the given order (§56–§57). Groups must be a non-empty list of non-empty lists of distinct Python coefficient indices; overlaps are allowed. Structured designs also return int64 `group_df`; whole-plot groups must lie in one stratum. The existing `power` and `df` are unchanged. Whole-plot F power is exact in §19.5 and otherwise uses approximate containment degrees of freedom. Groups require `effect`.

Pass omitted-term columns as `alias=G` to get `(F₁ᵀF₁)⁻¹F₁ᵀF₂` under the result key `"alias"`, with `F₁ = F[ids]` and `F₂ = G[ids]`. `G` must be two-dimensional and match the row count, dtype and device of `F`. The alias calculation requires an unstructured design with a full-rank fitted model.

Pass fixed per-run `covariates=Z` (finite `(len(ids), c)` tensor, `c >= 1`, matching F dtype/device; `[1 Z]` full rank) for covariate-adjusted correlation/VIF, centered SPV/FDS and t/F power (§68–§69). The information is `M_Z = F[ids].T @ Q @ F[ids]`, eliminating `[1 Z]`; residual df is `n - 1 - c - p` and must be positive for power. Cannot combine with blocks, whole plots or alias. F has no intercept; singular information raises in Cholesky.

Pass `strip=(r, c)` with positive `eta_row`, `eta_col` for crossed GLS diagnostics and row/column/cell containment power (exact in the balanced orthogonal case with an intercept, otherwise approximate; §117–§118); whole plots, subplots, blocks, covariates and alias are excluded.

Pass nested `subplots` with `whole_plots` and positive `eta_sub` for GLS diagnostics and three-stratum containment power (§114–§115); blocks, covariates and alias are excluded.

## Criterion scores

`d_optimality(M: torch.Tensor, *, max_elements: int = 1_000_000) -> torch.Tensor`

Return logdet M for numerically positive-definite information matrices.

`a_optimality(M: torch.Tensor, *, max_elements: int = 1_000_000) -> torch.Tensor`

Return −trace(M⁻¹) through Cholesky solves.

`e_optimality(M: torch.Tensor, *, max_elements: int = 1_000_000) -> torch.Tensor`

Return the minimum eigenvalue for numerically positive-semidefinite information matrices.

`c_optimality(M: torch.Tensor, c: torch.Tensor, *, max_elements: int = 1_000_000) -> torch.Tensor`

Return −cᵀM⁻¹c for a matched contrast vector and positive-definite information matrix.

`g_optimality(M: torch.Tensor, F: torch.Tensor, *, max_elements: int = 1_000_000) -> torch.Tensor`

`v_optimality(M: torch.Tensor, F: torch.Tensor, *, max_elements: int = 1_000_000) -> torch.Tensor`

`i_optimality(M: torch.Tensor, W: torch.Tensor, *, max_elements: int = 1_000_000) -> torch.Tensor`

G and V return negative maximum and mean prediction variance; I returns `-tr(M^-1 W)`.
Inputs match batches exactly: SPD `M`, candidate rows `F`, symmetric PSD moment `W`.

## Exchange and restarted design

`glm_power(X, beta, family, *, alpha=0.05, nsim=1000, groups=None, generator=None, max_elements=10_000_000)` returns Monte Carlo Wald power and standard errors from batched float64 binomial or Poisson IRLS, with nonconverged and separated fit counts (P2 gate §111–§113).


`compare_designs(F, designs, *, region=None, effect=1.0, alpha=0.05, quantiles=(0.0, 0.25, 0.5, 0.75, 1.0), criterion="D", generator=None)`

Compare at least two unstructured designs (1-D int64 candidate IDs, repeats allowed) or Python integer run sizes, generated sequentially with `optimal_design` and one shared generator. Returns `ids`, run counts `n`, relative total-information D/A/I/G `efficiency` tensors (first design is 1; higher is better), per-coefficient `power`, `fds_quantiles` and sorted unscaled prediction-variance `fds` curves over `region` (default F). CPU float64; each design needs n > p, and the region must contain a nonzero row. Structured, Bayesian and model-robust comparisons are outside this API (P2 gate §99–§101).

`fedorov_exchange(F: torch.Tensor, ids: torch.Tensor, *, max_iter: int = 100, tile_size: int = 64, allow_repeats: bool = False, max_elements: int = 1_000_000) -> dict[str, object]`

Run directly verified one-row D exchanges from supplied IDs.

`score_two_swaps(F: torch.Tensor, ids: torch.Tensor, *, tile_size: int = 64, max_elements: int = 1_000_000) -> dict[str, object]`

Score legal simultaneous two-row D exchanges without updating the design.

`k2_exchange(F: torch.Tensor, ids: torch.Tensor, *, max_iter: int = 100, tile_size: int = 64, max_elements: int = 1_000_000, _free_positions: torch.Tensor | None = None) -> dict[str, object]`

Run deterministic one-/two-swap D search from supplied IDs; `_free_positions` is the private augmentation hook.

`d_optimal_design(F: torch.Tensor, n: int, *, fixed_ids: list[int] | tuple[int, ...] | torch.Tensor | None = None, restarts: int = 10, generator: torch.Generator | None = None, max_iter: int = 100, tile_size: int = 64, max_elements: int = 1_000_000) -> dict[str, object]`

Return the best seeded feasible D restart, its IDs, objective and conservative D-efficiency bound. `fixed_ids` accepts at most `n` distinct IDs; with exactly `n`, the call only evaluates the given design (§80).

`optimal_design(F: torch.Tensor, n: int, *, criterion: str | dict[str, float] = "D", fixed_ids: list[int] | tuple[int, ...] | torch.Tensor | None = None, restarts: int = 10, generator: torch.Generator | None = None, max_iter: int = 100, max_elements: int = 1_000_000, blocks: list[int] | tuple[int, ...] | None = None, whole_plots: list[int] | tuple[int, ...] | None = None, hard: torch.Tensor | None = None, eta: int | float = 1.0, covariates: torch.Tensor | None = None, alias: torch.Tensor | None = None, alias_weight: float = 1.0, if_possible: list[int] | None = None, tau: float = 1.0, models: list[list[int]] | None = None, model_weights: list[float] | None = None, subplots: list[int] | tuple[int, ...] | None = None, subplot_hard: torch.Tensor | None = None, eta_sub: int | float = 1.0, strip: tuple[int, int] | None = None, row_hard: torch.Tensor | None = None, col_hard: torch.Tensor | None = None, eta_row: int | float = 1.0, eta_col: int | float = 1.0) -> dict[str, object]`

With `subplots` or `strip`, `fixed_ids` fill leading positions and lock touched unit labels; returned IDs are sorted within each subplot or kept in row-major strip order (§120–§121).

Set `models=[[column indices], ...]` for CPU float64 model-robust D: maximise the weighted geometric mean of per-model D-efficiencies, with n at least the number of columns in their union. Positive `model_weights` are normalised (equal by default); `fixed_ids` are honoured. Results include the robust objective and dual floor plus `model_efficiency_lower_bounds` in model order. Only unstructured string D is supported, without `if_possible` (P2 gate §96–§98).

Set `if_possible=[column indices]` and finite positive `tau` for CPU float64 Bayesian D or I designs, including n below the total column count. Potential columns are residualised on the primary columns and scaled to range 2 (DuMouchel–Jones); `scaled_features` returns that model. `objective` is log det(XᵀX + K/τ²) for D or tr((XᵀX + K/τ²)⁻¹ W) for I, W from the real scaled candidates. Real `fixed_ids` are retained; `efficiency_lower_bound` uses P2 gate §93.4–§93.5. Structured, alias and compound options are incompatible. With `if_possible=None`, existing behavior is unchanged.

With `alias=G`, maximise `(1/p) log det(M/n) − alias_weight * ‖A‖²`, where `A = M⁻¹ F[ids]ᵀG[ids]` (§71–§73). `G` must be finite, 2-D, with the same candidate row count, dtype and device as `F`; `alias_weight` is finite and nonnegative (default 1). CPU float64 and `m ≥ n ≥ p` are required. Only string `criterion="D"` is supported; compound criteria, blocks, whole plots, covariates and hard labels raise `ValueError`. The existing compound exchange driver retains `fixed_ids` and returns sorted distinct IDs, `criterion="D"`, the penalised `objective`, `alias_trace=‖A‖²`, and `LOCAL_ALIAS_EXCHANGE` at a local optimum (else `NOT_CERTIFIED`). The efficiency floor bounds D-efficiency relative to the approximate D-optimum; it does not bound the non-concave penalised criterion. `alias=None` preserves existing results. Choose the weight explicitly; the constrained Jones–Nachtsheim form is unsupported.

A dict `criterion={"D": w_D, "A": w_A, "I": w_I}` maximises the weighted log-efficiency criterion (§65–§67). Keys are a nonempty subset of D/A/I; finite nonnegative weights are normalised to sum to one. CPU float64, unstructured designs only: blocks, whole plots and covariates are unsupported. The objective is `(w_D/p) log det(M/n) − w_A log tr(n M⁻¹) − w_I log tr(n B M⁻¹)`, with `B=FᵀF/m`. IDs are sorted and `fixed_ids` retained. Results include the normalised weight dict, `LOCAL_COMPOUND_EXCHANGE` at a local optimum (otherwise `NOT_CERTIFIED`), and a dual efficiency floor relative to the compound approximate optimum. Zero-weight terms are omitted; reference optima are not computed.


`covariates=Z` supports CPU float64 D/A/I for finite `(n, c)` fixed per-run columns with full-rank `[1 Z]`, `m ≥ n ≥ 1+c+p`, and no intercept in F (§62–§63). It cannot combine with `blocks` or `whole_plots`. IDs remain in run order, `fixed_ids` occupy leading runs, and the input `covariates` is returned unchanged. Objectives use the projected factor information, with centered candidate moments for I; `LOCAL_COVARIATE_EXCHANGE` certifies exchange/interchange local optimality and the dual efficiency floor compares with the unstructured centered approximate optimum. Covariate diagnostics remain open.


`fixed_ids` accepts at most `n` distinct IDs; with exactly `n`, the call only evaluates the given design (§80). Return the best D/A/I restart with the selected criterion's objective, efficiency lower bound and local numerical certificate. CPU float64 supports fixed blocks and whole plots together: each block is a union of consecutive plots, `eta > 0`, and features omit the intercept. `fixed_ids` fill leading positions; fixed plots retain their hard label. Results include both layouts and the centered-candidate dual floor. The local certificate covers within-plot exchanges and greedy relabels; combined diagnostics are not yet supported.

Set `subplots`, `subplot_hard` and `eta_sub` for CPU float64 split-split-plot D/A/I: subplots nest within whole plots and align with every whole-plot boundary (P2 gate §102–§103). Results include `subplots` and `eta_sub`; `LOCAL_SPLIT_SPLIT_PLOT_EXCHANGE` covers within-subplot exchanges and greedy subplot/whole-plot relabels. The OLS dual efficiency floor remains conservative. See the [split-split-plot guide](industrial.md#split-split-plot-designs); blocks, covariates, alias, Bayesian and model-robust options are unsupported.

With `strip=(r, c)`, `row_hard`, `col_hard`, `eta_row` and `eta_col`, CPU float64 D/A/I search uses an `r × c` row-major grid with `r*c = n`. The `LOCAL_STRIP_PLOT_EXCHANGE` certificate covers the local exchanges and the OLS dual efficiency floor remains conservative (§105). Whole plots, subplots, blocks, covariates, alias, Bayesian, model-robust and compound criteria are excluded; see the [strip-plot guide](industrial.md#strip-plot-designs).

## Least-squares analysis

`fit_least_squares(X, Y, *, terms=None, intercept=True)`

`residual_bootstrap(X, y, *, B=999, level=0.95, generator=None, indices=None, max_elements=10_000_000)` returns replicate coefficients and percentile intervals; `fit_least_squares_batched(X, Y)` returns coefficients and SEs over designs (§143–§144).

Fit float64 full-rank OLS with one QR for `(n,)` or `(n, J)` responses (`n > p`, identical dtype/device). Returns coefficients, SEs, t/p, residual variance/df, sequential type I ANOVA, lack-of-fit vs pure error when replicated, leverage, internally studentized residuals and Cook distances and the shared `qr_r` factor. `terms` is a tuple of positive column counts; `None` means one column each. With `intercept=True`, the first term is assumed to be the constant column alone and omitted from ANOVA. Leverage within 10 eps of 1 is set to 1 with NaN diagnostics. CPU/CUDA share code; p-values have absolute accuracy only, degrading at large df (gate §123–§124). No formula parser, rank-deficient fits, weighted/GLS/REML fits.

## Prediction profiler data

`predict_least_squares(fit, X0, *, level=0.95)`

For a single-response `fit_least_squares` result and float64 model rows `X0` of shape `(m, p)` in the original coding, return `fit`, `se` (both `(m,)`), `ci`, `pi` (both `(m, 2)`) and scalar Student quantile `t`. Requires `0.5 <= level < 1`; multi-response fits raise `ValueError`. Unweighted intervals only, CPU float64; uses the saved QR factor (§139).

`prediction_trace(fit, rows, f0, j, grid, *, level=0.95)`

Return the same predictions along factor `j` on `grid`, holding the other factors at `f0`. The caller supplies `rows(F)` mapping factor settings to model rows. See the [guide](industrial.md#prediction-at-new-points-and-effect-traces) and [R comparison](https://github.com/JohnnyZPiggy/TorchOED/tree/main/reports/w18-predict). No plotting or formula parsing.

## REML analysis

`fit_reml(X, y, groups)`

Fit a CPU float64 random-intercept REML model (§136–§138): full-rank `(n, p)` X with `n > p`, response `(n,)`, and a tuple of 1–3 integer grouping label tensors `(n,)`. Labels may be noncontiguous. Returns `coef`, `se`, Satterthwaite `df`, `t`, `variance` (groupings in input order, then residual), `loglik` (lme4 REML convention) and `iterations`. Dense autograd Newton with Fisher-scoring fallback; no timing claim. Duplicate partitions, one row per level, and groupings spanned by X raise `ValueError`; a group variance approaching zero raises `ValueError` naming the grouping. Nonconvergence raises `RuntimeError` with the gradient norm. No random slopes, boundary estimates, Kenward–Roger df, ML, p-values or GPU support. See the [REML guide](industrial.md#analysing-blocked-and-split-plot-results-reml).

### `orthogonal_array(levels, *, device=None, max_elements=1_000_000)`

Strength-2 integer orthogonal arrays with factor levels in input order: one prime power, optionally mixed with its prime, using GF(q) replacement or two-level Plackett–Burman. Returns int64 labels starting at zero; run size is minimal within the family, not globally. Mixed primes and strength ≥ 3 are unsupported. See [gate §128–§130](https://github.com/JohnnyZPiggy/TorchOED/blob/main/docs/P2_GATE.md).

### `covering_array(k, v, *, strength=2, device=None, max_elements=1_000_000)`

Returns deterministic int64 covering arrays. The run size is optimal for binary strength 2 (Kleitman–Spencer) and for prime-power v with strength 2/3 and strength ≤ k ≤ v + 1 (index 1). Otherwise, for strength 2/3, 2 ≤ v ≤ 6 and k ≤ 100, it returns the general family H (index-1 bases, density greedy, products, doubling, row deletion), whose run size is not optimal in general: for strength 3, v = 6, k ≥ 50 it is 1.26–1.38× the NIST IPOG-F sizes. Other inputs raise ValueError. The element budget is checked before construction for the optimal families and on the result for family H. See [gate §131–§135](https://github.com/JohnnyZPiggy/TorchOED/blob/main/docs/P2_GATE.md).


## Desirability and multi-response optimisation

`d_max(y, low, high, scale=1.0)`, `d_min(y, low, high, scale=1.0)`, `d_target(y, low, target, high, low_scale=1.0, high_scale=1.0)`

Return Derringer–Suich desirabilities of the same shape as float64 `y`. Thresholds must be ordered and shape exponents positive. `d_overall(*d)` returns the geometric mean of equal-shape individual values; a zero individual makes the overall value zero. The target uses the lower branch at the join (§140–§141).

`maximize_desirability(overall, lower, upper, *, starts=None)`

CPU float64 box maximisation: `overall` maps `(m, k)` factor settings to `(m,)` desirabilities using torch operations. Bounds are length-k tensors, strictly ordered; optional starts have shape `(s, k)` and lie inside the box. The default is the 5-per-factor midpoint grid in cartesian-product order. Returns `x` `(k,)`, scalar `value`, every start's final `values` `(s,)`, and accepted-step counts `iterations` `(s,)`; ties select the first start. Sine-transformed BFGS uses weak Wolfe bisection, at most 500 steps and 60 line-search trials; zero desirability, zero/nonfinite gradient or line-search failure stops a start. Nonsmooth convergence is observed; no global guarantee, timing, GPU or plotting claim. See the [guide](industrial.md#desirability-and-multiple-responses) and [published comparison](https://github.com/JohnnyZPiggy/TorchOED/tree/main/reports/w18-desirability).
