# API reference

All 31 names are exported from `torchoed`; tensor shapes, numerical guards and resource limits are detailed in the linked [source contracts](https://github.com/JohnnyZPiggy/TorchOED/tree/6c9d63d3b48b5afe6a3189364e19614d9897854d/docs). Search accepts unbatched CPU float64 features; constructors expose their explicit device/dtype choices. Signatures below preserve the source parameters, defaults and annotations.

## Candidate construction

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

`mixture_vertices(lower, upper, *, face_dims=(), dtype=torch.float64, device=None) -> Tensor`

Box-constrained simplex vertices in exact lexicographic order, then vertex-mean centroids for each requested face dimension (1 through q−1). Bounds are int, float, Fraction or str; floats use their shortest decimal. Invalid or infeasible bounds raise `ValueError`. See [mixture guide](industrial.md#mixture-designs).

`simplex_lattice(q: int, m: int, *, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None, max_elements: int = 1_000_000) -> torch.Tensor`

`lhs(n_samples: int, n_factors: int, *, generator: torch.Generator | None = None, device: torch.device | str | None = None, dtype: torch.dtype = torch.float64, mode: str = "random", max_elements: int = 1_000_000) -> torch.Tensor`

Draw a random Latin hypercube with one sample in each marginal stratum.

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

`design_diagnostics(F: torch.Tensor, ids: torch.Tensor, *, region: torch.Tensor | None = None, effect: float | torch.Tensor | None = None, alpha: float = 0.05, blocks: list[int] | tuple[int, ...] | None = None, whole_plots: list[int] | tuple[int, ...] | None = None, eta: float = 1.0, covariates: torch.Tensor | None = None, alias: torch.Tensor | None = None, groups: list[list[int]] | None = None, centered: bool = True) -> dict[str, torch.Tensor | list[int]]` returns design-column correlations and VIFs, scaled prediction variances, and their sorted values for the fraction-of-design-space curve.

Pass `blocks` and `whole_plots` together when every block boundary is a plot boundary for `M_C` correlation/VIF, centered SPV/FDS and t/F power with approximate containment df `w - g - h` (whole plot) and `n - w - (p - h)` (subplot), requiring positive residual df for power (§74–§75).
With `centered=False` (bool, unstructured only), all columns use uncentred Gram correlations and VIFs; SPV/FDS, power and alias outputs are unchanged. The default `True` retains centred diagnostics, undefined on Scheffé designs.

With `effect`, `groups=[[j, ...], ...]` adds effect-group F-test `group_power` in the given order (§56–§57). Groups must be a non-empty list of non-empty lists of distinct Python coefficient indices; overlaps are allowed. Structured designs also return int64 `group_df`; whole-plot groups must lie in one stratum. The existing `power` and `df` are unchanged. Whole-plot F power is exact in §19.5 and otherwise uses approximate containment degrees of freedom. Groups require `effect`.

Pass omitted-term columns as `alias=G` to get `(F₁ᵀF₁)⁻¹F₁ᵀF₂` under the result key `"alias"`, with `F₁ = F[ids]` and `F₂ = G[ids]`. `G` must be two-dimensional and match the row count, dtype and device of `F`. The alias calculation requires an unstructured design with a full-rank fitted model.

Pass fixed per-run `covariates=Z` (finite `(len(ids), c)` tensor, `c >= 1`, matching F dtype/device; `[1 Z]` full rank) for covariate-adjusted correlation/VIF, centered SPV/FDS and t/F power (§68–§69). The information is `M_Z = F[ids].T @ Q @ F[ids]`, eliminating `[1 Z]`; residual df is `n - 1 - c - p` and must be positive for power. Cannot combine with blocks, whole plots or alias. F has no intercept; singular information raises in Cholesky.

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

`fedorov_exchange(F: torch.Tensor, ids: torch.Tensor, *, max_iter: int = 100, tile_size: int = 64, allow_repeats: bool = False, max_elements: int = 1_000_000) -> dict[str, object]`

Run directly verified one-row D exchanges from supplied IDs.

`score_two_swaps(F: torch.Tensor, ids: torch.Tensor, *, tile_size: int = 64, max_elements: int = 1_000_000) -> dict[str, object]`

Score legal simultaneous two-row D exchanges without updating the design.

`k2_exchange(F: torch.Tensor, ids: torch.Tensor, *, max_iter: int = 100, tile_size: int = 64, max_elements: int = 1_000_000, _free_positions: torch.Tensor | None = None) -> dict[str, object]`

Run deterministic one-/two-swap D search from supplied IDs; `_free_positions` is the private augmentation hook.

`d_optimal_design(F: torch.Tensor, n: int, *, fixed_ids: list[int] | tuple[int, ...] | torch.Tensor | None = None, restarts: int = 10, generator: torch.Generator | None = None, max_iter: int = 100, tile_size: int = 64, max_elements: int = 1_000_000) -> dict[str, object]`

Return the best seeded feasible D restart, its IDs, objective and conservative D-efficiency bound.

`optimal_design(F: torch.Tensor, n: int, *, criterion: str | dict[str, float] = "D", fixed_ids: list[int] | tuple[int, ...] | torch.Tensor | None = None, restarts: int = 10, generator: torch.Generator | None = None, max_iter: int = 100, max_elements: int = 1_000_000, blocks: list[int] | tuple[int, ...] | None = None, whole_plots: list[int] | tuple[int, ...] | None = None, hard: torch.Tensor | None = None, eta: int | float = 1.0, covariates: torch.Tensor | None = None, alias: torch.Tensor | None = None, alias_weight: float = 1.0) -> dict[str, object]`

With `alias=G`, maximise `(1/p) log det(M/n) − alias_weight * ‖A‖²`, where `A = M⁻¹ F[ids]ᵀG[ids]` (§71–§73). `G` must be finite, 2-D, with the same candidate row count, dtype and device as `F`; `alias_weight` is finite and nonnegative (default 1). CPU float64 and `m ≥ n ≥ p` are required. Only string `criterion="D"` is supported; compound criteria, blocks, whole plots, covariates and hard labels raise `ValueError`. The existing compound exchange driver retains `fixed_ids` and returns sorted distinct IDs, `criterion="D"`, the penalised `objective`, `alias_trace=‖A‖²`, and `LOCAL_ALIAS_EXCHANGE` at a local optimum (else `NOT_CERTIFIED`). The efficiency floor bounds D-efficiency relative to the approximate D-optimum; it does not bound the non-concave penalised criterion. `alias=None` preserves existing results. Choose the weight explicitly; the constrained Jones–Nachtsheim form is unsupported.

A dict `criterion={"D": w_D, "A": w_A, "I": w_I}` maximises the weighted log-efficiency criterion (§65–§67). Keys are a nonempty subset of D/A/I; finite nonnegative weights are normalised to sum to one. CPU float64, unstructured designs only: blocks, whole plots and covariates are unsupported. The objective is `(w_D/p) log det(M/n) − w_A log tr(n M⁻¹) − w_I log tr(n B M⁻¹)`, with `B=FᵀF/m`. IDs are sorted and `fixed_ids` retained. Results include the normalised weight dict, `LOCAL_COMPOUND_EXCHANGE` at a local optimum (otherwise `NOT_CERTIFIED`), and a dual efficiency floor relative to the compound approximate optimum. Zero-weight terms are omitted; reference optima are not computed.


`covariates=Z` supports CPU float64 D/A/I for finite `(n, c)` fixed per-run columns with full-rank `[1 Z]`, `m ≥ n ≥ 1+c+p`, and no intercept in F (§62–§63). It cannot combine with `blocks` or `whole_plots`. IDs remain in run order, `fixed_ids` occupy leading runs, and the input `covariates` is returned unchanged. Objectives use the projected factor information, with centered candidate moments for I; `LOCAL_COVARIATE_EXCHANGE` certifies exchange/interchange local optimality and the dual efficiency floor compares with the unstructured centered approximate optimum. Covariate diagnostics remain open.


Return the best D/A/I restart with the selected criterion's objective, efficiency lower bound and local numerical certificate. CPU float64 supports fixed blocks and whole plots together: each block is a union of consecutive plots, `eta > 0`, and features omit the intercept. `fixed_ids` fill leading positions; fixed plots retain their hard label. Results include both layouts and the centered-candidate dual floor. The local certificate covers within-plot exchanges and greedy relabels; combined diagnostics are not yet supported.
