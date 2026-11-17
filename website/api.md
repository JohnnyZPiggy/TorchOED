# API reference

All 30 names are exported from `torchoed`; tensor shapes, numerical guards and resource limits are detailed in the linked [source contracts](https://github.com/JohnnyZPiggy/TorchOED/tree/6c9d63d3b48b5afe6a3189364e19614d9897854d/docs). Search accepts unbatched CPU float64 features; constructors expose their explicit device/dtype choices. Signatures below preserve the source parameters, defaults and annotations.

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

Return the coded two-factor pair rows and then center rows for 3, 4 or 5 factors.

`simplex_lattice(q: int, m: int, *, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None, max_elements: int = 1_000_000) -> torch.Tensor`

`lhs(n_samples: int, n_factors: int, *, generator: torch.Generator | None = None, device: torch.device | str | None = None, dtype: torch.dtype = torch.float64, mode: str = "random", max_elements: int = 1_000_000) -> torch.Tensor`

Draw a random Latin hypercube with one sample in each marginal stratum.

`sobol(n: int, d: int, *, scramble: bool = False, seed: int | None = None, skip: int = 0, balance: str = "exact", dtype: torch.dtype = torch.float64, device: torch.device | str | None = None, max_elements: int = 1_000_000) -> torch.Tensor`

Draw exactly n points using native CPU Sobol generation followed by the requested device transfer.

## Features and constraints

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

`design_diagnostics(F: torch.Tensor, ids: torch.Tensor, *, region: torch.Tensor | None = None, effect: float | torch.Tensor | None = None, alpha: float = 0.05, blocks: list[int] | tuple[int, ...] | None = None, whole_plots: list[int] | tuple[int, ...] | None = None, eta: float = 1.0, alias: torch.Tensor | None = None) -> dict[str, torch.Tensor | list[int]]` returns design-column correlations and VIFs, scaled prediction variances, and their sorted values for the fraction-of-design-space curve.

Pass omitted-term columns as `alias=G` to get `(F₁ᵀF₁)⁻¹F₁ᵀF₂` under the result key `"alias"`, with `F₁ = F[ids]` and `F₂ = G[ids]`. `G` must be two-dimensional and match the row count, dtype and device of `F`. The alias calculation requires an unstructured design with a full-rank fitted model.

## Criterion scores

`d_optimality(M: torch.Tensor, *, max_elements: int = 1_000_000) -> torch.Tensor`

Return logdet M for numerically positive-definite information matrices.

`a_optimality(M: torch.Tensor, *, max_elements: int = 1_000_000) -> torch.Tensor`

Return −trace(M⁻¹) through Cholesky solves.

`e_optimality(M: torch.Tensor, *, max_elements: int = 1_000_000) -> torch.Tensor`

Return the minimum eigenvalue for numerically positive-semidefinite information matrices.

`c_optimality(M: torch.Tensor, c: torch.Tensor, *, max_elements: int = 1_000_000) -> torch.Tensor`

Return −cᵀM⁻¹c for a matched contrast vector and positive-definite information matrix.

## Exchange and restarted design

`fedorov_exchange(F: torch.Tensor, ids: torch.Tensor, *, max_iter: int = 100, tile_size: int = 64, allow_repeats: bool = False, max_elements: int = 1_000_000) -> dict[str, object]`

Run directly verified one-row D exchanges from supplied IDs.

`score_two_swaps(F: torch.Tensor, ids: torch.Tensor, *, tile_size: int = 64, max_elements: int = 1_000_000) -> dict[str, object]`

Score legal simultaneous two-row D exchanges without updating the design.

`k2_exchange(F: torch.Tensor, ids: torch.Tensor, *, max_iter: int = 100, tile_size: int = 64, max_elements: int = 1_000_000, _free_positions: torch.Tensor | None = None) -> dict[str, object]`

Run deterministic one-/two-swap D search from supplied IDs; `_free_positions` is the private augmentation hook.

`d_optimal_design(F: torch.Tensor, n: int, *, fixed_ids: list[int] | tuple[int, ...] | torch.Tensor | None = None, restarts: int = 10, generator: torch.Generator | None = None, max_iter: int = 100, tile_size: int = 64, max_elements: int = 1_000_000) -> dict[str, object]`

Return the best seeded feasible D restart, its IDs, objective and conservative D-efficiency bound.

`optimal_design(F: torch.Tensor, n: int, *, criterion: str = "D", fixed_ids: list[int] | tuple[int, ...] | torch.Tensor | None = None, restarts: int = 10, generator: torch.Generator | None = None, max_iter: int = 100, max_elements: int = 1_000_000) -> dict[str, object]`

Return the best D/A/I restart with the selected criterion's objective, efficiency lower bound and local numerical certificate.
