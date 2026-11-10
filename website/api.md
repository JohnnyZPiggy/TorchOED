# API reference

All 19 names are exported from `torchoed`; tensor shapes, numerical guards and resource limits are detailed in the linked [source contracts](https://github.com/JohnnyZPiggy/TorchOED/tree/6c9d63d3b48b5afe6a3189364e19614d9897854d/docs). Search accepts unbatched CPU float64 features; constructors expose their explicit device/dtype choices. Signatures below preserve the source parameters, defaults and annotations.

## Candidate construction

`full_factorial(levels: list[int] | tuple[int, ...], *, device: torch.device | str | None = None, max_elements: int = 1_000_000) -> torch.Tensor`

Return integer level-index rows, with the last factor varying fastest.

`ff2n(k: int, *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float64, max_elements: int = 1_000_000) -> torch.Tensor`

Return the signed corners of a k-factor two-level factorial.

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
from torchoed import jacobian_features, optimal_design

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
```

## Information

`exact_information(F: torch.Tensor, ids: torch.Tensor, *, normalized: bool = False, allow_repeats: bool = False, max_elements: int = 1_000_000) -> torch.Tensor`

Form the chosen rows' Gram matrix, optionally normalized by the number of observation events.

`weighted_information(F: torch.Tensor, w: torch.Tensor, *, max_elements: int = 1_000_000) -> torch.Tensor`

Form information from nonnegative candidate weights summing to one.

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
