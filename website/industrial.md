# Industrial designs

These examples use CPU float64 features. Each search uses a fresh generator seeded at zero, and each example can run on its own.

## Blocking

Pass fixed block sizes in `blocks` when runs are grouped by a nuisance factor, such as day or batch. The sizes must sum to the number of runs. This is a D-optimal search without `fixed_ids`; omit the intercept because centering the features within each block removes it. The returned IDs are grouped by block.

```python
import math
import torch
from torchoed import full_factorial, optimal_design, polynomial_features

C = full_factorial((3, 3)).double() - 1
F = polynomial_features(C, 2, include_bias=False)
blocks = [4, 4]
result = optimal_design(F, 8, blocks=blocks, generator=torch.Generator().manual_seed(0))
ids = result["ids"]
assert ids.numel() == ids.unique().numel() == 8

rows = F[ids].split(blocks)
centered = torch.cat([block - block.mean(dim=0) for block in rows])
logdet = torch.linalg.slogdet(centered.T @ centered).logabsdet.item()
assert math.isclose(result["objective"], logdet, rel_tol=1e-9)
```

## Split-plot designs

Use `whole_plots` for the plot sizes, `hard` for one integer setting label per candidate, and `eta` for the positive ratio of whole-plot to residual variance. In this example, factor A must stay fixed within each plot. Split-plot search supports D only and cannot be combined with `blocks` or `fixed_ids`.

```python
import math
import torch
from torchoed import ff2n, optimal_design

X = ff2n(3)
X2 = torch.cat([X] * 2)  # Two candidate IDs per factor setting
A, B, C = X2.T
F = torch.stack([torch.ones_like(A), A, B, C, A * B, A * C, B * C], dim=1)
hard = (A > 0).long()
sizes = [2] * 4
eta = 1.0
result = optimal_design(F, 8, whole_plots=sizes, hard=hard, eta=eta,
                        generator=torch.Generator().manual_seed(0))

ids = result["ids"]
assert all(plot.unique().numel() == 1 for plot in hard[ids].split(sizes))
Z = torch.eye(4, dtype=F.dtype).repeat_interleave(2, dim=0)
V = torch.eye(8, dtype=F.dtype) + eta * (Z @ Z.T)
selected = F[ids]
logdet = torch.linalg.slogdet(selected.T @ torch.linalg.solve(V, selected)).logabsdet.item()
assert math.isclose(result["objective"], logdet, rel_tol=1e-9)
```

## Mixture designs

Call `simplex_lattice(q, m)` with positive integers for the component count and the number of subdivisions. For a Scheffé quadratic, keep the linear and pair-product columns and drop the pure squares. Omit the intercept because the proportions sum to one, so the intercept would be a linear combination of the component columns.

```python
import torch
from torchoed import optimal_design, polynomial_features, simplex_lattice

L = simplex_lattice(3, 3)
terms = polynomial_features(L, 2, include_bias=False)
F = terms[:, [0, 1, 2, 6, 7, 8]]  # x1, x2, x3, x1*x2, x1*x3, x2*x3
ids = optimal_design(F, 8, generator=torch.Generator().manual_seed(0))["ids"]

torch.testing.assert_close(L.sum(dim=1), L.new_ones(len(L)))
assert L[ids].unique(dim=0).shape[0] == 8
```

## Plackett–Burman

Choose a multiple of four from 4 to 48. For fewer factors, take the leading columns. This example screens six factors in 12 runs.

```python
from torchoed import plackett_burman

X = plackett_burman(12)[:, :6]
assert X.shape == (12, 6)
```

## Central composite and Box–Behnken

Use a central composite design to fit a quadratic response surface. For blocked analysis, pass the cube and axial block sizes in row order. Box–Behnken designs use 3 to 5 factors.

```python
from torchoed import box_behnken, central_composite

X = central_composite(2, alpha="orthogonal", center=(3, 3))
blocks = [2**2 + 3, 2 * 2 + 3]
Y = box_behnken(3)
assert X.shape == (sum(blocks), 2) and Y.shape == (15, 3)
```

## Definitive screening

Pass a supported factor count to `definitive_screening(k)`: 3–20 or 23–30; `k = 21` and `22` raise because no conference matrix of order 22 exists. The returned design uses three levels (−1, 0, +1) and needs no search. Here six factors take 13 runs.

```python
import torch
from torchoed import definitive_screening

D = definitive_screening(6)
assert D.shape == (13, 6)
gram = D.T @ D
assert torch.equal(gram, torch.diag(gram.diagonal()))

i, j = torch.triu_indices(D.shape[1], D.shape[1], offset=1)
quadratic = torch.cat([D.square(), D[:, i] * D[:, j]], dim=1)
assert torch.count_nonzero(D.T @ quadratic).item() == 0
```

## Fractional factorials and fold-over

Use zero-based base columns to specify generators. This eight-run design moves from resolution III to IV under full fold-over. Use `factor=0` to reverse only the first factor in the added rows. The interaction of the first two factors aliases with the fourth factor, which uses their product as its generator.

```python
import torch
from torchoed import design_diagnostics, fold_over, fractional_factorial, fractional_factorial_generators, word_length_pattern

X = fractional_factorial(3, fractional_factorial_generators(5, 8))
assert word_length_pattern(X).tolist() == [0, 0, 2, 1, 0]
assert word_length_pattern(fold_over(X)).tolist() == [0, 0, 0, 1, 0]
F = torch.cat((X.new_ones((len(X), 1)), X), dim=1)
G = X[:, [0]] * X[:, [1]]
A = design_diagnostics(F, torch.arange(len(X)), alias=G)["alias"]
torch.testing.assert_close(A[4, 0], X.new_tensor(1.0))
```

## Diagnostics and power

Pass your features and selected IDs to `design_diagnostics` for variance inflation factors (VIFs) and the sorted scaled prediction variances used for the fraction-of-design-space curve. Set `effect` to a scalar or a tensor of per-coefficient effects to get power for two-sided t-tests at `alpha` (0.001 to 0.5), assuming residual standard deviation σ = 1 and more runs than coefficients (n > p).

```python
import torch
from torchoed import design_diagnostics, full_factorial, optimal_design, polynomial_features

C = full_factorial((3, 3)).double() - 1
F = polynomial_features(C, 2)
ids = optimal_design(F, 9, generator=torch.Generator().manual_seed(0))["ids"]
alpha = 0.05
diagnostics = design_diagnostics(F, ids, effect=1.0, alpha=alpha)
power = diagnostics["power"]

print("Max VIF:", diagnostics["vif"].max().item())
print("FDS quantiles (10%, 50%, 90%):",
      torch.quantile(diagnostics["fds"], F.new_tensor([0.1, 0.5, 0.9])).tolist())
print("Power:", power.tolist())
assert ((alpha <= power) & (power <= 1)).all()
```
