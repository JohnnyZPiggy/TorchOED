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

## Covariates

Fixed per-run covariates absorb nuisance effects while the design chooses the factors. Supply features without an intercept; the returned IDs keep run order and `fixed_ids` stay leading. Covariates cannot combine with blocks or whole plots. `design_diagnostics` accepts the same `covariates=` argument.

```python
import torch
from torchoed import optimal_design
F = torch.cartesian_prod(torch.tensor([-1., 1.], dtype=torch.float64), torch.tensor([-1., 0., 1.], dtype=torch.float64))
Z = torch.arange(5, dtype=torch.float64)[:, None]  # fixed linear time trend
result = optimal_design(F, 5, covariates=Z, generator=torch.Generator().manual_seed(0))
assert torch.equal(result["covariates"], Z)
```

## Split-plot designs

Use `whole_plots` for the plot sizes, `hard` for one integer setting label per candidate, and `eta` for the positive ratio of whole-plot to residual variance. In this example, factor A must stay fixed within each plot. Split-plot search supports D, A and I, including `fixed_ids` and fixed blocks whose boundaries coincide with plot boundaries. Omit the intercept when blocks are present.

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

For blocks with whole plots, use block run counts and a model without an intercept. This is the frozen §53.8 (C-a) example; the floor is relative to the centered unstructured approximate optimum.

```python
import math
import torch
from torchoed import optimal_design

h, x = torch.cartesian_prod(torch.tensor([-1, 1]), torch.arange(-1, 2)).repeat(2, 1).T
F = torch.stack((h, x, x.square(), h*x), 1).double()
result = optimal_design(F, 8, blocks=[4, 4], whole_plots=[2]*4, hard=h, eta=1,
                        generator=torch.Generator().manual_seed(0))
assert math.isclose(math.exp(result["objective"]), 640/9, rel_tol=1e-9)
assert result["certificate"] == "LOCAL_SPLIT_PLOT_EXCHANGE"
assert 0 < result["efficiency_lower_bound"] <= 1 + 1e-12
```

## Split-split-plot designs

Use `subplots` for subplot sizes nested within `whole_plots`, `subplot_hard` for one integer subplot-setting label per candidate, and `eta_sub` for the positive subplot-to-residual variance ratio. Subplot sizes sum to the run count and every whole-plot boundary must also be a subplot boundary; `hard` stays constant within each whole plot and `subplot_hard` within each subplot. D/A/I search requires CPU float64 and cannot combine with `blocks`, `fixed_ids`, `covariates`, `alias`, `if_possible` or `models` (P2 gate §102–§103). `LOCAL_SPLIT_SPLIT_PLOT_EXCHANGE` certifies local optimality under within-subplot exchanges and greedy subplot/whole-plot relabels; `efficiency_lower_bound` is a conservative floor against the OLS approximate optimum, not a global-optimum certificate.

```python
import torch
from torchoed import optimal_design

C = torch.cartesian_prod(torch.tensor([-1., 1.]), torch.tensor([-1., 1.]),
                         torch.tensor([-1., 0., 1.])).double()
F = torch.cat((torch.ones(12, 1), C, C[:, 2:]**2), 1)
result = optimal_design(F, 8, whole_plots=[4, 4], hard=C[:, 0].long(),
                        subplots=[2]*4, subplot_hard=C[:, 1].long(),
                        eta=1, eta_sub=1, criterion="D",
                        generator=torch.Generator().manual_seed(0))
ids = result["ids"]
assert ids.unique().numel() == 8
assert all(C[g, 0].unique().numel() == 1 for g in ids.split(4))
assert all(C[g, 1].unique().numel() == 1 for g in ids.split(2))
Zw = torch.eye(2, dtype=torch.float64).repeat_interleave(4, dim=0)
Zs = torch.eye(4, dtype=torch.float64).repeat_interleave(2, dim=0)
V = torch.eye(8, dtype=torch.float64) + Zw@Zw.T + Zs@Zs.T
X = F[ids]
torch.testing.assert_close(torch.linalg.det(X.T@torch.linalg.inv(V)@X),
                           F.new_tensor(1280/63), rtol=1e-9, atol=0)
```

## Mixture designs

Designs are without replacement; replicate a point (for example the centroid) by appending copies of its candidate row.

Use `scheffe_features` to build canonical linear, quadratic or special-cubic columns without an intercept or pure squares. Since proportions sum to one, an intercept or squares would make the quadratic model rank-deficient. Pass the features to the existing search and certificate machinery:

```python
import torch
from torchoed import mixture_vertices, scheffe_features, optimal_design

X = mixture_vertices([.2,.1,.1], [.6,.5,.6], face_dims=(1,2))
F = scheffe_features(X)
design = optimal_design(F, 10, criterion="D", generator=torch.Generator().manual_seed(0))
runs = X[design["ids"]]
assert runs.shape == (10,3)
assert design["efficiency_lower_bound"] <= 1 + 1e-12
```

D, A and I use the same engine and support `fixed_ids`. I uses the **candidate-average** moment `F.T @ F / len(F)`, not an integral over the mixture region.

Use `simplex_lattice(q, m)` for the `{q, m}` simplex lattice: every row is a mixture of `q` components whose proportions are multiples of `1/m`. Its rows are valid candidates for `scheffe_features` and `optimal_design` too.

Use `simplex_centroid` for one equal-proportion point per nonempty subset, or `mixture_vertices` for box-constrained vertices and face centroids. Centroids average the face's extreme vertices. Decimal float bounds are interpreted exactly as typed; pass `"1/3"` or `Fraction(1, 3)` for an exact third, because the float `1/3` repeated three times is below one in this decimal convention.

```python
from torchoed import simplex_centroid, mixture_vertices

C = simplex_centroid(3)
V = mixture_vertices([0.2, 0.1, 0.1], [0.6, 0.5, 0.6], face_dims=(1, 2))
assert C.shape == (7, 3)
assert V.shape == (13, 3)  # six vertices, six edge means, one overall mean
```

Cross mixture and process features with `mixture_process_features`; decode selected IDs by quotient and remainder:

```python
import torch
from torchoed import mixture_vertices, scheffe_features, ff2n, polynomial_features, mixture_process_features, optimal_design
X = mixture_vertices([.2,.1,.1], [.6,.5,.6], face_dims=(1,2))
Z = ff2n(2)
F = mixture_process_features(scheffe_features(X), polynomial_features(Z,1))
r = optimal_design(F, 24, generator=torch.Generator().manual_seed(0))
mixture_runs, process_runs = X[r["ids"] // len(Z)], Z[r["ids"] % len(Z)]
assert r["certificate"] == "LOCAL_2_NUMERICAL" and 0 < r["efficiency_lower_bound"] <= 1+1e-12
```

For no-intercept mixture models, request uncentred VIFs:

```python
from torchoed import design_diagnostics, simplex_centroid, scheffe_features
F = scheffe_features(simplex_centroid(3), model="quadratic")
print(design_diagnostics(F, torch.arange(len(F)), centered=False)["vif"])
```

## Plackett–Burman

Choose a multiple of four from 4 to 48. For fewer factors, take the leading columns. This example screens six factors in 12 runs.

```python
from torchoed import plackett_burman

X = plackett_burman(12)[:, :6]
assert X.shape == (12, 6)
```

In the 12-run design, each main effect is partially aliased with every two-factor interaction that does not contain it, with coefficient ±1/3 as reported by `design_diagnostics(..., alias=)`.

## Central composite and Box–Behnken

Use a central composite design to fit a quadratic response surface. For blocked analysis, pass the cube and axial block sizes in row order. Box–Behnken designs use 3 to 7 factors.

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

## Compound D/I compromise

Trade estimation and prediction using normalised nonnegative weights; the floor is relative to the compound optimum, not either single-criterion optimum.

```python
import torch
from torchoed import full_factorial, polynomial_features, optimal_design
F = polynomial_features(full_factorial([3, 3]).double() - 1)
result = optimal_design(F, 6, criterion={"D": 0.5, "I": 0.5},
                        generator=torch.Generator().manual_seed(0))
print(result["objective"], result["efficiency_lower_bound"])
assert result["efficiency_lower_bound"] <= 1 + 1e-12
```

## Trade D-efficiency against aliasing

Penalise bias from omitted interactions with an explicit weight. The reported floor bounds D-efficiency; the alias criterion has a local exchange certificate.

```python
import torch
from torchoed import full_factorial, polynomial_features, optimal_design
X = 2 * full_factorial([2, 2, 2]).double() - 1
F = polynomial_features(X, 1)
G = torch.stack([X[:, 0]*X[:, 1], X[:, 0]*X[:, 2], X[:, 1]*X[:, 2]], 1)
for weight in (0., 0.1):
    result = optimal_design(F, 6, alias=G, alias_weight=weight, restarts=20,
                            generator=torch.Generator().manual_seed(0))
    print(weight, result["alias_trace"], result["efficiency_lower_bound"])
```

## "If Possible" terms (Bayesian D and I)

Primary terms must be estimable; potential terms may be included even when the run count is below the total column count (m ≥ n ≥ the primary count). Potential columns are residualised on the primary columns and scaled to range 2; the result exposes these as `scaled_features`. `tau` is the prior standard deviation of the scaled potential coefficients.

```python
import torch
from torchoed import full_factorial, polynomial_features, optimal_design

x = full_factorial([3, 3]).double() - 1
base = polynomial_features(x)  # [1, x1, x2, x1², x2², x1x2]
F = base[:, [0, 1, 2, 5, 3, 4]]
for criterion in ("D", "I"):
    result = optimal_design(
        F, 4, criterion=criterion, if_possible=[3, 4, 5], tau=1.0,
        generator=torch.Generator().manual_seed(0),
    )
    print(criterion, result["ids"], result["objective"],
          result["efficiency_lower_bound"])
```

Bayesian A, structured designs, non-diagonal priors and automatic selection of τ are not supported (P2 gate §95).

## Model-robust D over a set of models

Represent each competing model by a list of feature-column indices. The criterion is Σ_k (w_k/p_k) log det(M_k/n), the log of a weighted geometric mean of per-model D-efficiencies; weights are normalised and equal by default, and n must be at least the number of columns in the union. `efficiency_lower_bound` is the robust floor; `model_efficiency_lower_bounds` gives the individual D floors.

```python
import torch
from torchoed import full_factorial, polynomial_features, optimal_design

F = polynomial_features(full_factorial([3, 3]).double() - 1)
# Columns: [1, x1, x2, x1², x2², x1x2]
result = optimal_design(
    F, 6, models=[[0, 1, 2], [0, 1, 2, 3, 4, 5]],
    generator=torch.Generator().manual_seed(0),
)
print(result["ids"], result["objective"], result["efficiency_lower_bound"],
      result["model_efficiency_lower_bounds"])
```

The objective is (1/6) log 140 + (1/12) log 256 − log 6 ≈ −0.506054278419874 (P2 gate §96.7, MR-b).
A/I, n below the union column count, maximin criteria, model priors and structured designs are unsupported (§98).

## Compare Designs and the sample-size explorer

Efficiencies are relative to the first design on total information, so different run counts compare directly and values above 1 are better; I uses the region's mean prediction variance and G its maximum. FDS quantiles use the unscaled variance v = spv/n. An integer entry requests a run size searched with `optimal_design`, in order with one shared generator.

```python
import torch
from torchoed import full_factorial, compare_designs

x = full_factorial([3, 3]).double() - 1
F = torch.cat((torch.ones(9, 1), x), 1)
result = compare_designs(F, [torch.tensor([0, 2, 6, 8]), torch.tensor([0, 2, 4, 6, 8])])
print(result["efficiency"])
print(result["fds_quantiles"])
```

For corners plus centre relative to corners: D = (5/4)^{1/3} ≈ 1.0772, A = 15/14, I = 35/32, G = 15/14. Quantile rows are [0.25, 0.5, 0.5, 0.75, 0.75] and [0.2, 0.45, 0.45, 0.7, 0.7].

```python
F2 = torch.cat((F, x.prod(1, keepdim=True)), 1)
result = compare_designs(F2, [5, 7, 9], effect=1.0, generator=torch.Generator().manual_seed(0))
print(result["n"])
print(result["power"])
```

Structured, Bayesian and model-robust comparisons and cross-design GPU batching are not supported (P2 gate §101).

## Published reference examples

A published example is a call printed in a reference package's help page or vignette at a recorded version, with its inputs frozen as repository data and re-scored by one shared TorchOED metric.
The [published-example index](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w12-published/summary.md) records 8 of 13 MATCH, 1 NO MATCH and 4 NR (packages unavailable in the run).
On the AlgDesign `Federov.Rd` example with the A criterion, TorchOED's trace is 2.4240924092409237 and AlgDesign's is 2.3 (lower is better).
