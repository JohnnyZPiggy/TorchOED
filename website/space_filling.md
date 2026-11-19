# Space filling

Space-filling designs spread runs across a unit box without a response model.

## Random LHS and Sobol

Random Latin hypercubes place one run in each stratum of each factor. Sobol produces a low-discrepancy sequence; powers of two preserve its usual balance.

```python
import torch
from torchoed import lhs, sobol

random = lhs(8, 3, generator=torch.Generator().manual_seed(0))
sequence = sobol(8, 3)
print(random.shape, sequence.shape)
```

## Maximin and MaxPro LHS

Optimised modes use midpoint levels `(r + 0.5) / n`, with every column a permutation. They operate on the box only, without fixed runs or candidate constraints.

The objectives from [gate §83–85](https://github.com/JohnnyZPiggy/TorchOED/blob/main/docs/P2_GATE.md#83-space-filling-latin-hypercubes-maximin-and-maxpro-by-batched-column-swaps-w13-f1) are minimised:

- **maximin:** Morris–Mitchell `φ_p(X) = (Σ_{i<j} D_ij^{−p/2})^{1/p}` with `p = 15`.
- **maxpro:** Joseph–Gul–Ba `ψ(X) = (C(n,2)⁻¹ Σ_{i<j} exp(−L_ij))^{1/k}`.

Here `D_ij = Σ_l (x_il − x_jl)²` and `L_ij = Σ_l log (x_il − x_jl)²`. Maximin penalises short Euclidean distances; MaxPro penalises close projected pairs.

```python
import torch
from torchoed import lhs

for mode in ("maximin", "maxpro"):
    design = lhs(8, 3, mode=mode, restarts=2, max_iter=20,
                 generator=torch.Generator().manual_seed(0))
    print(mode, design.shape)
```

Each restart starts from a random midpoint LHS. The search evaluates within-column swaps in batches and takes the best strictly improving swap. `restarts` chooses the best result across starts; `max_iter=None` continues until no swap improves. The generator controls the starts. This is local descent, with no general global-optimum guarantee.
