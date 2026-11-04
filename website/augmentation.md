# Augmentation, constraints and categorical factors

Pass distinct candidate IDs as `fixed_ids=` to keep existing runs and optimize the remaining positions. Their count must be less than n. Returned IDs include them; locality is restricted to free outgoing positions. IDs always refer to the candidate matrix actually supplied, including after filtering.

`feasible_candidates` keeps rows satisfying C Aᵀ ≤ b and excludes each matching `disallowed` dictionary. `effects_coding` maps levels 0 through q−2 to unit vectors and the final level to all −1, in float64.

```python
import torch
from torchoed import full_factorial, effects_coding, feasible_candidates, optimal_design
levels = full_factorial((3, 3, 3))
C = torch.cat((levels[:, :2].double() - 1, effects_coding(levels[:, 2], 3)), dim=1)
mask = feasible_candidates(C, A=torch.tensor([[1., 1., 0., 0.]], dtype=torch.float64),
                           b=torch.tensor([1.], dtype=torch.float64), disallowed=({0: -1., 2: 1.},))
allowed = C[mask]
F = torch.cat((torch.ones(len(allowed), 1, dtype=torch.float64), allowed), dim=1)
result = optimal_design(F, 7, fixed_ids=[0], generator=torch.Generator().manual_seed(0))
assert 0 in result["ids"].tolist()
print(result["ids"].tolist(), result["efficiency_lower_bound"])
```

Here fixed ID 0 means the first **filtered** row. This example uses an additive model; build interactions or polynomial terms explicitly when your scientific model needs them. Fixed runs must be candidate rows, without replication. The same conservative efficiency bounds remain valid; see [criteria](criteria.md).
