# TorchOED

Beta 0.5.0 provides PyTorch-native exact experimental design on CPU float64 model matrices. Python 3.11 or newer and torch 2.2 or newer are required; the API is experimental.

## Install from source

```sh
git clone https://github.com/JohnnyZPiggy/TorchOED.git
cd TorchOED
python -m pip install -e .
```

## Choose a D-optimal design

```python
import torch
from torchoed import full_factorial, polynomial_features, optimal_design
C = full_factorial((3, 3)).double() - 1
F = polynomial_features(C, 2)
result = optimal_design(F, 7, criterion="D", generator=torch.Generator().manual_seed(0))
print(result["ids"].tolist(), result["objective"], result["efficiency_lower_bound"])
```

`ids` indexes rows of F; `objective` is the maximized log determinant for D. The efficiency field is a certified lower bound, not achieved efficiency or a global-optimality certificate. Search uses seeded feasible restarts; recreate the generator to repeat a call.

Read [criteria](criteria.md), [augmentation and constraints](augmentation.md), the [PyDOE comparison](pydoe.md), and the [API reference](api.md).
