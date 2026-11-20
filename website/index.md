# TorchOED

![TorchOED](assets/torchoed-banner.png)

**Optimal experimental design in PyTorch — with a certificate on every design.**

TorchOED chooses which experiments to run: D-, A- and I-optimal designs, blocking, split-plot, mixtures, screening and response-surface designs, space-filling designs, power and diagnostics, in one tensor API that also accepts any PyTorch model. Every optimal design carries a proved lower bound on its efficiency.

!!! info "Status"
    Version 0.8.0 is a pre-release on [TestPyPI](https://test.pypi.org/project/torchoed/); the API may still change before 1.0. Python 3.11 or newer and torch 2.2 or newer are required.

## Install

```sh
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple torchoed==0.8.0
```

or from source:

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

Read [criteria](criteria.md), [augmentation and constraints](augmentation.md), [space-filling designs](space_filling.md), the [PyDOE comparison](pydoe.md), and the [API reference](api.md).
