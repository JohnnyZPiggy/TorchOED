# TorchOED

Torch-only M1 constructors/features, M2-A information/criteria and scoped M3-A Fedorov search (Issue #9). CPU acceptance only; experimental API, no release claim.

```powershell
python -m pip install --no-build-isolation -e .
python -m pytest -q
```

```python
from torchoed import full_factorial, ff2n, lhs, sobol, polynomial_features
C = ff2n(2)
F = polynomial_features(C, degree=2)  # shape (4,6), rank deficient by design
```

See [API contract](docs/M1_API.md), [91-export coverage](docs/ALGORITHM_COVERAGE.md), [M1 evidence](reports/m1/ENGINEER_HANDOFF.md), and [SAE entrypoint](.sae/README.md).

Historical M0 evidence remains under reports/m0 at its original tested commit; its hashes describe that historical snapshot, not current M1 sources. Runtime dependency is torch only; PyDOE is not installed or called by the M1 package/tests.

M2-A: see [API and numerical boundaries](docs/M2_API.md) and [acceptance evidence](reports/m2a/ENGINEER_HANDOFF.md).

```python
import torch
from torchoed import exact_information, d_optimality
F = torch.tensor([[1.,-1.],[1.,0.],[1.,1.]], dtype=torch.float64)
M = exact_information(F, torch.tensor([0,2]))  # diag(2,2), raw event count
score = d_optimality(M)  # log(4), maximize; singular inputs raise
```

M3-A adds `fedorov_exchange(F, ids)` for provided-initial-ID CPUfloat64 best-improvement search. [API/error/certificate policy](docs/M3_API.md), [CPU evidence](reports/m3a/ENGINEER_HANDOFF.md). Numerical 1-swap stopping is not global optimality or a rigorous certificate. No initializer, repeats, GPU or M3-B.

M3-B1 adds `score_two_swaps(F, ids)` for all legal simultaneous k=2 exchange scores and validity diagnostics. It does not select/apply a swap or search. [API/schema](docs/M3B1_API.md), [CPU exact-oracle evidence](reports/m3b1/ENGINEER_HANDOFF.md).
