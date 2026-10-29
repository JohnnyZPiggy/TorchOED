# TorchOED

Torch-only M1 constructors and polynomial features (Issue #5). CPU acceptance only; no M2/M3 algorithms or release claim.

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
