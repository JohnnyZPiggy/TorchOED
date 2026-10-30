# TorchOED

Experimental PyTorch-native design of experiments: M1 constructors/features,
M2-A information/criteria, M3-A initialized 1-swap search and M3-B1 simultaneous
two-swap scoring. Version 0.0.1; CPU acceptance only, not a stable release or
general PyDOE replacement. Python >=3.11; runtime dependency: torch>=2.0 only.

```powershell
python -m pip install -e .
python examples/cpu_quickstart.py
```

```python
from torchoed import full_factorial, ff2n, lhs, sobol, polynomial_features
C = ff2n(2)  # CPU float64 signed corners
F = polynomial_features(C, degree=1)  # shape (4,3): bias, x1, x2
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

M2-B adds `c_optimality(M, c)`; see [API and SPD-only boundaries](docs/M2B_API.md) and [acceptance evidence](reports/m2b/ENGINEER_HANDOFF.md).

M3-A adds `fedorov_exchange(F, ids)` for provided-initial-ID CPUfloat64 best-improvement 1-swap search. [API/error/certificate policy](docs/M3_API.md), [CPU evidence](reports/m3a/ENGINEER_HANDOFF.md). Numerical 1-swap stopping is not global optimality or a rigorous certificate. No initializer, repeats or GPU acceptance.

M3-B1 adds `score_two_swaps(F, ids)` for all legal simultaneous k=2 exchange scores and validity diagnostics. It does not select/apply a swap or search. [API/schema](docs/M3B1_API.md), [CPU exact-oracle evidence](reports/m3b1/ENGINEER_HANDOFF.md).

Both M3 APIs require finite CPU float64 model features, unique CPU int64 IDs
and an initially numerically SPD Gram matrix. Check two-swap statuses before
using gains: invalid/inconclusive rows carry undefined NaN gain sentinels.
Integer factorial indices must be explicitly converted/coded before features.
Degree-2 features on two-level corners can be rank deficient; D/A criteria
reject numerical singularity rather than adding a ridge. E accepts numerical
PSD matrices. See contracts for batch, normalization and dtype policies.
`max_elements` is not a bound on native backend peak memory.

The [tiny runnable CPU example](examples/cpu_quickstart.py) uses existing APIs
only. Installation is from this source checkout, not a published release;
build requirements include setuptools >=77. If build requirements are already
installed, `--no-build-isolation` can be added. The recorded full regression
result is 65 passing Windows CPU tests at the source SHA in M3-B1 evidence;
historical manifests describe their own snapshots, not later documentation.

No verified CUDA/TPU support, acceleration claim, full PyDOE parity, two-swap
search, Modified Fedorov or DETMAX.
M5-A improves usability only. See [release readiness](docs/RELEASE_READINESS.md)
and the [MIT license](LICENSE).
