# PyTorch model → experiments

Run `python examples/closed_loop_sim.py` for four seeded CPU rounds of
Michaelis–Menten experiments. Each round recomputes Jacobian features at the
current estimate, asks `next_batch` for two new candidate IDs while retaining
all previous IDs, measures them with a simulated noisy lab, and refits all
observations with LBFGS. It prints the new IDs and parameter estimates.
The `Backend` protocol has one method, `run(x) → y`; `SimulatedLab` supplies
Gaussian noise with σ = 0.05. Parameters `V = exp(u)` and `K = exp(v)` stay
positive, and fitting starts from the previous round's estimate.

Certificates refer to the **current linearization over the given candidates
only**, conditional on previously run IDs. They do not certify the nonlinear
model at its unknown true parameters. This example targets small-budget,
constrained, certified experiments.

For a scalar-output MLP, select only its last layer to design on its learned
features. Call `eval()` before constructing Jacobian rows:

```python
import torch
from torchoed import jacobian_features, next_batch

torch.manual_seed(0)
mlp = torch.nn.Sequential(
    torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1)
).double().eval()
X = torch.randn(30, 2, dtype=torch.float64)
F = jacobian_features(mlp, X, params=["2.weight", "2.bias"])
ids = next_batch(F, [], 4, generator=torch.Generator().manual_seed(0))
```

Here each row is the three penultimate activations followed by 1 (the bias
derivative). The hidden layers remain fixed; this is a linear design on those
features, independent of the current output-layer weights and bias.
