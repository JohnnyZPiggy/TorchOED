import torch
from torch import nn

from torchoed import jacobian_features, optimal_design, polynomial_features


def test_linear_and_last_layer_oracles():
    class Polynomial(nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = nn.Parameter(torch.arange(6, dtype=torch.float64))

        def forward(self, x):
            return polynomial_features(x, 2) @ self.theta

    x = torch.linspace(-1, 1, 20, dtype=torch.float64).reshape(10, 2)
    model = Polynomial().eval()
    F = jacobian_features(model, x)
    torch.testing.assert_close(F, polynomial_features(x, 2), atol=1e-12, rtol=0)
    assert not F.requires_grad and F.dtype == torch.float64
    mlp = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)).double().eval()
    before = {name: p.clone() for name, p in mlp.state_dict().items()}
    hidden = mlp[:2](x)
    F = jacobian_features(mlp, x, params=["2.weight", "2.bias"])
    torch.testing.assert_close(F, torch.cat((hidden, torch.ones(10, 1)), dim=1), atol=1e-12, rtol=0)
    for name, p in mlp.state_dict().items():
        assert torch.equal(p, before[name])


def test_exponential_decay_oracle():
    class Decay(nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = nn.Parameter(torch.tensor([2., 2.], dtype=torch.float64))

        def forward(self, x):
            a, b = self.theta
            return a * torch.exp(-b * x)

    x = torch.linspace(0, 2, 201, dtype=torch.float64)
    F = jacobian_features(Decay().eval(), x)
    result = optimal_design(F, 2, criterion="D", generator=torch.Generator().manual_seed(0))
    assert result["ids"].tolist() == [0, 50]


def test_michaelis_menten_oracle():
    class MichaelisMenten(nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = nn.Parameter(torch.ones(2, dtype=torch.float64))

        def forward(self, x):
            V, K = self.theta
            return V * x / (K + x)

    x = torch.linspace(0.01, 2, 200, dtype=torch.float64)
    F = jacobian_features(MichaelisMenten().eval(), x)
    result = optimal_design(F, 2, criterion="D", generator=torch.Generator().manual_seed(0))
    assert result["ids"].tolist() == [49, 199]
