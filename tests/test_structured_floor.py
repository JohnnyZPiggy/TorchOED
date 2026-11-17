from itertools import product

import pytest
import torch

from torchoed import ff2n, optimal_design


@pytest.mark.parametrize("criterion", "DAI")
def test_factorial_structured_floor(criterion):
    a, b, c = ff2n(3).unbind(1)
    F = torch.stack((a, b, c, a*b, a*c, b*c), dim=1)
    result = optimal_design(F, 8, blocks=[4, 4], criterion=criterion, generator=torch.Generator().manual_seed(0))
    assert result["efficiency_lower_bound"] == pytest.approx(1, rel=0, abs=1e-12)


@pytest.mark.parametrize("sizes,eta,optima", [
    ([3, 4], 0, (108, 70/27, 208/243)), ([4, 4], 0, (240, 121/60, 31/45)),
    ([3, 3], 1, (48, 35/12, 11/6)), ([2, 2, 2], 0.5, (96, 71/24, 103/72)),
])
def test_structured_floor_against_exhaustive_optima(sizes, eta, optima):
    levels = torch.arange(-1, 2)
    x, y = torch.cartesian_prod(torch.tensor([-1, 1]) if eta else levels, levels).T
    F = torch.stack((torch.ones_like(x), x, y, y*y, x*y) if eta else (x, y, x*x, y*y, x*y), dim=1).double()
    F = F.repeat(2, 1) if eta else F
    structure = dict(whole_plots=sizes, hard=x.repeat(2), eta=eta) if eta else dict(blocks=sizes)
    G = F if eta else F - F.mean(0)
    W = G.mT @ G / len(G)
    Y = torch.linalg.solve(W, G.mT)
    n, p = sum(sizes), F.shape[1]
    for (criterion, optimum), restarts in product(zip("DAI", optima), (1, 20)):
        result = optimal_design(F, n, **structure, criterion=criterion, restarts=restarts,
                                max_iter=0 if restarts == 1 else 100, generator=torch.Generator().manual_seed(0))
        value, floor = result["objective"], result["efficiency_lower_bound"]
        if criterion == "D":
            uniform = ((value - W.logdet()) / p).exp() * p / (n * (G * Y.mT).sum(1).max())
            efficiency = (torch.exp(torch.tensor(value, dtype=F.dtype)).item() / optimum)**(1/p)
        else:
            B = torch.eye(p, dtype=F.dtype) if criterion == "A" else W
            phi = torch.trace(torch.linalg.solve(W, B))
            uniform = phi.square() / ((Y * (B @ Y)).sum(0).max() * n * -value)
            efficiency = optimum / -value
        assert isinstance(floor, float) and 0 < floor <= efficiency + 1e-12
        assert floor >= uniform.item() - 1e-12
