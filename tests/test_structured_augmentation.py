import math

import pytest
import torch

from torchoed import full_factorial, optimal_design


def split_candidates():
    h, x = torch.cartesian_prod(torch.tensor([-1, 1]), torch.arange(-1, 2)).repeat(2, 1).T
    F = torch.stack((torch.ones_like(h), h, x, x.square(), h*x), dim=1).double()
    return F, h.long()


@pytest.mark.parametrize("sizes,eta,fixed,optima", [
    ([4, 4], 0, [0, 8], (204, 845/408, 109/153)),
    ([4, 4], 0, [0, 2, 6, 8], (48, 53/12, 11/9)),
    ([3, 3], 1, [0, 6, 1], (8, 21/4, 7/2)),
    ([2, 2, 2], 0.5, [3], (96, 71/24, 103/72)),
])
def test_structured_augmentation_matches_exhaustive_optima(sizes, eta, fixed, optima):
    if eta:
        F, hard = split_candidates()
        structure = dict(whole_plots=sizes, hard=hard, eta=eta)
    else:
        x, y = (full_factorial([3, 3]).double() - 1).T
        F = torch.stack((x, y, x.square(), y.square(), x*y), dim=1)
        structure = dict(blocks=sizes)

    # P2_GATE §25.6 gives the independently enumerated exact optima.
    n, p = sum(sizes), F.shape[1]
    for criterion, optimum in zip("DAI", optima):
        result = optimal_design(F, n, **structure, fixed_ids=fixed, criterion=criterion,
                                restarts=20, generator=torch.Generator().manual_seed(0))
        value = result["objective"]
        expected = math.log(optimum) if criterion == "D" else -optimum
        assert value == pytest.approx(expected, rel=0, abs=1e-10)
        efficiency = math.exp((value - expected)/p) if criterion == "D" else expected/value
        assert 0 < result["efficiency_lower_bound"] <= efficiency + 1e-12
        assert result["ids"].unique().numel() == n
        start = 0
        for group in result["ids"].split(sizes):
            assert set(fixed[start:start + len(group)]) <= set(group.tolist())
            if eta:
                assert hard[group].unique().numel() == 1
            start += len(group)


def test_structured_augmentation_rejects_incompatible_layouts():
    F, hard = split_candidates()
    with pytest.raises(ValueError, match="must share one hard label"):
        optimal_design(F, 6, whole_plots=[3, 3], hard=hard, fixed_ids=[0, 3])
    with pytest.raises(ValueError):
        optimal_design(F, 6, blocks=[3, 3], whole_plots=[3, 3], hard=hard, fixed_ids=[0])
