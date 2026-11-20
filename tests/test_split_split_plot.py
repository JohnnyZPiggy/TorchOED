import torch
import pytest
from torchoed import optimal_design
from torchoed.d_optimal_design import _nested_information


def covariance(sizes, subplots, eta, eta_sub):
    zw = torch.repeat_interleave(torch.eye(len(sizes), dtype=torch.float64), torch.tensor(sizes), dim=0)
    zs = torch.repeat_interleave(torch.eye(len(subplots), dtype=torch.float64), torch.tensor(subplots), dim=0)
    return torch.eye(sum(sizes), dtype=torch.float64) + eta*zw@zw.T + eta_sub*zs@zs.T


@pytest.mark.parametrize('eta,eta_sub,targets', [(1., 1., [1280/63, 25/8, 2083/840]), (2., .5, [16., 89/24, 371/120])])
def test_exhaustive_optima_and_floors(eta, eta_sub, targets):
    C = torch.cartesian_prod(torch.tensor([-1., 1.]), torch.tensor([-1., 1.]), torch.tensor([-1., 0., 1.])).double()
    F = torch.cat((torch.ones(12, 1), C, C[:, 2:]**2), 1)
    V = covariance([4, 4], [2]*4, eta, eta_sub)
    for criterion, optimum in zip(['D', 'A', 'I'], targets):
        result = optimal_design(F, 8, whole_plots=[4, 4], hard=C[:, 0].long(), subplots=[2]*4,
                                subplot_hard=C[:, 1].long(), eta=eta, eta_sub=eta_sub,
                                criterion=criterion, generator=torch.Generator().manual_seed(0))
        ids = result['ids']; X = F[ids]; M = X.T@torch.linalg.inv(V)@X
        value = torch.linalg.det(M) if criterion == 'D' else torch.trace(torch.linalg.solve(M, torch.eye(5, dtype=F.dtype) if criterion == 'A' else F.T@F/12))
        assert value.item() == pytest.approx(optimum, rel=1e-12)
        efficiency = (value.item()/optimum)**(1/5) if criterion == 'D' else optimum/value.item()
        assert 0 < result['efficiency_lower_bound'] <= efficiency+1e-12
        assert ids.unique().numel() == 8
        assert all(C[g, 0].unique().numel() == 1 for g in ids.split(4))
        assert all(C[g, 1].unique().numel() == 1 for g in ids.split(2))
        assert result['certificate'] == 'LOCAL_SPLIT_SPLIT_PLOT_EXCHANGE'


def test_information_and_split_reductions():
    F = torch.randn(12, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    sizes, eta, eta_sub = [4, 4], 2., .5
    for subplots in ([1]*8, [4, 4], [1, 3, 2, 2]):
        groups = torch.arange(8).split(subplots)
        whole = torch.repeat_interleave(torch.arange(2), torch.tensor(sizes))
        owner = torch.tensor([whole[g[0]] for g in groups])
        a = 1/(1+eta_sub*torch.tensor(subplots, dtype=F.dtype)); c = eta_sub*a
        d = eta/(1+eta*torch.zeros(2, dtype=F.dtype).index_add(0, owner, torch.tensor(subplots)*a))
        M, _ = _nested_information(F, torch.arange(8), groups, owner, a, c, d)
        dense = F[:8].T@torch.linalg.inv(covariance(sizes, subplots, eta, eta_sub))@F[:8]
        torch.testing.assert_close(M, dense, rtol=1e-12, atol=1e-12)
        if subplots in ([1]*8, [4, 4]):
            effective = eta/(1+eta_sub) if len(subplots) == 8 else eta+eta_sub
            from torchoed.d_optimal_design import _split_information
            counts = F.new_tensor(sizes); weights = 1/(counts*(1+effective*counts))
            reduced, _ = _split_information(F, torch.arange(8), sizes, weights)
            torch.testing.assert_close(M, reduced/(1+eta_sub) if len(subplots) == 8 else reduced, rtol=1e-12, atol=1e-12)
