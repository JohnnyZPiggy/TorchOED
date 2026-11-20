import torch
import pytest
from torchoed import optimal_design
from torchoed.d_optimal_design import _strip_information


def covariance(r, c, er, ec):
    zr = torch.eye(r, dtype=torch.float64).repeat_interleave(c, dim=0)
    zc = torch.eye(c, dtype=torch.float64).repeat(r, 1)
    return torch.eye(r*c, dtype=torch.float64)+er*zr@zr.T+ec*zc@zc.T


@pytest.mark.parametrize('er,ec,targets', [(1., 1., [80/9, 119/30, 167/60]), (2., .5, [6., 109/24, 571/168])])
def test_optima_feasibility_and_floors(er, ec, targets):
    C = torch.cartesian_prod(torch.tensor([-1., 1.]), torch.tensor([-1., 1.]), torch.tensor([-1., 0., 1.])).double()
    F = torch.cat((torch.ones(12, 1), C, C[:, 2:]**2), 1)
    for criterion, optimum in zip('DAI', targets):
        result = optimal_design(F, 6, strip=(2, 3), row_hard=C[:, 0].long(), col_hard=C[:, 1].long(),
                                eta_row=er, eta_col=ec, criterion=criterion, generator=torch.Generator().manual_seed(0))
        ids = result['ids']; X = F[ids]; M = X.T@torch.linalg.inv(covariance(2, 3, er, ec))@X
        value = torch.linalg.det(M) if criterion == 'D' else torch.trace(torch.linalg.solve(M, torch.eye(5, dtype=F.dtype) if criterion == 'A' else F.T@F/12))
        assert value.item() == pytest.approx(optimum, rel=1e-12)
        efficiency = (value.item()/optimum)**.2 if criterion == 'D' else optimum/value.item()
        assert 0 < result['efficiency_lower_bound'] <= efficiency+1e-12
        assert ids.unique().numel() == 6
        assert all(C[g, 0].unique().numel() == 1 for g in ids.reshape(2, 3))
        assert all(C[g, 1].unique().numel() == 1 for g in ids.reshape(2, 3).T)
        assert result['certificate'] == 'LOCAL_STRIP_PLOT_EXCHANGE'


def test_rectangular_information():
    F = torch.randn(12, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    r, c, er, ec = 3, 4, 2., .5
    ar, ac = er/(1+er*c), ec/(1+ec*r)
    b = (1-1/(1+er*c)-1/(1+ec*r)+1/(1+er*c+ec*r))/(r*c)
    M, _ = _strip_information(F, torch.arange(12), (r, c), ar, ac, b)
    torch.testing.assert_close(M, F.T@torch.linalg.inv(covariance(r, c, er, ec))@F, rtol=1e-12, atol=1e-12)
