import pytest
import torch
from torchoed import optimal_design


@pytest.mark.parametrize('nested,fixed,targets', [
    (True, [1, 0, 4, 3], [7936/441, 27/8, 1903/744]),
    (True, [1, 0, 3], [1280/63, 933/280, 2083/840]),
    (False, [1, 4], [128/27, 137/32, 103/32]),
    (False, [1, 4, 2, 7], [80/27, 111/20, 69/20]),
])
def test_fixed_oracles_and_invariant(nested, fixed, targets):
    C = torch.cartesian_prod(torch.tensor([-1., 1.]), torch.tensor([-1., 1.]),
                             torch.tensor([-1., 0., 1.])).double()
    F = torch.cat((torch.ones(12, 1), C, C[:, 2:]**2), 1)
    if nested:
        n = 8
        kwargs = dict(whole_plots=[4, 4], subplots=[2]*4, hard=C[:, 0].long(), subplot_hard=C[:, 1].long())
        zr = torch.eye(2, dtype=F.dtype).repeat_interleave(4, 0)
        zc = torch.eye(4, dtype=F.dtype).repeat_interleave(2, 0)
    else:
        n = 6
        kwargs = dict(strip=(2, 3), row_hard=C[:, 0].long(), col_hard=C[:, 1].long())
        zr = torch.eye(2, dtype=F.dtype).repeat_interleave(3, 0)
        zc = torch.eye(3, dtype=F.dtype).repeat(2, 1)
    V = torch.eye(n, dtype=F.dtype)+zr@zr.T+zc@zc.T
    for criterion, optimum in zip('DAI', targets):
        result = optimal_design(F, n, fixed_ids=fixed, criterion=criterion, restarts=20,
                                generator=torch.Generator().manual_seed(0), **kwargs)
        ids = result['ids']; X = F[ids]; M = X.T@torch.linalg.solve(V, X)
        value = torch.linalg.det(M) if criterion == 'D' else torch.trace(torch.linalg.solve(
            M, torch.eye(5, dtype=F.dtype) if criterion == 'A' else F.T@F/12))
        assert value.item() == pytest.approx(optimum, rel=1e-12)
        assert ids.unique().numel() == n
        if nested:
            assert all(t in ids[2*(i//2):2*(i//2)+2] for i, t in enumerate(fixed))
            assert all(C[g, 0].unique().numel() == 1 for g in ids.split(4))
            assert all(C[g, 1].unique().numel() == 1 for g in ids.split(2))
        else:
            assert ids[:len(fixed)].tolist() == fixed
            assert all(C[g, 0].unique().numel() == 1 for g in ids.reshape(2, 3))
            assert all(C[g, 1].unique().numel() == 1 for g in ids.reshape(2, 3).T)


def test_infeasible_fixed_labels():
    C = torch.cartesian_prod(torch.tensor([-1., 1.]), torch.tensor([-1., 1.]),
                             torch.tensor([-1., 0., 1.])).double()
    F = torch.cat((torch.ones(12, 1), C, C[:, 2:]**2), 1)
    for fixed in ([0, 3], [0, 2, 6]):
        with pytest.raises(ValueError):
            optimal_design(F, 8, fixed_ids=fixed, whole_plots=[4, 4], hard=C[:, 0].long(),
                           subplots=[2]*4, subplot_hard=C[:, 1].long())
    for fixed in ([0, 8], [0, 4, 2, 9]):
        with pytest.raises(ValueError):
            optimal_design(F, 6, fixed_ids=fixed, strip=(2, 3),
                           row_hard=C[:, 0].long(), col_hard=C[:, 1].long())
