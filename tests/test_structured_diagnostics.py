import pytest
import torch

from torchoed import design_diagnostics, full_factorial
from torchoed.information import _coefficient_power


def test_structured_diagnostics_dense():
    generator = torch.Generator().manual_seed(231)
    effect = torch.tensor([0.2, 0.4, 0.8], dtype=torch.float64)
    for name, sizes in (("blocks", [3, 4]), ("whole_plots", [3, 2, 3])):
        F = torch.randn(12, 3, dtype=torch.float64, generator=generator)
        ids = torch.randperm(len(F), generator=generator)[:sum(sizes)]
        groups = torch.arange(len(sizes)).repeat_interleave(torch.tensor(sizes))
        Z = torch.nn.functional.one_hot(groups).to(F.dtype)
        identity = torch.eye(len(ids), dtype=F.dtype)
        eta = 0.7
        if name == "blocks":
            metric = projection = identity - Z @ torch.linalg.inv(Z.mT @ Z) @ Z.mT
            df = [len(ids) - len(sizes) - F.shape[1]] * F.shape[1]
            columns = [0, 1, 2]
        else:
            F[:, 0] = 1
            F[ids, 1] = groups.to(F.dtype)
            metric = torch.linalg.inv(identity + eta * Z @ Z.mT)
            weights = metric.sum(1)
            projection = metric - torch.outer(weights, weights) / weights.sum()
            whole = [all(rows[:, j].unique().numel() == 1 for rows in F[ids].split(sizes))
                     for j in range(F.shape[1])]
            df = [len(sizes) - sum(whole) if w else len(ids) - len(sizes) - whole.count(False)
                  for w in whole]
            columns = [1, 2]
        R = F[2:]
        d = design_diagnostics(F, ids, region=R, effect=effect, eta=eta, **{name: sizes})
        S = F[ids]
        X = S[:, columns]
        C = X.mT @ projection @ X
        inverse = torch.linalg.inv(S.mT @ metric @ S)
        if name == "blocks":
            R = R - R.mean(0)
        spv = len(ids) * (R @ inverse @ R.mT).diagonal()
        power = torch.cat([_coefficient_power(lam[None], dof, 0.05)
                           for lam, dof in zip(effect.square() / inverse.diagonal(), df)])
        assert d["columns"] == columns
        assert torch.equal(d["df"], torch.tensor(df, dtype=torch.int64))
        expected = {"spv": spv, "fds": spv.sort().values, "power": power,
                    "correlation": C / torch.outer(C.diagonal().sqrt(), C.diagonal().sqrt()),
                    "vif": C.diagonal() * torch.linalg.inv(C).diagonal()}
        for key, value in expected.items():
            torch.testing.assert_close(d[key], value, rtol=1e-10, atol=1e-12)


def test_structured_diagnostics_gate_oracles():
    F = 2 * full_factorial([2, 2, 2]).double() - 1
    blocks = F.prod(1).argsort()
    h, x = F[:, 0], F[:, 2]
    split = torch.stack((torch.ones_like(h), h, x, h * x), dim=1)
    for features, ids, structure, spv, df, multiplier in (
        (F, blocks, {"blocks": (4, 4)}, 3, 3, [8, 8, 8]),
        (split, torch.arange(8), {"whole_plots": (2, 2, 2, 2), "eta": 1},
         8, 2, [8 / 3, 8 / 3, 8, 8]),
    ):
        effect = F.new_tensor([0.2, 0.4, 0.6, 0.8])[:features.shape[1]]
        d = design_diagnostics(features, ids, effect=effect, **structure)
        expected = {"correlation": torch.eye(3, dtype=F.dtype), "vif": F.new_ones(3),
                    "spv": F.new_full((8,), spv),
                    "power": _coefficient_power(effect.square() * F.new_tensor(multiplier), df, 0.05)}
        assert torch.equal(d["df"], ids.new_full((features.shape[1],), df))
        for key, value in expected.items():
            torch.testing.assert_close(d[key], value, rtol=0, atol=1e-12)


def test_structured_diagnostics_invalid_inputs():
    F = 2 * full_factorial([2, 2, 2]).double() - 1
    ids = F.prod(1).argsort()
    a, b, c = F.unbind(1)
    saturated = torch.cat((F, torch.stack((a * b, a * c, b * c), dim=1)), dim=1)
    with pytest.raises(ValueError, match="boundaries"):
        design_diagnostics(F, ids, blocks=[3, 5], whole_plots=[2, 2, 2, 2])
    with pytest.raises(ValueError, match="summing"):
        design_diagnostics(F, ids, blocks=[3, 4])
    with pytest.raises(ValueError, match="degrees of freedom"):
        design_diagnostics(saturated, ids, blocks=[4, 4], effect=0.5)


def test_group_power_scipy_oracles_and_one_term():
    F = 2 * full_factorial([2, 2, 2]).double() - 1
    h, x = F[:, 0], F[:, 2]
    split = torch.stack((torch.ones_like(h), h, x, h * x), dim=1)
    plain = torch.cat((F.new_ones((8, 1)), F), dim=1)
    for features, ids, structure, effect, group, expected, df in (
        (plain, torch.arange(8), {}, [0, .5, .5, .5], [1, 2, 3], 0.23428437233178745, None),
        (F, F.prod(1).argsort(), {"blocks": [4, 4]}, [.5, .5, 0], [0, 1], 0.18506177481105285, 3),
        (split, torch.arange(8), {"whole_plots": [2]*4, "eta": 1}, [.5, .5, 0, 0], [0, 1],
         0.08114470454209445, 2),
    ):
        groups = [group] + [[j] for j in range(features.shape[1])]
        d = design_diagnostics(features, ids, effect=F.new_tensor(effect), groups=groups, **structure)
        torch.testing.assert_close(d["group_power"][0], F.new_tensor(expected), rtol=0, atol=1e-10)
        assert torch.equal(d["group_power"][1:], d["power"])
        if df is not None:
            assert torch.equal(d["group_df"], ids.new_full((len(groups),), df))


def test_group_power_trust_boundary():
    F = 2 * full_factorial([2, 2, 2]).double() - 1
    h, x = F[:, 0], F[:, 2]
    split = torch.stack((torch.ones_like(h), h, x, h * x), dim=1)
    ids = torch.arange(8)
    with pytest.raises(ValueError, match="mixed"):
        design_diagnostics(split, ids, effect=.5, groups=[[1, 2]], whole_plots=[2]*4)
    for groups, effect in (([[1]], None), ([[]], .5), ([[4]], .5), ([], .5),
                           ([[1, 1]], .5), ([[True]], .5), (((1,),), .5)):
        with pytest.raises(ValueError, match="groups"):
            design_diagnostics(split, ids, effect=effect, groups=groups)


def test_covariate_diagnostics_blocks_reduction():
    F = 2 * full_factorial([2, 2, 2]).double() - 1
    a, b, c = F.unbind(1)
    F = torch.cat((F, torch.stack((a*b, a*c, b*c), 1)), 1)
    ids = (a*b*c).argsort()
    Z = F.new_tensor([0]*4 + [1]*4)[:, None]
    d = design_diagnostics(F, ids, covariates=Z)
    blocked = design_diagnostics(F, ids, blocks=[4, 4])
    for key in ('correlation', 'vif', 'spv', 'fds'):
        torch.testing.assert_close(d[key], blocked[key], rtol=0, atol=1e-12)
    for structure in ({'covariates': Z}, {'blocks': [4, 4]}):
        with pytest.raises(ValueError, match='degrees of freedom'):
            design_diagnostics(F, ids, effect=1.0, groups=[[0, 1, 2], [3]], **structure)
    F = F[:, :4]
    d = design_diagnostics(F, ids, covariates=Z, effect=1.0, groups=[[0, 1, 2], [3]])
    blocked = design_diagnostics(F, ids, blocks=[4, 4], effect=1.0, groups=[[0, 1, 2], [3]])
    for key in ('correlation', 'vif', 'spv', 'power', 'group_power'):
        torch.testing.assert_close(d[key], blocked[key], rtol=0, atol=1e-12)
    for key in ('df', 'group_df'):
        assert torch.equal(d[key], blocked[key])
        assert torch.equal(d[key], torch.full_like(d[key], 2))


def test_covariate_diagnostics_dense_oracle():
    h, x = torch.cartesian_prod(torch.tensor([-1., 1.]), torch.tensor([-1., 0., 1.])).double().unbind(1)
    F = torch.stack((h, x, h*x), 1)
    ids = torch.tensor([0, 2, 3, 5, 1])
    Z = torch.arange(5, dtype=F.dtype)[:, None]
    X = torch.cat((F.new_ones(5, 1), Z, F[ids]), 1)
    inverse = torch.linalg.inv(X.mT @ X)[2:, 2:]
    C = torch.linalg.inv(inverse)
    correlation = C / torch.outer(C.diagonal().sqrt(), C.diagonal().sqrt())
    R = F - F.mean(0)
    d = design_diagnostics(F, ids, covariates=Z)
    torch.testing.assert_close(d['vif'], torch.linalg.inv(correlation).diagonal(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(d['spv'], 5*(R @ inverse @ R.mT).diagonal(), rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match='degrees of freedom'):
        design_diagnostics(F, ids, covariates=Z, effect=1.0)


def test_covariate_diagnostics_trust_boundary():
    F = 2 * full_factorial([2, 2, 2]).double() - 1
    ids = torch.arange(8)
    Z = torch.arange(8, dtype=F.dtype)[:, None]
    for z, extra in ((Z, {'blocks': [4, 4]}), (Z, {'alias': F}),
                     (Z[:-1], {}), (torch.ones_like(Z), {})):
        with pytest.raises(ValueError):
            design_diagnostics(F, ids, covariates=z, **extra)


def _combined_candidates():
    h, x = torch.cartesian_prod(torch.tensor([-1., 1.]), torch.tensor([-1., 0., 1.])).double().unbind(1)
    return torch.stack((h, x, x*x, h*x), 1).repeat(2, 1), torch.tensor([0, 2, 3, 4, 9, 11, 7, 8])


def test_combined_diagnostics_dense_gls():
    F, ids = _combined_candidates()
    Z = torch.eye(4, dtype=F.dtype).repeat_interleave(2, 0)
    B = torch.eye(2, dtype=F.dtype).repeat_interleave(4, 0)
    X = torch.cat((B, F[ids]), 1)
    V = torch.eye(8, dtype=F.dtype) + Z @ Z.mT
    sigma = torch.linalg.inv(X.mT @ torch.linalg.solve(V, X))[2:, 2:]
    C = torch.linalg.inv(sigma)
    R = F - F.mean(0)
    d = design_diagnostics(F, ids, blocks=[4, 4], whole_plots=[2]*4, eta=1.,
                           effect=1., groups=[[1, 2, 3]])
    expected = {'correlation': C / torch.outer(C.diagonal().sqrt(), C.diagonal().sqrt()),
                'vif': C.diagonal()*sigma.diagonal(), 'spv': 8*(R @ sigma @ R.mT).diagonal(),
                'power': _coefficient_power(1/sigma.diagonal(), 1, .05)}
    for key, value in expected.items():
        torch.testing.assert_close(d[key], value, rtol=1e-12, atol=1e-12)
    assert torch.equal(d['df'], torch.ones(4, dtype=torch.int64))
    assert d['group_df'].tolist() == [1]


def test_combined_diagnostics_reductions():
    F, ids = _combined_candidates()
    d = design_diagnostics(F, ids, blocks=[8], whole_plots=[2]*4, effect=1.)
    w = design_diagnostics(torch.cat((F.new_ones(12, 1), F), 1), ids,
                           whole_plots=[2]*4, effect=1.)
    for key in ('correlation', 'vif', 'power', 'df'):
        expected = w[key][1:] if key in ('power', 'df') else w[key]
        torch.testing.assert_close(d[key], expected, rtol=1e-12, atol=1e-12)
    assert torch.equal(d['df'], w['df'][1:])
    d = design_diagnostics(F, ids, blocks=[4, 4], whole_plots=[1]*8, eta=.5,
                           effect=1., groups=[[0, 1], [2, 3]])
    b = design_diagnostics(F, ids, blocks=[4, 4], effect=1/1.5**.5, groups=[[0, 1], [2, 3]])
    for key in ('correlation', 'vif', 'spv', 'power', 'group_power', 'df', 'group_df'):
        torch.testing.assert_close(d[key], b[key]*(1.5 if key == 'spv' else 1), rtol=1e-12, atol=1e-12)
    for key in ('df', 'group_df'):
        assert torch.equal(d[key], b[key])


def test_combined_diagnostics_trust_boundary():
    F, ids = _combined_candidates()
    for extra in ({'blocks': [3, 5]}, {'groups': [[0, 1]]}, {'covariates': F.new_zeros(8, 1)}):
        args = dict(blocks=[4, 4], whole_plots=[2]*4, effect=1.)
        args.update(extra)
        with pytest.raises(ValueError):
            design_diagnostics(F, ids, **args)
