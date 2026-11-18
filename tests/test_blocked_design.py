import math
from fractions import Fraction
from itertools import combinations

import pytest
import torch

from oracle_exchange import linear_value
from torchoed import ff2n, full_factorial, optimal_design
from torchoed.d_optimal_design import (
    _block_information, _block_scores, _split_information, _split_ratios, _split_scores,
)


def test_blocked_scores_match_dense_projection():
    F = torch.randn(12, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(14))
    ids, blocks = torch.arange(7), [3, 4]
    Z = torch.eye(2, dtype=F.dtype)[torch.tensor([0]*3 + [1]*4)]
    projection = torch.eye(7, dtype=F.dtype) - Z @ torch.linalg.solve(Z.mT @ Z, Z.mT)
    dense = lambda ids: F[ids].mT @ projection @ F[ids]
    M, means = _block_information(F, ids, blocks)
    torch.testing.assert_close(M, dense(ids), rtol=0, atol=1e-12)
    logdet = torch.linalg.slogdet(dense(ids)).logabsdet
    counts = [0, 0]
    for i, choices, gains, interchange in _block_scores(F, ids, blocks, means, torch.linalg.cholesky(M)):
        for j, gain in zip(choices.tolist(), gains):
            trial = ids.clone()
            trial[i] = ids[j] if interchange else j
            if interchange:
                trial[j] = ids[i]
            expected = torch.linalg.slogdet(dense(trial)).logabsdet - logdet
            torch.testing.assert_close(gain, expected, rtol=0, atol=1e-10)
            counts[interchange] += 1
    assert counts == [35, 12]


def test_blocked_objective_matches_fraction_optimum():
    rows = [(-3, 0), (-2, 1), (0, -2), (0, 0), (1, 2), (2, -1), (3, 1)]
    determinants = []
    for subset in combinations(range(len(rows)), 5):
        for first in combinations(subset, 2):
            second = [i for i in subset if i not in first]
            xx, xy, yy = (
                sum(Fraction(rows[i][a] * rows[i][b]) for i in subset)
                - sum(Fraction(sum(rows[i][a] for i in block) * sum(rows[i][b] for i in block),
                               len(block)) for block in (first, second))
                for a, b in ((0, 0), (0, 1), (1, 1))
            )
            determinants.append(xx*yy - xy*xy)
    F = torch.tensor(rows, dtype=torch.float64)
    result = optimal_design(F, 5, blocks=[2, 3], generator=torch.Generator().manual_seed(0))
    assert result["objective"] == pytest.approx(math.log(max(determinants)), rel=0, abs=1e-12)


def test_factorial_blocking_confounds_only_abc():
    X = ff2n(3)
    a, b, c = X.unbind(1)
    F = torch.stack((a, b, c, a*b, a*c, b*c), dim=1)
    result = optimal_design(F, 8, blocks=[4, 4], generator=torch.Generator().manual_seed(0))
    assert result["objective"] == pytest.approx(6*math.log(8), rel=0, abs=1e-12)
    for block in result["ids"].split(4):
        assert (X[block].prod(1) == X[block[0]].prod()).all()
        assert torch.equal(block, block.sort().values)


@pytest.mark.parametrize("criterion", ["A", "I"])
def test_blocked_linear_scores_match_dense_projection(criterion):
    F = torch.randn(12, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(14))
    ids, blocks = torch.arange(7), [3, 4]
    Z = torch.eye(2, dtype=F.dtype)[torch.tensor([0]*3 + [1]*4)]
    projection = torch.eye(7, dtype=F.dtype) - Z @ torch.linalg.solve(Z.mT @ Z, Z.mT)
    B = torch.eye(3, dtype=F.dtype) if criterion == "A" else torch.cov(F.mT, correction=0)

    def dense(selection):
        rows = F[selection]
        return rows.mT @ projection @ rows

    M, means = _block_information(F, ids, blocks)
    baseline = dense(ids)
    phi = torch.trace(torch.linalg.solve(baseline, B))
    det = torch.linalg.det(baseline)
    counts = [0, 0]
    for i, choices, gains, interchange, ratios in _block_scores(F, ids, blocks, means, torch.linalg.cholesky(M), B):
        for j, gain, ratio in zip(choices.tolist(), gains, ratios):
            trial = ids.clone()
            trial[i] = ids[j] if interchange else j
            if interchange:
                trial[j] = ids[i]
            candidate = dense(trial)
            expected = phi - torch.trace(torch.linalg.solve(candidate, B))
            torch.testing.assert_close(gain, expected, rtol=1e-10, atol=0)
            torch.testing.assert_close(ratio, torch.linalg.det(candidate)/det, rtol=1e-10, atol=0)
            counts[interchange] += 1
    assert counts == [35, 12]

    result = optimal_design(F, 7, blocks=blocks, criterion=criterion, generator=torch.Generator().manual_seed(0))
    assert result["objective"] == pytest.approx(-torch.trace(torch.linalg.solve(dense(result["ids"]), B)).item(),
                                                 rel=1e-10, abs=0)
    assert result["blocks"] == blocks and result["criterion"] == criterion
    assert result["stop_reason"] == result["certificate"] == "LOCAL_BLOCK_EXCHANGE"
    assert 0 < result["efficiency_lower_bound"] <= 1 + 1e-12


def test_blocked_linear_objectives_match_fraction_optima():
    x, y = (full_factorial([3, 3]).long() - 1).unbind(1)
    F = torch.stack((x, y, x.square(), y.square(), x*y), dim=1)
    rows = F.tolist()
    a, b, c = ff2n(3).unbind(1)
    factorial = torch.stack((a, b, c, a*b, a*c, b*c), dim=1)
    for criterion, expected in (("A", Fraction(70, 27)), ("I", Fraction(208, 243))):
        values = []
        for subset in combinations(range(len(rows)), 7):
            for first in combinations(subset, 3):
                second = [i for i in subset if i not in first]
                value = linear_value(rows, subset, criterion, blocks=[first, second])
                if value is not None:
                    values.append(value)
        assert len(values) == 1024
        assert min(values) == expected
        for features, sizes, optimum in ((F, [3, 4], expected), (factorial, [4, 4], Fraction(3, 4))):
            result = optimal_design(features.double(), sum(sizes), blocks=sizes, criterion=criterion,
                                    restarts=20, generator=torch.Generator().manual_seed(0))
            assert result["objective"] == pytest.approx(-float(optimum), rel=0, abs=1e-10)


def test_split_plot_scores_match_dense_gls():
    F = torch.randn(12, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(14))
    ids, sizes, eta = torch.arange(7), [3, 4], 0.75
    Z = torch.eye(2, dtype=F.dtype)[torch.tensor([0]*3 + [1]*4)]
    V = torch.eye(7, dtype=F.dtype) + eta * (Z @ Z.mT)
    dense = lambda selection: F[selection].mT @ torch.linalg.solve(V, F[selection])
    counts = F.new_tensor(sizes)
    M, sums = _split_information(F, ids, sizes, 1 / (counts * (1 + eta * counts)))
    torch.testing.assert_close(M, dense(ids), rtol=0, atol=1e-12)
    reference = torch.linalg.slogdet(dense(ids)).logabsdet
    L = torch.linalg.cholesky(M)
    rows, totals = torch.linalg.solve_triangular(
        L, torch.cat((F, sums)).mT, upper=False).mT.split((len(F), len(sizes)))
    incoming = torch.arange(len(ids), len(F))
    for group, total in zip(torch.arange(len(ids)).split(sizes), totals):
        c = eta / (1 + eta * len(group))
        for pos in group.tolist():
            gains = _split_ratios(rows[incoming], rows[ids[pos]], total, c).log()
            for j, gain in zip(incoming.tolist(), gains):
                trial = ids.clone()
                trial[pos] = j
                expected = torch.linalg.slogdet(dense(trial)).logabsdet - reference
                torch.testing.assert_close(gain, expected, rtol=0, atol=1e-10)


def test_split_plot_matches_frozen_exhaustive_optimum():
    h, x = torch.cartesian_prod(torch.tensor([-1, 1]), torch.arange(-1, 2)).repeat(2, 1).T
    F = torch.stack((torch.ones_like(h), h, x, x.square(), h*x), dim=1).double()
    result = optimal_design(F, 6, whole_plots=[3, 3], hard=h, generator=torch.Generator().manual_seed(0))
    # P2_GATE §7.6 enumerates all 420 feasible designs; max det(M) = 48.
    assert result["objective"] == pytest.approx(math.log(48), rel=0, abs=1e-12)
    plots = h[result["ids"]].reshape(2, 3)
    assert sorted(plots[:, 0].tolist()) == [-1, 1]
    assert (plots == plots[:, :1]).all()


def test_split_plot_feasibility_and_small_eta_limit():
    h, x = torch.cartesian_prod(torch.tensor([-2, 0, 3]), torch.arange(-2, 3)).T
    F = torch.stack((torch.ones_like(h), h, x, x.square(), h*x), dim=1).double()
    sizes = [2, 3, 3]
    result = optimal_design(F, 8, whole_plots=sizes, hard=h, eta=1e-12,
                            generator=torch.Generator().manual_seed(0))
    ids = result["ids"]
    assert ids.unique().numel() == 8
    assert all((plot == plot[0]).all() for plot in h[ids].split(sizes))
    rows = F[ids]
    assert result["objective"] == pytest.approx(torch.linalg.slogdet(rows.mT @ rows).logabsdet.item(),
                                                rel=0, abs=1e-9)


@pytest.mark.parametrize("criterion", ["A", "I"])
def test_split_plot_linear_scores_match_dense_gls(criterion):
    F = torch.randn(12, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(14))
    hard, ids, sizes, eta = torch.arange(12)//6, torch.tensor([0, 1, 2, 6, 7, 8]), [3, 3], 0.7
    Z = torch.eye(2, dtype=F.dtype).repeat_interleave(3, dim=0)
    V = torch.eye(6, dtype=F.dtype) + eta * (Z @ Z.mT)
    B = torch.eye(3, dtype=F.dtype) if criterion == "A" else F.mT @ F / len(F)

    def dense(selection):
        rows = F[selection]
        return rows.mT @ torch.linalg.solve(V, rows)

    counts = F.new_tensor(sizes)
    M, sums = _split_information(F, ids, sizes, 1 / (counts * (1 + eta*counts)))
    unused = torch.ones(len(F), dtype=torch.bool)
    unused[ids] = False
    groups = torch.arange(len(ids)).split(sizes)
    coefficients = (eta / (1 + eta*counts)).tolist()
    baseline = dense(ids)
    phi, det = torch.trace(torch.linalg.solve(baseline, B)), torch.linalg.det(baseline)
    checked = 0
    for pos, pool, gains, ratios in _split_scores(F, ids, hard, groups, coefficients, sums,
                                                 torch.linalg.cholesky(M), unused, B):
        for j, gain, ratio in zip(pool, gains, ratios):
            trial = ids.clone()
            trial[pos] = j
            candidate = dense(trial)
            torch.testing.assert_close(gain, phi - torch.trace(torch.linalg.solve(candidate, B)),
                                       rtol=1e-10, atol=0)
            torch.testing.assert_close(ratio, torch.linalg.det(candidate)/det, rtol=1e-10, atol=0)
            checked += 1
    assert checked == 18

    result = optimal_design(F, 6, whole_plots=sizes, hard=hard, eta=eta, criterion=criterion,
                            generator=torch.Generator().manual_seed(0))
    plots = hard[result["ids"]].reshape(2, 3)
    assert (plots == plots[:, :1]).all()
    expected = -torch.trace(torch.linalg.solve(dense(result["ids"]), B)).item()
    assert result["objective"] == pytest.approx(expected, rel=1e-10, abs=0)
    assert result["criterion"] == criterion
    assert result["stop_reason"] == result["certificate"] == "LOCAL_SPLIT_PLOT_EXCHANGE"
    assert 0 < result["efficiency_lower_bound"] <= 1 + 1e-12


@pytest.mark.parametrize("size,eta,total,singular,a,i", [
    (3, Fraction(1), 420, 164, Fraction(35, 12), Fraction(11, 6)),
    (2, Fraction(1, 2), 1380, 516, Fraction(71, 24), Fraction(103, 72)),
])
def test_split_plot_linear_objectives_match_fraction_optima(size, eta, total, singular, a, i):
    h, x = torch.cartesian_prod(torch.tensor([-1, 1]), torch.arange(-1, 2)).repeat(2, 1).T
    F = torch.stack((torch.ones_like(h), h, x, x.square(), h*x), dim=1)
    rows = F.tolist()
    plots = [plot for plot in combinations(range(len(F)), size)
             if len({rows[j][1] for j in plot}) == 1]
    sizes = [size]*(6//size)
    layouts = [groups for groups in combinations(plots, len(sizes))
               if len({j for plot in groups for j in plot}) == 6]
    assert len(layouts) == total
    for criterion, expected in (("A", a), ("I", i)):
        values = [linear_value(rows, (), criterion, blocks=groups, eta=eta) for groups in layouts]
        assert values.count(None) == singular
        optimum = min(value for value in values if value is not None)
        assert optimum == expected
        result = optimal_design(F.double(), 6, whole_plots=sizes, hard=h, eta=float(eta),
                                criterion=criterion, restarts=20, generator=torch.Generator().manual_seed(0))
        assert result["objective"] == pytest.approx(-float(optimum), rel=0, abs=1e-10)



def test_combined_information_identities():
    F = torch.randn(7, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(14))
    sizes, blocks, eta = [3, 2, 2], [5, 2], 0.7
    Z = torch.eye(3, dtype=F.dtype).repeat_interleave(torch.tensor(sizes), dim=0)
    Zb = torch.eye(2, dtype=F.dtype).repeat_interleave(torch.tensor(blocks), dim=0)
    Vi = torch.linalg.inv(torch.eye(7, dtype=F.dtype) + eta * Z @ Z.mT)
    P = Vi - Vi @ Zb @ torch.linalg.solve(Zb.mT @ Vi @ Zb, Zb.mT @ Vi)
    weights = lambda sizes: 1 / (F.new_tensor(sizes) * (1 + eta * F.new_tensor(sizes)))
    M, _ = _split_information(F, torch.arange(7), sizes, weights(sizes), blocks)
    torch.testing.assert_close(M, F.mT @ P @ F, rtol=0, atol=1e-12)
    M, _ = _split_information(F, torch.arange(6), [1]*6, weights([1]*6), [3, 3])
    blocked, _ = _block_information(F, torch.arange(6), [3, 3])
    torch.testing.assert_close(M, blocked/(1+eta), rtol=0, atol=1e-12)
    M, _ = _split_information(F, torch.arange(7), sizes, weights(sizes), [7])
    MW, _ = _split_information(torch.cat((torch.ones(7, 1), F), 1), torch.arange(7), sizes, weights(sizes))
    tau = sum(size/(1+eta*size) for size in sizes)
    torch.testing.assert_close(torch.linalg.det(MW), tau*torch.linalg.det(M), rtol=0, atol=1e-12)


def test_combined_frozen_exhaustive_optima():
    h, x = torch.cartesian_prod(torch.tensor([-1, 1]), torch.arange(-1, 2)).repeat(2, 1).T
    F = torch.stack((h, x, x.square(), h*x), 1).double()
    for sizes, blocks, eta, optima in (([2]*4, [4, 4], 1, (640/9, 177/103, 127/144)),
                                        ([2]*3, [2, 4], 0.5, (16, 19/8, 97/72))):
        for criterion, optimum in zip('DAI', optima):
            result = optimal_design(F, sum(sizes), whole_plots=sizes, blocks=blocks, hard=h,
                                    eta=eta, criterion=criterion, generator=torch.Generator().manual_seed(0))
            expected = math.log(optimum) if criterion == 'D' else -optimum
            assert result['objective'] == pytest.approx(expected, rel=1e-9)
            assert result['efficiency_lower_bound'] <= 1+1e-12
            assert result['blocks'] == blocks
            assert all(plot.unique().numel() == 1 for plot in h[result['ids']].split(sizes))


def test_combined_fixed_ids_and_boundary():
    h, x = torch.cartesian_prod(torch.tensor([-1, 1]), torch.arange(-1, 2)).repeat(2, 1).T
    F = torch.stack((h, x, x.square(), h*x), 1).double()
    result = optimal_design(F, 8, whole_plots=[2]*4, blocks=[4, 4], hard=h, fixed_ids=[0, 6],
                            generator=torch.Generator().manual_seed(0))
    assert result['ids'][:2].tolist() == [0, 6]
    assert math.exp(result['objective']) == pytest.approx(220/9, rel=1e-9)
    with pytest.raises(ValueError):
        optimal_design(F, 8, whole_plots=[2]*4, blocks=[3, 5], hard=h)


@pytest.mark.parametrize("case", (0, 1))
def test_covariate_frozen_exhaustive_oracles(case):
    h, x = torch.cartesian_prod(torch.tensor([-1., 1.], dtype=torch.float64), torch.tensor([-1., 0., 1.], dtype=torch.float64)).mT
    F = torch.stack((h, x, h*x if case == 0 else x*x), 1)
    n = 5 + case
    t = torch.arange(n, dtype=F.dtype)
    Z = t[:, None] if case == 0 else torch.stack((t, (-1)**t), 1)
    X = torch.cat((torch.ones(n, 1, dtype=F.dtype), Z), 1)
    Q = torch.eye(n, dtype=F.dtype) - X @ torch.linalg.lstsq(X, torch.eye(n, dtype=F.dtype)).solution
    expected = (math.log(1568/25), -149/196, -7/12) if case == 0 else (math.log(256/9), -19/16, -25/48)
    for criterion, value in zip(("D", "A", "I"), expected):
        result = optimal_design(F, n, criterion=criterion, covariates=Z, restarts=20, generator=torch.Generator().manual_seed(0))
        ids = result["ids"]
        assert torch.equal(result["covariates"], Z) and len(set(ids.tolist())) == n
        M = F[ids].mT @ Q @ F[ids]
        B = torch.eye(3, dtype=F.dtype) if criterion == "A" else (F-F.mean(0)).mT @ (F-F.mean(0))/6
        dense = torch.linalg.slogdet(M).logabsdet.item() if criterion == "D" else -torch.trace(torch.linalg.solve(M, B)).item()
        assert result["objective"] == pytest.approx(value, abs=1e-12, rel=0)
        assert result["objective"] == pytest.approx(dense, abs=1e-12, rel=0)
        assert result["certificate"] == "LOCAL_COVARIATE_EXCHANGE" and 0 < result["efficiency_lower_bound"] <= 1 + 1e-12


def test_covariate_blocks_reduction_and_fixed_order():
    a, b, c = ff2n(3).mT
    F = torch.stack((a,b,c,a*b,a*c,b*c), 1)
    ids = (a*b*c).argsort(stable=True)
    Z = torch.tensor([[0.]]*4 + [[1.]]*4, dtype=F.dtype)
    X = torch.cat((torch.ones(8, 1, dtype=F.dtype), Z), 1)
    Q = torch.eye(8, dtype=F.dtype) - X @ torch.linalg.solve(X.mT @ X, X.mT)
    torch.testing.assert_close(F[ids].mT @ Q @ F[ids], _block_information(F, ids, [4,4])[0], atol=1e-12, rtol=0)
    result = optimal_design(F, 8, covariates=Z, fixed_ids=ids[:3], generator=torch.Generator().manual_seed(0))
    assert torch.equal(result["ids"][:3], ids[:3])


def test_covariate_boundary():
    F = torch.cartesian_prod(torch.tensor([-1., 1.], dtype=torch.float64), torch.tensor([-1., 0., 1.], dtype=torch.float64))
    Z = torch.arange(5, dtype=F.dtype)[:,None]
    for kwargs in ({"blocks":[5]}, {"covariates":Z[:4]}, {"covariates":torch.ones_like(Z)},
                   {"covariates":Z*float("nan")}, {"n":3,"covariates":Z[:3]}):
        args = dict(n=5, covariates=Z); args.update(kwargs)
        with pytest.raises(ValueError): optimal_design(F, **args)
