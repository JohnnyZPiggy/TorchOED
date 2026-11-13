import math
from fractions import Fraction
from itertools import combinations

import pytest
import torch

from torchoed import ff2n, optimal_design
from torchoed.d_optimal_design import _block_information, _block_scores, _split_information, _split_ratios


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
