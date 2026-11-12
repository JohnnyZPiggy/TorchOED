import math
from fractions import Fraction
from itertools import combinations

import pytest
import torch

from torchoed import ff2n, optimal_design
from torchoed.d_optimal_design import _block_information, _block_scores


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
