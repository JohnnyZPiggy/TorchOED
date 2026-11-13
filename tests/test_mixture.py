import math
from fractions import Fraction

import pytest
import torch

from torchoed import optimal_design, polynomial_features, simplex_lattice


def test_simplex_lattice_exact_rows():
    for q, m in ((3, 4), (4, 4), (3, 3)):
        rows = simplex_lattice(q, m).tolist()
        assert len(rows) == len({tuple(row) for row in rows}) == math.comb(q + m - 1, m)
        for row in rows:
            counts = [round(x * m) for x in row]
            assert row == [a / m for a in counts]
            assert sum(Fraction(a, m) for a in counts) == 1


@pytest.mark.parametrize("q,n,logdet", [(3, 6, -6 * math.log(4)), (4, 10, -12 * math.log(4))])
def test_scheffe_quadratic_kiefer_bound(q, n, logdet):
    p = polynomial_features(simplex_lattice(q, 4), 2, include_bias=False)
    f = torch.cat((p[:, :q], p[:, 2 * q:]), dim=1)
    result = optimal_design(f, n, generator=torch.Generator().manual_seed(0))
    # Kiefer's D-optimal bound is attained by the {q, 2} lattice.
    assert result["objective"] == pytest.approx(logdet, rel=0, abs=1e-12)
