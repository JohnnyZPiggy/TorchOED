import math
from fractions import Fraction
from itertools import combinations

import pytest
import torch

from torchoed import optimal_design, polynomial_features, simplex_lattice, simplex_centroid, mixture_vertices


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



@pytest.mark.parametrize("q", [3, 4])
def test_simplex_centroid(q):
    expected = [[float(Fraction(1, len(s))) if i in s else 0 for i in range(q)]
                for n in range(1, q + 1) for s in combinations(range(q), n)]
    actual = simplex_centroid(q)
    assert actual.shape == (2**q - 1, q)
    assert actual.tolist() == expected


@pytest.mark.parametrize("lower,upper,expected", [
    ([.2,.1,.1], [.6,.5,.6], [
        ("1/5","1/5","3/5"), ("1/5","1/2","3/10"), ("3/10","1/10","3/5"),
        ("2/5","1/2","1/10"), ("3/5","1/10","3/10"), ("3/5","3/10","1/10"),
        ("1/5","7/20","9/20"), ("1/4","3/20","3/5"), ("3/10","1/2","1/5"),
        ("9/20","1/10","9/20"), ("1/2","2/5","1/10"), ("3/5","1/5","1/5"),
        ("23/60","17/60","1/3")]),
    ([.2,.2,0], [1,1,1], [
        ("1/5","1/5","3/5"), ("1/5","4/5","0"), ("4/5","1/5","0"),
        ("1/5","1/2","3/10"), ("1/2","1/5","3/10"), ("1/2","1/2","0"),
        ("2/5","2/5","1/5")]),
])
def test_mixture_exact_gate_rows(lower, upper, expected):
    assert mixture_vertices(lower, upper, face_dims=(1,2)).tolist() == [
        [float(Fraction(x)) for x in row] for row in expected]


def test_mixture_r3_faces():
    lower, upper = [.1]*4, [.6,.5,.4,.3]
    counts = [len(mixture_vertices(lower, upper))]
    counts += [len(mixture_vertices(lower, upper, face_dims=(d,))) - counts[0] for d in (1,2,3)]
    assert counts == [11,17,8,1]
    assert mixture_vertices(lower, upper, face_dims=(1,2,3))[-1].tolist() == [
        float(Fraction(x)) for x in ("18/55","14/55","13/55","2/11")]


def test_mixture_collapsed_and_empty():
    assert mixture_vertices([0,.1,.9], [1,1,1], face_dims=(1,2)).tolist() == [[0,.1,.9]]
    with pytest.raises(ValueError):
        mixture_vertices([.5]*3, [1]*3)
