"""Independent CPU oracles, named after contract §9 gates."""
import itertools
import json
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from torchoed import ff2n, full_factorial, lhs, polynomial_features, sobol


def tensor(rows):
    return torch.tensor(rows, dtype=torch.float64)


def scalar_features(rows, degree=2, bias=True):
    return [[*([1] if bias else []), *r,
             *([x*x for x in r] + [r[i]*r[j] for i in range(len(r))
                                    for j in range(i+1, len(r))] if degree == 2 else [])]
            for r in rows]


def exact_rank_det(features):
    """Fraction Gaussian elimination, independent of torch.linalg."""
    p = len(features[0])
    a = [[sum(Fraction(r[i])*Fraction(r[j]) for r in features)
          for j in range(p)] for i in range(p)]
    rank, det = 0, Fraction(1)
    for col in range(p):
        pivot = next((i for i in range(rank, p) if a[i][col]), None)
        if pivot is None:
            det = Fraction(0)
            continue
        if pivot != rank:
            a[pivot], a[rank] = a[rank], a[pivot]
            det = -det
        value = a[rank][col]
        det *= value
        for i in range(rank+1, p):
            scale = a[i][col]/value
            a[i] = [x-scale*y for x, y in zip(a[i], a[rank])]
        rank += 1
    return rank, det


def strata(x):
    n, k = x.shape
    assert x.device.type == 'cpu'
    assert bool(((x >= 0) & (x < 1)).all())
    assert torch.equal(torch.floor(n*x).to(torch.int64).sort(dim=0).values,
                       torch.arange(n)[:, None].expand(n, k))


def test_F1_order():
    expected = list(itertools.product(range(2), range(3)))
    x = full_factorial((2, 3))
    assert x.dtype == torch.int64 and x.tolist() == [list(r) for r in expected]


def test_F2_singletons():
    assert full_factorial((1, 3)).tolist() == [[0, 0], [0, 1], [0, 2]]
    assert full_factorial([3]).tolist() == [[0], [1], [2]]


def test_F3_signed_corners():
    for k in (1, 2, 3):
        x = ff2n(k)
        assert x.dtype == torch.float64
        assert x.tolist() == [list(r) for r in itertools.product((-1, 1), repeat=k)]
        ids = full_factorial([2]*k)
        assert len(torch.unique(ids, dim=0)) == 2**k
    assert ff2n(2, dtype=torch.float32).dtype == torch.float32


def test_F4_reject_before_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('allocation attempted for invalid/budget input')
    monkeypatch.setattr(torch, 'arange', forbidden)
    for levels in ((), [], (0,), (-1,), (1.0,), (True,), (2**63,), (1001, 1001)):
        with pytest.raises(ValueError):
            full_factorial(levels)
    with pytest.raises(ValueError):
        full_factorial((2, 3), max_elements=11)
    for k in (0, True, 64, 10**9):
        with pytest.raises(ValueError):
            ff2n(k)


def test_L1_strata():
    for dtype in (torch.float64, torch.float32):
        for seed in range(10):
            strata(lhs(5, 3, dtype=dtype, generator=torch.Generator().manual_seed(seed)))


def test_L2_single_sample():
    strata(lhs(1, 4, generator=torch.Generator().manual_seed(17)))


def test_L3_generator_replay_isolation():
    state = torch.random.get_rng_state().clone()
    g = torch.Generator().manual_seed(42)
    before = g.get_state().clone()
    x = lhs(5, 3, generator=g)
    assert not torch.equal(before, g.get_state())
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(x, lhs(5, 3, generator=torch.Generator().manual_seed(42)))


def test_L4_invalid():
    for n, k in ((0, 2), (-1, 2), (2, 0), (True, 2), (2.5, 2)):
        with pytest.raises(ValueError):
            lhs(n, k)
    for mode in ('maximin', 'centered', 'OA', 'sliced'):
        with pytest.raises(ValueError):
            lhs(5, 3, mode=mode)
    with pytest.raises(TypeError):
        lhs(5, 3, dtype=torch.int64)


def test_lhs_boundary_rounding(monkeypatch):
    def near_one(n, *, dtype, device, **kwargs):
        return torch.nextafter(torch.ones(n, dtype=dtype, device=device),
                               torch.zeros(n, dtype=dtype, device=device))
    monkeypatch.setattr(torch, 'rand', near_one)
    for dtype in (torch.float32, torch.float64):
        strata(lhs(5, 3, dtype=dtype, generator=torch.Generator().manual_seed(4)))


def test_S1_exact_points():
    assert torch.equal(sobol(4, 2), tensor([[0, 0], [.5, .5], [.75, .25], [.25, .75]]))


def test_S2_exact_count():
    assert sobol(3, 2).shape == (3, 2)


def test_S3_skip():
    assert torch.equal(sobol(3, 2, skip=1), tensor([[.5, .5], [.75, .25], [.25, .75]]))


def test_S4_scramble():
    state = torch.random.get_rng_state().clone()
    a = sobol(7, 3, scramble=True, seed=91)
    assert torch.equal(a, sobol(7, 3, scramble=True, seed=91))
    assert not torch.equal(a, sobol(7, 3, scramble=True, seed=92))
    assert torch.equal(state, torch.random.get_rng_state())
    assert bool(((a >= 0) & (a < 1)).all())


def test_S5_invalid_zero():
    assert sobol(0, 2).shape == (0, 2)
    for args in ((-1, 2), (1, 0), (True, 2), (1, 21202)):
        with pytest.raises(ValueError):
            sobol(*args)
    for kw in ({'skip': -1}, {'skip': 2**30}, {'balance': 'power2'},
               {'seed': -1}, {'seed': 2**63}, {'seed': True}):
        with pytest.raises(ValueError):
            sobol(3, 2, **kw)
    with pytest.raises(TypeError):
        sobol(3, 2, dtype=torch.int64)
    with pytest.raises(TypeError):
        sobol(3, 2, scramble=1)


def test_P1_degrees_order():
    c = tensor([[-1], [0], [1]])
    assert torch.equal(polynomial_features(c, 1), tensor([[1, -1], [1, 0], [1, 1]]))
    assert torch.equal(polynomial_features(c, 2), tensor([[1, -1, 1], [1, 0, 0], [1, 1, 1]]))
    rows = [[2, 3, 5], [-1, 4, 2]]
    assert torch.equal(polynomial_features(tensor(rows)), tensor(scalar_features(rows)))


def test_P2_bias():
    rows = [[2, 3], [-1, 4]]
    for degree in (1, 2):
        assert torch.equal(polynomial_features(tensor(rows), degree, False),
                           tensor(scalar_features(rows, degree, False)))


def test_P3_rank():
    for rows, rank in (([[-1], [0], [1]], 3), ([[-1], [1]], 2),
                       (list(itertools.product((-1, 1), repeat=2)), 4)):
        f = polynomial_features(tensor(rows))
        assert torch.linalg.matrix_rank(f).item() == rank
        assert exact_rank_det(scalar_features(rows))[0] == rank


def test_P4_batch():
    rows = [[[-1, 2], [3, 4]], [[0, 1], [-2, -3]]]
    for degree in (1, 2):
        for bias in (True, False):
            f = polynomial_features(tensor(rows), degree, bias)
            assert torch.equal(f, tensor([scalar_features(r, degree, bias) for r in rows]))
            assert torch.equal(f, torch.stack([polynomial_features(tensor(r), degree, bias) for r in rows]))


def test_P5_autograd_dtype_device():
    c = tensor([[.2, -.3], [.7, .9]]).requires_grad_()
    for degree in (1, 2):
        assert torch.autograd.gradcheck(lambda x: polynomial_features(x, degree), (c,),
                                       eps=1e-6, atol=1e-5, rtol=1e-3)
    for dtype in (torch.float64, torch.float32):
        x = c.detach().to(dtype)
        f = polynomial_features(x)
        assert f.dtype == dtype and f.device == x.device
        torch.testing.assert_close(f.double(), tensor(scalar_features(x.tolist())),
                                   atol=1e-6 if dtype == torch.float32 else 1e-9,
                                   rtol=1e-6 if dtype == torch.float32 else 1e-9)


def test_P6_actual_e0_and_separate_negative():
    fixture = json.loads((Path(__file__).parent/'fixtures/e0_p1.json').read_text())
    positive = fixture['C']
    cube = [[a, b, c, a*b*c] for a, b, c in itertools.product((-1, 1), repeat=3)]
    negative = cube + positive[8:]
    for rows, rank, determinant in ((positive, 15, 18014398509481984), (negative, 12, 0)):
        f = polynomial_features(tensor(rows))
        assert f.shape == (20, 15)
        assert torch.equal(f, tensor(scalar_features(rows)))
        assert torch.linalg.matrix_rank(f).item() == rank
        assert exact_rank_det(f.tolist()) == (rank, determinant)


def test_feature_rejections_and_empty():
    for c in (torch.ones(2, 1, dtype=torch.int64), tensor([[float('nan')]]),
              tensor([[float('inf')]]), torch.empty(2, 0), torch.ones(2)):
        with pytest.raises((ValueError, TypeError)):
            polynomial_features(c)
    for degree in (0, 3, True, 2.0):
        with pytest.raises(ValueError):
            polynomial_features(tensor([[1]]), degree)
    with pytest.raises(ValueError):
        polynomial_features(torch.empty(0, 10000))
    assert polynomial_features(torch.empty(0, 2)).shape == (0, 6)
    assert polynomial_features(torch.empty(0, 3, 2)).shape == (0, 3, 6)
