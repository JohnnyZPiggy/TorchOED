"""C1-C9, independent Fraction expectations; also reusable on CUDA float64."""
from fractions import Fraction as F
import os

import pytest
import torch

from torchoed import c_optimality, a_optimality, exact_information, ff2n, polynomial_features
from oracle_contrast import cases, score, solve

DEVICE = os.environ.get('M2B_TEST_DEVICE', 'cpu')
DTYPES = [torch.float64] if DEVICE != 'cpu' else [torch.float64, torch.float32]
ERRORS = {}


@pytest.fixture(scope='session', autouse=True)
def report_errors():
    yield
    path = os.environ.get('M2B_ERRORS')
    if path:
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps(ERRORS, indent=2)+'\n', encoding='utf-8')


def t(x, dtype=torch.float64):
    if isinstance(x, (list, tuple)):
        x = [[float(v) for v in row] if isinstance(row, (list, tuple)) else float(row) for row in x]
    return torch.tensor(x, dtype=dtype, device=DEVICE)


def close(actual, expected, gate, tolerance=None):
    reference = torch.as_tensor(expected, dtype=actual.dtype, device=actual.device)
    tol = tolerance or (1e-9 if actual.dtype == torch.float64 else 1e-5)
    error = float((actual-reference).abs().max().item())
    ERRORS[gate] = max(ERRORS.get(gate, 0.), error)
    torch.testing.assert_close(actual, reference, atol=tol, rtol=tol)


@pytest.mark.parametrize('dtype', DTYPES)
def test_C1_diagonal(dtype):
    M = torch.diag(t([2, 5, F(1, 3)], dtype))
    value = c_optimality(M, t([1, -2, 3], dtype))
    assert value.dtype == dtype and value.device == M.device and value.shape == ()
    close(value, -float(F(1, 2)+F(4, 5)+27), 'C1')


def test_C2_line():
    features = t([[1, -1], [1, 0], [1, 1]])
    for ids, expected in (([0, 2], -F(13, 8)), ([0, 1, 2], -F(35, 24))):
        M = exact_information(features, torch.tensor(ids, device=DEVICE))
        close(c_optimality(M, t([1, F(3, 2)])), float(expected), 'C2')


@pytest.mark.parametrize('dtype', DTYPES)
def test_C3_units_independent(dtype):
    for matrix, c, _ in cases():
        M = t(matrix, dtype)
        p = M.shape[0]
        units = torch.eye(p, dtype=dtype, device=DEVICE)
        total = sum(c_optimality(M, e) for e in units)
        expected = sum(score(matrix, [int(i == j) for j in range(p)]) for i in range(p))
        close(total, float(expected), 'C3')
        close(total, a_optimality(M), 'C3')
        close(c_optimality(M, t(c, dtype)), float(score(matrix, c)), 'C3')


def test_C4_rotation():
    M, c = t([[3, 1], [1, 2]]), t([1, -2])
    Q = t([[F(3, 5), -F(4, 5)], [F(4, 5), F(3, 5)]])
    close(c_optimality(Q@M@Q.mT, Q@c), float(score([[3, 1], [1, 2]], [1, -2])), 'C4')


def test_C5_homogeneity():
    for matrix, contrast, _ in cases():
        M, c = t(matrix), t(contrast)
        expected = float(score(matrix, contrast))
        for alpha in (-3., .125, 4.):
            close(c_optimality(M, alpha*c), alpha**2*expected, 'C5')
        close(c_optimality(4*M, c), expected/4, 'C5')
    close(c_optimality(t([[1.]]), t([1e-100])), -1e-200, 'C5', tolerance=1e-210)


def test_C6_rank_one():
    for matrix, contrast, f in cases():
        p = len(contrast)
        updated = [[matrix[i][j]+f[i]*f[j] for j in range(p)] for i in range(p)]
        z, u = solve(matrix, contrast), solve(matrix, f)
        expected = sum(x*y for x, y in zip(contrast, u))**2 / (1+sum(x*y for x, y in zip(f, u)))
        assert score(updated, contrast)-score(matrix, contrast) == expected >= 0
        close(c_optimality(t(updated), t(contrast))-c_optimality(t(matrix), t(contrast)), float(expected), 'C6')


def test_C7_gradient():
    q = t([[1.2, .1], [.2, 1.8]]).requires_grad_()
    c = t([1.1, -.3]).requires_grad_()
    eye = torch.eye(2, dtype=q.dtype, device=DEVICE)
    assert torch.autograd.gradcheck(lambda x, y: c_optimality(x.mT@x+eye, y),
                                   (q, c), eps=1e-6, atol=1e-5, rtol=1e-3)
    M = t([[3, 1], [1, 2]]).requires_grad_()
    cc = t([1, -2]).requires_grad_()
    dm, dc = torch.autograd.grad(c_optimality(M, cc), (M, cc))
    z = t(solve([[3, 1], [1, 2]], [1, -2]))
    close(dm, z[:, None]*z[None, :], 'C7')
    close(dc, -2*z, 'C7')


@pytest.mark.parametrize('dtype', DTYPES)
def test_C8_batch(dtype):
    fixtures = list(cases())[3::4]
    M = torch.stack([t(m, dtype) for m, _, _ in fixtures])
    c = torch.stack([t(c, dtype) for _, c, _ in fixtures])
    actual = c_optimality(M, c)
    close(actual, [float(score(m, c)) for m, c, _ in fixtures], 'C8')
    close(actual, torch.stack([c_optimality(m, cc) for m, cc in zip(M, c)]), 'C8')


def test_C9_negatives():
    M, c = t([[2, 0], [0, 3]]), t([1, 2])
    bad = [([], c), (M, []), (M, t([1])), (M, t([[1], [2]])),
           (M.expand(2, 2, 2), c), (M.expand(2, 2, 2), c[None]),
           (M, c.float()), (M, c.long()), (M, c.bool()),
           (M, t([float('nan'), 1])), (M, t([float('inf'), 1])),
           (M, t([0, 0])), (t([[1, 1], [0, 1]]), c),
           (t([[1, 0], [0, 0]]), c), (t([[1, 0], [0, -1]]), c),
           (t([[1, 0], [0, 1e-20]]), c), (M, t([1e308, 1])),
           (t([[float('nan'), 0], [0, 1]]), c),
           (torch.empty(0, 2, 2, dtype=M.dtype, device=DEVICE), c.expand(0, 2)),
           (torch.ones(2, 3, dtype=M.dtype, device=DEVICE), c)]
    for matrix, contrast in bad:
        with pytest.raises((TypeError, ValueError)):
            c_optimality(matrix, contrast)
    with pytest.raises(ValueError, match='batch index 1'):
        c_optimality(M.expand(3, 2, 2), t([[1, 2], [0, 0], [0, 0]]))
    for budget in (3, 0, True):
        with pytest.raises(ValueError):
            c_optimality(M, c, max_elements=budget)
    # Metadata-only mismatch requires no GPU allocation on the CPU runner.
    with pytest.raises(ValueError, match='same device'):
        c_optimality(M, torch.ones(2, dtype=M.dtype, device='meta'))
    corners = ff2n(2, device=DEVICE)
    features = polynomial_features(corners, degree=2)
    with pytest.raises(ValueError):
        c_optimality(features.mT@features, torch.ones(6, dtype=M.dtype, device=DEVICE))


def test_budget_before_factorization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('factorized before rejecting shape budget')
    monkeypatch.setattr(torch.linalg, 'cholesky_ex', forbidden)
    with pytest.raises(ValueError, match='budget'):
        c_optimality(t([[2, 0], [0, 3]]), t([1, 2]), max_elements=3)
