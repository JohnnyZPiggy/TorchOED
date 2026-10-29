"""M2-A gate numbering follows M2_IMPLEMENTATION_GATE (not older contract IDs)."""
import itertools
import json
import math
from pathlib import Path

import pytest
import torch

from torchoed import (exact_information, weighted_information, d_optimality,
                      a_optimality, e_optimality, polynomial_features)
from test_m1 import exact_rank_det, scalar_features


def t(rows, dtype=torch.float64):
    return torch.tensor(rows, dtype=dtype)


def ids(values):
    return torch.tensor(values, dtype=torch.int64)


F = t([[1, -1], [1, 0], [1, 1]])
SCORES = (d_optimality, a_optimality, e_optimality)


def close(actual, expected, tol=1e-9):
    torch.testing.assert_close(actual, t(expected).to(actual.dtype), atol=tol, rtol=tol)


def test_A1_exact_normalized():
    assert torch.equal(exact_information(F, ids([0, 2])), t([[2, 0], [0, 2]]))
    assert torch.equal(exact_information(F, ids([0, 2]), normalized=True), torch.eye(2, dtype=F.dtype))


def test_A2_id_identity():
    with pytest.raises(ValueError, match='duplicate ID'):
        exact_information(F, ids([0, 0]))
    assert torch.equal(exact_information(F, ids([0, 0]), allow_repeats=True), t([[2, -2], [-2, 2]]))
    coordinates_equal = t([[1, 2], [1, 2]])
    assert torch.equal(exact_information(coordinates_equal, ids([0, 1])), t([[2, 4], [4, 8]]))


def test_A3_probability_weights():
    assert torch.equal(weighted_information(F, t([.5, 0, .5])), t([[1, 0], [0, 1]]))


def test_A4_diagonal():
    M = t([[2, 0], [0, 8]])
    for score, expected in zip(SCORES, (math.log(16), -.625, 2)):
        close(score(M), expected)


def test_A5_singular_psd():
    M = t([[1, 0], [0, 0]])
    for score in SCORES[:2]:
        with pytest.raises(ValueError, match='Cholesky'):
            score(M)
    assert e_optimality(M).item() == 0


def test_A6_indefinite():
    for score in SCORES:
        with pytest.raises(ValueError):
            score(t([[1, 0], [0, -1]]))


def test_A7_correlated():
    for score, expected in zip(SCORES, (math.log(3), -4/3, 1)):
        close(score(t([[2, 1], [1, 2]])), expected)


def test_A8_independent_event_scaling():
    assert torch.equal(exact_information(F, ids([0, 2, 0, 2]), allow_repeats=True), t([[4, 0], [0, 4]]))
    assert torch.equal(exact_information(F, ids([0, 2, 0, 2]), allow_repeats=True, normalized=True), t([[1, 0], [0, 1]]))


def test_A9_invalid_and_preallocation(monkeypatch):
    for w in ([-.1, .5, .6], [float('nan'), 0, 1], [float('inf'), 0, 0], [1, 1, 1]):
        with pytest.raises(ValueError):
            weighted_information(F, t(w))
    for bad_ids in (ids([-1]), ids([3]), torch.tensor([True]), torch.tensor([0], dtype=torch.int32), t([0])):
        with pytest.raises((TypeError, ValueError)):
            exact_information(F, bad_ids)
    with pytest.raises(ValueError):
        weighted_information(F, t([.5, .5, 0], dtype=torch.float32))
    for score in SCORES:
        for M in (t([[1, 1], [0, 1]]), t([[float('nan')]]), t([[float('inf')]]), torch.ones(2, 3)):
            with pytest.raises(ValueError):
                score(M)
    def forbidden(*args, **kwargs):
        pytest.fail('numerical computation attempted after invalid size')
    monkeypatch.setattr(torch, 'gather', forbidden)
    monkeypatch.setattr(torch.linalg, 'cholesky_ex', forbidden)
    with pytest.raises(ValueError, match='budget'):
        exact_information(F, ids([0, 2]), max_elements=3)
    with pytest.raises(ValueError, match='budget'):
        d_optimality(t([[2, 0], [0, 8]]), max_elements=3)


def test_A10_matched_batch_and_error_index():
    f = torch.stack([F, 2*F])
    selected = ids([[0, 2], [0, 1]])
    raw = t([[[2, 0], [0, 2]], [[8, -4], [-4, 4]]])
    assert torch.equal(exact_information(f, selected), raw)
    assert torch.equal(exact_information(f, selected, normalized=True), raw/2)
    w = t([[.5, 0, .5], [.25, .5, .25]])
    assert torch.equal(weighted_information(f, w), t([[[1, 0], [0, 1]], [[4, 0], [0, 2]]]))
    M = t([[[2, 0], [0, 8]], [[2, 1], [1, 2]]])
    for score, expected in zip(SCORES, ([math.log(16), math.log(3)], [-.625, -4/3], [2, 1])):
        close(score(M), expected)
        close(score(M), torch.stack([score(x) for x in M]).tolist())
    for bad in (ids([0, 2]), ids([[0, 2]])):
        with pytest.raises(ValueError):
            exact_information(f, bad)
    with pytest.raises(ValueError):
        weighted_information(f, t([.5, 0, .5]))
    for call in (lambda: exact_information(f, ids([[0, 2], [0, 0]])),
                 lambda: exact_information(f, ids([[0, 2], [0, 3]])),
                 lambda: weighted_information(f, t([[.5, 0, .5], [0, 0, 0]])),
                 lambda: exact_information(torch.stack([F, F*float('nan')]), selected)):
        with pytest.raises(ValueError, match='batch index 1'):
            call()
    for score in SCORES:
        with pytest.raises(ValueError, match='batch index 1'):
            score(t([[[2, 0], [0, 8]], [[1, 0], [0, -1]]]))


def test_A11_float64_gradcheck():
    q = t([[1.2, .1], [.2, 1.8]]).requires_grad_()
    for score in SCORES:
        # Perturb free factors, keeping the matrix symmetric and SPD.
        assert torch.autograd.gradcheck(lambda x: score(x.mT@x + torch.eye(2, dtype=x.dtype)),
                                       (q,), eps=1e-6, atol=1e-5, rtol=1e-3)
    f = F.clone().requires_grad_()
    assert torch.autograd.gradcheck(lambda x: exact_information(x, ids([0, 2])), (f,))
    logits = t([.1, .2, .3]).requires_grad_()
    assert torch.autograd.gradcheck(lambda x, y: weighted_information(x, y.softmax(0)), (f, logits))


def test_A12_actual_e0_vs_separate_negative():
    fixture = json.loads((Path(__file__).parent/'fixtures/e0_p1.json').read_text())
    assert fixture['source_sha256'] == '34ef4a2784ef6f5c528032377d6c0830f1a3bb986fbb21404a724a48a99a458f'
    positive = fixture['C']
    negative = [[a,b,c,a*b*c] for a,b,c in itertools.product((-1,1), repeat=3)] + positive[8:]
    for rows, rank, det in ((positive,15,18014398509481984),(negative,12,0)):
        features = scalar_features(rows)
        assert exact_rank_det(features) == (rank,det)
        f = polynomial_features(t(rows))
        assert torch.equal(f, t(features))
        M = exact_information(f, torch.arange(20))
        independent_gram = t([[sum(r[i]*r[j] for r in features) for j in range(15)] for i in range(15)])
        assert torch.equal(M, independent_gram)
        assert torch.linalg.matrix_rank(M).item() == rank
        if det:
            close(d_optimality(M), math.log(det))
            assert torch.linalg.cholesky_ex(M).info.item() == 0
        else:
            for score in SCORES[:2]:
                with pytest.raises(ValueError):
                    score(M)


def test_empty_flags_dtype_and_overflow():
    assert torch.equal(exact_information(torch.empty(0,2,dtype=torch.float64), ids([])), t([[0,0],[0,0]]))
    with pytest.raises(ValueError):
        exact_information(F, ids([]), normalized=True)
    for call in (lambda: exact_information(F, ids([0]), normalized=1),
                 lambda: weighted_information(torch.empty(0,2,dtype=torch.float64), t([])),
                 lambda: d_optimality(torch.empty(0,2,2,dtype=torch.float64)),
                 lambda: exact_information(torch.ones(3,2,dtype=torch.int64), ids([0]))):
        with pytest.raises((ValueError,TypeError)):
            call()
    with pytest.raises(ValueError, match='overflow'):
        exact_information(t([[1e308]]), ids([0]))


def test_float32_separate_tolerance():
    for dtype,tolerance in ((torch.float64,1e-9),(torch.float32,1e-5)):
        f = F.to(dtype)
        M = exact_information(f,ids([0,2]))
        assert M.dtype == dtype and M.device.type == 'cpu'
        for score,expected in zip(SCORES,(math.log(4),-1,2)):
            value=score(M)
            assert value.dtype==dtype and value.device==M.device
            close(value,expected,tolerance)


def test_symmetric_psd_roundoff_policy():
    for dtype in (torch.float64,torch.float32):
        eps=torch.finfo(dtype).eps
        assert e_optimality(t([[1,0],[0,-eps]],dtype)).item()==0
        with pytest.raises(ValueError):
            e_optimality(t([[1,0],[0,-1024*eps]],dtype))
        M=t([[2,1+eps],[1,2]],dtype)
        for score,expected in zip(SCORES,(math.log(3),-4/3,1)):
            close(score(M),expected,1e-5 if dtype==torch.float32 else 1e-9)
