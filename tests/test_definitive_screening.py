import pytest
import torch

from torchoed import definitive_screening

FACTORS = (3, 4, 5, 6, 7, 8, 11, 12, 14, 17, 18, 19, 20, 23, 24)


def test_conference_foldover():
    for k in FACTORS:
        c = 2 * ((k + 1) // 2)
        d = definitive_screening(k).to(torch.int64)
        assert d.shape == (2 * c + 1, k)
        assert torch.equal(d[c:2*c], -d[:c])
        assert (d[-1] == 0).all() and (d.abs() <= 1).all()
        assert torch.equal(d[:c].T @ d[:c], (c - 1) * torch.eye(k, dtype=torch.int64))
        assert ((d == 0).sum(dim=0) == 3).all()


def test_second_order_orthogonality():
    for k in FACTORS:
        d = definitive_screening(k).to(torch.int64)
        b, e = torch.triu_indices(k, k, offset=1)
        gram = d.T @ torch.cat((d, d.square(), d[:, b] * d[:, e]), dim=1)
        assert torch.equal(gram[:, :k], (d.shape[0] - 3) * torch.eye(k, dtype=torch.int64))
        assert (gram[:, k:] == 0).all()


def test_unsupported_factor_counts():
    for k in (2, 4.0):
        with pytest.raises(ValueError):
            definitive_screening(k)

SUPPORTED = (*range(3, 21), *range(23, 31))


@pytest.mark.parametrize("k", [k for k in SUPPORTED if k % 2 == 0])
def test_extended_conference(k):
    conference = definitive_screening(k)[:k].to(torch.int64)
    eye = torch.eye(k, dtype=torch.int64)
    assert torch.equal(conference.T @ conference, (k - 1) * eye)
    assert torch.equal(conference.abs(), 1 - eye)
    assert torch.equal(conference.T, conference if k % 4 == 2 else -conference)


@pytest.mark.parametrize("k", SUPPORTED)
def test_extended_design(k):
    c = k + k % 2
    d = definitive_screening(k).to(torch.int64)
    assert d.shape == (2 * c + 1, k)
    assert ((d == 0).sum(dim=0) == 3).all()
    assert torch.equal(d.T @ d, 2 * (c - 1) * torch.eye(k, dtype=torch.int64))
    a, b = torch.triu_indices(k, k, offset=1)
    assert not (d.T @ torch.cat((d.square(), d[:, a] * d[:, b]), dim=1)).any()


@pytest.mark.parametrize("k", (*range(3, 9), *range(11, 15), *range(17, 21), 23, 24))
def test_unchanged_prime_paley_design(k):
    q = k + k % 2 - 1
    squares = {i * i % q for i in range(1, q)}
    chi = [0] + [1 if i in squares else -1 for i in range(1, q)]
    rows = [[0] + [1] * (k - 1)] + [
        [chi[-1]] + [chi[(j - i) % q] for j in range(k - 1)] for i in range(q)
    ]
    expected = torch.tensor(rows + [[-x for x in row] for row in rows] + [[0] * k],
                            dtype=torch.float64)
    assert torch.equal(definitive_screening(k), expected)


@pytest.mark.parametrize("k", (1, 2, 21, 22, 31))
def test_outside_covered_orders(k):
    with pytest.raises(ValueError, match="order-22" if k in (21, 22) else None):
        definitive_screening(k)
