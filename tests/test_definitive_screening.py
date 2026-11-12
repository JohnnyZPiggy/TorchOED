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
    for k in (2, 9, 10, 15, 16, 25, 4.0):
        with pytest.raises(ValueError):
            definitive_screening(k)
