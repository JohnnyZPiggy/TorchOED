import pytest
import torch

from torchoed import plackett_burman


@pytest.mark.parametrize("n", range(4, 49, 4))
def test_plackett_burman_is_balanced_and_orthogonal(n):
    design = plackett_burman(n)
    assert design.dtype == torch.float64
    assert design.device.type == "cpu"
    assert design.shape == (n, n - 1)
    assert ((design == -1) | (design == 1)).all()
    x = design.to(torch.int64)
    assert torch.equal(x.T @ x, n * torch.eye(n - 1, dtype=x.dtype))
    assert torch.equal(x.sum(dim=0), torch.zeros(n - 1, dtype=x.dtype))
    if n == 36:
        assert x[-1].tolist() == [-1] * 17 + [1] + [-1] * 17


@pytest.mark.parametrize("q", (3, 7, 11, 19, 23, 31, 43, 47))
def test_plackett_burman_cyclic_form(q):
    squares = {i * i % q for i in range(1, q)}
    generator = [1] + [1 if j in squares else -1 for j in range(1, q)]
    expected = [[generator[(j - i) % q] for j in range(q)] for i in range(q)]
    expected.append([-1] * q)
    assert torch.equal(plackett_burman(q + 1), torch.tensor(expected, dtype=torch.float64))


@pytest.mark.parametrize("runs", (2, 6, 50, 52, 12.0))
def test_plackett_burman_rejects_unsupported_runs(runs):
    with pytest.raises(ValueError, match="runs"):
        plackett_burman(runs)
