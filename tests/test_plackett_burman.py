import pytest
import torch

from torchoed import design_diagnostics, plackett_burman


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


def test_plackett_burman_12_partial_aliasing():
    x = plackett_burman(12).to(torch.int64)
    a, b = torch.triu_indices(11, 11, offset=1)
    f1 = torch.cat((torch.ones((12, 1), dtype=x.dtype), x), dim=1)
    f2 = x[:, a] * x[:, b]
    cross = f1.T @ f2
    j = torch.arange(11)[:, None]
    expected_abs = 4 * ((j != a) & (j != b)).to(x.dtype)

    assert torch.count_nonzero(cross[0]).item() == 0
    torch.testing.assert_close(cross[1:].abs(), expected_abs)
    result = design_diagnostics(f1.double(), torch.arange(12), alias=f2.double())
    torch.testing.assert_close(result["alias"], cross.double() / 12, atol=1e-12, rtol=0)
