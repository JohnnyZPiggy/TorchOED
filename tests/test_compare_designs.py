from fractions import Fraction

import pytest
import torch

from torchoed import compare_designs, design_diagnostics, full_factorial, optimal_design


def test_comparison_fraction_oracle():
    x = full_factorial([3, 3]).double() - 1
    F = torch.cat((torch.ones(9, 1), x), 1)
    ids = [torch.tensor([0, 2, 6, 8]), torch.tensor([0, 2, 4, 6, 8])]
    result = compare_designs(F, ids)
    expected = (float(Fraction(5, 4)) ** (1 / 3), float(Fraction(15, 14)),
                float(Fraction(35, 32)), float(Fraction(15, 14)))
    for name, value in zip(("D", "A", "I", "G"), expected):
        torch.testing.assert_close(result["efficiency"][name], F.new_tensor([1, value]), atol=1e-14, rtol=0)
    torch.testing.assert_close(result["fds_quantiles"], F.new_tensor([
        [.25, .5, .5, .75, .75], [.2, .45, .45, .7, .7]]), atol=1e-14, rtol=0)
    assert torch.equal(result["power"], torch.stack([
        design_diagnostics(F, d, effect=1.0)["power"] for d in ids]))


def test_run_sizes_and_required_errors():
    x = full_factorial([3, 3]).double() - 1
    F = torch.cat((torch.ones(9, 1), x, x.prod(1, keepdim=True)), 1)
    result = compare_designs(F, [5, 7, 9], generator=torch.Generator().manual_seed(0))
    generator = torch.Generator().manual_seed(0)
    assert result["n"] == [5, 7, 9]
    for n, ids in zip(result["n"], result["ids"]):
        assert torch.equal(ids, optimal_design(F, n, generator=generator)["ids"])
    for designs in ([5], [True, 7], [torch.arange(4), torch.arange(5)]):
        with pytest.raises(ValueError):
            compare_designs(F, designs)


def test_repeated_events_and_zero_region():
    F = torch.tensor([[1., -1.], [1., 0.], [1., 1.]], dtype=torch.float64)
    ids = torch.tensor([0, 0, 1, 2])
    result = compare_designs(F, [ids, ids.repeat(2)])
    for values in result["efficiency"].values():
        torch.testing.assert_close(values, F.new_tensor([1, 2]))
    with pytest.raises(ValueError):
        compare_designs(F, [ids, ids], region=torch.zeros_like(F))
