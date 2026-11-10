import torch
from torchoed import next_batch, optimal_design, polynomial_features


def test_exact_equality_no_repeat_determinism():
    F = polynomial_features(2*torch.rand(30, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(0))-1, 2)
    run_ids = [0, 3, 7, 11, 19, 25]
    for criterion in ("D", "A", "I"):
        ids = next_batch(F, run_ids, 3, criterion=criterion, generator=torch.Generator().manual_seed(0))
        full = optimal_design(F, 9, criterion=criterion, fixed_ids=run_ids, generator=torch.Generator().manual_seed(0))["ids"]
        expected = full[~torch.isin(full, torch.tensor(run_ids))].sort().values
        assert torch.equal(ids, expected)
        assert len(ids) == 3 and not torch.isin(ids, torch.tensor(run_ids)).any()
        assert ids.dtype == torch.int64
        assert torch.equal(ids, next_batch(F, run_ids, 3, criterion=criterion, generator=torch.Generator().manual_seed(0)))


def test_leverage_oracle():
    F = polynomial_features(torch.linspace(-1, 1, 21, dtype=torch.float64)[:, None], 2)
    run_ids = torch.tensor([0, 5, 10])
    M = F[run_ids].mT @ F[run_ids]
    leverage = (F * torch.linalg.solve(M, F.mT).mT).sum(dim=1)
    leverage[run_ids] = -torch.inf
    ids = next_batch(F, run_ids, 1, criterion="D", generator=torch.Generator().manual_seed(0))
    assert ids.tolist() == [20] == [leverage.argmax().item()]
