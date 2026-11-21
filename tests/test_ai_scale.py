import importlib

import pytest
import torch

from torchoed import optimal_design, polynomial_features

s = importlib.import_module('torchoed.d_optimal_design')


@pytest.mark.parametrize('criterion,expected', [
    ('A', [-3.22418068136689, -3.246316796354173, -3.2651264914074245]),
    ('I', [-0.3799428292524771, -0.37710157549003204, -0.37894625829255574]),
])
def test_prechange_objectives(criterion, expected):
    F = polynomial_features(2*torch.rand(200, 5, dtype=torch.float64,
        generator=torch.Generator().manual_seed(0))-1, degree=2)
    for seed, value in enumerate(expected):
        result = optimal_design(F, 42, criterion=criterion, restarts=1,
            generator=torch.Generator().manual_seed(seed))
        assert result['objective'] == pytest.approx(value, rel=1e-12)


@pytest.mark.parametrize('criterion', ['A', 'I'])
def test_low_workspace_winner_matches_full_scan(criterion, monkeypatch):
    F = torch.randn(80, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(527))
    ids = torch.arange(6)
    original = s._linear_pair_scores
    removal, calls = s._linear_removal_tiles, []
    def record(*args, **kwargs):
        calls.append(True)
        yield from removal(*args, **kwargs)
    monkeypatch.setattr(s, '_linear_removal_tiles', record)
    actual = next(s._linear_pair_moves(F, ids, criterion, 20_000, None, 1e-12))
    assert calls
    def full(*args, **kwargs):
        kwargs['tolerance'] = None
        return original(*args, **kwargs)
    monkeypatch.setattr(s, '_linear_pair_scores', full)
    expected = next(s._linear_pair_moves(F, ids, criterion, 20_000, None, 1e-12))
    assert actual == expected
    # Independent dense information solves check the selected two-swap gain.
    trial = ids.clone()
    trial[actual[0]] = trial.new_tensor(actual[1])
    B = torch.eye(3, dtype=F.dtype) if criterion == 'A' else F.mT@F/len(F)
    phi = lambda rows: torch.trace(torch.linalg.solve(F[rows].mT@F[rows], B))
    assert phi(trial) < phi(ids)
