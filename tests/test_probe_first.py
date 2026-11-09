import importlib
from itertools import islice
import torch
s = importlib.import_module('torchoed.d_optimal_design')


def test_probe_multitile_winner_and_resume(monkeypatch):
    F = torch.randn(14,3,dtype=torch.float64,generator=torch.Generator().manual_seed(145));ids = torch.arange(6)
    scores,screen = s._linear_pair_scores,s._linear_screen
    for budget in (3000,4000):
        for criterion in ('A','I'):
            seen = []
            def record(*args):
                seen.append(args[4]);return screen(*args)
            with monkeypatch.context() as patch:
                patch.setattr(s,'_linear_screen',record)
                actual = list(islice(s._linear_pair_moves(F,ids,criterion,budget,None,1e-12),2))
            def full(*args,**kwargs): return scores(*args,**dict(kwargs,tolerance=None))
            with monkeypatch.context() as patch:
                patch.setattr(s,'_linear_pair_scores',full)
                expected = list(islice(s._linear_pair_moves(F,ids,criterion,budget,None,1e-12),2))
            assert sum(v is None for v in seen)>1 and sum(v is not None for v in seen)>1
            assert len(actual)==2 and torch.equal(torch.tensor(actual),torch.tensor(expected))
