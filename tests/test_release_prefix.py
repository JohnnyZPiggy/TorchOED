import importlib
import torch
s = importlib.import_module('torchoed.d_optimal_design')


def test_release_prefix_capacity_and_winner_resume(monkeypatch):
    F = torch.randn(14,3,dtype=torch.float64,generator=torch.Generator().manual_seed(145));ids = torch.arange(6)
    scores,tiles = s._linear_pair_scores,s._linear_removal_tiles
    for criterion in ('A','I'):
        widths = []
        def record(*args):
            widths.append(args[7]);assert args[0][1].numel()==1 and args[0][2] is None
            return tiles(*args)
        with monkeypatch.context() as patch:
            patch.setattr(s,'_linear_removal_tiles',record)
            actual = list(s._linear_pair_moves(F,ids,criterion,10000,None,1e-12))
        def full(*args,**kwargs): return scores(*args,**dict(kwargs,tolerance=None))
        with monkeypatch.context() as patch:
            patch.setattr(s,'_linear_pair_scores',full)
            expected = list(s._linear_pair_moves(F,ids,criterion,10000,None,1e-12))
        assert min(widths)>49  # Base: available=7958, block=15, tables=314, capacity=49.
        assert len(actual)>1 and torch.equal(torch.tensor(actual),torch.tensor(expected))
