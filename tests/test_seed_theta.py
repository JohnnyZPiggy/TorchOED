import importlib
import torch
s = importlib.import_module('torchoed.d_optimal_design')

def test_seed_preserves_winner_resume_and_lifts_first_theta(monkeypatch):
    F = torch.randn(16,3,dtype=torch.float64,generator=torch.Generator().manual_seed(0))
    F[:3] = torch.linalg.qr(F[:3]).Q
    scores = s._linear_pair_scores
    for n in (3,6):
        for criterion in ('A','I'):
            seen,first = [],[]
            def scan(*args,**kwargs):
                first.clear();return scores(*args,**kwargs)
            def limit(original):
                def record(*args):
                    if not first: seen.append(args[3]);first.append(True)
                    return original(*args)
                return record
            with monkeypatch.context() as patch:
                patch.setattr(s,'_linear_pair_scores',scan)
                for name in ('_linear_removal_limit','_linear_square_limit'): patch.setattr(s,name,limit(getattr(s,name)))
                actual = list(s._linear_pair_moves(F,torch.arange(n),criterion,1_000_000,None,1e-12))
            def full(*args,**kwargs):
                kwargs['tolerance'] = None
                return scores(*args,**kwargs)
            with monkeypatch.context() as patch:
                patch.setattr(s,'_linear_pair_scores',full)
                expected = list(s._linear_pair_moves(F,torch.arange(n),criterion,1_000_000,None,1e-12))
            assert max(seen)>0 and len(actual)>1 and torch.equal(torch.tensor(actual),torch.tensor(expected))
