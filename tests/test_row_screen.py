import importlib
import torch
s = importlib.import_module('torchoed.d_optimal_design')


def test_row_screen_dense_entries_and_resume(monkeypatch):
    F = torch.randn(16,3,dtype=torch.float64,generator=torch.Generator().manual_seed(0))
    F[:3] = 1.5*torch.linalg.qr(F[:3]).Q
    original,screen = s._linear_pair_scores,s._linear_screen
    for n,budget in ((3,3000),(5,3160)):
        ids = torch.arange(n);L = torch.linalg.cholesky(F[ids].mT@F[ids])
        Z = torch.linalg.solve_triangular(L,F.mT,upper=False)
        c,d = torch.triu_indices(len(F),len(F),offset=1)
        for criterion in ('A','I'):
            T = torch.linalg.solve_triangular(L.mT,Z,upper=True) if criterion=='A' else (Z@Z.mT/len(F))@Z
            G,Q = Z.mT@Z,T.mT@T if criterion=='A' else Z.mT@T
            g,q = s._linear_pair_entries(Z,T,c,d,criterion)
            assert (g-G[c,d]).abs().max()<=1e-12*G[c,d].abs().max()
            assert (q-Q[c,d]).abs().max()<=1e-12*Q[c,d].abs().max()
            seen = []
            def record(*args):
                assert args[0].shape==(n,len(F)) and args[6] is None and args[8] is not None
                seen.append(True);return screen(*args)
            with monkeypatch.context() as patch:
                patch.setattr(s,'_linear_screen',record)
                actual = list(s._linear_pair_moves(F,ids,criterion,budget,None,1e-12))
            def full(*args,**kwargs): return original(*args,**dict(kwargs,tolerance=None))
            with monkeypatch.context() as patch:
                patch.setattr(s,'_linear_pair_scores',full)
                expected = list(s._linear_pair_moves(F,ids,criterion,budget,None,1e-12))
            assert seen and len(actual)>1 and torch.equal(torch.tensor(actual),torch.tensor(expected))
