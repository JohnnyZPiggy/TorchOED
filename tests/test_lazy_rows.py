import torch
from torchoed.k2_exchange import _one_candidates, _keep_top


def test_lazy_rows_match_explicit_tiles_and_masks():
    g=torch.Generator().manual_seed(118)
    Gss=torch.randn(5,5,dtype=torch.float64,generator=g).round()
    Gsn=torch.randn(5,7,dtype=torch.float64,generator=g).round()
    diagonal=torch.randn(12,dtype=torch.float64,generator=g).round()
    incoming=torch.arange(5,12); width=3; tolerance=1e-8
    for free in (None,torch.tensor([True,False,True,False,True])):
        for anchor in (None,4.):
            expected=[]
            for start in range(0,7,width):
                end=min(start+width,7)
                ratios=(1-Gss.diagonal()[:,None])*(1+diagonal[incoming[start:end]])+Gsn[:,start:end].square()
                positions=torch.arange(5).repeat_interleave(end-start)
                rows=torch.stack((positions,incoming[start:end].repeat(5)),dim=1)
                if free is not None:
                    keep=free[positions];ratios,rows=ratios.flatten()[keep],rows[keep]
                _keep_top(expected,ratios.flatten(),rows.__getitem__,tolerance,anchor)
            actual=_one_candidates(Gss,Gsn,diagonal,incoming,width,tolerance,anchor,free_positions=free)
            assert actual==expected
