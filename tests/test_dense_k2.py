from itertools import combinations
import torch
from torchoed.k2_exchange import k2_exchange, _dense_two_ratios


def test_dense_streamed_equivalence():
    F=torch.randint(-3,4,(12,3),generator=torch.Generator().manual_seed(114)).double()
    ids=torch.arange(5)
    for free in (None,torch.tensor([True,False,True,False,True])):
        dense=k2_exchange(F,ids,_free_positions=free)
        stream=k2_exchange(F,ids,max_elements=6000,_free_positions=free)
        assert torch.equal(dense['ids'],stream['ids'])
        for key in ('objective_history','swap_history','stop_reason','certificate','efficiency_gap_bound'):
            assert dense[key]==stream[key],key


def test_dense_ratios_direct_determinants():
    F=torch.randint(-3,4,(8,3),generator=torch.Generator().manual_seed(114)).double()
    ids=torch.arange(4); incoming=torch.arange(4,8); M=F[ids].mT@F[ids]
    Z=torch.linalg.solve_triangular(torch.linalg.cholesky(M),F.mT,upper=False)
    ratios,rows=_dense_two_ratios(Z,ids,incoming,Z[:,ids].mT@Z[:,ids],Z[:,ids].mT@Z[:,incoming])
    assert len(rows)==36
    for ratio,row in zip(ratios,rows):
        trial=ids.clone();trial[row[:2]]=row[2:];X=F[trial]
        expected=torch.linalg.det(X.mT@X)/torch.linalg.det(M)
        torch.testing.assert_close(ratio,expected,rtol=1e-10,atol=1e-12)
