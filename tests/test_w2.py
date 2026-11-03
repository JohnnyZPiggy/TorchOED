from itertools import combinations
import math
import pytest
import torch
from torchoed import effects_coding, feasible_candidates, full_factorial, d_optimal_design


def test_effects_coding():
    torch.testing.assert_close(effects_coding(torch.tensor([0,1,2]),3),torch.tensor([[1.,0.],[0.,1.],[-1.,-1.]],dtype=torch.float64),rtol=0,atol=0)
    with pytest.raises(ValueError):
        effects_coding(torch.tensor([3]),3)


def test_constrained_exhaustive_oracle():
    I = full_factorial((3,3,3))
    C = torch.cat((I[:,:2]-1,I[:,2:]),dim=1).double()
    mask = feasible_candidates(C,A=torch.tensor([[1.,1.,0.]],dtype=C.dtype),b=torch.tensor([1.],dtype=C.dtype),disallowed=({0:1,2:2},))
    assert mask.sum()==22 and (~mask).nonzero().flatten().tolist()==[20,23,24,25,26]
    F = torch.cat((torch.ones(27,1,dtype=C.dtype),C[:,:2],effects_coding(I[:,2],3),(C[:,0]*C[:,1])[:,None]),dim=1)[mask]
    maximum = 0
    # Enumerate once without torch.combinations' Cartesian intermediate; score bounded batches.
    for ids in torch.tensor(list(combinations(range(len(F)),7))).split(512):
        X = F[ids]
        maximum = max(maximum,torch.linalg.det(X.mT@X).round().max().item())
    assert maximum==5760
    result = d_optimal_design(F,7,restarts=10,generator=torch.Generator().manual_seed(0))
    chosen = C[mask][result['ids']]
    assert bool(((chosen[:,0]+chosen[:,1])<=1).all()) and not bool(((chosen[:,0]==1)&(chosen[:,2]==2)).any())
    assert result['objective']<=math.log(maximum)+1e-12
    assert result['d_efficiency_lower_bound']<=math.exp((result['objective']-math.log(maximum))/6)+1e-12
    print('hits_5760:',round(math.exp(result['objective']))==5760)
