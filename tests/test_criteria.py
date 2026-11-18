import pytest
import torch
from torchoed import g_optimality, v_optimality, i_optimality


def test_prediction_criteria_dense():
    F = torch.randn(9,4,dtype=torch.float64,generator=torch.Generator().manual_seed(11))
    X = F[:6]; M = X.T@X/6; W = F.T@F/9
    d = torch.einsum('ij,jk,ik->i',F,torch.linalg.inv(M),F)
    torch.testing.assert_close(g_optimality(M,F),-d.max(),rtol=1e-12,atol=0)
    torch.testing.assert_close(v_optimality(M,F),-d.mean(),rtol=1e-12,atol=0)
    torch.testing.assert_close(i_optimality(M,W),v_optimality(M,F),rtol=1e-12,atol=0)
    assert -g_optimality(M,F) >= 4-1e-12


def test_prediction_criteria_batch():
    F = torch.randn(2,9,4,dtype=torch.float64,generator=torch.Generator().manual_seed(12))
    M = F[:,:6].mT@F[:,:6]/6
    for fn, arg in ((g_optimality,F),(v_optimality,F),(i_optimality,F.mT@F/9)):
        torch.testing.assert_close(fn(M,arg),torch.stack([fn(m,a) for m,a in zip(M,arg)]),rtol=0,atol=0)


def test_prediction_criteria_invalid():
    for fn in (g_optimality,v_optimality,i_optimality):
        with pytest.raises(ValueError): fn(torch.ones(2,2),torch.ones(2,2))
    with pytest.raises(ValueError): i_optimality(torch.eye(2),torch.eye(3))
