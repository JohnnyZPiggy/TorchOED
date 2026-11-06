import pytest
import torch
from torchoed import d_optimality, exact_information
from torchoed.k2_exchange import _verified_information


def test_trusted_verification_matches_public_score():
    F=torch.randn(12,3,dtype=torch.float64,generator=torch.Generator().manual_seed(114))
    ids=torch.arange(5)
    M,value=_verified_information(F,ids)
    assert torch.equal(M,exact_information(F,ids))
    assert torch.equal(torch.tensor(value,dtype=F.dtype),d_optimality(M))
    F[:,2]=F[:,0]+F[:,1]
    with pytest.raises(ValueError):
        d_optimality(exact_information(F,ids))
    assert _verified_information(F,ids) is None
