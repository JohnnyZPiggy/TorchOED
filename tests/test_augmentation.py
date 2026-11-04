from fractions import Fraction
from itertools import combinations
import importlib
import math
import pytest
import torch
from torchoed import full_factorial, polynomial_features, d_optimal_design, optimal_design


def fixture():
    return polynomial_features(full_factorial((4,4)).double()*2/3-1,2)


def test_augmented_exhaustive_oracle():
    F = fixture()
    ids = torch.tensor([[0,5,10,*c] for c in combinations([i for i in range(16) if i not in (0,5,10)],6)])
    X = F[ids]
    M = X.mT@X
    det = torch.linalg.det(M)
    maximum = det.max().item()
    minimum = torch.linalg.inv(M[det>1e-8]).diagonal(dim1=-2,dim2=-1).sum(-1).min().item()
    assert math.isclose(maximum,float(Fraction(14227079168,4782969)),rel_tol=1e-12)
    assert math.isclose(minimum,float(Fraction(1086297,434176)),rel_tol=1e-12)
    for criterion in ('D','A'):
        call = d_optimal_design if criterion=='D' else optimal_design
        kwargs = {} if criterion=='D' else dict(criterion='A')
        result = call(F,9,fixed_ids=[0,5,10],restarts=10,generator=torch.Generator().manual_seed(0),**kwargs)
        assert {0,5,10} <= set(result['ids'].tolist()) and len(result['ids'].unique())==len(result['ids'])==9
        optimum = math.log(maximum) if criterion=='D' else -minimum
        assert result['objective']<=optimum+1e-12
        achieved = math.exp((result['objective']-optimum)/6) if criterion=='D' else minimum/-result['objective']
        floor = result['d_efficiency_lower_bound'] if criterion=='D' else result['efficiency_lower_bound']
        assert 0<floor<=achieved+1e-12
        print(criterion,'hits augmented optimum:',math.isclose(result['objective'],optimum,abs_tol=1e-12))


def test_fixed_positions_and_invalid_ids(monkeypatch):
    module = importlib.import_module('torchoed.d_optimal_design')
    exchange = module._linear_exchange
    finals = []
    def capture(*args,**kwargs):
        result = exchange(*args,**kwargs)
        assert all(not ({0,5,10}&set(move['outgoing_ids'])) for move in result['swap_history'] if move['k']==2)
        finals.append(result['ids'].tolist())
        return result
    monkeypatch.setattr(module,'_linear_exchange',capture)
    result = optimal_design(fixture(),9,criterion='I',fixed_ids=[0,5,10],generator=torch.Generator().manual_seed(0))
    assert len(finals)==10 and all({0,5,10}<=set(ids) for ids in finals+result['starts'])
    for fixed in ([0,0],[16],[-1],list(range(9))):
        with pytest.raises(ValueError):
            optimal_design(fixture(),9,criterion='I',fixed_ids=fixed)
