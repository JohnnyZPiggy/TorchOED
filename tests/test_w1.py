from fractions import Fraction
from itertools import combinations
import math
import torch
from torchoed import optimal_design, d_optimal_design
from torchoed.d_optimal_design import _linear_scores, _linear_pair_scores, _linear_exchange
from torchoed.exchange import _tolerance
from oracle_exchange import linear_value
from test_m3c import ROWS


def test_L1_exact_linear_swap():
    rows = [[1,0],[0,1],[2,0],[1,1]]
    F = torch.tensor(rows,dtype=torch.float64)
    _,phi,_,incoming,ratio,delta = _linear_scores(F,torch.tensor([0,1]),torch.eye(2,dtype=F.dtype),'A')
    assert incoming.tolist()==[2,3]
    assert ratio[0,0].item()==4 and phi==2 and delta[0,0].item()==-0.75
    assert linear_value(rows,[2,1],'A')==Fraction(5,4)


def test_L2_fraction_all_single_swaps_and_floor():
    # Frozen integer fixture: Python random.Random(87), randint(-2,2), row-major 7x3.
    rows = [[-1, -1, 2], [-2, 0, 0], [2, 0, -2], [0, -2, 0], [1, -2, 1], [2, 2, -2], [-1, 0, -1]]
    F = torch.tensor(rows,dtype=torch.float64)
    for criterion in ('A','I'):
        B = torch.eye(3,dtype=F.dtype) if criterion=='A' else F.mT@F/7
        values = {ids:linear_value(rows,ids,criterion) for ids in combinations(range(7),4)}
        checked = 0
        pair_checked = 0
        for ids,old in values.items():
            if old is None:
                continue
            _,phi,_,incoming,ratio,delta = _linear_scores(F,torch.tensor(ids),B,criterion)
            assert math.isclose(phi,float(old),rel_tol=1e-12)
            for i in range(4):
                for j,new_id in enumerate(incoming.tolist()):
                    trial = list(ids); trial[i] = new_id
                    new = linear_value(rows,trial,criterion)
                    if new is not None:
                        assert ratio[i,j]>0
                        assert math.isclose(delta[i,j].item(),float(new-old),rel_tol=1e-12,abs_tol=1e-12)
                        checked += 1
            for pairs,gains in _linear_pair_scores(F,torch.tensor(ids),criterion,1_000_000):
                for row,gain in zip(pairs.tolist(),gains.tolist()):
                    trial = list(ids); trial[row[0]],trial[row[1]] = row[2:]
                    new = linear_value(rows,trial,criterion)
                    if new is not None:
                        assert math.isclose(gain,float(old-new),rel_tol=1e-10,abs_tol=1e-12)
                        pair_checked += 1
        assert checked>0
        assert pair_checked>0
        result = optimal_design(F,4,criterion=criterion,generator=torch.Generator().manual_seed(0))
        direct = linear_value(rows,result['ids'].tolist(),criterion)
        assert math.isclose(result['objective'],-float(direct),rel_tol=1e-12)
        optimum = min(x for x in values.values() if x is not None)
        assert 0<result['efficiency_lower_bound']<=float(optimum/direct)+1e-12
        M = F[result['ids']].mT@F[result['ids']]
        condition = torch.linalg.cond(M).item()
        tolerance = _tolerance(float(direct),float(direct),condition,condition,3)
        for ids,value in values.items():
            if value is not None and len(set(ids)-set(result['ids'].tolist()))<=2:
                assert float(direct-value)<=tolerance
        assert result['certificate']=='LOCAL_2_NUMERICAL'
        assert result['stop_reason']=='LOCAL_2_NUMERICAL'
        print(criterion,'legal pair identities checked:',pair_checked)


def test_D_delegates_unchanged():
    F = torch.tensor(ROWS,dtype=torch.float64)
    args = dict(n=4,restarts=2)
    expected = d_optimal_design(F,generator=torch.Generator().manual_seed(0),**args)
    actual = optimal_design(F,criterion='D',generator=torch.Generator().manual_seed(0),**args)
    assert torch.equal(actual['ids'],expected['ids'])
    assert actual['objective']==expected['objective'] and actual['starts']==expected['starts']
    assert actual['efficiency_lower_bound']==expected['d_efficiency_lower_bound']


def test_L4_two_swap_trap():
    rows = [[2,-2],[2,0],[0,-1],[-1,-1],[1,1],[-1,2]]
    F = torch.tensor(rows,dtype=torch.float64); ids = torch.tensor([1,3,5]); B = torch.eye(2,dtype=F.dtype)
    old = linear_value(rows,ids.tolist(),'A')
    assert old==Fraction(11,29)
    values = {s:linear_value(rows,s,'A') for s in combinations(range(6),3)}
    assert all(v>=old for s,v in values.items() if v is not None and len(set(s)-set(ids.tolist()))<=1)
    _,phi,_,_,ratio,delta = _linear_scores(F,ids,B,'A')
    assert not bool(((ratio>0)&(delta < -1e-12)).any()) and ids.tolist()==[1,3,5]
    for free in (None,torch.tensor([True,False,True])):
        result = _linear_exchange(F,ids,B,'A',100,_free_positions=free)
        assert result['ids'].tolist()==[0,3,4] and math.isclose(result['objective'],-3/8,rel_tol=1e-12)
        assert len(result['swap_history'])==1 and result['swap_history'][0]['k']==2
        assert result['swap_history'][0]['outgoing_ids']==[1,5]
        assert result['certificate']=='LOCAL_2_NUMERICAL'
    assert min(v for v in values.values() if v is not None)==Fraction(3,8)
