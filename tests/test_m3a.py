import math

import pytest
import torch

from torchoed import fedorov_exchange
from torchoed.exchange import _exchange_ratios
from oracle_exchange import determinant, exhaustive, neighborhood


SIMPLE = [[1,0],[0,1],[2,0]]


def run(rows, selected, **kwargs):
    return fedorov_exchange(torch.tensor(rows,dtype=torch.float64),
                            torch.tensor(selected,dtype=torch.int64),**kwargs)


def ratios(rows, selected):
    F=torch.tensor(rows,dtype=torch.float64)
    X=F[selected]
    L=torch.linalg.cholesky(X.mT@X)
    incoming=[j for j in range(len(rows)) if j not in selected]
    R=torch.linalg.solve_triangular(L,X.mT,upper=False)
    T=torch.linalg.solve_triangular(L,F[incoming].mT,upper=False)
    return _exchange_ratios(R,T),incoming


def test_F1_gain_log_four():
    result=run(SIMPLE,[0,1],max_iter=1)
    assert result['ids'].tolist()==[2,1]
    assert result['initial_objective']==0
    assert math.isclose(result['final_objective'],math.log(4),abs_tol=1e-9)
    assert math.isclose(result['swap_history'][0]['gamma'],4,abs_tol=1e-9)
    assert determinant(SIMPLE,[2,1])==4


def test_F2_saturated_delete_singular():
    rows=[[1,-1],[1,1],[1,0]]
    assert determinant(rows,[1])==0
    gamma,_=ratios(rows,[0,1])
    torch.testing.assert_close(gamma,torch.full((2,1),.25,dtype=torch.float64),atol=1e-9,rtol=1e-9)
    assert run(rows,[0,1])['stop_reason']=='LOCAL_1_SWAP_NUMERICAL'


def test_F3_different_id_equal_features_legal():
    rows=[[1,0],[0,1],[1,0]]
    gamma,pool=ratios(rows,[0,1])
    assert pool==[2] and gamma[0,0].item()==1
    result=run(rows,[0,1])
    assert result['evaluated_candidates']==2
    assert result['scans'][0]['verified_spd']==1
    assert result['accepted_swaps']==0


def test_F4_repeats_rejected():
    with pytest.raises(ValueError,match='duplicate ID'):
        run(SIMPLE,[0,0])
    with pytest.raises(NotImplementedError):
        run(SIMPLE,[0,1],allow_repeats=True)


def test_F5_no_initializer():
    with pytest.raises(ValueError,match='initial design'):
        run([[1,0],[2,0],[3,0]],[0,1])
    with pytest.raises(ValueError,match='m>=n>=p'):
        run(SIMPLE,[0])


def test_F6_best_full_scan_and_stop():
    result=run(SIMPLE,[0,1])
    assert result['ids'].tolist()==[2,1] and result['accepted_swaps']==1
    assert result['stop_reason']=='LOCAL_1_SWAP_NUMERICAL'
    assert result['full_neighborhood_checked'] and result['certificate']=='NUMERICAL_ONLY'
    assert result['evaluated_candidates']==4
    assert max(d for _,_,d in neighborhood(SIMPLE,[2,1]))<=4


def test_F7_deterministic_ties_tiles_rng_order():
    before=torch.random.get_rng_state().clone()
    for rows,initial,expected in (([[1,0],[0,1],[2,0],[2,0]],[0,1],[2,1]),
                                  ([[1,0],[0,1],[2,2]],[0,1],[2,1]),
                                  ([[1,0],[0,1],[2,0],[2,0]],[1,0],[1,2])):
        outcomes=[run(rows,initial,max_iter=1,tile_size=width)['ids'].tolist() for width in (1,2,64)]
        assert outcomes==[expected]*3
    assert torch.equal(before,torch.random.get_rng_state())


def test_F8_all_legal_ratios_exact_reference():
    rows=[[1,0],[0,1],[1,1],[2,-1],[0,2]]
    initial=[0,1,2]
    gamma,pool=ratios(rows,initial)
    old=determinant(rows,initial)
    assert len(neighborhood(rows,initial))==6
    for i,j,det in neighborhood(rows,initial):
        assert math.isclose(gamma[i,pool.index(j)].item(),float(det/old),abs_tol=1e-9,rel_tol=1e-9)


def test_F9_saturated_no_inverse(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('dense inverse is forbidden')
    monkeypatch.setattr(torch.linalg,'inv',forbidden)
    assert run(SIMPLE,[0,1])['ids'].tolist()==[2,1]


def test_F10_iteration_and_no_legal_statuses():
    zero=run(SIMPLE,[0,1],max_iter=0)
    assert zero['stop_reason']=='MAX_ITER' and not zero['full_neighborhood_checked']
    assert zero['evaluated_candidates']==zero['accepted_swaps']==0
    one=run(SIMPLE,[0,1],max_iter=1)
    assert one['stop_reason']=='MAX_ITER' and not one['full_neighborhood_checked']
    assert one['accepted_swaps']==1
    all_selected=run(SIMPLE,[0,1,2])
    assert all_selected['stop_reason']=='NO_LEGAL_SWAP'
    assert all_selected['evaluated_candidates']==0
    assert all(b>a for a,b in zip(one['objective_history'],one['objective_history'][1:]))


def test_F11_local_trap_all_six_subsets():
    rows=[[1,0],[0,1],[.9,.9],[.9,-.9]]
    all_designs=exhaustive(rows,2)
    assert len(all_designs)==6
    assert determinant(rows,[0,1])==1
    assert max(d for _,_,d in neighborhood(rows,[0,1]))==determinant(rows,[0,2])
    assert float(max(d for _,d in all_designs))==2.6244
    result=run(rows,[0,1])
    assert result['ids'].tolist()==[0,1] and result['accepted_swaps']==0
    assert result['stop_reason']=='LOCAL_1_SWAP_NUMERICAL'
    assert result['certificate']!='LOCAL_1_SWAP_CERTIFIED'
    for initial,_ in all_designs:
        alternative=run(rows,list(initial))
        if alternative['stop_reason']=='LOCAL_1_SWAP_NUMERICAL':
            det=determinant(rows,alternative['ids'].tolist())
            assert max(d for _,_,d in neighborhood(rows,alternative['ids'].tolist()))<=det


def test_F12_singular_near_singular_and_disagreement(monkeypatch):
    singular=run([[1,0],[0,1],[2,0]],[0,1])
    assert singular['scans'][0]['qr_zero']>=1
    rows=[[1,0],[0,1],[1,1e-10]]
    assert determinant(rows,[0,2])>0  # mathematically SPD, not rank deficient
    uncertain=run(rows,[0,1])
    assert uncertain['stop_reason']=='LOCAL_1_SWAP_NUMERICAL'
    assert uncertain['scans'][0]['uncertain_nonimproving']==1
    assert uncertain['scans'][0]['first_uncertain'] is None
    assert uncertain['certificate']=='NUMERICAL_ONLY' and uncertain['accepted_swaps']==0
    overflow=run([[1,0],[0,1],[1e200,0]],[0,1])
    assert overflow['stop_reason']=='NUMERICAL_INCONCLUSIVE'
    # Controlled inconsistent gain verifies that the direct check is a real gate.
    monkeypatch.setattr('torchoed.exchange._exchange_ratios',lambda R,T: torch.full((R.shape[1],T.shape[1]),-1.,dtype=R.dtype))
    inconsistent=run(SIMPLE,[0,1])
    assert inconsistent['stop_reason']=='NUMERICAL_INCONCLUSIVE'
    assert inconsistent['accepted_swaps']==0


def test_exhaustive_integer_fixture_best_move_and_local_check():
    rows=[[1,0,0],[1,1,0],[1,0,1],[1,1,1],[1,-1,0],[1,0,-1]]
    all_designs=exhaustive(rows,4)
    assert len(all_designs)==15
    initial=[0,1,2,3]
    expected=max(d for _,_,d in neighborhood(rows,initial))
    one=run(rows,initial,max_iter=1)
    assert determinant(rows,one['ids'].tolist())==expected
    result=run(rows,initial)
    if result['stop_reason']=='LOCAL_1_SWAP_NUMERICAL':
        det=determinant(rows,result['ids'].tolist())
        assert max(d for _,_,d in neighborhood(rows,result['ids'].tolist()))<=det
    assert determinant(rows,result['ids'].tolist())<=max(d for _,d in all_designs)


def test_input_budget_ownership_and_reproducibility():
    F=torch.tensor(SIMPLE,dtype=torch.float64);selected=torch.tensor([0,1])
    saved=F.clone(),selected.clone()
    for options in ({'max_iter':-1},{'max_iter':True},{'tile_size':0},{'max_elements':20}):
        with pytest.raises(ValueError):
            fedorov_exchange(F,selected,**options)
    for bad in (F.float(),F.long()):
        with pytest.raises(TypeError):
            fedorov_exchange(bad,selected)
    with pytest.raises(ValueError):
        fedorov_exchange(F[None],selected)
    with pytest.raises(ValueError):
        run([[float('nan'),0],[0,1]],[0,1])
    assert torch.equal(F,saved[0]) and torch.equal(selected,saved[1])
    result=fedorov_exchange(F.requires_grad_(),selected)
    assert not result['ids'].requires_grad
    assert result['workspace_element_estimate']<=1_000_000
    constrained=fedorov_exchange(F,selected,max_iter=1,tile_size=64,max_elements=100)
    assert constrained['ids'].tolist()==[2,1]
    assert constrained['workspace_element_estimate']<=100
