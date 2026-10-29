import itertools
import math
from fractions import Fraction

import pytest
import torch

from torchoed import score_two_swaps
from oracle_exchange import determinant, neighborhood
from oracle_two_swap import TRAP, INTEGER11, exact_gram, bareiss_det, pair_cases


def features(rows):
    return torch.tensor([[float(Fraction(x)) if isinstance(x,str) else float(x)
                          for x in r] for r in rows],dtype=torch.float64)


def score(rows, selected, **kwargs):
    return score_two_swaps(features(rows),torch.tensor(selected,dtype=torch.int64),**kwargs)


def test_B1_four_individual_swaps_are_worse():
    assert determinant(TRAP,[0,1])==1
    cases=neighborhood(TRAP,[0,1])
    assert len(cases)==4
    assert all(det==Fraction(81,100) for _,_,det in cases)


def test_B2_simultaneous_trap_exact_six_subsets():
    exact=[(s,bareiss_det(exact_gram(TRAP,s))) for s in itertools.combinations(range(4),2)]
    assert len(exact)==6 and max(d for _,d in exact)==Fraction(6561,2500)
    result=score(TRAP,[0,1])
    assert result['evaluated_pairs']==1 and result['status'].tolist()==[0]
    assert result['outgoing_positions'].tolist()==[[0,1]]
    assert result['incoming_ids'].tolist()==[[2,3]]
    assert math.isclose(result['ratios'][0].item(),6561/2500,abs_tol=1e-9,rel_tol=1e-9)
    assert math.isclose(result['gains'][0].item(),math.log(6561/2500),abs_tol=1e-9,rel_tol=1e-9)


def test_B4_add_drop_identity_not_two_swap():
    start={0,1}
    endpoints=[(start|{j})-{i} for j in (2,3) for i in start|{j}]
    assert all(len(start-s)<=1 and s!={2,3} for s in endpoints)
    result=score(TRAP,[0,1])
    assert result['outgoing_positions'][0].unique().numel()==2
    assert result['incoming_ids'][0].unique().numel()==2


def test_B6_k1_identity_independent_of_search():
    generator=torch.Generator().manual_seed(7421)
    rows=[[1,0],[0,1]]+torch.randint(-2,3,(4,2),generator=generator).tolist()
    selected=[0,1,2]
    F=features(rows);X=F[selected];M=X.mT@X;L=torch.linalg.cholesky(M)
    signs=torch.diag(torch.tensor([-1.,1.],dtype=torch.float64))
    old=determinant(rows,selected)
    for i,j,det in neighborhood(rows,selected):
        W=torch.stack((F[selected[i]],F[j]),dim=1)
        Z=torch.linalg.solve_triangular(L,W,upper=False)
        gamma=torch.linalg.det(torch.eye(2,dtype=torch.float64)+signs@(Z.mT@Z)).item()
        assert math.isclose(gamma,float(det/old),abs_tol=1e-9,rel_tol=1e-9)


def test_B7_all_150_integer_pairs_exact_fraction_bareiss():
    selected=list(range(6))
    reference=list(pair_cases(INTEGER11,selected))
    assert len(reference)==math.comb(6,2)*math.comb(5,2)==150
    initial=bareiss_det(exact_gram(INTEGER11,selected))
    assert initial==determinant(INTEGER11,selected)>0
    result=score(INTEGER11,selected,tile_size=17)
    assert result['evaluated_pairs']==150 and result['counts']['VALID']==150
    for k,(positions,incoming,trial,gram,det) in enumerate(reference):
        assert det==determinant(INTEGER11,trial)>0
        assert result['outgoing_positions'][k].tolist()==list(positions)
        assert result['incoming_ids'][k].tolist()==list(incoming)
        assert math.isclose(result['ratios'][k].item(),float(det/initial),abs_tol=1e-9,rel_tol=1e-9)
        assert math.isclose(result['gains'][k].item(),math.log(float(det/initial)),abs_tol=1e-9,rel_tol=1e-9)
        X=features(INTEGER11)[trial]
        torch.testing.assert_close(X.mT@X,features(gram),atol=0,rtol=0)


def test_B8_saturated_without_delete_only_inverse(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('dense inverse forbidden')
    monkeypatch.setattr(torch.linalg,'inv',forbidden)
    assert bareiss_det(exact_gram(TRAP,[]))==0
    result=score(TRAP,[0,1])
    assert result['status'].tolist()==[0] and result['gains'][0]>0


def test_B9_duplicate_features_distinct_ids():
    rows=[[1,0],[0,1],[1,0],[0,1],[1,0]]
    result=score(rows,[0,1])
    assert result['evaluated_pairs']==3
    assert result['incoming_ids'].tolist()==[[2,3],[2,4],[3,4]]
    assert result['status'].tolist()==[0,1,0]
    assert result['gains'][0].item()==result['gains'][2].item()==0


def test_B10_order_tiles_rng_unsorted_ids():
    selected=[5,0,4,1,3,2]
    before=torch.random.get_rng_state().clone()
    baseline=score(INTEGER11,selected,tile_size=1)
    for width in (7,64):
        result=score(INTEGER11,selected,tile_size=width)
        for key in ('outgoing_positions','incoming_ids','status','initial_ids'):
            assert torch.equal(result[key],baseline[key])
        for key in ('ratios','gains','direct_scores'):
            torch.testing.assert_close(result[key],baseline[key],atol=1e-9,rtol=1e-9,equal_nan=True)
        assert result['reasons']==baseline['reasons']
    assert torch.equal(before,torch.random.get_rng_state())


def test_B11_scoring_coverage_not_search_certificate():
    result=score(TRAP,[0,1])
    assert result['certificate']=='NOT_APPLICABLE'
    assert not {'winner','stop_reason','history','accepted_swaps','final_ids'} & result.keys()
    # Full counts already checked by B7; no stopping/monotonicity PASS is claimed.


def test_B12_boundaries_pathological_ratio_and_budget(monkeypatch):
    singular=score([[1,0],[0,1],[1,0],[2,0]],[0,1])
    assert singular['status'].tolist()==[1] and torch.isnan(singular['gains']).all()
    near=[[1,0],[0,1],[1,0],[1,'1/10000000000']]
    assert bareiss_det(exact_gram(near,[2,3]))>0
    uncertain=score(near,[0,1])
    assert uncertain['status'].tolist()==[2]
    assert uncertain['reasons']==['M2_REJECT_QR_NONZERO']
    assert torch.isnan(uncertain['gains']).all()
    overflow=score([[1,0],[0,1],[1e200,0],[0,1e200]],[0,1])
    assert overflow['status'].tolist()==[2]
    for selected in ([0,0],[-1,1],[0,4]):
        with pytest.raises(ValueError):
            score(TRAP,selected)
    for F,selected in ((features(TRAP).float(),torch.tensor([0,1])),
                       (features(TRAP),torch.tensor([True,False]))):
        with pytest.raises(TypeError):
            score_two_swaps(F,selected)
    with pytest.raises(ValueError):
        score([[1,0],[0,1],[float('inf'),1]],[0,1])
    with pytest.raises(ValueError):
        score(TRAP,[0,1],tile_size=0)
    def forbidden(*args,**kwargs):
        pytest.fail('allocation/linalg attempted before invalid budget rejection')
    with monkeypatch.context() as ctx:
        ctx.setattr(torch,'full',forbidden)
        ctx.setattr(torch.linalg,'cholesky_ex',forbidden)
        with pytest.raises(ValueError):
            score(INTEGER11,list(range(6)),max_elements=1000)
    for raw in (-1.,float('nan'),float('inf'),1.):
        monkeypatch.setattr('torchoed.two_swap._rank_four_ratios',lambda L,F,ids,pairs: torch.full((pairs.shape[0],),raw,dtype=F.dtype))
        disagreement=score(TRAP,[0,1])
        assert disagreement['status'].tolist()==[2] and torch.isnan(disagreement['gains']).all()


def test_empty_pairs_ownership_and_limited_tile():
    for rows,selected in (([[1,0],[0,1]],[0,1]),([[1],[2],[3]],[0])):
        result=score(rows,selected)
        assert result['evaluated_pairs']==0 and result['status'].shape==(0,)
        assert result['incoming_ids'].shape==(0,2)
    F=features(TRAP);ids=torch.tensor([0,1]);saved=F.clone(),ids.clone()
    result=score_two_swaps(F,ids,tile_size=64,max_elements=400)
    assert result['status'].tolist()==[0] and result['workspace_element_estimate']<=400
    assert torch.equal(F,saved[0]) and torch.equal(ids,saved[1])
