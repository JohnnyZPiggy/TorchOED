from itertools import combinations
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from gather_scores import _Scorer, _unrank, gather_two_swap_ratios


def test_pair_unranking_independent_combinations():
    for size in (2,3,6,20,48):
        expected=list(combinations(range(size),2))
        actual=_unrank(torch.arange(len(expected)),size)
        assert actual.tolist()==[list(pair) for pair in expected]


def test_scale_pair_row_boundaries():
    size=980;ranks=[];expected=[];offset=0
    for first in range(size-1):
        length=size-first-1
        ranks.extend((offset,offset+length-1))
        expected.extend(((first,first+1),(first,size-1)))
        offset+=length
    assert _unrank(torch.tensor(ranks),size).tolist()==[list(x) for x in expected]


@pytest.mark.parametrize('form',['gather4','add_first'])
def test_saturated_trap_and_tile_order(form):
    F=torch.tensor([[1.,0.],[0.,1.],[.9,.9],[.9,-.9]],dtype=torch.float64)
    ids=torch.tensor([0,1])
    for width in (1,16):
        result=torch.cat(list(gather_two_swap_ratios(F,ids,form=form,tile_size=width)))
        torch.testing.assert_close(result,torch.tensor([2.6244],dtype=torch.float64),atol=1e-12,rtol=1e-12)


def test_empty_neighborhood_and_input_ownership():
    F=torch.eye(2,dtype=torch.float64);ids=torch.arange(2);saved=F.clone(),ids.clone()
    assert list(gather_two_swap_ratios(F,ids))==[]
    assert torch.equal(F,saved[0]) and torch.equal(ids,saved[1])


def test_bounded_cache_and_eager_errors():
    x=torch.arange(-5,6,dtype=torch.float64);F=torch.stack((x*0+1,x,x*x),dim=1);ids=torch.arange(6)
    cache=_Scorer(F,ids,max_elements=990,tile_size=7)
    assert cache.nn is None and cache.workspace_elements<=990
    for badF,badids,options in [(F.float(),ids,{}),(F,torch.tensor([0,0,1,2,3,4]),{}),
                                (F,ids,{'max_elements':1}),(F,ids,{'form':'delete_first'})]:
        with pytest.raises((ValueError,TypeError)):
            gather_two_swap_ratios(badF,badids,**options)
