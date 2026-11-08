from fractions import Fraction
from itertools import combinations, product
import importlib

import torch
from oracle_exchange import determinant, linear_value

search = importlib.import_module('torchoed.d_optimal_design')
ROWS = [[1,0,0],[0,1,0],[0,0,1],[2,0,0],[0,2,0],[0,0,2],[2,2,2]]


def test_closed_add_first_fraction_saturated():
    g = [[sum(Fraction(x*y) for x,y in zip(a,b)) for b in ROWS] for a in ROWS]
    legal = singular = 0
    for criterion in ('A','I'):
        B = [[Fraction(a==b) if criterion=='A' else
              sum(Fraction(row[a]*row[b],7) for row in ROWS) for b in range(3)] for a in range(3)]
        q = [[sum(Fraction(a[i])*B[i][j]*b[j] for i in range(3) for j in range(3))
              for b in ROWS] for a in ROWS]
        old = linear_value(ROWS,[0,1,2],criterion)
        for out,inc in product(combinations(range(3),2),combinations(range(3,7),2)):
            idx = (*out,*inc)
            G = {(Ellipsis,i,j):g[idx[i]][idx[j]] for i in range(4) for j in range(4)}
            Q = {(Ellipsis,i,j):q[idx[i]][idx[j]] for i in range(4) for j in range(4)}
            add,num,tau,pi = search._linear_closed_terms(G,Q)
            assert (1-G[...,0,0])*(1-G[...,1,1])-G[...,0,1]**2 == 0
            trial = [0,1,2]
            trial[out[0]],trial[out[1]] = inc
            new = linear_value(ROWS,trial,criterion)
            assert pi>0 and pi*tau == determinant(ROWS,trial)
            assert (tau==0) == (new is None)
            if new is None:
                singular += 1
            else:
                assert add+num/tau == old-new
                legal += 1
    assert legal>0 and singular>0 and legal+singular==36


def test_gather4_subset_is_bitwise_full_batch():
    F = torch.randn(40,10,dtype=torch.float64,generator=torch.Generator().manual_seed(136))
    ids = torch.arange(20)
    L = torch.linalg.cholesky(F[ids].mT@F[ids])
    Z = torch.linalg.solve_triangular(L,F.mT,upper=False)
    pairs = next(search._pair_tiles(20,list(range(20,40)),1387))
    chosen = torch.arange(0,len(pairs),13)
    J = torch.diag(F.new_tensor([-1.,-1.,1.,1.]))
    for criterion in ('A','I'):
        T = torch.linalg.solve_triangular(L.mT,Z,upper=True) if criterion=='A' else (Z@Z.mT/40)@Z
        legal,full = search._linear_gather4(Z,T,ids,pairs,criterion,J)
        sublegal,subset = search._linear_gather4(Z,T,ids,pairs[chosen],criterion,J)
        assert legal.all() and sublegal.all()
        assert torch.equal(subset,full[chosen])


def test_screen_preserves_ties_and_whole_cutoff_sequence(monkeypatch):
    F = torch.tensor(ROWS,dtype=torch.float64)
    ids = torch.arange(3)
    original = search._linear_pair_scores
    for criterion in ('A','I'):
        actual = list(search._linear_pair_moves(F,ids,criterion,1_000_000,None,1e-12))
        def full(*args,**kwargs):
            kwargs['tolerance'] = None
            return original(*args,**kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(search,'_linear_pair_scores',full)
            expected = list(search._linear_pair_moves(F,ids,criterion,1_000_000,None,1e-12))
        assert actual == expected and len(actual)>1
