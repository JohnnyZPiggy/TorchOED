from fractions import Fraction
from itertools import combinations, product
import importlib
import torch
from oracle_exchange import linear_value

search = importlib.import_module('torchoed.d_optimal_design')
ROWS = [[0,1,0],[0,0,1],[2,2,-2],[-2,0,-1],[-2,-2,0],[0,-1,0],[-1,1,1]]  # randint seed 141


def test_tangent_bound_fraction_saturated():
    p = 3
    gram = lambda rows: [[sum(Fraction(ROWS[r][i]*ROWS[r][j]) for r in rows) for j in range(p)] for i in range(p)]
    aug = [row+[Fraction(i==j) for j in range(p)] for i,row in enumerate(gram(range(p)))]
    for k in range(p):
        pivot = next(i for i in range(k,p) if aug[i][k])
        aug[k],aug[pivot] = aug[pivot],aug[k]
        aug[k] = [v/aug[k][k] for v in aug[k]]
        for i in range(p):
            if i!=k:
                factor = aug[i][k]
                aug[i] = [a-factor*b for a,b in zip(aug[i],aug[k])]
    inverse = [row[p:] for row in aug]
    for criterion in ('A','I'):
        B = [[Fraction(i==j) for j in range(p)] for i in range(p)] if criterion=='A' else [[v/7 for v in row] for row in gram(range(7))]
        Y = [[sum(inverse[i][j]*r[j] for j in range(p)) for i in range(p)] for r in ROWS]
        diagonal = [sum(y[i]*B[i][j]*y[j] for i in range(p) for j in range(p)) for y in Y]
        phi = linear_value(ROWS,[0,1,2],criterion)
        checked = 0
        for out,inc in product(combinations(range(3),2),combinations(range(3,7),2)):
            trial = [0,1,2]
            trial[out[0]],trial[out[1]] = inc
            new = linear_value(ROWS,trial,criterion)
            if new is not None:
                delta = sum(diagonal[i] for i in inc)-sum(diagonal[i] for i in out)
                assert phi+delta>0 and phi-new<=phi*delta/(phi+delta)<=delta
                checked += 1
        assert checked>0


def test_prefix_prune_preserves_winner_resume(monkeypatch):
    F = torch.randn(14,3,dtype=torch.float64,generator=torch.Generator().manual_seed(141))
    ids = torch.arange(6)
    original,limit = search._linear_pair_scores,search._linear_prefix_limit
    for criterion in ('A','I'):
        skipped = []
        def count(*args):
            length = limit(*args)
            skipped.append(len(args[0])-length)
            return length
        with monkeypatch.context() as patch:
            patch.setattr(search,'_linear_prefix_limit',count)
            actual = list(search._linear_pair_moves(F,ids,criterion,1_000_000,None,1e-12))
        def full(*args,**kwargs):
            kwargs['tolerance'] = None
            return original(*args,**kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(search,'_linear_pair_scores',full)
            expected = list(search._linear_pair_moves(F,ids,criterion,1_000_000,None,1e-12))
        assert sum(skipped)>0 and len(actual)>1
        assert torch.equal(torch.tensor(actual),torch.tensor(expected))
