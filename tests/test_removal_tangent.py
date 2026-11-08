from fractions import Fraction as R
from itertools import combinations
import importlib
import torch
from oracle_exchange import linear_value
s = importlib.import_module('torchoed.d_optimal_design')
ROWS = [[-2,0,-2],[1,-1,2],[-2,1,0],[0,1,-2],[1,-1,1],[-2,0,2],[-1,-1,2],[-2,0,-2]]  # randint seed 145


def inv(M):
    p = len(M)
    a = [row+[R(i==j) for j in range(p)] for i,row in enumerate(M)]
    for k in range(p):
        pivot = next((i for i in range(k,p) if a[i][k]),None)
        if pivot is None: return None
        a[k],a[pivot] = a[pivot],a[k]
        a[k] = [v/a[k][k] for v in a[k]]
        for i in range(p):
            if i!=k:
                factor = a[i][k]
                a[i] = [x-factor*y for x,y in zip(a[i],a[k])]
    return [r[p:] for r in a]


def test_removal_fraction_identity_and_bound():
    gram = lambda ids: [[sum(R(ROWS[r][i]*ROWS[r][j]) for r in ids) for j in range(3)] for i in range(3)]
    dot = lambda a,B,b: sum(a[i]*B[i][j]*b[j] for i in range(3) for j in range(3))
    for criterion in ('A','I'):
        B = [[R(i==j) for j in range(3)] for i in range(3)] if criterion=='A' else [[v/8 for v in r] for r in gram(range(8))]
        singular = checked = 0
        for n in (4,5):
            ids = list(range(n));MI = inv(gram(ids));phi = linear_value(ROWS,ids,criterion)
            y = [[sum(MI[i][j]*r[j] for j in range(3)) for i in range(3)] for r in ROWS]
            for a,b in combinations(ids,2):
                RI = inv(gram([r for r in ids if r not in (a,b)]))
                ga,gb,gab = dot(ROWS[a],MI,ROWS[a]),dot(ROWS[b],MI,ROWS[b]),dot(ROWS[a],MI,ROWS[b])
                rho = (1-ga)*(1-gb)-gab*gab
                assert (rho==0)==(RI is None)
                if RI is None:
                    singular += 1
                    continue
                pr = sum(B[i][j]*RI[j][i] for i in range(3) for j in range(3))
                ts = {}
                for j in range(n,8):
                    yr = [sum(RI[i][k]*ROWS[j][k] for k in range(3)) for i in range(3)]
                    exact = dot(yr,B,yr)
                    rr,delta,t = s._linear_removal_terms((ga,gab,gb),(dot(y[a],B,y[a]),dot(y[a],B,y[b]),dot(y[b],B,y[b])),(dot(ROWS[a],MI,ROWS[j]),dot(ROWS[b],MI,ROWS[j])),(dot(y[a],B,y[j]),dot(y[b],B,y[j])),dot(y[j],B,y[j]))
                    assert rr==rho>0 and delta==pr-phi and t==exact
                    ts[j] = exact
                for c,d in combinations(range(n,8),2):
                    trial = ids[:];trial[a],trial[b] = c,d
                    T = ts[c]+ts[d]
                    assert phi-linear_value(ROWS,trial,criterion)<=pr*T/(pr+T)-(pr-phi)
                    checked += 1
        assert singular>0 and checked>0


def test_removal_preserves_resume_and_skips_beyond_d(monkeypatch):
    F = torch.randn(14,3,dtype=torch.float64,generator=torch.Generator().manual_seed(145));ids = torch.arange(6)
    original,data,limit = s._linear_pair_scores,s._linear_removal_data,s._linear_removal_limit
    for criterion in ('A','I'):
        B = torch.eye(3,dtype=F.dtype) if criterion=='A' else F.mT@F/len(F)
        phi = s._linear_scores(F,ids,B,criterion)[1]
        context = {};extra = []
        def record(*args):
            result = data(*args);context.update(chosen=args[2],pool=args[3],diagonal=args[4],t=result[2]);return result
        def count(*args):
            negative,pr,delta,theta,tol = args
            a,b = torch.triu_indices(len(negative),len(negative),offset=1)
            diag,pool,t = context['diagonal'],context['pool'],context['t']
            eta = min(theta+delta,pr/2)
            e = t[a]+t[b] < pr*eta/(pr-eta)-tol
            d = diag[pool[a]]+diag[pool[b]] >= diag[context['chosen']].sum()+phi*theta/(phi-theta)-tol
            extra.append((e&d).any().item())
            return limit(*args)
        with monkeypatch.context() as patch:
            patch.setattr(s,'_linear_removal_data',record);patch.setattr(s,'_linear_removal_limit',count)
            actual = list(s._linear_pair_moves(F,ids,criterion,1_000_000,None,1e-12))
        def full(*args,**kwargs):
            kwargs['tolerance'] = None
            return original(*args,**kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(s,'_linear_pair_scores',full)
            expected = list(s._linear_pair_moves(F,ids,criterion,1_000_000,None,1e-12))
        assert any(extra) and len(actual)>1 and torch.equal(torch.tensor(actual),torch.tensor(expected))
