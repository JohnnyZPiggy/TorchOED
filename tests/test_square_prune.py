from fractions import Fraction as R
from itertools import combinations
import torch
from oracle_exchange import linear_value
from test_removal_tangent import inv, ROWS, s


def test_square_fraction_bound():
    gram = lambda ids: [[sum(R(ROWS[r][i]*ROWS[r][j]) for r in ids) for j in range(3)] for i in range(3)]
    dot = lambda a,B,b: sum(a[i]*B[i][j]*b[j] for i in range(3) for j in range(3))
    MI = inv(gram(range(3)))
    y = [[sum(MI[i][j]*row[j] for j in range(3)) for i in range(3)] for row in ROWS]
    checked = 0
    for criterion in ('A','I'):
        B = [[R(i==j) for j in range(3)] for i in range(3)] if criterion=='A' else [[v/8 for v in row] for row in gram(range(8))]
        Q = [[dot(y[a],B,y[b]) for b in range(3)] for a in range(3)]
        P = [[sum(Q[i][k]*Q[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        phi = linear_value(ROWS,[0,1,2],criterion)
        for a,b in combinations(range(3),2):
            q = Q[a][a],Q[a][b],Q[b][b]
            alpha,det,trace = s._linear_square_terms(q,(P[a][a],P[a][b],P[b][b]))
            rest = next(r for r in range(3) if r not in (a,b))
            OI = inv([[q[0],q[1]],[q[1],q[2]]])
            tau = Q[rest][rest]-sum(Q[rest][u]*OI[i][j]*Q[v][rest] for i,u in enumerate((a,b)) for j,v in enumerate((a,b)))
            assert alpha==phi-tau
            for c,d in combinations(range(3,8),2):
                ids = [0,1,2];ids[a],ids[b] = c,d
                if inv(gram(ids)) is None: continue
                sigma = sum(dot(ROWS[u],MI,ROWS[v])**2 for u in (a,b) for v in (c,d))
                value = linear_value(ROWS,ids,criterion)
                exact = (value-tau)*sigma-trace
                assert exact>=0 and exact**2>=4*det
                checked += 1
    assert checked>0


def test_square_preserves_resume_and_skips_beyond_d(monkeypatch):
    F = torch.randn(16,3,dtype=torch.float64,generator=torch.Generator().manual_seed(0));ids = torch.arange(3)
    F[:3] = torch.linalg.qr(F[:3]).Q
    original,data,limit = s._linear_pair_scores,s._linear_square_data,s._linear_square_limit
    for criterion in ('A','I'):
        B = torch.eye(3,dtype=F.dtype) if criterion=='A' else F.mT@F/len(F)
        phi,diagonal = s._linear_scores(F,ids,B,criterion)[1:3]
        context,extra = {},[]
        def record(*args):
            result = data(*args);context.update(chosen=args[2],pool=args[3],scores=result[0]);return result
        def count(*args):
            negative,alpha,kappa,theta,tol = args
            a,b = torch.triu_indices(negative.shape[1],negative.shape[1],offset=1)
            scores,pool = context['scores'],context['pool'];den = alpha-theta+tol
            threshold = torch.where(den>0,kappa/den,float('inf'))
            f = scores[:,a]+scores[:,b]<threshold[:,None]
            d = diagonal[pool[a]]+diagonal[pool[b]] >= (diagonal[context['chosen']].sum(0)+phi*theta/(phi-theta)-tol)[:,None]
            extra.append((f&d).any().item())
            return limit(*args)
        with monkeypatch.context() as patch:
            patch.setattr(s,'_linear_square_data',record);patch.setattr(s,'_linear_square_limit',count)
            actual = list(s._linear_pair_moves(F,ids,criterion,1_000_000,None,1e-12))
        def full(*args,**kwargs):
            kwargs['tolerance'] = None
            return original(*args,**kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(s,'_linear_pair_scores',full)
            expected = list(s._linear_pair_moves(F,ids,criterion,1_000_000,None,1e-12))
        assert any(extra) and len(actual)>1 and torch.equal(torch.tensor(actual),torch.tensor(expected))


