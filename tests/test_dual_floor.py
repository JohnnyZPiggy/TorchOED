from itertools import combinations

import pytest
import torch
from torchoed import design_diagnostics, optimal_design, full_factorial, polynomial_features
from torchoed.d_optimal_design import _dual_floor
from oracle_exchange import determinant, linear_value


def test_W2_1_fraction_efficiencies():
    # Frozen random.Random(87) integer fixture, shared with the gate's L2 oracle.
    rows = [[-1,-1,2],[-2,0,0],[2,0,-2],[0,-2,0],[1,-2,1],[2,2,-2],[-1,0,-1]]
    F = torch.tensor(rows,dtype=torch.float64)
    for c in ('D','A','I'):
        values = {s:determinant(rows,s) if c=='D' else linear_value(rows,s,c) for s in combinations(range(7),4)}
        valid = {s:v for s,v in values.items() if v}
        optimum = max(valid.values()) if c=='D' else min(valid.values())
        B = torch.eye(3,dtype=F.dtype) if c=='A' else F.mT@F/7
        for ids,value in valid.items():
            X=F[list(ids)]; M=X.mT@X; Y=torch.linalg.solve(M,F.mT)
            current = 3/(4*(F.mT*Y).sum(0).max().item()) if c=='D' else float(value)/(4*(Y*(B@Y)).sum(0).max().item())
            exact = float(value/optimum)**(1/3) if c=='D' else float(optimum/value)
            floor = _dual_floor(F,M,4,c,current)
            assert current<=floor<=exact+1e-9
        print(c,len(valid),'exact-efficiency bounds PASS')


def test_W2_2_closed_form():
    F=torch.tensor([[1.,-1.],[1.,1.]],dtype=torch.float64)
    result=optimal_design(F,2,restarts=1,generator=torch.Generator().manual_seed(0))
    assert abs(result['efficiency_lower_bound']-1)<=1e-12
    assert result['efficiency_lower_bound']==result['d_efficiency_lower_bound']


def test_ai_floor_dense_reference():
    F = polynomial_features(full_factorial([3]*3).double()-1, degree=2)
    ids = torch.randperm(len(F), generator=torch.Generator().manual_seed(0))[:20]
    M = F[ids].mT@F[ids]/len(ids)
    for criterion in ('A', 'I'):
        B = torch.eye(F.shape[1], dtype=F.dtype) if criterion=='A' else F.mT@F/len(F)
        own = torch.trace(torch.linalg.solve(M, B))
        w = F.new_full((len(F),), 1/len(F))
        for _ in range(100000):
            Mw = F.mT@(w[:, None]*F)
            Y = torch.linalg.solve(Mw, F.mT)
            q = (Y*(B@Y)).sum(0)
            phi = torch.trace(torch.linalg.solve(Mw, B))
            if phi/q.max() >= 1-1e-12:
                break
            w *= q.sqrt()
            w /= w.sum()
        else:
            raise AssertionError('dense reference did not converge')
        bound = _dual_floor(F, M*len(ids), len(ids), criterion, 0.)
        assert bound >= (1-1e-3)*phi.square()/(q.max()*own)
        assert bound <= phi/own+1e-12


def test_design_diagnostics_factorial():
    F = polynomial_features(full_factorial([2, 2, 2]).double() * 2 - 1, 1)
    d = design_diagnostics(F, torch.arange(8))
    assert d["columns"] == [1, 2, 3]
    torch.testing.assert_close(d["correlation"], torch.eye(3, dtype=F.dtype), atol=1e-12, rtol=0)
    torch.testing.assert_close(d["vif"], F.new_ones(3), atol=1e-12, rtol=0)
    torch.testing.assert_close(d["spv"], F.new_full((8,), 4), atol=1e-12, rtol=0)
    assert torch.equal(d["fds"], d["spv"].sort().values)


def test_design_diagnostics_power_factorial():
    """SciPy 1.18.1 ncf.sf(f.ppf(1−α,1,ν),1,ν,λ), dev-only."""
    F = polynomial_features(2 * full_factorial([2] * 4).double() - 1, 1)
    power = design_diagnostics(F, torch.arange(16), effect=0.5, alpha=0.05)["power"]
    torch.testing.assert_close(power, power[0].expand_as(power), atol=1e-14, rtol=0)
    torch.testing.assert_close(power, F.new_full((5,), 0.44646615784540034), atol=1e-8, rtol=0)


def test_design_diagnostics_power_per_coefficient():
    """SciPy 1.18.1 ncf.sf(f.ppf(1−α,1,ν),1,ν,λ), dev-only."""
    F = polynomial_features(2 * full_factorial([2] * 3).double() - 1, 1)
    effect = F.new_tensor([0., 1., 2., 3.])
    power = design_diagnostics(F, torch.arange(8), effect=effect, alpha=0.1)["power"]
    expected = F.new_tensor([0.1, 0.7499667343577331, 0.9978740910366228, 0.9999996431473732])
    torch.testing.assert_close(power, expected, atol=1e-8, rtol=0)


def test_design_diagnostics_power_one_residual_df():
    """SciPy 1.18.1 ncf.sf(f.ppf(1−α,1,ν),1,ν,λ), dev-only."""
    F = polynomial_features(2 * full_factorial([2] * 2).double() - 1, 1)
    ids = torch.arange(4)
    power = design_diagnostics(F, ids, effect=2.0, alpha=0.05)["power"]
    torch.testing.assert_close(power, F.new_full((3,), 0.2463555872889929), atol=1e-8, rtol=0)
    with pytest.raises(ValueError):
        design_diagnostics(F, ids, effect=2.0, alpha=0.0005)
    with pytest.raises(ValueError):
        design_diagnostics(F, torch.arange(3), effect=1.0)


def test_design_diagnostics_dense_oracle():
    F = polynomial_features(full_factorial([3, 3]).double() - 1, 2)
    ids = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8])
    S = F[ids]
    d = design_diagnostics(F, ids)
    for j, column in enumerate(d["columns"]):
        y = S[:, column]
        X = S[:, [0] + [k for k in d["columns"] if k != column]]
        residual = y - X @ torch.linalg.lstsq(X, y).solution
        torch.testing.assert_close(d["vif"][j], (y - y.mean()).square().sum() / residual.square().sum(), atol=1e-12, rtol=0)
    torch.testing.assert_close(d["spv"], len(ids) * (F @ torch.linalg.inv(S.mT @ S) @ F.mT).diagonal(), atol=1e-12, rtol=0)
    torch.testing.assert_close(design_diagnostics(F, ids, region=S)["spv"].mean(), F.new_tensor(F.shape[1]), atol=1e-12, rtol=0)


def test_design_diagnostics_d_floor():
    F = polynomial_features(full_factorial([3, 3]).double() - 1, 2)
    r = optimal_design(F, 8, generator=torch.Generator().manual_seed(0))
    maximum = design_diagnostics(F, r["ids"])["spv"].max().item()
    assert maximum >= F.shape[1] * (1 - 1e-12)
    assert r["efficiency_lower_bound"] >= F.shape[1] / maximum * (1 - 1e-12)


@pytest.mark.parametrize("case,weights,values", [
    ('a', {'D':.5,'A':.5}, (256,11/2,3/2)),
    ('a', {'D':1/3,'A':1/3,'I':1/3}, (256,11/2,3/2)),
    ('b', {'D':.5,'I':.5}, (3/8,None,7/9)),
    ('b', {'D':1/3,'A':1/3,'I':1/3}, (3/8,70/3,7/9))])
def test_compound_oracles(case, weights, values):
    from torchoed import mixture_process_features, scheffe_features, simplex_centroid
    F = (polynomial_features(full_factorial([3,3]).double()-1) if case=='a' else
         mixture_process_features(scheffe_features(simplex_centroid(2),'quadratic'),
                                  torch.tensor([[1.,-1],[1,0],[1,1]],dtype=torch.float64)))
    n = 6 if case=='a' else 7; p = F.shape[1]; B = F.T@F/len(F)
    def objective(M):
        return sum(w * (torch.linalg.slogdet(M/n).logabsdet/p if k=='D' else
                   -torch.trace(torch.linalg.solve(M/n,torch.eye(p,dtype=F.dtype) if k=='A' else B)).log())
                   for k,w in weights.items())
    for fixed in ([],[4]) if case=='a' and 'I' not in weights else ([],):
        r = optimal_design(F,n,criterion=weights,fixed_ids=fixed,restarts=20,generator=torch.Generator().manual_seed(0))
        M = F[r['ids']].T@F[r['ids']]
        all_ids = [list(fixed)+list(s) for s in combinations([i for i in range(9) if i not in fixed],n-len(fixed))]
        optima = [objective(x.T@x).item() for s in all_ids if torch.linalg.eigvalsh((x:=F[s]).T@x)[0]>1e-12]
        assert r['objective'] == pytest.approx(max(optima),rel=1e-9)
        assert r['objective'] == pytest.approx(objective(M).item(),rel=1e-9)
        if not fixed:
            actual = (torch.linalg.det(M),torch.trace(torch.linalg.inv(M)),torch.trace(B@torch.linalg.inv(M)))
            for got, expected in zip(actual,values):
                if expected is not None: assert got.item() == pytest.approx(expected,rel=1e-9)
        assert r['ids'].tolist() == sorted(set(r['ids'].tolist()))
        assert set(fixed) <= set(r['ids'].tolist()) and r['efficiency_lower_bound'] <= 1+1e-12


def test_compound_weight_one_and_boundary():
    F = polynomial_features(full_factorial([3,3]).double()-1)
    for c in ('D','A','I'):
        r = optimal_design(F,6,criterion=c,generator=torch.Generator().manual_seed(0)); M=F[r['ids']].T@F[r['ids']]
        assert _dual_floor(F,M,6,{c:1.},0.) == pytest.approx(_dual_floor(F,M,6,c,0.),rel=1e-12)
    r = optimal_design(F,6,criterion={'D':2.},generator=torch.Generator().manual_seed(0)); M=F[r['ids']].T@F[r['ids']]
    assert r['criterion'] == {'D':1.}
    assert r['objective'] == pytest.approx((torch.linalg.slogdet(M/6).logabsdet/6).item(),rel=1e-12)
    for weights, options in [({'X':1},{}),({'D':-1},{}),({'D':0},{}),({'D':1},{'blocks':[3,3]})]:
        with pytest.raises(ValueError): optimal_design(F,6,criterion=weights,**options)


@pytest.mark.parametrize('case,weight,det,trace', [('a',0,1024,3/2),('a',.1,768,1/3),('b',.25,80,32/25),('b',1,20,8/25)])
def test_alias_search_oracles(case, weight, det, trace):
    X = (full_factorial([2]*3).double()*2-1 if case=='a' else full_factorial([3,3]).double()-1)
    F = polynomial_features(X,1); n = 6 if case=='a' else 5; p = F.shape[1]
    G = (torch.stack([X[:,0]*X[:,1],X[:,0]*X[:,2],X[:,1]*X[:,2]],1) if case=='a' else
         torch.stack([X[:,0]*X[:,1],X[:,0]**2,X[:,1]**2],1))
    def objective(ids):
        S=F[list(ids)]; M=S.T@S
        return torch.linalg.slogdet(M/n).logabsdet/p-weight*torch.linalg.solve(M,S.T@G[list(ids)]).square().sum()
    for fixed in ([],[4]) if case=='b' and weight==.25 else ([],):
        r=optimal_design(F,n,alias=G,alias_weight=weight,fixed_ids=fixed,restarts=20,generator=torch.Generator().manual_seed(0))
        M=F[r['ids']].T@F[r['ids']]
        assert r['objective']==pytest.approx((torch.log(F.new_tensor(det/n**p))/p-weight*trace).item(),rel=1e-9)
        assert torch.linalg.det(M).item()==pytest.approx(det,rel=1e-9)
        assert r['alias_trace']==pytest.approx(trace,rel=1e-9)
        assert r['ids'].tolist()==sorted(set(r['ids'].tolist()))
        assert set(fixed)<=set(r['ids'].tolist()) and r['efficiency_lower_bound']<=1+1e-12
        if fixed:
            completions=[fixed+list(s) for s in combinations([i for i in range(len(F)) if i not in fixed],n-len(fixed))]
            values=[objective(s).item() for s in completions if torch.linalg.eigvalsh(F[s].T@F[s])[0]>1e-12]
            assert r['objective']==pytest.approx(max(values),rel=1e-9)


def test_alias_local_certificate_and_boundary():
    X=full_factorial([3,3]).double()-1; F=polynomial_features(X,1)
    G=torch.stack([X[:,0]*X[:,1],X[:,0]**2,X[:,1]**2],1)
    r=optimal_design(F,5,alias=G,alias_weight=.25,restarts=1,generator=torch.Generator().manual_seed(0))
    assert r['alias_trace']==pytest.approx(design_diagnostics(F,r['ids'],alias=G)['alias'].square().sum().item(),rel=1e-12)
    if r['certificate']=='LOCAL_ALIAS_EXCHANGE':
        for i in range(5):
            for j in set(range(len(F)))-set(r['ids'].tolist()):
                ids=r['ids'].clone(); ids[i]=j; S=F[ids]; M=S.T@S
                if torch.linalg.eigvalsh(M)[0]<=1e-12: continue
                value=torch.linalg.slogdet(M/5).logabsdet/3-.25*torch.linalg.solve(M,S.T@G[ids]).square().sum()
                assert value.item()<=r['objective']+1e-12*abs(r['objective'])
    for options in ({'alias':G[:-1]}, {'alias_weight':-1}, {'criterion':'A'}, {'criterion':{'D':1}}, {'blocks':[3,3]}):
        with pytest.raises(ValueError): optimal_design(F,6,**({'alias':G}|options))


@pytest.mark.parametrize('case', ['D','A','I','compound','alias','blocks','split','both','covariates'])
def test_given_design_roundtrip(case):
    X=full_factorial([2]*3).double()*2-1; F=polynomial_features(X,1); n=6
    options={'criterion':case} if case in ('D','A','I') else {}
    if case in ('D','A','I'): n=8
    if case=='compound': options={'criterion':{'D':.5,'I':.5}}
    if case=='alias': options={'alias':X[:, :1]*X[:,1:2]}
    if case in ('blocks','split','both','covariates'): F=X
    if case in ('blocks','both'): options['blocks']=[4,4]; n=8
    if case in ('split','both'): options.update(whole_plots=[2]*(n//2),hard=X[:,0].long())
    if case=='covariates': options['covariates']=torch.arange(n,dtype=F.dtype)[:,None]
    r=optimal_design(F,n,restarts=2,generator=torch.Generator().manual_seed(0),**options)
    e=optimal_design(F,n,fixed_ids=r['ids'],restarts=1,**options)
    assert torch.equal(e['ids'],r['ids'])
    for key in ('objective','efficiency_lower_bound'):
        assert e[key]==pytest.approx(r[key],abs=1e-12,rel=0)


def test_given_design_poor_oracle_and_boundary():
    F=polynomial_features(torch.arange(-4,5,dtype=torch.float64)[:,None],2); ids=[0,1,2,3]; n=4
    M=F[ids].T@F[ids]; best=max(torch.linalg.det(F[list(s)].T@F[list(s)]).item() for s in combinations(range(9),n))
    e=optimal_design(F,n,fixed_ids=ids,restarts=1)
    kw=3/(n*(F*torch.linalg.solve(M,F.T).T).sum(1).max().item())
    assert torch.linalg.det(M).item()<best
    assert kw-1e-12<=e['efficiency_lower_bound']<=(torch.linalg.det(M).item()/best)**(1/3)+1e-12
    with pytest.raises(ValueError): optimal_design(F,n,fixed_ids=[0,1,2,3,4],restarts=1)
