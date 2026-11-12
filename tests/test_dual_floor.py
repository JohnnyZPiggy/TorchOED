from itertools import combinations
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
