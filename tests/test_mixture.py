import math
from fractions import Fraction
from itertools import combinations

import pytest
import torch

from torchoed import scheffe_features, optimal_design, polynomial_features, simplex_lattice, simplex_centroid, mixture_vertices


def test_simplex_lattice_exact_rows():
    for q, m in ((3, 4), (4, 4), (3, 3)):
        rows = simplex_lattice(q, m).tolist()
        assert len(rows) == len({tuple(row) for row in rows}) == math.comb(q + m - 1, m)
        for row in rows:
            counts = [round(x * m) for x in row]
            assert row == [a / m for a in counts]
            assert sum(Fraction(a, m) for a in counts) == 1


@pytest.mark.parametrize("q,n,logdet", [(3, 6, -6 * math.log(4)), (4, 10, -12 * math.log(4))])
def test_scheffe_quadratic_kiefer_bound(q, n, logdet):
    p = polynomial_features(simplex_lattice(q, 4), 2, include_bias=False)
    f = torch.cat((p[:, :q], p[:, 2 * q:]), dim=1)
    result = optimal_design(f, n, generator=torch.Generator().manual_seed(0))
    # Kiefer's D-optimal bound is attained by the {q, 2} lattice.
    assert result["objective"] == pytest.approx(logdet, rel=0, abs=1e-12)



@pytest.mark.parametrize("q", [3, 4])
def test_simplex_centroid(q):
    expected = [[float(Fraction(1, len(s))) if i in s else 0 for i in range(q)]
                for n in range(1, q + 1) for s in combinations(range(q), n)]
    actual = simplex_centroid(q)
    assert actual.shape == (2**q - 1, q)
    assert actual.tolist() == expected


@pytest.mark.parametrize("lower,upper,expected", [
    ([.2,.1,.1], [.6,.5,.6], [
        ("1/5","1/5","3/5"), ("1/5","1/2","3/10"), ("3/10","1/10","3/5"),
        ("2/5","1/2","1/10"), ("3/5","1/10","3/10"), ("3/5","3/10","1/10"),
        ("1/5","7/20","9/20"), ("1/4","3/20","3/5"), ("3/10","1/2","1/5"),
        ("9/20","1/10","9/20"), ("1/2","2/5","1/10"), ("3/5","1/5","1/5"),
        ("23/60","17/60","1/3")]),
    ([.2,.2,0], [1,1,1], [
        ("1/5","1/5","3/5"), ("1/5","4/5","0"), ("4/5","1/5","0"),
        ("1/5","1/2","3/10"), ("1/2","1/5","3/10"), ("1/2","1/2","0"),
        ("2/5","2/5","1/5")]),
])
def test_mixture_exact_gate_rows(lower, upper, expected):
    assert mixture_vertices(lower, upper, face_dims=(1,2)).tolist() == [
        [float(Fraction(x)) for x in row] for row in expected]


def test_mixture_r3_faces():
    lower, upper = [.1]*4, [.6,.5,.4,.3]
    counts = [len(mixture_vertices(lower, upper))]
    counts += [len(mixture_vertices(lower, upper, face_dims=(d,))) - counts[0] for d in (1,2,3)]
    assert counts == [11,17,8,1]
    assert mixture_vertices(lower, upper, face_dims=(1,2,3))[-1].tolist() == [
        float(Fraction(x)) for x in ("18/55","14/55","13/55","2/11")]


def test_mixture_collapsed_and_empty():
    assert mixture_vertices([0,.1,.9], [1,1,1], face_dims=(1,2)).tolist() == [[0,.1,.9]]
    with pytest.raises(ValueError):
        mixture_vertices([.5]*3, [1]*3)


@pytest.mark.parametrize("model,degree", [("linear",1), ("quadratic",2), ("special_cubic",3)])
def test_scheffe_columns_and_grad(model, degree):
    x = torch.rand(5,4,dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    x = (x/x.sum(1,keepdim=True)).requires_grad_()
    expected = torch.stack([x[:,list(s)].prod(1) for d in range(1,degree+1)
                            for s in combinations(range(4),d)],1)
    actual = scheffe_features(x,model)
    assert torch.equal(actual,expected)
    actual.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("n,fixed", [(6,()), (7,()), (8,(12,13))])
def test_scheffe_exhaustive_optima(n, fixed):
    f = scheffe_features(mixture_vertices([.2,.1,.1],[.6,.5,.6],face_dims=(1,2)))
    if fixed: f = torch.cat((f, f[12:13]))
    b = f.T@f/len(f)
    best = dict(D=-math.inf,A=math.inf,I=math.inf)
    rows = list(combinations(range(12) if fixed else range(13),n-len(fixed)))
    for offset in range(0,len(rows),2048):
        ids = torch.tensor([list(fixed)+list(r) for r in rows[offset:offset+2048]])
        x = f[ids]; m = x.mT@x
        # Dense independent enumeration; singular subsets are infeasible.
        sign, ld = torch.linalg.slogdet(m)
        good = (sign > 0) & (torch.linalg.eigvalsh(m)[:,0] > 1e-12)
        m,ld = m[good],ld[good]
        if not len(m): continue
        inv = torch.linalg.inv(m)
        best['D'] = max(best['D'],float(ld.max()))
        best['A'] = min(best['A'],float(inv.diagonal(dim1=-2,dim2=-1).sum(-1).min()))
        best['I'] = min(best['I'],float((inv*b.T).sum((-2,-1)).min()))
    for c in ('D',) if fixed else ('D','A','I'):
        result = optimal_design(f,n,criterion=c,fixed_ids=list(fixed),
                                generator=torch.Generator().manual_seed(0))
        x = f[result['ids']]; m = x.T@x
        value = float(torch.linalg.slogdet(m).logabsdet) if c=='D' else float(
            torch.trace(torch.linalg.inv(m) if c=='A' else torch.linalg.inv(m)@b))
        assert value == pytest.approx(best[c],rel=1e-9)
        assert float(result['efficiency_lower_bound']) <= 1+1e-12
        ids = result['ids'].tolist()
        assert len(ids) == len(set(ids)) == n
        assert set(fixed) <= set(ids)


@pytest.mark.parametrize("q,model,P,n,fixed,values", [
    (3,"linear",[[1.,-1],[1,1]],7,[],[96,17/6,139/216]),
    (3,"linear",[[1.,-1],[1,1]],8,[],[144,8/3,451/756]),
    (3,"linear",[[1.,-1],[1,1]],8,[12,13],[1024/9,11/4,17/28]),
    (2,"quadratic",[[1.,-1],[1,0],[1,1]],7,[],[3/8,70/3,7/9])])
def test_mixture_process_optima(q, model, P, n, fixed, values):
    from torchoed import mixture_process_features
    f = mixture_process_features(scheffe_features(simplex_centroid(q),model),torch.tensor(P,dtype=torch.float64))
    for c, value in zip(('D','A','I'), values):
        r = optimal_design(f,n,criterion=c,fixed_ids=fixed,restarts=20,generator=torch.Generator().manual_seed(0))
        assert r['objective'] == pytest.approx(math.log(value) if c=='D' else -value,rel=1e-9)
        assert r['efficiency_lower_bound'] <= 1+1e-12
        assert set(fixed) <= set(r['ids'].tolist())
        assert len(set(r['ids'].tolist())) == n


def test_mixture_process_rows_information_and_boundary():
    from torchoed import mixture_process_features, ff2n
    s = scheffe_features(mixture_vertices([.2,.1,.1],[.6,.5,.6],face_dims=(1,2)))
    p = polynomial_features(ff2n(2),1); f = mixture_process_features(s,p)
    assert f.shape == (52,18)
    assert all(torch.equal(f[i*4+j],torch.kron(s[i],p[j])) for i in range(13) for j in range(4))
    torch.testing.assert_close(f.T@f,torch.kron(s.T@s,p.T@p),rtol=1e-12,atol=1e-12)
    ld = lambda m: torch.linalg.slogdet(m).logabsdet
    torch.testing.assert_close(ld(f.T@f),3*ld(s.T@s)+6*ld(p.T@p),rtol=1e-12,atol=1e-12)
    with pytest.raises(ValueError): mixture_process_features(s,p.float())
    with pytest.raises(TypeError): mixture_process_features(s,[[1.]])
    with pytest.raises(ValueError): mixture_process_features(s,p,max_elements=935)


def test_mixture_constraints_box_redundant():
    bounds = ([.2, .1, .1], [.6, .5, .6])
    expected = mixture_vertices(*bounds, face_dims=(1, 2))
    assert torch.equal(expected, mixture_vertices(*bounds, constraints=(), face_dims=(1, 2)))
    assert torch.equal(expected, mixture_vertices(*bounds, constraints=[([1, 1, 0], 0, 1)], face_dims=(1, 2)))
    bounds = ([.2, 0, 0], [.2, 1, 1])
    expected = mixture_vertices(*bounds, face_dims=(1, 2))
    assert expected.shape == (3, 3)  # two vertices, one edge; no 2-face
    assert torch.equal(expected, mixture_vertices(*bounds, constraints=[([1, 0, 0], 0, None)], face_dims=(1, 2)))


def test_mixture_constraints_hand_solved():
    expected = [('0','1/5','4/5'), ('0','4/5','1/5'), ('1/5','0','4/5'), ('4/5','0','1/5'),
                ('0','1/2','1/2'), ('1/10','1/10','4/5'), ('2/5','2/5','1/5'), ('1/2','0','1/2'),
                ('1/4','1/4','1/2')]
    actual = mixture_vertices([0]*3, [1]*3, constraints=[([1, 1, 0], '1/5', '4/5')], face_dims=(1, 2))
    assert actual.tolist() == [[float(Fraction(x)) for x in row] for row in expected]


def test_mixture_constraints_vertex_oracle():
    # Independent Cramer's rule and determinant minors; no production solve/rank.
    from itertools import permutations
    def det(a):
        return sum((-1)**sum(p[i] > p[j] for i in range(len(p)) for j in range(i+1, len(p))) *
                   math.prod(a[i][p[i]] for i in range(len(p))) for p in permutations(range(len(a))))
    one = [Fraction(1)]*4
    normals = [[Fraction(i == j) for i in range(4)] for j in range(4)]
    halfspaces = [(a, Fraction(b), sign) for a in normals for b, sign in ((0, 1), (1, -1))]
    halfspaces += [(list(map(Fraction, a)), Fraction(b), sign) for a, b, sign in
                   (([1, 1, 0, 0], '1/4', 1), ([1, 1, 0, 0], '3/4', -1), ([0, 1, 1, 0], '1/2', -1))]
    feasible = lambda v: all(sign*(sum(x*y for x,y in zip(a,v))-b) >= 0 for a,b,sign in halfspaces)
    expected = set()
    for subset in combinations(halfspaces, 3):
        a = [one] + [row for row, _, _ in subset]
        denominator = det(a)
        if not denominator:
            continue
        rhs = [Fraction(1)] + [b for _, b, _ in subset]
        v = tuple(det([[rhs[i] if k == j else x for k,x in enumerate(row)] for i,row in enumerate(a)]) / denominator for j in range(4))
        if feasible(v):
            expected.add(v)
    actual = mixture_vertices([0]*4, [1]*4, constraints=[([1,1,0,0], '1/4', '3/4'), ([0,1,1,0], None, '1/2')])
    assert actual.tolist() == [[float(x) for x in v] for v in sorted(expected)]
    for row in actual.tolist():
        v = tuple(Fraction(x).limit_denominator(1000) for x in row)
        assert v in expected and sum(v) == 1 and feasible(v)
        active = [a for a,b,_ in halfspaces if sum(x*y for x,y in zip(a,v)) == b]
        assert any(det([one] + list(s)) != 0 for s in combinations(active, 3))
