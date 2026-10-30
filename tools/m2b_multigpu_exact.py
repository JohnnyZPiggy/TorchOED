"""Exact G3-C fixture algebra; independent stdlib Fraction only, no torch."""
from fractions import Fraction as F
import random


def product(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def transpose(a):
    return [list(row) for row in zip(*a)]


def solve_many(a, b):
    n = len(a)
    rows = [[F(x) for x in left+right] for left,right in zip(a,b)]
    for j in range(n):
        pivot = next(i for i in range(j,n) if rows[i][j])
        rows[j],rows[pivot] = rows[pivot],rows[j]
        scale = rows[j][j]
        rows[j] = [x/scale for x in rows[j]]
        for i in range(n):
            if i!=j:
                scale=rows[i][j]
                rows[i]=[x-scale*y for x,y in zip(rows[i],rows[j])]
    return [row[n:] for row in rows]


def cayley(p, seed):
    rng=random.Random(seed)
    s=[[F(0) for _ in range(p)] for _ in range(p)]
    for i in range(p):
        for j in range(i+1,p):
            s[i][j]=F(rng.randint(-3,3),32)
            s[j][i]=-s[i][j]
    identity=[[F(i==j) for j in range(p)] for i in range(p)]
    plus=[[identity[i][j]+s[i][j] for j in range(p)] for i in range(p)]
    minus=[[identity[i][j]-s[i][j] for j in range(p)] for i in range(p)]
    # I+S and I-S commute: left solve equals the stated right-sided Cayley transform.
    q=solve_many(plus,minus)
    assert product(transpose(q),q)==identity
    return q


def strict_rational_geometric(p, exponent):
    step=F(exponent,p-1)
    if step.denominator!=1:
        raise ValueError('RATIONAL_GEOMETRIC_SPECTRUM_UNREPRESENTABLE: clarification required')
    return [F(1,10**(step.numerator*j)) for j in range(p)]


def exact_case(q, spectrum, seed):
    p=len(q)
    assert spectrum[0]==1 and all(x>0 for x in spectrum)
    assert all(x>=y for x,y in zip(spectrum,spectrum[1:]))
    matrix=product([[q[i][j]*spectrum[j] for j in range(p)] for i in range(p)],transpose(q))
    rng=random.Random(seed)
    random_c=[F(rng.randint(-7,7),7) for _ in range(p)]
    if not any(random_c):random_c[0]=F(1)
    contrasts=[]
    for name,c in [('v_max',[row[0] for row in q]),('v_min',[row[-1] for row in q]),('random',random_c)]:
        coefficients=[sum(q[i][j]*c[i] for i in range(p)) for j in range(p)]
        quadratic=sum(a*a/lam for a,lam in zip(coefficients,spectrum))
        norm_solved=sum(a*a/(lam*lam) for a,lam in zip(coefficients,spectrum))
        if name!='random':
            assert sum(x*x for x in c)==1
            eigenvalue=spectrum[0] if name=='v_max' else spectrum[-1]
            assert quadratic==1/eigenvalue
            assert [sum(matrix[i][j]*c[j] for j in range(p)) for i in range(p)]==[eigenvalue*x for x in c]
        contrasts.append(dict(name=name,c=c,quadratic=quadratic,kappa_c=norm_solved/quadratic))
    return matrix,contrasts
