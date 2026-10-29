"""Independent rational/integer Gram and Bareiss oracle, no torch imports."""
from fractions import Fraction
from itertools import combinations
from math import lcm


def exact_gram(rows, ids):
    selected = [[Fraction(str(x)) for x in rows[i]] for i in ids]
    p = len(rows[0])
    return [[sum(r[i]*r[j] for r in selected) for j in range(p)] for i in range(p)]


def bareiss_det(matrix):
    p = len(matrix)
    scale = lcm(*(x.denominator for r in matrix for x in r))
    a = [[int(x*scale) for x in r] for r in matrix]
    previous, sign = 1, 1
    for k in range(p-1):
        pivot = next((i for i in range(k,p) if a[i][k]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != k:
            a[k],a[pivot] = a[pivot],a[k]
            sign = -sign
        value = a[k][k]
        for i in range(k+1,p):
            for j in range(k+1,p):
                numerator = a[i][j]*value-a[i][k]*a[k][j]
                assert numerator % previous == 0
                a[i][j] = numerator//previous
            a[i][k] = 0
        previous = value
    return Fraction(sign*a[-1][-1],scale**p)


def pair_cases(rows, selected):
    eligible = [i for i in range(len(rows)) if i not in selected]
    for positions in combinations(range(len(selected)),2):
        for incoming in combinations(eligible,2):
            trial = list(selected)
            for i,j in zip(positions,incoming):
                trial[i] = j
            gram = exact_gram(rows,trial)
            yield positions,incoming,trial,gram,bareiss_det(gram)


TRAP = [[1,0],[0,1],['9/10','9/10'],['9/10','-9/10']]
INTEGER11 = [[1,x,x*x] for x in range(-5,6)]
