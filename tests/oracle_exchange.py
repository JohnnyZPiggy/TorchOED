"""Tiny independent exact reference; never imports TorchOED or torch."""
from fractions import Fraction
from itertools import combinations


def determinant(F, ids):
    rows = [[Fraction(str(x)) for x in F[i]] for i in ids]
    p = len(F[0])
    M = [[sum(r[i]*r[j] for r in rows) for j in range(p)] for i in range(p)]
    result = Fraction(1)
    for k in range(p):
        pivot = next((i for i in range(k, p) if M[i][k]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != k:
            M[k], M[pivot] = M[pivot], M[k]
            result = -result
        value = M[k][k]
        result *= value
        for i in range(k+1, p):
            factor = M[i][k]/value
            M[i] = [a-factor*b for a,b in zip(M[i],M[k])]
    return result


def neighborhood(F, ids):
    result = []
    for i in range(len(ids)):
        for j in range(len(F)):
            if j not in ids:
                trial = list(ids)
                trial[i] = j
                result.append((i,j,determinant(F,trial)))
    return result


def exhaustive(F, n):
    if len(F) > 8:
        raise ValueError('tiny exact oracle limited to m<=8')
    return [(subset,determinant(F,subset)) for subset in combinations(range(len(F)),n)]
