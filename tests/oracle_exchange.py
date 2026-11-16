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


def linear_value(F, ids, criterion, blocks=None):
    """Direct Fraction inverse and trace, independent of exchange identities."""
    p = len(F[0])
    gram = lambda rows: [[sum(Fraction(F[r][i])*F[r][j] for r in rows) for j in range(p)] for i in range(p)]
    def centered_gram(groups):
        sums = [[sum(F[r][j] for r in group) for j in range(p)] for group in groups]
        raw = gram([r for group in groups for r in group])
        return [[raw[i][j] - sum(Fraction(s[i]*s[j], len(group)) for s, group in zip(sums, groups))
                 for j in range(p)] for i in range(p)]

    info = gram(ids) if blocks is None else centered_gram(blocks)
    M = [row+[Fraction(i==j) for j in range(p)] for i,row in enumerate(info)]
    for k in range(p):
        pivot = next((i for i in range(k,p) if M[i][k]),None)
        if pivot is None:
            return None
        M[k], M[pivot] = M[pivot], M[k]
        value = M[k][k]
        M[k] = [x/value for x in M[k]]
        for i in range(p):
            if i != k:
                factor = M[i][k]
                M[i] = [a-factor*b for a,b in zip(M[i],M[k])]
    if criterion == "A":
        B = [[Fraction(i==j) for j in range(p)] for i in range(p)]
    else:
        region = range(len(F))
        moment = gram(region) if blocks is None else centered_gram([region])
        B = [[x/len(F) for x in row] for row in moment]
    return sum(B[i][j]*M[j][p+i] for i in range(p) for j in range(p))
