from fractions import Fraction as Q
from itertools import combinations
import math
import pytest
import torch
from torchoed import full_factorial, polynomial_features, optimal_design


def determinant(matrix):
    A = [[Q(int(v)) for v in row] for row in matrix.tolist()]
    det = Q(1)
    for j in range(len(A)):
        pivot = next((i for i in range(j, len(A)) if A[i][j]), None)
        if pivot is None: return Q(0)
        if pivot != j:
            A[j], A[pivot] = A[pivot], A[j]; det = -det
        value = A[j][j]; det *= value
        for i in range(j+1, len(A)):
            ratio = A[i][j]/value
            A[i] = [a-ratio*b for a, b in zip(A[i], A[j])]
    return det


@pytest.mark.parametrize('models,weights,expected,count', [
    ([[0,1,2], [0,3,4]], [1,1], (90,10), 78),
    ([[0,1,2], [0,1,2,3,4,5]], [1,1], (140,256), 76),
    ([[0,1,2,5], [0,1,2,3,4]], [1,2], (560,64), 76),
])
def test_model_robust_fraction_oracle(models, weights, expected, count):
    F = polynomial_features(full_factorial([3,3]).double()-1)
    union = sorted(set(i for c in models for i in c))
    def metrics(ids):
        X = F[list(ids)]
        return tuple(determinant(X[:, c].T@X[:, c]) for c in models)
    def score(dets):
        return sum(w/sum(weights)/len(c)*math.log(float(d))
                   for c,w,d in zip(models, weights, dets))-math.log(6)
    feasible = [ids for ids in combinations(range(9),6)
                if determinant(F[list(ids)][:,union].T@F[list(ids)][:,union]) > 0]
    assert len(feasible) == count
    optimum = max(score(metrics(ids)) for ids in feasible)
    result = optimal_design(F,6,models=models,model_weights=weights,
                            generator=torch.Generator().manual_seed(0))
    assert metrics(result['ids'].tolist()) == expected
    assert result['objective'] == pytest.approx(optimum, abs=1e-12)
    assert 0 < result['efficiency_lower_bound'] <= math.exp(result['objective']-optimum)+1e-12
    assert len(result['model_efficiency_lower_bounds']) == len(models)
    assert all(0 < v <= 1+1e-12 for v in result['model_efficiency_lower_bounds'])
    assert result['certificate'] == 'LOCAL_MODEL_ROBUST_EXCHANGE'
    if count == 78:
        maxima = [max(metrics(ids)[k] for ids in feasible) for k in range(2)]
        assert maxima == [140,12] and all(a < b for a,b in zip(expected, maxima))
    if weights == [1,2]:
        with pytest.raises(ValueError): optimal_design(F,5,models=models)
    fixed = optimal_design(F,6,models=models,model_weights=weights,fixed_ids=[0],
                           generator=torch.Generator().manual_seed(0))
    assert 0 in fixed['ids'] and len(fixed['ids'].unique()) == 6
