from fractions import Fraction as Q
from itertools import combinations, product
import math
import pytest
import torch
from torchoed import optimal_design


def exact_metrics(rows):
    p = 6
    M = [[sum(r[i]*r[j] for r in rows) + Q(i == j and i >= 3)
          for j in range(p)] for i in range(p)]
    augmented = [r + [Q(i == j) for j in range(p)] for i, r in enumerate(M)]
    det = Q(1)
    for j in range(p):
        pivot = next((i for i in range(j, p) if augmented[i][j]), None)
        if pivot is None:
            return None
        if pivot != j:
            augmented[j], augmented[pivot] = augmented[pivot], augmented[j]
            det = -det
        value = augmented[j][j]
        det *= value
        augmented[j] = [v/value for v in augmented[j]]
        for i in range(p):
            if i != j:
                value = augmented[i][j]
                augmented[i] = [a-value*b for a,b in zip(augmented[i], augmented[j])]
    W = [[sum(r[i]*r[j] for r in SCALED)/9 for j in range(p)] for i in range(p)]
    return det, sum(augmented[i][p+j]*W[j][i] for i in range(p) for j in range(p))


GRID = list(product((-1, 0, 1), repeat=2))
SCALED = [[Q(1), Q(x), Q(y), Q(x*y), 2*(Q(x*x)-Q(2,3)),
           2*(Q(y*y)-Q(2,3))] for x,y in GRID]


@pytest.mark.parametrize('criterion', ['D', 'I'])
def test_bayesian_fraction_optimum(criterion):
    F = torch.tensor([[1,x,y,x*y,x*x,y*y] for x,y in GRID], dtype=torch.float64)
    result = optimal_design(F, 4, criterion=criterion, if_possible=[3,4,5],
                            generator=torch.Generator().manual_seed(0))
    expected = torch.tensor([[float(v) for v in r] for r in SCALED], dtype=F.dtype)
    torch.testing.assert_close(result['scaled_features'], expected, atol=1e-14, rtol=1e-14)
    metrics = [v for ids in combinations(range(9),4)
               if (v := exact_metrics([SCALED[i] for i in ids])) is not None]
    selected = exact_metrics([SCALED[i] for i in result['ids'].tolist()])
    optimum = max(v[0] for v in metrics) if criterion == 'D' else min(v[1] for v in metrics)
    assert selected[0 if criterion == 'D' else 1] == optimum
    target = math.log(float(optimum)) if criterion == 'D' else float(optimum)
    assert result['objective'] == pytest.approx(target, abs=1e-12)
    efficiency = (float(selected[0]/optimum)**(1/6) if criterion == 'D'
                  else float(optimum/selected[1]))
    assert 0 <= result['efficiency_lower_bound'] <= efficiency + 1e-12
    fixed = optimal_design(F, 4, criterion=criterion, if_possible=[3,4,5], fixed_ids=[0],
                           generator=torch.Generator().manual_seed(0))
    assert 0 in fixed['ids'] and len(fixed['ids'].unique()) == 4
    F[:,3] = F[:,1] + 0.1*F[:,2]
    with pytest.raises(ValueError, match='span'):
        optimal_design(F, 4, criterion=criterion, if_possible=[3,4,5])
