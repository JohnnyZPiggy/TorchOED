from itertools import combinations
import math

import torch

from torchoed import d_optimal_design
from oracle_exchange import determinant


ROWS = [[2,1,-2],[0,1,0],[-2,0,2],[-2,0,-2],[-1,1,-1],
        [-2,2,0],[1,0,-1],[1,-2,-1],[1,2,-2],[-1,2,-2]]


def test_quality_against_exact_global_oracle():
    optimum = max(determinant(ROWS, ids) for ids in combinations(range(10), 4))
    result = d_optimal_design(torch.tensor(ROWS, dtype=torch.float64), 4,
                             restarts=10, generator=torch.Generator().manual_seed(0))
    assert determinant(ROWS, result['ids'].tolist()) == optimum
    assert math.isclose(result['objective'], math.log(float(optimum)), abs_tol=1e-12, rel_tol=1e-12)
    assert result['objective'] == max(result['restart_objectives'])


def test_reproducibility_with_same_seed():
    F = torch.tensor(ROWS, dtype=torch.float64)
    first = d_optimal_design(F, 4, restarts=10, generator=torch.Generator().manual_seed(0))
    second = d_optimal_design(F, 4, restarts=10, generator=torch.Generator().manual_seed(0))
    assert torch.equal(first['ids'], second['ids'])
    assert first['starts'] == second['starts']
    assert first['restart_objectives'] == second['restart_objectives']
