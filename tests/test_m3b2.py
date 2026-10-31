import itertools
import math
from fractions import Fraction

import torch

from torchoed import fedorov_exchange, k2_exchange
from oracle_exchange import determinant
from oracle_two_swap import TRAP


def test_S1_trap():
    F = torch.tensor([[float(Fraction(x)) for x in row] for row in TRAP], dtype=torch.float64)
    ids = torch.tensor([0, 1])
    assert fedorov_exchange(F, ids)['accepted_swaps'] == 0
    result = k2_exchange(F, ids)
    assert result['ids'].tolist() == [2, 3]
    assert [move['k'] for move in result['swap_history']] == [2]
    assert math.isclose(math.exp(result['final_objective'] - result['initial_objective']),
                        6561/2500, rel_tol=1e-12)
    assert result['stop_reason'] == result['certificate'] == 'LOCAL_2_NUMERICAL'


def test_S2_all_205_starts_exact_locality_and_dominance():
    rows = [[2,1,-2],[0,1,0],[-2,0,2],[-2,0,-2],[-1,1,-1],
            [-2,2,0],[1,0,-1],[1,-2,-1],[1,2,-2],[-1,2,-2]]
    # Existing Fraction oracle imports neither torch nor TorchOED.
    determinants = {s: determinant(rows, s) for s in itertools.combinations(range(10), 4)}
    valid = [s for s, value in determinants.items() if value > 0]
    assert len(valid) == 205
    optimum = max(determinants.values())
    F = torch.tensor(rows, dtype=torch.float64)
    global_count = 0
    nonlocal_endpoints = []
    for start in valid:
        ids = torch.tensor(start)
        result = k2_exchange(F, ids)
        endpoint = tuple(sorted(result['ids'].tolist()))
        value = determinants[endpoint]
        assert (math.log(float(optimum)) - result['final_objective']
                <= result['efficiency_gap_bound'] + 1e-12)
        if any(value < other for s, other in determinants.items()
               if len(set(s) - set(endpoint)) <= 2):
            nonlocal_endpoints.append((start, endpoint, result['stop_reason']))
        global_count += value == optimum
        one = fedorov_exchange(F, ids)
        assert value >= determinants[tuple(sorted(one['ids'].tolist()))]
        assert all(b > a for a, b in zip(result['objective_history'], result['objective_history'][1:]))
    assert not nonlocal_endpoints, (len(nonlocal_endpoints), global_count, nonlocal_endpoints[0])
    assert global_count == 145


def test_S7_empty_two_swap_neighborhood_after_phase1():
    for rows, selected in [([[1], [2], [3]], [0]),
                           ([[1,0], [0,1], [2,2]], [0,1])]:
        F = torch.tensor(rows, dtype=torch.float64)
        ids = torch.tensor(selected)
        one = fedorov_exchange(F, ids)
        result = k2_exchange(F, ids)
        assert result['stop_reason'] == 'NO_K2_NEIGHBORHOOD'
        assert result['certificate'] == 'NOT_CERTIFIED'
        assert result['ids'].tolist() == one['ids'].tolist()
        assert result['scanned_pairs']['k2'] == 0


def test_grid4_default_budget_and_fedorov_dominance():
    from torchoed import full_factorial, polynomial_features
    F = polynomial_features(full_factorial([3]*4).double()-1, degree=2)
    ids = torch.randperm(81, generator=torch.Generator().manual_seed(0))[:30].sort().values
    one = fedorov_exchange(F, ids)
    result = k2_exchange(F, ids)
    assert result['stop_reason'] == 'LOCAL_2_NUMERICAL'
    assert result['final_objective'] >= one['final_objective'] - 1e-12
