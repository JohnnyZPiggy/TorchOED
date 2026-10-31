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


def test_S9_pruned_top8_against_public_scores(monkeypatch):
    from torchoed import full_factorial, polynomial_features, score_two_swaps
    from torchoed.k2_exchange import _two_candidates, _ordered_top, _keep_top, _tolerance
    rows = [[2,1,-2],[0,1,0],[-2,0,2],[-2,0,-2],[-1,1,-1],
            [-2,2,0],[1,0,-1],[1,-2,-1],[1,2,-2],[-1,2,-2]]
    starts = [s for s in itertools.combinations(range(10),4) if determinant(rows,s)>0]
    assert len(starts) == 205
    grid = polynomial_features(full_factorial([3]*3).double()-1, degree=2)
    seeded = torch.randperm(27, generator=torch.Generator().manual_seed(0))[:20].sort().values
    native_det = torch.linalg.det
    for F, selections, pruning in [(torch.tensor(rows,dtype=torch.float64), starts, False),
                                    (grid, [seeded.tolist()], True)]:
        for selection in selections:
            ids = torch.tensor(selection)
            incoming = torch.tensor([j for j in range(len(F)) if j not in selection])
            M = F[ids].mT @ F[ids]
            Z = torch.linalg.solve_triangular(torch.linalg.cholesky(M), F.mT, upper=False)
            condition = torch.linalg.cond(M).item()
            tol = _tolerance(1.,1.,condition,condition,F.shape[1])
            oracle = score_two_swaps(F,ids)
            rows4 = torch.cat((oracle['outgoing_positions'],oracle['incoming_ids']),dim=1)
            def reference(tolerance, anchor=None):
                top = []
                _keep_top(top,oracle['ratios'],rows4,tolerance,anchor)
                return top
            expected, _ = _ordered_top(reference,tol)
            seen = [0]
            def counted(block):
                seen[0] += len(block)
                return native_det(block)
            with monkeypatch.context() as patch:
                patch.setattr(torch.linalg,'det',counted)
                actual, passes = _ordered_top(_two_candidates,Z,ids,incoming,
                    Z[:,ids].mT@Z[:,ids],Z[:,ids].mT@Z[:,incoming],64,1024,tol)
            assert len(actual) == len(expected)
            ratios = dict(zip(map(tuple,rows4.tolist()),oracle['ratios'].tolist()))
            torch.testing.assert_close(torch.tensor([g for g,_ in actual],dtype=torch.float64),
                torch.tensor([ratios[row] for _,row in actual],dtype=torch.float64),atol=1e-9,rtol=1e-9)
            # The 8th place can be an exact tie.
            difference = {row for _,row in actual} ^ {row for _,row in expected}
            if difference:
                cutoff = min(g for g,_ in expected)
                band = tol*max(1.,abs(cutoff))
                assert all(abs(ratios[row]-cutoff) <= band for row in difference)
            if pruning:
                assert seen[0] < passes*oracle['evaluated_pairs']


def test_S10_saturated_top8_against_public_scores(monkeypatch):
    from torchoed import full_factorial, polynomial_features, score_two_swaps
    from torchoed.k2_exchange import _two_candidates, _ordered_top, _keep_top, _tolerance
    grid = polynomial_features(full_factorial([3]*3).double()-1, degree=2)
    random = torch.randn(40,6,dtype=torch.float64,generator=torch.Generator().manual_seed(0))
    native_det, seen = torch.linalg.det, [0]
    def counted(block):
        seen[0] += len(block)
        return native_det(block)
    monkeypatch.setattr(torch.linalg,'det',counted)
    for F in (grid,random):
        p = F.shape[1]
        ids = torch.randperm(len(F),generator=torch.Generator().manual_seed(0))[:p].sort().values
        incoming = torch.tensor(sorted(set(range(len(F)))-set(ids.tolist())))
        M = F[ids].mT@F[ids]
        Z = torch.linalg.solve_triangular(torch.linalg.cholesky(M),F.mT,upper=False)
        condition = torch.linalg.cond(M).item()
        tol = _tolerance(1.,1.,condition,condition,p)
        oracle = score_two_swaps(F,ids)
        rows4 = torch.cat((oracle['outgoing_positions'],oracle['incoming_ids']),dim=1)
        def reference(tolerance,anchor=None):
            top = []
            _keep_top(top,oracle['ratios'],rows4,tolerance,anchor)
            return top
        expected, _ = _ordered_top(reference,tol)
        seen[0] = 0
        actual, passes = _ordered_top(_two_candidates,Z,ids,incoming,
            Z[:,ids].mT@Z[:,ids],Z[:,ids].mT@Z[:,incoming],64,1024,tol)
        assert len(actual) == len(expected)
        ratios = dict(zip(map(tuple,rows4.tolist()),oracle['ratios'].tolist()))
        torch.testing.assert_close(torch.tensor([g for g,_ in actual],dtype=torch.float64),
            torch.tensor([ratios[row] for _,row in actual],dtype=torch.float64),atol=1e-9,rtol=1e-9)
        # The 8th place can be an exact tie.
        difference = {row for _,row in actual} ^ {row for _,row in expected}
        if difference:
            cutoff = min(g for g,_ in expected)
            assert all(abs(ratios[row]-cutoff)<=tol*max(1.,abs(cutoff)) for row in difference)
        assert seen[0] < passes*oracle['evaluated_pairs']
