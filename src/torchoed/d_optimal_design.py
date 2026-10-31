"""Seeded feasible random starts followed by CPU D-optimal VND restarts."""
import torch
from math import exp

from .criteria import d_optimality
from .information import exact_information
from .k2_exchange import k2_exchange


@torch.no_grad()
def d_optimal_design(F, n, *, restarts=10, generator=None, max_iter=100,
                     tile_size=64, max_elements=1_000_000):
    """Return the best restarted k2_exchange run; see docs/M3C_API.md."""
    m, p = F.shape
    if type(n) is not int or not m >= n >= p:
        raise ValueError("n must be an int with m>=n>=p")
    starts, objectives = [], []
    failures = 0
    best = None
    for restart in range(restarts):
        while True:
            ids = torch.randperm(m, generator=generator)[:n].sort().values
            M = exact_information(F, ids, max_elements=max_elements)
            try:
                d_optimality(M, max_elements=max_elements)
            except ValueError:
                failures += 1
                if failures >= 100 * restarts:
                    raise ValueError("No feasible start after 100*restarts failed draws; "
                                     "candidate set may be rank-deficient for this n") from None
                continue
            break
        starts.append(ids.tolist())
        result = k2_exchange(F, ids, max_iter=max_iter, tile_size=tile_size,
                             max_elements=max_elements)
        objectives.append(result['final_objective'])
        if best is None or result['final_objective'] > best['final_objective']:
            best, best_restart = result, restart
    return dict(ids=best['ids'], objective=best['final_objective'],
                efficiency_gap_bound=best['efficiency_gap_bound'],
                d_efficiency_lower_bound=exp(-best['efficiency_gap_bound'] / p),
                stop_reason=best['stop_reason'], certificate=best['certificate'],
                restart_objectives=objectives, best_restart=best_restart, starts=starts)
