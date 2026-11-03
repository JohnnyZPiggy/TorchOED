"""Seeded feasible random starts followed by CPU D-optimal VND restarts."""
import torch

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
                d_efficiency_lower_bound=torch.tensor(-best['efficiency_gap_bound'] / p,
                                                     dtype=torch.float64).exp().item(),
                stop_reason=best['stop_reason'], certificate=best['certificate'],
                restart_objectives=objectives, best_restart=best_restart, starts=starts)


def _linear_scores(F, ids, B, criterion):
    M = F[ids].mT@F[ids]
    L = torch.linalg.cholesky(M)
    Z = torch.linalg.solve_triangular(L,F.mT,upper=False)
    if criterion == "A":
        Y = torch.linalg.solve_triangular(L.mT,Z,upper=True)
        Qs, diagonal = Y[:,ids].mT@Y, Y.square().sum(dim=0)
    else:
        HZ = (Z@Z.mT/len(F))@Z
        Qs, diagonal = Z[:,ids].mT@HZ, (Z*HZ).sum(dim=0)
    Gs, gd = Z[:,ids].mT@Z, Z.square().sum(dim=0)
    incoming = torch.tensor(sorted(set(range(len(F)))-set(ids.tolist())))
    u, v, c = Gs[:,ids].diagonal()[:,None], gd[incoming], Gs[:,incoming]
    alpha, beta, gamma = Qs[:,ids].diagonal()[:,None], diagonal[incoming], Qs[:,incoming]
    ratio = (1-u)*(1+v)+c.square()
    delta = ((1+v)*alpha-2*c*gamma-(1-u)*beta)/ratio
    phi = torch.trace(torch.cholesky_solve(B,L)).item()
    return M, phi, diagonal, incoming, ratio, delta


def _linear_exchange(F, ids, B, criterion, max_iter):
    from .exchange import _tolerance
    p = F.shape[1]
    for iteration in range(max_iter+1):
        M, phi, diagonal, incoming, ratio, delta = _linear_scores(F,ids,B,criterion)
        condition = torch.linalg.cond(M).item()
        if iteration == max_iter:
            reason = "MAX_ITER"
            break
        tolerance = _tolerance(phi,phi,condition,condition,p)
        moves = ((ratio>0)&(delta < -tolerance)).flatten().nonzero().flatten()
        if not len(moves):
            reason = "LOCAL_1_SWAP_NUMERICAL"
            break
        reason = "NUMERICAL_INCONCLUSIVE"
        for flat in moves[delta.flatten()[moves].argsort(stable=True)].tolist():
            position, index = divmod(flat,len(incoming))
            trial = ids.clone()
            trial[position] = incoming[index]
            X = F[trial]
            candidate = X.mT@X
            L, info = torch.linalg.cholesky_ex(candidate)
            if info.item():
                continue
            value = torch.trace(torch.cholesky_solve(B,L)).item()
            if phi-value > _tolerance(phi,value,condition,torch.linalg.cond(candidate).item(),p):
                ids = trial
                break
        else:
            break
    return dict(ids=ids,objective=-phi,efficiency_lower_bound=phi/(len(ids)*diagonal.max().item()),
                stop_reason=reason,certificate="LOCAL_1_NUMERICAL" if reason=="LOCAL_1_SWAP_NUMERICAL" else "NOT_CERTIFIED")


@torch.no_grad()
def optimal_design(F, n, *, criterion="D", restarts=10, generator=None,
                   max_iter=100, max_elements=1_000_000):
    """D VND or A/I one-swap restarts with criterion-specific efficiency floor."""
    if criterion not in ("D","A","I"):
        raise ValueError('criterion must be D, A or I')
    if criterion == "D":
        result = d_optimal_design(F,n,restarts=restarts,generator=generator,
                                  max_iter=max_iter,max_elements=max_elements)
        return dict(result,criterion="D",efficiency_lower_bound=result['d_efficiency_lower_bound'])
    m, p = F.shape
    if type(n) is not int or not m >= n >= p:
        raise ValueError("n must be an int with m>=n>=p")
    # Conservative simultaneous solve, gather and n-by-m score workspace; no m-by-m matrices.
    if 8*m*p+16*n*m+8*p*p > max_elements:
        raise ValueError('insufficient max_elements for linear-criterion workspace')
    B = torch.eye(p,dtype=F.dtype) if criterion=="A" else F.mT@F/m
    starts, objectives, failures, best = [], [], 0, None
    for restart in range(restarts):
        while True:
            ids = torch.randperm(m,generator=generator)[:n].sort().values
            M = exact_information(F,ids,max_elements=max_elements)
            try:
                d_optimality(M,max_elements=max_elements)
            except ValueError:
                failures += 1
                if failures >= 100*restarts:
                    raise ValueError('No feasible start after 100*restarts failed draws; candidate set may be rank-deficient for this n') from None
                continue
            break
        starts.append(ids.tolist())
        result = _linear_exchange(F,ids,B,criterion,max_iter)
        objectives.append(result['objective'])
        if best is None or result['objective']>best['objective']:
            best, best_restart = result, restart
    return dict(best,criterion=criterion,restart_objectives=objectives,
                best_restart=best_restart,starts=starts)
