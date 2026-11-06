"""Seeded feasible random starts followed by CPU D-optimal VND restarts."""
import torch

from .criteria import d_optimality
from .information import exact_information, _features
from .k2_exchange import k2_exchange, _verified_information
from .two_swap import _pair_tiles


def _fixed_pool(fixed_ids, m, n):
    fixed = torch.as_tensor(fixed_ids)
    if (fixed.ndim != 1 or len(fixed) >= n or
            (len(fixed) and (fixed.dtype not in (torch.int32, torch.int64) or
             fixed.min() < 0 or fixed.max() >= m or len(fixed.unique()) != len(fixed)))):
        raise ValueError('fixed_ids must be distinct candidate IDs with length < n')
    fixed = fixed.to(dtype=torch.int64)
    return fixed, torch.tensor(sorted(set(range(m))-set(fixed.tolist())),dtype=torch.int64)


def _dual_floor(F, M, n, criterion, current):
    m, p = F.shape
    w = F.new_full((m,),1/m)
    B = torch.eye(p,dtype=F.dtype,device=F.device) if criterion=='A' else F.mT@F/m
    R = torch.linalg.cholesky(B).mT if criterion=='I' else None
    own = torch.linalg.slogdet(M/n).logabsdet if criterion=='D' else n*torch.trace(torch.linalg.solve(M,B))
    best = current
    for iteration in range(1000):
        L, info = torch.linalg.cholesky_ex(F.mT@(w[:,None]*F))
        Z = torch.linalg.solve_triangular(L,F.mT,upper=False)
        if criterion=='D':
            scores = Z.square().sum(0)
            ratio = p/scores.max()
            floor = ((own-2*L.diagonal().log().sum())/p).exp()*ratio
            update = scores/p
        else:
            Y = torch.linalg.solve_triangular(L.mT,Z,upper=True)
            scores = (Y if criterion=='A' else R@Y).square().sum(0)
            phi = torch.trace(torch.cholesky_solve(B,L))
            ratio = phi/scores.max()
            floor = phi*ratio/own
            update = scores.sqrt()
        info_v, floor_v, ratio_v = torch.stack((info.to(F.dtype),floor,ratio)).tolist()
        if info_v:
            break
        best = max(best,floor_v)
        if ratio_v>=1-1e-4:
            break
        w = w*update
        w = w/w.sum()
    return best


@torch.no_grad()
def d_optimal_design(F: torch.Tensor, n: int, *, fixed_ids: list[int] | tuple[int, ...] | torch.Tensor | None = None, restarts: int = 10, generator: torch.Generator | None = None, max_iter: int = 100,
                     tile_size: int = 64, max_elements: int = 1_000_000) -> dict[str, object]:
    """Return the best restarted k2_exchange run; see docs/M3C_API.md."""
    m, p = F.shape
    if type(n) is not int or not m >= n >= p:
        raise ValueError("n must be an int with m>=n>=p")
    _features(F,max_elements)
    if fixed_ids is not None:
        fixed, pool = _fixed_pool(fixed_ids,m,n)
    starts, objectives = [], []
    failures = 0
    best = None
    for restart in range(restarts):
        while True:
            if fixed_ids is None:
                ids = torch.randperm(m, generator=generator)[:n].sort().values
            else:
                ids = torch.cat((fixed,pool[torch.randperm(len(pool),generator=generator)[:n-len(fixed)]])).sort().values
            if _verified_information(F,ids) is None:
                failures += 1
                if failures >= 100 * restarts:
                    raise ValueError("No feasible start after 100*restarts failed draws; "
                                     "candidate set may be rank-deficient for this n") from None
                continue
            break
        starts.append(ids.tolist())
        result = k2_exchange(F, ids, max_iter=max_iter, tile_size=tile_size,
                             max_elements=max_elements,
                             _free_positions=None if fixed_ids is None else ~torch.isin(ids,fixed))
        objectives.append(result['final_objective'])
        if best is None or result['final_objective'] > best['final_objective']:
            best, best_restart = result, restart
    current = torch.tensor(-best['efficiency_gap_bound']/p,dtype=torch.float64).exp().item()
    floor = _dual_floor(F,F[best['ids']].mT@F[best['ids']],n,'D',current)
    return dict(ids=best['ids'], objective=best['final_objective'],
                efficiency_gap_bound=best['efficiency_gap_bound'],
                d_efficiency_lower_bound=floor,
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


def _linear_pair_scores(F, ids, criterion, max_elements, free_positions=None):
    m, p = F.shape
    free = torch.arange(len(ids)) if free_positions is None else free_positions.nonzero().flatten()
    incoming = sorted(set(range(m))-set(ids.tolist()))
    if len(free)<2 or len(incoming)<2:
        return
    capacity = (max_elements-(8*m*p+16*len(ids)*m+8*p*p))//(32*p+256)
    if capacity<1:
        raise ValueError('insufficient max_elements for one linear pair tile')
    L = torch.linalg.cholesky(F[ids].mT@F[ids])
    Z = torch.linalg.solve_triangular(L,F.mT,upper=False)
    T = torch.linalg.solve_triangular(L.mT,Z,upper=True) if criterion=='A' else (Z@Z.mT/m)@Z
    J = torch.diag(F.new_tensor([-1.,-1.,1.,1.]))
    for pairs in _pair_tiles(len(free),incoming,capacity):
        pairs[:,:2] = free[pairs[:,:2]]
        idx = torch.cat((ids[pairs[:,:2]],pairs[:,2:]),dim=1)
        U, V = Z[:,idx].permute(1,2,0), T[:,idx].permute(1,2,0)
        K = U@U.mT+J
        Q = V@V.mT if criterion=='A' else U@V.mT
        legal = torch.linalg.det(K)>0
        gain = torch.linalg.solve(K[legal],Q[legal]).diagonal(dim1=-2,dim2=-1).sum(-1)
        yield pairs[legal],gain


def _linear_pair_moves(F, ids, criterion, max_elements, free_positions, tolerance):
    cutoff = None
    while True:
        best = None
        for pairs,gains in _linear_pair_scores(F,ids,criterion,max_elements,free_positions):
            keep = gains>tolerance
            for row,gain in zip(pairs[keep].tolist(),gains[keep].tolist()):
                key = (-gain,tuple(row))
                if (cutoff is None or key>cutoff) and (best is None or key<best):
                    best = key
        if best is None:
            return
        # A failed direct verification resumes the exact ordering with bounded memory.
        cutoff = best
        yield list(best[1][:2]),list(best[1][2:])


def _linear_exchange(F, ids, B, criterion, max_iter, *, _free_positions=None, max_elements=1_000_000):
    from .exchange import _tolerance
    p = F.shape[1]
    history = []
    for iteration in range(max_iter+1):
        M, phi, diagonal, incoming, ratio, delta = _linear_scores(F,ids,B,criterion)
        condition = torch.linalg.cond(M).item()
        if iteration == max_iter:
            reason = "MAX_ITER"
            break
        tolerance = _tolerance(phi,phi,condition,condition,p)
        legal = (ratio>0)&(delta < -tolerance)
        if _free_positions is not None:
            legal &= _free_positions[:,None]
        moves = legal.flatten().nonzero().flatten()
        if len(moves):
            candidates = (([flat//len(incoming)],[incoming[flat%len(incoming)].item()])
                          for flat in moves[delta.flatten()[moves].argsort(stable=True)].tolist())
            reason = "NUMERICAL_INCONCLUSIVE"
        else:
            candidates = _linear_pair_moves(F,ids,criterion,max_elements,_free_positions,tolerance)
            reason = "LOCAL_2_NUMERICAL"
        for positions,incoming_ids in candidates:
            reason = "NUMERICAL_INCONCLUSIVE"
            trial = ids.clone()
            trial[positions] = trial.new_tensor(incoming_ids)
            X = F[trial]
            candidate = X.mT@X
            L, info = torch.linalg.cholesky_ex(candidate)
            if info.item():
                continue
            value = torch.trace(torch.cholesky_solve(B,L)).item()
            if phi-value > _tolerance(phi,value,condition,torch.linalg.cond(candidate).item(),p):
                history.append(dict(k=len(positions),outgoing_ids=ids[positions].tolist(),incoming_ids=incoming_ids))
                ids = trial
                break
        else:
            break
    return dict(ids=ids,objective=-phi,efficiency_lower_bound=phi/(len(ids)*diagonal.max().item()),
                stop_reason=reason,certificate="LOCAL_2_NUMERICAL" if reason=="LOCAL_2_NUMERICAL" else "NOT_CERTIFIED",swap_history=history)


@torch.no_grad()
def optimal_design(F: torch.Tensor, n: int, *, criterion: str = "D", fixed_ids: list[int] | tuple[int, ...] | torch.Tensor | None = None, restarts: int = 10, generator: torch.Generator | None = None,
                   max_iter: int = 100, max_elements: int = 1_000_000) -> dict[str, object]:
    """D/A/I VND restarts with criterion-specific efficiency floor."""
    if criterion not in ("D","A","I"):
        raise ValueError('criterion must be D, A or I')
    if criterion == "D":
        result = d_optimal_design(F,n,fixed_ids=fixed_ids,restarts=restarts,generator=generator,
                                  max_iter=max_iter,max_elements=max_elements)
        return dict(result,criterion="D",efficiency_lower_bound=result['d_efficiency_lower_bound'])
    m, p = F.shape
    if type(n) is not int or not m >= n >= p:
        raise ValueError("n must be an int with m>=n>=p")
    if fixed_ids is not None:
        fixed, pool = _fixed_pool(fixed_ids,m,n)
    # Conservative simultaneous solve, gather and n-by-m score workspace; no m-by-m matrices.
    if 8*m*p+16*n*m+8*p*p > max_elements:
        raise ValueError('insufficient max_elements for linear-criterion workspace')
    B = torch.eye(p,dtype=F.dtype) if criterion=="A" else F.mT@F/m
    starts, objectives, failures, best = [], [], 0, None
    for restart in range(restarts):
        while True:
            if fixed_ids is None:
                ids = torch.randperm(m,generator=generator)[:n].sort().values
            else:
                ids = torch.cat((fixed,pool[torch.randperm(len(pool),generator=generator)[:n-len(fixed)]])).sort().values
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
        result = _linear_exchange(F,ids,B,criterion,max_iter,
                                  _free_positions=None if fixed_ids is None else ~torch.isin(ids,fixed),max_elements=max_elements)
        objectives.append(result['objective'])
        if best is None or result['objective']>best['objective']:
            best, best_restart = result, restart
    best['efficiency_lower_bound'] = _dual_floor(F,F[best['ids']].mT@F[best['ids']],n,criterion,best['efficiency_lower_bound'])
    return dict(best,criterion=criterion,restart_objectives=objectives,
                best_restart=best_restart,starts=starts)
