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


def _linear_closed_terms(G, Q, incoming=None):
    if incoming is None:
        gaa,gab,gbb,gcc,w,gdd,gac,gad,gbc,gbd = G
        qaa,qab,qbb,qcc,qcd,qdd,qac,qad,qbc,qbd = Q
        u, v = 1+gcc, 1+gdd
        pi = u*v-w*w
        addition = (v*qcc+u*qdd-2*w*qcd)/pi
    else:
        gaa,gab,gbb,gac,gad,gbc,gbd = G
        qaa,qab,qbb,qac,qad,qbc,qbd = Q
        u,v,w,pi,qcc,qcd,qdd,addition = incoming
    e00 = (gac*v-gad*w)/pi
    e01 = (gad*u-gac*w)/pi
    e10 = (gbc*v-gbd*w)/pi
    e11 = (gbd*u-gbc*w)/pi
    t00 = gaa-1-e00*gac-e01*gad
    t01 = gab-e00*gbc-e01*gbd
    t11 = gbb-1-e10*gbc-e11*gbd
    r00 = qaa-2*(e00*qac+e01*qad)+e00**2*qcc+2*e00*e01*qcd+e01**2*qdd
    r01 = (qab-e00*qbc-e01*qbd-e10*qac-e11*qad
           +e00*e10*qcc+(e00*e11+e01*e10)*qcd+e01*e11*qdd)
    r11 = qbb-2*(e10*qbc+e11*qbd)+e10**2*qcc+2*e10*e11*qcd+e11**2*qdd
    return addition, t11*r00+t00*r11-2*t01*r01, t00*t11-t01*t01, pi


def _linear_screen(G, Q, idx, tolerance, cutoff, best, tables, flat):
    if tables is None:
        row = idx[:,(0,0,1,2,2,3,0,0,1,1)]
        col = idx[:,(0,1,1,2,3,3,2,3,2,3)]
        addition, numerator, tau, pi = _linear_closed_terms(G[row,col].unbind(1),Q[row,col].unbind(1))
    else:
        incoming, outgoing = tables
        out, local = flat//incoming.shape[1], flat%incoming.shape[1]
        gaa,gab,gbb,qaa,qab,qbb = outgoing[:,out].unbind(0)
        row, col = idx[:,(0,0,1,1)], idx[:,(2,3,2,3)]
        addition, numerator, tau, pi = _linear_closed_terms(
            (gaa,gab,gbb,*G[row,col].unbind(1)),(qaa,qab,qbb,*Q[row,col].unbind(1)),incoming[:,local].unbind(0))
    closed = addition+numerator/tau
    determinant = pi*tau
    band = (determinant.abs()<=1e-8) | ~torch.isfinite(determinant) | ~torch.isfinite(closed)
    # The state tolerance is exactly the gate's delta (exchange._tolerance).
    keep = (closed>tolerance-tolerance/2) & (closed>=best-tolerance)
    if cutoff is not None:
        keep &= closed<=-cutoff[0]+tolerance/2
    return keep | band


def _linear_gather4(Z, T, ids, pairs, criterion, J):
    idx = torch.cat((ids[pairs[:,:2]],pairs[:,2:]),dim=1)
    U, V = Z[:,idx].permute(1,2,0), T[:,idx].permute(1,2,0)
    K = U@U.mT+J
    Q = V@V.mT if criterion=='A' else U@V.mT
    legal = torch.linalg.det(K)>0
    gain = torch.linalg.solve(K[legal],Q[legal]).diagonal(dim1=-2,dim2=-1).sum(-1)
    return legal, gain


def _linear_eligible(pairs, gains, tolerance, cutoff):
    keep = gains>tolerance
    if cutoff is not None:
        row = pairs.new_tensor(cutoff[1])
        first = (pairs != row).int().argmax(dim=1)
        later = pairs.gather(1,first[:,None]).flatten() > row[first]
        keep &= (gains < -cutoff[0]) | ((gains == -cutoff[0]) & later)
    return keep


def _linear_prefix_limit(negative, outgoing, phi, tolerance, best):
    theta = min(max(tolerance,best)-tolerance,phi/2)
    shift = phi*theta/(phi-theta)
    return torch.searchsorted(negative,-outgoing-shift+tolerance,right=True)


def _linear_prefix_tiles(n, prefix, selected_scores, phi, tolerance, width, get_best):
    pool, negative, order = prefix
    q, count = len(pool), len(order)
    # At the capped threshold no pair can be pruned: preserve the original raw sequence.
    if -negative[-1]>=selected_scores.topk(2).values.sum()+phi-tolerance:
        offset = 0
        for pairs in _pair_tiles(n,pool.tolist(),width):
            yield pairs,torch.arange(offset,offset+len(pairs))
            offset += len(pairs)
        return
    ends = torch.arange(q).flip(0)[:-1].cumsum(0)
    i,k = torch.triu_indices(n,n,offset=1)
    outgoing = selected_scores[i]+selected_scores[k]
    start, previous = torch.zeros_like(i), None
    while True:
        best = get_best()
        if best!=previous:
            limits = _linear_prefix_limit(negative,outgoing,phi,tolerance,best)
            previous = best
        remaining = (limits-start).clamp_min(0)
        boundaries = remaining.cumsum(0)
        total = boundaries[-1].item()
        if not total:
            return
        flat = torch.arange(min(width,total))
        out = torch.searchsorted(boundaries,flat,right=True)
        rank = flat-(boundaries[out]-remaining[out])+start[out]
        local = order[rank]
        a = torch.searchsorted(ends,local,right=True)
        b = a+1+local-(ends[a]-(q-a-1))
        last = out[-1].item()
        start[:last],start[last] = limits[:last],rank[-1]+1
        yield torch.stack((i[out],k[out],pool[a],pool[b]),dim=1),out*count+local


def _linear_pair_scores(F, ids, criterion, max_elements, free_positions=None, *, tolerance=None, cutoff=None):
    m, p = F.shape
    free = torch.arange(len(ids)) if free_positions is None else free_positions.nonzero().flatten()
    incoming = sorted(set(range(m))-set(ids.tolist()))
    if len(free)<2 or len(incoming)<2:
        return
    fixed = 8*m*p+16*len(ids)*m+8*p*p
    capacity = (max_elements-fixed)//(32*p+256)
    if capacity<1:
        raise ValueError('insufficient max_elements for one linear pair tile')
    L = torch.linalg.cholesky(F[ids].mT@F[ids])
    Z = torch.linalg.solve_triangular(L,F.mT,upper=False)
    T = torch.linalg.solve_triangular(L.mT,Z,upper=True) if criterion=='A' else (Z@Z.mT/m)@Z
    J = torch.diag(F.new_tensor([-1.,-1.,1.,1.]))
    count, outgoing_count = len(incoming)*(len(incoming)-1)//2, len(free)*(len(free)-1)//2
    available_base = max_elements-fixed-2*len(incoming)**2-2*count-8*outgoing_count
    table_elements = 8*count+6*outgoing_count
    hoisted = table_elements<=available_base//2
    available = available_base-table_elements if hoisted else available_base
    tables = None
    screened_capacity = available//96
    screened = tolerance is not None and screened_capacity>=1 and max_elements-fixed>=6*count+8*outgoing_count
    if screened:
        # Closed temporaries die before gather4; retain at most two sets of row IDs.
        rescore_capacity = (available-8*screened_capacity)//(32*p+256)
        screened = rescore_capacity>=1
    if screened:
        capacity = screened_capacity
        diagonal = T.square().sum(0) if criterion=='A' else (Z*T).sum(0)
        B = torch.eye(p,dtype=F.dtype) if criterion=='A' else F.mT@F/m
        phi = torch.trace(torch.cholesky_solve(B,L)).item()
        pool = torch.tensor(incoming)
        a,b = torch.triu_indices(len(pool),len(pool),offset=1)
        values, order = (diagonal[pool[a]]+diagonal[pool[b]]).sort(descending=True,stable=True)
        prefix = pool,-values,order
        del a,b,values
        G = Z.mT@Z
        Q = T.mT@T if criterion=='A' else Z.mT@T
        if hoisted:
            c,d = torch.combinations(torch.tensor(incoming),r=2).mT
            u,v,w = 1+G[c,c],1+G[d,d],G[c,d]
            pi = u*v-w*w
            qcc,qcd,qdd = Q[c,c],Q[c,d],Q[d,d]
            incoming_table = torch.stack((u,v,w,pi,qcc,qcd,qdd,(v*qcc+u*qdd-2*w*qcd)/pi))
            a,b = torch.combinations(ids[free],r=2).mT
            outgoing_table = torch.stack((G[a,a],G[a,b],G[b,b],Q[a,a],Q[a,b],Q[b,b]))
            del a,b,c,d,u,v,w,pi,qcc,qcd,qdd
            tables = incoming_table,outgoing_table
    best = -float('inf')
    tiles = (_linear_prefix_tiles(len(free),prefix,diagonal[ids[free]],phi,tolerance,capacity,lambda:best)
             if screened else ((pairs,None) for pairs in _pair_tiles(len(free),incoming,capacity)))
    for pairs,flat in tiles:
        pairs[:,:2] = free[pairs[:,:2]]
        if screened:
            idx = torch.cat((ids[pairs[:,:2]],pairs[:,2:]),dim=1)
            pairs = pairs[_linear_screen(G,Q,idx,tolerance,cutoff,best,tables,flat)]
            del idx
        for chunk in pairs.split(rescore_capacity) if screened else (pairs,):
            legal, gain = _linear_gather4(Z,T,ids,chunk,criterion,J)
            if screened:
                eligible = _linear_eligible(chunk[legal],gain,tolerance,cutoff) & torch.isfinite(gain)
                if eligible.any():
                    best = max(best,gain[eligible].max().item())
            yield chunk[legal],gain


def _linear_pair_moves(F, ids, criterion, max_elements, free_positions, tolerance):
    cutoff = None
    while True:
        best = None
        for pairs,gains in _linear_pair_scores(F,ids,criterion,max_elements,free_positions,
                                             tolerance=tolerance,cutoff=cutoff):
            keep = _linear_eligible(pairs,gains,tolerance,cutoff)
            eligible = keep.nonzero().flatten()
            if len(eligible):
                index = eligible[gains[eligible].argmax()]
                tied = eligible[gains[eligible]==gains[index]]
                if len(tied)>1:
                    for column in (3,2,1,0):
                        tied = tied[pairs[tied,column].argsort(stable=True)]
                    index = tied[0]
                key = (-gains[index].item(),tuple(pairs[index].tolist()))
                if best is None or key<best:
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
