"""G-based, score-first / verify-top deterministic k=2 VND."""
import torch

from ._validation import integer
from .criteria import d_optimality
from .exchange import _tolerance
from .information import exact_information


def _keep_top(top, ratios, rows, ratio_tolerance, anchor):
    eligible = ((ratios - 1) > ratio_tolerance * ratios.abs().clamp_min(1)).nonzero().flatten()
    if anchor is None:
        chosen = eligible[ratios[eligible].argsort(descending=True, stable=True)[:8]]
        key = lambda item: (-item[0], item[1])
    else:
        cutoff = anchor - ratio_tolerance * max(1., abs(anchor))
        band = ratios[eligible] >= cutoff
        lower = eligible[~band]
        chosen = torch.cat((eligible[band][:8],
                            lower[ratios[lower].argsort(descending=True, stable=True)[:8]]))
        key = lambda item: (0 if item[0] >= cutoff else 1,
                            0 if item[0] >= cutoff else -item[0], item[1])
    top.extend(zip(ratios[chosen].tolist(), map(tuple, rows[chosen].tolist())))
    top.sort(key=key)
    del top[8:]


def _ordered_top(scorer, *args):
    top = scorer(*args)
    if len(top) > 1 and top[0][0] - top[1][0] <= args[-1] * max(1., abs(top[0][0])):
        # A second bounded scan anchors near-ties to the global maximum.
        return scorer(*args, anchor=top[0][0]), 2
    return top, 1


def _one_candidates(Gss, Gsn, diagonal, incoming, width, tolerance, anchor=None, *, free_positions=None):
    top = []
    n = Gss.shape[0]
    for start in range(0, incoming.numel(), width):
        end = min(start + width, incoming.numel())
        ratios = ((1 - Gss.diagonal()[:, None]) * (1 + diagonal[incoming[start:end]])
                  + Gsn[:, start:end].square())
        positions = torch.arange(n, device=incoming.device).repeat_interleave(end-start)
        rows = torch.stack((positions, incoming[start:end].repeat(n)), dim=1)
        if free_positions is not None:
            keep = free_positions[positions]
            ratios, rows = ratios.flatten()[keep], rows[keep]
        _keep_top(top, ratios.flatten(), rows, tolerance, anchor)
    return top


def _pruned_candidates(top, Z, current, incoming, Gss, Gsn, outgoing, rho,
                       capacity, tolerance, anchor):
    q = incoming.numel()
    diagonal = Z[:, incoming].square().sum(dim=0)
    indices = torch.arange(q, device=Z.device)
    # Reserve half the tile budget for prefixes.
    batch = max(1, capacity*128 // (32*q))
    # Closed-form survivors: three p-sized gathers/products plus scalar/index slack.
    capacity = max(1, capacity*64 // (4*Z.shape[0]+64))
    for start in range(0, len(outgoing), batch):
        O, r = outgoing[start:start+batch], rho[start:start+batch, None]
        a, b = O.mT
        x, y = Gsn[a], Gsn[b]
        if len(current) == Z.shape[0]:
            w = x.square()+y.square()
        else:
            w = (r*(1+diagonal) + (1-Gss[b,b,None])*x.square()
                 + (1-Gss[a,a,None])*y.square() + 2*Gss[a,b,None]*x*y)
        values, order = w.sort(dim=1, descending=True, stable=True)
        ends = torch.searchsorted(-values, -(1+tolerance/2)*r/values)
        counts = (ends-indices-1).clamp_min(0)
        cumulative = counts.flatten().cumsum(0)
        total = cumulative[-1].item()
        for offset in range(0, total, capacity):
            flat = torch.arange(offset, min(offset+capacity, total), device=Z.device)
            slot = torch.searchsorted(cumulative, flat, right=True)
            first, index = slot % q, slot // q
            second = first+1+flat-(cumulative-counts.flatten())[slot]
            N = order[index[:,None],torch.stack((first, second), dim=1)].sort(dim=1).values
            rows = torch.cat((O[index], incoming[N]), dim=1)
            key = ((rows[:,0]*len(current)+rows[:,1])*Z.shape[1]+rows[:,2])*Z.shape[1]+rows[:,3]
            permutation = key.argsort()
            rows, index, N = rows[permutation], index[permutation], N[permutation]
            xc, xd = x[index,N[:,0]], x[index,N[:,1]]
            yc, yd = y[index,N[:,0]], y[index,N[:,1]]
            if len(current) == Z.shape[0]:
                v = xc*xd+yc*yd
            else:
                a, b = rows[:, :2].mT
                gcd = (Z[:,rows[:,2]]*Z[:,rows[:,3]]).sum(dim=0)
                v = (r[index,0]*gcd + (1-Gss[b,b])*xc*xd + (1-Gss[a,a])*yc*yd
                     + Gss[a,b]*(xc*yd+yc*xd))
            ratios = (w[index,N[:,0]]*w[index,N[:,1]]-v.square())/r[index,0]
            _keep_top(top, ratios, rows, tolerance, anchor)


def _two_candidates(Z, current, incoming, Gss, Gsn, width, capacity, tolerance, anchor=None, *, free_positions=None):
    top = []
    n, q = current.numel(), incoming.numel()
    outgoing = torch.triu_indices(n, n, 1, device=Z.device).mT
    if free_positions is not None:
        outgoing = outgoing[free_positions[outgoing].all(dim=1)]
    a, b = outgoing.mT
    rho = (1-Gss[a,a])*(1-Gss[b,b])-Gss[a,b].square()
    if n == Z.shape[0]:
        rho = torch.ones_like(rho)
    guarded = rho >= 1e-2
    if guarded.any():
        _pruned_candidates(top, Z, current, incoming, Gss, Gsn, outgoing[guarded],
                           rho[guarded], capacity, tolerance, anchor)
        outgoing = outgoing[~guarded]
        if not len(outgoing):
            return top
    signs = Z.new_tensor([-1., -1., 1., 1.])
    identity = torch.eye(4, dtype=Z.dtype, device=Z.device)
    for a in range(0, q, width):
        A = incoming[a:a+width]
        for b in range(a, q, width):
            B = incoming[b:b+width]
            pairs = (torch.triu_indices(len(A), len(B), 1, device=Z.device).mT if a == b
                     else torch.cartesian_prod(torch.arange(len(A), device=Z.device),
                                               torch.arange(len(B), device=Z.device)))
            if not pairs.numel():
                continue
            # Only incoming-ID blocks of G_NN are ever resident.
            columns = torch.cat((A, B))
            cross = torch.cat((Gsn[:, a:a+len(A)], Gsn[:, b:b+len(B)]), dim=1)
            Gnn = Z[:, columns].mT @ Z[:, columns]
            G = torch.cat((torch.cat((Gss, cross), dim=1),
                           torch.cat((cross.mT, Gnn), dim=1)), dim=0)
            count = outgoing.shape[0] * pairs.shape[0]
            for start in range(0, count, capacity):
                flat = torch.arange(start, min(start+capacity, count), device=Z.device)
                O = outgoing[flat // pairs.shape[0]]
                N = pairs[flat % pairs.shape[0]]
                indices = torch.cat((O, n + N + N.new_tensor([0, len(A)])), dim=1)
                block = G[indices[:, :, None], indices[:, None, :]]
                block *= signs[None, :, None]
                block += identity
                ratios = torch.linalg.det(block)
                rows = torch.cat((O, A[N[:, :1]], B[N[:, 1:]]), dim=1)
                _keep_top(top, ratios, rows, tolerance, anchor)
    return top


def _dense_two_ratios(Z, current, incoming, Gss, Gsn, free_positions=None):
    n, q = len(current), len(incoming)
    outgoing = torch.triu_indices(n,n,1,device=Z.device).mT
    if free_positions is not None:
        outgoing = outgoing[free_positions[outgoing].all(1)]
    pairs = torch.triu_indices(q,q,1,device=Z.device).mT
    O = outgoing.repeat_interleave(len(pairs),dim=0)
    N = pairs.repeat(len(outgoing),1)
    rows = torch.cat((O,incoming[N]),1)
    Gnn = Z[:,incoming].mT@Z[:,incoming]
    G = torch.cat((torch.cat((Gss,Gsn),1),torch.cat((Gsn.mT,Gnn),1)),0)
    idx = torch.cat((O,n+N),1)
    blocks = G[idx[:,:,None],idx[:,None,:]]
    blocks *= Z.new_tensor([-1.,-1.,1.,1.])[None,:,None]
    blocks += torch.eye(4,dtype=Z.dtype,device=Z.device)
    ratios = torch.linalg.det(blocks)
    a,b = O.mT
    rho = (1-Gss[a,a])*(1-Gss[b,b])-Gss[a,b].square()
    if n==Z.shape[0]:
        rho = torch.ones_like(rho)
    keep = rho>=1e-2
    a,b = a[keep],b[keep]; c,d = N[keep].mT; r = rho[keep]
    xc,xd,yc,yd = Gsn[a,c],Gsn[a,d],Gsn[b,c],Gsn[b,d]
    if n==Z.shape[0]:
        wc,wd,v = xc.square()+yc.square(),xd.square()+yd.square(),xc*xd+yc*yd
    else:
        diagonal = Z[:,incoming].square().sum(0)
        wc = r*(1+diagonal[c])+(1-Gss[b,b])*xc.square()+(1-Gss[a,a])*yc.square()+2*Gss[a,b]*xc*yc
        wd = r*(1+diagonal[d])+(1-Gss[b,b])*xd.square()+(1-Gss[a,a])*yd.square()+2*Gss[a,b]*xd*yd
        gcd = (Z[:,incoming[c]]*Z[:,incoming[d]]).sum(0)
        v = r*gcd+(1-Gss[b,b])*xc*xd+(1-Gss[a,a])*yc*yd+Gss[a,b]*(xc*yd+yc*xd)
    # Preserve the streamed closed-form scores for guarded pairs (including tie keys).
    ratios[keep] = (wc*wd-v.square())/r
    return ratios,rows


def _dense_candidates(ratios, rows, tolerance, anchor=None):
    top = []
    _keep_top(top,ratios,rows,tolerance,anchor)
    return top


@torch.no_grad()
def k2_exchange(F: torch.Tensor, ids: torch.Tensor, *, max_iter: int = 100, tile_size: int = 64, max_elements: int = 1_000_000,
                _free_positions: torch.Tensor | None = None) -> dict[str, object]:
    """Stream G-based ratios, verify at most eight per phase; see M3B2_API.md."""
    integer(max_iter, 'max_iter', 0)
    integer(tile_size, 'tile_size')
    M = exact_information(F, ids, max_elements=max_elements)
    m, p = F.shape
    n, q = ids.numel(), m - ids.numel()
    if not m >= n >= p >= 1:
        raise ValueError('require m>=n>=p>=1')
    current = ids.clone()
    free_count = n if _free_positions is None else int(_free_positions.sum())
    one = lambda *args, **kw: _one_candidates(*args, free_positions=_free_positions, **kw)
    two = lambda *args, **kw: _two_candidates(*args, free_positions=_free_positions, **kw)
    score = d_optimality(M, max_elements=max_elements).item()
    # Fixed state + conservative block/4x4 determinant workspace; no N2 output.
    fixed = 2*m*p + 2*n*m + 8*p*p + 8*m + 8*n + 128
    width = min(tile_size, max(1, q))
    while True:
        capacity = (max_elements - fixed - 2*(n+2*width)**2 - 4*p*width) // 128
        if capacity >= max(1, n*width):
            break
        if width == 1:
            raise ValueError('insufficient max_elements for one score tile/workspace')
        width = max(1, width // 2)
    dense = fixed+m*m+128*(free_count*(free_count-1)//2)*(q*(q-1)//2)+8*n*q <= max_elements
    if dense:
        width = max(1,q)
    history, objectives = [], [score]
    scanned = dict(k1=0, k2=0)
    verification_attempts = 0
    while True:
        L = torch.linalg.cholesky(M)
        Z = torch.linalg.solve_triangular(L, F.mT, upper=False)
        diagonal = Z.square().sum(dim=0)
        gap_bound = (p * (n * diagonal.max() / p).log()).item()
        if n == m:
            reason = 'NO_LEGAL_SWAP'
            break
        empty_two = free_count < 2 or q < 2
        if len(history) >= max_iter:
            reason = 'NO_K2_NEIGHBORHOOD' if empty_two else 'MAX_ITER'
            break
        condition = torch.linalg.cond(M).item()
        selected = torch.zeros(m, dtype=torch.bool, device=F.device)
        selected[current] = True
        incoming = (~selected).nonzero().flatten()
        Gss = Z[:, current].mT @ Z[:, current]
        Gsn = Z[:, current].mT @ Z[:, incoming]
        ratio_tolerance = _tolerance(1., 1., condition, condition, p)
        top, passes = _ordered_top(one, Gss, Gsn, diagonal, incoming, width, ratio_tolerance)
        scanned['k1'] += passes*free_count*q
        k = 1
        if not top:
            if empty_two:
                reason = 'NO_K2_NEIGHBORHOOD'
                break
            k = 2
            if dense:
                ratios, rows = _dense_two_ratios(Z,current,incoming,Gss,Gsn,_free_positions)
                top, passes = _ordered_top(_dense_candidates,ratios,rows,ratio_tolerance)
            else:
                top, passes = _ordered_top(two, Z, current, incoming, Gss, Gsn, width,
                                           capacity, ratio_tolerance)
            scanned['k2'] += passes * (free_count*(free_count-1)//2) * (q*(q-1)//2)
            if not top:
                reason = 'LOCAL_2_NUMERICAL'
                break
        for gamma, row in top:
            positions, incoming_ids = list(row[:k]), list(row[k:])
            trial = current.clone()
            trial[positions] = trial.new_tensor(incoming_ids)
            verification_attempts += 1
            trial_M = exact_information(F, trial, max_elements=max_elements)
            try:
                value = d_optimality(trial_M, max_elements=max_elements).item()
            except ValueError:
                continue
            tolerance = _tolerance(score, value, condition,
                                   torch.linalg.cond(trial_M).item(), p)
            if value - score > tolerance:
                history.append(dict(k=k, outgoing_positions=positions,
                    outgoing_ids=current[positions].tolist(), incoming_ids=incoming_ids,
                    gamma=gamma, gain=value-score, tolerance=tolerance))
                current, M, score = trial, trial_M, value
                objectives.append(score)
                break
        else:
            reason = 'NUMERICAL_INCONCLUSIVE'
            break
    return dict(ids=current, initial_objective=objectives[0], final_objective=score,
        objective_history=objectives, swap_history=history, accepted_swaps=len(history),
        scanned_pairs=scanned, verification_attempts=verification_attempts, stop_reason=reason,
        full_neighborhood_checked=reason == 'LOCAL_2_NUMERICAL',
        certificate='LOCAL_2_NUMERICAL' if reason == 'LOCAL_2_NUMERICAL' else 'NOT_CERTIFIED',
        efficiency_gap_bound=gap_bound)
