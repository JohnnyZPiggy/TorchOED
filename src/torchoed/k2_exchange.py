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


def _one_candidates(Gss, Gsn, diagonal, incoming, width, tolerance, anchor=None):
    top = []
    n = Gss.shape[0]
    for start in range(0, incoming.numel(), width):
        end = min(start + width, incoming.numel())
        ratios = ((1 - Gss.diagonal()[:, None]) * (1 + diagonal[incoming[start:end]])
                  + Gsn[:, start:end].square())
        positions = torch.arange(n, device=incoming.device).repeat_interleave(end-start)
        rows = torch.stack((positions, incoming[start:end].repeat(n)), dim=1)
        _keep_top(top, ratios.flatten(), rows, tolerance, anchor)
    return top


def _two_candidates(Z, current, incoming, Gss, Gsn, width, capacity, tolerance, anchor=None):
    top = []
    n, q = current.numel(), incoming.numel()
    outgoing = torch.triu_indices(n, n, 1, device=Z.device).mT
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


@torch.no_grad()
def k2_exchange(F, ids, *, max_iter=100, tile_size=64, max_elements=1_000_000):
    """Stream G-based ratios, verify at most eight per phase; see M3B2_API.md."""
    integer(max_iter, 'max_iter', 0)
    integer(tile_size, 'tile_size')
    M = exact_information(F, ids, max_elements=max_elements)
    if F.dtype != torch.float64:
        raise TypeError('F must be float64')
    if F.ndim != 2:
        raise ValueError('F must be unbatched')
    m, p = F.shape
    n, q = ids.numel(), m - ids.numel()
    if not m >= n >= p >= 1:
        raise ValueError('require m>=n>=p>=1')
    current = ids.clone()
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
        empty_two = n < 2 or q < 2
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
        top, passes = _ordered_top(_one_candidates, Gss, Gsn, diagonal, incoming, width, ratio_tolerance)
        scanned['k1'] += passes*n*q
        k = 1
        if not top:
            if empty_two:
                reason = 'NO_K2_NEIGHBORHOOD'
                break
            k = 2
            top, passes = _ordered_top(_two_candidates, Z, current, incoming, Gss, Gsn, width,
                                       capacity, ratio_tolerance)
            scanned['k2'] += passes * (n*(n-1)//2) * (q*(q-1)//2)
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
