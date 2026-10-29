"""CPU float64 exact-ID, best-improvement Fedorov 1-swap search."""
import torch

from ._validation import budget, integer
from .criteria import d_optimality
from .information import exact_information


def _exchange_ratios(R, T):
    """Whitened selected/incoming columns; only bounded tiles, no inverse."""
    u = R.square().sum(dim=0)
    v = T.square().sum(dim=0)
    c = R.mT @ T
    return (1 - u[:, None]) * (1 + v[None, :]) + c.square()


def _tolerance(old_score, new_score, old_condition, new_condition, p):
    # Heuristic error policy, not an interval bound or rigorous certificate.
    return (128 * torch.finfo(torch.float64).eps * p
            * max(1., abs(old_score), abs(new_score))
            * max(1., old_condition, new_condition))


@torch.no_grad()
def fedorov_exchange(F, ids, *, max_iter=100, tile_size=64,
                     allow_repeats=False, max_elements=1_000_000):
    """Search from provided IDs. See docs/M3_API.md for stop/error policies."""
    if not isinstance(F, torch.Tensor) or F.dtype != torch.float64:
        raise TypeError("F must be a float64 torch.Tensor")
    if F.device.type != "cpu" or F.ndim != 2:
        raise ValueError("M3-A supports unbatched CPU features only")
    if not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64:
        raise TypeError("ids must be an int64 torch.Tensor")
    if ids.device.type != "cpu" or ids.ndim != 1:
        raise ValueError("ids must be a CPU vector")
    if type(allow_repeats) is not bool:
        raise TypeError("allow_repeats must be bool")
    if allow_repeats:
        raise NotImplementedError("M3-A has no with-replacement search")
    integer(max_iter, "max_iter", 0)
    integer(tile_size, "tile_size")
    integer(max_elements, "max_elements")
    m, p = F.shape
    n = ids.numel()
    if not m >= n >= p >= 1:
        raise ValueError("require m>=n>=p>=1")
    budget(F.shape, max_elements)
    budget((max_iter + 1, 8), max_elements)
    # Conservative major-tensor workspace estimate, excluding caller-owned F.
    fixed = 4*n*p + 8*p*p + 2*m + 4*n
    per_column = 6*n + 4*p + 8
    capacity = (max_elements - fixed) // per_column
    if capacity < 1:
        raise ValueError("insufficient max_elements for one score tile/workspace")
    width = min(tile_size, capacity)
    current = ids.clone()
    M = exact_information(F, current, max_elements=max_elements)
    try:
        score = d_optimality(M, max_elements=max_elements).item()
    except ValueError as exc:
        raise ValueError("invalid initial design under M2-A numerical SPD policy: " + str(exc)) from exc
    initial = score
    objectives, swaps, scans = [score], [], []
    evaluated = 0
    full_checked = False
    while True:
        if n == m:
            reason = "NO_LEGAL_SWAP"
            break
        if len(swaps) >= max_iter:
            reason = "MAX_ITER"
            break
        L = torch.linalg.cholesky(M)
        R = torch.linalg.solve_triangular(L, F[current].mT, upper=False)
        condition = torch.linalg.cond(M).item()
        selected = torch.zeros(m, dtype=torch.bool)
        selected[current] = True
        incoming = (~selected).nonzero().flatten()  # exact IDs, ascending
        diagnostics = dict(evaluated=0, verified_spd=0, qr_zero=0, uncertain=0,
                           marginal=0, marginal_upper_gain=0.,
                           max_ratio_log_error=0., condition=condition,
                           max_gain_tolerance=0., first_uncertain=None)
        # Only the best score and the tie winner are retained, never n*m records.
        best_score = None
        best_move = None
        for start in range(0, incoming.numel(), width):
            tile = incoming[start:start + width]
            T = torch.linalg.solve_triangular(L, F[tile].mT, upper=False)
            ratios = _exchange_ratios(R, T)
            for i in range(n):
                for column, j_tensor in enumerate(tile):
                    j = j_tensor.item()
                    gamma = ratios[i, column].item()
                    trial = current.clone()
                    trial[i] = j
                    diagnostics['evaluated'] += 1
                    try:
                        trial_M = exact_information(F, trial, max_elements=max_elements)
                        value = d_optimality(trial_M, max_elements=max_elements).item()
                    except ValueError:
                        # A numerical zero QR diagonal diagnoses rank loss; a tiny
                        # nonzero pivot is uncertain, never proof of singularity.
                        qr = torch.linalg.qr(F[trial], mode='r').R
                        if bool((qr.diagonal() == 0).any()):
                            diagnostics['qr_zero'] += 1
                            continue
                        diagnostics['uncertain'] += 1
                        if diagnostics['first_uncertain'] is None:
                            diagnostics['first_uncertain'] = dict(position=i, incoming_id=j,
                                                                  reason="M2_REJECT_QR_NONZERO")
                        continue
                    diagnostics['verified_spd'] += 1
                    trial_condition = torch.linalg.cond(trial_M).item()
                    tolerance = _tolerance(score, value, condition, trial_condition, p)
                    diagnostics['max_gain_tolerance'] = max(diagnostics['max_gain_tolerance'], tolerance)
                    delta = value - score
                    ratio_log = torch.tensor(gamma, dtype=torch.float64).log().item() if gamma > 0 else float('nan')
                    error = abs(ratio_log - delta)
                    if not (torch.isfinite(torch.tensor([gamma, ratio_log, tolerance], dtype=torch.float64)).all()
                            and error <= tolerance):
                        diagnostics['uncertain'] += 1
                        if diagnostics['first_uncertain'] is None:
                            diagnostics['first_uncertain'] = dict(position=i, incoming_id=j,
                                                                  reason="RATIO_DIRECT_DISAGREEMENT")
                        continue
                    diagnostics['max_ratio_log_error'] = max(diagnostics['max_ratio_log_error'], error)
                    if 0 < delta <= tolerance:
                        diagnostics['marginal'] += 1
                        diagnostics['marginal_upper_gain'] = max(
                            diagnostics['marginal_upper_gain'], delta + tolerance)
                        continue
                    if delta <= tolerance:
                        continue
                    key = (i, j)
                    move = dict(position=i, outgoing_id=current[i].item(), incoming_id=j,
                                score=value, gain=delta, gamma=gamma, tolerance=tolerance,
                                condition=trial_condition)
                    if (best_score is None or value > best_score
                            or (value == best_score and key < (best_move['position'], best_move['incoming_id']))):
                        best_score, best_move = value, move
        evaluated += diagnostics['evaluated']
        scans.append(diagnostics)
        full_checked = True
        if diagnostics['uncertain']:
            reason = "NUMERICAL_INCONCLUSIVE"
            break  # Unknown neighbors could invalidate a claimed best move.
        if diagnostics['marginal'] and (best_move is None or
                best_move['gain'] - best_move['tolerance'] <= diagnostics['marginal_upper_gain']):
            reason = "NUMERICAL_INCONCLUSIVE"
            break
        if best_move is None:
            reason = "LOCAL_1_SWAP_NUMERICAL"
            break
        # Anchor ties to the global verified maximum, not a nontransitive chain.
        # Recheck a lexicographic prefix instead of storing the whole neighborhood.
        tie_tolerance = best_move['tolerance']
        chosen = None
        diagnostics['tie_rechecks'] = 0
        for i in range(n):
            for j in incoming.tolist():
                trial = current.clone()
                trial[i] = j
                diagnostics['tie_rechecks'] += 1
                try:
                    trial_M = exact_information(F, trial, max_elements=max_elements)
                    value = d_optimality(trial_M, max_elements=max_elements).item()
                except ValueError:
                    continue
                trial_condition = torch.linalg.cond(trial_M).item()
                tolerance = _tolerance(score, value, condition, trial_condition, p)
                if value >= best_score - tie_tolerance and value - score > tolerance:
                    T = torch.linalg.solve_triangular(L, F[j:j+1].mT, upper=False)
                    gamma = _exchange_ratios(R[:, i:i+1], T).item()
                    chosen = dict(position=i, outgoing_id=current[i].item(), incoming_id=j,
                                  score=value, gain=value-score, gamma=gamma,
                                  tolerance=tolerance, tie_tolerance=tie_tolerance,
                                  best_verified_score=best_score)
                    break
            if chosen is not None:
                break
        if chosen is None:
            reason = "NUMERICAL_INCONCLUSIVE"
            break
        best_move = chosen
        current[best_move['position']] = best_move['incoming_id']
        M = exact_information(F, current, max_elements=max_elements)
        score = best_move['score']
        swaps.append(best_move)
        objectives.append(score)
        full_checked = False  # The newly accepted state has not yet been scanned.
    return dict(ids=current, initial_objective=initial, final_objective=score,
                objective_history=objectives, swap_history=swaps, stop_reason=reason,
                accepted_swaps=len(swaps), evaluated_candidates=evaluated,
                direct_trial_evaluations=evaluated + sum(s.get('tie_rechecks', 0) for s in scans),
                full_neighborhood_checked=full_checked, scans=scans,
                certificate="NUMERICAL_ONLY" if reason == "LOCAL_1_SWAP_NUMERICAL" else "NONE",
                effective_tile_size=width, workspace_element_estimate=fixed + per_column*width)
