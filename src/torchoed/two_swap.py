"""All legal simultaneous two-row exchange scores; no search/state updates."""
import torch

from ._validation import budget, integer
from .criteria import d_optimality
from .information import exact_information


def _pair_tiles(n, incoming, width):
    rows = []
    for i in range(n - 1):
        for k in range(i + 1, n):
            for a in range(len(incoming) - 1):
                for b in range(a + 1, len(incoming)):
                    rows.append((i, k, incoming[a], incoming[b]))
                    if len(rows) == width:
                        yield torch.tensor(rows, dtype=torch.int64)
                        rows = []
    if rows:
        yield torch.tensor(rows, dtype=torch.int64)


def _rank_four_ratios(L, F, ids, pairs):
    outgoing = F[ids[pairs[:, :2]]]
    incoming = F[pairs[:, 2:]]
    W = torch.cat((outgoing, incoming), dim=1).transpose(-2, -1)
    Z = torch.linalg.solve_triangular(L, W, upper=False)
    H = Z.mT @ Z
    signs = F.new_tensor([-1., -1., 1., 1.])
    return torch.linalg.det(torch.eye(4, dtype=F.dtype) + signs[None, :, None] * H)


@torch.no_grad()
def score_two_swaps(F, ids, *, tile_size=64, max_elements=1_000_000):
    """Score every unordered k=2 exchange. Schema/policies: docs/M3B1_API.md."""
    if not isinstance(F, torch.Tensor) or F.dtype != torch.float64:
        raise TypeError("F must be a float64 torch.Tensor")
    if F.ndim != 2 or F.device.type != "cpu":
        raise ValueError("M3-B1 supports unbatched CPU features only")
    if not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64:
        raise TypeError("ids must be an int64 torch.Tensor")
    if ids.ndim != 1 or ids.device.type != "cpu":
        raise ValueError("ids must be a CPU vector")
    integer(tile_size, "tile_size")
    integer(max_elements, "max_elements")
    m, p = F.shape
    n = ids.numel()
    if not m >= n >= p >= 1:
        raise ValueError("require m>=n>=p>=1")
    budget(F.shape, max_elements)
    q = m - n
    count = (n*(n-1)//2) * (q*(q-1)//2)
    # Count retained numeric/index outputs before materializing any neighborhood.
    output_elements = 10*count + n
    budget((output_elements,), max_elements)
    fixed = 2*m*p + 6*n*p + 10*p*p + 4*m + 2*n + 64
    per_pair = 16*p + 128
    available = max_elements - output_elements - fixed
    if available < 0 or (count and available < per_pair):
        raise ValueError("insufficient max_elements for outputs plus one score tile/workspace")
    width = min(tile_size, available // per_pair) if count else 0
    M = exact_information(F, ids, max_elements=max_elements)
    try:
        initial = d_optimality(M, max_elements=max_elements).item()
    except ValueError as exc:
        raise ValueError("invalid initial design under M2-A numerical SPD policy: " + str(exc)) from exc
    positions = torch.empty((count, 2), dtype=torch.int64)
    incoming_ids = torch.empty((count, 2), dtype=torch.int64)
    ratios = torch.full((count,), float('nan'), dtype=F.dtype)
    gains, scores, tolerances, errors = (torch.full_like(ratios, float('nan')) for _ in range(4))
    status = torch.full((count,), 2, dtype=torch.int8)
    reasons = ["UNRESOLVED"] * count
    result = dict(initial_ids=ids.clone(), initial_objective=initial,
                  outgoing_positions=positions, incoming_ids=incoming_ids,
                  ratios=ratios, gains=gains, direct_scores=scores,
                  tolerances=tolerances, log_ratio_errors=errors,
                  status=status, reasons=reasons,
                  status_names={0: "VALID", 1: "INVALID", 2: "NUMERICAL_INCONCLUSIVE"},
                  evaluated_pairs=count, effective_tile_size=width,
                  workspace_element_estimate=output_elements + fixed + per_pair*width,
                  certificate="NOT_APPLICABLE")
    counts = dict(VALID=0, INVALID=0, NUMERICAL_INCONCLUSIVE=0)
    result['counts'] = counts
    if not count:
        return result
    L = torch.linalg.cholesky(M)
    condition = torch.linalg.cond(M).item()
    selected = torch.zeros(m, dtype=torch.bool)
    selected[ids] = True
    eligible = (~selected).nonzero().flatten().tolist()
    offset = 0
    for pairs in _pair_tiles(n, eligible, width):
        size = pairs.shape[0]
        positions[offset:offset+size] = pairs[:, :2]
        incoming_ids[offset:offset+size] = pairs[:, 2:]
        gamma = _rank_four_ratios(L, F, ids, pairs)
        ratios[offset:offset+size] = gamma
        for local, pair in enumerate(pairs):
            index = offset + local
            trial = ids.clone()
            trial[pair[:2]] = pair[2:]
            try:
                trial_M = exact_information(F, trial, max_elements=max_elements)
                value = d_optimality(trial_M, max_elements=max_elements).item()
            except ValueError:
                qr = torch.linalg.qr(F[trial], mode='r').R
                if bool((qr.diagonal() == 0).any()):
                    status[index] = 1
                    reasons[index] = "NUMERICAL_QR_ZERO"
                    counts['INVALID'] += 1
                else:
                    reasons[index] = "M2_REJECT_QR_NONZERO"
                    counts['NUMERICAL_INCONCLUSIVE'] += 1
                continue
            scores[index] = value
            trial_condition = torch.linalg.cond(trial_M).item()
            tolerance = (256 * torch.finfo(F.dtype).eps * p
                         * max(1., abs(initial), abs(value))
                         * max(1., condition, trial_condition))
            tolerances[index] = tolerance
            ratio = gamma[local].item()
            if not bool(torch.isfinite(gamma[local])) or ratio <= 0:
                reasons[index] = "NONPOSITIVE_OR_NONFINITE_RATIO_ON_SPD_TRIAL"
                counts['NUMERICAL_INCONCLUSIVE'] += 1
                continue
            error = abs(gamma[local].log().item() - (value - initial))
            errors[index] = error
            if not bool(torch.isfinite(tolerances[index])) or error > tolerance:
                reasons[index] = "RATIO_DIRECT_DISAGREEMENT"
                counts['NUMERICAL_INCONCLUSIVE'] += 1
                continue
            status[index] = 0
            reasons[index] = "DIRECT_SPD_AND_RATIO_AGREE"
            gains[index] = value - initial
            counts['VALID'] += 1
        offset += size
    return result
