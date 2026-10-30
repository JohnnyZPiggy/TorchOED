"""Deterministic CPU float64 VND using the accepted one/two-swap APIs."""
import torch

from .exchange import fedorov_exchange
from .two_swap import score_two_swaps


@torch.no_grad()
def k2_exchange(F, ids, *, max_iter=100, tile_size=64, max_elements=1_000_000):
    """Run one-swap descent, then best verified two-swaps; see M3B2_API.md."""
    current, remaining = ids, max_iter
    history, objectives = [], []
    scanned = dict(k1=0, k2=0)
    verification_attempts = 0
    while True:
        phase1 = fedorov_exchange(F, current, max_iter=remaining,
                                  tile_size=tile_size, max_elements=max_elements)
        if not objectives:
            objectives.append(phase1['initial_objective'])
        objectives.extend(phase1['objective_history'][1:])
        for move in phase1['swap_history']:
            history.append(dict(k=1, outgoing_positions=[move['position']],
                outgoing_ids=[move['outgoing_id']], incoming_ids=[move['incoming_id']],
                gamma=move['gamma'], gain=move['gain'], tolerance=move['tolerance']))
        current = phase1['ids']
        remaining -= phase1['accepted_swaps']
        scanned['k1'] += phase1['evaluated_candidates']
        verification_attempts += phase1['direct_trial_evaluations']
        reason = phase1['stop_reason']
        if reason != 'NO_LEGAL_SWAP' and (current.numel() < 2 or F.shape[0] - current.numel() < 2):
            reason = 'NO_K2_NEIGHBORHOOD'
        if reason != 'LOCAL_1_SWAP_NUMERICAL':
            break
        phase2 = score_two_swaps(F, current, tile_size=tile_size,
                                 max_elements=max_elements)
        scanned['k2'] += phase2['evaluated_pairs']
        verification_attempts += phase2['evaluated_pairs']
        eligible = ((phase2['status'] == 0) &
                    (phase2['gains'] > phase2['tolerances'])).nonzero().flatten()
        if not eligible.numel():
            unresolved_improving = ((phase2['status'] == 2) &
                (phase2['ratios'].isnan() | (phase2['ratios'] > 1))).any().item()
            reason = ('NUMERICAL_INCONCLUSIVE' if unresolved_improving
                      else 'LOCAL_2_NUMERICAL')
            break
        # argmax returns the first maximum in the scorer's deterministic row order.
        best = eligible[phase2['gains'][eligible].argmax()].item()
        positions = phase2['outgoing_positions'][best]
        incoming = phase2['incoming_ids'][best]
        history.append(dict(k=2, outgoing_positions=positions.tolist(),
            outgoing_ids=current[positions].tolist(), incoming_ids=incoming.tolist(),
            gamma=phase2['ratios'][best].item(), gain=phase2['gains'][best].item(),
            tolerance=phase2['tolerances'][best].item()))
        current[positions] = incoming
        objectives.append(phase2['direct_scores'][best].item())
        remaining -= 1
    return dict(ids=current, initial_objective=objectives[0], final_objective=objectives[-1],
        objective_history=objectives, swap_history=history, accepted_swaps=len(history),
        scanned_pairs=scanned, verification_attempts=verification_attempts, stop_reason=reason,
        full_neighborhood_checked=reason == 'LOCAL_2_NUMERICAL',
        certificate='LOCAL_2_NUMERICAL' if reason == 'LOCAL_2_NUMERICAL' else 'NOT_CERTIFIED')
