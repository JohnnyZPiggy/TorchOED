from itertools import combinations
import json
from pathlib import Path

import pytest
import torch
from torchoed import minimum_aberration, fractional_factorial, fractional_factorial_generators, word_length_pattern, ff2n
from torchoed.factorial import _FRF2_GENERATORS


def brute(k, runs, blocks=1):
    m = runs.bit_length() - 1
    masks = [g for g in range(1, runs) if g.bit_count() >= 2]
    block_indicators = []
    base = ff2n(m)
    for B in combinations(range(1, runs), blocks.bit_length()-1) if blocks > 1 else []:
        span = {0}
        for beta in B:
            span |= {s ^ beta for s in span}
        if len(span) == blocks:
            labels = sum((base[:, [c for c in range(m) if beta >> c & 1]].prod(1) == -1).long()*2**t
                         for t, beta in enumerate(B))
            block_indicators.append((labels[:, None] == torch.arange(blocks)).double())
    indicators = torch.cat(block_indicators, 1) if block_indicators else None
    best = None
    for G in combinations(masks, k-m):
        generators = [[c for c in range(m) if g >> c & 1] for g in G]
        X = fractional_factorial(m, generators)
        wlp = tuple(word_length_pattern(X).tolist())
        feasible = blocks == 1
        if blocks > 1:
            effects = torch.stack([X[:, list(ix)].prod(1) for r in (1, 2) for ix in combinations(range(k), r)], 1)
            sums = (indicators.T @ effects).reshape(-1, blocks, effects.shape[1])
            confounded = (sums.abs() == runs/blocks).all(1)
            feasible = (~confounded.any(1)).any().item()
        if feasible and (best is None or wlp < best[0]):
            best = wlp, generators
    return best


def test_exhaustive_and_catalogue():
    for k, runs in [(6, 16), (8, 16), (9, 32)]:
        result = minimum_aberration(k, runs)
        expected = brute(k, runs)
        assert (tuple(result['wlp'].tolist()), result['generators']) == expected
        assert torch.equal(result['wlp'], word_length_pattern(result['design']))
    for k, runs in _FRF2_GENERATORS:
        X = fractional_factorial(runs.bit_length()-1, fractional_factorial_generators(k, runs))
        assert torch.equal(minimum_aberration(k, runs)['wlp'], word_length_pattern(X))


def test_blocked_oracle():
    for k, runs, blocks in [(6, 16, 2), (7, 32, 4)]:
        result = minimum_aberration(k, runs, blocks=blocks)
        assert tuple(result['wlp'].tolist()) == brute(k, runs, blocks)[0]
        X, labels = result['design'], result['block']
        expected_labels = sum((X[:, gen].prod(1) == -1).long()*2**t
                              for t, gen in enumerate(result['block_generators']))
        assert torch.equal(labels, expected_labels)
        confounded = []
        for r in (1, 2, 3):
            for ix in combinations(range(k), r):
                effect = X[:, list(ix)].prod(1)
                constant = all(effect[labels == t].unique().numel() == 1 for t in labels.unique())
                if r < 3:
                    assert not constant
                elif constant:
                    confounded.append(ix)
        assert result['block_confounded'] == confounded
    with pytest.raises(ValueError, match='unconfounded'):
        minimum_aberration(5, 16, blocks=4)


def test_frozen_frf2():
    cases = json.loads((Path(__file__).parents[1]/'reports/w15-frf2-ext/results.json').read_text())['cases']
    sizes = {'MA1': (13, 32, 1), 'MA2': (14, 32, 1), 'MA3': (12, 64, 1),
             'BL1': (6, 16, 2), 'BL3': (7, 32, 4), 'RD1': (3, 8, 2)}
    for name, (k, runs, blocks) in sizes.items():
        result = minimum_aberration(k, runs, blocks=blocks)
        assert result['wlp'].tolist() == cases[name]['WLP']
        assert all(len(ix) == 3 for ix in result['block_confounded'])
    with pytest.raises(ValueError, match='unconfounded'):
        minimum_aberration(5, 16, blocks=4)
