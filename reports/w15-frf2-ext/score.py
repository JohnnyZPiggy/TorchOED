import csv
import json
from itertools import combinations
from pathlib import Path
import torch
from torchoed import word_length_pattern

p = Path(__file__).parent
torch.set_num_threads(1)
metadata = json.loads((p/'frf2.json').read_text())
runtime = json.loads((p/'runtime.json').read_text())
rows = list(csv.DictReader((p/'designs.csv').open()))
assert runtime['R_version'].startswith('R version 4.5.2') and runtime['packages']['FrF2'] == '2.3-5'
results = dict(run_id=runtime['run_id'], source_sha=runtime['source_sha'], R_version=runtime['R_version'], FrF2='2.3-5', RD1_Rd_line=metadata['RD1_Rd_line'], cases={})
lines = ['# FrF2 C2 frozen references', f"Run {runtime['run_id']}; R 4.5.2; FrF2 2.3-5; reference freeze only, no verdict.", 'RD1 verbatim FrF2.Rd example: `'+metadata['RD1_Rd_line']+'` (randomize=FALSE added).']
for name, info in metadata['cases'].items():
    result = dict(info)
    if info['status'] == 'RECORDED':
        selected = [row for row in rows if row['case'] == name]
        k = info['nfactors']
        X = torch.tensor([[float(row[f'x{j+1}']) for j in range(k)] for row in selected], dtype=torch.float64)
        assert ((X == -1) | (X == 1)).all()
        result['WLP'] = word_length_pattern(X).tolist()
        block = torch.tensor([int(row['block']) for row in selected])
        confounded = [list(effect) for order in (1, 2) for effect in combinations(range(1, k+1), order) if block.unique().numel() > 1 and all(X[block == b][:, [i-1 for i in effect]].prod(1).unique().numel() == 1 for b in block.unique())]
        result.update(block_confounded_effects=confounded, main_effect_confounded=any(len(effect) == 1 for effect in confounded), factor_numbering='1-based FrF2 factor order', reported_generators='No generators/gen field present in design.info; available alias and block.gen metadata frozen in frf2.json')
        lines.append(f"{name}: WLP {result['WLP']}; confounded {confounded}; main effect confounded {result['main_effect_confounded']}.")
    else:
        lines.append(f"{name}: NOT_RUN — {info['error']}")
    results['cases'][name] = result
assert len(results['cases']) == 7
(p/'results.json').write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
(p/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
