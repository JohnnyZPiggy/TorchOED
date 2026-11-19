"""G0 single-cell adapter; transport success preserves science failures."""
import json, os, pathlib, subprocess, sys
root = pathlib.Path('/content/torchoed-g0')
sha = os.environ['SAE_CANDIDATE_SHA']
subprocess.run(['git', 'clone', 'https://github.com/JohnnyZPiggy/TorchOED.git', str(root)], check=True, capture_output=True)
subprocess.run(['git', 'checkout', sha], cwd=root, check=True, capture_output=True)
data = dict(source_sha=sha, status='PASS', rows=[], calls=[],
    device_observed=json.loads(os.environ['SAE_DEVICE_OBSERVED']))
for index, method in enumerate(('cuda', 'cpu', 'cuda')):
    run = subprocess.run([sys.executable, '-B', 'tools/bench_crossover.py', '--call', '3', '1000', method],
        cwd=root, capture_output=True, text=True)
    data['calls'].append(dict(method=method, warmup=index==0, exit_code=run.returncode, stderr=run.stderr))
    if index and run.returncode==0:
        data['rows'].append(json.loads(run.stdout))
data['scientific_status'] = 'PASS' if all(c['exit_code']==0 for c in data['calls']) else 'ERROR'
pathlib.Path('/content/g0-probe.json').write_text(json.dumps(data, allow_nan=False)+'\n')
