"""One same-runtime base/head CPU benchmark pair; accelerator unused."""
import json, os, pathlib, subprocess, sys
root = pathlib.Path('/content/w1-pair'); sha = os.environ['SAE_CANDIDATE_SHA']
base = '91244d841befbbe5d0a9f80faf6e19f269970d69'
subprocess.run(['git','clone','https://github.com/JohnnyZPiggy/TorchOED.git',str(root)],check=True,capture_output=True)
subprocess.run([sys.executable,'-m','pip','download','pydoe==1.6.0','--no-deps','-d','/content/w1-wheel'],check=True,capture_output=True)
wheel = str(next(pathlib.Path('/content/w1-wheel').glob('*.whl')))
subprocess.run([sys.executable,'-m','pip','install',wheel],check=True,capture_output=True)
data = dict(status='PASS',source_sha=sha,base_sha=base,device_observed=json.loads(os.environ['SAE_DEVICE_OBSERVED']),runs={})
for name,commit in (('base',base),('head',sha)):
    checkout = root.parent/('w1-'+name)
    subprocess.run(['git','worktree','add','--detach',str(checkout),commit],cwd=root,check=True,capture_output=True)
    subprocess.run([sys.executable,'-B','tools/bench_b0.py',wheel,'w1-pair'],cwd=checkout,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True,capture_output=True)
    data['runs'][name] = json.loads((checkout/'reports/w1-pair/results.json').read_text())
pathlib.Path('/content/w1-pair.json').write_text(json.dumps(data,allow_nan=False)+'\n')
