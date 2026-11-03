"""One same-runtime base/head CPU benchmark pair; accelerator unused."""
import json, os, pathlib, subprocess, sys
root = pathlib.Path('/content/w95-pair'); sha = os.environ['SAE_CANDIDATE_SHA']
base = 'a99fe6d7d808b692b4562eb5ba7acc587e59be71'
subprocess.run(['git','clone','https://github.com/JohnnyZPiggy/TorchOED.git',str(root)],check=True,capture_output=True)
subprocess.run([sys.executable,'-m','pip','download','pydoe==1.6.0','--no-deps','-d','/content/w95-wheel'],check=True,capture_output=True)
wheel = str(next(pathlib.Path('/content/w95-wheel').glob('*.whl')))
subprocess.run([sys.executable,'-m','pip','install',wheel],check=True,capture_output=True)
data = dict(status='PASS',source_sha=sha,base_sha=base,device_observed=json.loads(os.environ['SAE_DEVICE_OBSERVED']),runs={})
for name,commit in (('base',base),('head',sha)):
    checkout = root.parent/('w95-'+name)
    subprocess.run(['git','worktree','add','--detach',str(checkout),commit],cwd=root,check=True,capture_output=True)
    subprocess.run([sys.executable,'-B','tools/bench_b0.py',wheel,'w95-pair'],cwd=checkout,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True,capture_output=True)
    data['runs'][name] = json.loads((checkout/'reports/w95-pair/results.json').read_text())
pathlib.Path('/content/w95-pair.json').write_text(json.dumps(data,allow_nan=False)+'\n')
