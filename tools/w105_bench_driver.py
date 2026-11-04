"""One same-runtime base/head CPU benchmark pair; accelerator unused."""
import json, os, pathlib, subprocess, sys
root = pathlib.Path('/content/w105-pair'); sha = os.environ['SAE_CANDIDATE_SHA']
base = '9f2bab790508bf8270d0c597e3ebc4833dc09d60'
subprocess.run(['git','clone','https://github.com/JohnnyZPiggy/TorchOED.git',str(root)],check=True,capture_output=True)
subprocess.run([sys.executable,'-m','pip','download','pydoe==1.6.0','--no-deps','-d','/content/w105-wheel'],check=True,capture_output=True)
wheel = str(next(pathlib.Path('/content/w105-wheel').glob('*.whl')))
subprocess.run([sys.executable,'-m','pip','install',wheel],check=True,capture_output=True)
data = dict(status='PASS',source_sha=sha,base_sha=base,device_observed=json.loads(os.environ['SAE_DEVICE_OBSERVED']),runs={})
for name,commit in (('base',base),('head',sha)):
    checkout = root.parent/('w105-'+name)
    subprocess.run(['git','worktree','add','--detach',str(checkout),commit],cwd=root,check=True,capture_output=True)
    subprocess.run([sys.executable,'-B','tools/bench_b0.py',wheel,'w105-pair'],cwd=checkout,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True,capture_output=True)
    data['runs'][name] = json.loads((checkout/'reports/w105-pair/results.json').read_text())
measurement = "import json,time,statistics,torch; from torchoed import full_factorial,polynomial_features,optimal_design; torch.set_num_threads(1); F=polynomial_features(full_factorial((3,3,3)).double()-1,2); out={}\nfor c in ('A','I'):\n times=[]\n for repeat in range(3):\n  begin=time.perf_counter(); r=optimal_design(F,20,criterion=c,generator=torch.Generator().manual_seed(0)); times.append(time.perf_counter()-begin)\n out[c]=dict(seconds=statistics.median(times),objective=r['objective'],floor=r['efficiency_lower_bound'],certificate=r['certificate'])\nprint(json.dumps(out))"
data['linear_grid3_n20'] = json.loads(subprocess.check_output([sys.executable,'-c',measurement],cwd=checkout,env=dict(os.environ,PYTHONPATH=str(checkout/'src'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),text=True))
pathlib.Path('/content/w105-pair.json').write_text(json.dumps(data,allow_nan=False)+'\n')
