"""Issue 131: profile the unchanged CPU A random500/n20 benchmark case."""
import cProfile, io, json, os, pathlib, pstats, subprocess, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
sha = os.environ['SAE_CANDIDATE_SHA']
root = pathlib.Path('/content/torchoed-w131')
subprocess.run(['git','clone','https://github.com/JohnnyZPiggy/TorchOED.git',str(root)],check=True,capture_output=True)
subprocess.run(['git','checkout',sha],cwd=root,check=True,capture_output=True)
sys.path.insert(0,str(root/'src'))
import torch
from torchoed import optimal_design, polynomial_features
torch.set_num_threads(1)
C = 2*torch.rand(500,3,dtype=torch.float64,generator=torch.Generator().manual_seed(0))-1
F = polynomial_features(C,degree=2)
profile = cProfile.Profile()
result = profile.runcall(optimal_design,F,20,criterion='A',restarts=1,generator=torch.Generator().manual_seed(0))
stream = io.StringIO()
stats = pstats.Stats(profile,stream=stream).sort_stats('cumulative')
stats.print_stats()
pathlib.Path('/content/w131-profile.txt').write_text(stream.getvalue())
entries = [{'file':key[0], 'line':key[1], 'name':key[2], 'primitive_calls':v[0], 'calls':v[1], 'self_seconds':v[2], 'cumulative_seconds':v[3]} for key,v in stats.stats.items()]
pathlib.Path('/content/w131-profile.json').write_text(json.dumps(dict(status='PASS',source_sha=sha,production_base='4a5c60439b74b06ae05c77053c07d6decbc8805a',python=sys.version,torch=torch.__version__,device=str(F.device),threads=torch.get_num_threads(),ids=result['ids'].tolist(),entries=entries),allow_nan=False))
