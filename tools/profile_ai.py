"""CPU A/I scale profile: python tools/profile_ai.py M CRITERION OUTPUT [SECONDS]."""
import _thread
import cProfile
import json
from pathlib import Path
import platform
import pstats
import subprocess
import sys
import threading
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from torchoed import optimal_design, polynomial_features

torch.set_num_threads(1)
m, criterion, output = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
F = polynomial_features(2*torch.rand(m, 5, dtype=torch.float64,
    generator=torch.Generator().manual_seed(0))-1, degree=2)
row = dict(m=m, criterion=criterion, threads=1, max_elements=20_000_000,
    python=platform.python_version(), torch=torch.__version__, platform=platform.platform(),
    cpu=platform.processor(), source_sha=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=Path(sys.modules[optimal_design.__module__].__file__).resolve().parents[2], text=True).strip())
profile = cProfile.Profile()
timer = threading.Timer(float(sys.argv[4]) if len(sys.argv)>4 else 600, _thread.interrupt_main)
timer.daemon = True
timer.start()
start = time.perf_counter()
profile.enable()
try:
    result = optimal_design(F, 42, criterion=criterion, restarts=1,
        generator=torch.Generator().manual_seed(0), max_elements=row['max_elements'])
    row.update(status='PASS', objective=result['objective'], ids=result['ids'].tolist(),
        floor=result['efficiency_lower_bound'], stop_reason=result['stop_reason'])
except KeyboardInterrupt:
    row['status'] = 'TIMEOUT'
finally:
    profile.disable()
    timer.cancel()
row['seconds'] = time.perf_counter()-start
stats = pstats.Stats(profile)
times = {name: sum(v[3] for k,v in stats.stats.items() if k[2]==name)
    for name in ('_linear_scores', '_linear_pair_moves', '_linear_exchange', '_dual_floor')}
row['phases'] = dict(initial=row['seconds']-times['_linear_exchange']-times['_dual_floor'],
    one_swap=times['_linear_scores'], pairs=times['_linear_pair_moves'],
    verification=times['_linear_exchange']-times['_linear_scores']-times['_linear_pair_moves'],
    certificate=times['_dual_floor'])
row['phase_times_valid'] = row['status']=='PASS'
if not row['phase_times_valid']:
    row['interrupted_profile_phases'] = row.pop('phases')
    row['phases'] = None
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(row)+'\n', encoding='utf-8')
print(json.dumps(row), flush=True)
