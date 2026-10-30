"""Issue #18 bounded G1-G5 rerun; production source remains frozen."""
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import torch
from torchoed import c_optimality
from m2b_multigpu_condition import run_condition

def main(root):
    root = Path(root)
    path = Path('/content/m2b-gpu.json')
    report = dict(status='PASS', source_sha=os.environ['SAE_CANDIDATE_SHA'],
                  scientific_status='IN_PROGRESS', G1={}, G2=[], G4=[], G5=[])
    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    def timing(fn, gpu=False):
        for _ in range(3):
            fn()
        samples = []
        for _ in range(5):
            if gpu:
                torch.cuda.synchronize()
            begin = time.perf_counter()
            fn()
            if gpu:
                torch.cuda.synchronize()
            samples.append(time.perf_counter()-begin)
        return statistics.median(samples)
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        report['observed'] = json.loads(os.environ['SAE_DEVICE_OBSERVED'])
        report['environment'] = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
            cuda=torch.version.cuda, python=sys.version, cpu_threads=4, tf32=False,
            driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version',
                                            '--format=csv,noheader'], text=True).strip())
        env = dict(os.environ, M2B_TEST_DEVICE='cuda', M2B_ERRORS=str(root/'errors.json'),
                   PYTHONPATH=os.pathsep.join([str(root/'src'), str(root/'tests')]))
        result = subprocess.run([sys.executable,'-B','-m','pytest','tests/test_m2b.py',
            '-q','-p','no:cacheprovider'], cwd=root, env=env, capture_output=True, text=True, timeout=180)
        report['G1'] = dict(status='PASS' if result.returncode==0 else 'FAIL',
                            output=result.stdout+result.stderr)
        assert result.returncode==0, 'G1 numerical failure; frozen src'
        report['G3-C'] = run_condition(('cpu','cuda'))
        report['G3'] = report['G3-C']['status']
        save()
        assert report['G3-C']['status']!='FAIL', 'G3-C numerical failure; frozen src'
        for p in (3,10,30,100):
            for B in (1,16,256,4096,16384):
                generator = torch.Generator().manual_seed(182000+p*10000+B)
                q = torch.randn(B,p,p,generator=generator,dtype=torch.float64)/p**.5
                hostM = q.mT@q+torch.eye(p,dtype=torch.float64)
                hostc = torch.randn(B,p,generator=generator,dtype=torch.float64)
                del q
                M,c = hostM.cuda(),hostc.cuda()
                torch.cuda.reset_peak_memory_stats()
                v = c_optimality(M,c,max_elements=M.numel())
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                z = torch.linalg.solve(hostM,hostc.unsqueeze(-1)).squeeze(-1)
                reference = -(hostc*z).sum(-1)
                error = ((v.cpu()-reference).abs()/reference.abs()).max().item()
                report['G2'].append(dict(p=p,B=B,criterion_peak_bytes=peak,
                    max_relative_error=error,status='PASS' if error<=1e-9 else 'FAIL'))
                assert error<=1e-9, 'G2 dense CPU oracle failure; frozen src'
                del v
                torch.cuda.reset_peak_memory_stats()
                mg,cg = M.detach().requires_grad_(),c.detach().requires_grad_()
                dm,dc = torch.autograd.grad(c_optimality(mg,cg,max_elements=M.numel()).sum(),(mg,cg))
                expected_m = z[:,:,None]*z[:,None,:]
                em = ((dm.cpu()-expected_m).abs().amax()/expected_m.abs().amax()).item()
                ec = ((dc.cpu()+2*z).abs().amax()/(2*z).abs().amax()).item()
                report['G4'].append(dict(p=p,B=B,relative_inf_M=em,relative_inf_c=ec,
                    status='PASS' if max(em,ec)<=1e-8 else 'FAIL',
                    diagnostic_peak_bytes=torch.cuda.max_memory_allocated()))
                assert max(em,ec)<=1e-8, 'G4 gradient failure; frozen src'
                del mg,cg,dm,dc,z,expected_m,reference
                # CPU oracle/gradient diagnostics use at most the GO's 16,384 matrices.
                for dtype in (torch.float64,torch.float32):
                    cpuM,cpuc = hostM.to(dtype),hostc.to(dtype)
                    del M,c
                    gc.collect(); torch.cuda.empty_cache()
                    M,c = cpuM.cuda(),cpuc.cuda()
                    gpu = timing(lambda:c_optimality(M,c,max_elements=M.numel()),True)
                    cpu = timing(lambda:c_optimality(cpuM,cpuc,max_elements=cpuM.numel()))
                    del M,c
                    gc.collect(); torch.cuda.empty_cache()
                    transfer = timing(lambda:c_optimality(cpuM.cuda(),cpuc.cuda(),
                                                         max_elements=cpuM.numel()).cpu(),True)
                    report['G5'].append(dict(p=p,B=B,dtype=str(dtype),cpu_seconds=cpu,
                        gpu_seconds=gpu,gpu_with_transfer_seconds=transfer,warmup=3,repeats=5))
                    M,c = cpuM.cuda(),cpuc.cuda()
                del M,c,hostM,hostc,cpuM,cpuc
                gc.collect(); torch.cuda.empty_cache(); save()
        report['scientific_status'] = 'PASS_CONSTRUCTIBLE_GRID_WITH_BLOCKS'
    except Exception as exc:
        report.update(scientific_status='FAIL' if isinstance(exc,AssertionError) else 'BLOCKED',
                      failure_type=type(exc).__name__,failure=str(exc))
    finally:
        report['break_even'] = []
        for p in (3,10,30,100):
            for dtype in ('torch.float64','torch.float32'):
                rows = sorted((r for r in report['G5'] if r['p']==p and r['dtype']==dtype),key=lambda r:r['B'])
                i = next((i for i,r in enumerate(rows) if r['gpu_with_transfer_seconds']<r['cpu_seconds']),None)
                report['break_even'].append(dict(p=p,dtype=dtype,interval_open_closed=None if i is None
                    else [0 if i==0 else rows[i-1]['B'],rows[i]['B']]))
        save()
    return path
