"""Issue18 G1-G5 driver, called only in an authorized CUDA session."""
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import torch
from torchoed import c_optimality


def main(root):
    root = Path(root)
    started = time.monotonic()
    deadline = started + 340  # Reserve transport's 420s workload window for final output.
    report = dict(source_sha=os.environ['SAE_CANDIDATE_SHA'], status='PASS',
                  transport_semantics='DELIVERY_ONLY_CHECK_SCIENTIFIC_STATUS',
                  scientific_status='IN_PROGRESS', G1={}, G2=[], G3=[], G4=[], G5=[])
    path = Path('/content/m2b-gpu.json')

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    def remaining():
        if time.monotonic() >= deadline:
            raise TimeoutError('FINITE_CONTROLLER_WORKLOAD_WINDOW')

    def timing(fn, gpu=False):
        for _ in range(3):
            remaining()
            fn()
        measured = []
        for _ in range(5):
            remaining()
            if gpu:
                torch.cuda.synchronize()
            begin = time.perf_counter()
            fn()
            if gpu:
                torch.cuda.synchronize()
            measured.append(time.perf_counter()-begin)
        return statistics.median(measured)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA_UNAVAILABLE')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.cuda.get_device_properties(0)
        driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=15)
        report['environment'] = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
            cuda=torch.version.cuda, python=sys.version, driver=driver.stdout.strip(),
            device_total_bytes=device.total_memory, cpu_threads=4, tf32=False)
        env = dict(os.environ, M2B_TEST_DEVICE='cuda', M2B_ERRORS=str(root/'gpu-errors.json'),
                   PYTHONPATH=os.pathsep.join([str(root/'src'), str(root/'tests')]))
        report['active_stage']='G1'; save()
        result = subprocess.run([sys.executable, '-B', '-m', 'pytest', 'tests/test_m2b.py',
                                 '-q', '-p', 'no:cacheprovider'], cwd=root, env=env,
                                capture_output=True, text=True, timeout=100)
        report['G1'] = dict(status='PASS' if result.returncode == 0 else 'FAIL',
                            exit_code=result.returncode, log=result.stdout+result.stderr,
                            oracle_errors=json.loads((root/'gpu-errors.json').read_text())
                              if (root/'gpu-errors.json').exists() else None)
        save()
        if result.returncode:
            raise AssertionError('G1_FAILED')

        # Reproduce each matrix identically on CPU/GPU; controlled orthogonal rotation.
        for exponent in range(15):
            remaining()
            report['active_stage']=f'G3/condition10^{exponent}'; save()
            seed = 18300+exponent
            gen = torch.Generator().manual_seed(seed)
            Q, _ = torch.linalg.qr(torch.randn(5, 5, generator=gen, dtype=torch.float64))
            eigenvalues = torch.logspace(0, -exponent, 5, dtype=torch.float64)
            M = (Q*eigenvalues)@Q.mT
            c = torch.ones(5, dtype=torch.float64)
            row = dict(seed=seed, target_condition=10**exponent)
            for label, where, dtype in [('cpu64','cpu',torch.float64),
                                        ('gpu64','cuda',torch.float64),('gpu32','cuda',torch.float32)]:
                try:
                    value = c_optimality(M.to(where,dtype), c.to(where,dtype))
                    row[label] = dict(accepted=True, score=float(value.item()))
                except ValueError as exc:
                    row[label] = dict(accepted=False, reason=str(exc))
            row['acceptance_disagreement'] = len({row[x]['accepted'] for x in ('cpu64','gpu64','gpu32')}) > 1
            report['G3'].append(row)
        report['G3_summary'] = {x:dict(rejected=sum(not row[x]['accepted'] for row in report['G3']),
                                         total=len(report['G3'])) for x in ('cpu64','gpu64','gpu32')}
        report['float32_first_rejected_target_condition'] = next(
            (r['target_condition'] for r in report['G3'] if not r['gpu32']['accepted']), None)
        save()

        # Increasing batches; stop only for measured memory, allocation failure or finite deadline.
        for p in (3, 10, 30, 100):
            B = 1
            while True:
                remaining()
                report['active_stage']=f'G2/p{p}/B{B}/allocate'; save()
                seed = 182000+p*10000+B
                gen = torch.Generator(device='cuda').manual_seed(seed)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                q = torch.randn(B,p,p,device='cuda',dtype=torch.float64,generator=gen)/math.sqrt(p)
                M = q.mT@q + torch.eye(p,device='cuda',dtype=torch.float64)
                del q
                c = torch.randn(B,p,device='cuda',dtype=torch.float64,generator=gen)
                limit = B*p*p
                v64 = c_optimality(M,c,max_elements=limit)
                M32, c32 = M.float(), c.float()
                v32 = c_optimality(M32,c32,max_elements=limit)
                # At least 10k matrix elements when the config contains that many.
                sample_count = min(B,max(10000,math.ceil(10000/(p*p))))
                sample = torch.linspace(0,B-1,sample_count,device='cuda').long()
                hostM, hostc = M[sample].cpu(), c[sample].cpu()
                reference = c_optimality(hostM,hostc,max_elements=hostM.numel())
                relative = (v64[sample].cpu()-reference).abs()/reference.abs().clamp_min(torch.finfo(torch.float64).tiny)
                report['active_stage']=f'G2/p{p}/B{B}/eigvalsh'; save()
                eigen = torch.linalg.eigvalsh(M)
                condition = eigen[:,-1]/eigen[:,0]
                normalized = (v32.double()-v64).abs()/v64.abs().clamp_min(torch.finfo(torch.float64).tiny)
                normalized /= torch.finfo(torch.float32).eps*condition
                report['G2'].append(dict(p=p,B=B,seed=seed,status='PASS' if relative.max().item()<=1e-9 else 'FAIL',
                    sampled_scores=sample_count,sampled_matrix_elements=hostM.numel(),
                    max_relative_error=relative.max().item(), normalized_fp32_max=normalized.max().item(),
                    normalized_fp32_p99=torch.quantile(normalized,.99).item(),float32_status='OBSERVED'))
                if relative.max().item()>1e-9:
                    raise AssertionError('G2_FP64_PARITY_FAILED')
                del eigen,condition,normalized,hostM,hostc,reference

                report['active_stage']=f'G4/p{p}/B{B}'; save()
                mg=M.detach().requires_grad_(); cg=c.detach().requires_grad_()
                dm,dc=torch.autograd.grad(c_optimality(mg,cg,max_elements=limit).sum(),(mg,cg))
                z=torch.linalg.solve(M,c.unsqueeze(-1)).squeeze(-1)
                expectedM=z[:,:,None]*z[:,None,:]; expectedc=-2*z
                # Global relative infinity norm avoids zero-entry division artifacts.
                errorM=((dm-expectedM).abs().amax()/expectedM.abs().amax()).item()
                errorc=((dc-expectedc).abs().amax()/expectedc.abs().amax()).item()
                report['G4'].append(dict(p=p,B=B,status='PASS' if max(errorM,errorc)<=1e-8 else 'FAIL',
                                         relative_inf_M=errorM,relative_inf_c=errorc))
                if max(errorM,errorc)>1e-8:
                    raise AssertionError('G4_ANALYTIC_FAILED')
                del mg,cg,dm,dc,z,expectedM,expectedc
                diagnostic_peak=torch.cuda.max_memory_allocated()

                # Equal full inputs and criterion/validation semantics, float32 and float64.
                for dtype, gpuM, gpuc in [('float64',M,c),('float32',M32,c32)]:
                    remaining()
                    report['active_stage']=f'G5/p{p}/B{B}/{dtype}'; save()
                    cpuM,cpuc=gpuM.cpu(),gpuc.cpu()
                    torch.cuda.reset_peak_memory_stats()
                    cpu=timing(lambda:c_optimality(cpuM,cpuc,max_elements=limit))
                    gpu=timing(lambda:c_optimality(gpuM,gpuc,max_elements=limit),True)
                    transfer=timing(lambda:c_optimality(cpuM.cuda(),cpuc.cuda(),max_elements=limit).cpu(),True)
                    report['G5'].append(dict(p=p,B=B,dtype=dtype,cpu_seconds=cpu,gpu_seconds=gpu,
                        gpu_including_H2D_D2H_seconds=transfer,warmup=3,repeats=5,statistic='median',
                        cpu_threads=4,peak_max_memory_allocated=torch.cuda.max_memory_allocated(),
                        includes_generation=False,scope='MEASURED_CONFIG_ONLY'))
                    del cpuM,cpuc
                peak=max(diagnostic_peak,torch.cuda.max_memory_allocated())
                report['G2'][-1].update(peak_bytes=peak,peak_fraction=peak/device.total_memory)
                save()
                del M,c,M32,c32,v64,v32,sample,gpuM,gpuc
                gc.collect(); torch.cuda.empty_cache()
                if peak >= .68*device.total_memory:
                    break
                # Double until the next measured-peak extrapolation crosses 70%;
                # interpolate that final batch rather than overshooting device capacity.
                B=min(B*2,max(B+1,int(B*.70*device.total_memory/peak)))
        report['scientific_status']='PASS'
        report['active_stage']='COMPLETE'
    except Exception as exc:
        report['scientific_status']='FAIL' if isinstance(exc,AssertionError) else 'BLOCKED'
        report['failure_type']=type(exc).__name__
        report['failure']=str(exc)
        report['traceback']=traceback.format_exc()
        report['uncompleted_configs']='NOT_RUN'
    finally:
        # No unmeasured extrapolation: first measured batch with GPU median below CPU median.
        report['break_even_measured_B']={}
        for p in (3,10,30,100):
            for dtype in ('float64','float32'):
                rows=[r for r in report['G5'] if r['p']==p and r['dtype']==dtype]
                report['break_even_measured_B'][f'{p}/{dtype}']={
                    'without_transfer':next((r['B'] for r in rows if r['gpu_seconds']<r['cpu_seconds']),None),
                    'with_transfer':next((r['B'] for r in rows if r['gpu_including_H2D_D2H_seconds']<r['cpu_seconds']),None),
                    'missing_means':'NOT_OBSERVED_IN_MEASURED_CONFIGS'}
        report['elapsed_seconds']=time.monotonic()-started
        save()
    return path
