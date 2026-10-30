"""Source-pinned rerun, frozen src; sample diagnostics separately from full criterion."""
import gc
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

import torch
from torchoed import c_optimality
from m2b_multigpu_condition import run_condition
from m2b_multigpu_plan import next_batch, sample_positions, break_even_union


def main(root, start_p=3, start_B=1, reuse_condition=False):
    root=Path(root)
    source=os.environ['SAE_CANDIDATE_SHA']
    device_requested=os.environ['SAE_DEVICE_REQUESTED']
    started=time.monotonic();deadline=started+3300
    path=Path('/content/m2b-multigpu.json')
    condition_path=Path('/content/m2b-g3c.json')
    g1_path=Path('/content/m2b-g1.json')
    report=dict(status='PASS',source_sha=source,scientific_status='IN_PROGRESS',
                G2=[],G3=[],G4=[],G5=[],requested=device_requested,
                diagnostic_limit=10000,criterion_target_fraction=.70,
                construction_policy='STRICT_RATIONAL_GEOMETRIC_SCIENTIST_CLARIFICATION',
                controller_observed=json.loads(os.environ['SAE_DEVICE_OBSERVED']))
    report['workload_plan']=dict(start_p=start_p,start_B=start_B,reuse_condition=reuse_condition)
    def save():path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def stage(name):report['active_stage']=name;save()
    def check():
        if time.monotonic()>=deadline:raise TimeoutError('FINITE_MATRIX_WINDOW3300')
    def timing(fn,gpu):
        for _ in range(3):check();fn()
        samples=[]
        for _ in range(5):
            check()
            if gpu:torch.cuda.synchronize()
            before=time.perf_counter();fn()
            if gpu:torch.cuda.synchronize()
            samples.append(time.perf_counter()-before)
        return dict(median=statistics.median(samples),samples=samples,warmup=3,repeats=5)
    def available_host_bytes():
        values={line.split(':')[0]:int(line.split()[1])*1024
                for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
        return values['MemAvailable'],values['MemTotal']
    g1=dict(status='PASS',source_sha=source,scientific_status='NOT_RUN_REUSED_ACCEPTED_T4')
    condition=dict(status='PASS',source_sha=source,scientific_status='NOT_RUN')
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA_UNAVAILABLE')
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        properties=torch.cuda.get_device_properties(0)
        smi=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],
                           capture_output=True,text=True,check=True,timeout=15)
        cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                  if line.startswith('model name')),platform.processor())
        available,total_host=available_host_bytes()
        report['environment']=dict(gpu=torch.cuda.get_device_name(0),gpu_bytes=properties.total_memory,
            cpu_model=cpu,cpu_threads=4,host_total_bytes=total_host,host_available_bytes=available,
            torch=torch.__version__,cuda=torch.version.cuda,driver=smi.stdout.strip(),python=sys.version,
            cuda_arches=torch.cuda.get_arch_list(),tf32=False)
        if device_requested!='T4':
            stage('G1')
            env=dict(os.environ,M2B_TEST_DEVICE='cuda',M2B_ERRORS=str(root/'oracle-errors.json'),
                     PYTHONPATH=os.pathsep.join([str(root/'src'),str(root/'tests')]))
            result=subprocess.run([sys.executable,'-B','-m','pytest','tests/test_m2b.py','-q','-p','no:cacheprovider'],
                                  cwd=root,env=env,capture_output=True,text=True,timeout=180)
            g1.update(scientific_status='PASS' if result.returncode==0 else 'FAIL',exit_code=result.returncode,
                      log=result.stdout+result.stderr,errors=json.loads((root/'oracle-errors.json').read_text())
                       if (root/'oracle-errors.json').exists() else None)
            g1_path.write_text(json.dumps(g1,indent=2)+'\n')
            if result.returncode:raise AssertionError('G1_FAILED_STOP_FROZEN_SRC')
        stage('G3-C')
        def save_condition(data):
            data=dict(data,status='PASS',scientific_status=data['status'],source_sha=source)
            condition_path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
        condition=(dict(status='NOT_RUN_REUSED_ACCEPTED_T4',reused_from='t4-fail-1',
                        source_sha='ebf0ee6aa23ba9ef2855ad4986cb6cca0b0ad1b9') if reuse_condition
                   else run_condition(('cpu','cuda'),check,save_condition))
        save_condition(condition)
        if condition['status']=='FAIL':raise AssertionError('G3C_NUMERICAL_FAIL_STOP_FROZEN_SRC')
        if device_requested!='T4':
            stage('G3/legacy-controlled-conditioning')
            for exponent in range(15):
                check()
                seed=18300+exponent
                generator=torch.Generator().manual_seed(seed)
                q,_=torch.linalg.qr(torch.randn(5,5,generator=generator,dtype=torch.float64))
                eigenvalues=torch.logspace(0,-exponent,5,dtype=torch.float64)
                matrix=(q*eigenvalues)@q.mT
                contrast=torch.ones(5,dtype=torch.float64)
                observation=dict(seed=seed,target_condition=10**exponent)
                for label,where,dtype in [('cpu64','cpu',torch.float64),('cpu32','cpu',torch.float32),
                                          ('gpu64','cuda',torch.float64),('gpu32','cuda',torch.float32)]:
                    try:
                        value=c_optimality(matrix.to(where,dtype),contrast.to(where,dtype)).item()
                        observation[label]=dict(accepted=True,score=value)
                    except ValueError as exc:
                        observation[label]=dict(accepted=False,reason=str(exc))
                observation['acceptance_disagreement']=len({observation[k]['accepted']
                            for k in ('cpu64','cpu32','gpu64','gpu32')})>1
                report['G3'].append(observation)
            del q,matrix,contrast,eigenvalues
            save()
        for p in (3,10,30,100):
            if p<start_p:continue
            batch=start_B if p==start_p else 1
            while True:
                check();stage(f'G2/p{p}/B{batch}/generate')
                seed=182000+p*10000+batch
                generator=torch.Generator(device='cuda').manual_seed(seed)
                q=torch.randn(batch,p,p,device='cuda',dtype=torch.float64,generator=generator)/math.sqrt(p)
                M=q.mT@q+torch.eye(p,device='cuda',dtype=torch.float64);del q
                c=torch.randn(batch,p,device='cuda',dtype=torch.float64,generator=generator)
                limit=M.numel()
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                v64=c_optimality(M,c,max_elements=limit)
                torch.cuda.synchronize()
                criterion_peak=torch.cuda.max_memory_allocated()
                row=dict(p=p,B=batch,seed=seed,criterion_peak_bytes=criterion_peak,
                         criterion_peak_fraction=criterion_peak/properties.total_memory,
                         peak_scope='FULL_FP64_CRITERION_FORWARD_ONLY_INCLUDING_INPUT_OUTPUT',status='IN_PROGRESS')
                report['G2'].append(row);save()
                positions=torch.tensor(sample_positions(batch),device='cuda',dtype=torch.int64)
                hostM,hostc=M[positions].cpu(),c[positions].cpu()
                reference=c_optimality(hostM,hostc,max_elements=hostM.numel())
                small64=v64[positions].cpu()
                error=(small64-reference).abs()/reference.abs().clamp_min(torch.finfo(torch.float64).tiny)
                if error.max().item()>1e-9:raise AssertionError('G2_FP64_PARITY_FAIL_STOP_FROZEN_SRC')
                stage(f'G2/p{p}/B{batch}/float32')
                M32,c32=M.float(),c.float()
                v32=c_optimality(M32,c32,max_elements=limit)
                small32=v32[positions].double().cpu()
                full_raw_max=0.
                for start in range(0,batch,1048576):
                    part=(v32[start:start+1048576].double()-v64[start:start+1048576]).abs()
                    part/=v64[start:start+1048576].abs().clamp_min(torch.finfo(torch.float64).tiny)
                    full_raw_max=max(full_raw_max,part.max().item())
                del v32,M32,c32,v64,part
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                diagnostic_baseline=torch.cuda.memory_allocated()
                eigen=torch.linalg.eigvalsh(hostM)
                cond=eigen[:,-1]/eigen[:,0]
                normalized=(small32-small64).abs()/small64.abs().clamp_min(torch.finfo(torch.float64).tiny)
                normalized/=torch.finfo(torch.float32).eps*cond
                row.update(status='PASS',diagnostic_sample_matrices=len(positions),
                    sampled_matrix_elements=hostM.numel(),max_fp64_relative_error=error.max().item(),
                    fp32_full_batch_raw_relative_max=full_raw_max,
                    fp32_sample_normalized_max=normalized.max().item(),
                    fp32_sample_normalized_p99=torch.quantile(normalized,.99).item(),fp32_status='OBSERVED',
                    condition_backend='CPU eigvalsh on deterministic sample only')
                # Large-scale gradient diagnostic is capped independently of criterion scaling.
                if device_requested!='T4':
                    stage(f'G4/p{p}/B{batch}')
                    mg=hostM.cuda().requires_grad_();cg=hostc.cuda().requires_grad_()
                    dm,dc=torch.autograd.grad(c_optimality(mg,cg,max_elements=mg.numel()).sum(),(mg,cg))
                    torch.cuda.synchronize()
                    z=torch.linalg.solve(hostM,hostc.unsqueeze(-1)).squeeze(-1).cuda()
                    expected_m=z[:,:,None]*z[:,None,:];expected_c=-2*z
                    em=((dm-expected_m).abs().amax()/expected_m.abs().amax()).item()
                    ec=((dc-expected_c).abs().amax()/expected_c.abs().amax()).item()
                    report['G4'].append(dict(p=p,B=batch,sampled_matrices=len(positions),
                        relative_inf_M=em,relative_inf_c=ec,status='PASS' if max(em,ec)<=1e-8 else 'FAIL'))
                    if max(em,ec)>1e-8:raise AssertionError('G4_GRADIENT_FAIL_STOP_FROZEN_SRC')
                    del mg,cg,dm,dc,z,expected_m,expected_c
                row.update(diagnostic_peak_bytes=torch.cuda.max_memory_allocated(),
                           diagnostic_baseline_bytes=diagnostic_baseline)
                del hostM,hostc,reference,small64,small32,eigen,cond,normalized,error,positions
                gc.collect();torch.cuda.empty_cache();save()
                for dtype in ('float64','float32'):
                    check();stage(f'G5/p{p}/B{batch}/{dtype}')
                    if dtype=='float32':
                        # Preserve the same fp64 data, then cast on CUDA outside timing.
                        M=cpuM.cuda().float();c=cpuc.cuda().float()
                        del cpuM,cpuc
                        gc.collect();torch.cuda.empty_cache()
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    gpu=timing(lambda:c_optimality(M,c,max_elements=limit),True)
                    gpu_peak=torch.cuda.max_memory_allocated()
                    input_bytes=M.numel()*M.element_size()+c.numel()*c.element_size()
                    # Structural major-tensor bound from frozen criterion; not a quota.
                    host_forward_estimate=4*M.numel()*M.element_size()+3*c.numel()*c.element_size()+8*batch*8
                    available,total=available_host_bytes()
                    timing_row=dict(device=device_requested,p=p,B=batch,dtype=dtype,
                        measurement_protocol='UNLOADED_TRANSFER_V2',
                        gpu_seconds=gpu['median'],gpu_samples_seconds=gpu['samples'],warmup=3,repeats=5,
                        criterion_gpu_peak_bytes=gpu_peak,host_available_bytes=available,
                        host_forward_major_tensor_estimate_bytes=host_forward_estimate,
                        CPU_threads=4,status='IN_PROGRESS')
                    report['G5'].append(timing_row);save()
                    if input_bytes>available:
                        timing_row.update(status='BLOCKED_HOST_MEMORY',cpu_status='NOT_RUN',transfer_status='NOT_RUN')
                        raise RuntimeError('HOST_INPUT_STAGING_MEMORY_UNAVAILABLE')
                    else:
                        cpuM,cpuc=M.cpu(),c.cpu()
                        available_after,_=available_host_bytes()
                        if host_forward_estimate-input_bytes>available_after:
                            timing_row.update(status='BLOCKED_HOST_MEMORY',cpu_status='NOT_RUN')
                        else:
                            cpu=timing(lambda:c_optimality(cpuM,cpuc,max_elements=limit),False)
                            timing_row.update(status='PASS',cpu_seconds=cpu['median'],cpu_samples_seconds=cpu['samples'],
                                              cpu_status='PASS')
                        # Transaction timing starts with no preloaded duplicate GPU input.
                        del M,c
                        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                        timing_row['transfer_resident_baseline_bytes']=torch.cuda.memory_allocated()
                        transfer=timing(lambda:c_optimality(cpuM.cuda(),cpuc.cuda(),max_elements=limit).cpu(),True)
                        timing_row.update(gpu_with_transfer_seconds=transfer['median'],
                            gpu_with_transfer_samples_seconds=transfer['samples'],transfer_status='PASS',
                            transfer_peak_bytes=torch.cuda.max_memory_allocated())
                    save()
                del cpuM,cpuc
                gc.collect();torch.cuda.empty_cache()
                if criterion_peak>=.68*properties.total_memory:break
                batch=next_batch(batch,criterion_peak,properties.total_memory)
        report['scientific_status']=('PARTIAL_HOST_MEMORY_BLOCKS' if any(r['status']!='PASS' for r in report['G5'])
                                     else 'PASS_CONSTRUCTIBLE_GRID_WITH_BLOCKS')
        report['active_stage']='COMPLETE'
    except Exception as exc:
        report.update(scientific_status='FAIL' if isinstance(exc,AssertionError) else 'BLOCKED',
                      failure_type=type(exc).__name__,failure=str(exc),traceback=traceback.format_exc())
    finally:
        g1_path.write_text(json.dumps(g1,indent=2,allow_nan=False)+'\n')
        if not condition_path.exists():
            condition_path.write_text(json.dumps(dict(condition,status='PASS',source_sha=source),indent=2)+'\n')
        report['break_even_union']=break_even_union(report['G5'])
        report['elapsed_seconds']=time.monotonic()-started
        save()
    return [path,condition_path,g1_path]
