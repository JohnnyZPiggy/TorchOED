"""Eleven same-host cells; transport PASS preserves scientific failures."""
import json, os, pathlib, signal, subprocess, sys, time

def cell(mode, n, device):
    import torch
    from torchoed import lhs, fast_flexible_filling, space_filling_diagnostics
    A, b = torch.ones((1, 3), dtype=torch.float64, device=device), torch.tensor([1.5], dtype=torch.float64, device=device)
    if device == 'cuda': torch.cuda.synchronize()
    start = time.perf_counter()
    X = (fast_flexible_filling(50, 3, A=A, b=b, n_candidates=n, seed=0, device=device) if mode == 'fff' else
         lhs(n, 5, mode=mode, restarts=1, max_iter=20, generator=torch.Generator(device).manual_seed(0), device=device))
    if device == 'cuda': torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    metrics = {k: float(v) for k, v in space_filling_diagnostics(X).items()}
    if mode == 'fff':
        validity = dict(max_violation=float((X @ A.T - b).max())); valid = validity['max_violation'] <= 0
    else:
        r = torch.round(n * X - 0.5)
        validity = dict(permutation=bool((r.sort(dim=0).values == torch.arange(n, device=device)[:, None]).all()), max_residual=float((n * X - 0.5 - r).abs().max()))
        valid = validity['permutation'] and validity['max_residual'] <= 1e-9
    return dict(cell=f'{mode}-{n}', device=device, seconds=seconds, diagnostics=metrics, validity=validity, scientific_status='PASS' if valid else 'FAIL')

def timeout(*_):
    raise TimeoutError('CPU call exceeded 600 s')

if __name__ == '__main__':
    sha = os.environ['SAE_CANDIDATE_SHA']; root = pathlib.Path('/content/torchoed-sfgpu')
    subprocess.run(['git', 'clone', 'https://github.com/JohnnyZPiggy/TorchOED.git', str(root)], check=True, capture_output=True)
    subprocess.run(['git', 'checkout', sha], cwd=root, check=True, capture_output=True); sys.path.insert(0, str(root / 'src'))
    data = dict(source_sha=sha, status='PASS', rows=[], skipped=[], device_observed=json.loads(os.environ['SAE_DEVICE_OBSERVED'])); signal.signal(signal.SIGALRM, timeout)
    for mode in ('maximin', 'maxpro', 'uniform', 'fff'):
        for n in ((2**14, 2**17) if mode == 'fff' else (50, 200, 400)):
            try:
                cell(mode, n, 'cuda'); signal.alarm(600)
                data['rows'].append(cell(mode, n, 'cpu')); signal.alarm(0)
                data['rows'].append(cell(mode, n, 'cuda'))
            except TimeoutError as error:
                signal.alarm(0); data['rows'].append(dict(cell=f'{mode}-{n}', device='cpu', scientific_status='TIMEOUT', error=str(error))); data['skipped'].append(dict(mode=mode, larger_than=n)); break
    data['scientific_status'] = 'PASS' if all(r['scientific_status']=='PASS' for r in data['rows']) else 'FAIL'
    pathlib.Path('/content/sf-gpu.json').write_text(json.dumps(data, allow_nan=False)+'\n')
