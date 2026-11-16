"""One CPU R probe inside the allocated Colab Python runtime."""
import csv, json, os, pathlib, shutil, subprocess, sys
root = pathlib.Path('/content/torchoed-r-heldout'); sha = os.environ['SAE_CANDIDATE_SHA']
subprocess.run(['git','clone','https://github.com/JohnnyZPiggy/TorchOED.git',str(root)],check=True,capture_output=True)
subprocess.run(['git','checkout',sha],cwd=root,check=True,capture_output=True)
installed = shutil.which('Rscript') is None
out = root/'reports/w8-skpr'
if (out/'run.py').exists():
    host = json.loads(os.environ['SAE_DEVICE_OBSERVED'])['nvidia_smi'][0]['name']
    data = dict(status='PASS', source_sha=sha, observed_host=host,
                device_used='CPU (CUDA hidden from R)', reference_status='NOT_RUN')
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    try:
        if installed: subprocess.run(['apt-get','install','-y','r-base-core'],check=True)
        data['r_version'] = subprocess.check_output(['R','--version'],text=True).splitlines()[0]
        subprocess.run(['Rscript','-e','install.packages("skpr",repos="https://cloud.r-project.org")'],
                       env=cpu_env,check=True)
        result = subprocess.run(['Rscript','tools/r_heldout.R','K4'],cwd=root,
                                env=cpu_env,capture_output=True,text=True)
        for name in ('sessionInfo.txt', 'example-lines.txt', 'candidates.csv', 'skpr.csv', 'model.txt'):
            path = out/name
            if path.exists(): data[name] = path.read_text()
        if result.returncode == 0:
            data['reference_status'] = 'PASS'
        else:
            data['reference_error'] = result.stderr.strip()
    except subprocess.CalledProcessError as exc:
        data['reference_error'] = str(exc)
    pathlib.Path('/content/r-heldout.json').write_text(json.dumps(data,allow_nan=False)+'\n')
    sys.exit()
out = root/'reports/w7-heldout'
if (out/'H14.csv').exists():
    data = dict(status='PASS', source_sha=sha, runtime='CPU Rscript in Colab Python runtime',
                device_observed=json.loads(os.environ['SAE_DEVICE_OBSERVED']), reference_status='NOT_RUN')
    try:
        if installed: subprocess.run(['apt-get','install','-y','r-base-core'],check=True)
        data['r_version'] = subprocess.check_output(['R','--version'],text=True).splitlines()[0]
        subprocess.run(['Rscript','-e','install.packages("skpr",repos="https://cloud.r-project.org")'],check=True)
        result = subprocess.run(['Rscript','tools/r_heldout.R','H14'],cwd=root,
                                env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
        session = out/'sessionInfo.txt'
        if result.returncode == 0 and session.exists():
            with (out/'skpr.csv').open() as file:
                rows = list(csv.DictReader(file))
            data.update(reference_status='PASS', sessionInfo=session.read_text(),
                        rows_by_plot=[[[float(row[k]) for k in 'ABC'] for row in rows if int(row['plot']) == p]
                                      for p in range(1,7)])
        else:
            data['reference_error'] = result.stderr.strip() or 'Missing sessionInfo()'
            if session.exists(): data['sessionInfo'] = session.read_text()
    except subprocess.CalledProcessError as exc:
        data['reference_error'] = str(exc)
    pathlib.Path('/content/r-heldout.json').write_text(json.dumps(data,allow_nan=False)+'\n')
    sys.exit()
if installed: subprocess.run(['apt-get','install','-y','r-base-core'],check=True)
version = subprocess.check_output(['R','--version'],text=True)
subprocess.run(['Rscript','-e','install.packages("AlgDesign",repos="https://cloud.r-project.org")'],check=True)
out = root/'reports/w6-heldout'
if (out/'H11.csv').exists():
    result = subprocess.run(['Rscript','tools/r_heldout.R','H11'],cwd=root,
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
    session = out/'sessionInfo.txt'
    data = dict(status='PASS', source_sha=sha, runtime='CPU Rscript in Colab Python runtime',
                r_install='apt-get r-base-core' if installed else 'preinstalled Rscript',
                r_version=version, device_observed=json.loads(os.environ['SAE_DEVICE_OBSERVED']),
                reference_status='PASS' if result.returncode == 0 and session.exists() else 'NOT_RUN',
                sessionInfo=session.read_text() if session.exists() else None)
    if data['reference_status'] == 'PASS':
        data['algdesign'] = json.loads((out/'algdesign_out.json').read_text())
    else:
        data['reference_error'] = result.stderr.strip() or 'Missing sessionInfo()'
    pathlib.Path('/content/r-heldout.json').write_text(json.dumps(data,allow_nan=False)+'\n')
    sys.exit()
subprocess.run(['Rscript','tools/r_heldout.R'],cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
out = root/'reports/w3-heldout/final'
data = dict(status='PASS',source_sha=sha,runtime='Rscript in Colab Python runtime',r_install='apt-get r-base-core' if installed else 'preinstalled Rscript',r_version=version,device_observed=json.loads(os.environ['SAE_DEVICE_OBSERVED']),algdesign=json.loads((out/'algdesign_out.json').read_text()),sessionInfo=(out/'sessionInfo.txt').read_text())
pathlib.Path('/content/r-heldout.json').write_text(json.dumps(data,allow_nan=False)+'\n')
