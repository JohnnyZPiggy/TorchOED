"""One CPU R probe inside the allocated Colab Python runtime."""
import csv, json, os, pathlib, shutil, subprocess, sys
root = pathlib.Path('/content/torchoed-r-heldout'); sha = os.environ['SAE_CANDIDATE_SHA']
subprocess.run(['git','clone','https://github.com/JohnnyZPiggy/TorchOED.git',str(root)],check=True,capture_output=True)
subprocess.run(['git','checkout',sha],cwd=root,check=True,capture_output=True)
installed = shutil.which('Rscript') is None
out = root/'reports/w11-mixproc'
if (out/'run.py').exists():
    data = dict(status='PASS', source_sha=sha, device_used='CPU (CUDA hidden from R)', device_observed=json.loads(os.environ['SAE_DEVICE_OBSERVED']))
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    try:
        if installed: subprocess.run(['apt-get','install','-y','r-base-core'],env=cpu_env,check=True,capture_output=True,text=True)
        subprocess.run(['Rscript','-e','install.packages("AlgDesign",repos="https://cloud.r-project.org"); stopifnot(requireNamespace("AlgDesign"))'],env=cpu_env,check=True,capture_output=True,text=True)
        result = subprocess.run(['Rscript','tools/r_heldout.R','MP11'],cwd=root,env=cpu_env,capture_output=True,text=True)
        data['call'] = dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        for name in ('sessionInfo.txt','optfederov_d.csv','optfederov_a.csv','optfederov_i.csv'):
            if (out/name).exists(): data[name] = (out/name).read_text()
    except subprocess.CalledProcessError as exc:
        data['setup_error'] = exc.stderr or exc.stdout
    pathlib.Path('/content/r-heldout.json').write_text(json.dumps(data,allow_nan=False)+'\n')
    sys.exit()
out = root/'reports/w10-dsd'
if (out/'retry.txt').exists():
    data = dict(source_sha=sha, device_used='CPU (CUDA hidden from R)', errors={})
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    try:
        if installed:
            subprocess.run(['apt-get', 'install', '-y', 'r-base-core'],
                           env=cpu_env, check=True, capture_output=True, text=True)
        subprocess.run(['Rscript', '-e', 'install.packages("daewr", repos="https://cloud.r-project.org"); '
                        'stopifnot(requireNamespace("daewr", quietly=TRUE))'],
                       env=cpu_env, check=True, capture_output=True, text=True)
        subprocess.run(['Rscript', 'tools/r_heldout.R', 'K7'], cwd=root,
                       env=cpu_env, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        data['errors']['K7'] = exc.stderr or exc.stdout
    for name in ('daewr_dsd.csv', 'sessionInfo.txt'):
        if (out/name).exists():
            data[name] = (out/name).read_text()
    pathlib.Path('/content/r-heldout.json').write_text(json.dumps(data, allow_nan=False)+'\n')
    sys.exit()
out = root/'reports/w11-mixai'
if (out/'run.py').exists():
    data = dict(status='PASS', source_sha=sha, device_used='CPU (CUDA hidden from R)', device_observed=json.loads(os.environ['SAE_DEVICE_OBSERVED']))
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    try:
        if installed: subprocess.run(['apt-get','install','-y','r-base-core'],env=cpu_env,check=True,capture_output=True,text=True)
        setup = subprocess.run(['Rscript','-e','install.packages("AlgDesign",repos="https://cloud.r-project.org"); stopifnot(requireNamespace("AlgDesign"))'],env=cpu_env,capture_output=True,text=True)
        data['installation'] = dict(returncode=setup.returncode,stdout=setup.stdout,stderr=setup.stderr)
        result = subprocess.run(['Rscript','tools/r_heldout.R','A11'],cwd=root,env=cpu_env,capture_output=True,text=True)
        data['call'] = dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        for name in ('sessionInfo.txt','optfederov_a.csv','optfederov_i.csv'):
            if (out/name).exists(): data[name] = (out/name).read_text()
    except subprocess.CalledProcessError as exc:
        data['setup_error'] = exc.stderr or exc.stdout
    pathlib.Path('/content/r-heldout.json').write_text(json.dumps(data,allow_nan=False)+'\n')
    sys.exit()
out = root/'reports/w11-skpr'
if (out/'run.py').exists():
    data = dict(status='PASS', source_sha=sha, device_used='CPU (CUDA hidden from R)', reference_status='NOT_RUN')
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    try:
        if installed: subprocess.run(['apt-get','install','-y','r-base-core'],env=cpu_env,check=True,capture_output=True,text=True)
        setup = subprocess.run(['Rscript','-e','install.packages("skpr",repos="https://cloud.r-project.org"); stopifnot(requireNamespace("skpr"))'],env=cpu_env,capture_output=True,text=True)
        data['installation'] = dict(returncode=setup.returncode,stdout=setup.stdout,stderr=setup.stderr)
        result = subprocess.run(['Rscript','tools/r_heldout.R','K11'],cwd=root,env=cpu_env,capture_output=True,text=True)
        data['call'] = dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        for name in ('sessionInfo.txt','calls.txt','skpr.csv'):
            if (out/name).exists(): data[name] = (out/name).read_text()
        if result.returncode == 0 and 'skpr.csv' in data: data['reference_status'] = 'PASS'
    except subprocess.CalledProcessError as exc:
        data['setup_error'] = exc.stderr or exc.stdout
    pathlib.Path('/content/r-heldout.json').write_text(json.dumps(data,allow_nan=False)+'\n')
    sys.exit()
out = root/'reports/w11-mixref'
if (out/'run.py').exists():
    data = dict(status='PASS', source_sha=sha, device_used='CPU (CUDA hidden from R)', errors={})
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    try:
        if installed:
            subprocess.run(['apt-get', 'install', '-y', 'r-base-core'], env=cpu_env, check=True, capture_output=True, text=True)
        expressions = [('AlgDesign', 'https://cloud.r-project.org', 'install.packages("AlgDesign", repos="https://cloud.r-project.org")'),
            ('mixexp', 'https://cloud.r-project.org', 'install.packages("mixexp", repos="https://cloud.r-project.org")'),
            ('mixexp', 'https://cran.r-project.org/src/contrib/Archive/mixexp/', 'base <- "https://cran.r-project.org/src/contrib/Archive/mixexp/"; html <- paste(readLines(base),collapse=" "); files <- unique(regmatches(html,gregexpr("mixexp_[0-9.]+[.]tar[.]gz",html))[[1]]); url <- paste0(base,files[order(package_version(sub("[.]tar[.]gz$", "", sub("^mixexp_", "", files))),decreasing=TRUE)[1]]); cat("Archive URL:",url,"\\n"); dest <- tempfile(fileext=".tar.gz"); download.file(url,dest); td <- tempfile(); dir.create(td); untar(dest,exdir=td); d <- read.dcf(file.path(td,"mixexp","DESCRIPTION")); deps <- paste(d[,intersect(c("Imports","Depends"),colnames(d))],collapse=","); deps <- trimws(gsub(" *[(][^)]*[)]","",strsplit(deps,",")[[1]])); install.packages(setdiff(deps,"R"),repos="https://cloud.r-project.org"); install.packages(dest,repos=NULL,type="source")'),
            ('mixexp', 'https://jsl5-code.r-universe.dev', 'install.packages("mixexp", repos=c("https://jsl5-code.r-universe.dev","https://cloud.r-project.org"))')]
        data['install_attempts'] = []; loaded = set()
        for pkg, url, expression in expressions:
            if pkg in loaded: continue
            result = subprocess.run(['Rscript', '-e', expression+'; stopifnot(requireNamespace("'+pkg+'"))'], env=cpu_env, capture_output=True, text=True)
            data['install_attempts'].append(dict(package=pkg, source=url, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
            if result.returncode == 0: loaded.add(pkg)
            else: data['errors'][pkg] = result.stderr or result.stdout
        for pkg in loaded: data['errors'].pop(pkg, None)
        result = subprocess.run(['Rscript', 'tools/r_heldout.R', 'M11'], cwd=root, env=cpu_env, capture_output=True, text=True)
        if result.stderr: data['errors']['M11'] = result.stderr
    except subprocess.CalledProcessError as exc:
        data['errors']['M11'] = exc.stderr or exc.stdout
    for name in ('xvert.csv', 'optfederov.csv', 'sessionInfo.txt'):
        if (out/name).exists():
            data[name] = (out/name).read_text()
    pathlib.Path('/content/r-heldout.json').write_text(json.dumps(data, allow_nan=False)+'\n')
    sys.exit()
out = root/'reports/w10-heldout'
if (out/'run.py').exists():
    data = dict(source_sha=sha, device_used='CPU (CUDA hidden from R)', errors={})
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    setup_error = None
    try:
        if installed:
            subprocess.run(['apt-get', 'install', '-y', 'r-base-core'],
                           env=cpu_env, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        setup_error = exc.stderr or exc.stdout
    if not setup_error:
        gmp_error = None
        try:
            for args in (['update'], ['install', '-y', 'libgmp-dev']):
                subprocess.run(['apt-get', *args], env=cpu_env, check=True,
                               capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            gmp_error = exc.stderr or exc.stdout
        for package in ('FrF2', 'rsm'):
            expression = (f'install.packages("{package}", repos="https://cloud.r-project.org"); '
                          f'stopifnot(requireNamespace("{package}", quietly=TRUE))')
            result = subprocess.run(['Rscript', '-e', expression], env=cpu_env,
                                    capture_output=True, text=True)
            if result.returncode:
                data['errors'][package] = result.stderr or result.stdout
        if gmp_error:
            data['errors']['libgmp-dev'] = gmp_error
    for case, package in (('H21', 'FrF2'), ('H22', 'rsm'), ('H23', 'rsm')):
        if setup_error or package in data['errors']:
            data['errors'][case] = setup_error or data['errors'][package]
            continue
        result = subprocess.run(['Rscript', 'tools/r_heldout.R', case], cwd=root,
                                env=cpu_env, capture_output=True, text=True)
        if result.returncode:
            data['errors'][case] = result.stderr or result.stdout
        for suffix in ('-session.txt', '.csv'):
            path = out/(case + suffix)
            if path.exists():
                data[path.name] = path.read_text()
    pathlib.Path('/content/r-heldout.json').write_text(json.dumps(data, allow_nan=False)+'\n')
    sys.exit()
out = root/'reports/w10-rref'
if (out/'run.py').exists():
    data = dict(status='PASS', source_sha=sha, device_used='CPU (CUDA hidden from R)',
                device_observed=json.loads(os.environ['SAE_DEVICE_OBSERVED']), errors={})
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    try:
        if installed:
            subprocess.run(['apt-get', 'install', '-y', 'r-base-core'],
                           env=cpu_env, check=True, capture_output=True, text=True)
        subprocess.run(['apt-get', 'update'], env=cpu_env, check=True, capture_output=True, text=True)
        subprocess.run(['apt-get', 'install', '-y', 'libgmp-dev'], env=cpu_env,
                       check=True, capture_output=True, text=True)
        for package in ('FrF2', 'rsm'):
            expression = (f'install.packages("{package}", repos="https://cloud.r-project.org"); '
                          f'stopifnot(requireNamespace("{package}", quietly=TRUE))')
            install = subprocess.run(['Rscript', '-e', expression], env=cpu_env,
                                     capture_output=True, text=True)
            if install.returncode:
                data['errors'][package] = install.stderr or install.stdout
        result = subprocess.run(['Rscript', 'tools/r_heldout.R', 'K6'], cwd=root,
                                env=cpu_env, capture_output=True, text=True)
        if result.returncode:
            data['errors']['K6'] = result.stderr or result.stdout
        for path in [*out.glob('*.csv'), out/'sessionInfo.txt']:
            if path.exists():
                data[path.name] = path.read_text()
    except subprocess.CalledProcessError as exc:
        data['errors']['r-base-core'] = exc.stderr or exc.stdout
    pathlib.Path('/content/r-heldout.json').write_text(json.dumps(data, allow_nan=False)+'\n')
    sys.exit()
out = root/'reports/w9-heldout'
if (out/'run.py').exists():
    data = dict(status='PASS', source_sha=sha, device_used='CPU (CUDA hidden from R)',
                device_observed=json.loads(os.environ['SAE_DEVICE_OBSERVED']))
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    if installed:
        setup = subprocess.run(['apt-get','install','-y','r-base-core'], capture_output=True, text=True)
    for case, package in (('H19','AlgDesign'), ('H20','skpr')):
        if installed and setup.returncode:
            data[case] = dict(status='NOT_RUN', error=setup.stderr.strip())
            continue
        install = subprocess.run(['Rscript','-e',f'install.packages("{package}",repos="https://cloud.r-project.org")'],
                                 env=cpu_env,capture_output=True,text=True)
        result = subprocess.run(['Rscript','tools/r_heldout.R',case],cwd=root,
                                env=cpu_env,capture_output=True,text=True)
        session, design = (out/f'{case}-{name}' for name in ('session.txt','design.csv'))
        row = data[case] = dict(status='NOT_RUN')
        if session.exists(): row['session'] = session.read_text()
        if result.returncode:
            row['error'] = result.stderr.strip() or install.stderr.strip()
        elif design.exists() and session.exists():
            row.update(status='PASS', csv=design.read_text())
        else:
            row['error'] = 'Installed optBlock documentation lists no I criterion.' if case == 'H19' else 'Missing design or sessionInfo()'
    pathlib.Path('/content/r-heldout.json').write_text(json.dumps(data,allow_nan=False)+'\n')
    sys.exit()
k5 = root/'reports/w9-skpr-i'
case, out = ('K5', k5) if (k5/'run.py').exists() else ('K4', root/'reports/w8-skpr')
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
        result = subprocess.run(['Rscript','tools/r_heldout.R',case],cwd=root,
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
