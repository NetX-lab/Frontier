"""Read-only image and mounted interpreter inspection; no simulator execution."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
mounted = Path('/data/ycfeng/envs/frontier-py310/bin/python')
repo = Path('/data/ycfeng/Frontier/.worktrees/step5-preview')
result = dict(image_python=sys.version, image_executable=sys.executable,
    image_platform=platform.platform(), mounted_python=str(mounted),
    mounted_is_symlink=mounted.is_symlink(), mounted_target=os.readlink(mounted),
    mounted_target_exists=mounted.exists(), repo_exists=repo.is_dir())
probe = "import sys,json; import frontier.main; print(json.dumps(dict(python=sys.version,executable=sys.executable,frontier_main_import=True)))"
for label, executable in [('canonical', str(mounted)), ('image', sys.executable)]:
    try:
        call=subprocess.run([executable,'-B','-c',probe],cwd=repo,
            env={**os.environ,'PYTHONPATH':str(repo),'PYTHONDONTWRITEBYTECODE':'1',
                'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                'NUMEXPR_NUM_THREADS':'1','TMPDIR':'/scratch','XDG_CACHE_HOME':'/scratch/cache'},
            capture_output=True,text=True,timeout=35)
        result[label]=dict(exit=call.returncode,stdout=call.stdout,stderr=call.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        result[label]=dict(exception=type(error).__name__,message=str(error))
result['image_packages']={}
for name in ['numpy','pandas','scipy','scikit-learn','plotly','torch','vllm','cloudpickle']:
    try: result['image_packages'][name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: result['image_packages'][name]=None
print(json.dumps(result,indent=2))
