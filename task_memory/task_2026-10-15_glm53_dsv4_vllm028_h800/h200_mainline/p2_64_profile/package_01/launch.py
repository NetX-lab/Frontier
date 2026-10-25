"""Submit the prepared native package; run only by the root agent."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / 'manifest.json').read_text())
for name, expected in manifest['files'].items():
    if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f'Prepared input changed: {name}')
config = json.loads((HERE / 'config.json').read_text())
assert (config['gpu_type'], config['charged_group'], config['gpu'], config['replica']) == ('H200', 'step_main', 8, 8)
assert config['engine']['data_parallel_size'] == 4
assert config['engine']['load_format'] == 'dummy'
assert Path(config['observer_bootstrap']).name == 'native_dummy_bootstrap.py'
assert Path(config['overlay_manifest']).name == 'overlay.json'
assert config['mode'] == 'clean'
assert config['serve_observers'] == {'prefill': None, 'decode': None}
assert json.loads((HERE / 'serve_observers.json').read_text()) is None
if sys.argv[1:] == ['--check-only']:
    print('INPUT_HASHES_PASS_NO_SUBMISSION')
    raise SystemExit(0)
if sys.argv[1:]:
    raise ValueError('Only --check-only or no arguments are supported')
assert os.environ['STEPMIND_BACKEND'] == 'rjob'
sys.path[:0] = [str(HERE), str(HERE / 'tests/comparison/dp_placement_pp')]
from native_pd_worker import write_run_manifest
run_manifest = write_run_manifest(config, os.environ['AUTHORIZATION'])
print('PRELAUNCH_MANIFEST', str(run_manifest), flush=True)
credentials = Path('/data/ycfeng/tmp/brainctl-auth-i-fengyicheng')
os.environ['BRAINPP_ACCESS_KEY'] = (credentials / 'accesskey_id').read_text().strip()
os.environ['BRAINPP_SECRET_KEY'] = (credentials / 'accesskey_secret').read_text().strip()
status = dict(start_utc=datetime.now(timezone.utc).isoformat(), launcher_pid=os.getpid(),
              config=str(HERE / 'config.json'))
(HERE / 'launcher_started.json').write_text(json.dumps(status, indent=2) + '\n')
sys.argv = ['submit_native_diagnostic.py', '--config', str(HERE / 'config.json')]
exit_code = 0
try:
    runpy.run_path(str(HERE / 'submit_native_diagnostic.py'), run_name='__main__')
except BaseException as error:
    exit_code = error.code if isinstance(error, SystemExit) and isinstance(error.code, int) else 1
    status['exception_type'] = type(error).__name__
    raise
finally:
    status.update(exit_code=exit_code, end_utc=datetime.now(timezone.utc).isoformat())
    (HERE / 'launcher_exit.json').write_text(json.dumps(status, indent=2) + '\n')
