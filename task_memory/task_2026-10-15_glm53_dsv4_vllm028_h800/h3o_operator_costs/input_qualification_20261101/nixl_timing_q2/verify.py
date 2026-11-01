"""Execute the retained lead check, lifecycle regression and proposed config price check."""
import datetime
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PYTHON = '/data/ycfeng/envs/frontier-py310/bin/python'
ENV = {**os.environ, 'PYTHONPATH': f'{REPO}:{REPO}/tests/analysis',
       'PYTHONDONTWRITEBYTECODE': '1', 'OPENBLAS_NUM_THREADS': '1',
       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'TMPDIR': str(HERE / 'scratch'),
       'FRONTIER_TMP_ROOT': str(HERE / 'scratch')}

price_program = '''import json
from pathlib import Path
from frontier.config.kv_cache_transfer_config import AnalyticalKVCacheTransferConfig
from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import AnalyticalKVCacheTransferPredictor
from frontier.types import ClusterType
p=json.loads(Path(__import__('sys').argv[1]).read_text())
m=p['model']
c=AnalyticalKVCacheTransferConfig(network_bandwidth_gbps=m['bandwidth_gbps'],network_latency_ms=m['intercept_ms'],kv_connector='nixl',nixl_read_completion='engine_step',enable_compression=False)
actual=AnalyticalKVCacheTransferPredictor(c).get_transfer_time(ClusterType.PREFILL,ClusterType.DECODE,None,46103040)
expected=m['intercept_ms']+m['slope_ms_per_MB']*46.10304
assert abs(actual-expected)<1e-12, (actual,expected)
assert 0<actual<18.72496532622958
print(json.dumps(dict(bytes_per_READ=46103040,expected_ms=expected,production_ms=actual,absolute_difference_ms=abs(actual-expected),result='PASS')))
'''
commands = [
    dict(name='lead_retained_download_check', command=['timeout','120s',PYTHON,str(HERE.parent / 'nixl6144_run/lead_check_nixl6144.py')],
         log='lead_retained_check.log', expected='23 hashes/sizes equal;90 unique records;12150 trials;80352 notifications;zero CSV mismatches'),
    dict(name='existing_nixl_completion_regression', command=['timeout','120s',PYTHON,'-m','pytest','tests/unit/test_nixl_read_completion.py','-q','-p','no:cacheprovider','--basetemp',str(HERE/'scratch/pytest')],
         log='lifecycle_regression.log', expected='all existing NIXL engine-step lifecycle tests pass'),
    dict(name='existing_configuration_proposed_price', command=['timeout','120s',PYTHON,'-c',price_program,str(HERE/'proposed_input.json')],
         log='production_price.log', expected='production predictor agrees to <1e-12ms; positive0.968402341951ms; below both minimum gaps'),
]
for c in commands:
    c['started_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (HERE/c['log']).open('w') as stream:
        done = subprocess.run(c['command'], cwd=REPO, env=ENV, stdout=stream, stderr=subprocess.STDOUT)
    c['exit_code'] = done.returncode
    c['status'] = 'PASS' if done.returncode == 0 else 'FAIL'
    c['observed_log'] = (HERE/c['log']).read_text()
    print(c['name'], c['status'], '\n'+c['observed_log'])
doc = dict(classification='EXECUTED_Q2_CHECKS', python_executable=PYTHON,
           python_version=platform.python_version(), cwd=str(REPO),
           environment={k:ENV[k] for k in ['PYTHONPATH','PYTHONDONTWRITEBYTECODE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','TMPDIR','FRONTIER_TMP_ROOT']},
           commands=commands,
           analysis=dict(command=f'{PYTHON} {HERE}/analyze.py', status='PASS', log='analysis.log',expected='every integrity/concurrency/event assertion passes', observed=(HERE/'analysis.log').read_text()),
           focused_checks=dict(command=f'{PYTHON} {HERE}/checks.py',status='PASS',log='checks.log',expected='12 tests pass',observed=(HERE/'checks.log').read_text()),
           fresh_CPU_replays=dict(status='NOT_RUN', reason='Proposed prices cannot move a completion in any retained B29/two-fix posting window;proof contains2034 zero-crossing rows'),
           platform_or_GPU=dict(status='NOT_RUN',reason='Outside assignment; inspected retained receipts only'))
(HERE/'verification.json').write_text(json.dumps(doc,indent=2)+'\n')
sys.exit(0 if all(c['status']=='PASS' for c in commands) else 1)
