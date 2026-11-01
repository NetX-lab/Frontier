"""Copy only completed supplement-pass raw records and selected reports from personal storage."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

evidence = Path(__file__).resolve().parent
remote_root = '/mnt/ws-jfs/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_attention/a6_supplement_passes_20261101_01'
local = Path('/data/ycfeng/tmp/glm53_dsv4/a6_supplement_passes_20261101_download')
phase_name = 'A6_SUPPLEMENT_PASSES'
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'shai-ycfeng']
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
clock = time.monotonic()
status_command = ssh + ['test "$(id -un)" = i-fengyicheng && cat ' + shlex.quote(remote_root + '/status.json')]
status_result = subprocess.run(status_command, check=True, capture_output=True, timeout=60)
status = json.loads(status_result.stdout)
phase = status['phases'][phase_name]
assert phase['state'] == 'succeeded', phase['state']
files = ['status.json', f'{phase_name}/bridge/bridge_run.json', f'{phase_name}/bridge/bridge_report.json',
         f'{phase_name}/logs/bridge_run.log', f'{phase_name}/logs/bridge_report.log']
for engine in (0, 1):
    prefix = f'{phase_name}/bridge/FULL_AND_PIECEWISE/'
    files.extend([prefix + f'raw/dsv4_raw_e{engine}.jsonl',
                  prefix + f'bootstrap_config_e{engine}.json', prefix + f'run_e{engine}.log'])
assert not local.exists(), local
local.mkdir()
receipts = []
for name in files:
    destination = local / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = remote_root + '/' + name
    command = ssh + ['test "$(id -un)" = i-fengyicheng && cat ' + shlex.quote(source)]
    with destination.open('xb') as output:
        subprocess.run(command, stdout=output, check=True, timeout=300)
    digest_command = ssh + ['test "$(id -un)" = i-fengyicheng && sha256sum ' + shlex.quote(source)]
    remote_digest = subprocess.check_output(digest_command, text=True, timeout=120).split()[0]
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    assert digest == remote_digest, name
    receipts.append({'relative_path': name, 'local_path': str(destination), 'remote_path': source,
                     'bytes': destination.stat().st_size, 'sha256': digest, 'remote_sha256': remote_digest,
                     'copy_command': command, 'hash_command': digest_command})
result = {'status': 'PASS_SELECTED_DOWNLOAD', 'started_utc': started,
          'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'elapsed_seconds': time.monotonic() - clock, 'files': receipts,
          'downloaded_bytes': sum(r['bytes'] for r in receipts),
          'cloud_status': status, 'local_root': str(local),
          'limits': 'Selected files only; completed traces remain in personal cloud storage.'}
(evidence / 'download.json').write_text(json.dumps(result, indent=2) + '\n')
(evidence / 'worker_status_final.json').write_bytes(status_result.stdout)
print('PASS', len(receipts), 'files', result['downloaded_bytes'], 'bytes')
