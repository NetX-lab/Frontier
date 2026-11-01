"""Verify the completed supplement passes against the frozen package and selected artifacts."""
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = Path('/data/ycfeng/tmp/glm53_dsv4/a6_supplement_passes_20261101_download')
PHASE = 'A6_SUPPLEMENT_PASSES'
REMOTE = '/mnt/ws-jfs/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_attention/a6_supplement_passes_20261101_01'
status = json.loads((ROOT / 'status.json').read_text())
phase = status['phases'][PHASE]
platform = json.loads((HERE / 'platform_final.json').read_text())
launch = json.loads((HERE / 'launch_command.json').read_text())
analysis = json.loads((HERE / 'repeatability_result.json').read_text())
mapping = json.loads((HERE / 'case_mapping.json').read_text())
predicted = json.loads((HERE / 'predicted_steps.json').read_text())
bridge = json.loads((ROOT / PHASE / 'bridge/bridge_report.json').read_text())
run = json.loads((ROOT / PHASE / 'bridge/bridge_run.json').read_text())
download = json.loads((HERE / 'download.json').read_text())
assert phase['state'] == 'succeeded'
assert status['source_revision'] == 'a689ed05f462eb269b9dffe640da151e6614a5ef'
assert status['runtime']['device_name'] == 'NVIDIA H200'
assert status['runtime']['vllm'] == '0.28.0'
assert platform['creator'] == 'i-fengyicheng'
assert platform['labels']['quotagroup.brainpp.cn/preemptible'] == 'no'
assert platform['status']['phase'] == 'Succeeded'
assert platform['stop_annotations'] == {}
assert len(platform['replicas']) == 1
replica = platform['replicas'][0]
assert replica['phase'] == 'Succeeded'
container = replica['container_statuses'][0]
terminal = container['state']['terminated']
assert terminal['exitCode'] == 0 and container['restart_count'] == 0
assert launch['final_status'] == 'succeeded'
assert bridge['verdict'] == 'PASS' and bridge['reasons'] == []
assert set(run['runs']) == set(bridge['runs']) == {'FULL_AND_PIECEWISE/e0', 'FULL_AND_PIECEWISE/e1'}
assert all(r['exit_code'] == 0 and r['error'] is None for r in run['runs'].values())
traces = [t for r in bridge['runs'].values() for t in r['traces']]
assert len(traces) == len(mapping) == 864
assert {Path(t['path']).stem for t in traces} == set(mapping)
assert all(t['error'] is None and t['hold_interval_count'] == 1 for t in traces)
expected_modes = {f'FULL_AND_PIECEWISE/e{e}': dict(Counter(
    step['cudagraph_dispatch']['cg_mode'] for step in predicted if step['engine_index'] == e)) for e in (0, 1)}
for name, counts in expected_modes.items():
    assert bridge['runs'][name]['cg_mode_counts'] == counts, (name, bridge['runs'][name]['cg_mode_counts'], counts)
assert analysis['status'] == 'PASS_IDENTITY_AND_GEOMETRY'
assert analysis['measured_rows'] == 4320
assert analysis['geometry']['measured_steps'] == analysis['geometry']['cases_seen'] == 864
assert analysis['identity_mismatches'] == analysis['count_mismatches'] == []
assert analysis['reconstructed_reference_medians'] == 29376
for item in download['files']:
    assert hashlib.sha256(Path(item['local_path']).read_bytes()).hexdigest() == item['sha256'] == item['remote_sha256']
remote_code = f'''import json, pathlib, getpass
assert getpass.getuser() == 'i-fengyicheng'
root = pathlib.Path({REMOTE!r})
phase = json.loads((root/'status.json').read_text())['phases'][{PHASE!r}]
files = {{name: (root/{PHASE!r}/name).stat().st_size for name in phase['copied_files']}}
print(json.dumps({{'files':files,'state':phase['state']}}))
'''
command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'shai-ycfeng', 'timeout 60s python3 -']
response = subprocess.run(command, input=remote_code, text=True, capture_output=True, check=True, timeout=65)
archive = json.loads(response.stdout)
assert archive['state'] == 'succeeded'
assert set(archive['files']) == set(phase['copied_files'])
assert all(size > 0 for size in archive['files'].values())
trace_files = {name: size for name, size in archive['files'].items() if '/traces/' in name}
assert {Path(name).stem for name in trace_files} == set(mapping)
assert sum(trace_files.values()) == phase['trace_bytes_archived']
started = datetime.datetime.fromisoformat(terminal['startedAt'].replace('Z', '+00:00'))
finished = datetime.datetime.fromisoformat(terminal['finishedAt'].replace('Z', '+00:00'))
seconds = (finished - started).total_seconds()
result = {'status': 'PASS', 'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'job': platform['name'], 'source_revision': status['source_revision'], 'node': replica['node'],
          'platform_phase': platform['status']['phase'], 'container_exit_code': terminal['exitCode'],
          'container_seconds': seconds, 'container_gpu_hours': seconds / 3600,
          'launcher_seconds': launch['elapsed_seconds'], 'launcher_gpu_hours_window': launch['elapsed_seconds'] / 3600,
          'phase_seconds': phase.get('seconds'), 'cg_mode_counts': expected_modes,
          'measured_steps': 864, 'layer_group_rows': 4320, 'reference_medians_reproduced': 29376,
          'archive_files': archive['files'], 'trace_files': len(trace_files),
          'trace_bytes': sum(trace_files.values()), 'selected_download_files': len(download['files']),
          'remote_command': command, 'remote_source': remote_code,
          'limits': 'Trace names/sizes verified remotely; complete traces remain in cloud and are not rehashed here. '
                    'Cost windows are observations, not a platform bill. Qualification remains open.'}
with (HERE / 'business_verification.json').open('x') as out:
    json.dump(result, out, indent=2)
print(json.dumps({k: v for k, v in result.items() if k not in {'archive_files', 'remote_command', 'remote_source'}}))
