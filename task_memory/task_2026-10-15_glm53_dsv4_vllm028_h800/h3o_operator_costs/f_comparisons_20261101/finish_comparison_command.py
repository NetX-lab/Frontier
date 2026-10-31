"""Run the existing terminal checks once after the original watcher finishes."""
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3))
repo = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028')
task = repo / 'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800'
root = task / 'h3o_operator_costs/f_comparisons_20261101'
stage = Path('/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4')
run = stage / 'f_variants/runs/f2_p2_64_k18_codex_20261101'
python = '/data/ycfeng/envs/frontier-py310/bin/python'
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR='/data/ycfeng/tmp')
def read(path):
    return json.loads(path.read_text())
def write(name, value):
    with (root / name).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')

end = time.monotonic() + 1800
while True:
    watch = read(root / 'watcher.json')['cases']['f2_p2_64']
    if watch['status'] == 'READING_COMPLETE':
        break
    if watch['status'] in ('RUN_FAILED', 'GATE_FAILED'):
        raise RuntimeError(watch)
    if time.monotonic() > end:
        raise TimeoutError('Original replay/watcher has not completed; no check was run')
    time.sleep(15)
assert not (root / 'complete').exists()
assert not (root / 'f2_p2_64_a8.json').exists()
receipt = read(run / 'receipt.json')
assert receipt['status'] == 'FINISHED'
assert receipt['source_commit'] == '662a427ed32bc1bc2f29b177e7c073cbd460c0b4'
assert receipt['source_dirty_paths'] == []
assert receipt['a11']['stale_stage_end_events'] == 0
with gzip.open(run / 'forwards.jsonl.gz', 'rt') as stream:
    forward_count = sum(1 for line in stream if line.strip())
assert forward_count == receipt['a11']['forwards']['rows'] == receipt['a11']['stage_end_events']
for key in ('forwards', 'collectives'):
    record = receipt['a11'][key]
    assert hashlib.sha256(Path(record['path']).read_bytes()).hexdigest() == record['sha256']
records = read(run / 'collectives.json')
rows = records['rows']
calls = sum(row['call_count'] for row in rows)
exact_calls = sum(row['call_count'] for row in rows if row['exact'])
assert calls == receipt['a11']['collectives']['calls']
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
command = [python, str(task / 'h3o_operator_costs/a10/a8_collectives.py'), str(run)]
result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=60)
with (root / 'f2_p2_64_a8_stdout.txt').open('x') as stream:
    stream.write(result.stdout + result.stderr)
assert result.returncode == 0, result.stderr
assert result.stdout.splitlines()[-1] == 'violations 0', result.stdout
write('f2_p2_64_a8.json', dict(observed_utc=started, command=command, exit_code=result.returncode,
      distinct_requests=len(rows), calls=calls, exact_calls=exact_calls, violations=0,
      actual_forward_rows=forward_count, source_evidence=str(root / 'f2_p2_64_collectives.json')))
write('f2_p2_64_receipt.json', receipt)
write('f2_p2_64_collectives.json', records)
command = [python, 'tests/analysis/v4_f_variant_comparison.py', '--task-root', str(task.relative_to(repo)),
           '--stage-root', str(stage), '--output', str((root / 'complete').relative_to(repo))]
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=60)
write('complete_command.json', dict(command=command, cwd=str(repo), start_utc=started,
      end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), exit_code=result.returncode,
      environment={'PYTHONDONTWRITEBYTECODE': '1', 'TMPDIR': '/data/ycfeng/tmp'},
      python_version=subprocess.check_output([python, '--version'], text=True).strip(),
      collector_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
      stdout=result.stdout, stderr=result.stderr, timeout_s=60, rlimit_as_gib=30))
assert result.returncode == 0, result.stderr
assert result.stdout.strip() == 'PASS_EVIDENCE_CHECKS readings=8 metrics=128 unavailable=3'
comparison = read(root / 'complete/comparison.json')
case = next(item for item in comparison['results'] if item['variant'] == 'F2' and item['case'] == 'p2_64')
write('f2_p2_64_verification.json', dict(status=f"PASS_EVIDENCE_NUMERICAL_{case['verdict']}", result=case))
print(result.stdout.strip(), flush=True)
print(json.dumps({'F2_P2_64': case['verdict'], 'forwards': forward_count, 'calls': calls,
                  'exact_calls': exact_calls, 'pooled': case['metrics'][:4]}, indent=2), flush=True)
