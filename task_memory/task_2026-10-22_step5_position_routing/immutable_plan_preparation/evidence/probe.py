"""Read-only real-wave preparation diagnosis; no trace emission or numeric memo."""
import ast
from collections import Counter
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/data/ycfeng/Frontier/.worktrees/step5-d29-artifact-bytes')
REV = '56529fcd19e7e209e577c5aac3c581b4c5df8b6a'
LABEL = sys.argv[1]
assert LABEL == 'red'
OUT = ROOT / LABEL
OUT.mkdir(exist_ok=False)
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip() == REV
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE).strip()
fixture = Path('/data/ycfeng/tmp/d29-parent-position-fixture03')

def state(root):
    return {str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns,
            hashlib.sha256(p.read_bytes()).hexdigest()] for p in root.rglob('*') if p.is_file()}

before = state(fixture)
original = Path('/data/ycfeng/tmp/d29_artifact_bytes_repair/verified03/candidate_setup.py')
setup = original.read_text().replace('2cfbb34474c1d4d31f6daf8f9c626d03ad972f58', REV)
tree = ast.parse(setup)
prefix = []
for node in tree.body:
    if node.lineno >= 126:
        break
    if 34 <= node.lineno <= 39:
        continue
    prefix.append(node)
wave = next(node for node in tree.body if node.lineno == 137)
(OUT / 'setup.py').write_text(ast.unparse(ast.Module(body=prefix, type_ignores=[])) + '\n')
cwd = OUT / 'cwd'
model_rel = Path('data/config/models/stepfun-ai__Step-5-Preview.json')
(cwd / model_rel).parent.mkdir(parents=True)
(cwd / model_rel).write_bytes((SOURCE / model_rel).read_bytes())
os.chdir(cwd)  # Before config parsing; all generated paths are scratch-only.
sys.path.insert(0, str(SOURCE))
real_read_bytes = Path.read_bytes
collective_paths = list((fixture / 'fixture_collectives').rglob('collective_local.json'))
assert len(collective_paths) == 1
fixed_path = collective_paths[0].resolve()
fixed_bytes = real_read_bytes(fixed_path)
read_counts = Counter()

def fixed_file_byte_adapter(path, *args, **kwargs):
    if path.resolve() == fixed_path:
        read_counts['fixed_byte_requests'] += 1
        return fixed_bytes
    return real_read_bytes(path, *args, **kwargs)

Path.read_bytes = fixed_file_byte_adapter
namespace = {'__name__': '__main__', '__file__': str(OUT / 'setup.py')}
sys.argv = [str(OUT / 'setup.py'), '--output', str(OUT / 'setup-output')]
exec(compile(ast.Module(body=prefix, type_ignores=[]), str(OUT / 'setup.py'), 'exec'), namespace)
from frontier.cc_backend.collective_plan import CollectivePlan
real_key = CollectivePlan.cache_key
key_counts = Counter()

def counted_key(plan):
    value = real_key(plan)
    key_counts[value] += 1
    return value

def payload(phases):
    rows = []
    for record in phases.lane_records:
        layer = record.execution_time.layer_execution_times[0]
        moe = layer.moe_or_mlp_time_component.operator_times
        comm = layer.communication_time_component.operator_times
        rows.append({'ep': record.ep_id, 'moe': moe.op_times, 'comm': comm.op_times,
                     'plans': {key: real_key(plan) for key, plan in comm.plans.items()},
                     'proofs': moe.physical_frame['operator_costs']})
    assert len(rows) == 16
    assert sum(len(row['moe']) + len(row['comm']) for row in rows) == 208
    return json.dumps(rows, sort_keys=True).encode()

wave_code = compile(ast.Module(body=[wave], type_ignores=[]), str(OUT / 'setup.py'), 'exec')
timings = []
payloads = []
for index in range(4):
    key_counts.clear()
    read_counts.clear()
    CollectivePlan.cache_key = counted_key
    prof = cProfile.Profile() if index == 3 else None
    if prof:
        prof.enable()
    start = time.perf_counter()
    exec(wave_code, namespace)
    seconds = time.perf_counter() - start
    if prof:
        prof.disable()
        prof.dump_stats(str(OUT / 'wave.prof'))
    CollectivePlan.cache_key = real_key
    timings.append({'index': index, 'pricing_s': seconds, 'profiled': bool(prof),
                    'key_calls': sum(key_counts.values()), 'distinct_keys': len(key_counts),
                    'key_repetitions': sum(key_counts.values()) - len(key_counts), **read_counts})
    payloads.append(payload(namespace['phases']))
assert len(set(payloads)) == 1
assert hashlib.sha256(payloads[0]).hexdigest() == '3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83'
(OUT / 'pricing_payload.json').write_bytes(payloads[0])
Path.read_bytes = real_read_bytes
assert before == state(fixture)
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE).strip()
stats = pstats.Stats(str(OUT / 'wave.prof'))
selected = [{'file': file, 'line': line, 'function': name, 'primitive': cc,
             'calls': nc, 'self_s': tt, 'cumulative_s': ct}
            for (file, line, name), (cc, nc, tt, ct, callers) in stats.stats.items()
            if 'collective_plan.py' in file or 'moe_routing_workload.py' in file or
               (name in ('dumps', 'loads', 'asdict', 'predict') and
                ('json/' in file or 'dataclasses.py' in file or 'sklearn/' in file))]
selected.sort(key=lambda row: row['cumulative_s'], reverse=True)
result = {'task_id': 'D29-IMMUTABLE-PLAN-PREPARATION', 'source': str(SOURCE), 'revision': REV,
          'adapter': 'ONE SINGLE-ROLE DIAGNOSTIC FIXED-FILE-BYTE ADAPTER, not run-wide P/D snapshot',
          'fixed_file': str(fixed_path), 'fixed_sha256': hashlib.sha256(fixed_bytes).hexdigest(),
          'timings': timings, 'warm_unprofiled_median_s': statistics.median(t['pricing_s'] for t in timings[1:3]),
          'payload_sha256': hashlib.sha256(payloads[0]).hexdigest(), 'fixture_unchanged': True,
          'profile': selected, 'verdict': 'RED: repeated identical plan-key preparation' if timings[-1]['key_repetitions'] else 'GREEN: no repeated keys'}
(OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'profile'}, indent=2))
assert not timings[-1]['key_repetitions'], result['verdict']
