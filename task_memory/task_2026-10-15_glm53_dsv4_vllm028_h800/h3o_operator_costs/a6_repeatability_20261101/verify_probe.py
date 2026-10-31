"""Verify the fixed A6 probe against the retained native steps and frozen source."""
import dataclasses
import datetime
import hashlib
import io
import json
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import time

resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3))
repo = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028')
sys.path[:0] = [str(repo), str(repo / 'tests/analysis')]
from v4_attention_coverage import plan_from_spec, prediction_steps, read_steps, check_steps
root = repo / 'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs'
here = root / 'a6_repeatability_20261101'
build = Path('/data/ycfeng/Frontier/.worktrees/g1h-supplement-build')
package = build / here.relative_to(repo) / 'package_01'
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
clock = time.monotonic()
def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(path.read_bytes()).hexdigest()
def save(name, value):
    (here / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
manifest = json.loads((package / 'manifest.json').read_text())
assert len(manifest['files']) == 546
for name, digest in manifest['files'].items():
    assert sha(package / name) == digest, name
revision = manifest['source_revision']
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=build, text=True).strip() == revision
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=build, text=True).strip()
archive = subprocess.check_output(['git', 'archive', revision, 'frontier', 'data/config'], cwd=build)
source_count = 0
with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
    for member in tree.getmembers():
        if member.isfile():
            assert (package / member.name).read_bytes() == tree.extractfile(member).read(), member.name
            source_count += 1
for path in ('tests/comparison/calibration/dsv4_graph_bridge_check.py',
             'tests/comparison/calibration/profile_dsv4_prefill.py',
             'frontier/profiling/moe/balanced_routing.py'):
    assert (package / Path(path).name).read_bytes() == subprocess.check_output(['git', 'show', revision + ':' + path], cwd=build)
plan = plan_from_spec(here / 'package_plan.json')
mapping = json.loads((here / 'case_mapping.json').read_text())
original_plans = [plan_from_spec(root / name / 'package_plan.json') for name in ('g1h', 'g1h_supplement')]
originals = {case.case_id: case for item in original_plans for case in item.cases}
original_steps = {s['case_id']: s for item in original_plans for s in prediction_steps(item)}
assert len(mapping) == len(plan.cases) == 128
assert len({m['original'] for m in mapping.values()}) == 8
predictions = list(prediction_steps(plan))
assert len(predictions) == 128
for case in plan.cases:
    definition = dataclasses.asdict(case)
    definition['case_id'] = mapping[case.case_id]['original']
    assert definition == dataclasses.asdict(originals[definition['case_id']]), case.case_id
    assert case.warmup == 0 and case.measured == 1
for step in predictions:
    original = original_steps[mapping[step['case_id']]['original']]
    for field in ('phase', 'requests', 'cudagraph_dispatch'):
        assert step[field] == original[field], (step['case_id'], field)
predicted_check = check_steps(plan, [('prediction:probe', n, s) for n, s in enumerate(predictions, 1)], predicted=True)
assert predicted_check['prediction_matches']
raw_paths = [Path('/data/ycfeng/tmp/glm53_dsv4/h200/g1h_02/download/G1H_ATTENTION/bridge/FULL_AND_PIECEWISE/raw') / f'dsv4_raw_e{e}.jsonl' for e in (0, 1)]
raw_paths.append(Path('/data/ycfeng/tmp/glm53_dsv4/h200/g1h_s01/download/G1H_SUPPLEMENT/bridge/FULL_AND_PIECEWISE/raw/dsv4_raw_e0.jsonl'))
selected = {item['original'] for item in mapping.values()}
raw = {}
for file, line, step in read_steps(raw_paths):
    if step['case_id'] in selected:
        assert step['case_id'] not in raw, step['case_id']
        raw[step['case_id']] = (file, line, step)
assert raw.keys() == selected
cloned = []
for case_id, item in mapping.items():
    file, line, step = raw[item['original']]
    cloned.append((file, line, {**step, 'case_id': case_id}))
observed_check = check_steps(plan, cloned)
assert observed_check['prediction_matches'], observed_check['mismatches'][:3]
save('geometry_check.json', {'predicted': predicted_check, 'retained_raw_step_check': observed_check,
                            'original_steps': raw,
                            'raw_inputs': {str(p): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in raw_paths}})
save('predicted_steps.json', predictions)
job = json.loads((package / 'jobs/A6_REPEATABILITY.json').read_text())
assert (job['gpu'], job['gpu_type'], job['charged_group'], job['preemptible'], job['timeout_seconds']) == (1, 'H200', 'step_main', False, 5400)
assert job['output_dir'].startswith('/mnt/codesign-exp/ycfeng/')
steps = sum(case.estimated_steps for case in plan.cases)
assert steps == 1568
seconds = 2 * 486.104 + steps * 0.205 + len(plan.cases) * 0.2
save('verification.json', {'status': 'PASS', 'started_utc': started,
    'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'elapsed_seconds': time.monotonic() - clock, 'python': sys.version, 'source_revision': revision,
    'verified_manifest_files': len(manifest['files']), 'verified_snapshot_files': source_count,
    'package': str(package), 'manifest_sha256': sha(package / 'manifest.json'),
    'original_cases': 8, 'cloned_cases': len(plan.cases), 'measured_steps': len(predictions),
    'estimated_engine_steps': steps, 'estimated_seconds': seconds, 'estimated_gpu_hours': seconds / 3600,
    'timeout_exposure_gpu_hours': job['timeout_seconds'] / 3600,
    'recorded_previous_gpu_hours_approximate': 4.52, 'group_cap_gpu_hours': 7,
    'ceiling_with_payload_timeout_approximate': 4.52 + job['timeout_seconds'] / 3600,
    'remaining_after_payload_timeout_approximate': 7 - 4.52 - job['timeout_seconds'] / 3600,
    'limits': 'Geometry and source check only; prompt contents and persistent GPU cache/clock state are not fixed. No native probe has run.'})
print((here / 'verification.json').read_text())
