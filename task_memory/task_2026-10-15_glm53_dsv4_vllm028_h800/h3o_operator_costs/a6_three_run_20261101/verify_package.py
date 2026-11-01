"""Verify the supplement-pass package against the frozen source and the retained G1h-s01 steps."""
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
here = root / 'a6_three_run_20261101'
build = Path('/data/ycfeng/Frontier/.worktrees/g1h-supplement-build')
package = build / here.relative_to(repo) / 'package_01'
REMAINING_GPU_HOURS = 2.160175
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
clock = time.monotonic()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (here / name).open('x') as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


for name in ('package_plan.json', 'pilot_cases.json', 'case_mapping.json'):
    assert (here / name).read_bytes() == (package.parent / name).read_bytes(), name
manifest = json.loads((package / 'manifest.json').read_text())
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
supplement = plan_from_spec(root / 'g1h_supplement/package_plan.json')
originals = {case.case_id: case for case in supplement.cases}
original_steps = {step['case_id']: step for step in prediction_steps(supplement)}
assert len(mapping) == len(plan.cases) == 864
assert len({item['original'] for item in mapping.values()}) == 432
assert {item['repeat'] for item in mapping.values()} == {1, 2}
predictions = list(prediction_steps(plan))
assert len(predictions) == 864
for case in plan.cases:
    definition = dataclasses.asdict(case)
    definition['case_id'] = mapping[case.case_id]['original']
    assert definition == dataclasses.asdict(originals[definition['case_id']]), case.case_id
    assert case.warmup == 0 and case.measured == 1
for step in predictions:
    original = original_steps[mapping[step['case_id']]['original']]
    for field in ('phase', 'requests', 'cudagraph_dispatch'):
        assert step[field] == original[field], (step['case_id'], field)
predicted_check = check_steps(plan, [('prediction:passes', n, s) for n, s in enumerate(predictions, 1)], predicted=True)
assert predicted_check['prediction_matches']
raw_root = Path('/data/ycfeng/tmp/glm53_dsv4/h200/g1h_s01/download/G1H_SUPPLEMENT/bridge/FULL_AND_PIECEWISE/raw')
raw_paths = [raw_root / f'dsv4_raw_e{engine}.jsonl' for engine in (0, 1)]
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
                             'raw_inputs': {str(p): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in raw_paths}})
save('predicted_steps.json', predictions)
job = json.loads((package / 'jobs/A6_SUPPLEMENT_PASSES.json').read_text())
assert (job['gpu'], job['gpu_type'], job['charged_group'], job['preemptible'], job['timeout_seconds']) == (1, 'H200', 'step_main', False, 5400)
assert job['output_dir'].startswith('/mnt/codesign-exp/ycfeng/')
steps = sum(case.estimated_steps for case in plan.cases)
engines = {case.engine_index for case in plan.cases}
seconds = len(engines) * 486.104 + steps * 0.205 + len(plan.cases) * 0.2
timeout_gpu_hours = job['timeout_seconds'] / 3600
assert timeout_gpu_hours <= REMAINING_GPU_HOURS
save('verification.json', {'status': 'PASS', 'started_utc': started,
    'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'elapsed_seconds': time.monotonic() - clock, 'python': sys.version, 'source_revision': revision,
    'verified_manifest_files': len(manifest['files']), 'verified_snapshot_files': source_count,
    'package': str(package), 'manifest_sha256': sha(package / 'manifest.json'),
    'original_cases': 432, 'cloned_cases': len(plan.cases), 'measured_steps': len(predictions),
    'estimated_engine_steps': steps, 'estimated_seconds': seconds, 'estimated_gpu_hours': seconds / 3600,
    'timeout_exposure_gpu_hours': timeout_gpu_hours,
    'remaining_gpu_hours_before_run_approximate': REMAINING_GPU_HOURS,
    'remaining_after_timeout_exposure_approximate': REMAINING_GPU_HOURS - timeout_gpu_hours,
    'limits': 'Geometry and source check only; prompt contents and persistent GPU cache/clock state are not fixed. No native run yet.'})
print((here / 'verification.json').read_text())
