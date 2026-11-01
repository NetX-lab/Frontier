"""Build and CPU-verify a targeted package with the retained B30b planner/runner.

Does not build a submission lock, launch a worker, or access cloud/platform APIs.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
REPO = TASK.parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    base = TASK / 'h3o_operator_costs/b30b/stage_b30b_links_kv_02.json'
    config = copy.deepcopy(json.loads(base.read_text()))
    config.update(stage='p2_nixl6144_qualification', job_tag='dsv4-h200-nixl6144-q1-20261102',
        timeout_seconds=1800, preemptible=True, output_dir='/mnt/codesign-exp/ycfeng/glm53_dsv4/comm/h200/p2_nixl6144_q1_20261102')
    # No repeated failed collective phases; this experiment addresses exact KV bytes
    # and repeated/concurrent completion at the formal prompt and its boundaries.
    phase = next(p for p in config['phases'] if p['name'] == 'C_nixl')
    phase['timeout_seconds'] = 1200
    phase['config']['shuffle_seed'] = 20261102
    phase['config']['grid'] = [dict(prompt_tokens=[6143, 6144, 6145], active_pairs=[1, 8],
                                    requests_in_flight=[1, 2, 4, 8, 16])]
    config['phases'] = [phase]
    path = output / 'stage_nixl6144.json'
    config['lock_sources'] = list(dict.fromkeys(config['lock_sources'] + [
        str(path.relative_to(REPO)), str(Path(__file__).resolve().relative_to(REPO))]))
    path.write_text(json.dumps(config, indent=2) + '\n')
    leaf = dict(phase['config'], output_dir=config['output_dir'] + '/C_nixl')
    leaf_path = output / 'nixl_leaf.json'
    leaf_path.write_text(json.dumps(leaf, indent=2) + '\n')
    planner = module('q1_b30b_planner', TASK / 'h3o_operator_costs/b30b/planner.py')
    report = planner.dry_run(config, planner.read_timings(planner.DEFAULT_G5H),
                             planner.read_timings([planner.DEFAULT_K5]), repo=REPO)
    cases = report['phases'][0]['cases']
    assert report['allocated_gpus'] == 16 and report['total_cases'] == 30
    assert report['wall_seconds'] < phase['timeout_seconds'] < config['timeout_seconds']
    assert len({(c['prompt_tokens'], c['active_pairs'], c['requests_in_flight']) for c in cases}) == 30
    assert {c['blocks_per_request'] for c in cases if c['prompt_tokens'] in (6143, 6144)} == {46}
    assert {c['blocks_per_request'] for c in cases if c['prompt_tokens'] == 6145} == {51}
    assert all(c['bytes_per_reader'] == c['blocks_per_request'] * c['requests_in_flight'] * 1002240 for c in cases)
    command = [sys.executable, '-m', 'frontier.profiling.collectives.nixl_kv_read_benchmark',
               '--config', str(leaf_path), '--list-cases']
    listed = subprocess.run(command, cwd=REPO, check=True, capture_output=True, text=True, timeout=60)
    assert listed.stdout.splitlines()[-1] == 'cases 30 passes 3 measurements 90'
    (output / 'list_cases.txt').write_text(listed.stdout)
    runtime = module('q1_runtime_check', TASK / 'h3o_operator_costs/g5h/launch/runtime_check.py')
    worker = runtime.worker_command(config, path, output / 'submission_lock.json')
    # This is review-only. Committed-source lock creation remains with Claude.
    (output / 'worker_command.txt').write_text(worker + '\n')
    sources = [base, Path(__file__).resolve(), path, leaf_path,
        TASK / config['node_entry'].split(str(TASK.relative_to(REPO)) + '/')[-1]]
    sources += [REPO / p for p in config['lock_sources']]
    sources += [REPO / config['runtime_check'], Path(config['p2_overlay'])]
    sources = list(dict.fromkeys(sources))
    for source in sources:
        if source.suffix == '.py':
            compile(source.read_text(), str(source), 'exec')
    layout = json.loads((REPO / config['kv_layout_evidence']).read_text())
    source_hashes = {str(source): sha256(source) for source in sources}
    native = Path(layout['native_source'])
    assert sha256(native) == layout['native_source_sha256']
    source_hashes[str(native)] = sha256(native)
    notifications = sum(c['active_pairs'] * c['requests_in_flight'] * (leaf['warmup'] + leaf['repeats'])
                        for c in cases) * leaf['passes']
    reader_trials = sum(c['active_pairs'] * leaf['repeats'] for c in cases) * leaf['passes']
    report.update(status='PASS_CPU_PACKAGE_CHECKS', config=str(path), config_sha256=sha256(path),
        base_config=str(base), source_sha256=source_hashes,
        cpu_command=command, expected_raw_records=90, expected_reader_trials=reader_trials,
        expected_notifications=notifications, measured_gpu_checks='NOT_RUN',
        submission_lock='NOT_BUILT: Claude must review, commit sources and call existing build_lock before submission',
        resource_policy=dict(gpu_type='H200', nodes=2, gpus_per_node=8, planned_preemptible=True,
            non_preemptible='Only if total personal reserved/queued non-preemptible H200 including this job is <=32; otherwise wholly preemptible',
            platform_check='NOT_RUN; belongs to lead /ds4-exec'))
    (output / 'cpu_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:report[k] for k in ['status','total_cases','expected_raw_records',
        'expected_reader_trials','expected_notifications','wall_seconds','gpu_hours','timeout_gpu_hours']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'packages/nixl6144')
    build(parser.parse_args().output.resolve())
