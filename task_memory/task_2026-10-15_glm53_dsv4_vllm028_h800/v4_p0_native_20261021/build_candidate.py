"""Stage an unsealed V4 review candidate using existing native launch mechanisms."""

import hashlib
import json
from pathlib import Path
import subprocess

import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PRIMARY = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028')
TD = PRIMARY / HERE.parent.relative_to(REPO)
V4 = TD / 'h200_dsv4_g0b_20261021'
TEMPLATE = TD / 'h200_glm_canonical_20261021/package'
PACKAGE = HERE / 'candidate_h800_p0_06'
NATIVE = Path('/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69')
CHECKER = REPO / 'task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py'
FRONTIER = PRIMARY / 'task_memory/task_2026-10-22_v4_balanced_routing/balanced_frontier_input/frontier.json'
WORKLOAD = 'data/processed_traces/canonical_fixed_6144_1024_req128_qps1'
MODULES = ['tests/comparison/calibration/' + name + '.py' for name in
    ('native_pd_worker', 'pd_replay', 'pd_proxy', 'pd_metrics', 'client_metrics',
     'e2e_metrics_gap', 'native_full_structure', 'native_dummy_initialization')]
MODULES += ['tests/comparison/dp_placement_pp/vllm_replay.py',
            'tests/comparison/stage_admission_pp/vllm_burst_driver.py',
            'tests/integration/native_extension_startup_probe.py']
MODULES += ['frontier/profiling/moe/balanced_routing.py']


def sha(content):
    return hashlib.sha256(content).hexdigest()


def git(repo, *args):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(repo), *args]).strip()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def main(generation='06'):
    global PACKAGE
    PACKAGE = HERE / f'candidate_h800_p0_{generation}'
    assert generation == '06', 'Only the root-selected fresh native06 generation is authorized'
    commit = git(REPO, 'rev-parse', 'HEAD').decode()
    assert not git(REPO, 'status', '--short'), 'Commit verified source before staging'
    native_commit = git(NATIVE, 'rev-parse', 'HEAD').decode()
    assert native_commit == '2cf0a6915ce544dc493a0990f2ea38d81601128a'
    assert not git(NATIVE, 'status', '--short'), 'The original native snapshot must remain clean'
    PACKAGE.mkdir(exist_ok=False)
    assert not (PACKAGE / 'manifest.json').exists(), 'Never refresh a sealed candidate'
    lock = dict(harness_commit=commit, native_commit=native_commit,
        native_checkout=str(NATIVE), committed_files={}, retained_files={},
        adopted_import_prerequisite='9c54f663da0b20ce7fe2dfe60319d7153b469dce',
        qualification='UNVERIFIED; source/CPU evidence only; primary review required')

    def stage(name, content):
        path = PACKAGE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    config = json.loads((TEMPLATE / 'config.json').read_text())
    prior = json.loads((V4 / 'config.json').read_text())
    config.update(workspace=str(PACKAGE), script=str(PACKAGE / 'native_pd_worker.py'),
        harness_dir=str(PACKAGE / 'tests/comparison/calibration'), harness_worktree=str(REPO),
        model_dir=str(PACKAGE / 'model'), case_dir=str(PACKAGE / 'case'),
        run_tag=f'dsv4-h800-p0-canonical-20261021-{generation}', job_tag=f'dsv4-h800-p0-canonical-20261021-{generation}',
        output_dir='/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/'
                   'dsv4_h800_p0_canonical_20261021/case/runs/groundtruth_clean/'
                   f'dsv4-h800-p0-canonical-20261021-{generation}',
        gpu_type='H800', charged_group='codesign', replica=4, timeout_seconds=3600,
        peer_wait_seconds=1200, startup_timeout_s=1800, replay_timeout_s=1800,
        model_hashes=prior['model_hashes'], harness_files={}, retained_files={}, tool_hashes={},
        observer_bootstrap=str(PACKAGE / 'native_dummy_bootstrap.py'),
        overlay_manifest=str(PACKAGE / 'overlay.json'))
    engine = config['engine']
    for name in ('attention_backend', 'quantization', 'num_gpu_blocks_override'):
        engine.pop(name, None)
    engine.update(data_parallel_size=2, tensor_parallel_size=8, pipeline_parallel_size=1,
        seed=prior['llm']['seed'], block_size=None, kv_cache_dtype='fp8',
        worker_extension_cls='native_full_structure.FullModelInventory', moe_gating_routing_runtime_path='balanced_topk')
    # Native compilation/capture defaults are kept unresolved, as in accepted G0B.
    engine.pop('compilation_config', None)
    accepted_model_sha = prior['expected_source_hashes'][
        'vllm/models/deepseek_v4/nvidia/model.py']
    config['runtime_source_overlay'] = dict(base_source_sha=native_commit,
        approved_c26_model_sha256=accepted_model_sha,
        effective_runtime_source='base2cf plus the declared accepted C26 model leaf')
    config['expected_source_hashes'].update(prior['expected_source_hashes'])
    for name in ('vllm/distributed/parallel_state.py', 'vllm/forward_context.py',
                 'vllm/v1/cudagraph_dispatcher.py', 'vllm/compilation/breakable_cudagraph.py',
                 'vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py',
                 'vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py',
                 'vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py'):
        config['expected_source_hashes'][name] = sha((NATIVE / name).read_bytes())
    for name in ('vllm/v1/worker/gpu/dp_utils.py', 'vllm/v1/worker/gpu/states.py',
                 'vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py'):
        config['expected_source_hashes'][name] = sha((NATIVE / name).read_bytes())
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('native_checker', CHECKER)
    checker = module_from_spec(spec)
    spec.loader.exec_module(checker)
    assert len(config['expected_source_hashes']) == 70, 'Dispatch source binding must have70 pins'
    for name in checker.ROUTING_PINS:
        config['expected_source_hashes'][name] = sha((NATIVE / name).read_bytes())
    assert len(config['expected_source_hashes']) == 74, 'Routing source binding must have74 pins'
    for name in MODULES:
        target = 'native_pd_worker.py' if name.endswith('/native_pd_worker.py') else name
        if name == checker.BALANCED_MODULE:
            target = 'tests/comparison/calibration/balanced_routing.py'
        content = subprocess.check_output(['git', '-C', str(REPO), 'show', f'{commit}:{name}'])
        stage(target, content)
        config['harness_files'][name] = target
        lock['committed_files'][target] = dict(path=name, commit=commit, sha256=sha(content))
        config['tool_hashes'][str(PACKAGE / target)] = sha(content)
    content = FRONTIER.read_bytes()
    stage('case/inputs/frontier.json', content)
    config['retained_files']['case/inputs/frontier.json'] = sha(content)
    config['tool_hashes'][str(PACKAGE / 'case/inputs/frontier.json')] = sha(content)
    checker.verify_balanced_frontier(config | {'harness_commit': commit}, engine)
    lock['retained_files']['case/inputs/frontier.json'] = dict(source=str(FRONTIER), sha256=sha(content))
    for name in ('probe_runtime_inventory.py', 'probe_nixl_read.py',
                 'native_dummy_bootstrap.py', 'overlay_model.py', 'submit_native_diagnostic.py'):
        content = original = (V4 / name).read_bytes()
        if name == 'submit_native_diagnostic.py':
            text = content.decode().replace('"VLLM_LOGGING_LEVEL": "DEBUG"', '"VLLM_LOGGING_LEVEL": "INFO"')
            marker = '    print("RJOB_NAME", workers.rjob.meta.name, flush=True)'
            assert text.count(marker) == 1
            text = text.replace(marker, marker + '''
    try:
        from native_pd_worker import write_launcher_receipt
        write_launcher_receipt(config, workers.rjob.meta.name, path)
    except Exception as error:
        print("LAUNCHER_OBSERVATION_ERROR", repr(error), flush=True)
''')
            content = text.encode()
        stage(name, content)
        config['retained_files'][name] = sha(content)
        config['tool_hashes'][str(PACKAGE / name)] = sha(content)
        lock['retained_files'][name] = dict(source=str(V4 / name), source_sha256=sha(original),
            sha256=sha(content), changed=content != original)
    overlay = json.loads((V4 / 'overlay.json').read_text())
    overlay['staged_dir'] = str(PACKAGE)
    leaf = overlay['files']['models/deepseek_v4/nvidia/model.py']
    leaf['source'] = str(PACKAGE / 'overlay_model.py')
    assert leaf['sha256'] == accepted_model_sha
    assert sha((NATIVE / 'vllm/models/deepseek_v4/nvidia/model.py').read_bytes()) == leaf['base_sha256']
    write_json(PACKAGE / 'overlay.json', overlay)
    config['retained_files']['overlay.json'] = sha((PACKAGE / 'overlay.json').read_bytes())
    lock['runtime_overlay'] = overlay
    original_model = Path(prior['llm']['model'])
    for name, expected in config['model_hashes'].items():
        content = (original_model / name).read_bytes()
        assert sha(content) == expected
        stage('model/' + name, content)
    hashes = {}
    for name in ('trace.csv', 'request_ids.json', 'warmup_pool.csv', 'manifest.json', 'SHA256SUMS'):
        content = subprocess.check_output(['git', '-C', str(REPO), 'show', f'68f891b:{WORKLOAD}/{name}'])
        stage('trace/' + name, content)
        stage('case/inputs/trace/' + name, content)
        hashes[name] = sha(content)
    for name, expected in config['expected_source_hashes'].items():
        if name == 'vllm/models/deepseek_v4/nvidia/model.py':
            assert sha((PACKAGE / 'overlay_model.py').read_bytes()) == expected
        else:
            assert sha((NATIVE / name).read_bytes()) == expected, name
    write_json(PACKAGE / 'config.json', config)
    write_json(PACKAGE / 'engine.json', engine)
    write_json(PACKAGE / 'case/inputs/engine.json', engine)
    write_json(PACKAGE / 'case/inputs/native_config.json', config)
    write_json(PACKAGE / 'serve_observers.json', None)
    case = dict(case_id='dsv4-h800-p0-canonical-fixed128-qps1',
        run_generation=f'v4-native-v028-h800-p0-20261021-{generation}',
        purpose='P0 original V4 H80032; all four metric errors at most10percent',
        model='DeepSeek-V4-Flash', architecture='pd-disaggregation',
        groundtruth_checkout_path=str(NATIVE), groundtruth_commit=native_commit,
        groundtruth_branch=git(NATIVE, 'branch', '--show-current').decode(),
        workload_sha256=hashes, image=config['image'],
        groundtruth_checker_path=str(CHECKER), groundtruth_checker_sha256=sha(CHECKER.read_bytes()),
        primary_checker_schema_review_required=True,
        runtime_source_overlay=dict(manifest='overlay.json', sha256=sha((PACKAGE / 'overlay.json').read_bytes()),
            accepted_c26_model_sha256=leaf['sha256'], native_checkout_is_base_only=True),
        topology={'vllm': {role: dict(tp=8, dp=2, ep=16, pp=1) for role in ('prefill', 'decode')},
                  'frontier': {'sys_arch': 'pd-disaggregation'}},
        policy=dict(priority='P0', gpu_type='H800', charged_group='codesign', preemptible=False,
            metric_relative_error_limit=0.10, all_four_metrics_required=True,
            root_cap_check_required=True, root_package_acceptance_required=True,
            auxiliary_validators_soft=True, H200_nonpreemptible_running_plus_queued_limit=32,
            formal64_requires_same_model_same_hardware32_all_four_errors_at_most=0.30))
    stage('case/manifest.yaml', yaml.safe_dump(case, sort_keys=False).encode())
    launch = (TEMPLATE / 'launch.py').read_text()
    launch = launch.replace("('H200', 'step_main', 8, 8)", "('H800', 'codesign', 8, 4)")
    launch = launch.replace("config['engine']['data_parallel_size'] == 4", "config['engine']['data_parallel_size'] == 2")
    launch = launch.replace("assert not any(name in config for name in ('observer_bootstrap', 'overlay_manifest'))",
        "assert Path(config['observer_bootstrap']).name == 'native_dummy_bootstrap.py'\n"
        "assert Path(config['overlay_manifest']).name == 'overlay.json'")
    before = "assert os.environ['STEPMIND_BACKEND'] == 'rjob'"
    after = before + "\nfrom native_pd_worker import write_run_manifest\n" \
        "run_manifest = write_run_manifest(config, os.environ['AUTHORIZATION'])\n" \
        "print('PRELAUNCH_MANIFEST', str(run_manifest), flush=True)"
    assert launch.count(before) == 1
    stage('launch.py', launch.replace(before, after).encode())
    seal = (TEMPLATE / 'seal.py').read_text()
    seal = seal.replace("('model', 'tests', 'trace')", "('model', 'tests', 'trace', 'case/inputs')")
    seal = seal.replace('files = sorted([path', "files = sorted([HERE / 'case/manifest.yaml'] + [path")
    stage('seal.py', seal.encode())
    write_json(PACKAGE / 'source_lock.json', lock)
    write_json(PACKAGE / 'evidence_readiness.json', dict(
        status='UNSEALED_REVIEW_CANDIDATE', submitted=False, runtime_qualified=False,
        target='H800/codesign32 nonpreemptible; P16+D16 TP8/DP2/EP16/PP1',
        runtime='native2cf base + accepted C26 one-file bootstrap; binaries unchanged',
        source_instrumentation='Existing NativeServingInventory; V1 and supported V2 actual calls',
        native_limits=['H800 grouped memory and actual allocation', 'native HMA P/D transfer',
            'all32rank forwards/replays with conditional NONE semantics', 'ordered shutdown',
            'complete 3x(16+128) client records', 'four calibration errors<=10percent'],
        primary_review=['Unsealed candidate', 'Existing checker V4/V2/schema/C26 binding'],
        cost=dict(payload_guard_seconds=3600, max_payload_gpu_hours=32,
            scheduled_arrival_span_seconds=426, minimum_schedule_gpu_hours=32*426/3600,
            h800_measured_startup_or_timing=False),
        reused_h200_evidence='Source/bootstrap/native dtype and geometry functions only; no H800 memory/timing claim'))
    print(json.dumps(dict(package=str(PACKAGE), source_commit=commit,
        native_source_pins=len(config['expected_source_hashes']), unsealed=True,
        actual_submission_or_native_admission=False)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', required=True, type=int)
    args = parser.parse_args()
    if args.generation < 1:
        parser.error('generation must be positive')
    main(f'{args.generation:02d}')
