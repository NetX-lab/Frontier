"""Build exclusive D28 capture03 with the approved existing decode-only control."""

from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parent
REFERENCE = Path('/data/ycfeng/tmp/step5_d24_native_20261022/capture02')
WORKSPACE = Path('/data/ycfeng/tmp/step5_d24_native_20261022/capture03')
NATIVE = Path('/data/ycfeng/tmp/step5_profiling_20261020/image_src')
OUTPUT = Path('/mnt/codesign-exp/ycfeng/frontier/step5_native_pd/d24_capture03')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def git_bytes(relative):
    return subprocess.check_output(['git', 'show', f'HEAD:{relative}'], cwd=REPO)


def main():
    start = datetime.now(timezone.utc).isoformat()
    sys.path.insert(0, str(REPO))
    from tests.integration.step5_pd import build_stage, r2_canonical

    source = build_stage.committed_source()
    assert source['repository'] == str(REPO)
    original = json.loads((REFERENCE / 'stage/config.json').read_text())
    manifest = json.loads((REFERENCE / 'stage_manifest.json').read_text())
    for name, entry in manifest['entries'].items():
        assert digest(REFERENCE / name) == entry['sha256'], name
    assert 'role_num_gpu_blocks_override' not in original
    native_pins = original['expected_source_hashes']
    assert len(native_pins) == 50
    for name, expected in native_pins.items():
        assert digest(NATIVE / name) == expected, name
    definition = r2_canonical.definition(NATIVE)
    assert definition == json.loads((REFERENCE / 'stage/r2_run.json').read_text())
    definition['role_num_gpu_blocks_override'] = {'decode': 32000}
    definition['description'] += ' D28: only D uses32000 native GPU blocks; P pool remains automatic.'
    r2_canonical.validate_definition(definition)
    assert not WORKSPACE.exists(), 'A distinct unused candidate root is required'
    for name in ('tests.integration.step5_pd.build_stage', 'tests.integration.step5_pd.r2_canonical',
                 'tests.integration.step5_pd.canonical_stage', 'tests.integration.step5_pd.canonical_replay',
                 'native_pd_worker', 'pd_replay', 'pd_proxy', 'serving_metrics', 'vllm_replay', 'vllm_burst_driver'):
        assert Path(importlib.import_module(name).__file__).resolve().is_relative_to(REPO), name

    WORKSPACE.mkdir(parents=True, exist_ok=False)
    shutil.copytree(REFERENCE / 'reference_inputs', WORKSPACE / 'reference_inputs')
    listing = WORKSPACE / 'installed_source_pins.txt'
    shutil.copy2(REFERENCE / 'installed_source_pins.txt', listing)
    write_new(WORKSPACE / 'run_definition.json', definition)
    stage = WORKSPACE / 'stage'
    config = build_stage.build(definition, stage=stage, workspace=WORKSPACE, output_dir=OUTPUT,
        model_dir=WORKSPACE / 'reference_inputs', image_source_hashes=listing,
        probe_dir=REFERENCE / 'stage', job_tag='d28h80032g303', source=source)
    assert config['expected_source_hashes'] == native_pins
    assert config['engine'] == original['engine']
    assert config['role_num_gpu_blocks_override'] == {'decode': 32000}
    from tests.integration.step5_pd import submit_native_pd, r2_stage
    assert submit_native_pd.resource_arguments(config) == submit_native_pd.resource_arguments(original)
    with (stage / 'tests/__init__.py').open('x') as stream:
        stream.write('"""Staged calibration tools."""\n')
    shutil.copy2(REPO / 'tests/integration/step5_pd/submit_native_pd.py', stage / 'submit_native_pd.py')
    shutil.copy2(REFERENCE / 'stage/launcher_scope.py', stage / 'launcher_scope.py')
    with (stage / 'submit.sh').open('x') as stream:
        stream.write('#!/usr/bin/env bash\nset -euo pipefail\n'
            f'stage={stage}\n'
            f'python3 "$stage/verify_r2_stage.py" --stage-dir "$stage" '
            f'--output {WORKSPACE}/presubmit_verification.json\n'
            'source /data/ycfeng/tmp/step5_native_pd_r1/stepmind_env.sh\n'
            'export STEPMIND_BACKEND=rjob\n'
            'exec "$STEPMIND_PYTHON" "$stage/submit_native_pd.py" '
            '--config "$stage/config.json" --launcher-scope "$stage/launcher_scope.py"\n')
    write_new(stage / 'routing_status.json', dict(status='BLOCKED_NATIVE_BALANCED_UNSUPPORTED',
        required='Balanced token distribution on both Frontier and native vLLM',
        declared_native_routing='ORIGINAL_BIASED_SIGMOID_UNCHANGED',
        effective_native_routing='UNVERIFIED_NOT_RUN',
        simulation_strategy_applied=False,
        allowed_existing_strategies=['uniform_random', 'normal_routing'],
        not_submission_ready_for_balanced_calibration=True,
        evidence_receipt_sha256=digest(TASK / 'native_routing_receipt.json'),
        next_action='Parent must resolve exact balanced runtime mechanism before launching aligned run'))

    origins = {}
    for name, module in sorted(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.is_relative_to(REPO) and not path.is_relative_to(TASK):
            relative = str(path.relative_to(REPO))
            assert path.read_bytes() == git_bytes(relative), name
            origins[name] = dict(path=str(path), repository=str(REPO), relative=relative,
                commit=source['commit'], sha256=digest(path), git_blob_matches=True)
    assert 'tests.integration.step5_pd.r2_stage' in origins
    source_paths = {name: name for name in build_stage.HARNESS_FILES}
    source_paths.update({Path(build_stage.WORKER).name: build_stage.WORKER})
    source_paths.update({target: origin for origin, target in r2_stage.CAPTURE_FILES.items()})
    for name in ('r2_reference.py', 'r2_controls.py', 'r2_runtime.py', 'submit_native_pd.py'):
        source_paths[name] = 'tests/integration/step5_pd/' + name
    for name in ('r2_replay.py', 'canonical_replay.py'):
        source_paths['tests/integration/step5_pd/' + name] = 'tests/integration/step5_pd/' + name
    source_paths['compare_attention_state.py'] = 'tests/integration/step5_environment/compare_attention_state.py'
    copied = {}
    for target, origin in sorted(source_paths.items()):
        assert (stage / target).read_bytes() == git_bytes(origin), target
        copied[target] = dict(repository=str(REPO), relative=origin,
            commit=source['commit'], sha256=digest(stage / target))
    write_new(stage / 'source_origins.json', dict(source=source, imported_modules=origins,
        copied_git_files=copied, external_frozen_inputs=dict(reference_stage=str(REFERENCE / 'stage'),
            reference_manifest_sha256=digest(REFERENCE / 'stage_manifest.json'),
            launcher_scope_sha256=digest(stage / 'launcher_scope.py'), probe_pins=definition['probe_helpers'],
            native_source_root=str(NATIVE), native_pins=native_pins),
        package_marker='Existing canonical_stage package marker prevents installed regular tests shadowing'))
    config['tool_hashes'] = {str(path.relative_to(stage)): digest(path)
        for path in sorted(stage.rglob('*')) if path.is_file() and path.name != 'config.json'}
    (stage / 'config.json').write_text(json.dumps(config, indent=1))
    write_new(TASK / 'build_receipt.json', dict(task_id='D28-D24-PRESSURE-STAGE', status='PASS_BUILD',
        utc_start=start, utc_end=datetime.now(timezone.utc).isoformat(), interpreter=sys.executable,
        python=sys.version, cwd=str(Path.cwd()), source=source, stage=str(stage),
        config_sha256=digest(stage / 'config.json'), imported_modules_bound=len(origins),
        copied_git_files=len(copied), installed_native_pins=len(native_pins),
        capture02_manifest_unchanged=True, base_engine_resources_unchanged=True,
        role_num_gpu_blocks_override={'decode': 32000}, prefill_pool='AUTOMATIC_UNCHANGED',
        native_pressure='UNVERIFIED_WITHOUT_OBSERVED_NATIVE_ALLOCATION_PRESSURE', submissions=0))
    print(json.dumps(dict(stage=str(stage), source=source, config_sha256=digest(stage / 'config.json')), indent=2))


if __name__ == '__main__':
    main()
