"""Build one exclusive D24 package through the existing canonical G3 builder."""

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
REFERENCE = Path('/data/ycfeng/tmp/step5_canonical_g3_20261021/capture01')
WORKSPACE = Path('/data/ycfeng/tmp/step5_d24_native_20261022/capture02')
NATIVE = Path('/data/ycfeng/tmp/step5_profiling_20261020/image_src')
OUTPUT = Path('/mnt/codesign-exp/ycfeng/frontier/step5_native_pd/d24_capture02')


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
    assert source['commit'] == 'ebd5baf61757d48ff4ca6287903a01e6828c25f4'
    original = json.loads((REFERENCE / 'stage/config.json').read_text())
    immutable = json.loads((REFERENCE / 'stage_source_manifest.json').read_text())
    for name, expected in immutable.items():
        assert digest(REFERENCE / 'stage' / name) == expected, name
    native_pins = original['expected_source_hashes']
    assert len(native_pins) == 50
    for name, expected in native_pins.items():
        assert digest(NATIVE / name) == expected, name

    definition = r2_canonical.definition(NATIVE)
    assert definition == json.loads((REFERENCE / 'stage/r2_run.json').read_text())
    assert not WORKSPACE.exists(), 'A distinct unused candidate root is required'
    origins = {}
    for name, module in sorted(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.is_relative_to(REPO) and not path.is_relative_to(TASK):
            relative = str(path.relative_to(REPO))
            value = git_bytes(relative)
            assert path.read_bytes() == value, (name, path)
            origins[name] = dict(path=str(path), repository=str(REPO), relative=relative,
                                 commit=source['commit'], sha256=digest(path), git_blob_matches=True)
    expected_modules = ('tests.integration.step5_pd.build_stage',
                        'tests.integration.step5_pd.r2_canonical',
                        'tests.integration.step5_pd.canonical_stage',
                        'tests.integration.step5_pd.canonical_replay',
                        'native_pd_worker', 'pd_replay', 'pd_proxy',
                        'serving_metrics', 'vllm_replay', 'vllm_burst_driver')
    for name in expected_modules:
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(REPO), (name, module.__file__)

    WORKSPACE.mkdir(parents=True, exist_ok=False)
    shutil.copytree(REFERENCE / 'reference_inputs', WORKSPACE / 'reference_inputs')
    listing = WORKSPACE / 'installed_source_pins.txt'
    with listing.open('x') as stream:
        for name, expected in sorted(native_pins.items()):
            stream.write(f'{expected}  /usr/local/lib/python3.12/dist-packages/{name}\n')
    write_new(WORKSPACE / 'run_definition.json', definition)
    stage = WORKSPACE / 'stage'
    config = build_stage.build(definition, stage=stage, workspace=WORKSPACE,
        output_dir=OUTPUT, model_dir=WORKSPACE / 'reference_inputs',
        image_source_hashes=listing, probe_dir=REFERENCE / 'stage',
        job_tag='d24h80032g302', source=source)
    assert config['expected_source_hashes'] == native_pins
    assert config['engine'] == original['engine']
    from tests.integration.step5_pd import submit_native_pd, r2_stage
    assert submit_native_pd.resource_arguments(config) == submit_native_pd.resource_arguments(original)

    # Use the same package marker that the existing clean-stage builder writes.
    with (stage / 'tests/__init__.py').open('x') as stream:
        stream.write('"""Staged calibration tools."""\n')
    for name in ('submit_native_pd.py',):
        shutil.copy2(REPO / 'tests/integration/step5_pd' / name, stage / name)
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

    for name, module in sorted(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if (filename and Path(filename).resolve().is_relative_to(REPO)
                and not Path(filename).resolve().is_relative_to(TASK)):
            path = Path(filename).resolve()
            relative = str(path.relative_to(REPO))
            assert path.read_bytes() == git_bytes(relative)
            origins[name] = dict(path=str(path), repository=str(REPO), relative=relative,
                                 commit=source['commit'], sha256=digest(path), git_blob_matches=True)
    assert 'tests.integration.step5_pd.r2_stage' in origins
    copied = {}
    source_paths = {name: name for name in build_stage.HARNESS_FILES}
    source_paths.update({Path(build_stage.WORKER).name: build_stage.WORKER})
    source_paths.update({target: origin for origin, target in r2_stage.CAPTURE_FILES.items()})
    for name in ('r2_reference.py', 'r2_controls.py', 'r2_runtime.py', 'submit_native_pd.py'):
        source_paths[name] = 'tests/integration/step5_pd/' + name
    for name in ('r2_replay.py', 'canonical_replay.py'):
        source_paths['tests/integration/step5_pd/' + name] = 'tests/integration/step5_pd/' + name
    source_paths['compare_attention_state.py'] = 'tests/integration/step5_environment/compare_attention_state.py'
    for target, origin in sorted(source_paths.items()):
        assert (stage / target).read_bytes() == git_bytes(origin), target
        copied[target] = dict(repository=str(REPO), relative=origin,
                              commit=source['commit'], sha256=digest(stage / target))
    write_new(stage / 'source_origins.json', dict(source=source, imported_modules=origins,
        copied_git_files=copied, external_frozen_inputs=dict(
            reference_stage=str(REFERENCE / 'stage'), reference_manifest_sha256=digest(REFERENCE / 'stage_source_manifest.json'),
            launcher_scope_sha256=digest(stage / 'launcher_scope.py'),
            probe_pins=definition['probe_helpers'], native_source_root=str(NATIVE), native_pins=native_pins),
        package_marker='Existing canonical_stage package marker prevents installed regular tests shadowing'))
    config['tool_hashes'] = {str(path.relative_to(stage)): digest(path)
                            for path in sorted(stage.rglob('*')) if path.is_file() and path.name != 'config.json'}
    (stage / 'config.json').write_text(json.dumps(config, indent=1))
    write_new(TASK / 'build_receipt.json', dict(task_id='D24-NATIVE-STAGE', status='PASS_BUILD',
        utc_start=start, utc_end=datetime.now(timezone.utc).isoformat(), interpreter=sys.executable,
        python=sys.version, cwd=str(Path.cwd()), source=source, stage=str(stage),
        config_sha256=digest(stage / 'config.json'), imported_modules_bound=len(origins),
        copied_git_files=len(copied), installed_native_pins=len(native_pins),
        original_manifest_unchanged=True, definition_engine_resources_unchanged=True,
        submissions=0, network_api_auth_cloud_operations=0))
    print(json.dumps(dict(stage=str(stage), config_sha256=digest(stage / 'config.json'),
                          source=source, native_pins=len(native_pins)), indent=2))


if __name__ == '__main__':
    main()
