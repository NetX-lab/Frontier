"""Hash-bound R5 startup source binding; synthetic packages live only in scratch.

The fixtures copy only sealed generation 01/02 inputs and reconstruct their
prelaunch manifests in scratch. Native bytes come from the pinned Git objects;
FRONTIER_NATIVE_SOURCE_REPO selects their repository when it is elsewhere.
Every rejection changes one field or file of that coherent package.
"""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
DISPATCH = REPO / 'task_memory/task_2026-10-22_v4_native_dispatch'
R3 = DISPATCH / 'check_groundtruth_run_successor_r3.py'
R4 = DISPATCH / 'check_groundtruth_run_successor_r4.py'
R5 = DISPATCH / 'check_groundtruth_run_successor_r5.py'
MAINLINE = REPO / 'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline'
PACKAGES = {
    'startup': MAINLINE / 'p2_64_formal_recovery/package_02',
    'base': MAINLINE / 'p2_64_formal/package_02',
}
SOURCE_REPO_ENV = 'FRONTIER_NATIVE_SOURCE_REPO'
STARTUP_SOURCE_SHA = 'a06a72a4326cba02234df6cfbdc21b78d3a15d26'
BASE_SOURCE_SHA = '2cf0a6915ce544dc493a0990f2ea38d81601128a'
C26_MODEL = 'vllm/models/deepseek_v4/nvidia/model.py'
C26_MODEL_BASE_SHA256 = 'c01be6d313f494eb38982c8f56c314267bbf67310e7f13ca53b596b31aef4e44'
LEAF = 'models/deepseek_v4/nvidia/model.py'
STARTUP_SOURCES = {
    'vllm/config/parallel.py': (
        'native_config_parallel.py',
        'f29f984ac39f4ae11622c2c64f92a657951f3a28e624a2a60d1040b49e025311',
        'f489eb26a3784ce5691faebad99a33b6e2e7c5988605bd7427415efc685f7440'),
    'vllm/distributed/parallel_state.py': (
        'native_distributed_parallel_state.py',
        'c6d65b96a260ea0d779da1073d1799392813eee74c9ed44753dfa56f4e7a9e4c',
        'a9b20efd63ece3bd4e4436f774a7be7156bfc410b310944c4ef4d7e19a091572'),
    'vllm/distributed/utils.py': (
        'native_distributed_utils.py',
        '792146b357139e705064a68d3b66858de8a0baaca4dc7db1232fa217d45db231',
        'a06292492a5cc24a17be299a5890d617fc8d1fe82e9a95b1cfc6be47d0d250ab'),
    'vllm/v1/engine/utils.py': (
        'native_engine_utils.py',
        '6ce83b0552b6207505c9bd7ac1fd67d2771628eac01572738765923aa91c1c0f',
        '533d60761f2505e93b318c349baf733a1ec7e386f1c2ad3f768e41bdab017599'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def load(path, name=None):
    spec = importlib.util.spec_from_file_location(name or f'native_{Path(path).stem}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_repo(checkout):
    """Locate the repository holding the sealed native commits, never a worktree copy."""
    override = os.environ.get(SOURCE_REPO_ENV)
    if override:
        return Path(override)
    try:
        common = subprocess.run(['git', '-C', str(checkout), 'rev-parse', '--path-format=absolute',
                                 '--git-common-dir'], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pytest.fail(f'Committed native sources unavailable: set {SOURCE_REPO_ENV} to a vLLM repository '
                    f'containing the sealed commits (no usable repository at {checkout})')
    return Path(common).parent


def committed_blobs(repo, commit, paths):
    """Read committed native source bytes from the object store."""
    request = ''.join(f'{commit}:{path}\n' for path in paths).encode()
    process = subprocess.run(['git', '-C', str(repo), 'cat-file', '--batch'], input=request,
                             capture_output=True)
    if process.returncode:
        pytest.fail(f'Committed native bytes unavailable in {repo}: {commit} '
                    f'({process.stderr.decode().strip()}); set {SOURCE_REPO_ENV}')
    stream = io.BytesIO(process.stdout)
    blobs = {}
    for path in paths:
        header = stream.readline().split()
        if len(header) != 3 or header[1] != b'blob':
            pytest.fail(f'Committed native blob missing: {commit}:{path}; set {SOURCE_REPO_ENV}')
        blobs[path] = stream.read(int(header[2]))
        stream.read(1)
    return blobs


@pytest.fixture(scope='session')
def checker():
    return load(R5)


@pytest.fixture(scope='session')
def frozen_r4():
    return load(R4)


class Generation:
    """One sealed generation rebuilt in scratch from sealed inputs and committed native bytes."""

    def __init__(self, root, generation):
        self.generation = generation
        self.source = PACKAGES[generation]
        self.stage = Path(root) / 'package'
        self.seal = json.loads((self.source / 'manifest.json').read_text())
        for name in [*self.seal['files'], 'manifest.json']:
            target = self.stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.source / name, target)
        self.case = self.stage / 'case'
        self.config = json.loads((self.stage / 'config.json').read_text())
        self.overlay = json.loads((self.stage / 'overlay.json').read_text())
        self.lock = json.loads((self.stage / 'source_lock.json').read_text())
        self.seal = json.loads((self.stage / 'manifest.json').read_text())
        self.engine = json.loads((self.stage / 'engine.json').read_text())
        self.case_manifest = yaml.safe_load((self.case / 'manifest.yaml').read_text())
        self.seal_path = self.stage / 'manifest.json'
        self.repo = source_repo(Path(self.case_manifest['groundtruth_checkout_path']))
        self.overlay['staged_dir'] = str(self.stage)
        for row in self.overlay['files'].values():
            row['source'] = str(self.stage / Path(row['source']).name)
        prefix = str(Path(self.config['script']).parent)
        for key in ('script', 'case_dir', 'overlay_manifest', 'harness_dir',
                    'observer_bootstrap', 'model_dir'):
            value = self.config[key]
            assert value.startswith(prefix), key
            self.config[key] = str(self.stage / Path(value).relative_to(prefix))
        self.config['harness_worktree'] = str(self.stage)
        self.config['tool_hashes'] = {
            str(self.stage / Path(name).relative_to(prefix)) if name.startswith(prefix) else name: None
            for name in self.config['tool_hashes']}
        self.checkout = self.build_checkout(Path(root) / 'checkout')
        self.manifest_path = (self.case / 'runs/groundtruth_clean' / self.config['run_tag']
                              / 'run_manifest.json')
        self.manifest = self.derive_manifest()
        self.commit()

    def build_checkout(self, target):
        """Materialize the sealed commit's native sources from committed blobs, not a worktree."""
        commit = self.case_manifest['groundtruth_commit']
        blobs = committed_blobs(self.repo, commit, sorted(self.config['expected_source_hashes']))
        for name, payload in blobs.items():
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        for name, expected in self.config['expected_source_hashes'].items():
            if name != C26_MODEL:
                assert digest(target / name) == expected, name
        assert digest(target / C26_MODEL) == C26_MODEL_BASE_SHA256
        return target

    def derive_inputs(self):
        retained = {name: digest(self.stage / name) for name in self.config['retained_files']}
        inputs = {str(self.stage / name): value for name, value in retained.items()}
        inputs.update({name.removeprefix('case/'): value for name, value in retained.items()
                       if name.startswith('case/')})
        inputs['inputs/native_config.json'] = digest(self.case / 'inputs/native_config.json')
        inputs.update({str(self.stage / staged): digest(self.stage / staged)
                       for staged in self.config['harness_files'].values()})
        return inputs

    def derive_manifest(self):
        """A prelaunch manifest derived from sealed inputs, not from a launcher run directory."""
        topology = self.case_manifest['topology']['vllm']['prefill']
        return dict(
            run_tag=self.config['run_tag'], mode='clean', case_id=self.case_manifest['case_id'],
            run_generation=self.case_manifest['run_generation'],
            execution_config='inputs/native_config.json',
            groundtruth_commit=self.case_manifest['groundtruth_commit'],
            groundtruth_checkout_path=str(self.checkout),
            harness_commit=self.lock['harness_commit'], harness_worktree=str(self.stage),
            resources=dict(gpus=2 * topology['tp'] * topology['dp'],
                           gpu_type=self.case_manifest['policy']['gpu_type'],
                           preemptible=self.case_manifest['policy']['preemptible']),
            inputs_sha256=self.derive_inputs(),
            harness_sha256={name: digest(self.stage / staged)
                            for name, staged in self.config['harness_files'].items()})

    def write_config(self):
        payload = json.dumps(self.config, indent=2) + '\n'
        (self.stage / 'config.json').write_text(payload)
        (self.case / 'inputs/native_config.json').write_text(payload)

    def commit(self):
        """Rebind every derived digest to the scratch bytes, from the sealed inputs only."""
        write(self.stage / 'overlay.json', self.overlay)
        for name in self.config['retained_files']:
            self.config['retained_files'][name] = digest(self.stage / name)
        for name in self.config['tool_hashes']:
            self.config['tool_hashes'][name] = digest(Path(name))
        self.write_config()
        self.lock['runtime_overlay'] = self.overlay
        for name in self.lock['retained_files']:
            self.lock['retained_files'][name]['sha256'] = digest(self.stage / name)
        write(self.stage / 'source_lock.json', self.lock)
        for name in self.seal['files']:
            self.seal['files'][name] = digest(self.stage / name)
        write(self.seal_path, self.seal)
        self.manifest['inputs_sha256'] = self.derive_inputs()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write(self.manifest_path, self.manifest)
        stamp = self.manifest_path.stat().st_mtime_ns - 1_000_000_000
        os.utime(self.seal_path, ns=(stamp, stamp))
        self.config = json.loads((self.stage / 'config.json').read_text())
        return self

    def prepare_rank_graph(self, root):
        """Minimal run geometry that reaches the binding inside the inherited rank evidence."""
        run_dir = Path(root) / 'run'
        run_dir.mkdir()
        (run_dir / 'run_manifest.json').write_text('{}\n')
        nodes = {f'serving{index:02d}': None for index in range(2 * self.engine['data_parallel_size'])}
        return run_dir, nodes, {'repetitions': {'repeat_indices': [0]}}

    def publish_overlay(self):
        """Persist an edited overlay with the lock copy the predicate compares."""
        write(self.stage / 'overlay.json', self.overlay)
        self.lock['runtime_overlay'] = self.overlay
        write(self.stage / 'source_lock.json', self.lock)

    def publish_lock(self):
        write(self.stage / 'source_lock.json', self.lock)


@pytest.fixture
def startup(tmp_path):
    fixture = Generation(tmp_path, 'startup')
    (tmp_path / 'generation').mkdir()
    return fixture


@pytest.fixture
def old_generation(tmp_path):
    return Generation(tmp_path, 'base')


def bind(checker, fixture, manifest=None, engine=None, config=None):
    return checker.native_v4_source_binding(manifest or fixture.manifest,
                                            engine or fixture.engine, config or fixture.config)


def test_r5_binds_actual_generation02_startup_package(startup, checker, frozen_r4):
    bind(checker, startup)
    with pytest.raises(ValueError, match='exact approved base2cf plus C26'):
        bind(frozen_r4, startup)
    assert digest(startup.stage / 'native_engine_utils.py') == STARTUP_SOURCES['vllm/v1/engine/utils.py'][2]
    assert set(startup.overlay['files']) == {LEAF, *[p.removeprefix('vllm/') for p in STARTUP_SOURCES]}


def test_r5_delegates_generation01_package_to_frozen_binding(old_generation, checker, frozen_r4):
    assert old_generation.manifest['groundtruth_commit'] == BASE_SOURCE_SHA
    bind(checker, old_generation)
    bind(frozen_r4, old_generation)
    assert set(old_generation.overlay['files']) == {LEAF}


@pytest.mark.parametrize('fault', ['manifest', 'engine', 'declaration'])
def test_r5_rejects_unapproved_native_commit(startup, checker, fault):
    if fault == 'manifest':
        startup.manifest['groundtruth_commit'] = '0' * 40
    elif fault == 'engine':
        startup.engine['native_reference']['source_sha'] = BASE_SOURCE_SHA
    else:
        startup.config['runtime_source_overlay']['base_source_sha'] = BASE_SOURCE_SHA
    with pytest.raises(ValueError, match='Unknown V4 native source identity'):
        bind(checker, startup)


@pytest.mark.parametrize('fault', ['extra', 'missing'])
def test_r5_rejects_overlay_membership(startup, checker, fault):
    if fault == 'extra':
        startup.overlay['files']['config/cache.py'] = dict(startup.overlay['files'][LEAF])
    else:
        del startup.overlay['files']['distributed/utils.py']
    startup.publish_overlay()
    with pytest.raises(ValueError, match='exactly the approved C26 model and four startup sources'):
        bind(checker, startup)


@pytest.mark.parametrize('fault', ['staged_bytes', 'overlay_new_hash', 'overlay_base_hash',
                                   'expected_hash', 'checkout_bytes'])
def test_r5_rejects_forged_startup_source(startup, checker, fault):
    source = 'vllm/config/parallel.py'
    staged_name = STARTUP_SOURCES[source][0]
    row = startup.overlay['files'][source.removeprefix('vllm/')]
    if fault == 'staged_bytes':
        staged = startup.stage / staged_name
        staged.write_bytes(staged.read_bytes() + b'\n# forged\n')
        startup.commit()
    elif fault == 'overlay_new_hash':
        row['sha256'] = '0' * 64
        startup.publish_overlay()
    elif fault == 'overlay_base_hash':
        row['base_sha256'] = STARTUP_SOURCES[source][2]
        startup.publish_overlay()
    elif fault == 'expected_hash':
        startup.config['expected_source_hashes'][source] = STARTUP_SOURCES[source][1]
    else:
        (startup.checkout / source).write_bytes(b'# unapproved checkout bytes\n')
    with pytest.raises(ValueError, match='startup source overlay differs from the approved commit'):
        bind(checker, startup)


@pytest.mark.parametrize('fault', ['seal', 'retained', 'prelaunch', 'committed_commit',
                                   'committed_path', 'committed_hash'])
def test_r5_rejects_missing_package_pins(startup, checker, fault):
    source = 'vllm/config/parallel.py'
    staged_name = STARTUP_SOURCES[source][0]
    if fault == 'seal':
        del startup.seal['files'][staged_name]
        startup.commit()
    elif fault == 'retained':
        del startup.config['retained_files'][staged_name]
    elif fault == 'prelaunch':
        del startup.manifest['inputs_sha256'][str(startup.stage / staged_name)]
    else:
        key = fault.removeprefix('committed_')
        startup.lock['committed_files'][staged_name][key] = (
            '0' * 40 if key == 'commit' else 'vllm/config/cache.py' if key == 'path' else '0' * 64)
        startup.publish_lock()
    with pytest.raises(ValueError, match='not sealed and prelaunch-bound|not bound to the launch inputs|'
                                         'startup source commit binding mismatch'):
        bind(checker, startup)


def test_r5_rejects_unapproved_native_commit_binding(startup, checker):
    startup.lock['committed_files']['native_config_cache.py'] = dict(
        commit=STARTUP_SOURCE_SHA, path='vllm/config/cache.py',
        sha256=digest(startup.stage / 'native_config_parallel.py'))
    startup.publish_lock()
    with pytest.raises(ValueError, match='source_lock binds unapproved native sources'):
        bind(checker, startup)


@pytest.mark.parametrize('fault', ['overlay_new_hash', 'staged_bytes', 'declaration_hash',
                                   'declaration_empty'])
def test_r5_rejects_forged_c26_identity(startup, checker, fault):
    row = startup.overlay['files'][LEAF]
    if fault == 'overlay_new_hash':
        row['sha256'] = '0' * 64
        startup.publish_overlay()
    elif fault == 'staged_bytes':
        path = startup.stage / 'overlay_model.py'
        path.write_bytes(path.read_bytes() + b'\n# forged\n')
        startup.commit()
    elif fault == 'declaration_hash':
        startup.config['runtime_source_overlay']['approved_c26_model_sha256'] = '0' * 64
    else:
        startup.config['runtime_source_overlay']['effective_runtime_source'] = ''
    with pytest.raises(ValueError, match='C26 model/base/observed-source mapping mismatch|'
                                         'must declare the approved startup-repair source identity'):
        bind(checker, startup)


@pytest.mark.parametrize('fault', ['overlay_outside_stage', 'lock_overlay', 'harness_commit',
                                   'config_mismatch', 'stale_seal', 'other_native_source'])
def test_r5_rejects_unapproved_package_identity(startup, checker, fault):
    if fault == 'overlay_outside_stage':
        startup.config['overlay_manifest'] = str(startup.stage.parent / 'overlay.json')
    elif fault == 'lock_overlay':
        startup.lock['runtime_overlay'] = dict(startup.overlay, staged_dir='elsewhere')
        startup.publish_lock()
    elif fault == 'harness_commit':
        startup.lock['harness_commit'] = '0' * 40
        startup.publish_lock()
    elif fault == 'config_mismatch':
        startup.config['job_tag'] = 'unapproved'
    elif fault == 'stale_seal':
        stamp = startup.manifest_path.stat().st_mtime_ns + 1_000_000_000
        os.utime(startup.seal_path, ns=(stamp, stamp))
    else:
        (startup.checkout / 'vllm/config/vllm.py').write_bytes(b'# unapproved checkout bytes\n')
    with pytest.raises(ValueError, match='outside the sealed execution package|'
                                         'runtime overlay differs|native and harness commits|'
                                         'sealed package/source lock differs|'
                                         'Unapproved V4 native source change'):
        bind(checker, startup)


def test_r5_keeps_frozen_chain_and_normal_entry_point(tmp_path, checker, frozen_r4):
    assert checker.R4_PATH == R4
    assert checker.R4_SHA256 == digest(R4)
    assert checker.R3_SHA256 == digest(R3)
    assert checker.frozen_native_v4_source_binding is not checker.native_v4_source_binding
    # The frozen dependency remains the routing-owner seam of R4, not a copy.
    assert checker.routing_successor_source() == frozen_r4.routing_successor_source()
    assert checker.successor_source() == frozen_r4.successor_source()
    tampered = tmp_path / 'check_groundtruth_run_successor_r4.py'
    tampered.write_bytes(R4.read_bytes() + b'\n# unapproved dependency bytes.\n')
    with pytest.raises(ValueError, match='Frozen R4 dependency hash mismatch'):
        checker.frozen_r4_source(tampered)
    with pytest.raises(ValueError, match='Frozen R4 dependency hash mismatch'):
        checker.load_frozen_r4(tampered)
    assert all(callable(getattr(checker, name)) for name in
               ('check', 'main', 'sha', 'native_idle_evidence', 'native_rank_graph_evidence'))
    entered = subprocess.run([sys.executable, str(R5), str(tmp_path / 'missing'), 'tag', 'clean'],
                             capture_output=True, text=True, timeout=300)
    assert entered.returncode != 0 and 'FileNotFoundError' in entered.stderr


def test_r5_call_chain_binds_through_rank_evidence(tmp_path, checker):
    chain = checker.native_rank_graph_evidence.__globals__
    assert checker.check.__globals__ is chain and checker.main.__globals__ is chain
    assert chain['native_rank_graph_evidence'] is checker.native_rank_graph_evidence
    assert chain['native_v4_source_binding'] is checker.native_v4_source_binding
    assert chain['frozen_native_v4_source_binding'] is not chain['native_v4_source_binding']

    fixture = Generation(tmp_path, 'startup')
    run_dir, nodes, workload = fixture.prepare_rank_graph(tmp_path)
    accepted = checker.native_rank_graph_evidence(
        nodes, fixture.manifest, fixture.engine, run_dir, workload, [], fixture.config)
    assert accepted == (False, 'UNVERIFIED: ValueError: Missing canonical client phase IDs'), accepted

    fixture.manifest['groundtruth_commit'] = 'f' * 40
    rejected = checker.native_rank_graph_evidence(
        nodes, fixture.manifest, fixture.engine, run_dir, workload, [], fixture.config)
    assert rejected[0] is False
    assert rejected[1].startswith("UNVERIFIED: ValueError: Unknown V4 native source identity: ('ffff"), rejected


def test_compare_selects_one_immutable_reference_reader(tmp_path, monkeypatch):
    directory = (DISPATCH.parent / 'task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs'
                 / 'formal_p2_64_native_20261102')
    local = load(directory / 'coverage.py', 'p2_64_reader_coverage')
    monkeypatch.setitem(sys.modules, 'coverage', local)
    compare = load(directory / 'compare.py', 'p2_64_reader_compare')
    assert compare.CHECKERS == {'r4': R4, 'r5': R5}
    assert compare.select_reference_checker() == R4
    assert compare.select_reference_checker('r5') == R5
    with pytest.raises(ValueError, match='Unknown reference checker'):
        compare.select_reference_checker('r3')

    download = tmp_path / 'download'
    artifacts = download / 'artifacts'
    artifacts.mkdir(parents=True)
    (artifacts / 'payload.txt').write_text('incomplete CPU fixture without a run manifest')
    inventory = dict(files=[dict(path='payload.txt', bytes=(artifacts / 'payload.txt').stat().st_size,
                                sha256=digest(artifacts / 'payload.txt'))])
    (download / 'cloud_inventory.json').write_text(json.dumps(inventory))
    (download / 'download_receipt.json').write_text(json.dumps(dict(
        status='HASH_IDENTICAL_DOWNLOAD', mismatched=[],
        inventory_sha256=digest(download / 'cloud_inventory.json'))))
    for choice, expected in (('r4', R4), ('r5', R5)):
        result = compare.run_pipeline(tmp_path / 'case', download, tmp_path / f'out_{choice}',
                                      reference_checker=choice)
        assert result['status'] == 'UNVERIFIED' and result['errors'] == ['read_run_manifest']
        assert result['reference_checker'] == dict(name=choice, path=str(expected),
                                                   sha256=digest(expected))
    assert compare.run_pipeline(tmp_path / 'case', download, tmp_path / 'out_default')
    assert json.loads((tmp_path / 'out_default/result.json').read_text())['reference_checker']['name'] == 'r4'

    # The selected reader drives the subprocess and the recorded digest.
    pipeline = tmp_path / 'pipeline'
    artifacts = pipeline / 'download/artifacts'
    artifacts.mkdir(parents=True)
    receipt = pipeline / 'promotion_receipt.json'
    receipt.write_text(json.dumps(dict(issued_utc='2026-01-01T00:00:00+00:00')))
    worker = pipeline / 'harness/tests/comparison/calibration/native_pd_worker.py'
    worker.parent.mkdir(parents=True)
    worker.write_text('def validate_promotion_receipt(config):\n'
                      '    return {"path": config["promotion_receipt"]}\n')
    config = dict(harness_worktree=str(pipeline / 'harness'), pair_warmup=True, preemptible=True,
                  promotion_receipt=str(receipt))
    case_source = pipeline / 'case_source'
    (case_source / 'inputs').mkdir(parents=True)
    (case_source / 'manifest.yaml').write_text('{}\n')
    (case_source / 'inputs/native_config.json').write_text(json.dumps(config))
    tag = 'p2-fixture'
    (artifacts / 'run_manifest.json').write_text(json.dumps(dict(
        run_tag=tag, case_id='fixture', execution_config='inputs/native_config.json',
        promotion_receipt={'path': str(receipt)},
        written_before_launch_utc='2026-01-02T00:00:00+00:00', resources={'preemptible': True})))
    (artifacts / 'launcher_receipt.json').write_text('{}\n')
    (artifacts / 'platform_terminal_receipt.json').write_text(json.dumps(dict(preemptible_label='yes')))
    inventory = dict(files=[dict(path=path.relative_to(artifacts).as_posix(),
                                 bytes=path.stat().st_size, sha256=digest(path))
                            for path in sorted(artifacts.rglob('*')) if path.is_file()])
    download = pipeline / 'download'
    (download / 'cloud_inventory.json').write_text(json.dumps(inventory))
    (download / 'download_receipt.json').write_text(json.dumps(dict(
        status='HASH_IDENTICAL_DOWNLOAD', mismatched=[],
        inventory_sha256=digest(download / 'cloud_inventory.json'))))
    for choice, expected in (('r4', R4), ('r5', R5)):
        result = compare.run_pipeline(case_source, download, pipeline / f'out_{choice}',
                                      reference_checker=choice)
        assert result['reference_checker'] == dict(name=choice, path=str(expected),
                                                   sha256=digest(expected))
        command = result['commands'][0]
        assert command['argv'][1] == result['reference_checker']['path']
        assert command['argv'][2:] == [str(pipeline / f'out_{choice}/r4_case'), tag, 'clean']
        assert command['exit_code'] != 0 and digest(command['argv'][1]) == result['reference_checker']['sha256']
