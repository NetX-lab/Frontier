"""Build package_01 for the H4b-(a) D-side handoff measurement (design.md section 7).

Copies the native06 record copy, applies only the listed changes, compares
every file with native06, seals the package, runs the run-manifest checks on a
scratch copy of the case, and runs ``launch.py --check-only``:

    python3 build_package.py
"""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'v4_p0_native_20261021' / 'candidate_h800_p0_06'
TARGET = HERE / 'package_01'
# native06 ran from this directory; its configs carry it in every path field.
SOURCE_ROOT = ('/data/ycfeng/Frontier/.worktrees/v4-balanced-native-model/task_memory/'
               'task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_06')
HARNESS = Path('/data/ycfeng/Frontier/.worktrees/h4b-lifecycle-harness')
NATIVE = Path('/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69')
NATIVE_COMMIT = '2cf0a6915ce544dc493a0990f2ea38d81601128a'
OBSERVER = 'tests/comparison/calibration/native_scheduler_lifecycle.py'
REPLAY = 'tests/comparison/dp_placement_pp/vllm_replay.py'
SCHEDULER_CLS = 'native_scheduler_lifecycle.LifecycleObservedScheduler'
NEW_PINS = ('vllm/v1/core/sched/async_scheduler.py', 'vllm/config/scheduler.py', 'vllm/v1/outputs.py')
RUN_TAG = 'dsv4-h800-p0-lifecycle-20261025-01'
OUTPUT_DIR = ('/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h800_p0_lifecycle_20261025/'
              f'case/runs/groundtruth_clean/{RUN_TAG}')
CASE_LINES = {
    'case_id: ': 'case_id: dsv4-h800-p0-lifecycle-fixed128-qps1',
    'run_generation: ': 'run_generation: v4-native-v028-h800-p0-lifecycle-20261025-01',
    'purpose: ': ('purpose: DIAGNOSTIC H4b-(a) D-side handoff measurement; native06 case and workload '
                  'plus the engine-core scheduler lifecycle observer; not an E2E reference\n'
                  'diagnostic_only: true\n'
                  'e2e_reference_admission: false'),
}
# Outputs of the native06 launch, not inputs.
LAUNCH_OUTPUTS = ('launcher_started.json', 'launcher_exit.json', 'manifest.json')
JSON_CHANGED = ('config.json', 'case/inputs/native_config.json', 'engine.json', 'case/inputs/engine.json',
                'overlay.json', 'source_lock.json', 'evidence_readiness.json')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(['git', '--no-optional-locks', '-C', str(HARNESS), *args],
                          check=True, capture_output=True).stdout


def dumps(value):
    return json.dumps(value, indent=2) + '\n'


def relocate(value):
    """Move native06 package paths, in values and keys, into package_01."""
    if isinstance(value, dict):
        return {relocate(key): relocate(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item) for item in value]
    if isinstance(value, str) and value.startswith(SOURCE_ROOT):
        return str(TARGET) + value[len(SOURCE_ROOT):]
    return value


def leaf_differences(old, new, prefix=()):
    if isinstance(old, dict) and isinstance(new, dict):
        paths = []
        for key in sorted(old.keys() | new.keys()):
            if key not in old or key not in new:
                paths.append(prefix + (key,))
            else:
                paths += leaf_differences(old[key], new[key], prefix + (key,))
        return paths
    return [] if old == new else [prefix]


def source_files():
    files = {}
    for path in sorted(SOURCE.rglob('*')):
        relative = path.relative_to(SOURCE).as_posix()
        if path.is_file() and relative not in LAUNCH_OUTPUTS and not relative.startswith('case/runs/'):
            files[relative] = path
    return files


def build(commit):
    TARGET.mkdir()
    for relative, path in source_files().items():
        (TARGET / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, TARGET / relative)
    config = relocate(json.loads((SOURCE / 'config.json').read_text()))
    harness_files = dict(config['harness_files'], **{OBSERVER: OBSERVER})
    for name, target in harness_files.items():
        (TARGET / target).write_bytes(git('show', f'{commit}:{name}'))

    overlay = relocate(json.loads((SOURCE / 'overlay.json').read_text()))
    (TARGET / 'overlay.json').write_text(dumps(overlay))
    pins = {name: sha256(NATIVE / name) for name in NEW_PINS}
    if pins.keys() & config['expected_source_hashes'].keys():
        raise ValueError('A new pin is already pinned in native06')
    engine = dict(config['engine'], scheduler_cls=SCHEDULER_CLS)
    config.update(engine=engine, harness_worktree=str(HARNESS), harness_files=harness_files,
                  expected_source_hashes=config['expected_source_hashes'] | pins,
                  run_tag=RUN_TAG, job_tag=RUN_TAG, output_dir=OUTPUT_DIR)
    config['retained_files']['overlay.json'] = sha256(TARGET / 'overlay.json')
    config['tool_hashes'] = {name: sha256(name) for name in
                             [*config['tool_hashes'], str(TARGET / OBSERVER)]}
    for name in ('config.json', 'case/inputs/native_config.json'):
        (TARGET / name).write_text(dumps(config))
    for name in ('engine.json', 'case/inputs/engine.json'):
        (TARGET / name).write_text(dumps(engine))

    lock = relocate(json.loads((SOURCE / 'source_lock.json').read_text()))
    lock['harness_commit'] = commit
    lock['committed_files'] = {target: dict(commit=commit, path=name, sha256=sha256(TARGET / target))
                               for name, target in harness_files.items()}
    lock['added_source_pins'] = pins
    (TARGET / 'source_lock.json').write_text(dumps(lock))
    readiness = json.loads((SOURCE / 'evidence_readiness.json').read_text())
    readiness.update(
        purpose='DIAGNOSTIC H4b-(a): native D-side handoff states; client metrics enter no gate',
        harness_commit=commit, added_source_pins=sorted(pins),
        scheduler_observer=dict(scheduler_cls=SCHEDULER_CLS, module=OBSERVER,
                                sha256=sha256(TARGET / OBSERVER)))
    (TARGET / 'evidence_readiness.json').write_text(dumps(readiness))

    lines = []
    for line in (SOURCE / 'case/manifest.yaml').read_text().splitlines():
        prefix = next((key for key in CASE_LINES if line.startswith(key)), None)
        if prefix is not None:
            line = CASE_LINES[prefix]
        elif line.startswith('  sha256: ') and lines[-1] == '  manifest: overlay.json':
            line = '  sha256: ' + config['retained_files']['overlay.json']
        lines.append(line)
    (TARGET / 'case/manifest.yaml').write_text('\n'.join(lines) + '\n')
    return config, pins


def compare(config, pins):
    """Fail unless package_01 differs from native06 only by the listed changes."""
    import yaml
    old_files, new_files = source_files(), {
        path.relative_to(TARGET).as_posix(): path for path in TARGET.rglob('*') if path.is_file()}
    if new_files.keys() - old_files.keys() != {OBSERVER} or old_files.keys() - new_files.keys():
        raise ValueError(f'Unexpected file set: {sorted(new_files.keys() ^ old_files.keys())}')
    replay, observer = str(TARGET / REPLAY), str(TARGET / OBSERVER)
    config_paths = {('harness_worktree',), ('harness_files', OBSERVER), ('tool_hashes', replay),
                    ('tool_hashes', observer), ('run_tag',), ('job_tag',), ('output_dir',),
                    ('engine', 'scheduler_cls'), ('retained_files', 'overlay.json'),
                    *(('expected_source_hashes', name) for name in pins)}
    lock_paths = {('harness_commit',), ('added_source_pins',), ('committed_files', OBSERVER),
                  ('committed_files', REPLAY, 'sha256'),
                  *(('committed_files', name, 'commit') for name in config['harness_files'].values()
                    if name != OBSERVER)}
    allowed = {'config.json': config_paths, 'case/inputs/native_config.json': config_paths,
               'engine.json': {('scheduler_cls',)}, 'case/inputs/engine.json': {('scheduler_cls',)},
               'overlay.json': set(), 'source_lock.json': lock_paths,
               'evidence_readiness.json': {('purpose',), ('harness_commit',), ('added_source_pins',),
                                           ('scheduler_observer',)}}
    report = {}
    for relative in sorted(old_files):
        old, new = old_files[relative].read_bytes(), new_files[relative].read_bytes()
        if relative in JSON_CHANGED:
            changed = set(leaf_differences(relocate(json.loads(old)), json.loads(new)))
            if not changed <= allowed[relative]:
                raise ValueError(f'{relative}: unexpected changes {sorted(changed - allowed[relative])}')
            report[relative] = sorted('.'.join(path) for path in changed)
        elif relative == 'case/manifest.yaml':
            changed = set(leaf_differences(yaml.safe_load(old), yaml.safe_load(new)))
            expected = {('case_id',), ('run_generation',), ('purpose',), ('diagnostic_only',),
                        ('e2e_reference_admission',), ('runtime_source_overlay', 'sha256')}
            if changed != expected:
                raise ValueError(f'case/manifest.yaml: changes {sorted(changed)}')
            report[relative] = sorted('.'.join(path) for path in changed)
        elif relative == REPLAY:
            if old == new:
                raise ValueError('vllm_replay.py lacks the scheduler_cls passthrough')
            report[relative] = 'restaged from the harness commit'
        elif old != new:
            raise ValueError(f'Unexpected byte change: {relative}')
    return report


def preflight(config, commit):
    """Run the launcher's run-manifest checks against a scratch copy of the case."""
    scratch = Path(tempfile.mkdtemp(prefix='package_01_preflight_', dir='/data/ycfeng/tmp/glm53_dsv4/h4b'))
    case = scratch / 'case'
    shutil.copytree(TARGET / 'case', case)
    trial = dict(config, case_dir=str(case))
    (case / 'inputs/native_config.json').write_text(dumps(trial))
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(TARGET))
    from native_pd_worker import write_run_manifest
    manifest = json.loads(write_run_manifest(trial, 'preflight only; no submission').read_text())
    if (manifest['harness_commit'] != commit or manifest['groundtruth_commit'] != NATIVE_COMMIT
            or OBSERVER not in manifest['harness_sha256'] or manifest['run_tag'] != RUN_TAG):
        raise ValueError('Preflight run manifest does not bind the observer harness')
    return str(scratch)


def main():
    if git('status', '--short', '--untracked-files=all').strip():
        raise ValueError('Commit the harness worktree before building')
    commit = git('rev-parse', 'HEAD').decode().strip()
    native = subprocess.run(['git', '-C', str(NATIVE), 'rev-parse', 'HEAD'],
                            check=True, capture_output=True, text=True).stdout.strip()
    if native != NATIVE_COMMIT:
        raise ValueError(f'NATIVE checkout moved: {native}')
    config, pins = build(commit)
    report = compare(config, pins)
    sealed = json.loads(subprocess.run([sys.executable, str(TARGET / 'seal.py')], check=True,
                                       capture_output=True, text=True).stdout)
    scratch = preflight(config, commit)
    check = subprocess.run([sys.executable, str(TARGET / 'launch.py'), '--check-only'],
                           capture_output=True, text=True)
    if check.returncode or check.stdout.strip() != 'INPUT_HASHES_PASS_NO_SUBMISSION':
        raise ValueError(f'launch.py --check-only failed: {check.stdout} {check.stderr}')
    result = dict(harness_commit=commit, native_commit=native, package=str(TARGET),
                  manifest_sha256=sha256(TARGET / 'manifest.json'), sealed=sealed, added_pins=pins,
                  changes=report, preflight_scratch=scratch, check_only=check.stdout.strip())
    (HERE / 'package_01_build.json').write_text(dumps(result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
