"""Build a sealed H200 mainline case (design.md sections 4, 5 and 9).

Copies the case's H800 source package, applies only the listed changes,
compares every file with the source, seals the package, and runs the checks
that need no GPU: the run-manifest preflight on a scratch copy of the case, the
R3 Frontier reconstruction, the offline launch path, and ``launch.py --check-only``:

    python3 build_package.py p2_32_reference [--package package_01]

The package directory must not exist yet.
"""

import argparse
import ast
import difflib
from datetime import datetime, timezone
import hashlib
import importlib
import json
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREES = Path('/data/ycfeng/Frontier/.worktrees')
NATIVE = Path('/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69')
NATIVE_COMMIT = '2cf0a6915ce544dc493a0990f2ea38d81601128a'
R3_CHECKER = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/'
                  'task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py')
SCRATCH = Path('/data/ycfeng/tmp/glm53_dsv4/h200')
CLOUD = '/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18'
HARDWARE = dict(gpu_type='H200', charged_group='step_main')
TASK_DIR = 'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800'
# Each source package path is relative to the task directory. A source config carries the
# package directory inside the worktree it was built in (`built_in`) in every path field.
SOURCES = {
    'reference': dict(package='v4_p0_native_20261021/candidate_h800_p0_06', built_in='v4-balanced-native-model',
                      harness='v4-balanced-native-model', harness_commit='b62c155abd981a71e12e62aa0c0fe3990741d2f8'),
    'profile': dict(package='v4_p0_native_20261021/candidate_h800_p0_profile_02', built_in='v4-native-profile-run',
                    harness='v4-native-profile-run', harness_commit='a90c9dbbee53e2cabbfac73ba8490ffb958ce8eb'),
    'formal': dict(package='h200_mainline/p2_64_reference/package_01', built_in='glm53-dsv4-vllm028',
                   harness='p2-64-pair-warmup', harness_commit=None),
    'wave': dict(package='h200_mainline/p2_64_lifecycle/package_01', built_in='glm53-dsv4-vllm028',
                 harness='p2-64-wave-harness', harness_commit=None),
    # H4b-(a): native06 plus the engine-core scheduler lifecycle observer.
    'lifecycle': dict(package='h4b_lifecycle_measurement/package_01', built_in='glm53-dsv4-vllm028',
                      harness='h4b-lifecycle-harness', harness_commit='7ac25c7471fc2ae73657fc529a89bc651c3326ba'),
}
# h200_limit: the session's non-preemptible H200 limit, running plus queued, when the case was built.
CASES = {
    'p2_64_formal': dict(source='formal', dp=4, h200_limit=32,
                         case_id='dsv4-h200-p2-64-canonical-fixed128-qps1',
                         run_tag='dsv4-h200-p2-64-formal-20261102-01',
                         purpose='Formal P2 H200-64 native reference under D-P2-64-FORMAL-B; '
                                 'pair-warmed, wholly preemptible; issued 30% receipt precedes build'),
    'p2_64_wave': dict(source='wave', dp=4, h200_limit=64, case_id='dsv4-h200-p2-64-wave-fixed128-qps1',
                       run_tag='dsv4-h200-p2-64-wave-20261030-01',
                       purpose='DIAGNOSTIC P2-64 native DP wave restart under D-P2-64-WAVE-DIAGNOSTIC; '
                               'one repetition, 16 warmup and 128 formal requests; no E2E reference admission; '
                               'never qualifies H800 cells'),
    'p2_32_reference': dict(source='reference', dp=2, h200_limit=64, case_id='dsv4-h200-p2-canonical-fixed128-qps1',
                            run_tag='dsv4-h200-p2-canonical-20261025-01',
                            purpose='P2 H200-32 native reference for DSV4 under D-H200-TEMP-MAINLINE; '
                                    'Frontier comparisons against it are DIAGNOSTIC'),
    'p2_32_profile': dict(source='profile', dp=2, h200_limit=64, case_id='dsv4-h200-p2-profile-fixed128-qps1',
                          run_tag='dsv4-h200-p2-profile-20261025-01',
                          purpose='Diagnostic native serving profile on H200-32; source of the H200 step-cost '
                                  'table; no E2E or reference admission'),
    'p2_64_reference': dict(source='reference', dp=4, h200_limit=64, case_id='dsv4-h200-p2-64-canonical-fixed128-qps1',
                            run_tag='dsv4-h200-p2-64-canonical-20261025-01',
                            purpose='P2 H200-64 native reference and basic validation for DSV4 under '
                                    'D-H200-TEMP-MAINLINE; Frontier comparisons against it are DIAGNOSTIC'),
    'p2_64_profile': dict(source='profile', dp=4, h200_limit=64, case_id='dsv4-h200-p2-64-profile-fixed128-qps1',
                          run_tag='dsv4-h200-p2-64-profile-20261025-01',
                          purpose='Diagnostic native serving profile on H200-64; source of the H200-64 '
                                  'step-cost table; no E2E or reference admission'),
    'p2_32_lifecycle': dict(source='lifecycle', dp=2, h200_limit=32, case_id='dsv4-h200-p2-lifecycle-fixed128-qps1',
                            run_tag='dsv4-h200-p2-lifecycle-20261025-01',
                            purpose='DIAGNOSTIC H4b-(a) D-side handoff measurement on H200-32 under '
                                    'D-H200-H4B-LIFECYCLE; P2-32 case and workload plus the engine-core scheduler '
                                    'lifecycle observer; not an E2E reference; never qualifies H800 cells'),
    'p2_64_lifecycle': dict(source='lifecycle', dp=4, h200_limit=64, case_id='dsv4-h200-p2-64-lifecycle-fixed128-qps1',
                            run_tag='dsv4-h200-p2-64-lifecycle-20261026-01',
                            purpose='DIAGNOSTIC H4b-(a) D-side handoff measurement on H200-64 under '
                                    'D-H200-MAINLINE-CONTINUE; P2-64 case and workload plus the engine-core '
                                    'scheduler lifecycle observer; not an E2E reference; never qualifies H800 cells'),
}
FRONTIER_INPUT = 'case/inputs/frontier.json'
ROLE_KEY = 'cluster_config_{}_replica_config_{}'
ROLES = ('prefill', 'decode')
# Outputs of the source launch, not inputs.
LAUNCH_OUTPUTS = ('launcher_started.json', 'launcher_exit.json', 'manifest.json')
CONFIGS = ('config.json', 'case/inputs/native_config.json')
ENGINES = ('engine.json', 'case/inputs/engine.json')
JSON_CHANGED = (*CONFIGS, *ENGINES, FRONTIER_INPUT, 'overlay.json', 'source_lock.json', 'evidence_readiness.json')


class Case:
    def __init__(self, name, package):
        case = CASES[name]
        source = SOURCES[case['source']]
        self.wave = case['source'] == 'wave'
        self.formal = case['source'] == 'formal'
        self.name, self.profile, self.dp, self.h200_limit = name, case['source'] == 'profile', case['dp'], case['h200_limit']
        self.replicas = 2 * self.dp
        self.case_id, self.run_tag, self.purpose = case['case_id'], case['run_tag'], case['purpose']
        self.source = HERE.parent / source['package']
        self.harness = WORKTREES / source['harness']
        self.harness_commit = source['harness_commit']
        self.source_root = str(WORKTREES / source['built_in'] / TASK_DIR / source['package'])
        self.target = HERE / name / package
        self.output_dir = f"{CLOUD}/{self.run_tag.replace('-', '_')}/case/runs/groundtruth_clean/{self.run_tag}"
        self.launch_lines = [
            ("assert (config['gpu_type'], config['charged_group'], config['gpu'], config['replica']) == "
             "('H800', 'codesign', 8, 4)",
             "assert (config['gpu_type'], config['charged_group'], config['gpu'], config['replica']) == "
             f"('{HARDWARE['gpu_type']}', '{HARDWARE['charged_group']}', 8, {self.replicas})", 1)]
        if self.wave or self.formal:
            self.launch_lines = []
        if self.dp != 2 and not (self.wave or self.formal):
            self.launch_lines.append(("assert config['engine']['data_parallel_size'] == 2",
                                      f"assert config['engine']['data_parallel_size'] == {self.dp}", 1))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(repository, *args):
    return subprocess.run(['git', '--no-optional-locks', '-C', str(repository), *args],
                          check=True, capture_output=True, text=True).stdout


def dumps(value):
    return json.dumps(value, indent=2) + '\n'


def relocate(case, value):
    """Move source package paths, in values and keys, into the new package."""
    if isinstance(value, dict):
        return {relocate(case, key): relocate(case, item) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(case, item) for item in value]
    if isinstance(value, str) and value.startswith(case.source_root):
        return str(case.target) + value[len(case.source_root):]
    return value


def replace_lines(text, replacements):
    lines = text.splitlines()
    for old, new, count in replacements:
        if lines.count(old) != count:
            raise ValueError(f'Expected {count} line(s) {old!r}, found {lines.count(old)}')
        lines = [new if line == old else line for line in lines]
    return '\n'.join(lines) + '\n'


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


def source_files(case):
    files = {}
    for path in sorted(case.source.rglob('*')):
        relative = path.relative_to(case.source).as_posix()
        if path.is_file() and relative not in LAUNCH_OUTPUTS and not relative.startswith('case/runs/'):
            files[relative] = path
    return files


def engine_update(case, engine):
    engine = dict(engine, data_parallel_size=case.dp)
    if case.wave:
        engine['scheduler_cls'] = 'native_dp_wave.WaveObservedScheduler'
    if case.profile:
        engine['profiler_config'] = dict(engine['profiler_config'], torch_profiler_dir=f'{case.output_dir}/torch_traces')
    return engine


def manifest_lines(case, config):
    """case/manifest.yaml line replacements as (old, new, occurrences)."""
    text = (case.source / 'case/manifest.yaml').read_text()
    lines = text.splitlines()
    overlay = lines[lines.index('  manifest: overlay.json') + 1]
    top = {line.split(': ', 1)[0]: line for line in lines if not line.startswith(' ') and ': ' in line}
    replacements = [
        (top['case_id'], f'case_id: {case.case_id}', 1),
        (top['run_generation'], f"run_generation: v4-native-v028-{case.run_tag.removeprefix('dsv4-')}", 1),
        (top['purpose'], f'purpose: {case.purpose}', 1),
        (overlay, '  sha256: ' + config['retained_files']['overlay.json'], 1),
        ('  priority: P0', '  priority: P2', 1),
        ('  gpu_type: H800', f"  gpu_type: {HARDWARE['gpu_type']}", 1),
        ('  charged_group: codesign', f"  charged_group: {HARDWARE['charged_group']}", 1)]
    limit = '  H200_nonpreemptible_running_plus_queued_limit: '
    old_limit = next((line for line in lines if line.startswith(limit)), None)
    if old_limit is not None and old_limit != f'{limit}{case.h200_limit}':
        replacements.append((old_limit, f'{limit}{case.h200_limit}', 1))
    if case.dp != 2:
        replacements += [('      dp: 2', f'      dp: {case.dp}', 2), ('      ep: 16', f'      ep: {8 * case.dp}', 2)]
    return replacements


def copy_source(case):
    case.target.mkdir(parents=True)
    for relative, path in source_files(case).items():
        (case.target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, case.target / relative)


def build(case):
    if case.formal:
        return build_formal(case)
    if case.wave:
        return build_wave(case)
    copy_source(case)
    (case.target / 'launch.py').write_text(replace_lines((case.source / 'launch.py').read_text(), case.launch_lines))

    old_config = json.loads((case.source / 'config.json').read_text())
    ep = old_config['engine']['tensor_parallel_size'] * case.dp
    frontier = json.loads((case.source / FRONTIER_INPUT).read_text())
    frontier.update({ROLE_KEY.format(role, 'device'): HARDWARE['gpu_type'].lower() for role in ROLES})
    if case.dp != 2:
        frontier.update({'replica_config_attn_dp': case.dp,
                         **{ROLE_KEY.format(role, 'moe_expert_parallel_size'): ep for role in ROLES}})
    (case.target / FRONTIER_INPUT).write_text(dumps(frontier))
    (case.target / 'overlay.json').write_text(dumps(relocate(case, json.loads((case.source / 'overlay.json').read_text()))))

    config = relocate(case, old_config)
    engine = engine_update(case, config['engine'])
    config.update(HARDWARE, engine=engine, replica=case.replicas, run_tag=case.run_tag, job_tag=case.run_tag,
                  output_dir=case.output_dir)
    changed = {}
    for name in config['retained_files']:
        new = sha256(case.target / name)
        if new != config['retained_files'][name]:
            changed[name] = config['retained_files'][name]
            config['retained_files'][name] = new
    config['tool_hashes'] = {name: sha256(name) for name in config['tool_hashes']}
    for name in CONFIGS:
        (case.target / name).write_text(dumps(config))
    for name in ENGINES:
        (case.target / name).write_text(dumps(engine_update(case, json.loads((case.source / name).read_text()))))

    lock = relocate(case, json.loads((case.source / 'source_lock.json').read_text()))
    if lock['native_commit'] != NATIVE_COMMIT:
        raise ValueError('The source lock names another NATIVE commit')
    for name in changed.keys() & lock['retained_files'].keys():
        entry = lock['retained_files'][name]
        entry.update(sha256=config['retained_files'][name], source=str(case.source / name),
                     source_sha256=changed[name], changed=True)
        if 'staged_sha256' in entry:
            entry['staged_sha256'] = config['retained_files'][name]
    (case.target / 'source_lock.json').write_text(dumps(lock))

    readiness = json.loads((case.source / 'evidence_readiness.json').read_text())
    gpus = config['replica'] * config['gpu']
    if gpus != old_config['replica'] * old_config['gpu']:
        readiness['cost'].update(
            max_payload_gpu_hours=readiness['cost']['payload_guard_seconds'] * gpus / 3600,
            minimum_schedule_gpu_hours=readiness['cost']['scheduled_arrival_span_seconds'] * gpus / 3600)
    readiness.update(purpose=case.purpose,
                     target=(f"{HARDWARE['gpu_type']}/{HARDWARE['charged_group']}{gpus} nonpreemptible; "
                             f'P{gpus // 2}+D{gpus // 2} TP8/DP{case.dp}/EP{ep}/PP1'))
    (case.target / 'evidence_readiness.json').write_text(dumps(readiness))

    manifest = case.target / 'case/manifest.yaml'
    manifest.write_text(replace_lines(manifest.read_text(), manifest_lines(case, config)))
    return config, sorted(changed)


def compare(case, config, changed_retained):
    """Fail unless the package differs from its source only by the listed changes."""
    if case.formal:
        return compare_formal(case, config)
    if case.wave:
        return compare_wave(case, config)
    import yaml
    old_files, new_files = source_files(case), {
        path.relative_to(case.target).as_posix(): path for path in case.target.rglob('*') if path.is_file()}
    if new_files.keys() != old_files.keys():
        raise ValueError(f'Unexpected file set: {sorted(new_files.keys() ^ old_files.keys())}')
    dp_changed = case.dp != 2
    engine_paths = ({('data_parallel_size',)} if dp_changed else set()) | (
        {('profiler_config', 'torch_profiler_dir')} if case.profile else set())
    config_paths = ({('gpu_type',), ('charged_group',), ('run_tag',), ('job_tag',), ('output_dir',)}
                    | {('engine', *path) for path in engine_paths}
                    | ({('replica',)} if dp_changed else set())
                    | {('retained_files', name) for name in changed_retained}
                    | {('tool_hashes', str(case.target / name)) for name in changed_retained
                       if str(case.target / name) in config['tool_hashes']})
    frontier_paths = {(ROLE_KEY.format(role, 'device'),) for role in ROLES} | (
        {('replica_config_attn_dp',)} | {(ROLE_KEY.format(role, 'moe_expert_parallel_size'),) for role in ROLES}
        if dp_changed else set())
    # A changed retained file's lock entry may change in any of its own keys and must change its hash.
    locked = json.loads(old_files['source_lock.json'].read_text())['retained_files'].keys() & set(changed_retained)
    lock_keys = ('sha256', 'source', 'source_sha256', 'changed', 'staged_sha256')
    allowed = {**{name: config_paths for name in CONFIGS}, **{name: engine_paths for name in ENGINES},
               FRONTIER_INPUT: frontier_paths, 'overlay.json': set(),
               'evidence_readiness.json': {('purpose',), ('target',), ('cost', 'max_payload_gpu_hours'),
                                           ('cost', 'minimum_schedule_gpu_hours')}}
    if not dp_changed:
        allowed['evidence_readiness.json'] -= {('cost', 'max_payload_gpu_hours'), ('cost', 'minimum_schedule_gpu_hours')}
    report = {}
    for relative in sorted(old_files):
        old, new = old_files[relative].read_bytes(), new_files[relative].read_bytes()
        if relative == 'source_lock.json':
            changed = set(leaf_differences(relocate(case, json.loads(old)), json.loads(new)))
            required = {('retained_files', name, 'sha256') for name in locked}
            if not required <= changed <= {('retained_files', name, key) for name in locked for key in lock_keys}:
                raise ValueError(f'source_lock.json: changes {sorted(changed)}')
            report[relative] = sorted('.'.join(path) for path in changed)
        elif relative in JSON_CHANGED:
            changed = set(leaf_differences(relocate(case, json.loads(old)), json.loads(new)))
            if changed != allowed[relative]:
                raise ValueError(f'{relative}: changes {sorted(changed)} differ from {sorted(allowed[relative])}')
            report[relative] = sorted('.'.join(path) for path in changed)
        elif relative == 'case/manifest.yaml':
            changed = set(leaf_differences(yaml.safe_load(old), yaml.safe_load(new)))
            expected = {('case_id',), ('run_generation',), ('purpose',), ('runtime_source_overlay', 'sha256'),
                        ('policy', 'priority'), ('policy', 'gpu_type'), ('policy', 'charged_group')}
            old_limit = yaml.safe_load(old)['policy'].get('H200_nonpreemptible_running_plus_queued_limit')
            if old_limit not in (None, case.h200_limit):
                expected.add(('policy', 'H200_nonpreemptible_running_plus_queued_limit'))
            if dp_changed:
                expected |= {('topology', 'vllm', role, key) for role in ROLES for key in ('dp', 'ep')}
            if changed != expected:
                raise ValueError(f'case/manifest.yaml: changes {sorted(changed)}')
            report[relative] = sorted('.'.join(path) for path in changed)
        elif relative == 'launch.py':
            diff = [line for line in difflib.unified_diff(old.decode().splitlines(), new.decode().splitlines(),
                                                          lineterm='', n=0)
                    if line[:1] in '+-' and not line.startswith(('+++', '---'))]
            if sorted(diff) != sorted([f'-{old}' for old, _, _ in case.launch_lines]
                                      + [f'+{new}' for _, new, _ in case.launch_lines]):
                raise ValueError(f'launch.py: unexpected changes {diff}')
            report[relative] = diff
        elif old != new:
            raise ValueError(f'Unexpected byte change: {relative}')
    return report


# D-P2-64-FORMAL-B keeps the retained engine, workload and observers intact.
FORMAL_MODULES = ('tests/comparison/calibration/native_pd_worker.py',
                  'tests/comparison/calibration/pd_replay.py',
                  'tests/comparison/calibration/pd_proxy.py',
                  'tests/comparison/dp_placement_pp/vllm_replay.py')
PROMOTION_PATH = HERE.parent / 'h3o_operator_costs/formal_p2_20261102/promotion/receipt_p2_32_to_p2_64.json'
PROMOTION_SHA256 = 'd505eecea90c0ea0a285a3cff76086e11b2c3914a88ad2a188dd9c94323f7596'
FORMAL_LAUNCH_INSERT = "from native_pd_worker import validate_promotion_receipt\nvalidate_promotion_receipt(config)\n"
PREEMPTIBLE_LINE = '        replica=config["replica"], image=config["image"], positive_tags=[config["gpu_type"]],'


def formal_harness_differences(case):
    config = json.loads((case.source / 'config.json').read_text())
    return [name for name, target in config['harness_files'].items()
            if name not in FORMAL_MODULES
            and sha256(case.harness / name) != sha256(case.source / target)]


def formal_documents(case):
    """Construct the exact allowed target; comparison rederives it from source."""
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location('formal_receipt_validator', case.harness / FORMAL_MODULES[0])
    worker = module_from_spec(spec)
    spec.loader.exec_module(worker)
    binding = dict(path=str(PROMOTION_PATH), sha256=PROMOTION_SHA256, build_utc=case.build_utc)
    worker.validate_promotion_receipt(dict(promotion_receipt=binding))
    if formal_harness_differences(case):
        raise ValueError('Unapproved retained harness changes: ' + str(formal_harness_differences(case)))
    files = {name: path.read_bytes() for name, path in source_files(case).items()}
    config = relocate(case, json.loads(files['config.json']))
    config.update(run_tag=case.run_tag, job_tag=case.run_tag, output_dir=case.output_dir,
                  harness_worktree=str(case.harness), pair_warmup=True, preemptible=True,
                  promotion_receipt=binding)
    lock = relocate(case, json.loads(files['source_lock.json']))
    lock['harness_commit'] = case.harness_commit
    # Rebind unchanged staged modules to the same new reviewed commit too.
    for name, target in config['harness_files'].items():
        if name in FORMAL_MODULES:
            files[target] = (case.harness / name).read_bytes()
        lock['committed_files'][target] = dict(commit=case.harness_commit, path=name,
                                              sha256=hashlib.sha256(files[target]).hexdigest())
    launch = files['launch.py'].decode()
    marker = "if sys.argv[1:] == ['--check-only']:"
    if launch.count(marker) != 1:
        raise ValueError('Non-unique launch receipt validation boundary')
    files['launch.py'] = launch.replace(marker, FORMAL_LAUNCH_INSERT + marker).encode()
    submit = files['submit_native_diagnostic.py'].decode()
    if submit.splitlines().count(PREEMPTIBLE_LINE) != 1:
        raise ValueError('Non-unique ResourceConfig preemptible boundary')
    files['submit_native_diagnostic.py'] = submit.replace(
        PREEMPTIBLE_LINE, PREEMPTIBLE_LINE + '\n        preemptible=config.get("preemptible", False),').encode()
    overlay = relocate(case, json.loads(files['overlay.json']))
    files['overlay.json'] = dumps(overlay).encode()
    lock['runtime_overlay'] = overlay
    changed = []
    for name, old_digest in config['retained_files'].items():
        digest = hashlib.sha256(files[name]).hexdigest()
        if digest != old_digest:
            changed.append(name)
            config['retained_files'][name] = digest
            if name in lock['retained_files']:
                entry = lock['retained_files'][name]
                entry.update(sha256=digest, source=str(case.source / name),
                             source_sha256=old_digest, changed=True)
                if 'staged_sha256' in entry:
                    entry['staged_sha256'] = digest
    # Paths have been relocated; derive hashes from the candidate bytes rather
    # than reading a partially built package.
    for path in config['tool_hashes']:
        relative = Path(path).relative_to(case.target).as_posix()
        config['tool_hashes'][path] = hashlib.sha256(files[relative]).hexdigest()
    for name in CONFIGS:
        files[name] = dumps(config).encode()
    files['source_lock.json'] = dumps(lock).encode()
    readiness = json.loads(files['evidence_readiness.json'])
    readiness.update(purpose=case.purpose, target='H200/step_main64 preemptible; P32+D32 TP8/DP4/EP32/PP1')
    readiness['cost'].update(preemptible=True, H200_nonpreemptible_running_plus_queued_limit=32,
                            requested_gpus=64, timeout_gpu_hours=64.0,
                            estimated_container_gpu_hours=24.322222222222223)
    files['evidence_readiness.json'] = dumps(readiness).encode()
    manifest = files['case/manifest.yaml'].decode()
    purpose = next(line for line in manifest.splitlines() if line.startswith('purpose: '))
    generation = next(line for line in manifest.splitlines() if line.startswith('run_generation: '))
    files['case/manifest.yaml'] = replace_lines(manifest, [
        (purpose, 'purpose: ' + case.purpose, 1),
        (generation, f"run_generation: v4-native-v028-{case.run_tag.removeprefix('dsv4-')}", 1),
        ('  preemptible: false', '  preemptible: true', 1),
        ('  H200_nonpreemptible_running_plus_queued_limit: 64',
         '  H200_nonpreemptible_running_plus_queued_limit: 32', 1),
        ('  sha256: ' + json.loads((case.source / 'config.json').read_text())['retained_files']['overlay.json'],
         '  sha256: ' + config['retained_files']['overlay.json'], 1),
    ]).encode()
    return files, config, changed


def build_formal(case):
    case.build_utc = datetime.now(timezone.utc).isoformat()
    files, config, changed = formal_documents(case)  # receipt check precedes mkdir
    case.target.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        destination = case.target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    # Loading the worker above searched sys.path before the package root
    # existed, which cached that root as having no importer.
    importlib.invalidate_caches()
    return config, changed


def compare_formal(case, config):
    """Fail on any byte beyond the declared transformations/restaged harness."""
    expected, expected_config, _ = formal_documents(case)
    actual = {p.relative_to(case.target).as_posix(): p for p in case.target.rglob('*') if p.is_file()}
    if actual.keys() != expected.keys() or config != expected_config:
        raise ValueError('Unexpected formal package file set or config')
    report = {}
    for name, content in expected.items():
        if actual[name].read_bytes() != content:
            raise ValueError(f'Unexpected formal package bytes: {name}')
        original = (case.source / name).read_bytes()
        if original != content:
            if name.endswith('.json'):
                paths = leaf_differences(relocate(case, json.loads(original)), json.loads(content))
                report[name] = ['.'.join(path) for path in paths]
            else:
                report[name] = list(difflib.unified_diff(original.decode().splitlines(), content.decode().splitlines(), n=0))
    return report


# The approved wave package changes only these harness files. Existing drivers
# remain the lifecycle package's bytes and must exist at the actual harness HEAD.
WAVE_MODULES = ('native_dp_wave.py', 'native_dp_wave_worker.py')
WAVE_RESTAGED = ('native_scheduler_lifecycle.py', 'native_full_structure.py', *WAVE_MODULES)
TRACE_MANIFEST = 'case/inputs/trace/manifest.json'
TRACE_SUMS = 'case/inputs/trace/SHA256SUMS'


def wave_harness_differences(case):
    config = json.loads((case.source / 'config.json').read_text())
    differences = [name for name, target in config['harness_files'].items()
        if Path(name).name not in WAVE_RESTAGED
        and sha256(case.harness / name) != sha256(case.source / target)]
    # The inventory may gain the worker bootstrap, but its existing executable
    # code must remain the source package's. Ignore comments/spacing only.
    name = 'tests/comparison/calibration/native_full_structure.py'
    source = ast.parse((case.source / config['harness_files'][name]).read_text())
    current = ast.parse((case.harness / name).read_text())
    bootstraps = [node for node in current.body if isinstance(node, ast.If) and
        any(isinstance(part, ast.ImportFrom) and part.module=='native_dp_wave_worker'
            for part in ast.walk(node))]
    current.body = [node for node in current.body if node not in bootstraps]
    if len(bootstraps)!=1 or ast.dump(source)!=ast.dump(current):
        differences.append(name)
    return differences

def build_wave(case):
    """Copy P2-64 lifecycle bytes, changing only the approved observer bindings."""
    copy_source(case)
    old = json.loads((case.source / 'config.json').read_text())
    config = relocate(case, old)
    engine = engine_update(case, config['engine'])
    config.update(engine=engine, run_tag=case.run_tag, job_tag=case.run_tag,
                  output_dir=case.output_dir, harness_worktree=str(case.harness))
    lock = relocate(case, json.loads((case.source / 'source_lock.json').read_text()))
    lock['harness_commit'] = case.harness_commit
    lock['qualification'] = case.purpose
    for module_name in WAVE_RESTAGED:
        name = f'tests/comparison/calibration/{module_name}'
        target = config['harness_files'].setdefault(name, name)
        content = (case.harness / name).read_bytes()
        (case.target / target).write_bytes(content)
        digest = sha256(case.target / target)
        config['tool_hashes'][str(case.target / target)] = digest
    # Rebind every staged file to one actual clean commit; never silently copy
    # unrelated HEAD changes into the diagnostic generation.
    for source, target in config['harness_files'].items():
        lock['committed_files'][target] = dict(commit=case.harness_commit,
                                               path=source, sha256=sha256(case.target / target))
    trace = json.loads((case.source / TRACE_MANIFEST).read_text())
    repeat_indices = [1]
    trace['repetitions'].update(clean_repetitions=len(repeat_indices), repeat_indices=repeat_indices,
        request_id_binding='the harness qualifies every formal and warmup template ID with repeat=1 before dispatch',
        sequence='warmup -> both roles idle and request-owned KV released -> formal -> both roles idle and request-owned KV released; once')
    (case.target / TRACE_MANIFEST).write_text(dumps(trace))
    shutil.copy2(case.target / TRACE_MANIFEST, case.target / 'trace/manifest.json')
    manifest_hash = sha256(case.target / TRACE_MANIFEST)
    sums = (case.source / TRACE_SUMS).read_text().splitlines()
    sums = [f'{manifest_hash}  manifest.json' if line.endswith('  manifest.json') else line for line in sums]
    (case.target / TRACE_SUMS).write_text('\n'.join(sums)+'\n')
    shutil.copy2(case.target / TRACE_SUMS, case.target / 'trace/SHA256SUMS')
    overlay = relocate(case, json.loads((case.source / 'overlay.json').read_text()))
    (case.target / 'overlay.json').write_text(dumps(overlay))
    lock['runtime_overlay'] = overlay
    changed = []
    for name in config['retained_files']:
        digest = sha256(case.target / name)
        if digest != config['retained_files'][name]:
            previous = config['retained_files'][name]
            config['retained_files'][name] = digest
            changed.append(name)
            if name in lock['retained_files']:
                lock['retained_files'][name].update(sha256=digest, source=str(case.source / name),
                    source_sha256=previous, changed=True)
                if 'staged_sha256' in lock['retained_files'][name]:
                    lock['retained_files'][name]['staged_sha256'] = digest
    config['tool_hashes'] = {name: sha256(name) for name in config['tool_hashes']}
    for name in CONFIGS:
        (case.target / name).write_text(dumps(config))
    for name in ENGINES:
        (case.target / name).write_text(dumps(engine))
    (case.target / 'source_lock.json').write_text(dumps(lock))
    import yaml
    manifest = yaml.safe_load((case.source / 'case/manifest.yaml').read_text())
    manifest.update(case_id=case.case_id, run_generation=f"v4-native-v028-{case.run_tag.removeprefix('dsv4-')}",
                    purpose=case.purpose)
    manifest['runtime_source_overlay']['sha256'] = config['retained_files']['overlay.json']
    manifest['workload_sha256'].update({'manifest.json':manifest_hash, 'SHA256SUMS':sha256(case.target / TRACE_SUMS)})
    (case.target / 'case/manifest.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
    readiness = json.loads((case.source / 'evidence_readiness.json').read_text())
    readiness.update(purpose=case.purpose, harness_commit=case.harness_commit,
        wave_observers={name:sha256(case.target / f'tests/comparison/calibration/{name}') for name in WAVE_RESTAGED})
    readiness['scheduler_observer'].update(scheduler_cls=engine['scheduler_cls'],
        module='tests/comparison/calibration/native_dp_wave.py',
        sha256=sha256(case.target / 'tests/comparison/calibration/native_dp_wave.py'))
    span = ((trace['arrivals']['last_arrival_s'] - trace['arrivals']['first_arrival_s']) +
            (trace['warmup']['last_arrival_s'] - trace['warmup']['first_arrival_s'])) * len(repeat_indices)
    readiness['cost'].update(scheduled_arrival_span_seconds=span,
        minimum_schedule_gpu_hours=span * case.replicas * config['gpu'] / 3600)
    readiness['native_limits'] = [text.replace('3x(16+128)', '1x(16+128)') for text in readiness['native_limits']]
    (case.target / 'evidence_readiness.json').write_text(dumps(readiness))
    return config, changed


def compare_wave(case, config):
    """Exact file and leaf comparison, with the immutable source as evidence."""
    import yaml
    old_files = source_files(case)
    new_files = {p.relative_to(case.target).as_posix():p for p in case.target.rglob('*') if p.is_file()}
    added = {f'tests/comparison/calibration/{name}' for name in WAVE_MODULES}
    if new_files.keys() != old_files.keys() | added:
        raise ValueError('Unexpected wave package file set')
    changed_modules = {f'tests/comparison/calibration/{name}' for name in WAVE_RESTAGED}
    config_allowed = {('engine','scheduler_cls'),('run_tag',),('job_tag',),('output_dir',),('harness_worktree',),
        ('retained_files','overlay.json')}
    if str(case.target/'overlay.json') in config['tool_hashes']:
        config_allowed.add(('tool_hashes',str(case.target/'overlay.json')))
    config_allowed |= {('tool_hashes',str(case.target/name)) for name in changed_modules}
    config_allowed |= {('harness_files',name) for name in added}
    trace_allowed = {('repetitions',name) for name in ('clean_repetitions','repeat_indices','request_id_binding','sequence')}
    manifest_allowed = {('case_id',),('run_generation',),('purpose',),('runtime_source_overlay','sha256'),
                        ('workload_sha256','manifest.json'),('workload_sha256','SHA256SUMS')}
    allowed = {**{name:config_allowed for name in CONFIGS}, **{name:{('scheduler_cls',)} for name in ENGINES},
        TRACE_MANIFEST:trace_allowed, 'trace/manifest.json':trace_allowed,
        'case/manifest.yaml':manifest_allowed, 'overlay.json':set()}
    report = {}
    for relative in sorted(new_files):
        if relative in added:
            report[relative] = ['added observer module']
            continue
        original = old_files[relative].read_bytes()
        actual = new_files[relative].read_bytes()
        if relative in changed_modules:
            name = relative
            if actual != (case.harness / name).read_bytes():
                raise ValueError(f'Observer module is not the harness bytes: {name}')
            report[relative] = ['restaged observer module']
        elif relative in (TRACE_SUMS,'trace/SHA256SUMS'):
            before, after = original.decode().splitlines(), actual.decode().splitlines()
            if len(before)!=len(after) or [line for line in before if not line.endswith('  manifest.json')] != [line for line in after if not line.endswith('  manifest.json')]:
                raise ValueError('Only the trace manifest hash may change in SHA256SUMS')
            report[relative] = ['manifest.json hash']
        elif relative in allowed:
            load = yaml.safe_load if relative.endswith('.yaml') else json.loads
            before, after = relocate(case,load(original)), load(actual)
            leaves = set(leaf_differences(before,after))
            if leaves != allowed[relative]:
                raise ValueError(f'{relative}: {sorted(leaves)} != {sorted(allowed[relative])}')
            report[relative] = sorted('.'.join(path) for path in leaves)
        elif relative in ('source_lock.json','evidence_readiness.json'):
            before, after = relocate(case,json.loads(original)), json.loads(actual)
            leaves = set(leaf_differences(before,after))
            if relative=='source_lock.json':
                permitted = {('harness_commit',),('qualification',),('runtime_overlay',)}
                permitted |= {('committed_files',name) for name in added}
                permitted |= {('committed_files',name,key) for name in before['committed_files'] for key in ('commit','sha256')}
                permitted |= {('retained_files','overlay.json',key) for key in ('sha256','staged_sha256','source','source_sha256','changed')}
            else:
                permitted = {('purpose',),('harness_commit',),('wave_observers',),('native_limits',),
                    ('scheduler_observer','scheduler_cls'),('scheduler_observer','module'),('scheduler_observer','sha256'),
                    ('cost','scheduled_arrival_span_seconds'),('cost','minimum_schedule_gpu_hours')}
            # runtime_overlay is a relocated nested document; validate separately.
            if relative=='source_lock.json':
                expected_overlay = json.loads((case.target/'overlay.json').read_text())
                if after['runtime_overlay'] != expected_overlay:
                    raise ValueError('Runtime overlay lock mismatch')
                leaves = {leaf for leaf in leaves if leaf[0]!='runtime_overlay'}
            if not leaves <= permitted:
                raise ValueError(f'Unexpected {relative} changes: {sorted(leaves-permitted)}')
            report[relative] = sorted('.'.join(path) for path in leaves)
        elif original != actual:
            raise ValueError(f'Unexpected wave byte change: {relative}')
    # Both trace copies and the workload contract are verified by the existing consumer.
    for name in ('manifest.json','SHA256SUMS'):
        if (case.target/'trace'/name).read_bytes() != (case.target/'case/inputs/trace'/name).read_bytes():
            raise ValueError('Trace copies differ')
    sys.path[:0] = [config['harness_dir'], str(case.target), str(case.target/'tests/comparison/dp_placement_pp')]
    from pd_replay import canonical_workload
    canonical_workload(case.target/'trace')
    if config['engine']['worker_extension_cls']!='native_full_structure.FullModelInventory':
        raise ValueError('Worker extension contract changed')
    return report


def preflight(case, config):
    """Run the launcher's run-manifest checks against a scratch copy of the case."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f'{case.name}_preflight_', dir=SCRATCH))
    shutil.copytree(case.target / 'case', scratch / 'case')
    trial = dict(config, case_dir=str(scratch / 'case'))
    (scratch / 'case/inputs/native_config.json').write_text(dumps(trial))
    from native_pd_worker import write_run_manifest
    manifest = json.loads(write_run_manifest(trial, 'preflight only; no submission').read_text())
    resources = manifest['resources']
    if (manifest['harness_commit'] != case.harness_commit or manifest['groundtruth_commit'] != NATIVE_COMMIT
            or manifest['harness_tree_dirty_paths'] or manifest['groundtruth_tree_dirty']
            or manifest['run_tag'] != case.run_tag or resources['gpus'] != case.replicas * config['gpu']
            or (resources['gpu_type'], resources['charged_group']) != (HARDWARE['gpu_type'], HARDWARE['charged_group'])):
        raise ValueError('Preflight run manifest does not bind the harness, NATIVE commit and H200 resources')
    return str(scratch)


def frontier_reconstruction(case, config):
    """Rebuild the Frontier P/D roles with R3's own check at the harness commit."""
    checker = runpy.run_path(str(R3_CHECKER), run_name='h200_preflight')
    observed = checker['verify_balanced_frontier'](dict(config, harness_commit=case.harness_commit), config['engine'])
    tp = config['engine']['tensor_parallel_size']
    expected = dict(device=HARDWARE['gpu_type'].lower(), attn_tensor_parallel_size=tp, attn_dp=case.dp,
                    moe_expert_parallel_size=tp * case.dp, num_pipeline_stages=1, world_size=tp * case.dp)
    for role in ROLES:
        if any(observed['roles'][role][name] != value for name, value in expected.items()):
            raise ValueError(f'Frontier {role} role differs from the case layout: {observed["roles"][role]}')
    return observed['roles']


def launch_path(case, config):
    """Plan every host and build its server command, as the worker does, for placeholder addresses."""
    sys.path[:0] = [str(case.target), config['harness_dir'], str(Path(config['harness_dir']).parent / 'dp_placement_pp')]
    from native_pd_worker import node_plan
    from pd_replay import kv_transfer_config
    from vllm_replay import server_command
    peers = [dict(host=f'h{i}', ip=f'10.0.0.{i + 1}') for i in range(case.replicas)]
    plans = {}
    for index, peer in enumerate(peers):
        plan = node_plan(config, peers, peer['host'], [str(i) for i in range(config['gpu'])])
        port = config['http_ports'][plan['role']]
        command = server_command(plan['engine'], Path(config['model_dir']), port) + [
            '--kv-transfer-config', json.dumps(kv_transfer_config(plan['engine'], plan['role'], port))]
        value = lambda flag: command[command.index(flag) + 1] if flag in command else None
        role, lane = ROLES[index // case.dp], index % case.dp
        row = dict(role=plan['role'], lane=plan['lane'], headless=plan['engine']['headless'],
                   data_parallel_size=value('--data-parallel-size'),
                   data_parallel_start_rank=value('--data-parallel-start-rank'),
                   data_parallel_address=value('--data-parallel-address'),
                   data_parallel_rpc_port=value('--data-parallel-rpc-port'),
                   profiler_config=json.loads(value('--profiler-config')) if value('--profiler-config') else None,
                   scheduler_cls=value('--scheduler-cls'),
                   worker_extension_cls=value('--worker-extension-cls'),
                   kv_transfer_config=json.loads(value('--kv-transfer-config')))
        trace_dir = f'{case.output_dir}/torch_traces/{role}/dp{lane}' if case.profile else None
        if (row['role'] != role or row['lane'] != lane or row['headless'] != (lane != 0)
                or (row['profiler_config'] or {}).get('torch_profiler_dir') != trace_dir
                or row['scheduler_cls'] != config['engine'].get('scheduler_cls')
                or row['worker_extension_cls'] != config['engine'].get('worker_extension_cls')
                or row['data_parallel_size'] != str(case.dp)
                or row['data_parallel_start_rank'] != (str(lane) if lane else None)
                or row['data_parallel_address'] != peers[case.dp * (index // case.dp)]['ip']):
            raise ValueError(f'Unexpected plan for host {index}: {row}')
        plans[peer['host']] = row
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=sorted(CASES))
    parser.add_argument('--package', help='New directory name; default package_02 for wave, package_01 otherwise')
    parser.add_argument('--harness', type=Path, help='Clean harness containing the retained source files plus the observer changes')
    parser.add_argument('--dry-check', type=Path, help='Wave/formal: scratch package comparison and offline launch plan; no sealing/preflight/commit required')
    arguments = parser.parse_args()
    package = arguments.package or ('package_02' if arguments.case == 'p2_64_wave' else 'package_01')
    case = Case(arguments.case, package)
    if case.wave or case.formal:
        if arguments.harness:
            case.harness = arguments.harness.resolve()
        case.harness_commit = git(case.harness, 'rev-parse', 'HEAD').strip()
        if arguments.dry_check:
            case.target = arguments.dry_check.resolve()
            config, changed = build(case)
            report = compare(case, config, changed)
            plans = launch_path(case, config)
            result = dict(status='DRY_CHECK_PASS', changes=report, launch_path=plans,
                          harness_prerequisites=(formal_harness_differences(case) if case.formal else wave_harness_differences(case)),
                          receipt_checks=('PASS' if case.formal else 'NOT_APPLICABLE'),
                          not_run=['seal', 'run_manifest_preflight', 'R3_reconstruction', 'launch_check_only'])
            print(json.dumps(result, indent=2))
            return
        differences = formal_harness_differences(case) if case.formal else wave_harness_differences(case)
        if differences:
            raise ValueError(f'Retained harness differs from the lifecycle package: {differences}; prepare a clean composed harness commit with the source drivers and inventory plus the worker bootstrap')
    elif arguments.harness or arguments.dry_check:
        raise ValueError('--harness and --dry-check apply only to p2_64_wave and p2_64_formal')
    if git(case.harness, 'status', '--short', '--untracked-files=all').strip():
        raise ValueError(f'The harness worktree {case.harness} is not clean')
    if git(case.harness, 'rev-parse', 'HEAD').strip() != case.harness_commit:
        raise ValueError(f'The harness worktree {case.harness} moved from {case.harness_commit}')
    if git(NATIVE, 'rev-parse', 'HEAD').strip() != NATIVE_COMMIT:
        raise ValueError('NATIVE checkout moved')
    sys.dont_write_bytecode = True
    # As launch.py: the profile worker imports vllm_replay at module level.
    sys.path[:0] = [str(case.target), str(case.target / 'tests/comparison/dp_placement_pp')]
    config, changed_retained = build(case)
    report = compare(case, config, changed_retained)
    sealed = json.loads(subprocess.run([sys.executable, str(case.target / 'seal.py')], check=True,
                                       capture_output=True, text=True).stdout)
    scratch = preflight(case, config)
    roles = frontier_reconstruction(case, config)
    plans = launch_path(case, config)
    check = subprocess.run([sys.executable, str(case.target / 'launch.py'), '--check-only'],
                           capture_output=True, text=True)
    if check.returncode or check.stdout.strip() != 'INPUT_HASHES_PASS_NO_SUBMISSION':
        raise ValueError(f'launch.py --check-only failed: {check.stdout} {check.stderr}')
    result = dict(case=case.name, source=str(case.source), harness=str(case.harness),
                  harness_commit=case.harness_commit, native_commit=NATIVE_COMMIT, package=str(case.target),
                  manifest_sha256=sha256(case.target / 'manifest.json'), sealed=sealed, changes=report,
                  changed_retained_files=changed_retained, preflight_scratch=scratch, frontier_roles=roles,
                  launch_path=plans, check_only=check.stdout.strip())
    (HERE / case.name / f'{case.target.name}_build.json').write_text(dumps(result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
