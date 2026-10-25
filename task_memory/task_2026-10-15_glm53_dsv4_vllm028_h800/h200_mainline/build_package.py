"""Build package_01 of one H200 mainline case (design.md sections 4 and 5).

Copies the case's H800 source package, applies only the listed changes,
compares every file with the source, seals the package, and runs the checks
that need no GPU: the run-manifest preflight on a scratch copy of the case, the
R3 Frontier reconstruction, the offline launch path, and ``launch.py --check-only``:

    python3 build_package.py p2_32_reference [--package package_01]

The package directory must not exist yet.
"""

import argparse
import difflib
import hashlib
import json
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGES = HERE.parent / 'v4_p0_native_20261021'
WORKTREES = Path('/data/ycfeng/Frontier/.worktrees')
NATIVE = Path('/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69')
NATIVE_COMMIT = '2cf0a6915ce544dc493a0990f2ea38d81601128a'
R3_CHECKER = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/'
                  'task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py')
SCRATCH = Path('/data/ycfeng/tmp/glm53_dsv4/h200')
CLOUD = '/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18'
HARDWARE = dict(gpu_type='H200', charged_group='step_main')
PACKAGE_DIR = 'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/'
# The source packages ran from their harness worktrees; their configs carry
# these directories in every path field.
SOURCES = {
    'reference': dict(package='candidate_h800_p0_06', harness='v4-balanced-native-model',
                      harness_commit='b62c155abd981a71e12e62aa0c0fe3990741d2f8'),
    'profile': dict(package='candidate_h800_p0_profile_02', harness='v4-native-profile-run',
                    harness_commit='a90c9dbbee53e2cabbfac73ba8490ffb958ce8eb'),
}
CASES = {
    'p2_32_reference': dict(source='reference', dp=2, case_id='dsv4-h200-p2-canonical-fixed128-qps1',
                            run_tag='dsv4-h200-p2-canonical-20261025-01',
                            purpose='P2 H200-32 native reference for DSV4 under D-H200-TEMP-MAINLINE; '
                                    'Frontier comparisons against it are DIAGNOSTIC'),
    'p2_32_profile': dict(source='profile', dp=2, case_id='dsv4-h200-p2-profile-fixed128-qps1',
                          run_tag='dsv4-h200-p2-profile-20261025-01',
                          purpose='Diagnostic native serving profile on H200-32; source of the H200 step-cost '
                                  'table; no E2E or reference admission'),
    'p2_64_reference': dict(source='reference', dp=4, case_id='dsv4-h200-p2-64-canonical-fixed128-qps1',
                            run_tag='dsv4-h200-p2-64-canonical-20261025-01',
                            purpose='P2 H200-64 native reference and basic validation for DSV4 under '
                                    'D-H200-TEMP-MAINLINE; Frontier comparisons against it are DIAGNOSTIC'),
    'p2_64_profile': dict(source='profile', dp=4, case_id='dsv4-h200-p2-64-profile-fixed128-qps1',
                          run_tag='dsv4-h200-p2-64-profile-20261025-01',
                          purpose='Diagnostic native serving profile on H200-64; source of the H200-64 '
                                  'step-cost table; no E2E or reference admission'),
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
        self.name, self.profile, self.dp = name, case['source'] == 'profile', case['dp']
        self.replicas = 2 * self.dp
        self.case_id, self.run_tag, self.purpose = case['case_id'], case['run_tag'], case['purpose']
        self.source = PACKAGES / source['package']
        self.harness = WORKTREES / source['harness']
        self.harness_commit = source['harness_commit']
        self.source_root = str(self.harness / PACKAGE_DIR / source['package'])
        self.target = HERE / name / package
        self.output_dir = f"{CLOUD}/{self.run_tag.replace('-', '_')}/case/runs/groundtruth_clean/{self.run_tag}"
        self.launch_lines = [
            ("assert (config['gpu_type'], config['charged_group'], config['gpu'], config['replica']) == "
             "('H800', 'codesign', 8, 4)",
             "assert (config['gpu_type'], config['charged_group'], config['gpu'], config['replica']) == "
             f"('{HARDWARE['gpu_type']}', '{HARDWARE['charged_group']}', 8, {self.replicas})", 1)]
        if self.dp != 2:
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
    if '  H200_nonpreemptible_running_plus_queued_limit: 32' in lines:
        replacements.append(('  H200_nonpreemptible_running_plus_queued_limit: 32',
                             '  H200_nonpreemptible_running_plus_queued_limit: 64', 1))
    if case.dp != 2:
        replacements += [('      dp: 2', f'      dp: {case.dp}', 2), ('      ep: 16', f'      ep: {8 * case.dp}', 2)]
    return replacements


def build(case):
    case.target.mkdir(parents=True)
    for relative, path in source_files(case).items():
        (case.target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, case.target / relative)
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
            if 'H200_nonpreemptible_running_plus_queued_limit' in yaml.safe_load(old)['policy']:
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
    sys.path[:0] = [config['harness_dir'], str(Path(config['harness_dir']).parent / 'dp_placement_pp')]
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
                   kv_transfer_config=json.loads(value('--kv-transfer-config')))
        trace_dir = f'{case.output_dir}/torch_traces/{role}/dp{lane}' if case.profile else None
        if (row['role'] != role or row['lane'] != lane or row['headless'] != (lane != 0)
                or (row['profiler_config'] or {}).get('torch_profiler_dir') != trace_dir
                or row['data_parallel_size'] != str(case.dp)
                or row['data_parallel_start_rank'] != (str(lane) if lane else None)
                or row['data_parallel_address'] != peers[case.dp * (index // case.dp)]['ip']):
            raise ValueError(f'Unexpected plan for host {index}: {row}')
        plans[peer['host']] = row
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=sorted(CASES))
    parser.add_argument('--package', default='package_01')
    arguments = parser.parse_args()
    case = Case(arguments.case, arguments.package)
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
