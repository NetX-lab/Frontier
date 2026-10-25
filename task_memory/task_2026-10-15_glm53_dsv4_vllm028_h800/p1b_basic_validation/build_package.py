"""Build package_01 for the P1b H800-64 basic native validation (design.md section 4).

Copies the native06 package, applies only the listed changes, compares every
file with native06, seals the package, and runs the checks of design section 5
that need no GPU: the run-manifest preflight on a scratch copy of the case, the
R3 Frontier reconstruction, the offline launch path for 8 hosts, and
``launch.py --check-only``:

    python3 build_package.py
"""

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
SOURCE = HERE.parent / 'v4_p0_native_20261021' / 'candidate_h800_p0_06'
TARGET = HERE / 'package_01'
# native06 ran from this directory; its configs carry it in every path field.
SOURCE_ROOT = ('/data/ycfeng/Frontier/.worktrees/v4-balanced-native-model/task_memory/'
               'task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_06')
HARNESS = Path('/data/ycfeng/Frontier/.worktrees/v4-balanced-native-model')
HARNESS_COMMIT = 'b62c155abd981a71e12e62aa0c0fe3990741d2f8'
NATIVE = Path('/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69')
NATIVE_COMMIT = '2cf0a6915ce544dc493a0990f2ea38d81601128a'
R3_CHECKER = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/'
                  'task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py')
SCRATCH = Path('/data/ycfeng/tmp/glm53_dsv4/p1b')
# P1 layout: one host per DP lane, so 2 x DP replicas.
DP = 4
REPLICAS = 2 * DP
RUN_TAG = 'dsv4-h800-p1-basic-20261025-01'
OUTPUT_DIR = ('/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h800_p1_basic_20261025/'
              f'case/runs/groundtruth_clean/{RUN_TAG}')
CASE_LINES = {
    'case_id: ': 'case_id: dsv4-h800-p1-basic-fixed128-qps1',
    'run_generation: ': 'run_generation: v4-native-v028-h800-p1-basic-20261025-01',
    'purpose: ': ('purpose: DIAGNOSTIC P1b basic_validation of the 64-GPU P32+D32 native P/D runtime; '
                  'native06 case and workload at DP4/EP32; not the P1 formal reference and not an E2E reference\n'
                  'diagnostic_only: true\n'
                  'e2e_reference_admission: false'),
}
# (old line, new line, occurrences) in case/manifest.yaml and launch.py.
TOPOLOGY_LINES = (('      dp: 2', f'      dp: {DP}', 2), ('      ep: 16', f'      ep: {8 * DP}', 2),
                  ('  priority: P0', '  priority: P1', 1))
LAUNCH_LINES = (
    ("assert (config['gpu_type'], config['charged_group'], config['gpu'], config['replica']) == "
     "('H800', 'codesign', 8, 4)",
     "assert (config['gpu_type'], config['charged_group'], config['gpu'], config['replica']) == "
     f"('H800', 'codesign', 8, {REPLICAS})", 1),
    ("assert config['engine']['data_parallel_size'] == 2",
     f"assert config['engine']['data_parallel_size'] == {DP}", 1),
)
FRONTIER_INPUT = 'case/inputs/frontier.json'
# Outputs of the native06 launch, not inputs.
LAUNCH_OUTPUTS = ('launcher_started.json', 'launcher_exit.json', 'manifest.json')
JSON_CHANGED = ('config.json', 'case/inputs/native_config.json', 'engine.json', 'case/inputs/engine.json',
                FRONTIER_INPUT, 'overlay.json', 'source_lock.json', 'evidence_readiness.json')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(repository, *args):
    return subprocess.run(['git', '--no-optional-locks', '-C', str(repository), *args],
                          check=True, capture_output=True, text=True).stdout


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


def source_files():
    files = {}
    for path in sorted(SOURCE.rglob('*')):
        relative = path.relative_to(SOURCE).as_posix()
        if path.is_file() and relative not in LAUNCH_OUTPUTS and not relative.startswith('case/runs/'):
            files[relative] = path
    return files


def build():
    TARGET.mkdir()
    for relative, path in source_files().items():
        (TARGET / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, TARGET / relative)

    old_config = json.loads((SOURCE / 'config.json').read_text())
    ep = old_config['engine']['tensor_parallel_size'] * DP
    frontier = json.loads((SOURCE / FRONTIER_INPUT).read_text())
    frontier.update({'replica_config_attn_dp': DP,
                     'cluster_config_prefill_replica_config_moe_expert_parallel_size': ep,
                     'cluster_config_decode_replica_config_moe_expert_parallel_size': ep})
    (TARGET / FRONTIER_INPUT).write_text(dumps(frontier))
    overlay = relocate(json.loads((SOURCE / 'overlay.json').read_text()))
    (TARGET / 'overlay.json').write_text(dumps(overlay))

    config = relocate(old_config)
    engine = dict(config['engine'], data_parallel_size=DP)
    config.update(engine=engine, replica=REPLICAS, run_tag=RUN_TAG, job_tag=RUN_TAG, output_dir=OUTPUT_DIR)
    for name in (FRONTIER_INPUT, 'overlay.json'):
        config['retained_files'][name] = sha256(TARGET / name)
    config['tool_hashes'] = {name: sha256(name) for name in config['tool_hashes']}
    for name in ('config.json', 'case/inputs/native_config.json'):
        (TARGET / name).write_text(dumps(config))
    for name in ('engine.json', 'case/inputs/engine.json'):
        (TARGET / name).write_text(dumps(engine))

    lock = relocate(json.loads((SOURCE / 'source_lock.json').read_text()))
    if lock['harness_commit'] != HARNESS_COMMIT or lock['native_commit'] != NATIVE_COMMIT:
        raise ValueError('native06 source lock names another harness or NATIVE commit')
    old_frontier = lock['retained_files'][FRONTIER_INPUT]
    lock['retained_files'][FRONTIER_INPUT] = dict(
        sha256=config['retained_files'][FRONTIER_INPUT], source=str(SOURCE / FRONTIER_INPUT),
        source_sha256=old_frontier['sha256'], changed=True)
    (TARGET / 'source_lock.json').write_text(dumps(lock))

    readiness = json.loads((SOURCE / 'evidence_readiness.json').read_text())
    gpus = config['replica'] * config['gpu']
    readiness['cost'].update(
        max_payload_gpu_hours=readiness['cost']['payload_guard_seconds'] * gpus / 3600,
        minimum_schedule_gpu_hours=readiness['cost']['scheduled_arrival_span_seconds'] * gpus / 3600)
    readiness.update(
        purpose='DIAGNOSTIC P1b basic_validation: 64-rank native P/D runtime evidence; client metrics enter no gate',
        target=f'H800/codesign{gpus} nonpreemptible; P32+D32 TP8/DP{DP}/EP{ep}/PP1')
    (TARGET / 'evidence_readiness.json').write_text(dumps(readiness))

    lines = []
    for line in (SOURCE / 'case/manifest.yaml').read_text().splitlines():
        prefix = next((key for key in CASE_LINES if line.startswith(key)), None)
        if prefix is not None:
            line = CASE_LINES[prefix]
        elif line.startswith('  sha256: ') and lines[-1] == '  manifest: overlay.json':
            line = '  sha256: ' + config['retained_files']['overlay.json']
        lines.append(line)
    (TARGET / 'case/manifest.yaml').write_text(replace_lines('\n'.join(lines), TOPOLOGY_LINES))
    (TARGET / 'launch.py').write_text(replace_lines((SOURCE / 'launch.py').read_text(), LAUNCH_LINES))
    return config


def compare(config):
    """Fail unless package_01 differs from native06 only by the listed changes."""
    import yaml
    old_files, new_files = source_files(), {
        path.relative_to(TARGET).as_posix(): path for path in TARGET.rglob('*') if path.is_file()}
    if new_files.keys() != old_files.keys():
        raise ValueError(f'Unexpected file set: {sorted(new_files.keys() ^ old_files.keys())}')
    config_paths = {('replica',), ('engine', 'data_parallel_size'), ('run_tag',), ('job_tag',), ('output_dir',),
                    ('retained_files', FRONTIER_INPUT), ('retained_files', 'overlay.json'),
                    ('tool_hashes', str(TARGET / FRONTIER_INPUT))}
    allowed = {'config.json': config_paths, 'case/inputs/native_config.json': config_paths,
               'engine.json': {('data_parallel_size',)}, 'case/inputs/engine.json': {('data_parallel_size',)},
               FRONTIER_INPUT: {('replica_config_attn_dp',),
                                ('cluster_config_prefill_replica_config_moe_expert_parallel_size',),
                                ('cluster_config_decode_replica_config_moe_expert_parallel_size',)},
               'overlay.json': set(),
               'source_lock.json': {('retained_files', FRONTIER_INPUT, key)
                                    for key in ('sha256', 'source', 'source_sha256', 'changed')},
               'evidence_readiness.json': {('purpose',), ('target',), ('cost', 'max_payload_gpu_hours'),
                                           ('cost', 'minimum_schedule_gpu_hours')}}
    report = {}
    for relative in sorted(old_files):
        old, new = old_files[relative].read_bytes(), new_files[relative].read_bytes()
        if relative in JSON_CHANGED:
            changed = set(leaf_differences(relocate(json.loads(old)), json.loads(new)))
            if changed != allowed[relative]:
                raise ValueError(f'{relative}: changes {sorted(changed)} differ from {sorted(allowed[relative])}')
            report[relative] = sorted('.'.join(path) for path in changed)
        elif relative == 'case/manifest.yaml':
            changed = set(leaf_differences(yaml.safe_load(old), yaml.safe_load(new)))
            expected = {('case_id',), ('run_generation',), ('purpose',), ('diagnostic_only',),
                        ('e2e_reference_admission',), ('runtime_source_overlay', 'sha256'), ('policy', 'priority'),
                        *(('topology', 'vllm', role, key) for role in ('prefill', 'decode') for key in ('dp', 'ep'))}
            if changed != expected:
                raise ValueError(f'case/manifest.yaml: changes {sorted(changed)}')
            report[relative] = sorted('.'.join(path) for path in changed)
        elif relative == 'launch.py':
            diff = [line for line in difflib.unified_diff(old.decode().splitlines(), new.decode().splitlines(),
                                                          lineterm='', n=0)
                    if line[:1] in '+-' and not line.startswith(('+++', '---'))]
            if sorted(diff) != sorted([f'-{old}' for old, _, _ in LAUNCH_LINES]
                                      + [f'+{new}' for _, new, _ in LAUNCH_LINES]):
                raise ValueError(f'launch.py: unexpected changes {diff}')
            report[relative] = diff
        elif old != new:
            raise ValueError(f'Unexpected byte change: {relative}')
    return report


def preflight(config):
    """Run the launcher's run-manifest checks against a scratch copy of the case."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix='package_01_preflight_', dir=SCRATCH))
    case = scratch / 'case'
    shutil.copytree(TARGET / 'case', case)
    trial = dict(config, case_dir=str(case))
    (case / 'inputs/native_config.json').write_text(dumps(trial))
    from native_pd_worker import write_run_manifest
    manifest = json.loads(write_run_manifest(trial, 'preflight only; no submission').read_text())
    if (manifest['harness_commit'] != HARNESS_COMMIT or manifest['groundtruth_commit'] != NATIVE_COMMIT
            or manifest['harness_tree_dirty_paths'] or manifest['groundtruth_tree_dirty']
            or manifest['run_tag'] != RUN_TAG or manifest['resources']['gpus'] != REPLICAS * config['gpu']):
        raise ValueError('Preflight run manifest does not bind the native06 harness, NATIVE commit and P1 resources')
    return str(scratch)


def frontier_reconstruction(config):
    """Rebuild the Frontier P/D roles with R3's own check at the harness commit."""
    checker = runpy.run_path(str(R3_CHECKER), run_name='p1b_preflight')
    observed = checker['verify_balanced_frontier'](dict(config, harness_commit=HARNESS_COMMIT), config['engine'])
    tp = config['engine']['tensor_parallel_size']
    expected = dict(attn_tensor_parallel_size=tp, attn_dp=DP, moe_expert_parallel_size=tp * DP,
                    num_pipeline_stages=1, world_size=tp * DP)
    for role in ('prefill', 'decode'):
        if any(observed['roles'][role][name] != value for name, value in expected.items()):
            raise ValueError(f'Frontier {role} role differs from the P1 layout: {observed["roles"][role]}')
    return observed['roles']


def launch_path(config):
    """Plan every host and build its server command, as the worker does, for placeholder addresses."""
    sys.path[:0] = [config['harness_dir'], str(Path(config['harness_dir']).parent / 'dp_placement_pp')]
    from native_pd_worker import node_plan
    from pd_replay import kv_transfer_config
    from vllm_replay import server_command
    peers = [dict(host=f'h{i}', ip=f'10.0.0.{i + 1}') for i in range(REPLICAS)]
    plans = {}
    for index, peer in enumerate(peers):
        plan = node_plan(config, peers, peer['host'], [str(i) for i in range(config['gpu'])])
        port = config['http_ports'][plan['role']]
        command = server_command(plan['engine'], Path(config['model_dir']), port) + [
            '--kv-transfer-config', json.dumps(kv_transfer_config(plan['engine'], plan['role'], port))]
        value = lambda flag: command[command.index(flag) + 1] if flag in command else None
        role, lane = ('prefill', 'decode')[index // DP], index % DP
        row = dict(role=plan['role'], lane=plan['lane'], headless=plan['engine']['headless'],
                   data_parallel_size=value('--data-parallel-size'),
                   data_parallel_start_rank=value('--data-parallel-start-rank'),
                   data_parallel_address=value('--data-parallel-address'),
                   data_parallel_rpc_port=value('--data-parallel-rpc-port'),
                   kv_transfer_config=json.loads(value('--kv-transfer-config')))
        if (row['role'] != role or row['lane'] != lane or row['headless'] != (lane != 0)
                or row['data_parallel_size'] != str(DP)
                or row['data_parallel_start_rank'] != (str(lane) if lane else None)
                or row['data_parallel_address'] != peers[DP * (index // DP)]['ip']):
            raise ValueError(f'Unexpected plan for host {index}: {row}')
        plans[peer['host']] = row
    return plans


def main():
    if git(HARNESS, 'status', '--short', '--untracked-files=all').strip():
        raise ValueError('The harness worktree is not clean')
    if git(HARNESS, 'rev-parse', 'HEAD').strip() != HARNESS_COMMIT:
        raise ValueError('The harness worktree moved from the native06 commit')
    if git(NATIVE, 'rev-parse', 'HEAD').strip() != NATIVE_COMMIT:
        raise ValueError('NATIVE checkout moved')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(TARGET))
    config = build()
    report = compare(config)
    sealed = json.loads(subprocess.run([sys.executable, str(TARGET / 'seal.py')], check=True,
                                       capture_output=True, text=True).stdout)
    scratch = preflight(config)
    roles = frontier_reconstruction(config)
    plans = launch_path(config)
    check = subprocess.run([sys.executable, str(TARGET / 'launch.py'), '--check-only'],
                           capture_output=True, text=True)
    if check.returncode or check.stdout.strip() != 'INPUT_HASHES_PASS_NO_SUBMISSION':
        raise ValueError(f'launch.py --check-only failed: {check.stdout} {check.stderr}')
    result = dict(harness_commit=HARNESS_COMMIT, native_commit=NATIVE_COMMIT, package=str(TARGET),
                  manifest_sha256=sha256(TARGET / 'manifest.json'), sealed=sealed, changes=report,
                  preflight_scratch=scratch, frontier_roles=roles, launch_path=plans,
                  check_only=check.stdout.strip())
    (HERE / 'package_01_build.json').write_text(dumps(result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
