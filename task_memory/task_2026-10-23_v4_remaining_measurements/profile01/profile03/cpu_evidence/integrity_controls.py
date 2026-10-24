"""Read-only PROFILE03 source, input closure, resource and historical binding checks."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = Path('/data/ycfeng/Frontier/.worktrees/v4-native-profile03-run')
TASK = REPO / 'task_memory/task_2026-10-23_v4_remaining_measurements'
RECORDS = TASK / 'profile01/profile03'
PACKAGE = REPO / 'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_profile_03'
PRIMARY = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028')
RUNTIME = '1437ace6f4a438c1bb879d904e2bede0c777303c'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args])


def file_map(path):
    return {str(p.relative_to(path)): sha(p) for p in path.rglob('*') if p.is_file()}


config = json.loads((PACKAGE / 'config.json').read_text())
lock = json.loads((PACKAGE / 'source_lock.json').read_text())
plan = json.loads((RECORDS / 'profile_plan.json').read_text())
freeze = json.loads((RECORDS / 'input_freeze.json').read_text())
prior = json.loads((TASK / 'profile01/delivery.json').read_text())
old_package = Path(prior['package'])
old_config = json.loads((old_package / 'config.json').read_text())
overlay = json.loads((PACKAGE / 'overlay.json').read_text())
assert overlay['binary_sha256'] == json.loads((old_package / 'overlay.json').read_text())['binary_sha256']
assert len(overlay['binary_sha256']) == 20
assert config['expected_source_hashes'] == old_config['expected_source_hashes']
assert len(config['expected_source_hashes']) == 87
assert config['model_hashes'] == old_config['model_hashes']
assert config['engine'] == old_config['engine'] | {'profiler_config': config['engine']['profiler_config']}
assert {k: v for k, v in config['engine']['profiler_config'].items() if k != 'torch_profiler_dir'} == {
    k: v for k, v in old_config['engine']['profiler_config'].items() if k != 'torch_profiler_dir'}
changed = []
for name, target in config['harness_files'].items():
    row = freeze['runtime_modules'][name]
    content = (PACKAGE / target).read_bytes()
    assert row['commit'] == RUNTIME and row['sha256'] == sha(PACKAGE / target)
    assert content == git('show', f'{RUNTIME}:{name}') == git('show', f"{lock['harness_commit']}:{name}")
    if sha(PACKAGE / target) != sha(old_package / target):
        changed.append(name)
assert len(config['harness_files']) == 13
assert set(changed) == {f'tests/comparison/calibration/{name}.py' for name in (
    'native_pd_worker', 'native_full_structure', 'native_serve_profile')}
primary_receipt = PRIMARY / 'task_memory/task_2026-10-23_v4_remaining_measurements/profile01/integration_cpu/primary_acceptance.json'
assert (RECORDS / 'primary_integration_acceptance.json').read_bytes() == primary_receipt.read_bytes()
acceptance = json.loads(primary_receipt.read_text())
assert acceptance['status'] == 'ACCEPTED_CPU_DIAGNOSTIC_COLLECTION_REPAIRS'
assert acceptance['source_revision'] == RUNTIME
assert (acceptance['passed'], acceptance['unchanged_baseline_failures'], acceptance['counts']['skipped']) == (465, 1, 0)
assert set(acceptance['independent_reviews'].values()) == {'NO_FINDINGS'}
for name in changed:
    assert acceptance['source_hashes'][name] == freeze['runtime_modules'][name]['sha256']
resources = dict(gpu=8, replica=4, cpu=64, mem_gb=512, gpu_type='H800', charged_group='codesign',
                 timeout_seconds=3600, startup_timeout_s=1800, replay_timeout_s=1800,
                 peer_wait_seconds=1200, stop_timeout_s=90, data_parallel_size_local=1)
for name, value in resources.items():
    assert config[name] == value, (name, config.get(name))
assert config['cloud_mount'] == old_config['cloud_mount'] == 'juicefs+s3://oss.i.shaipower.com/codesign-exp:/mnt/codesign-exp'
assert config['role_profiler_max_iterations'] == {'prefill': 512, 'decode': 64}
assert config['serving_profile_limits'] == {'records': 600000, 'bytes': 268435456}
assert config['run_tag'] == plan['run_tag'] == 'dsv4-h800-p0-profile-20261024-03'
assert config['output_dir'] == plan['cloud_root'] + '/case/runs/groundtruth_clean/' + plan['run_tag']
assert plan['run_generation'] in (PACKAGE / 'case/manifest.yaml').read_text()
assert 'auxiliary_validators_soft: true' in (PACKAGE / 'case/manifest.yaml').read_text()
for name in ('config.json', 'profile_plan.json', 'source_lock.json', 'evidence_readiness.json'):
    value = json.loads((PACKAGE / name).read_text())
    assert value['diagnostic_only'] and not value['e2e_reference_admission']
launcher = (PACKAGE / 'submit_native_diagnostic.py').read_text()
assert launcher == (old_package / 'submit_native_diagnostic.py').read_text()
assert 'mem_gb=config["mem_gb"]' in launcher and 'mounts=[config["cloud_mount"]]' in launcher
assert 'code_mount_point=config["workspace"]' in launcher
assert 'preemptible: false' in (PACKAGE / 'case/manifest.yaml').read_text()
# Evaluate only the existing seal's path-list expression. Do not execute its write or invoke seal.py.
tree = ast.parse((PACKAGE / 'seal.py').read_text())
expression = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'files' for t in n.targets))
closure_paths = eval(compile(ast.Expression(expression), 'existing_seal_input_list', 'eval'), {'HERE': PACKAGE})
closure = {str(p.relative_to(PACKAGE)): sha(p) for p in closure_paths}
all_files = file_map(PACKAGE)
assert closure == all_files and len(closure) == 45
payload = dict(files=closure, submitted=False, native_qualification=False)
manifest_bytes = (json.dumps(payload, indent=2, sort_keys=True) + '\n').encode()
(ROOT / 'prospective_manifest.json').write_bytes(manifest_bytes)
matrix = json.loads((RECORDS / 'matrix_binding.json').read_text())
assert matrix['source_revision'] == RUNTIME and matrix['identical_cases'] == 77
assert matrix['mismatched_cases'] == 0 and not matrix['new_capture']
assert sha(Path(matrix['comparison_path'])) == matrix['comparison_sha256']
for name, expected in matrix['unchanged_domain_trees'].items():
    assert git('rev-parse', f'{RUNTIME}:{name}').decode().strip() == expected
    assert git('rev-parse', f"{matrix['capture_commit']}:{name}").decode().strip() == expected
inventory = json.loads((RECORDS / 'prior_hash_inventory.json').read_text())
for name, expected in inventory['original_files'].items():
    assert sha(Path(name)) == expected, name
for name, expected in inventory['preserved_primary_records'].items():
    assert sha(Path(name)) == expected, name
for name, row in inventory['prior_packages'].items():
    package = Path(name)
    for relative, expected in row['inputs'].items():
        assert sha(package / relative) == expected
    if row['sealed']:
        assert sha(package / 'manifest.json') == row['seal_manifest_sha256']
        assert json.loads((package / 'manifest.json').read_text())['files'] == row['inputs']
builder = 'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/build_candidate.py'


class Native06(ast.NodeTransformer):
    def visit_If(self, node):
        test = ast.unparse(node.test)
        if test == 'plan':
            return [self.visit(x) for x in node.orelse]
        if test == 'plan is None':
            return [self.visit(x) for x in node.body]
        return self.generic_visit(node)


assert ast.dump(Native06().visit(ast.parse(git('show', f'{RUNTIME}:{builder}').decode()))) == ast.dump(
    Native06().visit(ast.parse((REPO / builder).read_text())))
assert not (PACKAGE / 'manifest.json').exists() and not (PACKAGE / 'case/runs').exists()
assert not list(PACKAGE.rglob('__pycache__'))
receipt = dict(status='PASS_CPU_PROFILE03_CLOSURE_AND_BINDINGS', runtime_modules=13,
               changed_runtime_modules=changed, unchanged_runtime_modules=10, native_pins=87,
               retained_binary_pins=20, actual_image_binary_verification='ROOT_PENDING',
               integration_counts={'passed': 465, 'known_baseline_failures': 1, 'skipped': 0},
               acceptance_sha256=sha(primary_receipt), resources=resources,
               package_files=len(all_files), package_bytes=sum(p.stat().st_size for p in closure_paths),
               package_hashes=all_files, prospective_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
               sealed=False, submitted=False, manifest_closure='EXACT_45_FILES_USING_EXISTING_SEAL_LIST_EXPRESSION',
               matrix_cases=77, domain_trees=5, native06_ast_equal=True,
               prior_original_files=123, prior_checkers=3, prior_profile_packages=2,
               runtime_qualification=False, cloud_access=False, python=sys.version)
(ROOT / 'integrity.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k: v for k, v in receipt.items() if k != 'package_hashes'}))
