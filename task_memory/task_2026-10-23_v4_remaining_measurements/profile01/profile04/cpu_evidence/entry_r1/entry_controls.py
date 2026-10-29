"""Disposable CPU-only entry controls; never invoke a platform or native GPU run."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path('/data/ycfeng/Frontier/.worktrees/v4-native-profile04-run')
ROOT = Path(__file__).resolve().parent
TASK = REPO / 'task_memory/task_2026-10-23_v4_remaining_measurements/profile01'
INITIAL = json.loads((ROOT / 'prior_reference.json').read_text())
OLD = Path(INITIAL['package'])
NEW = Path('/data/ycfeng/Frontier/.worktrees/v4-native-profile04-run/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_profile_04')
PYTHON = '/data/ycfeng/envs/frontier-py310/bin/python'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(path):
    return {str(p.relative_to(path)): sha(p) for p in path.rglob('*') if p.is_file()}


def run(name, command, *, env):
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    (ROOT / (name + '.stdout.log')).write_text(result.stdout)
    (ROOT / (name + '.stderr.log')).write_text(result.stderr)
    (ROOT / (name + '.receipt.json')).write_text(json.dumps(dict(
        command=command, returncode=result.returncode, cpu_only=True), indent=2) + '\n')
    return result


before = {str(p): files(p) for p in (OLD, NEW)}
assert before[str(OLD)] == INITIAL['package_hashes']
env = {k: v for k, v in os.environ.items() if k not in (
    'PYTHONPATH', 'PYTHONHOME', 'AUTHORIZATION', 'BRAINPP_ACCESS_KEY', 'BRAINPP_SECRET_KEY')}
env.update(PYTHONDONTWRITEBYTECODE='1', STEPMIND_BACKEND='rjob',
           AUTHORIZATION='CPU_ONLY_SENTINEL_NEVER_SUBMIT')

for source, name in ((NEW, 'new_launcher_green'),):
    copied = ROOT / name
    shutil.copytree(source, copied)
    # Test-only input manifest on a disposable copy; neither real candidate is sealed.
    (copied / 'manifest.json').write_text(json.dumps(dict(
        files=before[str(source)], test_only=True, submitted=False)))
    if source == OLD:
        result = run(name, [PYTHON, '-B', str(copied / 'launch.py')], env=env)
        assert result.returncode == 1
        assert "ModuleNotFoundError: No module named 'vllm_replay'" in result.stderr
    else:
        code = '''import json,pathlib,runpy,sys
launch=pathlib.Path(sys.argv[1]); receipt=pathlib.Path(sys.argv[2]); root=launch.parent
def trace(frame,event,arg):
    if event == 'call' and frame.f_code.co_name == 'write_run_manifest':
        assert pathlib.Path(frame.f_code.co_filename) == root/'native_pd_worker.py'
        assert 'pd_replay' not in sys.modules, 'Forbidden preload'
        assert pathlib.Path(sys.modules['vllm_replay'].__file__) == root/'tests/comparison/dp_placement_pp/vllm_replay.py'
        receipt.write_text(json.dumps(dict(status='GREEN_BEFORE_WRITE_RUN_MANIFEST_BODY', worker=frame.f_code.co_filename, dependency=sys.modules['vllm_replay'].__file__, sys_path=sys.path, pd_replay_preloaded=False, run_files_written=False, submitted=False),indent=2)+'\\n')
        print('FRESH_LAUNCHER_GREEN_BEFORE_WRITE_RUN_MANIFEST_BODY')
        raise SystemExit(0)
    return trace
sys.argv=[str(launch)]; sys.settrace(trace); runpy.run_path(str(launch),run_name='__main__')
raise AssertionError('Sentinel not reached')'''
        result = run(name, [PYTHON, '-I', '-B', '-c', code, str(copied / 'launch.py'),
                           str(ROOT / 'new_launcher_sentinel.json')], env=env)
        assert result.returncode == 0, result.stderr
        assert 'FRESH_LAUNCHER_GREEN_BEFORE_WRITE_RUN_MANIFEST_BODY' in result.stdout
        checked = run('launch_check_only', [PYTHON, '-I', '-B', str(copied / 'launch.py'), '--check-only'], env=env)
        assert checked.returncode == 0 and 'INPUT_HASHES_PASS_NO_SUBMISSION' in checked.stdout
    assert not (copied / 'launcher_started.json').exists()
    assert not (copied / 'launcher_exit.json').exists()
    assert not (copied / 'case/runs').exists()

# Execute the actual generated bootstrap through its source-verification/exec seam.
# Fake installed vLLM source and binary files are explicitly verification stand-ins.
# They do not establish validity of the locked image or a native runtime.
bootstrap = ROOT / 'bootstrap_standins'
bootstrap.mkdir()
standin = bootstrap / 'installed/vllm'
(standin / 'models/deepseek_v4/nvidia').mkdir(parents=True)
(standin / '__init__.py').write_text('"""CPU source-verification stand-in, not native vLLM."""\n')
base = standin / 'models/deepseek_v4/nvidia/model.py'
base.write_bytes(b'# CPU-only base source-verification stand-in\n')
binary_hashes = {}
for name in ('_C_stable_libtorch.abi3.so', '_moe_C_stable_libtorch.abi3.so'):
    path = standin / name
    path.write_bytes(b'CPU-only binary hash-verification stand-in; not a shared library\n')
    binary_hashes[str(path)] = sha(path)
stage = bootstrap / 'stage'
shutil.copytree(NEW, stage)
config = json.loads((NEW / 'config.json').read_text())
overlay = json.loads((NEW / 'overlay.json').read_text())
overlay.update(installed_package=str(standin), staged_dir=str(stage), binary_sha256=binary_hashes)
leaf = overlay['files']['models/deepseek_v4/nvidia/model.py']
leaf.update(source=str(stage / 'overlay_model.py'), base_sha256=sha(base))
(stage / 'overlay.json').write_text(json.dumps(overlay))
config.update(script=str(stage / 'native_pd_worker.py'), overlay_manifest=str(stage / 'overlay.json'))
config['tool_hashes'] = {str(stage / Path(name).relative_to(NEW)): sha(stage / Path(name).relative_to(NEW))
                         for name in config['tool_hashes']}
(stage / 'config.json').write_text(json.dumps(config))
bootstrap_receipt = ROOT / 'bootstrap_environment.json'
code = '''import importlib.util,json,pathlib,subprocess,sys
stage=pathlib.Path(sys.argv[1]); package=pathlib.Path(sys.argv[2]); receipt=pathlib.Path(sys.argv[3])
s=importlib.util.spec_from_file_location('generated_bootstrap',stage/'native_dummy_bootstrap.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
class InstalledSourceVerificationStandIn:
    @staticmethod
    def find_spec(name,paths):
        assert name=='vllm'; return type('Spec',(),{'origin':str(package/'__init__.py')})()
m.PathFinder=InstalledSourceVerificationStandIn
def launch(executable,args,env):
    assert 'pd_replay' not in sys.modules and 'native_pd_worker' not in sys.modules
    assert env['PYTHONPATH']==str(stage)+':'+str(stage/'tests/comparison/dp_placement_pp')
    assert args==[sys.executable,'-B',str(stage/'native_pd_worker.py'),'--config',str(stage/'config.json')]
    child_code="import json,pathlib,sys; import native_pd_worker; assert 'pd_replay' not in sys.modules; assert pathlib.Path(sys.modules['vllm_replay'].__file__)==pathlib.Path('tests/comparison/dp_placement_pp/vllm_replay.py').resolve(); print('BOOTSTRAP_CHILD_IMPORT_GREEN'); print(json.dumps(dict(worker=native_pd_worker.__file__,dependency=sys.modules['vllm_replay'].__file__,sys_path=sys.path,pd_replay_preloaded=False)))"
    child=subprocess.run([executable,'-B','-c',child_code],env=env,text=True,capture_output=True)
    receipt.write_text(json.dumps(dict(status='GREEN' if child.returncode==0 else 'FAIL',standins='installed source/binary/PathFinder only; generated bootstrap and13runtime bytes unchanged',generated_argv=args,generated_pythonpath=env['PYTHONPATH'],child_command=[executable,'-B','-c',child_code],returncode=child.returncode,stdout=child.stdout,stderr=child.stderr,submitted=False),indent=2)+'\\n')
    assert child.returncode==0,child.stderr
    print(child.stdout,end='')
m.bootstrap(stage/'config.json',package=package,launch=launch)
'''
result = run('bootstrap_green', [PYTHON, '-I', '-B', '-c', code, str(stage), str(standin),
                                 str(bootstrap_receipt)], env=env)
assert result.returncode == 0, result.stderr
assert 'BOOTSTRAP_CHILD_IMPORT_GREEN' in result.stdout
assert sha(base) == leaf['sha256']
for name, expected in binary_hashes.items():
    assert sha(Path(name)) == expected
for path in (OLD, NEW):
    assert files(path) == before[str(path)]
    if path == NEW:
        assert not (path / 'manifest.json').exists()
    if path == NEW:
        assert not (path / 'case/runs').exists()
    assert not list(path.rglob('__pycache__'))
print(json.dumps(dict(old_launcher='NOT_RERUN_IMMUTABLE', new_launcher='GREEN_traced_pre_body',
                      bootstrap_child='GREEN_generated_env', real_candidate_bytes_unchanged=True)))
