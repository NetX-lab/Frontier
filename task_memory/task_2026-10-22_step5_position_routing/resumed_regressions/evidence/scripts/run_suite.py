"""Execute the five delegated suites without writing to frozen source."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path('/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05')
EXPECTED = 'e41abc073500d61f224eda4fca1123e74e40544d'
PY310 = '/data/ycfeng/envs/frontier-py310/bin/python'
PY312 = '/data/ycfeng/envs/asyncopd-py312/bin/python'
BASELINE = '/data/ycfeng/tmp/d30_workflow_fidelity_20261022/candidate'


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def state():
    sub = REPO / 'frontier/cc_backend/backends/collective-sim'
    return {
        'head': git('rev-parse', 'HEAD'),
        'status': git('status', '--porcelain=v1', '--untracked-files=all', '--ignore-submodules=none'),
        'submodule_head': subprocess.check_output(['git', '-C', str(sub), 'rev-parse', 'HEAD'], text=True).strip(),
        'submodule_status': subprocess.check_output(['git', '-C', str(sub), 'status', '--porcelain=v1', '--untracked-files=all'], text=True).strip(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', choices=['strict77', 'unit', 'supplement101', 'metadata196', 'catalogue53'])
    suite = parser.parse_args().suite
    dest = ROOT / suite
    dest.mkdir(exist_ok=False)
    view = dest / 'generated_repo_root'
    view.mkdir()
    before = state()
    assert before['head'] == EXPECTED and not before['status'] and not before['submodule_status'], before
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR='/data/ycfeng/tmp', FRONTIER_TMP_ROOT='/data/ycfeng/tmp',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONPATH=str(REPO) if suite == 'strict77' else f'{REPO}:{REPO}/tests/unit')
    env.pop('FRONTIER_TEST_CPU_WRITER_FILE', None)
    env.pop('FRONTIER_TEST_CPU_WRITER_SHA256', None)
    py = PY312 if suite == 'supplement101' else PY310
    if suite == 'supplement101':
        env.update(STEP5_NATIVE_SOURCE_ROOT='/data/ycfeng/tmp/d29_published_sources01/source/vllm', VLLM_TARGET_DEVICE='cpu')
    mount = ['bwrap', '--die-with-parent', '--ro-bind', '/', '/', '--dev-bind', '/dev', '/dev',
             '--bind', '/data/ycfeng/tmp', '/data/ycfeng/tmp', '--bind', str(view), str(REPO)]
    for entry in sorted(REPO.iterdir()):
        mount.extend(['--ro-bind', str(entry), str(entry)])
    mount.extend(['--chdir', str(REPO), '--'])
    version_code = "import json,platform,sys,importlib.metadata as m;pkgs=['pytest','numpy','pandas','scipy','scikit-learn','plotly','fasteners','ddsketch','torch','aiohttp'];out={'python':platform.python_version(),'executable':sys.executable};\nfor p in pkgs:\n try: out[p]=m.version(p)\n except m.PackageNotFoundError: out[p]=None\nprint(json.dumps(out,sort_keys=True))"
    versions = json.loads(subprocess.check_output([py, '-B', '-c', version_code], env=env, text=True))
    if suite == 'strict77':
        output = ROOT / 'strict77_final04'
        output.mkdir(exist_ok=False)
        commands = [[py, '-B', 'tests/e2e/refactor_fidelity/run_matrix.py', 'run', '--repo-root', str(REPO),
                     '--python-bin', PY310, '--label', 'candidate', '--output-root', str(output),
                     '--jobs', '4', '--clean-cache', '--continue-on-failure'],
                    [py, '-B', 'tests/e2e/refactor_fidelity/run_matrix.py', 'compare', '--output-root', str(output),
                     '--baseline-label', BASELINE, '--candidate-label', str(output / 'candidate')]]
    else:
        tests = {
            'unit': ['tests/unit'],
            'supplement101': ['tests/unit/test_step5_canonical_g3.py', 'tests/unit/test_step5_native_position_cyclic.py'],
            'metadata196': ['tests/unit/test_comm_operator_families.py', 'tests/unit/test_ep_wave_trace_context.py',
                            'tests/unit/test_quantization_manager_registry.py', 'tests/unit/test_quantization_module_precision.py',
                            'tests/unit/test_op_trace_utils.py', 'tests/unit/test_ep_trace.py', 'tests/unit/test_collective_sim_cc_backend.py'],
            'catalogue53': ['tests/unit/test_moe_share_expert_operator_families.py', 'tests/unit/test_dsv4_precision_metadata.py'],
        }[suite]
        seconds = {'unit':1800, 'supplement101':180, 'metadata196':180, 'catalogue53':120}[suite]
        commands = [['timeout', f'{seconds}s', py, '-B', '-m', 'pytest', *tests, '-q', '-p', 'no:cacheprovider']]
        if suite == 'unit':
            commands[0].append('--continue-on-collection-errors')
        commands[0].extend([f'--basetemp={dest}/basetemp', f'--junitxml={dest}/results.xml'])
    evidence = {'task_id':'D29-E41-FINAL-REGRESSIONS', 'suite':suite, 'source_before':before,
                'versions':versions, 'environment':{k:env[k] for k in sorted(env) if k.startswith(('PYTHON','FRONTIER','STEP5','VLLM','OMP','OPENBLAS','MKL','TMP'))},
                'source_entries_readonly':[p.name for p in sorted(REPO.iterdir())],
                'generated_file_root':str(view), 'executions':[]}
    (dest / 'launch.json').write_text(json.dumps({**evidence,'commands':[mount+c for c in commands]}, indent=2)+'\n')
    for n, command in enumerate(commands):
        full = mount + command
        log = dest / f'command{n+1}.log'
        started = time.monotonic()
        with log.open('w') as stream:
            result = subprocess.run(full, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
        evidence['executions'].append({'command':full, 'returncode':result.returncode,
                                      'duration_seconds':time.monotonic()-started, 'log':str(log)})
        if suite == 'strict77' and n == 0 and not (ROOT/'strict77_final04/candidate/manifest.json').is_file():
            break
    evidence['source_after'] = state()
    (dest / 'execution.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps(evidence, indent=2))
    return max([x['returncode'] for x in evidence['executions']] + [0])


if __name__ == '__main__':
    raise SystemExit(main())
