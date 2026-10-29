"""Rebind the retained regression runner to the approved post-key frozen source."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path('/data/ycfeng/Frontier/.worktrees/d29-ledger-final-1f6b4314')
HEAD = '1461573de388b72450189bc0c1b88b4ac32491f0'
TREE = '447a285aa403af747f838a3bf404b2b17e2eacf7'
SUB = '3dac53c544e546574cb6e8078cd14053d0916113'
PY310 = '/data/ycfeng/envs/frontier-py310/bin/python'
PY312 = '/data/ycfeng/envs/asyncopd-py312/bin/python'
STRICT = Path('/data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/regressions/strict77_output')
BASELINE = '/data/ycfeng/tmp/d29_native_cache_key_20261024.xBHOmb/frozen-matrix/candidate'
NATIVE = Path('frontier/cc_backend/backends/collective-sim/sim/datacenter')
RETAINED_BINARY = Path('/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-06') / NATIVE / 'htsim_ndp'


def state():
    def git(path, *args):
        return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()
    sub = REPO / 'frontier/cc_backend/backends/collective-sim'
    return {'head':git(REPO,'rev-parse','HEAD'), 'tree':git(REPO,'rev-parse','HEAD^{tree}'),
            'status':git(REPO,'status','--porcelain=v1','--untracked-files=all','--ignore-submodules=none'),
            'submodule_head':git(sub,'rev-parse','HEAD'),
            'submodule_status':git(sub,'status','--porcelain=v1','--untracked-files=all')}


def historical_projected_namespace(view, env, readonly=False):
    command = ['bwrap','--die-with-parent','--ro-bind','/','/','--dev-bind','/dev','/dev',
               '--bind','/data/ycfeng/tmp','/data/ycfeng/tmp',
               '--ro-bind' if readonly else '--bind',str(view),str(REPO)]
    for entry in sorted(REPO.iterdir()):
        command.extend(['--ro-bind',str(entry),str(entry)])
    native_view = view.parent/'native_datacenter'
    native_view.mkdir(exist_ok=True)
    command.extend(['--bind',str(native_view),str(REPO/NATIVE)])
    for entry in sorted((REPO/NATIVE).iterdir()):
        command.extend(['--ro-bind',str(entry),str(entry)])
    command.extend(['--ro-bind',str(RETAINED_BINARY),str(REPO/NATIVE/'htsim_ndp')])
    assignments = [f'{k}={v}' for k,v in sorted(env.items())
                   if k.startswith(('PYTHON','FRONTIER','STEP5','VLLM','OMP','OPENBLAS','MKL','NUMEXPR','TMP'))]
    return ['sudo','-n','systemd-run','--scope','--quiet','-p','MemoryMax=30G',
            'runuser','-u','brainpp','--','env',*assignments,*command,'--chdir',str(REPO),'--']



def namespace(view, env, readonly=False):
    import runpy
    retained = runpy.run_path('/data/ycfeng/tmp/d29_performance_regressions_final05/run_suite.py')
    retained_namespace = retained['namespace']
    retained_namespace.__globals__['REPO'] = REPO
    assignments = [f'{key}={value}' for key, value in sorted(env.items())
                   if key.startswith(('PYTHON', 'FRONTIER', 'STEP5', 'VLLM', 'OMP', 'OPENBLAS', 'MKL', 'NUMEXPR', 'TMP'))]
    command = retained_namespace(view, readonly)
    if view.parent.name == 'strict77':
        command[-1:-1] = ['--ro-bind', '/data/ycfeng/tmp/d29_native_cache_key_20261024.xBHOmb/frozen-matrix/generated_repo_root',
                         '/data/ycfeng/Frontier/.worktrees/step5-d29-native-cache-key']
    return ['sudo', '-n', 'systemd-run', '--scope', '--quiet', '-p', 'MemoryMax=30G',
            'runuser', '-u', 'brainpp', '--', 'env', *assignments, *command]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('suite',choices=['strict77','unit','supplement101','metadata196','catalogue53','governance52'])
    suite = parser.parse_args().suite
    dest = ROOT/suite
    dest.mkdir(exist_ok=False)
    view = dest/'generated_repo_root'
    view.mkdir()
    before = state()
    assert not (REPO/NATIVE/'htsim_ndp').exists(), 'Canonical unit namespace requires the pinned source binary to remain absent'
    assert (before['head'],before['tree'],before['submodule_head']) == (HEAD,TREE,SUB)
    assert not before['status'] and not before['submodule_status'], before
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR='/data/ycfeng/tmp',FRONTIER_TMP_ROOT='/data/ycfeng/tmp',
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
               PYTHONPATH=str(REPO) if suite=='strict77' else f'{REPO}:{REPO}/tests/unit')
    for key in ('FRONTIER_TEST_CPU_WRITER_FILE','FRONTIER_TEST_CPU_WRITER_SHA256'):
        env.pop(key,None)
    py = PY312 if suite in ('supplement101','governance52') else PY310
    if suite=='supplement101':
        env.update(STEP5_NATIVE_SOURCE_ROOT='/data/ycfeng/tmp/d29_published_sources01/source/vllm',VLLM_TARGET_DEVICE='cpu')
    if suite=='governance52':
        env.update(PYTHONPATH=f'{REPO}:{REPO}/tests/unit:/data/ycfeng/tmp/d29_py312_cpu_extras01/site',VLLM_TARGET_DEVICE='cpu')
    code = "import json,platform,sys,importlib.metadata as m;out={'python':platform.python_version(),'executable':sys.executable};\nfor p in ['pytest','numpy','pandas','scipy','scikit-learn','plotly','fasteners','ddsketch','torch','aiohttp']:\n try: out[p]=m.version(p)\n except m.PackageNotFoundError: out[p]=None\nprint(json.dumps(out))"
    versions = json.loads(subprocess.check_output([py,'-B','-c',code],env=env,text=True))
    junit = None
    if suite=='strict77':
        STRICT.mkdir(exist_ok=False)
        commands = [['timeout','1800s',py,'-B','tests/e2e/refactor_fidelity/run_matrix.py','run','--repo-root',str(REPO),
                     '--python-bin',py,'--label','candidate','--output-root',str(STRICT),'--jobs','4',
                     '--clean-cache','--continue-on-failure'],
                    ['timeout','180s',py,'-B','tests/e2e/refactor_fidelity/run_matrix.py','compare','--output-root',str(STRICT),
                     '--baseline-label',BASELINE,'--candidate-label',str(STRICT/'candidate')]]
        logs = [STRICT/'run.log',STRICT/'compare.log']
    else:
        tests = {
            'unit':['tests/unit'],
            'supplement101':['tests/unit/test_step5_canonical_g3.py','tests/unit/test_step5_native_position_cyclic.py'],
            'metadata196':['tests/unit/test_comm_operator_families.py','tests/unit/test_ep_wave_trace_context.py',
                           'tests/unit/test_quantization_manager_registry.py','tests/unit/test_quantization_module_precision.py',
                           'tests/unit/test_op_trace_utils.py','tests/unit/test_ep_trace.py','tests/unit/test_collective_sim_cc_backend.py'],
            'catalogue53':['tests/unit/test_moe_share_expert_operator_families.py','tests/unit/test_dsv4_precision_metadata.py'],
            'governance52':['tests/unit/test_profiling_governance_minimal_red.py'],
        }[suite]
        seconds = {'unit':1800,'supplement101':180,'metadata196':180,'catalogue53':120,'governance52':120}[suite]
        stem = Path('/data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/regressions/unit/results') if suite=='unit' else dest/'results'
        basetemp = stem if suite=='unit' else dest/'basetemp'
        junit = Path(str(stem)+'.xml')
        logs = [Path(str(stem)+'.log')]
        assert not any(p.exists() for p in (basetemp,junit,*logs))
        command = ['timeout',f'{seconds}s',py,'-B','-m','pytest',*tests,'-q','-p','no:cacheprovider']
        if suite=='unit':
            command.append('--continue-on-collection-errors')
        commands = [command+[f'--basetemp={basetemp}',f'--junitxml={junit}']]
    mount = namespace(view,env)
    evidence = {'task_id':'D29-LEDGER-FINAL-FROZEN-REGRESSIONS','suite':suite,'source_before':before,
                'versions':versions,'environment':{k:env[k] for k in sorted(env) if k.startswith(('PYTHON','FRONTIER','STEP5','VLLM','OMP','OPENBLAS','MKL','NUMEXPR','TMP'))},
                'source_entries_readonly':[p.name for p in sorted(REPO.iterdir())],
                'generated_file_root':str(view),'junit':None if junit is None else str(junit),'executions':[],
                'native_binary_projection':False,'source_binary_present':(REPO/NATIVE/'htsim_ndp').exists(),
                'runtime_user':'brainpp','MemoryMax_bytes':30*1024**3}
    (dest/'launch.json').write_text(json.dumps({**evidence,'commands':[mount+c for c in commands]},indent=2)+'\n')
    for command,log in zip(commands,logs):
        started = time.monotonic()
        with log.open('x') as stream:
            result = subprocess.run(mount+command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=False)
        evidence['executions'].append({'command':mount+command,'returncode':result.returncode,
                                      'duration_seconds':time.monotonic()-started,'log':str(log)})
        if suite=='strict77' and not (STRICT/'candidate/manifest.json').is_file():
            break
    evidence['source_after'] = state()
    (dest/'execution.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'suite':suite,'raw_exits':[e['returncode'] for e in evidence['executions']],
                      'junit':evidence['junit'],'receipt':str(dest/'execution.json'),
                      'source_after':evidence['source_after']},indent=2))
    return max([e['returncode'] for e in evidence['executions']]+[0])


if __name__=='__main__':
    raise SystemExit(main())
