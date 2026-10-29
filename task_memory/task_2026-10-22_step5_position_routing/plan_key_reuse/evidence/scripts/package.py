"""Archive the bounded verification and Git-derived two-path accounting."""
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CODE = Path('/data/ycfeng/Frontier/.worktrees/step5-d29-plan-key')
RECORD = CODE / 'task_memory/task_2026-10-22_step5_position_routing/plan_key_reuse'
OUT = RECORD / 'evidence'
OUT.mkdir(exist_ok=True)
BASE = '087db1bfcb3dbc2377d4032f650b29c48fb104fd'
ORIGIN = '1aee0af11cd58691e64603b99eca0ac459a5b350'
PATHS = ['frontier/cc_backend/collective_plan.py', 'tests/unit/test_step5_moe_pricing.py']
DONORS = ['frontier/cc_backend/backends/collective_sim_cc_backend.py',
          'frontier/cc_backend/base_cc_backend.py', 'tests/unit/test_collective_sim_cc_backend.py']
git = lambda *args: subprocess.check_output(['git', *args], cwd=CODE, text=True)
assert git('diff', '--name-only', BASE, '--', *DONORS) == ''
diff = git('diff', BASE, '--', *PATHS)
(OUT / 'implementation.diff').write_text(diff)
(OUT / 'changed_files.txt').write_text('\n'.join(PATHS) + '\n')
def counts(base, paths):
    result = {}
    for line in git('diff', '--numstat', base, '--', *paths).splitlines():
        added, removed, path = line.split('\t')
        added, removed = int(added), int(removed)
        result[path] = {'added': added, 'removed': removed, 'gross': added + removed,
                        'retained_growth': added - removed}
    return result
step = counts(BASE, PATHS)
assert step[PATHS[0]]['gross'] == 5 and step[PATHS[1]]['gross'] == 11
assert sum(value['retained_growth'] for value in step.values()) == 14
cumulative = counts(ORIGIN, PATHS + DONORS)
assert cumulative[PATHS[0]]['gross'] == 139 and cumulative[PATHS[1]]['gross'] == 311
assert [cumulative[path]['gross'] for path in DONORS] == [93, 14, 236]
copies = {'freeze.json': ROOT / 'freeze.json', 'verification.json': ROOT / 'verification.json'}
original = json.loads((ROOT / 'freeze.json').read_text())
protected = Path(original['protected_body'])
assert hashlib.sha256(protected.read_bytes()).hexdigest() == original['protected_body_sha256']
copies['scripts/protected_body.py'] = protected
for name in ('preflight.py', 'fixture_wiring.py', 'run.py', 'verify.py', 'package.py', 'manifest.py'):
    copies['scripts/' + name] = ROOT / name
for name in ('red', 'green', 'affected', 'baseline-tests'):
    for extension in ('log', 'xml'):
        copies['tests/' + name + '.' + extension] = ROOT.parent / '01' / (name + '.' + extension)
for arm in ('baseline', 'candidate'):
    for name in ('wave_result.json', 'pricing.prof'):
        copies[arm + '/' + name] = ROOT / arm / name
    copies[arm + '/run.log'] = ROOT / (arm + '.log')
    copies[arm + '/output_result.json'] = ROOT / arm / 'accepted-output' / arm / 'result.json'
    copies[arm + '/emission.pstats'] = ROOT / arm / 'accepted-output' / arm / 'emission.pstats'
for name, path in {'pricing_payload.json': ROOT / 'candidate/pricing_payload.json',
                   'ledger.json': ROOT / 'candidate/accepted-output/candidate/ledger.json',
                   'op_traces.jsonl': ROOT / 'candidate/accepted-output/candidate/iteration-0/op_traces.jsonl'}.items():
    copies['exact_output/' + name] = path
for relative, source in copies.items():
    destination = OUT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
frozen = json.loads((ROOT / 'freeze.json').read_text())
tests_prefix = ('timeout 240s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env '
    'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 '
    'PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/d29_plan_key_reuse/01/cache '
    'XDG_CACHE_HOME=/data/ycfeng/tmp/d29_plan_key_reuse/01/cache FRONTIER_TMP_ROOT=/data/ycfeng/tmp/d29_plan_key_reuse/01 '
    '/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest ')
commands = []
for label in ('red', 'green'):
    command = tests_prefix + ('-q tests/unit/test_step5_moe_pricing.py::test_lossless_plan_order_identity_and_count_separation '
        '-o cache_dir=/data/ycfeng/tmp/d29_plan_key_reuse/01/cache/pytest-{0} '
        '--basetemp=/data/ycfeng/tmp/d29_plan_key_reuse/01/pytest-{0} '
        '--junitxml=/data/ycfeng/tmp/d29_plan_key_reuse/01/{0}.xml '
        '> /data/ycfeng/tmp/d29_plan_key_reuse/01/{0}.log 2>&1').format(label)
    commands.append({'label': label, 'cwd': str(CODE), 'command': command, 'exit': int(label == 'red')})
for label, arm, cache in (('affected', 'candidate', 'affected'), ('baseline-tests', 'baseline', 'baseline')):
    command = tests_prefix + ('-q -ra tests/unit/test_step5_moe_pricing.py tests/unit/test_collective_sim_cc_backend.py '
        'tests/unit/test_step5_contract_reuse.py -o cache_dir=/data/ycfeng/tmp/d29_plan_key_reuse/01/cache/pytest-{0} '
        '--basetemp=/data/ycfeng/tmp/d29_plan_key_reuse/01/pytest-{0} '
        '--junitxml=/data/ycfeng/tmp/d29_plan_key_reuse/01/{1}.xml '
        '> /data/ycfeng/tmp/d29_plan_key_reuse/01/{1}.log 2>&1').format(cache, label)
    commands.append({'label': label, 'cwd': frozen['sources'][arm]['root'], 'command': command, 'exit': 0})
measure_prefix = ('timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env '
    'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ')
commands.append({'label': 'preflight', 'cwd': str(ROOT), 'exit': 0, 'command': measure_prefix +
    '/data/ycfeng/envs/frontier-py310/bin/python -B ' + str(ROOT / 'preflight.py') +
    ' > ' + str(ROOT / 'preflight.log') + ' 2>&1'})
for arm in ('baseline', 'candidate'):
    commands.append({'label': arm + '-wave', 'cwd': str(ROOT), 'exit': 0, 'command': measure_prefix +
        'TMPDIR=' + str(ROOT / 'cache') + ' XDG_CACHE_HOME=' + str(ROOT / 'cache') +
        ' MPLCONFIGDIR=' + str(ROOT / 'cache/mpl') + ' NUMBA_CACHE_DIR=' + str(ROOT / 'cache/numba') +
        ' /data/ycfeng/envs/frontier-py310/bin/python -B ' + str(ROOT / 'run.py') + ' ' + arm +
        ' > ' + str(ROOT / (arm + '.log')) + ' 2>&1'})
for name in ('verify.py', 'package.py', 'manifest.py'):
    commands.append({'label': name, 'cwd': str(ROOT), 'exit': 0, 'command': measure_prefix +
        'TMPDIR=' + str(ROOT / 'cache') + ' XDG_CACHE_HOME=' + str(ROOT / 'cache') +
        ' /data/ycfeng/envs/frontier-py310/bin/python -B ' + str(ROOT / name)})
receipt = {'task_id': 'D29-PLAN-KEY-REUSE', 'session_id': frozen['session_id'], 'status': 'PASS',
    'source_commit': git('rev-parse', 'HEAD').strip(),
    'base': BASE, 'd29_origin': ORIGIN, 'step_counts': step, 'cumulative_path_counts': cumulative,
    'step_gross': 16, 'retained_growth': 14, 'owner_ceiling_unchanged': 3956,
    'donors_unchanged': True, 'global_union_accounting': 'Parent-owned; not rerun in this two-path slice.',
    'source_sha256': {path: hashlib.sha256((CODE / path).read_bytes()).hexdigest() for path in PATHS},
    'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                    'user': 'brainpp', 'blas_threads': 1, 'memory': 'MemoryMax=30G'},
    'verification': json.loads((ROOT / 'verification.json').read_text()), 'commands': commands,
    'local_submodule': {'pin': '3dac53c544e546574cb6e8078cd14053d0916113',
        'preparation': 'git clone --shared --no-checkout from protected snapshot local repository, then git checkout --detach at existing pin; no network or pin change.'},
    'limitations': 'Cached manager/predictor helper with explicit real input owner and real P/D backends; not full Simulator/CLI/native qualification. One torch-dependent test skipped in both arms. No numerical/proof cache or diagnostic key/byte adapter. Parent reviews, integration, union accounting and full workflow remain pending.'}
(OUT / 'evidence.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'status': receipt['status'], 'step_counts': step, 'cumulative_path_counts': cumulative}, indent=2))
