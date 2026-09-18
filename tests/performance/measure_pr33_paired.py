"""Measure paired revisions using the existing simulator wall-time runner."""

from __future__ import annotations
import argparse, json, os, resource, subprocess, sys, time
from pathlib import Path

BASELINE: Path
CANDIDATE: Path
OUT: Path

CASES = [
    {
        'name': 'small_dense', 'model': 'dense', 'total_gpus': 16,
        'shape': {'attn_tp': 4, 'attn_dp': 1, 'moe_tp': 1, 'moe_ep': 1, 'pp': 2},
        'mode': 'sequential', 'num_requests': 2, 'qps': 2.0,
        'prefill_tokens': 16, 'decode_tokens': 8,
    },
    {
        'name': 'longer_dense', 'model': 'dense', 'total_gpus': 16,
        'shape': {'attn_tp': 4, 'attn_dp': 1, 'moe_tp': 1, 'moe_ep': 1, 'pp': 2},
        'mode': 'sequential', 'num_requests': 4, 'qps': 10.0,
        'prefill_tokens': 64, 'decode_tokens': 16,
    },
    {
        'name': 'representative_moe', 'model': 'moe', 'total_gpus': 32,
        'shape': {'attn_tp': 8, 'attn_dp': 1, 'moe_tp': 1, 'moe_ep': 8, 'pp': 2},
        'mode': 'sequential', 'num_requests': 2, 'qps': 5.0,
        'prefill_tokens': 32, 'decode_tokens': 8,
    },
]

def usage():
    u = resource.getrusage(resource.RUSAGE_CHILDREN)
    return u.ru_utime, u.ru_stime

def git_info(repo):
    sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(['git','status','--porcelain'], cwd=repo, text=True)
    return sha, dirty

def run_one(repo, case, attempt):
    side = 'baseline' if repo == BASELINE else 'candidate'
    out = OUT / case['name'] / side / f'attempt-{attempt}'
    out.mkdir(parents=True, exist_ok=False)
    payload = {key: value for key, value in case.items() if key != 'name'}
    payload['attempt_index'] = attempt
    payload.update({'seed': 42, 'simulation_mode': 'online', 'dummy_execution_time_ms': 1.0,
                    'device': 'h800', 'network_device': 'h800_dgx'})
    case_json = out / 'case.json'
    result_json = out / 'result.json'
    case_json.write_text(json.dumps(payload, indent=2) + '\n')
    command = [sys.executable, str(repo / 'tests/performance/sim_walltime_scaling/run_case.py'),
               '--case-json', str(case_json), '--result-json', str(result_json)]
    env = os.environ.copy()
    env.update({'PYTHONPATH': str(repo), 'PYTHONDONTWRITEBYTECODE':'1',
                'WANDB_DISABLED':'true', 'VIDUR_DISABLE_WANDB':'1',
                'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
                'MKL_NUM_THREADS':'1', 'NUMEXPR_NUM_THREADS':'1',
                'FRONTIER_LOG_LEVEL':'WARNING', 'TMPDIR':str(OUT), 'FRONTIER_TMP_ROOT':str(OUT)})
    log_path = out / 'run.log'
    u0, s0 = usage()
    wall0 = time.perf_counter()
    proc = subprocess.run(command, cwd=repo, env=env, stdout=log_path.open('w'), stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - wall0
    u1, s1 = usage()
    result = json.loads(result_json.read_text()) if result_json.exists() else {'status':'missing'}
    result['runner_elapsed_s'] = elapsed
    result['child_user_s'] = u1-u0
    result['child_system_s'] = s1-s0
    result['child_cpu_s'] = (u1-u0)+(s1-s0)
    result['command_exit_code'] = proc.returncode
    result['log_path'] = str(log_path)
    result_path = out / 'measurement.json'
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result

def main():
    global BASELINE, CANDIDATE, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    BASELINE, CANDIDATE, OUT = args.baseline.resolve(), args.candidate.resolve(), args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = {'baseline':{}, 'candidate':{}, 'cases':CASES, 'repetitions':args.repetitions,
                'python':sys.executable, 'environment':{k:os.environ.get(k) for k in ('CONDA_PREFIX','VIRTUAL_ENV')}}
    for name, repo in [('baseline', BASELINE), ('candidate', CANDIDATE)]:
        sha, dirty = git_info(repo)
        manifest[name] = {'repo':str(repo), 'sha':sha, 'dirty':dirty}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    results=[]
    for attempt in range(args.repetitions):
        for case in CASES:
            for repo in ((BASELINE, CANDIDATE) if attempt % 2 == 0 else (CANDIDATE, BASELINE)):
                r = run_one(repo, case, attempt)
                row = {'case':case['name'], 'side':'baseline' if repo==BASELINE else 'candidate', 'attempt':attempt,
                       'status':r.get('status'), 'sim_wallclock_s':r.get('sim_wallclock_s'), 'init_s':r.get('init_s'),
                       'total_proc_s':r.get('total_proc_s'), 'event_count':r.get('event_count'),
                       'completed_requests':r.get('completed_requests'), 'runner_elapsed_s':r.get('runner_elapsed_s'),
                       'child_cpu_s':r.get('child_cpu_s'), 'peak_rss_mb':r.get('peak_rss_mb')}
                results.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                (OUT/'results.json').write_text(json.dumps(results, indent=2, sort_keys=True)+'\n')
    print(f'completed {len(results)} measurements; output={OUT}')

if __name__ == '__main__':
    main()
