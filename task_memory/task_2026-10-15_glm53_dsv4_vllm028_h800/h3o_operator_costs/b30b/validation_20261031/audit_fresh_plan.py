"""Independently reconcile existing B30b fresh artifacts using their raw inputs."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

root = Path.cwd()
area = root / 'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b30b'
fresh = area / 'fresh_dry_run_20261031_01'
checks = []
jobs = []
expected_counts = {'A_intra': (132, 2), 'B_same_rail': (142, 1), 'B_ring_boundary': (142, 1), 'C_nixl': (80, 1), 'F2_launch_depth': (84, 1)}
def check(name, actual, expected):
    ok = math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9) if isinstance(expected, float) else actual == expected
    checks.append(dict(name=name, expected=expected, actual=actual, status='PASS' if ok else 'FAIL'))

for stage in ('b30b_links_kv', 'b30b_launch_depth'):
    config = json.loads((area / ('stage_' + stage + '.json')).read_text())
    plan = json.loads((fresh / (stage + '.json')).read_text())
    tables = {}
    for source in plan['timing_sources']:
        path = Path(source['path'])
        check('timing_sha256:' + str(path), hashlib.sha256(path.read_bytes()).hexdigest(), source['sha256'])
        with path.open(newline='') as handle:
            tables[str(path)] = list(csv.DictReader(handle))
    g5h = [r for p, rows in tables.items() if '/h200/' in p for r in rows]
    k5 = [r for p, rows in tables.items() if '/w5_comm_k5/' in p for r in rows]
    walls, phases, total, pass_rows = [], [], 0, 0
    for cfg, phase in zip(config['phases'], plan['phases']):
        name, cases = phase['name'], phase['cases']
        count, copies = expected_counts[name]
        check(name + ':case_array_length', len(cases), count)
        check(name + ':copies', phase['copies'], copies)
        check(name + ':case_total', phase['total_cases'], len(cases) * copies)
        check(name + ':case_uniqueness', len({json.dumps(c, sort_keys=True) for c in cases}), len(cases))
        settings = cfg['config']
        if name == 'C_nixl':
            coefficient = max(float(r['complete_median_us']) / int(r['bytes_per_reader']) for r in k5)
            service = sum(c['bytes_per_reader'] for c in cases) * coefficient * settings['passes'] * (settings['warmup'] + settings['repeats']) / 1e6
            overhead = len(cases) * settings['passes'] * config['estimate']['nixl_case_overhead_seconds']
            check(name + ':max_pool_use', max(c['blocks_per_request'] * c['requests_in_flight'] for c in cases), 864)
            check(name + ':pool_admits_all_cases', all(c['blocks_per_request'] * c['requests_in_flight'] <= settings['kv_cache']['num_gpu_blocks'] for c in cases), True)
            check(name + ':T6144_live_blocks', 24 + 2 + 2 + 2 + 16, 46)
            check(name + ':T8192_live_blocks', {c['blocks_per_request'] for c in cases if c['prompt_tokens'] == 8192}, {54})
        elif name == 'F2_launch_depth':
            enqueued = sum(min(settings['max_launches'], settings['max_enqueued_kernels'] // c['kernels_per_graph']) * c['kernels_per_graph'] for c in cases)
            check(name + ':kernel_enqueue_ceiling_sum', enqueued, 44040192)
            check(name + ':mode_counts', dict(Counter(c['mode'] for c in cases)), {'graph': 78, 'plain': 6})
            service = sum(c['sleep_ms'] for c in cases) / 1000 + enqueued * min(float(r['median_us']) for r in g5h) / 1e6
            overhead = sum(settings['kernels_per_graph']) * config['estimate']['capture_seconds_per_kernel']
        else:
            envelopes = []
            for case in cases:
                rows = [r for r in g5h if r['op'] == case['op'] and r['cuda_graph'].lower() == str(case['cuda_graph']).lower()]
                envelopes.append(max(float(r['median_us']) for r in rows) * max(1, case['max_input_bytes'] / max(int(r['max_input_bytes']) for r in rows)))
            multiplier = settings['passes'] * (settings['warmup_calls'] + 4 * settings['repeats']) * settings['calls_per_repeat'] + 1
            check(name + ':service_multiplier', multiplier, 1216 if name == 'A_intra' else 376)
            service = sum(envelopes) * multiplier / 1e6
            overhead = len(cases) * settings['passes'] * config['estimate']['collective_case_overhead_seconds']
            check(name + ':graph_cases', sum(c['cuda_graph'] for c in cases), 42 if name == 'A_intra' else 14)
        wall = service + overhead + config['estimate']['process_startup_seconds']
        for field, value in (('service_seconds', service), ('overhead_seconds', overhead), ('wall_seconds', wall)):
            check(name + ':' + field, phase['estimate'][field], value)
        check(name + ':padded_phase_below_timeout', wall * config['estimate']['safety_factor'] < cfg['timeout_seconds'], True)
        total += len(cases) * copies
        pass_rows += len(cases) * copies * settings.get('passes', 1)
        walls.append(wall)
        phases.append(dict(name=name, cases_per_copy=len(cases), copies=copies, total_cases=len(cases)*copies, planned_pass_rows=len(cases)*copies*settings.get('passes', 1), wall_seconds=wall, timeout_seconds=cfg['timeout_seconds']))
    allocated = config['gpu'] * config['replica']
    unpadded = config['estimate']['startup_seconds'] + sum(walls)
    wall = unpadded * config['estimate']['safety_factor']
    gpuh = wall * allocated / 3600
    check(stage + ':phase_count', len(plan['phases']), len(config['phases']))
    check(stage + ':total_cases', plan['total_cases'], total)
    check(stage + ':unpadded_wall_seconds', plan['unpadded_wall_seconds'], unpadded)
    check(stage + ':wall_seconds', plan['wall_seconds'], wall)
    check(stage + ':allocated_gpus', plan['allocated_gpus'], allocated)
    check(stage + ':gpu_hours', plan['gpu_hours'], gpuh)
    check(stage + ':timeout_gpu_hours', plan['timeout_gpu_hours'], config['timeout_seconds'] * allocated / 3600)
    check(stage + ':padded_job_below_timeout', wall < config['timeout_seconds'], True)
    jobs.append(dict(stage=stage, total_cases=total, planned_pass_rows=pass_rows, phases=phases, allocated_gpus=allocated, unpadded_wall_seconds=unpadded, wall_seconds=wall, gpu_hours=gpuh, timeout_seconds=config['timeout_seconds'], timeout_gpu_hours=plan['timeout_gpu_hours'], plan_path=str(fresh / (stage + '.json'))))
for record in checks:
    for key in ('expected', 'actual'):
        if isinstance(record[key], set):
            record[key] = sorted(record[key])
result = dict(status='PASS' if all(c['status'] == 'PASS' for c in checks) else 'FAIL', jobs=jobs, combined_gpu_hours=sum(j['gpu_hours'] for j in jobs), combined_timeout_gpu_hours=sum(j['timeout_gpu_hours'] for j in jobs), checks=checks, limits='CPU arithmetic of fresh generated arrays and local retained timings; no GPU measurements or platform actions.')
output = area / 'validation_20261031'
output.mkdir(exist_ok=True)
(output / 'generated_plan_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(result['status'], len(checks), 'independent checks; combined GPUh', result['combined_gpu_hours'])
for job in jobs:
    print(job['stage'], 'cases', job['total_cases'], 'pass rows/trials', job['planned_pass_rows'], 'wall', job['wall_seconds'], 'GPUh', job['gpu_hours'])
raise SystemExit(0 if result['status'] == 'PASS' else 1)
