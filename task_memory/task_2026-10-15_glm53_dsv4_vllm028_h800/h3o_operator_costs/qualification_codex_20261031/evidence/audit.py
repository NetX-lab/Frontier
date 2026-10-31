"""Capture and verify only retained P2 qualification artifacts on the local host."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

AREA = Path(__file__).resolve().parents[1]
MAIN = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028')
TASK = MAIN / 'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800'
NATIVE = Path('/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69')
STAGE = Path('/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4')
PIN = '2cf0a6915ce544dc493a0990f2ea38d81601128a'
ANSI = re.compile(r'\x1b\[[0-9;]*m')
SOURCES = {}


def retain(source_id, path, *, pointers=(), lines=(), expected_sha256=None):
    path = Path(path)
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None:
        assert digest == expected_sha256, str(path)
    record = {'path': str(path), 'bytes': len(content), 'sha256': digest,
              'pointers': list(pointers), 'excerpts': []}
    text_lines = content.decode().splitlines() if lines else []
    for first, last in lines:
        record['excerpts'].append({'first_line': first, 'last_line': last,
                                  'text': '\n'.join(text_lines[first - 1:last])})
    SOURCES[source_id] = record
    return content


def retained_json(source_id, path, **kwargs):
    return json.loads(retain(source_id, path, **kwargs))


def capture():
    initial = json.loads((AREA / 'evidence/initial_sources.json').read_text())
    selectors = [(), ((147, 150), (269, 270), (435, 447)), ((142, 163),),
                 ((338, 356),), ((79, 84), (274, 283), (471, 488)),
                 ((56, 62), (272, 293), (410, 418)),
                 ((28, 33), (71, 103)), ((28, 33), (194, 207), (223, 248)),
                 ((23, 27), (105, 160)), ((393, 402), (666, 678), (817, 826))]
    for index, (source, excerpts) in enumerate(zip(initial['sources'], selectors), 1):
        source_id = f'A{index:02}'
        retain(source_id, source['path'], lines=excerpts)
        SOURCES[source_id]['initial_sha256'] = source['sha256']
        SOURCES[source_id]['changed_since_initial'] = SOURCES[source_id]['sha256'] != source['sha256']
    source_specs = {
        'NIXL': (NATIVE / 'vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py',
                 ((99, 131), (960, 1062), (1388, 1414), (1425, 1457), (2412, 2446))),
        'NATIVE_MEMORY': (NATIVE / 'vllm/v1/worker/gpu_worker.py',
                          ((515, 565), (754, 806))),
        'NATIVE_COUNTERS': (NATIVE / 'vllm/utils/mem_utils.py', ((30, 31), (305, 326))),
        'PLANNER_SOURCE': (MAIN / 'frontier/scheduler/utils/memory_planner.py',
                           ((239, 260), (304, 329))),
        'CAPACITY_SOURCE': (MAIN / 'frontier/attention/memory.py', ((211, 224),)),
        'PAYLOAD_SOURCE': (MAIN / 'frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py',
                          ((19, 65), (107, 136))),
        'GROUP_COUNTS': (MAIN / 'task_memory/task_2026-10-21_v4_grouped_nixl/native_retention.md',
                         ((14, 74),)),
        'EX_SITU': (TASK / 'test_report_2026-10-31_ex_situ_input_audit.md', ((88, 94),)),
    }
    for source_id, (path, excerpts) in source_specs.items():
        expected = '4860f51b6ac3301c243794b7626b5536f6b0ec41204e426ee6b61bc0e1d0bdb7' if source_id == 'NIXL' else None
        retain(source_id, path, lines=excerpts, expected_sha256=expected)
    cases = {}
    for scale, expected_blocks in ((32, 109934), (64, 112914)):
        prefix = f'P{scale}'
        case_dir = TASK / f'h200_mainline/diagnostic/inputs/p2_{scale}r'
        receipts = {}
        for name in ('run_check_status.groundtruth_clean', 'run_manifest', 'launcher_receipt',
                     'platform_terminal_receipt', 'replay_finished', 'download_receipt', 'cloud_inventory'):
            receipts[name] = retained_json(f'{prefix}_{name}', case_dir / f'{name}.json', pointers=('/',))
        manifest = receipts['run_manifest']
        launcher = receipts['launcher_receipt']
        assert launcher['run_manifest_sha256'] == SOURCES[f'{prefix}_run_manifest']['sha256']
        run_tag = manifest['run_tag']
        r4 = receipts['run_check_status.groundtruth_clean']['runs'][run_tag]
        assert r4['status'] == 'PASS' and all(value.startswith('PASS:') for value in r4['checks'].values())
        assert manifest['groundtruth_commit'] == PIN and manifest['resources']['gpus'] == scale
        graph = json.loads(r4['checks']['rank_runtime_graph'].removeprefix('PASS: '))
        assert graph['physical_ranks'] == scale and graph['phases_per_rank'] == 6
        terminal = receipts['platform_terminal_receipt']
        assert terminal['status']['phase'] == 'Succeeded'
        assert terminal['creator'] == 'i-fengyicheng' and terminal['preemptible_label'] == 'no'
        assert len(terminal['replicas']) == scale // 8
        assert all(c['restartCount'] == 0 and c['state']['terminated']['exitCode'] == 0
                   for replica in terminal['replicas'] for c in replica['containers'])
        local = Path(receipts['download_receipt']['local_root'])
        selected = [entry for entry in receipts['cloud_inventory']['files']
                    if entry['path'].endswith(('/server.log', '/result.json'))
                    or re.search(r'/serving/rank\d+\.json$', entry['path'])]
        nodes, ranks, log_rows = [], [], []
        for entry in selected:
            relative = entry['path']
            source_id = f'{prefix}_{relative}'
            content = retain(source_id, local / relative, expected_sha256=entry['sha256'])
            assert len(content) == entry['bytes']
            if relative.endswith('/server.log'):
                rows = []
                for number, text in enumerate(content.decode().splitlines()[:1600], 1):
                    if any(token in text for token in ('Model loading took', 'Available KV cache memory:',
                                                       'GPU KV cache size:', 'Registering packed KV cache:',
                                                       'Actual usage is')):
                        rows.append({'line': number, 'text': ANSI.sub('', text)})
                log_rows.append({'source': source_id, 'rows': rows, 'inspection_limit_lines': 1600})
            elif relative.endswith('/result.json'):
                result = json.loads(content)
                assert result['workload_returncode'] == 0 and result['stage'] == 'finished'
                gpus = [line for line in result['nvidia_smi'].splitlines() if line.startswith('GPU ')]
                assert len(gpus) == 8 and all('NVIDIA H200' in line for line in gpus)
                assert result['run_manifest_sha256'] == launcher['run_manifest_sha256']
                nodes.append({'source': source_id, 'host': result['host'], 'role': result['plan']['role'],
                              'lane': result['plan']['lane'], 'gpu_count': len(gpus),
                              'versions': result['versions'], 'engine': result['plan']['engine']})
            else:
                rank = json.loads(content)
                inventory = rank['inventory']
                assert inventory['device_identity']['name'] == 'NVIDIA H200'
                assert inventory['cache']['num_blocks'] == expected_blocks
                memory = inventory['memory']
                scalars = memory['native_scalars']
                non_kv = scalars['requested_memory'] - scalars['available_kv_cache_memory_bytes']
                assert non_kv == scalars['total_consumed'] + scalars['peak_activation_memory']
                assert scalars['cudagraph_memory_estimate'] == 0
                ranks.append({'source': source_id, 'pointers': ['/inventory/memory', '/inventory/workspace_bytes',
                                                              '/inventory/cache/num_blocks', '/inventory/effective_graph'],
                              'host': rank['host'], 'role': rank['role'], 'memory': memory,
                              'workspace_bytes': inventory['workspace_bytes'],
                              'num_blocks': inventory['cache']['num_blocks'],
                              'graph_mode': inventory['effective_graph']['configured_mode'],
                              'pre_graph_non_kv_bytes': non_kv,
                              'non_load_residual_bytes': non_kv - memory['model_load_delta_bytes']})
        assert len(nodes) == scale // 8 and len(ranks) == scale
        assert {node['role'] for node in nodes} == {'prefill', 'decode'}
        planner = retained_json(f'{prefix}_F4', STAGE / f'f_variants/f4/p2_{scale}_init/planner_blocks.json',
                                pointers=('/planner', '/native_num_blocks', '/decode_lanes'))
        comparison = retained_json(f'{prefix}_F4_B28', TASK / f'h3o_operator_costs/codex_resume_20261031/f4_p2_{scale}_b28_comparison.json',
                                   pointers=('/statuses', '/metrics', '/config_differences'))
        reading = retained_json(f'{prefix}_B29', TASK / f'h3o_operator_costs/codex_resume_20261031/p2_{scale}_warm_k19b29/reading.json',
                                pointers=('/verdict', '/repetitions', '/routing_status'))
        config_path = STAGE / f'a10/p2_{scale}_warm_k19b29/metrics/deepseek_ai_deepseek_v4_flash/online_serving/a10_p2_{scale}_operator_path/config.json'
        config = retained_json(f'{prefix}_B29_CONFIG', config_path,
                               pointers=('/kv_cache_transfer_config', '/cluster_config/replica_scheduler_config'))
        terms = planner['planner']
        predicted = (terms['requested_memory_bytes'] - terms['parameter_memory_per_device_bytes']
                     - terms['non_kv_cache_overhead_bytes']) // terms['physical_block_size_bytes']
        assert set(planner['decode_lanes'].values()) == {predicted}
        assert planner['native_num_blocks'] == expected_blocks
        assert comparison['statuses'] == ['FINISHED', 'FINISHED']
        assert len(comparison['metrics']) == 5 and all(item['equal'] for item in comparison['metrics'].values())
        metric_rows = []
        for repeat, repetition in reading['repetitions'].items():
            assert len(repetition['metrics']) == 4
            for metric in repetition['metrics']:
                calculated = abs(metric['frontier'] - metric['vllm']) / abs(metric['vllm'])
                assert math.isfinite(calculated) and math.isclose(calculated, metric['relative_error'], abs_tol=1e-14)
                metric_rows.append({'repeat': repeat, **metric})
        assert len(metric_rows) == 12
        cases[f'p2_{scale}'] = {'run_tag': run_tag, 'rjob_name': r4['rjob_name'], 'original_r4_status': r4['status'],
                               'original_authorization': manifest['authorization'], 'prelaunch_timestamp': manifest['written_before_launch_utc'],
                               'manifest_promotion_fields': [key for key in manifest if 'promo' in key or 'admiss' in key],
                               'r4_checks_except_graph': {key: value for key, value in r4['checks'].items() if key != 'rank_runtime_graph'},
                               'graph_summary': {key: graph[key] for key in ('physical_ranks', 'phases_per_rank', 'coverage')},
                               'native_blocks': expected_blocks, 'nodes': nodes, 'ranks': ranks, 'startup_logs': log_rows,
                               'planner': planner, 'planner_excess_blocks': predicted - expected_blocks,
                               'planner_relative_error': (predicted - expected_blocks) / expected_blocks,
                               'existing_f4_comparison': comparison, 'b29_verdict': reading['verdict'], 'b29_metrics': metric_rows,
                               'b29_transfer_config': config['kv_cache_transfer_config'],
                               'b29_scheduler_config': config['cluster_config']['replica_scheduler_config']}
    group_names = ['compressed', 'swa_0', 'swa_1', 'csa_state_0', 'hca_state_0']
    counts = [24, 2, 2, 2, 16]
    page_bytes = [1002240, 823680, 786240, 870912, 656640]
    structural = sum(count * size for count, size in zip(counts, page_bytes))
    packed = sum(counts) * 1002240
    assert (sum(counts), packed, structural, packed - structural) == (46, 46103040, 39521664, 6581376)
    absence_roots = [TASK / 'h200_mainline', TASK / 'h3o_operator_costs/codex_resume_20261031']
    argv = ['rg', '--files', '-g', '*promotion*', '-g', '*admission*', '-g', '!*.py', '-g', '!*.jsonl',
            *map(str, absence_roots)]
    listing = subprocess.run(argv, text=True, capture_output=True)
    assert listing.returncode in (0, 1), listing.stderr
    proof = {'schema_version': 1, 'captured_at_utc': datetime.now(timezone.utc).isoformat(),
             'main_head': subprocess.check_output(['git', '-C', str(MAIN), 'rev-parse', 'HEAD'], text=True).strip(),
             'sidecar_initial_head': initial['sidecar_initial_head'], 'sources': SOURCES, 'cases': cases,
             'payload': {'counts_source': 'GROUP_COUNTS', 'native_descriptor_source': 'NIXL',
                         'groups': [{'group': name, 'live_blocks': count, 'component_bytes_per_page': size,
                                     'packed_descriptor_bytes_per_page': 1002240}
                                    for name, count, size in zip(group_names, counts, page_bytes)],
                         'live_blocks': sum(counts), 'packed_descriptor_bytes_per_lane': packed,
                         'simulator_component_bytes_per_lane': structural, 'difference_bytes': packed - structural,
                         'relative_undercount': (packed - structural) / packed,
                         'qualification_limit': 'Packed registration is observed; T6144 request-level descriptor selection is source-derived, not a wire-byte observation.'},
             'receipt_presence_check': {'argv': argv, 'returncode': listing.returncode, 'matched_paths': listing.stdout.splitlines(),
                                        'limit': 'Only these two retained P2 evidence roots and original manifest bindings were inspected; no global absence claim.'}}
    (AREA / 'evidence/qualification_evidence.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({'capture': 'PASS', 'sources': len(SOURCES), 'rank_inventory_rows': 96,
                      'node_results': 12, 'startup_logs': 12, 'b29_metric_rows': 24}))


def verify():
    proof = json.loads((AREA / 'evidence/qualification_evidence.json').read_text())
    native_head = subprocess.check_output(['git', '-C', str(NATIVE), 'rev-parse', 'HEAD'], text=True).strip()
    assert native_head == PIN, native_head
    source_objects = []
    for source_id in ('NIXL', 'NATIVE_MEMORY', 'NATIVE_COUNTERS'):
        record = proof['sources'][source_id]
        relative = str(Path(record['path']).relative_to(NATIVE))
        original = subprocess.check_output(['git', '-C', str(NATIVE), 'show', f'{PIN}:{relative}'])
        assert original == Path(record['path']).read_bytes(), source_id
        source_objects.append({'source': source_id, 'git_object': f'{PIN}:{relative}',
                               'matches_captured_local_bytes': True})
    selectors = 0
    for source_id, record in proof['sources'].items():
        content = Path(record['path']).read_bytes()
        assert len(content) == record['bytes'] and hashlib.sha256(content).hexdigest() == record['sha256'], source_id
        for excerpt in record['excerpts']:
            selected = '\n'.join(content.decode().splitlines()[excerpt['first_line'] - 1:excerpt['last_line']])
            assert selected == excerpt['text'], source_id
            selectors += 1
        if record['pointers']:
            document = json.loads(content)
            for pointer in record['pointers']:
                selected = document
                if pointer != '/':
                    for key in pointer.lstrip('/').split('/'):
                        selected = selected[int(key)] if isinstance(selected, list) else selected[key.replace('~1', '/').replace('~0', '~')]
                selectors += 1
    rank_pointers = 0
    registration_rows = 0
    graph_memory_rows = 0
    for case in proof['cases'].values():
        for rank in case['ranks']:
            original = json.loads(Path(proof['sources'][rank['source']]['path']).read_text())['inventory']
            assert original['memory'] == rank['memory'] and original['workspace_bytes'] == rank['workspace_bytes']
            assert original['cache']['num_blocks'] == rank['num_blocks']
            assert original['effective_graph']['configured_mode'] == rank['graph_mode']
            scalars = original['memory']['native_scalars']
            assert scalars['requested_memory'] - scalars['available_kv_cache_memory_bytes'] == scalars['total_consumed'] + scalars['peak_activation_memory']
            rank_pointers += 4
        for log in case['startup_logs']:
            content = Path(proof['sources'][log['source']]['path']).read_text().splitlines()
            registrations = []
            graph_rows = []
            for row in log['rows']:
                assert ANSI.sub('', content[row['line'] - 1]) == row['text']
                if 'Registering packed KV cache:' in row['text']:
                    match = re.search(r'total_size=(\d+), block_stride=(\d+), num_blocks=(\d+), num_regions=(\d+)', row['text'])
                    assert match is not None, log['source']
                    values = tuple(map(int, match.groups()))
                    assert values == (case['native_blocks'] * 1002240, 1002240, case['native_blocks'], 1), log['source']
                    registrations.append(row)
                if 'Actual usage is' in row['text']:
                    match = re.search(r'and ([0-9.]+) GiB for CUDAGraph memory', row['text'])
                    assert match is not None and 2.35 <= float(match.group(1)) <= 2.38, log['source']
                    graph_rows.append(row)
            assert len(registrations) == len(graph_rows) == 8, log['source']
            registration_rows += len(registrations)
            graph_memory_rows += len(graph_rows)
    payload = proof['payload']
    assert sum(group['live_blocks'] for group in payload['groups']) == payload['live_blocks'] == 46
    assert sum(group['live_blocks'] * group['component_bytes_per_page'] for group in payload['groups']) == payload['simulator_component_bytes_per_lane'] == 39521664
    assert payload['packed_descriptor_bytes_per_lane'] == 46 * 1002240 == 46103040
    assert payload['difference_bytes'] == 6581376
    assert math.isclose(payload['relative_undercount'], 6581376 / 46103040, abs_tol=1e-14)
    metric_rows = 0
    for case in proof['cases'].values():
        terms = case['planner']['planner']
        predicted = (terms['requested_memory_bytes'] - terms['parameter_memory_per_device_bytes'] - terms['non_kv_cache_overhead_bytes']) // terms['physical_block_size_bytes']
        assert set(case['planner']['decode_lanes'].values()) == {predicted}
        assert case['planner_excess_blocks'] == predicted - case['native_blocks']
        assert math.isclose(case['planner_relative_error'], (predicted - case['native_blocks']) / case['native_blocks'], abs_tol=1e-14)
        for row in case['b29_metrics']:
            assert row['vllm'] != 0 and math.isfinite(row['frontier']) and math.isfinite(row['vllm'])
            assert math.isclose(row['relative_error'], abs(row['frontier'] - row['vllm']) / abs(row['vllm']), abs_tol=1e-14)
            metric_rows += 1
    report = (AREA / 'qualification_report.md').read_text()
    references = set(re.findall(r'`S:([^`]+)`', report))
    assert references and references.issubset(proof['sources']), sorted(references - proof['sources'].keys())
    criteria = [row for row in report.splitlines() if re.match(r'\| Q\d+ \|', row)]
    assert len(criteria) == 14 and all('`S:' in row for row in criteria)
    for relative in re.findall(r'\[[^\]]+\]\(([^)]+)\)', report):
        assert (AREA / relative.split('#', 1)[0]).is_file(), relative
    result = {'status': 'PASS', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
              'python_executable': sys.executable, 'python_version': sys.version,
              'environment': {'PYTHONDONTWRITEBYTECODE': os.environ.get('PYTHONDONTWRITEBYTECODE'), 'TMPDIR': os.environ.get('TMPDIR')},
              'source_hashes_pass': len(proof['sources']), 'source_excerpts_and_json_pointers_pass': selectors,
              'rank_inventory_field_pointers_pass': rank_pointers, 'arithmetic': 'PASS',
              'packed_registration_rows_pass': registration_rows, 'graph_memory_log_rows_pass': graph_memory_rows,
              'report_source_references_pass': len(references), 'report_criterion_rows_pass': len(criteria),
              'native_r4_cases': 2, 'native_rank_rows': 96, 'retained_b29_metric_rows': metric_rows,
              'native_source_pin': PIN, 'native_source_files_pass': source_objects,
              'qualification_claim': False, 'limits': ['No GPU, platform or cloud access.', 'No A6/A10 replay or new gate evaluation.',
                                                     'No independent wire trace, phase peak or promotion receipt was created.']}
    (AREA / 'evidence/verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true', help='Verify retained evidence without recapturing it.')
    arguments = parser.parse_args()
    verify() if arguments.verify else capture()
