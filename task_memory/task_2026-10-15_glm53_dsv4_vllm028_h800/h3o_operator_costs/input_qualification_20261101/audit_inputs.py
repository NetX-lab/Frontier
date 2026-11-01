"""Read retained P2 inputs; write a bounded qualification census, never run a simulator."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
REPO = TASK.parents[1]
RUNS = Path('/data/ycfeng/tmp/glm53_dsv4/two_fix_20261101/runs')
B29 = Path('/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def row_key(row):
    return row['case_id'], row['attention_variant'], str(row['plans_tile_schedule']).lower()


def query_key(mode, phase, padded, prefill, cached, decode):
    # Order does not change attention features; every request remains counted.
    if len(prefill) != len(cached):
        raise ValueError('Prefill query/context pairing is incomplete')
    paired = sorted(zip(prefill, cached))
    return (mode, phase, int(padded), tuple(q for q, _ in paired), tuple(c for _, c in paired), tuple(sorted(decode)))


def workload_key(coordinate):
    return query_key(coordinate['graph_runtime_mode'], coordinate['phase'], coordinate['padded_rows'],
                     coordinate['prefill_seq_lens'] or [], coordinate['prefill_kv_cache_sizes'] or [],
                     coordinate['decode_kv_cache_sizes'] or [])


def replay_queries(root):
    """Reconstruct request geometry from retained real forwards, checking every token.

    NIXL's first replay has cached prompt-1, then advances one cached token per
    occurrence. Completion records may reorder lanes, but a request stays on its
    lane and its own completions are ordered. Dummy forwards carry no request.
    """
    counts, queries, seen, prefill_seen = Counter(), Counter(), Counter(), Counter()
    with gzip.open(root / 'forwards.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            ids = row['request_ids']
            counts[(row['role'], row['mode'], row['step_class'])] += 1
            padded = row['descriptor']['padded_total_tokens']
            if not row['real']:
                if ids:
                    raise ValueError('Dummy forward has request IDs')
                key = query_key(row['mode'], 'dummy', padded, [], [], [])
            elif row['role'] == 'prefill':
                if len(ids) != 1 or row['local_tokens'] != 6144:
                    raise ValueError('Unexpected canonical prefill geometry')
                prefill_seen.update(ids)
                key = query_key(row['mode'], 'prefill', padded, [6144], [0], [])
            else:
                if len(set(ids)) != len(ids):
                    raise ValueError('Request duplicated inside decode forward')
                if row['local_tokens'] != len(ids):
                    raise ValueError('Decode is not one token per request')
                cached = [6143 + seen[request] for request in ids]
                flags = [seen[request] != 0 for request in ids]
                if row['step']['context_requests'] != flags.count(False):
                    raise ValueError('Reconstructed first tails disagree with recorded classifier')
                prefill = [1 for flag in flags if not flag]
                prefill_cached = [c for c, flag in zip(cached, flags) if not flag]
                decoding = [c for c, flag in zip(cached, flags) if flag]
                phase = 'first_tail' if prefill else 'decode'
                key = query_key(row['mode'], phase, padded, prefill, prefill_cached, decoding)
                seen.update(ids)
            queries[key] += 1
    if set(prefill_seen) != set(seen) or set(prefill_seen.values()) != {1}:
        raise ValueError('Prefill request population differs from decode')
    if len(seen) != 128 or set(seen.values()) != {1024}:
        raise ValueError(f'Incomplete request/token census: {dict(Counter(seen.values()))}')
    return queries, {'root': str(root), 'forwards_sha256': digest(root / 'forwards.jsonl.gz'),
                     'request_count': len(seen), 'decode_tokens_per_request': 1024,
                     'unique_attention_workloads': len(queries),
                     'forward_classes': [{'role': k[0], 'mode': k[1], 'step_class': k[2], 'count': v}
                                         for k, v in sorted(counts.items())]}


def independent_attention(phase, original_rows):
    """Compare exact original scope composition with both independent pass windows.

    This is a proposed cross-pass check, not the frozen workload-heldout gate.
    No train/test reassignment or production-table edit is performed.
    """
    from frontier.config.model_config import BaseModelConfig
    from v4_attention_phase_heldout import compose, model_layer_counts, statistics_of
    from v4_attention_heldout import build_p2_config, row_features
    from v4_attention_exact_errors import measured_lookup
    from frontier.execution_time_predictor.dsv4_predictor import (
        load_dsv4_attention_dataframe, train_dsv4_attention_models, DeepSeekV4AttentionPredictor)
    from frontier.types import ClusterType, MeasurementType
    scratch = Path('/data/ycfeng/tmp/codex-implement/65c06d69-e1a5-43c1-836f-440fe3bd1713/q1_config')
    config, binding = build_p2_config(scratch)
    cluster = config.get_clusters()[ClusterType.DECODE]
    replica = cluster.replica_config
    frame = load_dsv4_attention_dataframe(phase['input_table']['path'], replica.model_config,
        replica, cluster.replica_scheduler_config.block_size, MeasurementType.KERNEL_ONLY)
    assert len(frame) == len(original_rows)
    predictor = DeepSeekV4AttentionPredictor(train_dsv4_attention_models(frame, measured_lookup))
    manifest = load(TASK / 'h3o_operator_costs/a6_pass_replays_20261101/tables_manifest.json')
    frames = {}
    for name in ['pass1_all', 'pass2_all']:
        record = manifest['tables'][name]
        assert digest(record['path']) == record['sha256']
        with open(record['path'], newline='') as stream:
            rows = list(csv.DictReader(stream))
        frames[name] = {row_key(r): r for r in rows}
        reference = {row_key(r): r for r in original_rows}
        assert len(rows) == len(frames[name]) == len(original_rows) == 7105
        assert set(frames[name]) == set(reference)
        identity_columns = ['phase', 'attention_variant', 'compress_ratio', 'engine_cudagraph_mode',
            'graph_runtime_mode', 'padded_rows', 'prefill_seq_lens', 'prefill_kv_cache_sizes',
            'decode_kv_cache_sizes', 'model_name', 'attention_backend', 'attention_kv_cache_dtype',
            'num_tensor_parallel_workers', 'kv_cache_dtype', 'multi_stream_gemm_token_threshold',
            'operator_scopes', 'device_name', 'device_capability', 'model_config_sha256', 'weight_source']
        for key, other in frames[name].items():
            for column in identity_columns:
                assert reference[key][column] == other[column], (name, key, column)
            window = float(other['layer_window_stats.median'])
            assert math.isfinite(window) and window > 0
    model = BaseModelConfig.create_from_name('deepseek-ai/DeepSeek-V4-Flash')
    weights = model_layer_counts(model)
    output = []
    for omission in phase['attention']['omissions']:
        members = [frame.loc[i] for i in omission['row_indices'] + omission['missing_row_indices']]
        assert len(members) == 5
        predicted = measured_original = 0.0
        windows = Counter()
        for raw in members:
            row = raw
            scopes = list(json.loads(raw['operator_scopes']))
            prediction = predictor.scope_times(row_features(row), variant=row.attention_variant,
                graph_runtime_mode=row.graph_runtime_mode, plans_tile_schedule=bool(row.plans_tile_schedule),
                scopes=scopes)
            costs, gaps = prediction.cost_ms, prediction.gap_ms
            threshold = int(float(raw['multi_stream_gemm_token_threshold']))
            weight = weights[(raw['attention_variant'], row['plans_tile_schedule'])]
            predicted += weight * compose(row, costs, gaps, threshold)
            measured_original += weight * float(raw['layer_window_stats.median'])
            for name, rows in frames.items():
                other = rows[row_key(raw)]
                windows[name] += weight * float(other['layer_window_stats.median'])
        output.append({'unit_id': omission['unit_id'], 'class': list(workload_key(omission['workload'])[:2]),
                       'original_exact_composition_us': predicted * 1000,
                       'original_measured_window_us': measured_original * 1000,
                       'independent_pass_windows_us': {n: w * 1000 for n, w in windows.items()},
                       'signed_errors': {n: (predicted - w) / w for n, w in windows.items()}})
    grouped = defaultdict(list)
    for item in output:
        grouped[tuple(item['class'])].extend(item['signed_errors'].values())
    return {'classification': 'PROPOSED_CROSS_PASS_EXACT_POINT_CHECK_NOT_FORMAL_A6',
            'production_lookup': 'train_dsv4_attention_models + _build_exact_feature_lookup + scope_times',
            'exact_layer_queries': len(output) * 5, 'interpolated_queries': 0,
            'independent_pass_identity_columns_checked': identity_columns,
            'rows_per_independent_pass': 7105, 'positive_windows_per_independent_pass': 7105,
            'config_binding': binding,
            'table_records': {n: manifest['tables'][n] for n in frames}, 'workloads': output,
            'by_class': [{'class': list(k), **statistics_of([abs(v) for v in values], values),
                          'maximum_absolute_error': max(map(abs, values))}
                         for k, values in sorted(grouped.items())]}


def main(output):
    phase = load(TASK / 'h3o_operator_costs/a6_forward_attention_20261101/current_phase_heldout.json')
    with open(phase['input_table']['path'], newline='') as stream:
        table = list(csv.DictReader(stream))
    assert digest(phase['input_table']['path']) == phase['input_table']['sha256']
    from frontier.config.model_config import BaseModelConfig
    from v4_attention_phase_heldout import aggregate_forward_attention
    a6 = load(phase['input_result'])
    aggregated = aggregate_forward_attention(phase['rows'], BaseModelConfig.create_from_name(
        'deepseek-ai/DeepSeek-V4-Flash'), a6['attention']['fold_assignment'])
    assert aggregated == phase['attention']
    skip = {r['row_index']: r for r in phase['coverage_skips']}
    queries = {}
    result = {'classification': 'P2_INPUT_QUALIFICATION_RETAINED_CPU_AUDIT', 'checks': [],
              'inputs': [], 'runs': [], 'omissions': [], 'formal_attention': {
                  k: aggregated[k] for k in ['attention_verdict', 'complete_workloads', 'omitted_workloads',
                                            'partially_evaluable_workloads', 'fully_unevaluable_workloads', 'by_group']}}
    result['checks'].append({'name': 'frozen_forward_reaggregation', 'status': 'PASS', 'rows': len(phase['rows'])})
    for case in ['p2_32', 'p2_64']:
        queries[case], census = replay_queries(RUNS / case)
        baseline_queries, _ = replay_queries(B29 / f'{case}_warm_k19b29')
        assert queries[case] == baseline_queries
        receipt = load(RUNS / case / 'receipt.json')
        baseline = load(B29 / f'{case}_warm_k19b29' / 'receipt.json')
        assert receipt['status'] == baseline['status'] == 'FINISHED'
        signature = lambda rows: sorted((Path(r['path']).name, r['sha256'], r['rows'], tuple(r['consumers'])) for r in rows)
        assert signature(receipt['profiling_tables']) == signature(baseline['profiling_tables'])
        assert receipt['trace'] == baseline['trace']
        assert receipt['input_sha256'] == baseline['input_sha256'] == digest(receipt['input'])
        assert digest(receipt['trace']['path']) == receipt['trace']['sha256']
        for record in receipt['profiling_tables']:
            assert digest(record['path']) == record['sha256']
            with open(record['path'], newline='') as stream:
                assert len(list(csv.DictReader(stream))) == record['rows']
        config = load(Path(receipt['metrics_dir']) / 'config.json')
        census.update(receipt_sha256=digest(RUNS / case / 'receipt.json'),
                      config_sha256=digest(Path(receipt['metrics_dir']) / 'config.json'),
                      input=receipt['input'], input_sha256=receipt['input_sha256'], trace=receipt['trace'],
                      source_commit=receipt['source_commit'], kv_admission=receipt['kv_admission'],
                      kv_transfer_config=config['kv_cache_transfer_config'],
                      a1=receipt['a1'])
        result['runs'].append(census)
        result['inputs'] = receipt['profiling_tables']
        result['checks'].append({'name': case + '_formal_input_hashes_and_query_equality', 'status': 'PASS',
                                 'tables': 7, 'requests': 128, 'tokens_per_request': 1024})
    for omission in aggregated['omissions']:
        missing = omission['missing_row_indices']
        timings = [float(table[i]['layer_window_stats.median']) for i in missing]
        result['omissions'].append({**omission,
            'measured_source_rows_present': len(missing), 'missing_measured_layer_windows': sum(v <= 0 for v in timings),
            'heldout_rejections': [skip[i] for i in missing],
            'exact_workload_forward_queries': {case: queries[case][workload_key(omission['workload'])] for case in queries},
            'status': 'NEEDS_DECISION',
            'missing_evidence': 'Frozen workload-heldout prediction is inadmissible; measured rows/windows exist. '
                                'A cross-pass exact-point rule is proposed, not applied.'})
    result['independent_attention'] = independent_attention(phase, table)
    prior = load(TASK / 'h3o_operator_costs/qualification_codex_20261031/evidence/qualification_evidence.json')
    verified, identities = [], []
    for case, facts in prior['cases'].items():
        for item in facts['ranks'] + facts['nodes']:
            record = prior['sources'][item['source']]
            assert digest(record['path']) == record['sha256']
            assert Path(record['path']).stat().st_size == record['bytes']
            verified.append(item['source'])
        for item in facts['ranks']:
            s = item['memory']['native_scalars']
            assert s['requested_memory'] - s['available_kv_cache_memory_bytes'] == s['total_consumed'] + s['peak_activation_memory']
            assert item['num_blocks'] == facts['native_blocks']
        identities.append({'case': case, 'native_blocks': facts['native_blocks'], 'ranks': len(facts['ranks']),
                           'nodes': len(facts['nodes']), 'r4_status': facts['original_r4_status'],
                           'original_authorization': facts['original_authorization'],
                           'graph_summary': facts['graph_summary'], 'planner': facts['planner'],
                           'planner_relative_error': facts['planner_relative_error'],
                           'existing_f4_comparison': facts['existing_f4_comparison']})
    result['native_identity_and_capacity'] = identities
    result['checks'].append({'name': 'native_inventory_and_memory_identity', 'status': 'PASS',
                             'hashed_node_rank_files': len(verified), 'memory_identities': 96})
    result['status'] = 'PASS_EVIDENCE_CHECKS_NOT_FORMAL_ACCEPTANCE'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'checks': result['checks'],
                     'queried_omissions': {case: sum(x['exact_workload_forward_queries'][case] > 0 for x in result['omissions'])
                                           for case in queries},
                     'cross_pass_classes': result['independent_attention']['by_class']}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
