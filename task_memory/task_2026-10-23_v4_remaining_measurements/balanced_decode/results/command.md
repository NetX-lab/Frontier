# Balanced q1 CPU audit commands

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Independent immutable CPU verification artifacts. |

Task: `W128-BALANCED-DECODE-DATA`. Working directory: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. No conda activation, Torch/vLLM import, network or platform operation. Each CPU analysis has an actual systemd 30 GiB memory cgroup and 60-second timeout. Raw imports the pinned package snapshot and changes only its process cwd.

The inspected environment is Python 3.10.6; numpy/scipy reside in `/data/ycfeng/envs/frontier-py310/lib/python3.10/site-packages/`. The trace parser and finalizer use `/usr/bin/python3`, also Python 3.10.6, with stdlib only. `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` apply to raw/CSV reconstruction. No dependency install. Trace resource planning inspected the 1,500,192,771-byte file header and supported loader/aggregate complexity; MemAvailable was 121,101,300 KiB. Full trace load took 30.251 s; complete audit took 49.260 s and peak RSS 6,062,156 KiB.

## Exact execution commands

```bash
sudo -n systemd-run --scope --quiet -p MemoryMax=30G timeout 60s /usr/bin/time -v runuser -u brainpp -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_balanced_decode_data_audit.py raw > /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_balanced_decode_raw_audit.log 2>&1
sudo -n systemd-run --scope --quiet -p MemoryMax=30G timeout 60s /usr/bin/time -v runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_balanced_decode_data_audit.py trace > /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_balanced_decode_trace_audit.log 2>&1
sudo -n systemd-run --scope --quiet -p MemoryMax=30G timeout 60s runuser -u brainpp -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_balanced_decode_data_audit.py raw > /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_balanced_decode_raw_audit_r1.log 2>&1
sudo -n systemd-run --scope --quiet -p MemoryMax=30G timeout 60s runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_balanced_decode_data_audit.py trace > /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_balanced_decode_trace_audit_r1.log 2>&1
sudo -n systemd-run --scope --quiet -p MemoryMax=30G timeout 60s runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_balanced_decode_finalize.py > /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_balanced_decode_finalize.log 2>&1
```

The first pair exited 127 before running Python: `/usr/bin/timeout: failed to run command ‘/usr/bin/time’: No such file or directory`. Removed only the unavailable timing binary; both 60-second/30 GiB caps remained. Scripts record monotonic elapsed time and `resource.getrusage().ru_maxrss`. The second pair passed. No data repair or rerun of GPU measurement.

## Why a task-local adapter is necessary

The reused warmup auditor hard-codes a historical package, job, routing and paths in `main`; only its existing `scan_raw`, `audit_tables`, and `statistics_for` are imported unchanged. Current routing checks additionally compare the exact flat strategy origin/SHA, source-defined nine native modules and six FFN router/runner identities. The existing `tests/analysis/v4_kernel_coverage.py` CLI requires a different accepted-prefill receipt schema/native inputs and also computes nested scope sums, forbidden here. This adapter calls its underlying supported retained DSA parser/aggregate directly and limits the extra census to ownership/correlation counts. No new maintained tool.

## Full temporary script: `/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_balanced_decode_data_audit.py`

SHA-256: `14ca5c195f07866856ff275631fd715fc147133bde8082a4935db2cfdd3cf39a`.

```python
"""Task-scoped CPU audit; retained inputs are never modified."""
import argparse
import bisect
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

ROOT = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028')
TASK = ROOT / 'task_memory/task_2026-10-23_v4_remaining_measurements/balanced_decode'
TEMP = Path('/data/ycfeng/tmp/opencode-v2-tmp/opencode')
DOWNLOAD = Path('/data/ycfeng/tmp/glm53_dsv4/balanced_decode02_results_6zkg5vz4/download_receipt.json')
RESULTS = TASK / 'results'
ANALYZER = ROOT / 'task_memory/task_2026-10-21_v4_q1_warm_remeasure/observed/analyze_warmup_results.py'
TASK_ID = 'W128-BALANCED-DECODE-DATA'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def binding():
    receipt = load(DOWNLOAD)
    package_binding = load(TASK / 'package_binding.json')
    package = Path(package_binding['package_path'])
    local = Path(receipt['local_root'])
    phase = local / 'Q1_WARM_6143_7166_BALANCED'
    assert phase.is_dir()
    return receipt, package_binding, package, local, phase


def timing(start):
    return dict(seconds=time.monotonic() - start,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                python_executable=sys.executable, python_version=sys.version,
                finished_utc=datetime.now(timezone.utc).isoformat())


def raw_audit():
    start = time.monotonic()
    receipt, bound, package, local, phase = binding()
    manifest_sha = sha(package / 'manifest.json')
    assert manifest_sha == bound['manifest_sha256'] == load(TASK / 'root_seal.json')['manifest_sha256']
    assert sha(TASK / 'package_plan.json') == bound['package_plan_sha256']
    config = load(package / 'config.json')
    snapshot = Path(config['source_snapshot'])
    source = bound['source_revision']
    assert snapshot == package / 'source' / source
    before = []
    for item in receipt['files']:
        path = local / item['path']
        actual = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
        assert actual['bytes'] == item['bytes'] and actual['sha256'] == item['sha256'], path
        before.append(actual)
    assert len(before) == 8 and sum(x['bytes'] for x in before) == 1601865978
    sys.path.insert(0, str(snapshot))
    os.chdir(snapshot)
    from frontier.profiling.attention.dsv4_attention_main import build_parser, build_plan
    from frontier.profiling.attention.dsv4_attention_plan import expected_scopes, scope_operators
    from frontier.profiling.attention.dsv4_attention_rows import build_dsv4_rows, routing_columns
    from frontier.profiling.attention.dsv4_attention_runner import ROUTING_NATIVE_MODULES
    from frontier.profiling.attention.dsa_vllm_runner import _case_window
    from frontier.profiling.common.runtime_identity import RuntimeIdentity, attach_identity_columns
    from frontier.profiling.common.model_config import ModelConfig
    analyzer = module('retained_warmup_analyzer', ANALYZER)
    plan = build_plan(build_parser().parse_args(config['profile_args']))
    assert json.loads(json.dumps(plan.to_json())) == load(package / 'plan.json')
    case, = plan.cases_for_engine(0)
    assert (case.case_id, case.phase, case.batch_size, case.prompt_len, case.warmup, case.measured) == (
        'e0-decode-q1-c6143-7166', 'decode', 1, 6141, 2, 1024)
    assert _case_window(case, plan.engine(0)) == (2, 1024)
    engine, steps, calls, series, raw = analyzer.scan_raw(phase, 6141, expected_scopes)
    assert raw['call_counts'] == {'CUDA_EVENT': 51200, 'KERNEL_ONLY': 45056}
    assert all(v['requests'][0]['num_computed_tokens'] not in (6141, 6142) for v in steps.values())
    assert {key[2] for key in steps if key[0] == 'CUDA_EVENT'} == set(range(1024))
    assert {key[2] for key in steps if key[0] == 'KERNEL_ONLY'} == set(range(1024, 2048))
    assert set(calls) == {(method, context, layer, scope)
        for method in analyzer.METHOD_FILES for context in analyzer.CONTEXTS
        for layer, (variant, _) in analyzer.LAYER_ROLES.items()
        for scope in expected_scopes(variant, True, method == 'KERNEL_ONLY')}
    routing = engine['moe_routing']
    routing_fields = routing_columns(routing)
    strategy = package / 'balanced_routing.py'
    assert routing['origin'] == str(strategy) and routing['sha256'] == sha(strategy)
    assert routing['sha256'] == bound['manifest_files']['balanced_routing.py'] == config['tool_hashes'][str(strategy)]
    assert routing['environment'] == routing['runtime_path'] == config['moe_gating_routing_runtime_path'] == 'balanced_topk'
    expected_routers = {f'model.layers.{layer}.ffn.experts' for layer in analyzer.LAYER_ROLES}
    assert set(routing['routers']) == set(routing['runners']) == expected_routers
    expected_native = {name.replace('.', '/') + '.py' for name in ROUTING_NATIVE_MODULES}
    assert set(routing['native_sources']) == expected_native
    for name, identity in routing['native_sources'].items():
        assert identity == dict(origin=str(Path(config['native_package']) / name.removeprefix('vllm/')),
                                sha256=config['expected_source_hashes'][name])
    assert engine['use_v2_model_runner'] and engine['engine_policy'] == 'eager_chunked'
    assert (engine['num_local_heads'], engine['padded_heads'], engine['num_local_groups']) == (8, 64, 1)
    assert [(v['global_layer_id'], v['attention_variant']) for v in engine['layers']] == [
        (layer, role[0]) for layer, role in analyzer.LAYER_ROLES.items()]
    summary = load(phase / 'raw/dsv4_aggregate_summary.json')
    assert summary['missing_engines'] == [] and summary['plan'] == json.loads(json.dumps(plan.summary()))
    identity = RuntimeIdentity(runtime_versions=engine['runtime_versions'], source_revision=source,
        model_config_revision=load(TASK / 'package_plan.json')['model_config_revision'],
        model_config_sha256=sha(package / 'model/config.json'), device_name=engine['device_name'],
        device_capability=engine['device_capability'], schema_version='dsv4_attention.v1',
        quant_signature=ModelConfig.from_model_name(plan.shape.model_name).get_quant_signature(),
        parallel_groups={'engine_tensor_parallel': 1, 'target_tensor_parallel': 8}, graph_mode='eager')
    assert identity.to_json() == summary['identity']
    tables, rows = analyzer.audit_tables(phase, source, calls, expected_scopes, scope_operators, identity.to_json())
    routing_cells = 0
    for method, selected in rows.items():
        for row in selected:
            assert row['routing_runtime_path'] == routing_fields['routing_runtime_path']
            assert row['routing_strategy_sha256'] == routing_fields['routing_strategy_sha256']
            assert row['moe_routing'] == routing_fields['moe_routing'] and json.loads(row['moe_routing']) == routing
            routing_cells += 3
        assert summary['outputs'][method]['rows'] == len(selected)
    records = [engine, *steps.values(), *calls.values()]
    rebuilt = build_dsv4_rows(plan, records)
    reproduction = {}
    for method, selected in rebuilt.items():
        attached = attach_identity_columns(selected, identity)
        columns = list(dict.fromkeys(name for row in attached for name in row))
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=columns, restval='')
        writer.writeheader()
        writer.writerows(attached)
        regenerated = buffer.getvalue().encode()
        assert regenerated == analyzer.csv_path(phase, method).read_bytes()
        reproduction[method] = dict(rows=len(selected), sha256=hashlib.sha256(regenerated).hexdigest(),
                                    every_cell_and_csv_bytes_exact=True)
    all_stats = []
    for (method, layer, scope), points in sorted(series.items()):
        item = dict(measurement=method, layer=layer, variant=analyzer.LAYER_ROLES[layer][0],
                    plans_tile_schedule=analyzer.LAYER_ROLES[layer][1], scope=scope,
                    **analyzer.statistics_for(points))
        if method == 'KERNEL_ONLY':
            item['kernel_count_distribution'] = dict(sorted(Counter(
                calls[(method, context, layer, scope)]['kernel_count'] for context, _ in points).items()))
        all_stats.append(item)
    csv_shape = {}
    for method in rows:
        with analyzer.csv_path(phase, method).open(newline='') as stream:
            reader = csv.DictReader(stream)
            assert len(reader.fieldnames) == len(set(reader.fieldnames))
            parsed = list(reader)
            assert all(None not in row and all(value is not None for value in row.values()) for row in parsed)
            csv_shape[method] = dict(columns=len(reader.fieldnames), malformed_rows=0, duplicate_columns=0)
    module_hashes = []
    for name, imported in sorted(sys.modules.items()):
        path = getattr(imported, '__file__', None)
        if name.startswith('frontier.') and path and path.endswith('.py'):
            path = Path(path)
            assert path.is_relative_to(snapshot)
            relative = str(path.relative_to(package))
            assert sha(path) == bound['manifest_files'][relative]
            module_hashes.append(dict(module=name, path=str(path), sha256=sha(path)))
    assert not any(name in sys.modules for name in ('torch', 'vllm'))
    after = [{**item, 'sha256_after': sha(item['path'])} for item in before]
    assert all(item['sha256_after'] == item['sha256'] for item in after)
    output = dict(task_id=TASK_ID, job=receipt['job'], status='PASS_CPU_RAW_TABLE_IDENTITY_CHECKS',
        source_revision=source, builder_revision=bound['builder_commit'],
        merged_revision=load(TASK / 'root_seal.json')['merged_commit'], manifest_sha256=manifest_sha,
        package=str(package), local_root=str(local), phase_root=str(phase),
        download_receipt=dict(path=str(DOWNLOAD), sha256=sha(DOWNLOAD), cloud_checks='ROOT_RECEIPT_NOT_REQUERIED'),
        inputs_before_after=after, imported_source_hashes=module_hashes,
        reused_analyzer=dict(path=str(ANALYZER), sha256=sha(ANALYZER),
            functions=['scan_raw', 'audit_tables', 'statistics_for'], hard_coded_main_invoked=False),
        raw=raw, tables=tables, csv_shape=csv_shape, byte_exact_reproduction=reproduction,
        routing=dict(engine=routing, checked_csv_rows=12288, checked_csv_cells=routing_cells,
            exact_flat_strategy_and_nine_native_origins=True,
            source_evidence=['dsv4_attention_runner.py:192-244', 'dsv4_attention_rows.py:57-76',
                             'balanced_routing.py:11-35,38-72', 'official_model.py:691-709,840-846'],
            observed='Constructed router/runner, strategy and source identities; exact same identity in every CSV row.',
            not_observed='No routed expert-ID/weight tensors retained; numerical dispatch balance not independently checked.'),
        warmup=dict(plan_skip=2, raw_prompt_len=6141, measured_context_range=[6143,7166],
            excluded_cached_contexts=[6141,6142], measured_steps_per_method=1024,
            evidence='Pinned case window and recorder skip semantics plus exact raw contexts; skipped steps have no raw IDs.'),
        runtime_identity=identity.to_json(), engine=engine,
        per_scope_statistics=all_stats, execution=timing(start),
        predicates={name:'PASS' for name in ['eight_original_hashes', 'exact_q1_context_layer_scope_keys',
            'no_missing_duplicate_or_orphan_calls', 'no_malformed_csv_keys', 'positive_finite_call_costs',
            'positive_integer_kernel_counts', 'every_scope_statistic_from_raw', 'csv_bytes_reproduced',
            'balanced_strategy_origin_sha_and_all_rows', 'nine_native_identity_origins_and_pins',
            'two_unmeasured_warmups_source_and_raw_window']},
        limits=['Only physical TP1/eager/six-layer isolated attention measurements.',
            'target TP8 per-rank shape labels are not physical TP8 evidence.',
            'Single retained sample per context/layer/scope; no stationarity/repetition qualification.',
            'CUDA_EVENT and KERNEL_ONLY are separate instrumented passes, not identical observations.',
            'No nested scope addition, CPU gap, serving total or routing-variant equivalence computed.',
            'Root owns platform/package/native constructor evidence and all admissions.'])
    (TEMP / 'w128_balanced_decode_raw_audit.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=output['status'], raw=raw, tables=tables, scope_series=len(all_stats),
                         routing_rows=12288, execution=output['execution']), indent=2))


def trace_audit():
    start = time.monotonic()
    _, bound, package, _, phase = binding()
    parser_path = package / 'source' / bound['source_revision'] / 'frontier/profiling/attention/dsa_kernel_trace.py'
    assert sha(parser_path) == bound['manifest_files'][str(parser_path.relative_to(package))]
    parser = module('retained_dsa_trace_parser', parser_path)
    trace_path = phase / 'profiler_traces/e0-decode-q1-c6143-7166.json'
    trace_sha = sha(trace_path)
    assert trace_sha == 'd1207a5e3879202fc5c41cedf897495a62d6c2c09cd03935fb7a5a58ed105855'
    events = parser.load_trace_events(trace_path)
    load_seconds = time.monotonic() - start
    print(json.dumps(dict(stage='trace_loaded', events=len(events), elapsed_seconds=load_seconds,
                         peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True)
    aggregate = parser.aggregate_kernel_trace(events)
    raw_calls = {}
    with (phase / 'raw/dsv4_raw_e0.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] == 'call' and row['measurement'] == 'KERNEL_ONLY':
                key = parser.ScopeKey(row['step_index'], row['global_layer_id'], row['scope'])
                assert key not in raw_calls
                raw_calls[key] = row
    assert set(aggregate) == set(raw_calls) and len(aggregate) == 45056
    for key, value in aggregate.items():
        assert value.kernel_count == raw_calls[key]['kernel_count']
        assert value.kernel_time_ms == raw_calls[key]['time_ms']
    ranges, launches, parents = defaultdict(list), defaultdict(list), Counter()
    categories, sync_names, sync_fields = Counter(), Counter(), set()
    annotation_keys = Counter()
    kernel_fields = set()
    for index, event in enumerate(events):
        category = event.get('cat')
        categories[category] += 1
        if category == 'user_annotation':
            key = parser.parse_annotation_name(event.get('name'))
            if key is not None:
                annotation_keys[key] += 1
                ranges[parser._thread(event)].append((float(event['ts']), float(event['ts'])+float(event['dur']), key))
        elif category in ('cuda_runtime','cuda_driver'):
            correlation = event.get('args', {}).get('correlation')
            launches[correlation].append((index,event))
        elif category == 'cpu_op':
            parents[event.get('args', {}).get('External id')] += 1
        if event.get('name') in ('cudaEventRecordWithFlags','cudaStreamWaitEvent','cudaEventSynchronize','cudaDeviceSynchronize'):
            sync_names[event['name']] += 1
            sync_fields.update(event.get('args', {}))
    assert set(annotation_keys) == set(raw_calls) and set(annotation_keys.values()) == {1}
    starts = {}
    for thread, values in ranges.items():
        values.sort(key=lambda value:value[0])
        assert all(right[0] >= left[1] for left,right in zip(values, values[1:]))
        starts[thread] = [value[0] for value in values]
    first = min(value[0] for values in ranges.values() for value in values)
    last = max(value[1] for values in ranges.values() for value in values)
    def owner(launch):
        thread = parser._thread(launch)
        position = bisect.bisect_right(starts.get(thread, []), launch['ts'])-1
        if position >= 0 and launch['ts'] <= ranges[thread][position][1]:
            return ranges[thread][position][2]
        return None
    scoped, regions, streams, missing_parent, kernel_launch_cardinality = Counter(), Counter(), Counter(), 0, Counter()
    unowned_samples = []
    for index,event in enumerate(events):
        if event.get('cat') != 'kernel':
            continue
        assert math.isfinite(event['dur']) and event['dur'] > 0
        args = event.get('args', {})
        kernel_fields.update(args)
        matches = launches.get(args.get('correlation'), [])
        kernel_launch_cardinality[len(matches)] += 1
        streams[str(args.get('stream'))] += 1
        missing_parent += not bool(parents.get(args.get('External id')))
        key = owner(matches[0][1]) if len(matches) == 1 else None
        if key is not None:
            scoped[key] += 1
        else:
            stamp = matches[0][1]['ts'] if len(matches) == 1 else None
            region = ('no_unique_launch' if stamp is None else 'before_first_measured_scope' if stamp < first
                      else 'after_last_measured_scope' if stamp > last else 'between_first_and_last_measured_scope')
            regions[region] += 1
            if len(unowned_samples) < 12:
                unowned_samples.append(dict(trace_event_index=index, region=region, event=event))
    assert set(scoped) == set(aggregate)
    assert all(scoped[key] == value.kernel_count for key,value in aggregate.items())
    assert sha(trace_path) == trace_sha
    output = dict(task_id=TASK_ID, status='PASS_EXACT_KERNEL_AGGREGATION_COVERAGE_LIMITED',
        trace=dict(path=str(trace_path), bytes=trace_path.stat().st_size, sha256=trace_sha, sha256_after=trace_sha),
        parser=dict(path=str(parser_path), sha256=sha(parser_path),
                    functions=['load_trace_events','parse_annotation_name','aggregate_kernel_trace']),
        census=dict(events=len(events), categories=dict(categories),
            annotations=len(annotation_keys), duplicate_annotations=0,
            raw_kernel_only_calls_reproduced=len(aggregate),
            scoped_unique_kernels=sum(scoped.values()), unowned_kernels=sum(regions.values()),
            unowned_regions=dict(regions), kernel_launch_cardinality=dict(kernel_launch_cardinality),
            kernels_without_external_cpu_parent=missing_parent, kernel_streams=dict(streams),
            sync_calls=dict(sync_names), kernel_arg_fields=sorted(kernel_fields), sync_arg_fields=sorted(sync_fields)),
        raw_time_and_count_comparison=dict(exact_scope_keys=True, every_time_exact=True, every_kernel_count_exact=True),
        unowned_kernel_samples=unowned_samples,
        predicates=dict(exact_45056_annotations_and_raw_calls='PASS', duplicate_malformed_annotations='PASS_ZERO',
                        scope_kernel_counts_times_exact='PASS', all_kernels_have_unique_launch='PASS' if set(kernel_launch_cardinality)=={1} else 'FAIL',
                        all_trace_kernels_owned_by_measured_scope='PASS' if not regions else 'FAIL',
                        complete_attention_or_forward_ownership='UNVERIFIED', actual_event_handle_dependency_edges='UNVERIFIED'),
        limits=['Uniqueness of correlation joins does not prove trace capture completeness.',
            'Unowned kernels include unmeasured seed/warmup/prime/FFN/norm/metadata work; names or envelope positions do not assign it.',
            'No enclosing measured attention/forward annotations or retained request/warmup IDs.',
            'Synchronization call counts alone do not establish actual event-handle edges.',
            'No full serving topology, GPU critical path or CPU gap reconstructed.'],
        load_seconds=load_seconds, execution=timing(start))
    assert not any(name in sys.modules for name in ('torch','vllm'))
    (TEMP / 'w128_balanced_decode_trace_audit.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=output['status'],census=output['census'],execution=output['execution']),indent=2))


if __name__ == '__main__':
    assert sys.flags.dont_write_bytecode
    args = argparse.ArgumentParser()
    args.add_argument('mode', choices=['raw','trace'])
    mode = args.parse_args().mode
    (raw_audit if mode == 'raw' else trace_audit)()
```

## Full temporary script: `/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_balanced_decode_finalize.py`

SHA-256: `51de7b9b785b21486e2dff779e09617ab268f88174cec47d160e78beddefbf4f`.

```python
"""Generate the three owner-scoped audit artifacts from completed CPU receipts."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
from datetime import datetime, timezone

TEMP = Path('/data/ycfeng/tmp/opencode-v2-tmp/opencode')
TASK = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/balanced_decode')
OUT = TASK / 'results'
SCRIPT = TEMP / 'w128_balanced_decode_data_audit.py'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def reject_duplicate_fields(pairs):
    result = {}
    for name, value in pairs:
        assert name not in result, f'Duplicate JSON field: {name}'
        result[name] = value
    return result


assert sys.flags.dont_write_bytecode
start = time.monotonic()
raw = json.loads((TEMP / 'w128_balanced_decode_raw_audit.json').read_text())
trace = json.loads((TEMP / 'w128_balanced_decode_trace_audit.json').read_text())
records = 0
with (Path(raw['phase_root']) / 'raw/dsv4_raw_e0.jsonl').open() as stream:
    for line in stream:
        row = json.loads(line, object_pairs_hook=reject_duplicate_fields)
        assert isinstance(row, dict)
        records += 1
assert records == 98305
for item in raw['inputs_before_after']:
    item['sha256_final'] = sha(item['path'])
    assert item['sha256_final'] == item['sha256']
raw['predicates']['duplicate_raw_json_fields'] = 'PASS_ZERO_IN_98305_RECORDS'
raw['status'] = 'PASS_SCOPED_CPU_DATA_CHECKS_TRACE_TOTAL_COVERAGE_UNQUALIFIED'
raw['Modification History'] = [{'date':'2026-10-23', 'change':'Independent immutable balanced q1 raw/table/routing/trace CPU audit.'}]
raw['trace_audit'] = trace
raw['script_sha256'] = sha(SCRIPT)
raw['finalizer_sha256'] = sha(Path(__file__))
raw['admissions'] = {'data':False, 'root':False, 'native':False, 'serving':False, 'physical_TP8':False}
raw['actions'] = {'platform':0, 'cloud':0, 'network':0, 'Docker':0, 'Git':0,
                  'source_or_input_edits':0, 'delegation':0, 'outlier_deletion':0,
                  'nested_scope_sums':0, 'CPU_gap_computation':0, 'serving_total_computation':0}
raw['phase_checkpoint'] = {'scoped_raw_table_routing':'PASS', 'bounded_trace_inspection':'PASS',
    'three_artifact_delivery':'PASS_AFTER_FINAL_OUTPUT_CHECK',
    'D1':'CLEAR','D2':'CLEAR','D3':'CLEAR','D4':'CLEAR','D5':'CLEAR','D6':'CLEAR','D7':'CLEAR',
    'S1_to_S6':'No defect in the assigned CPU verification slice; root records retain root-owned work.',
    'route':'Publish only three assigned local artifacts; return scoped facts and unresolved root admission checks.'}
raw['finalization'] = dict(seconds=time.monotonic()-start,
    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    python_executable=sys.executable, python_version=sys.version,
    finished_utc=datetime.now(timezone.utc).isoformat(), duplicate_raw_json_fields=0,
    raw_json_records_checked=records, original_byte_hashes_retained=True)
raw['execution_records'] = []
for name, code in [('raw_audit.log',127),('trace_audit.log',127),('raw_audit_r1.log',0),('trace_audit_r1.log',0)]:
    path = TEMP / ('w128_balanced_decode_'+name)
    raw['execution_records'].append(dict(path=str(path), sha256=sha(path), exit_code=code))
OUT.mkdir(exist_ok=True)
assert not any((OUT/name).exists() for name in ('data_audit.json','command.md','test_report_2026-10-23_data.md'))
(OUT/'data_audit.json').write_text(json.dumps(raw,indent=2,allow_nan=False)+'\n')
prefix = 'sudo -n systemd-run --scope --quiet -p MemoryMax=30G timeout 60s '
command_base = prefix + 'runuser -u brainpp -- env '
raw_cmd = command_base + 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B '+str(SCRIPT)+' raw'
trace_cmd = command_base + 'PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B '+str(SCRIPT)+' trace'
final_cmd = command_base + 'PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B '+str(Path(__file__))
header = '## Modification History\n\n| Date | Change |\n| --- | --- |\n| 2026-10-23 | Independent immutable CPU verification artifacts. |\n\n'
command = '# Balanced q1 CPU audit commands\n\n'+header
command += 'Task: `W128-BALANCED-DECODE-DATA`. Working directory: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. No conda activation, Torch/vLLM import, network or platform operation. Each CPU analysis has an actual systemd 30 GiB memory cgroup and 60-second timeout. Raw imports the pinned package snapshot and changes only its process cwd.\n\n'
command += 'The inspected environment is Python 3.10.6; numpy/scipy reside in `/data/ycfeng/envs/frontier-py310/lib/python3.10/site-packages/`. The trace parser and finalizer use `/usr/bin/python3`, also Python 3.10.6, with stdlib only. `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` apply to raw/CSV reconstruction. No dependency install. Trace resource planning inspected the 1,500,192,771-byte file header and supported loader/aggregate complexity; MemAvailable was 121,101,300 KiB. Full trace load took 30.251 s; complete audit took 49.260 s and peak RSS 6,062,156 KiB.\n\n'
command += '## Exact execution commands\n\n```bash\n'
command += prefix+'/usr/bin/time -v '+raw_cmd.removeprefix(prefix)+' > '+str(TEMP/'w128_balanced_decode_raw_audit.log')+' 2>&1\n'
command += prefix+'/usr/bin/time -v '+trace_cmd.removeprefix(prefix)+' > '+str(TEMP/'w128_balanced_decode_trace_audit.log')+' 2>&1\n'
command += raw_cmd+' > '+str(TEMP/'w128_balanced_decode_raw_audit_r1.log')+' 2>&1\n'
command += trace_cmd+' > '+str(TEMP/'w128_balanced_decode_trace_audit_r1.log')+' 2>&1\n'
command += final_cmd+' > '+str(TEMP/'w128_balanced_decode_finalize.log')+' 2>&1\n```\n\n'
command += 'The first pair exited 127 before running Python: `/usr/bin/timeout: failed to run command ‘/usr/bin/time’: No such file or directory`. Removed only the unavailable timing binary; both 60-second/30 GiB caps remained. Scripts record monotonic elapsed time and `resource.getrusage().ru_maxrss`. The second pair passed. No data repair or rerun of GPU measurement.\n\n'
command += '## Why a task-local adapter is necessary\n\nThe reused warmup auditor hard-codes a historical package, job, routing and paths in `main`; only its existing `scan_raw`, `audit_tables`, and `statistics_for` are imported unchanged. Current routing checks additionally compare the exact flat strategy origin/SHA, source-defined nine native modules and six FFN router/runner identities. The existing `tests/analysis/v4_kernel_coverage.py` CLI requires a different accepted-prefill receipt schema/native inputs and also computes nested scope sums, forbidden here. This adapter calls its underlying supported retained DSA parser/aggregate directly and limits the extra census to ownership/correlation counts. No new maintained tool.\n\n'
for path in (SCRIPT,Path(__file__)):
    command += '## Full temporary script: `'+str(path)+'`\n\nSHA-256: `'+sha(path)+'`.\n\n```python\n'+path.read_text()+'```\n\n'
command += '## Exact logs\n\n'
for entry in raw['execution_records']:
    command += '`'+entry['path']+'` (exit '+str(entry['exit_code'])+', SHA-256 `'+entry['sha256']+'`)\n\n```text\n'+Path(entry['path']).read_text()+'```\n\n'
(OUT/'command.md').write_text(command)
report = '# Balanced q1 decode CPU data verification\n\n'+header
report += 'Task `W128-BALANCED-DECODE-DATA`; job `exp-1023-213905-916183`. **PASS for the scoped CPU integrity, raw/table reproduction and constructed routing identity checks. No data/native/root/serving admission.** Trace-wide attention ownership and device dependencies remain unqualified.\n\n'
report += '## Inputs and source binding\n\nRuntime source `'+raw['source_revision']+'`; builder `'+raw['builder_revision']+'`; merge `'+raw['merged_revision']+'`. Manifest SHA-256 `'+raw['manifest_sha256']+'`. Original package: `'+raw['package']+'`. Local phase: `'+raw['phase_root']+'`. Receipt: `'+raw['download_receipt']['path']+'`, SHA-256 `'+raw['download_receipt']['sha256']+'`. Local eight-file total is 1,601,865,978 bytes; all hashes match the root download receipt and remain unchanged before/after/final. Cloud hashes were not independently queried.\n\n'
report += '| Original file | Bytes | SHA-256 |\n| --- | ---: | --- |\n'
for item in raw['inputs_before_after']:
    report += '| `'+str(Path(item['path']).relative_to(Path(raw['local_root'])))+'` | '+str(item['bytes'])+' | `'+item['sha256']+'` |\n'
report += '\n## Acceptance predicates\n\n| Predicate | Expected | Observed / result |\n| --- | --- | --- |\n'
for line in [
    ('Measured keys','q1, C6143..7166 inclusive, six layers, both methods','PASS; exact context/layer/scope Cartesian sets; 1024 steps/method'),
    ('Warmups','two unmeasured decode steps','PASS source/raw-window check; prompt 6141, skip=2, contexts 6141/6142 absent; skipped-step IDs not retained'),
    ('Call counts','CUDA_EVENT 51200; KERNEL_ONLY 45056','PASS; enclosing layer scope only in CUDA_EVENT explains the 6144 difference'),
    ('Rows','6144 per CSV; one sample/context/layer/scope','PASS; no missing/duplicate/malformed row or scope keys, zero duplicate CSV columns'),
    ('Raw JSON fields','no duplicate/malformed fields','PASS; 98305 raw records parse with duplicate-field rejection'),
    ('Costs/counts','finite positive raw time; positive integer KERNEL_ONLY count','PASS; event kernel_count is null; std=0 and count=1 for singleton scope stats'),
    ('Every scope statistic','exact raw join','PASS; 307200 CUDA_EVENT and 315392 KERNEL_ONLY numeric cells exact'),
    ('Full table reconstruction','all original CSV bytes equal','PASS; pinned builder plus identity rebuild both byte-identical CSVs'),
    ('Routing identity','balanced_topk strategy, exact origin/SHA and native origins in every row','PASS; six constructed routers/runners, nine source pins, 12288 rows / 36864 routing cells'),
    ('Trace scope reconstruction','45056 annotations, exact raw time/count','PASS; every key, time and kernel count exact'),
    ('Trace-wide kernel ownership','all kernels assigned to measured attention scopes','FAIL / not established; 227901 of 352829 kernels unowned by measured scopes'),
    ('Actual event dependency edges','event handles/generations and stream edges','UNVERIFIED; synchronization args only External id/cbid/correlation'),
    ('Serving or physical TP8 admission','not in this assignment','NOT ATTEMPTED; physical TP1/eager/six-layer isolated evidence only')]:
    report += '| '+' | '.join(line)+' |\n'
report += '\n## Scope timings and sample/count coverage\n\nAll values are milliseconds. Each listed method/scope has 1024 retained context samples; each original table row has one sample, not 1024 repeated measurements at one context. Layers 0/2/3 plan their variant tile schedule; 1/4/5 reuse it. No nested scopes are summed. Full min/median/p95/p99/max, maxima contexts, top five points and every KERNEL_ONLY count distribution are in `data_audit.json`; all raw outliers remain intact.\n\n'
report += '| Layer / variant | Scope | CUDA_EVENT median / p95 / max | KERNEL_ONLY median / p95 / max | Kernel count per call |\n| --- | --- | ---: | ---: | --- |\n'
stats = {(s['measurement'],s['layer'],s['scope']):s for s in raw['per_scope_statistics']}
for method,layer,scope in sorted(stats):
    if method != 'CUDA_EVENT':
        continue
    ce = stats[method,layer,scope]
    ko = stats.get(('KERNEL_ONLY',layer,scope))
    def times(item):
        return ' / '.join(f'{item[name]:.6f}' for name in ('median_ms','p95_ms','max_ms')) if item else 'not annotated'
    counts = ', '.join(f'{k}×{v}' for k,v in ko['kernel_count_distribution'].items()) if ko else '—'
    report += f'| {layer} / {ce["variant"]} | `{scope}` | {times(ce)} | {times(ko)} | {counts} |\n'
report += '\n## Trace census and precise limits\n\nSupported retained parser `load_trace_events` + `aggregate_kernel_trace` read all 5,177,456 events. There are 45,056 unique nonoverlapping measured annotations, 124,928 uniquely scoped kernels and 227,901 kernels outside measured scopes. All 352,829 kernels have exactly one launch correlation. Unowned regions: 1,163 before first annotation, 226,705 between first/last, 33 after last. These positions do not classify a kernel as attention, FFN, warmup or priming. 73,463 kernels have no External-id CPU-op parent. Five kernel streams are observed (7/21/25/29/33); no critical path is inferred. Synchronization calls: 47,733 event records, 46,187 stream waits, 1,027 event synchronizations, three device synchronizations; event handles/generations and wait-stream identity are absent. Source synchronization semantics are not runtime dependency edges.\n\n'
report += 'No full attention/forward annotation or retained request/warmup identity supplies a completeness denominator. CUDA_EVENT and KERNEL_ONLY are separate instrumented passes. No CPU gap, serving total, graph equivalence, full-layer/TP8/DP2/EP16 equivalence or comparison to other routing variants is inferred. Router tensors/expert-ID values are not retained: the balanced assignment formula is source-backed, while numerical dispatch balance is not independently observed. Root retains platform terminal, actual installed package/device/constructor checks and final admission.\n\n'
report += '## Commands, execution and failure handling\n\n`command.md` retains full exact commands, both temporary scripts and execution logs. Raw/CSV: Python 3.10.6 at `/data/ycfeng/envs/frontier-py310/bin/python`, 17.675 s, peak RSS 932204 KiB. Trace: `/usr/bin/python3` 3.10.6, 49.260 s, peak RSS 6062156 KiB. Each used `sudo -n systemd-run --scope --quiet -p MemoryMax=30G timeout 60s`, with user-owned output via `runuser -u brainpp`; no conda activation or dependency install. Initial commands exited127 because `/usr/bin/time` is absent, before analysis started. Removing only that wrapper resolved it; caps were unchanged. Final duplicate-field/hash checks use the same cap. No Torch/vLLM import in either analysis. The existing experimental-architecture warning concerns simulator support; no simulation/admission ran here.\n\n'
report += '## 1. 规划完成情况\n\n**汇总：** PASS 3｜BLOCKED/FAILED/SKIPPED 0｜TODO 0\n\n| 子任务 | 状态 | 简要依据 |\n| --- | --- | --- |\n| 原始数据、统计表、routing 身份 | PASS | 两表字节重建和全部 keys 精确一致 |\n| 有界完整 trace 检查 | PASS | 49.260 秒内完整解析，明确未归属 kernel 和依赖证据缺口 |\n| 三个指定产物交付 | PASS | 本次生成且仅占用指定 results 三文件 |\n\n'
report += '## 2. 偏离检测\n\n**结论：** 未偏移，正常推进。\n**检测窗口：** 当前 CPU 分配开始至今｜**当前前沿：** 产物交给 root，保持 admission 边界\n\n| 模式 | 判定 | 依据 |\n| --- | --- | --- |\n| D1 工具打磨 | CLEAR | 只复用所需函数和临时适配器，无维护工具扩张 |\n| D2 细节反复重试 | CLEAR | 缺少 time 二进制一次修正，caps 不变 |\n| D3 支线问题修补 | CLEAR | 无 source/platform 修补 |\n| D4 可并行任务停滞 | CLEAR | raw/table 与 trace 独立并行执行 |\n| D5 超范围镀金 | CLEAR | 只交付指定三文件 |\n| D6 分析空转 | CLEAR | 每项检查产出确定结果或明确证据限制 |\n| D7 路线违背 | CLEAR | 无求和、CPU gap、serving 推断或 admission |\n\n**文档检查：** 本次明确分配的 CPU 验证范围内，S1–S6 均未发现明显缺陷；父记录的平台状态和后续接受仍由 root 更新。\n**归因：** 无\n\n'
report += '## 3. 纠正建议\n\n**分支：** A：spec/tickets 合理，无需改动\n\n**需停止的偏离行为：** 无\n\n**接下去的路线：**\n1. Root 核对三文件及实际 terminal/package/device/constructor 证据 → 完成标志：独立 root 接受记录，不自动提升 data/native/serving admission。\n\n'
report += '## 4. 最值得关注的 3 点\n\n1. 94 个 scope series 的原始统计和两表完整字节重建一致。\n2. 227901 个 kernel 不属于已测 scope；唯一 correlation 不等于完整 attention 覆盖。\n3. 无 event-handle 依赖边、TP8/graph/serving 证据；后续判定由 root 负责。\n'
(OUT/'test_report_2026-10-23_data.md').write_text(report)
print(json.dumps(dict(task_id=raw['task_id'],status=raw['status'],finalization=raw['finalization'],
    artifacts=[dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)) for path in sorted(OUT.iterdir())]),indent=2))
```

## Exact logs

`/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_balanced_decode_raw_audit.log` (exit 127, SHA-256 `9e445dc4a6d24e53e93127402618a278aa63dcdba82cb67802731aa66d6c4c93`)

```text
/usr/bin/timeout: failed to run command ‘/usr/bin/time’: No such file or directory
```

`/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_balanced_decode_trace_audit.log` (exit 127, SHA-256 `9e445dc4a6d24e53e93127402618a278aa63dcdba82cb67802731aa66d6c4c93`)

```text
/usr/bin/timeout: failed to run command ‘/usr/bin/time’: No such file or directory
```

`/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_balanced_decode_raw_audit_r1.log` (exit 0, SHA-256 `cc66b9671fd1a1fb7bb1093e03ec6b3e73e783343b77f26d43148ecb16c93b76`)

```text
WARNING 22:21:26 model_architectures.py:1299] Experimental model architecture profile selected: deepseek_v4 (name=deepseek-ai/DeepSeek-V4-Flash, _model_name=deepseek-ai/DeepSeek-V4-Flash, model_type=deepseek_v4, model_arch=generic). Runtime limitations: compressed attention, mHC and mixed precision are unsupported; shared FFN sequence-parallel token/collective execution is unsupported
{
  "status": "PASS_CPU_RAW_TABLE_IDENTITY_CHECKS",
  "raw": {
    "kind_counts": {
      "engine": 1,
      "step": 2048,
      "call": 96256
    },
    "call_counts": {
      "CUDA_EVENT": 51200,
      "KERNEL_ONLY": 45056
    },
    "step_counts": {
      "CUDA_EVENT": 1024,
      "KERNEL_ONLY": 1024
    },
    "all_cached_contexts": [
      6143,
      7166
    ],
    "excluded_contexts": [
      6141,
      6142
    ],
    "no_duplicate_or_orphan_calls": true
  },
  "tables": {
    "CUDA_EVENT": {
      "rows": 6144,
      "contexts": 1024,
      "layers": [
        0,
        1,
        2,
        3,
        4,
        5
      ],
      "variant_tile_combinations": 6,
      "samples_per_context_layer_scope": 1,
      "raw_join_numeric_cells_exact": 307200,
      "expected_finite_positive_timings": true,
      "sha256": "d51ce6a312c3ce108d10377125b2f8d2ecfd9368fa77f3094f911a8d48b685be",
      "bytes": 38983593
    },
    "KERNEL_ONLY": {
      "rows": 6144,
      "contexts": 1024,
      "layers": [
        0,
        1,
        2,
        3,
        4,
        5
      ],
      "variant_tile_combinations": 6,
      "samples_per_context_layer_scope": 1,
      "raw_join_numeric_cells_exact": 315392,
      "expected_finite_positive_timings": true,
      "sha256": "e92981d3fd5d910c7b333af5290cc0a8ac636a8a02ad4be2ca556d841e838399",
      "bytes": 37164812
    }
  },
  "scope_series": 94,
  "routing_rows": 12288,
  "execution": {
    "seconds": 17.674971762113273,
    "peak_rss_kib": 932204,
    "python_executable": "/data/ycfeng/envs/frontier-py310/bin/python",
    "python_version": "3.10.6 (main, Nov 14 2022, 16:10:14) [GCC 11.3.0]",
    "finished_utc": "2026-10-23T14:21:41.631259+00:00"
  }
}
```

`/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_balanced_decode_trace_audit_r1.log` (exit 0, SHA-256 `55f0fc65e4258141ab0d0dad61da7aadf7e7735c2f9603a3921985aac6219256`)

```text
{"stage": "trace_loaded", "events": 5177456, "elapsed_seconds": 30.250771994236857, "peak_rss_kib": 6062156}
{
  "status": "PASS_EXACT_KERNEL_AGGREGATION_COVERAGE_LIMITED",
  "census": {
    "events": 5177456,
    "categories": {
      "null": 70,
      "cpu_op": 2169026,
      "user_annotation": 45056,
      "overhead": 109,
      "kernel": 352829,
      "ac2g": 1453949,
      "cuda_runtime": 1001437,
      "gpu_memcpy": 15920,
      "cuda_driver": 82734,
      "gpu_memset": 1029,
      "gpu_user_annotation": 55296,
      "Trace": 1
    },
    "annotations": 45056,
    "duplicate_annotations": 0,
    "raw_kernel_only_calls_reproduced": 45056,
    "scoped_unique_kernels": 124928,
    "unowned_kernels": 227901,
    "unowned_regions": {
      "before_first_measured_scope": 1163,
      "between_first_and_last_measured_scope": 226705,
      "after_last_measured_scope": 33
    },
    "kernel_launch_cardinality": {
      "1": 352829
    },
    "kernels_without_external_cpu_parent": 73463,
    "kernel_streams": {
      "7": 285091,
      "21": 20534,
      "29": 8212,
      "25": 8212,
      "33": 30780
    },
    "sync_calls": {
      "cudaDeviceSynchronize": 3,
      "cudaEventRecordWithFlags": 47733,
      "cudaStreamWaitEvent": 46187,
      "cudaEventSynchronize": 1027
    },
    "kernel_arg_fields": [
      "External id",
      "block",
      "blocks per SM",
      "channel",
      "channel_type",
      "context",
      "correlation",
      "device",
      "est. achieved occupancy %",
      "graph id",
      "graph node id",
      "grid",
      "occupancy",
      "queued",
      "registers per thread",
      "shared memory",
      "stream",
      "warps per SM"
    ],
    "sync_arg_fields": [
      "External id",
      "cbid",
      "correlation"
    ]
  },
  "execution": {
    "seconds": 49.25959668401629,
    "peak_rss_kib": 6062156,
    "python_executable": "/usr/bin/python3",
    "python_version": "3.10.6 (main, Nov 14 2022, 16:10:14) [GCC 11.3.0]",
    "finished_utc": "2026-10-23T14:22:13.892227+00:00"
  }
}
```
