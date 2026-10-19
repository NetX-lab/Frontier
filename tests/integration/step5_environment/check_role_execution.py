#!/usr/bin/env python3
"""Prepare a private TP16 native request run using the existing capacity lifecycle."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import check_role_tp16 as capacity
import role_execution_observation as observation
import role_model_input_observation as model_observation
import role_profiler_observation as profiler_observation

GEN5_MANIFEST_SHA = '68b9233b16b59832a0061fe2ecf75cc4e8f95b22c2600cf29ddc7ecfd6540137'
# Only census bodies absent from the immutable capacity/request ledgers need
# these existing retained roots; their identities remain in CENSUS_SOURCES.
CENSUS_SOURCE_ROOTS = {
    'unquantized': 'native_ffn_execution_sources/observed',
    'modular': 'native_workspace_sources/observed',
    'triton_experts': 'native_ffn_execution_sources/observed',
    'triton_dispatch': 'native_execution_sources/observed',
    'block_table': 'native_state_20261019_01/native_sources',
    'workspace': 'native_workspace_sources/observed',
    'prepare_no_dp_ep': 'native_state_20261019_01/native_sources',
}
POLICY_FIELDS = ('repetition_penalty', 'presence_penalty', 'frequency_penalty',
                 'temperature', 'top_k', 'top_p', 'min_p', 'seed')
LENGTH_FIELDS = dict(n=1, max_tokens=9, min_tokens=9, ignore_eos=True, stop=[], stop_token_ids=[])
REQUEST_SCOPE = 'native_request_results_and_cpu_metadata_only_not_G2'
UNMEASURED = dict.fromkeys(('device_inputs', 'content', 'finite', 'gpu_completion',
    'changed_input_all_rank_graph_replay', 'public_request_join',
    'precision_and_exclusion_binding', 'full_physical_scope_census', 'G2'), 'UNMEASURED')


def request_inputs(config):
    """Read explicit A/B IDs from the retained candidate without promoting its status."""
    path = Path(config['request_manifest'])
    if not path.is_absolute() or path != path.resolve():
        raise ValueError('An explicit canonical request manifest path is required')
    if capacity.sha256(path) != config['request_manifest_sha256']:
        raise ValueError('Request manifest source identity mismatch')
    manifest = json.loads(path.read_text())
    rows = manifest['requests']
    workload = dict(stage='native_requests', requests=2, prompt_tokens=4105, output_tokens=9)
    if config['workload'] != workload or len(rows) != workload['requests']:
        raise ValueError('This caller requires two sequential 4105-token requests with nine output tokens')
    if manifest['sampling_length_fields'] != LENGTH_FIELDS:
        raise ValueError('The explicit native length/stop settings must remain unchanged')
    tokenizer = json.loads((capacity.TASK/'reference_inputs/tokenizer.json').read_text())
    tokenizer_config = json.loads((capacity.TASK/'reference_inputs/tokenizer_config.json').read_text())
    vocab = tokenizer['model']['vocab']
    bos = vocab[tokenizer_config['bos_token']]
    added = {row['id'] for row in tokenizer['added_tokens']}
    added.update(int(token_id) for token_id in tokenizer_config['added_tokens_decoder'])
    ordinary = manifest['ordinary_tokens']
    if len(ordinary) != 2 or any(vocab[row['vocabulary_entry']] != row['token_id'] or
            row['token_id'] in added for row in ordinary):
        raise ValueError('Two ordinary IDs must match the locked BPE vocabulary')
    for name, row in zip(('A', 'B'), rows):
        tokens = row['prompt_token_ids']
        if row['name'] != name or type(tokens) is not list or len(tokens) != workload['prompt_tokens'] or \
                any(type(token_id) is not int or token_id < 0 for token_id in tokens):
            raise ValueError('Explicit ordered A/B prompt token IDs are required')
        if row['prompt_sha256'] != token_hash(tokens):
            raise ValueError('Prompt token hash mismatch: '+name)
        if tokens[0] != bos or tokens[1:-1] != [ordinary[0]['token_id']]*(len(tokens)-2) or \
                tokens[-1] != ordinary[0 if name == 'A' else 1]['token_id']:
            raise ValueError('Prompt IDs differ from the locked BOS/BPE A/B construction')
    changed = [index for index, (a, b) in enumerate(zip(
        rows[0]['prompt_token_ids'], rows[1]['prompt_token_ids'])) if a != b]
    if changed != [workload['prompt_tokens']-1]:
        raise ValueError('Only the last ordinary prompt token may differ between A and B')
    return dict(requests=copy.deepcopy(rows), sampling_length_fields=copy.deepcopy(LENGTH_FIELDS),
        prompt_tokens=workload['prompt_tokens'], output_tokens=workload['output_tokens'],
        changed_prompt_positions=changed, native_input_acceptance='UNMEASURED',
        manifest=str(path), manifest_sha256=config['request_manifest_sha256'])


def token_hash(tokens):
    import hashlib
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode('ascii')).hexdigest()


def preflight(config, *, new_generation=True):
    """Reuse inert capacity admission, then bind the distinct request slice and sources."""
    for key in ('task_root', 'generation', 'resources', 'image', 'vllm_version', 'workload',
                'engine', 'rpc_seconds', 'record_bytes', 'archive_bytes', 'archive_reserve_seconds',
                'marker_root', 'worker_local_root', 'archive_root', 'native_cli', 'caller_sha256',
                'observer_sha256', 'supervisor_sha256', 'request_manifest', 'request_manifest_sha256',
                'request_caller_sha256', 'request_observer_sha256', 'model_input_observer_sha256',
                'state_observer_sha256', 'profiler_observer_sha256'):
        if key not in config:
            raise ValueError(key+' must be explicitly configured')
    original = copy.deepcopy(config)
    original['workload'] = dict(stage='capacity_initialization')
    retained = capacity.preflight(original, new_generation=new_generation)
    if config['rpc_seconds'] > 60 or config['archive_reserve_seconds'] != 300 or \
            config['record_bytes'] > 512*1024**2 or config['archive_bytes'] > 4*1024**3:
        raise ValueError('Preserve the 60s RPC, 300s reserve, 512MiB records and 4GiB archive ceilings')
    if 'physical_observation' in config:
        limits = profiler_observation.ProfilerObservationLimits(**config['physical_observation'])
        if limits.trace_bytes > config['record_bytes']//8:
            raise ValueError('Native diagnostic traces share the original rank allowance')
        if capacity.sha256(Path(__file__).with_name('role_profiler_observation.py')) != config.get('profiler_observer_sha256'):
            raise ValueError('Native profiler observer source identity mismatch')
    if 'device_observation' in config:
        limits = model_observation.DeviceObservationLimits(**config['device_observation'])
        if max(limits.owned_bytes, limits.rpc_bytes) > config['record_bytes']//8:
            raise ValueError('Device destinations share the original rank record allowance')
    archive = Path(config['archive_root'])
    cloud = Path('/mnt/codesign-exp/ycfeng/step5-preview')
    if archive == cloud or not archive.is_relative_to(cloud):
        raise ValueError('Request archives require a distinct personal Step5 generation')
    if new_generation and archive.exists():
        raise ValueError('Request archive generation must be new')
    for filename, key in (('check_role_execution.py', 'request_caller_sha256'),
                          ('role_execution_observation.py', 'request_observer_sha256'),
                          ('role_model_input_observation.py', 'model_input_observer_sha256'),
                          ('role_state_observation.py', 'state_observer_sha256'),
                          ('role_profiler_observation.py', 'profiler_observer_sha256')):
        if capacity.sha256(Path(__file__).with_name(filename)) != config[key]:
            raise ValueError('Request source identity mismatch: '+filename)
    path = capacity.TASK/'native_role_sources/observed_generation5/source_manifest.json'
    if capacity.sha256(path) != GEN5_MANIFEST_SHA:
        raise ValueError('Retained request source manifest mismatch')
    sources = dict(retained['sources'])
    for row in json.loads(path.read_text())['files']:
        name, digest = row['relative_path'], row.get('sha256', row.get('source_sha256'))
        if capacity.sha256(path.parent/name) != digest:
            raise ValueError('Retained request source body mismatch: '+name)
        if name in sources and sources[name] != digest:
            raise ValueError('Conflicting native request source identity: '+name)
        sources[name] = digest
    for key, (module, digest) in model_observation.CENSUS_SOURCES.items():
        name = module.replace('.', '/')+'.py'
        if key in CENSUS_SOURCE_ROOTS:
            body = capacity.TASK/CENSUS_SOURCE_ROOTS[key]/name
            if capacity.sha256(body) != digest:
                raise ValueError('Retained census source body mismatch: '+name)
        elif name not in sources:
            raise ValueError('Missing retained census source identity: '+name)
        if name in sources and sources[name] != digest:
            raise ValueError('Conflicting native census source identity: '+name)
        sources[name] = digest
    return dict(scope='inert_request_preparation_not_native_qualification',
        capacity_preflight=retained, sources=sources, request_inputs=request_inputs(config),
        evidence_status=dict(UNMEASURED))


def profiler_rows(rows, initial, config, *, started):
    """Verify source, worker, lifecycle and trace completion without node attribution."""
    if not isinstance(rows, list) or len(rows) != len(initial) or not initial:
        raise ValueError('Every original native worker must return its profiler observation')
    workers = {row['rank']: row for row in initial}
    if len(workers) != len(initial) or set(workers) != set(range(16)):
        raise ValueError('The original TP16 worker census must be complete and unique')
    source_owners = {
        'wrapper': {'TorchProfilerWrapper.'+name for name in
                    ('__init__', '_start', '_stop', 'annotate_context_manager')},
        'worker': {'Worker.profile', 'Worker._get_cudagraph_capture_context'},
        'torch_graphs': {'CUDAGraph.raw_cuda_graph_exec'},
    }
    identity_keys = ('rank', 'local_rank', 'pid', 'start_time', 'parent_pid', 'parent_start_time',
                     'generation', 'node_rank', 'boot_id', 'pid_namespace', 'observer_sha256')
    seen = set()
    for row in rows:
        rank = row['rank']
        if type(rank) is not int or rank in seen or rank not in workers:
            raise ValueError('Native profiler workers are missing, duplicated or substituted')
        seen.add(rank)
        native = workers[rank]
        identity = {key: native[key] for key in identity_keys}
        observed = row['profiler_observation']
        if (row['source'] != observation.SOURCE or row['pid'] != native['pid'] or
                row['local_rank'] != native['local_rank'] or row['worker_identity'] != identity or
                observed['worker_identity'] != identity or observed['errors'] or
                observed['status'] != 'NATIVE_PROFILER_DIAGNOSTIC_ONLY_NOT_G2' or
                observed['observer_sha256'] != config['profiler_observer_sha256'] or
                observed['limits'] != config['physical_observation']):
            raise ValueError('Native profiler source, worker, limit or error state is unverified')
        sources = observed['sources']
        if len(sources) != len(profiler_observation.SOURCE_PINS):
            raise ValueError('Native profiler must bind its original wrapper and worker sources')
        for key, (relative, digest) in profiler_observation.SOURCE_PINS.items():
            selected = [source for path, source in sources.items() if Path(path).as_posix().endswith('/'+relative)]
            if (len(selected) != 1 or selected[0]['sha256'] != digest or
                    len(selected[0]['code_owners']) != len(source_owners[key]) or
                    {owner['name'] for owner in selected[0]['code_owners']} != source_owners[key] or
                    any(type(owner['first_line']) is not int or owner['first_line'] <= 0
                        for owner in selected[0]['code_owners'])):
                raise ValueError('Native profiler loaded code differs from the pinned original owners')
        windows = observed['windows']
        expected = ['capture'] if started else ['capture', 'requests']
        if [window['label'] for window in windows] != expected:
            raise ValueError('Original capture/request profiling windows are missing or out of order')
        previous = None
        for window in windows:
            trace = window['trace']
            if (window['status'] != 'COMPLETE_DIAGNOSTIC_WINDOW_NOT_G2' or window['handler_calls'] != 1 or
                    window['native_start'] != 'ORIGINAL_PROFILER_RUNNING' or
                    window['native_stop'] != 'ORIGINAL_PROFILER_STOPPED_AND_TRACE_COMPLETE' or
                    type(window['profiler_id']) is not int or window['profiler_id'] <= 0 or
                    window['wrapper_source'] != sources or trace is None or
                    trace['status'] != 'COMPLETE_NATIVE_TRACE_NOT_G2' or not trace['reader_complete'] or
                    trace['error'] is not None or trace['deadline_exceeded'] or
                    trace['byte_limit'] != config['physical_observation']['trace_bytes'] or
                    trace['bytes_seen'] != trace['retained_bytes'] or
                    type(trace['retained_bytes']) is not int or not 0 < trace['retained_bytes'] <= trace['byte_limit'] or
                    type(trace['events']) is not int or not 0 < trace['events'] <= config['physical_observation']['trace_events'] or
                    any(type(trace[key]) is not int or trace[key] < 0 for key in ('annotations', 'graph_launches', 'kernels')) or
                    trace['annotations'] <= 0 or
                    any(trace[key] > trace['events'] for key in ('annotations', 'graph_launches', 'kernels')) or
                    type(trace['sha256']) is not str or len(trace['sha256']) != 64 or
                    any(character not in '0123456789abcdef' for character in trace['sha256'])):
                raise ValueError('Native trace handler, source, activity, hash or byte completion is unverified')
            if Path(trace['path']).name != f'profile_rank_{rank}_pid_{native["pid"]}_{window["label"]}.json':
                raise ValueError('Native trace output differs from its original worker/window identity')
            times = [window['start'], trace['started'], trace['finished'], trace['inspection_started'],
                     trace['inspection_finished'], window['stop'], trace['deadline']]
            if (any(type(value) not in (int, float) or not math.isfinite(value) for value in times) or
                    times != sorted(times) or window['stop'] >= trace['deadline'] or
                    trace['seconds'] != trace['finished']-trace['started']):
                raise ValueError('Native trace export and inspection did not complete within the original deadline')
            if previous is not None and (previous['stop'] > window['start'] or
                    previous['profiler_id'] != window['profiler_id']):
                raise ValueError('Native request trace must follow capture with the same original profiler')
            previous = window
        if started:
            window = observed['current']
            if (window is None or window['label'] != 'requests' or
                    window['status'] != 'COLLECTING_DIAGNOSTIC_TRACE_NOT_G2' or
                    window['native_start'] != 'ORIGINAL_PROFILER_RUNNING' or
                    window['native_stop'] != 'UNMEASURED' or window['trace'] is not None or
                    window['handler_calls'] != 0 or window['wrapper_source'] != sources or
                    previous['profiler_id'] != window['profiler_id'] or previous['stop'] > window['start']):
                raise ValueError('Native profiler RPC returned without original request collection')
        elif observed['current'] is not None:
            raise ValueError('Native profiler stop left an unfinished diagnostic window')
    return dict(status='ALL_RANK_NATIVE_PROFILER_CONTROL_ONLY_NOT_G2', ranks=sorted(seen),
                operation='start' if started else 'stop', physical_kernel_attribution='UNVERIFIED')


def analyze_profiler(config, root):
    """Verify detached original worker files before joining diagnostic launch evidence."""
    root = Path(root).resolve()
    rank_limit = config['record_bytes']//8

    def read(path, cap):
        if not path.is_file() or path.is_symlink() or path.stat().st_size > cap:
            raise ValueError('Missing, substituted or over-budget retained profiler evidence: '+str(path))
        return path.read_bytes()

    node0 = root/'node_0'
    initial = json.loads(read(node0/'all_rank_snapshots.json', config['record_bytes']))['workers']
    stopped = json.loads(read(node0/'profiler_stop.json', config['record_bytes']))['workers']
    control = profiler_rows(stopped, initial, config, started=False)
    nodes = {rank: json.loads(read(root/f'node_{rank}'/'node.json', config['record_bytes'])) for rank in (0, 1)}
    for rank, node in nodes.items():
        if node['node_rank'] != rank or node['generation'] != config['generation']:
            raise ValueError('Retained native node differs from the original generation')
    results = []
    for worker in stopped:
        identity = worker['worker_identity']
        node = nodes[identity['node_rank']]
        directory = root/f'node_{node["node_rank"]}'/'workers'
        if observation.rank_record_bytes(directory, worker['rank']) > rank_limit:
            raise ValueError('Retained profiler and worker files exceed the original rank allowance')
        original = json.loads(read(directory/f'worker_{worker["rank"]}_{worker["pid"]}.json', rank_limit))
        if original != identity:
            raise ValueError('Retained profiler worker differs from the original census')
        source = read(directory/f'request_rank_{worker["rank"]}_pid_{worker["pid"]}.jsonl', rank_limit)
        records, completions = [], {}
        for line in source.splitlines():
            row = json.loads(line)
            if (row['rank'] != worker['rank'] or row['local_rank'] != worker['local_rank'] or
                    row['pid'] != worker['pid'] or row['worker_identity'] != identity):
                raise ValueError('Retained profiler/model row differs from its original worker')
            if row['scope'] == 'native_profiler_completion_only_not_G2':
                label = row['window']['label']
                if label in completions or row['observer_sha256'] != config['profiler_observer_sha256'] or row['errors']:
                    raise ValueError('Retained native profiler completion is duplicated or unverified')
                completions[label] = row['window']
            else:
                if row['source'] != observation.SOURCE:
                    raise ValueError('Retained model observation source differs from the profiler generation')
                observed = row.get('source_pinned_model_observation')
                if observed is not None:
                    if observed['status'] != 'OBSERVED_CPU_METADATA_ONLY_NOT_G2':
                        raise ValueError('Retained native model annotation scope is unverified')
                    records.append(observed)
        joined = []
        windows = worker['profiler_observation']['windows']
        if set(completions) != set(profiler_observation.WINDOWS):
            raise ValueError('Retained native profiler completion records are missing')
        for window in windows:
            if completions[window['label']] != window or window['trace']['deadline'] != node['workload_deadline']:
                raise ValueError('Retained native trace completion/deadline differs from the original receipt')
            trace = window['trace']
            original_path = Path(trace['path'])
            if original_path.parent != Path(node['observer_output']):
                raise ValueError('Native trace output has no original node observer ownership')
            payload = read(directory/original_path.name, config['physical_observation']['trace_bytes'])
            if len(payload) != trace['retained_bytes'] or hashlib.sha256(payload).hexdigest() != trace['sha256']:
                raise ValueError('Actual retained trace bytes/hash differ from native completion')
            events = json.loads(payload)['traceEvents']
            if (len(events) != trace['events'] or any(event.get('cat') == 'user_annotation' and
                    event.get('name', '').startswith(profiler_observation.ANNOTATION_PREFIX) and
                    event.get('pid') != worker['pid'] for event in events)):
                raise ValueError('Actual native trace events differ from original worker completion')
            correlation = profiler_observation.correlate_trace(events, records)
            if (correlation['annotations'], correlation['graph_launches'], correlation['kernels']) != (
                    trace['annotations'], trace['graph_launches'], trace['kernels']):
                raise ValueError('Actual native trace activities differ from original completion counts')
            joined.append(dict(label=window['label'], path=str(directory/original_path.name),
                               sha256=trace['sha256'], correlation=correlation))
        results.append(dict(rank=worker['rank'], worker_identity=identity, windows=joined))
    return dict(status='RETAINED_NATIVE_PROFILER_ANALYSIS_ONLY_NOT_G2', generation=config['generation'],
        control=control, workers=results, physical_kernel_attribution='UNVERIFIED_NO_CAPTURE_NODE_BINDING',
        transient_scale_content='UNMEASURED', clean_performance='UNVERIFIED', G2='UNVERIFIED')


def engine_kwargs(config, node):
    kwargs = capacity.engine_kwargs(config, node)
    kwargs['worker_cls'] = 'check_role_execution.RequestWorker'
    if 'physical_observation' in config:
        kwargs['profiler_config'] = profiler_observation.native_config(node['observer_output'])
    return kwargs


def headless_argv(config, node):
    argv = capacity.headless_argv(config, node)
    argv[argv.index('--worker-cls')+1] = 'check_role_execution.RequestWorker'
    if 'physical_observation' in config:
        argv.extend(['--profiler-config', json.dumps(profiler_observation.native_config(
            node['observer_output']), separators=(',', ':'))])
    return argv



def non_cache_bytes(local):
    local = Path(local)
    return sum(path.stat().st_size for path in local.rglob('*')
               if path.is_file() and 'cache' not in path.relative_to(local).parts)


def owned_device_bytes(config, node, *, require_complete=False):
    if 'device_observation' not in config and 'physical_observation' not in config:
        return 0
    return observation.device_reservation_bytes(node['observer_output'], require_complete=require_complete)


def retain(config, node, filename, value, *, clock=time.monotonic):
    """Charge detached caller receipts to the existing node allowance before each write."""
    local = Path(node['local_output'])
    encoded = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode('utf-8')
    if clock() >= node['workload_deadline']:
        raise TimeoutError('Request evidence exceeded the original workload deadline')
    used = non_cache_bytes(local)
    if used+len(encoded)+owned_device_bytes(config, node) > config['record_bytes']:
        raise ValueError('Node aggregate evidence exceeds the original record budget')
    capacity.save(local/filename, value)
    return str(local/filename)


def public_output(outputs):
    """Detach the inspected public output fields without consuming worker async outputs."""
    return [dict(request_id=output.request_id, finished=output.finished,
        prompt_token_ids=list(output.prompt_token_ids) if output.prompt_token_ids is not None else None,
        encoder_prompt_token_ids=(list(output.encoder_prompt_token_ids)
            if output.encoder_prompt_token_ids is not None else None),
        outputs=[dict(index=item.index, token_ids=list(item.token_ids), text=item.text,
                      finish_reason=item.finish_reason, stop_reason=item.stop_reason)
                 for item in output.outputs]) for output in outputs]


def validate_output(outputs, request, output_tokens):
    """Accept length support only from a completed, unchanged public native output."""
    if len(outputs) != 1:
        raise ValueError('Exactly one public RequestOutput is required: '+request['name'])
    output = outputs[0]
    if not isinstance(output['request_id'], str) or not output['request_id']:
        raise ValueError('Native public request ID is missing: '+request['name'])
    if (output['finished'] is not True or output['prompt_token_ids'] != request['prompt_token_ids'] or
            output['encoder_prompt_token_ids'] is not None):
        raise ValueError('Native output must finish and preserve the exact text prompt: '+request['name'])
    if len(output['outputs']) != 1:
        raise ValueError('Exactly one native completion is required: '+request['name'])
    item = output['outputs'][0]
    if (type(item['index']) is not int or item['index'] != 0 or
            len(item['token_ids']) != output_tokens or
            any(type(token) is not int or token < 0 for token in item['token_ids']) or
            item['finish_reason'] != 'length' or item['stop_reason'] is not None):
        raise ValueError('Native output does not demonstrate the proposed nine-token length settings: '+request['name'])
    return dict(name=request['name'], public_request_id=output['request_id'],
        prompt_sha256=token_hash(output['prompt_token_ids']),
        generated_token_ids=list(item['token_ids']), generated_tokens=len(item['token_ids']),
        finish_reason=item['finish_reason'], stop_reason=item['stop_reason'],
        length_evidence='completed_public_output_only_not_device_or_sampling_semantics')


IDENTITY_FIELDS = ('rank', 'local_rank', 'pid', 'start_time', 'parent_pid', 'parent_start_time',
                   'generation', 'node_rank', 'boot_id', 'pid_namespace', 'observer_sha256')


def request_metadata(rows, initial, previous, config):
    """Check separate CPU summaries against startup identities without joining public IDs."""
    if len(rows) != 16 or {row.get('rank') for row in rows} != set(range(16)):
        raise ValueError('All sixteen separate request metadata summaries are required')
    baseline = {row['rank']: row for row in initial}
    earlier = {row['rank']: row for row in previous} if previous is not None else {}
    for row in rows:
        rank = row['rank']
        if (any(row['worker_identity'][field] != baseline[rank][field] for field in IDENTITY_FIELDS) or
                any(row[field] != baseline[rank][field] for field in ('rank', 'local_rank', 'pid'))):
            raise ValueError('Request metadata worker identity changed')
        if (row['scope'] != 'cpu_request_metadata_only_not_G2' or
                row['status'] != 'OBSERVED_CPU_METADATA_ONLY_NOT_G2' or
                any(row['source'][key] != config[config_key] for key, config_key in (
                    ('observer_sha256', 'request_observer_sha256'),
                    ('capacity_observer_sha256', 'observer_sha256'),
                    ('model_input_observer_sha256', 'model_input_observer_sha256'),
                    ('state_observer_sha256', 'state_observer_sha256'),
                    ('profiler_observer_sha256', 'profiler_observer_sha256')))):
            raise ValueError('Request metadata or observer source is unverified')
        if (row.get('runner_observer_error') is not None or
                row.get('model_input_observer_error') is not None):
            raise ValueError('Native runner or model-input observation is unverified')
        model = row.get('source_pinned_model_observation')
        if model is not None and (model['observer_failures'] or model['collection_stopped']):
            raise ValueError('Model-input metadata collection is unverified')
        counts = row['request_metadata']
        for field in ('invocations', 'records_written', 'metadata_snapshots'):
            count = counts[field]
            old = earlier[rank]['request_metadata'][field] if rank in earlier else 0
            if type(count) is not int or count <= old:
                raise ValueError('Request CPU metadata must advance on every rank: '+field)
        if (counts['collection_stopped'] or row['errors']['observer'] is not None or
                row['errors']['native'] is not None or row['errors']['native_failures'] != 0 or
                counts['last_invocation']['native_outcome'] != 'returned'):
            raise ValueError('Request metadata collection is unverified')
    return dict(status='OBSERVED_CPU_METADATA_ONLY_NOT_G2', ranks=len(rows),
                public_request_join='UNMEASURED', device_inputs='UNMEASURED')


def device_preparation_rows(rows, initial, preparation, config):
    """Bind every prepared diagnostic allocation to the original worker identities."""
    if len(rows) != len(initial) or {row.get('rank') for row in rows} != {row['rank'] for row in initial}:
        raise ValueError('Every original worker must return its own device preparation result')
    baseline = {row['rank']: row for row in initial}
    for row in rows:
        before = baseline[row['rank']]
        if (row['status'] != 'PREPARED_DIAGNOSTIC_DEVICE_WITNESSES_NOT_G2' or
                row['preparation'] != preparation or
                any(row['worker_identity'][field] != before[field] for field in IDENTITY_FIELDS) or
                any(row[field] != before[field] for field in ('rank', 'local_rank', 'pid')) or
                any(row['source'][key] != config[config_key] for key, config_key in (
                    ('observer_sha256', 'request_observer_sha256'),
                    ('capacity_observer_sha256', 'observer_sha256'),
                    ('model_input_observer_sha256', 'model_input_observer_sha256'),
                    ('state_observer_sha256', 'state_observer_sha256'),
                    ('profiler_observer_sha256', 'profiler_observer_sha256')))):
            raise ValueError('Device preparation is absent, unverified or from a different worker/source')
        observed = row['device_observation']
        if (observed['status'] != 'PREPARED_NOT_EXECUTED' or observed['collected_steps'] or
                observed['owned_bytes'] > preparation['limits']['owned_bytes'] or
                observed['clean_performance'] != 'UNVERIFIED' or observed['G2'] != 'UNVERIFIED'):
            raise ValueError('Prepared allocation cannot establish native device execution or G2')
    return dict(status='PREPARED_DIAGNOSTIC_DEVICE_WITNESSES_NOT_G2', ranks=len(rows))


def device_request_rows(rows, previous, association, request, completed, sampling_fields):
    """Reconcile full internal IDs and actual device ranges for one returned request."""
    if association is None:
        raise ValueError('Device evidence requires the retained original frontend association')
    if len(rows) != 16 or {row.get('rank') for row in rows} != set(range(16)):
        raise ValueError('All sixteen original workers are required for device evidence')
    internal = association['internal_request_id']
    prior = {row['rank']: row.get('device_observation', {}).get('drained_steps', 0)
             for row in previous} if previous is not None else {}
    expected_tokens = list(request['prompt_token_ids'])+list(completed['generated_token_ids'][:-1])
    counts = []
    for worker in rows:
        observed = worker.get('device_observation')
        if (observed is None or observed['status'] != 'OBSERVED_DIAGNOSTIC_DEVICE_WITNESSES_NOT_G2' or
                observed['collection_stopped'] or observed['errors'] or
                observed['drained_steps'] != observed['collected_steps'] or
                observed['native_async_output_consumed_by_observer']):
            raise ValueError('A worker has incomplete diagnostic device/completion evidence')
        registered = [row for row in observed['registered_requests'] if row['internal_request_id'] == internal]
        if (len(registered) != 1 or registered[0]['prompt_token_ids'] != request['prompt_token_ids'] or
                registered[0]['resolved_sampling_fields'] != sampling_fields or
                registered[0]['name'] != request['name']):
            raise ValueError('Original frontend and worker registration IDs, prompt or sampling disagree')
        steps = observed['steps'][prior.get(worker['rank'], 0):]
        if not steps or any(row['prepared_cpu_batch']['req_ids'] != [internal] for row in steps):
            raise ValueError('Device steps do not belong exclusively to the original complete request ID')
        cursor = 0
        for step in steps:
            if (step['status'] != 'OBSERVED_DIAGNOSTIC_DEVICE_WITNESS_NOT_G2' or
                    step['completion'] != 'OWNED_COPY_EVENT_QUERY_READY_AFTER_NATIVE_GENERATE' or
                    step['finite_elements_observed'] != step['finite_elements_expected'] or
                    step['native_async_output_consumed_by_observer']):
                raise ValueError('Native valid hidden finite/completion witness is unverified')
            batch, values = step['prepared_cpu_batch'], step['values']
            valid, padded = batch['num_tokens'], batch['num_tokens_after_padding']
            if (batch['num_reqs'] != 1 or batch['query_start_loc_np'][:2] != [0, valid] or
                    values['query_start_loc'][:2] != [0, valid] or
                    values['positions'][:valid] != list(range(cursor, cursor+valid)) or
                    values['final_positions'][:valid] != values['positions'][:valid] or
                    values['seq_lens'][0] != cursor+valid or
                    values['input_ids'][:valid] != expected_tokens[cursor:cursor+valid] or
                    (step['padding_mask_used'] and any(values['is_padding'][:valid])) or
                    values['idx_mapping'] != [registered[0]['persistent_index']] or
                    values['request_state'][0] != registered[0]['persistent_index'] or
                    any(value != -1 for value in values['request_state'][1:])):
                raise ValueError('Actual device token/position/request ranges do not conserve native request work')
            groups = step['slot_groups']
            if not groups or any(not group for group in groups):
                raise ValueError('Native layer slot groups are absent')
            for index in range(len(groups)):
                slots = values['slot_mapping_'+str(index)]
                if len(slots) != padded or any(type(slot) is not int or slot < 0 for slot in slots[:valid]):
                    raise ValueError('Valid native token slot bindings are missing or padded')
            if not step['model_input_ids_none'] and values['final_input_ids'][:valid] != values['input_ids'][:valid]:
                raise ValueError('Finalized model token input differs from the observed prepared token content')
            if step['model_input_ids_none'] and step['model_inputs_embeds_none']:
                raise ValueError('Finalized model has no observed token or embedding input')
            hidden = values['hidden_states']
            if hidden['shape'][0] != 1 or step['finite_elements_expected'] != valid*hidden['shape'][1]:
                raise ValueError('Finite reduction does not cover every valid hidden element')
            if not step['model_inputs_embeds_none'] and values['inputs_embeds']['shape'] != hidden['shape']:
                raise ValueError('Finalized embedding witness does not cover the valid native request row')
            if (len(values['input_ids']) != padded or len(values['final_positions']) != padded or
                    batch['num_computed_tokens_np'] != [cursor] or
                    batch['num_computed_prefill_tokens_np'] != [min(cursor, len(request['prompt_token_ids']))] or
                    batch['prefill_len_np'] != [len(request['prompt_token_ids'])]):
                raise ValueError('Device witnesses and original CPU frontiers disagree in the approved scope')
            cursor += valid
        if cursor != len(expected_tokens):
            raise ValueError('Device witnesses omit or duplicate useful prompt/decode token work')
        counts.append(dict(rank=worker['rank'], steps=len(steps), valid_tokens=cursor))
    return dict(status='OBSERVED_DIAGNOSTIC_FRONTEND_WORKER_DEVICE_ASSOCIATION_NOT_G2',
        name=request['name'], public_request_id=association['public_request_id'], internal_request_id=internal,
        workers=counts, native_async_output_consumed_by_observer=False,
        global_step_causality='UNVERIFIED', physical_scope_census='UNVERIFIED', clean_performance='UNVERIFIED',
        precision_and_exclusion_binding='UNVERIFIED', G2='UNVERIFIED')


def compare_capacity(initial, final):
    for row in final:
        before = next(item for item in initial if item['rank'] == row['rank'])
        if (any(row[field] != before[field] for field in IDENTITY_FIELDS) or
                row['gpu_uuid'].removeprefix('GPU-') != before['gpu_uuid'].removeprefix('GPU-')):
            raise ValueError('Initial/final native worker identity changed')
    return True


def frontend_identity(config, node):
    """Read the actual caller process in the original supervisor's PID namespace."""
    stat = Path('/proc/self/stat').read_text()
    pid = os.getpid()
    if int(stat.split(' ', 1)[0]) != pid:
        raise ValueError('Frontend /proc identity does not match os.getpid()')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    namespace = os.readlink('/proc/self/ns/pid')
    if (boot, namespace) != (node['boot_id'], node['pid_namespace']):
        raise ValueError('Frontend and original supervisor identity namespaces differ')
    return dict(pid=pid, start_time=int(stat.rsplit(')', 1)[1].split()[19]),
                parent_pid=os.getppid(), boot_id=boot, pid_namespace=namespace,
                generation=config['generation'], node_rank=node['node_rank'],
                caller_sha256=capacity.sha256(Path(__file__)),
                observer_sha256=capacity.sha256(Path(model_observation.__file__)))


def validate_admission(record, completed, request, inputs, identity):
    """Join public completion to a retained original admission, never an ID suffix."""
    if record is None or record['status'] != 'OBSERVED_FRONTEND_ADMISSION_ONLY_NOT_G2' or len(record['records']) != 1:
        raise ValueError('Original frontend admission is unverified')
    row = record['records'][0]
    if (row['public_request_id'] != completed['public_request_id'] or
            row['prompt_token_ids'] != request['prompt_token_ids'] or
            row['context'] != dict(name=request['name'], manifest=inputs['manifest'],
                                   manifest_sha256=inputs['manifest_sha256'],
                                   prompt_sha256=request['prompt_sha256']) or
            row['frontend_identity'] != identity):
        raise ValueError('Completed public output differs from its original frontend admission')
    return dict(status='OBSERVED_ORIGINAL_FRONTEND_PUBLIC_INTERNAL_ASSOCIATION_NOT_G2',
                name=request['name'], public_request_id=row['public_request_id'],
                internal_request_id=row['internal_request_id'],
                resolved_sampling_fields=copy.deepcopy(row['resolved_sampling_fields']),
                worker_request_join='UNMEASURED', device_consumption='UNMEASURED')


def execute_leader(config, node, sources, *, llm_factory=None, sampling_factory=None,
                   verify=capacity.installed_sources, clock=time.monotonic,
                   publish=capacity.publish_terminal,
                   bind_admission=model_observation.AdmissionSites.bind,
                   identify_frontend=frontend_identity):
    """Run normal sequential generation and retain capacity and request evidence separately."""
    if node['node_rank'] != 0:
        raise ValueError('Only node0 owns native requests')
    end = node['workload_deadline']
    if type(end) not in (int, float) or not math.isfinite(end):
        raise ValueError('Finite local monotonic workload deadline required')
    inputs = request_inputs(config)
    kwargs = engine_kwargs(config, node)
    receipt = dict(scope=REQUEST_SCOPE, status='UNVERIFIED', generation=config['generation'],
        native_generation_returns=[], completed_requests=[], requests=[], frontend_associations=[],
        device_associations=[], profiler_observations=[],
        evidence_status=dict(UNMEASURED), evidence_verdict='PASS', evidence_errors=[],
        result='PASS', return_code=0, reasons=[], request_manifest=inputs['manifest'],
        request_manifest_sha256=inputs['manifest_sha256'],
        sampling_observation_scope='caller_parameters_only_engine_clones_unobserved',
        lifecycle_scope='original_capacity_only_not_G2_status_retained_separately',
        terminal_publication='UNMEASURED',
        terminal_marker=str(Path(config['marker_root'])/'node_0_workload_terminal.json'))
    llm, phase, request, sampling_fields = None, 'deadline_before_initialization', None, None
    profiler_start_attempted, profiler_stop_attempted = False, False

    def remaining():
        value = end-clock()
        if value <= 0:
            raise TimeoutError('Native request workload uses the original absolute deadline')
        return value

    def evidence(operation, call):
        return capacity.supervision.evidence(receipt, operation, call)

    def record(filename, value):
        return evidence('save '+filename, lambda: retain(config, node, filename, value, clock=clock))

    def stop_profiler(operation):
        nonlocal profiler_stop_attempted
        if llm is None or not profiler_start_attempted or profiler_stop_attempted:
            return
        profiler_stop_attempted = True
        try:
            profiled = evidence(operation, lambda: llm.collective_rpc('physical_profile',
                args=('requests', False), timeout=min(remaining(), config['rpc_seconds'])))
            if profiled is not None:
                record('profiler_stop.json', dict(workers=profiled))
                verified = evidence('validate all-rank completed native traces', lambda: profiler_rows(
                    profiled, initial, config, started=False))
                if verified is not None:
                    receipt['profiler_observations'].append(verified)
        except BaseException as failure:
            # Auxiliary cleanup must not replace a native generation exception.
            receipt['evidence_verdict'] = 'UNVERIFIED'
            receipt['evidence_errors'].append(dict(operation=operation, error=repr(failure)))
            receipt['result'], receipt['return_code'] = 'FAIL', 1
            receipt['reasons'].append('evidence unavailable: '+operation)

    def census(rows, raw_name, census_name):
        if record(raw_name, dict(workers=rows)) is None:
            return None
        def inspect():
            remaining()
            nodes = [json.loads((Path(config['marker_root'])/f'node_{rank}_inventory.json').read_text())
                     for rank in (0, 1)]
            return capacity.all_rank_census(rows, nodes, capacity.expected_census())
        result = evidence('validate '+census_name, inspect)
        if result is not None:
            record(census_name, dict(census=result, workers=rows))
        return result

    try:
        remaining()
        phase = 'installed_source_verification'
        verify(config, sources)
        phase = 'deadline_before_initialization'
        remaining()
        if llm_factory is None:
            from vllm import LLM
            llm_factory = LLM
        phase = 'native_initialization'
        llm = llm_factory(**kwargs)
        phase = 'native_sampling_defaults'
        defaults = llm.get_default_sampling_params()
        policy = {field: copy.deepcopy(getattr(defaults, field)) for field in POLICY_FIELDS}
        if policy['seed'] is not None:
            raise ValueError('The native default seed must remain None; no sampling substitute is authorized')
        receipt['native_default_sampling'] = dict(policy, **{
            field: copy.deepcopy(getattr(defaults, field)) for field in LENGTH_FIELDS})
        sampling_fields = dict(policy, **copy.deepcopy(inputs['sampling_length_fields']))
        if sampling_factory is None:
            from vllm import SamplingParams
            sampling_factory = SamplingParams
        phase = 'native_sampling_constructor'
        params = sampling_factory(**sampling_fields)
        receipt['submitted_sampling_fields'] = copy.deepcopy(sampling_fields)
        phase = 'deadline_before_initial_capacity_rpc'
        timeout = min(remaining(), config['rpc_seconds'])
        phase = 'native_initial_capacity_rpc'
        initial = llm.collective_rpc('capacity_snapshot', timeout=timeout)
        initial_census = census(initial, 'all_rank_snapshots.json', 'all_rank_census.json')
        admission_sites = evidence('bind original frontend admission', lambda: bind_admission(llm.llm_engine))
        identity = evidence('read frontend process identity', lambda: identify_frontend(config, node))
        preparation = None
        if 'device_observation' in config:
            preparation = dict(limits=copy.deepcopy(config['device_observation']),
                requests=copy.deepcopy(inputs['requests']), sampling_fields=copy.deepcopy(sampling_fields))
            phase = 'deadline_before_device_preparation_rpc'
            timeout = min(remaining(), config['rpc_seconds'])
            phase = 'native_device_preparation_rpc'
            prepared = llm.collective_rpc('prepare_device_observation', args=(preparation,), timeout=timeout)
            record('device_preparation.json', dict(workers=prepared))
            evidence('validate all-worker device preparation',
                lambda: device_preparation_rows(prepared, initial, preparation, config))
        if 'physical_observation' in config:
            profiler_start_attempted = True
            phase = 'native_profiler_start_rpc'
            profiled = evidence('start original native request profiler', lambda: llm.collective_rpc(
                'physical_profile', args=('requests', True), timeout=min(remaining(), config['rpc_seconds'])))
            if profiled is not None:
                record('profiler_start.json', dict(workers=profiled))
                evidence('validate original all-rank profiler start', lambda: profiler_rows(
                    profiled, initial, config, started=True))
        previous = None
        for request in inputs['requests']:
            name = request['name']
            phase = 'deadline_before_generate_'+name
            remaining()
            context = dict(name=name, manifest=inputs['manifest'],
                           manifest_sha256=inputs['manifest_sha256'],
                           prompt_sha256=request['prompt_sha256'],
                           prompt_token_ids=request['prompt_token_ids'])
            scope = (evidence('prepare frontend admission '+name, lambda: model_observation.AdmissionScope(
                llm.llm_engine, admission_sites, context, params, sampling_fields, identity))
                if admission_sites is not None and identity is not None else None)
            if scope is not None:
                evidence('install frontend admission '+name, scope.__enter__)
            admission_record = None
            phase = 'native_generate_'+name
            try:
                outputs = llm.generate([dict(prompt_token_ids=list(request['prompt_token_ids']))],
                                       sampling_params=params, use_tqdm=False)
            finally:
                native_exception = sys.exc_info()
                if scope is not None:
                    def finalize_admission():
                        scope.__exit__(*native_exception)
                        if scope.record is None:
                            raise ValueError('Original frontend admission has no retained record')
                        return scope.record
                    admission_record = evidence('finalize frontend admission '+name, finalize_admission)
                    if admission_record is not None:
                        record('request_'+name+'_admission.json', admission_record)
            receipt['native_generation_returns'].append(name)
            detached = evidence('detach public output '+name, lambda: public_output(outputs))
            checked, association = None, None
            if detached is not None:
                record('request_'+name+'_outputs.json', dict(name=name, outputs=detached))
                checked = evidence('validate public output '+name,
                    lambda: validate_output(detached, request, inputs['output_tokens']))
                if checked is not None:
                    receipt['completed_requests'].append(name)
                    receipt['requests'].append(checked)
                    association = evidence('validate frontend admission '+name, lambda: validate_admission(
                        admission_record, checked, request, inputs, identity))
                    if association is not None:
                        receipt['frontend_associations'].append(association)
            phase = 'deadline_before_request_'+name+'_rpc'
            timeout = min(remaining(), config['rpc_seconds'])
            phase = 'native_request_'+name+'_rpc'
            rows = llm.collective_rpc('request_snapshot', timeout=timeout)
            if record('request_'+name+'_snapshots.json', dict(workers=rows)) is not None:
                metadata = evidence('validate request '+name+' CPU metadata',
                         lambda: request_metadata(rows, initial, previous, config))
                if preparation is not None and metadata is not None and checked is not None:
                    device = evidence('validate request '+name+' device witnesses', lambda: device_request_rows(
                        rows, previous, association, request, checked, sampling_fields))
                    if device is not None:
                        receipt['device_associations'].append(device)
            previous = rows
            if checked is None:
                break
        phase = 'native_profiler_stop_rpc'
        stop_profiler('stop and drain original native request profiler')
        phase = 'deadline_before_final_capacity_rpc'
        timeout = min(remaining(), config['rpc_seconds'])
        phase = 'native_final_capacity_rpc'
        final = llm.collective_rpc('capacity_snapshot', timeout=timeout)
        final_census = census(final, 'final_all_rank_snapshots.json', 'final_all_rank_census.json')
        if initial_census is not None and final_census is not None:
            evidence('compare initial/final capacity identities', lambda: compare_capacity(initial, final))
        receipt['sampling_observation_scope'] = ('original_frontend_resolved_fields_device_consumption_unmeasured'
            if len(receipt['frontend_associations']) == 2 else 'caller_parameters_only_engine_clones_unobserved')
        if receipt['completed_requests'] == ['A', 'B'] and receipt['result'] == 'PASS':
            receipt['status'] = 'COMPLETED_PUBLIC_OUTPUTS_ONLY_NOT_G2'
        else:
            receipt['result'] = 'FAIL'
        if receipt['result'] != 'PASS':
            receipt['status'] = 'UNVERIFIED'
        phase = 'request_evidence_retention'
        saved = record('request_evidence.json', receipt)
        digest = evidence('hash retained request_evidence.json', lambda: capacity.sha256(saved)) if saved else None
        if saved is None or digest is None:
            receipt['status'] = 'UNVERIFIED'
        retained = 'request_evidence.json retained sha256='+digest if digest else 'request_evidence.json unverified'
        reason = ('Native request results '+receipt['status']+'; phase='+phase+'; '+retained+
                  '; remaining G2 unmeasured')
        if receipt['evidence_errors']:
            reason += '; '+repr(receipt['evidence_errors'])
        phase = 'request_terminal_publication'
        evidence('publish native workload terminal', lambda: publish(config, node, receipt['result'], reason))
        return llm
    except BaseException as failure:
        stop_profiler('cleanup original native request profiler after failure')
        error = dict(phase=phase, error_type=observation.class_name(failure), traceback=traceback.format_exc(),
            request=copy.deepcopy(request), engine_kwargs=kwargs,
            submitted_sampling_fields=sampling_fields, config=config, sources=sources)
        deadline = phase.startswith('deadline_before_')
        native = phase in ('native_initialization', 'native_sampling_constructor',
            'native_generate_A', 'native_generate_B', 'native_initial_capacity_rpc', 'native_device_preparation_rpc',
            'native_request_A_rpc', 'native_request_B_rpc', 'native_final_capacity_rpc')
        receipt.update(status='BLOCKED_ON_RUNTIME' if native else
                       'UNVERIFIED_WORKLOAD_DEADLINE' if deadline else 'UNVERIFIED',
                       result='FAIL', return_code=1, failure=error)
        if native:
            record('native_failure.json', dict(status='BLOCKED_ON_RUNTIME', **error))
        else:
            receipt['evidence_verdict'] = 'UNVERIFIED'
            receipt['evidence_errors'].append(dict(operation=phase, error=repr(failure)))
        record('request_evidence.json', receipt)
        capacity.diagnostic(lambda: publish(config, node, 'FAIL',
            receipt['status']+'; phase='+phase+': '+error['traceback']))
        raise
    finally:
        stop_profiler('cleanup original native request profiler')



def run_node(config, sources, config_path, *, env=None, run=subprocess.run, popen=subprocess.Popen,
             supervise=capacity.supervision.supervise, proc_root=Path('/proc'), deadline=None):
    """Adapt the real supervisor argv and retain its existing node, PID and archive ownership."""
    env = dict(os.environ if env is None else env)
    env.update(STEP5_REQUEST_CALLER_SHA=config['request_caller_sha256'],
               STEP5_REQUEST_OBSERVER_SHA=config['request_observer_sha256'],
               STEP5_MODEL_INPUT_OBSERVER_SHA=config['model_input_observer_sha256'],
               STEP5_STATE_OBSERVER_SHA=config['state_observer_sha256'],
               STEP5_PROFILER_OBSERVER_SHA=config['profiler_observer_sha256'])

    def request_supervision(root, generation, passed_env, engine, argv, *args, **kwargs):
        argv = list(argv)
        argv[2] = str(Path(__file__).resolve())
        if 'physical_observation' in config:
            passed_env = dict(passed_env,
                STEP5_PHYSICAL_OBSERVATION=json.dumps(config['physical_observation'], separators=(',', ':')),
                STEP5_PROFILER_OBSERVER_SHA=config['profiler_observer_sha256'])
        engine = dict(engine, request_sources=dict(caller_sha256=config['request_caller_sha256'],
            observer_sha256=config['request_observer_sha256'],
            model_input_observer_sha256=config['model_input_observer_sha256'],
            state_observer_sha256=config['state_observer_sha256'],
            profiler_observer_sha256=config['profiler_observer_sha256']),
            request_manifest_sha256=config['request_manifest_sha256'])
        node = json.loads(Path(passed_env['STEP5_CAPACITY_NODE_FILE']).read_text())
        if 'physical_observation' in config:
            node.update(physical_observation=copy.deepcopy(config['physical_observation']),
                profiler_observer_sha256=config['profiler_observer_sha256'])
            capacity.save(Path(passed_env['STEP5_CAPACITY_NODE_FILE']), node)
            capacity.save(Path(root)/f"node_{node['node_rank']}_inventory.json", node)
            engine['physical_observation'] = copy.deepcopy(config['physical_observation'])
        original_finalize = kwargs['finalize_evidence']

        def finalize_request_records(status):
            local = Path(node['local_output'])
            # Reserve the exact metadata written by the unchanged capacity finalizer.
            pending = dict(status, result='UNVERIFIED', return_code=None,
                evidence_verdict='UNVERIFIED', phase='before_archive_completion_and_peer_exchange',
                final_result_retained=False)
            metadata_bytes = len((json.dumps(pending, indent=2, default=str)+'\n').encode('utf-8'))
            used = non_cache_bytes(local)
            try:
                owned = owned_device_bytes(config, node, require_complete=True)
            except BaseException as failure:
                return dict(verdict='UNVERIFIED', record_budget=dict(node_rank=node['node_rank'],
                    record_bytes=config['record_bytes'], observed_bytes=used, owned_device_bytes=None,
                    finalization_metadata_bytes=metadata_bytes, verdict='UNVERIFIED', error=repr(failure)))
            budget = dict(node_rank=node['node_rank'], record_bytes=config['record_bytes'],
                observed_bytes=used, owned_device_bytes=owned, finalization_metadata_bytes=metadata_bytes,
                required_bytes=used+metadata_bytes+owned,
                scope='aggregate_non_cache_files_and_reserved_tensor_storage_before_archive',
                verdict='PASS' if used+metadata_bytes+owned <= config['record_bytes'] else 'UNVERIFIED')
            if budget['verdict'] != 'PASS':
                return dict(verdict='UNVERIFIED', record_budget=budget)
            archive = original_finalize(status)
            budget['retained_bytes'] = archive['retained_bytes']
            if archive['retained_bytes']+owned > config['record_bytes']:
                budget['verdict'] = 'UNVERIFIED'
            return dict(archive, verdict='UNVERIFIED' if budget['verdict'] != 'PASS' else archive['verdict'],
                        record_budget=budget)

        kwargs['finalize_evidence'] = finalize_request_records
        return supervise(root, generation, passed_env, engine, argv, *args, **kwargs)

    return capacity.run_node(config, sources, config_path, env=env, run=run, popen=popen,
        supervise=request_supervision, proc_root=proc_root, deadline=deadline)


def __getattr__(name):
    if name != 'RequestWorker':
        raise AttributeError(name)
    for variable in ('STEP5_CAPACITY_TASK', 'STEP5_CAPACITY_NODE_FILE',
                     'STEP5_REQUEST_CALLER_SHA', 'STEP5_REQUEST_OBSERVER_SHA',
                     'STEP5_MODEL_INPUT_OBSERVER_SHA', 'STEP5_STATE_OBSERVER_SHA', 'STEP5_PROFILER_OBSERVER_SHA',
                     'STEP5_CAPACITY_OBSERVER_SHA'):
        if variable not in os.environ:
            raise ValueError(variable+' must be explicitly configured')
    if Path(os.environ['STEP5_CAPACITY_TASK']).resolve() != capacity.TASK:
        raise ValueError('Request worker requires the exact trusted task root')
    for filename, variable in (('check_role_execution.py', 'STEP5_REQUEST_CALLER_SHA'),
        ('role_execution_observation.py', 'STEP5_REQUEST_OBSERVER_SHA'),
        ('role_model_input_observation.py', 'STEP5_MODEL_INPUT_OBSERVER_SHA'),
        ('role_state_observation.py', 'STEP5_STATE_OBSERVER_SHA'),
        ('role_profiler_observation.py', 'STEP5_PROFILER_OBSERVER_SHA'),
        ('role_capacity_observation.py', 'STEP5_CAPACITY_OBSERVER_SHA')):
        if capacity.sha256(Path(__file__).with_name(filename)) != os.environ[variable]:
            raise ValueError('Request worker source identity mismatch: '+filename)
    if 'STEP5_PHYSICAL_OBSERVATION' in os.environ and os.environ.get('STEP5_PROFILER_OBSERVER_SHA') != profiler_observation.SOURCE_SHA256:
        raise ValueError('Request worker native profiler source identity mismatch')
    observation.capacity.observation_output()
    from vllm.v1.worker.gpu_worker import Worker
    return observation.observed_worker(Worker)



def main():
    deadline = time.monotonic()+capacity.supervision.SECONDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'node', 'leader', 'headless', 'analyze-profiler'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--node-file', type=Path)
    parser.add_argument('--evidence-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.mode == 'analyze-profiler':
        if args.evidence_root is None or args.output is None:
            parser.error('--evidence-root and --output are required for detached profiler analysis')
        result = analyze_profiler(config, args.evidence_root)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return 0
    if args.mode == 'preflight':
        print(json.dumps(preflight(config), indent=2))
        return 0
    if 'native_sources' not in config:
        raise ValueError('native_sources must be explicitly configured')
    receipt = preflight(config, new_generation=False)
    sources = receipt['sources']
    if sources != config['native_sources']:
        raise ValueError('Frozen preflight native source ledger mismatch')
    if args.mode == 'node':
        return run_node(config, sources, args.config, deadline=deadline)['return_code']
    if args.node_file is None:
        parser.error('--node-file is required for the actual native child')
    node = json.loads(args.node_file.read_text())
    if args.mode == 'headless':
        capacity.installed_sources(config, sources)
        argv = headless_argv(config, node)
        os.execv(argv[0], argv)
    else:
        execute_leader(config, node, sources)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
