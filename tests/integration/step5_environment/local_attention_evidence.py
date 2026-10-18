"""Retain actual timed local attention outputs outside priced native scopes.

This private evidence slice reuses frozen attention construction and ownership.
It records bracketing resource queries, not continuous exclusivity or clean timing.
The existing collector and all earlier generations remain unchanged.
"""
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import local_attention_measurement as attention
from check_precision_resources import cgroup_memory, process_tree


def resource_snapshot(query_timeout_seconds):
    """Retain separate query intervals; the measurements are not atomic."""
    if query_timeout_seconds <= 0:
        raise ValueError('A positive resource-query timeout is required')
    result = {'started_ns': time.time_ns(), 'started_monotonic_ns': time.monotonic_ns(),
              'pid': os.getpid(), 'parent_pid': os.getppid(),
              'scope': 'bracketing_queries_not_continuous_exclusivity',
              'nic_counter_unit': 'RDMA port data counters in four-byte words', 'queries': {}}
    queries = {
        'processes': lambda: process_tree(os.getpid()),
        'cgroup': cgroup_memory,
        'nic': lambda: {str(path): int(path.read_text()) for path in
            sorted(Path('/sys/class/infiniband').glob('*/ports/*/counters/port_*_data'))},
    }
    for name, query in (
        ('physical_gpu', '--query-gpu=name,uuid,memory.used,memory.total'),
        ('compute_apps', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory'),
    ):
        def gpu(query=query):
            completed = subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=query_timeout_seconds)
            return {'command': completed.args, 'exit_code': completed.returncode,
                    'stdout': completed.stdout, 'stderr': completed.stderr}
        queries[name] = gpu
    for name, query in queries.items():
        row = {'started_ns': time.time_ns(), 'started_monotonic_ns': time.monotonic_ns()}
        try:
            row['value'] = query()
        except Exception:
            row['error'] = traceback.format_exc()
        row['finished_ns'] = time.time_ns()
        row['finished_monotonic_ns'] = time.monotonic_ns()
        result['queries'][name] = row
    stat = Path('/proc/self/stat').read_text()
    result['pid_start_time'] = int(stat.rsplit(')', 1)[1].split()[19])
    parent_stat = (Path('/proc')/str(os.getppid())/'stat').read_text()
    result['parent_start_time'] = int(parent_stat.rsplit(')', 1)[1].split()[19])
    result['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    result['pid_namespace'] = os.readlink('/proc/self/ns/pid')
    result['finished_ns'] = time.time_ns()
    result['finished_monotonic_ns'] = time.monotonic_ns()
    return result


def timed_passes(torch, cell, kind, owners, forward, prepare, restore, *, repeats,
                 tensor_copy_bytes, evidence_bytes, observe):
    """Measure sibling-owner/whole passes and save each actual output after drain.

    ``prepare`` and ``restore`` run outside annotations. ``forward`` executes the
    unchanged native call once. This internal seam is used by native sample setup
    and fake regression tests; it does not construct a runtime or submit a job.
    """
    if kind not in attention.OWNERS or repeats < 2:
        raise ValueError('Known attention kind and at least two repeats required')
    if tensor_copy_bytes <= 0 or evidence_bytes <= 0:
        raise ValueError('Positive copy and evidence byte budgets required')
    cell = Path(cell)
    timed = cell/'timed_outputs'
    timed.mkdir(exist_ok=False)
    rows, errors = [], []

    def diagnostic(label, function):
        try:
            return function()
        except Exception:
            errors.append({'evidence': label, 'error': traceback.format_exc()})
            return None

    def persist(row, path):
        payload = (json.dumps(row, indent=2)+'\n').encode()
        used = sum(path.stat().st_size for path in cell.rglob('*') if path.is_file())
        if used+len(payload) > evidence_bytes:
            raise ValueError('Pass record would exceed the sample evidence byte budget')
        with path.open('xb') as stream:
            stream.write(payload)

    def retain(value, destination):
        logical_bytes = value.numel()*value.element_size()
        if logical_bytes > tensor_copy_bytes:
            raise ValueError('Timed output exceeds the explicit tensor-copy budget')
        cpu = value.detach().cpu()
        stream = io.BytesIO()
        torch.save(cpu, stream)
        payload = stream.getvalue()
        used = sum(path.stat().st_size for path in cell.rglob('*') if path.is_file())
        if used+len(payload) > evidence_bytes:
            raise ValueError('Timed output would exceed the sample evidence byte budget')
        with destination.open('xb') as output:
            output.write(payload)
        import hashlib
        return {'path': str(destination), 'sha256': hashlib.sha256(payload).hexdigest(),
                'serialized_bytes': len(payload), 'logical_bytes': logical_bytes,
                'dtype': str(value.dtype), 'shape': list(value.shape), 'stride': list(value.stride()),
                'scope': 'actual_timed_output_after_device_drain_not_numeric_reference'}

    for repeat in range(repeats):
        for annotated, name in ((True, 'owner_pass'), (False, 'forward_pass')):
            row = {'repeat': repeat, 'pass': name, 'kind': kind, 'sample': cell.name,
                   'timing_scope': 'profiled_kernel_only_not_clean_device_duration'}
            restore()
            arguments, host_metadata_ms = prepare()
            torch.cuda.synchronize()
            row['before'] = diagnostic(name+'_before', observe)
            owners.enabled, owners.calls = annotated, []
            row['forward_started_ns'] = time.time_ns()
            row['forward_started_monotonic_ns'] = time.monotonic_ns()
            started = time.perf_counter_ns()
            failure_phase = 'native_forward'
            try:
                try:
                    if annotated:
                        value = forward(*arguments)
                    else:
                        with torch.profiler.record_function('vidur_'+attention.WHOLE_FORWARD):
                            value = forward(*arguments)
                finally:
                    owners.enabled = False
                    row['forward_finished_ns'] = time.time_ns()
                    row['forward_finished_monotonic_ns'] = time.monotonic_ns()
                    row['host_profiled_launch_ms'] = (time.perf_counter_ns()-started)/1e6
                    row['host_metadata_build_ms'] = host_metadata_ms
                    row['owner_calls'] = list(owners.calls)
                    rows.append(row)
                # The price annotation has ended before drain, queries and DtoH copy.
                failure_phase = 'post_forward_device_drain'
                torch.cuda.synchronize()
            except BaseException:
                row['native_error'] = traceback.format_exc()
                row['native_failure_phase'] = failure_phase
                try:
                    diagnostic(name+'_native_failure', lambda: persist(
                        row, timed/f'{repeat:04d}_{name}_failure.json'))
                except BaseException:
                    errors.append({'evidence': name+'_native_failure', 'error': traceback.format_exc()})
                raise
            row['drained_ns'] = time.time_ns()
            row['drained_monotonic_ns'] = time.monotonic_ns()
            row['after'] = diagnostic(name+'_after', observe)
            if annotated and tuple(owners.calls) != attention.OWNERS[kind]:
                raise ValueError('Native owner order differs from the frozen attention contract')
            row['output'] = diagnostic(name+'_timed_output', lambda: retain(
                value, timed/f'{repeat:04d}_{name}.pt'))
            row['content_status'] = 'retained_not_numerically_verified' if row['output'] else 'UNVERIFIED'
            diagnostic(name+'_pass_record', lambda: persist(
                row, timed/f'{repeat:04d}_{name}.json'))
    return {'passes': rows, 'evidence_errors': errors, 'status': 'raw_unadmitted',
            'device_copy_activity': 'UNMEASURED', 'device_gaps': 'UNMEASURED',
            'clean_timing': 'UNVERIFIED', 'continuous_identity_exclusivity': 'UNVERIFIED'}


@contextmanager
def _sample_tracer(tracer, staging):
    """Preserve a body exception if the unchanged tracer also fails on exit."""
    body_error = None
    try:
        with tracer:
            try:
                yield tracer
            except BaseException as error:
                body_error, body_traceback = error, error.__traceback__
                body_details = traceback.format_exc()
                raise
    except BaseException as error:
        if body_error is not None and error is not body_error:
            record = {'body_error': body_details, 'tracer_cleanup_error': traceback.format_exc(),
                      'trace': tracer.trace_path, 'status': 'raw_unadmitted',
                      'retained_sample_cap_applies': False}
            try:
                with (staging/'tracer_cleanup_error.json').open('x') as stream:
                    json.dump(record, stream, indent=2)
                    stream.write('\n')
            except Exception:
                print(json.dumps(record)+'\n'+traceback.format_exc(), file=sys.stderr)
            finally:
                # Diagnostic I/O must not replace the native exception.
                raise body_error.with_traceback(body_traceback)
        raise


def measure_samples(torch, output, native, kind, selected, plan, module, spec, state,
                    initial, builder, table, record, groups, owners, warmups, repeats,
                    tensor_copy_bytes, evidence_bytes, query_timeout_seconds):
    """Run the frozen native sample setup with independently retained timed passes.

    Construction/cache binding/owner annotations are supplied by the existing
    collector. This is a new generation path; it cannot backfill old observations.
    """
    if warmups < 1 or repeats < 2 or kind not in attention.OWNERS:
        raise ValueError('Known kind, positive warmups and at least two repeats required')
    if min(tensor_copy_bytes, evidence_bytes, query_timeout_seconds) <= 0:
        raise ValueError('Positive copy, sample evidence and query budgets required')
    from vllm.forward_context import set_forward_context
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from frontier.profiling.utils.record_function_tracer import RecordFunctionTracer
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    staging_root = output.parent/(output.name+'_trace_staging')
    staging_root.mkdir(exist_ok=False)
    layer = module.attn
    window = plan['geometry']['W']
    hidden = native.model_config.hf_config.hidden_size
    rows = []
    for sample in selected:
        cell = output/sample['label']
        (cell/'profiler_traces').mkdir(parents=True, exist_ok=False)
        staging = staging_root/sample['label']
        (staging/'profiler_traces').mkdir(parents=True, exist_ok=False)
        tracer = RecordFunctionTracer(str(staging))
        trace_staging = {'trace': tracer.trace_path, 'sample': sample['label'],
                         'status': 'raw_unadmitted', 'retained_sample_cap_applies': False,
                         'retention': 'worker_local_raw_preserved_without_cleanup'}
        with (staging/'trace_staging.json').open('x') as stream:
            json.dump(trace_staging, stream, indent=2)
            stream.write('\n')
        for tensor, value in zip(state, initial):
            tensor.copy_(value)
        generator = torch.Generator(device='cuda').manual_seed(sample['content_seed'])
        schedule = attention.chunks(sample, window)
        previous = 0

        def step(count):
            positions = torch.arange(previous, previous+count, device='cuda')
            slots = table[0, positions//spec.block_size].long()*spec.block_size+positions%spec.block_size
            starts = torch.tensor([0, count], dtype=torch.int32, device='cpu')
            lengths = torch.tensor([previous+count], dtype=torch.int32, device='cpu')
            common = CommonAttentionMetadata(query_start_loc=starts.cuda(), query_start_loc_cpu=starts,
                seq_lens=lengths.cuda(), num_reqs=1, num_actual_tokens=count, max_query_len=count,
                max_seq_len=previous+count, block_table_tensor=table, slot_mapping=slots,
                positions=positions, seq_lens_cpu_upper_bound=lengths)
            started = time.perf_counter_ns()
            with torch.device('cpu'):
                metadata = builder.build(0, common, fast_build=False)
            return (positions, slots, metadata), (time.perf_counter_ns()-started)/1e6

        def forward(positions, slots, metadata, inputs):
            with set_forward_context({layer.layer_name: metadata}, native, num_tokens=inputs.shape[0],
                                     slot_mapping={layer.layer_name: slots}):
                return module(positions, inputs)

        for count in schedule[:-1]:
            inputs = torch.randn((count, hidden), generator=generator, device='cuda', dtype=torch.bfloat16)
            arguments, _ = step(count)
            forward(*arguments, inputs)
            previous += count
        inputs = torch.randn((sample['query'], hidden), generator=generator, device='cuda', dtype=torch.bfloat16)
        prefix_state = [tensor.clone() for tensor in state]

        def restore():
            for tensor, value in zip(state, prefix_state):
                tensor.copy_(value)

        def prepare():
            arguments, host_ms = step(sample['query'])
            return (*arguments, inputs), host_ms

        for _ in range(warmups):
            restore()
            arguments, _ = prepare()
            owners.enabled = False
            forward(*arguments)
        torch.cuda.synchronize()
        with _sample_tracer(tracer, staging):
            evidence = timed_passes(torch, cell, kind, owners, forward, prepare, restore,
                repeats=repeats, tensor_copy_bytes=tensor_copy_bytes, evidence_bytes=evidence_bytes,
                observe=lambda: resource_snapshot(query_timeout_seconds))
        raw_trace = Path(tracer.trace_path)
        used = sum(path.stat().st_size for path in cell.rglob('*') if path.is_file())
        if used+raw_trace.stat().st_size > evidence_bytes:
            raise ValueError('Profiler trace would exceed the sample evidence byte budget')
        retained_trace = cell/'profiler_traces'/raw_trace.name
        with retained_trace.open('xb') as stream:
            stream.write(raw_trace.read_bytes())
        tracer.trace_path = str(retained_trace)
        stats = tracer.get_operation_time_stats()
        if set(stats) != set(attention.OWNERS[kind]+(attention.WHOLE_FORWARD,)) or any(
                value['count'] != repeats for value in stats.values()):
            raise ValueError('Unexpected annotated scopes or repeat counts')
        row = {'kind': kind, 'sample': sample, 'schedule': schedule, 'groups': groups,
               'dispatch': record, 'evidence': evidence, 'kernel_only_stats': stats,
               'kernel_only_reconciliation': attention.reconcile(stats, kind),
               'trace': tracer.trace_path, 'trace_staging': trace_staging, 'status': 'raw_unadmitted',
               'coverage': 'local_eager_prefill_not_graph_or_full_role',
               'sample_evidence_budget_bytes': evidence_bytes}
        payload = (json.dumps(row, indent=2, default=float)+'\n').encode()
        used = sum(path.stat().st_size for path in cell.rglob('*') if path.is_file())
        if used+len(payload) > evidence_bytes:
            raise ValueError('Completed trace and sample record exceed the sample evidence budget')
        with (cell/'observation.json').open('xb') as stream:
            stream.write(payload)
        rows.append(row)
    return rows

