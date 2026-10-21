"""Observe MRv2 serving forwards after DP dispatch without GPU reads or fences."""

from dataclasses import asdict
from collections import Counter, defaultdict
import sys


SOURCE_MODULES = (
    'vllm.v1.worker.gpu.dp_utils',
    'vllm.v1.worker.gpu.cudagraph_utils',
    'vllm.v1.worker.gpu.async_utils',
    'vllm.v1.worker.gpu.input_batch',
    'vllm.v1.worker.gpu.block_table',
    'vllm.v1.worker.gpu.model_states.default',
    'vllm.v1.worker.gpu.attn_utils',
    'vllm.models.step5.nvidia.model_state',
)


def install(module, observer, wrap):
    """Wrap existing host seams; native calls, results and exceptions are unchanged."""
    for native in (module.dispatch_cg_and_sync_dp, module.StepTimingCollector):
        name = native.__module__
        observer.verify_module(sys.modules[name], observer.config['schema_source_hashes'][name], 'native_schema')
    sequence = 0

    def before(runner, arguments):
        if (getattr(observer.local, 'worker_startup', False)
                or not getattr(observer.local, 'worker_execution_kind', None)):
            return None  # Startup capture/profiling is not serving execution.
        context = dict(identity=observer.identity(runner), runner=runner,
                       dummy_run=bool(arguments.get('dummy_run', False)),
                       is_profile=bool(arguments.get('is_profile', False)),
                       worker_execution_kind=observer.local.worker_execution_kind)
        observer.local.runtime = context
        return context

    def dispatched(manager, result, arguments, token):
        context = getattr(observer.local, 'runtime', None)
        if context is not None:
            descriptor, _ = result
            fields = asdict(descriptor)
            fields['cg_mode'] = descriptor.cg_mode.name
            context['descriptor'] = fields
            context['local_num_tokens'] = arguments['num_tokens']
            context['local_num_reqs'] = arguments['num_reqs']

    def record(timing, arguments):
        nonlocal sequence
        context = getattr(observer.local, 'runtime', None)
        if context is None:
            return
        batch = arguments['input_batch']
        sequence += 1
        fields = dict(context['identity'], forward_index=sequence,
            dummy_run=context['dummy_run'], is_profile=context['is_profile'],
            worker_execution_kind=context['worker_execution_kind'],
            descriptor=context['descriptor'], local_num_tokens=context['local_num_tokens'],
            local_num_reqs=context['local_num_reqs'],
            num_tokens=batch.num_tokens, num_tokens_after_padding=batch.num_tokens_after_padding,
            num_reqs=batch.num_reqs, num_reqs_after_padding=batch.num_reqs_after_padding,
            query_lengths=batch.num_scheduled_tokens.tolist(),
            context_lengths=batch.num_computed_tokens_np.tolist(),
            native_request_ids=list(batch.req_ids), has_prefill=bool(batch.has_prefill),
            full_cudagraph=bool(arguments['full_cudagraph']),
            metadata=context.get('metadata'), observer_fence=False,
            boundary='native_step_timing_record_batch_before_model_forward')
        context['forward'] = fields
        observer.emit('native_forward_start', **fields)

    def after(runner, result, arguments, context):
        if context is not None and 'forward' in context:
            fields = context['forward']
            observer.emit('native_forward_end', **context['identity'],
                forward_index=fields['forward_index'],
                boundary='native_execute_model_return_no_observer_cuda_fence',
                observer_fence=False)

    wrap(module.GPUModelRunner, 'execute_model', observer, before=before, after=after, runtime_scope=True)
    wrap(module, 'dispatch_cg_and_sync_dp', observer, after=dispatched)
    wrap(module.StepTimingCollector, 'record_batch', observer, before=record)


def bind_model_state(worker, observer, wrap):
    cls = type(worker.model_runner.model_state)
    for name in SOURCE_MODULES:
        observer.verify_module(sys.modules[name], observer.config['schema_source_hashes'][name], 'native_schema')

    def prepared(state, result, arguments, token):
        context = getattr(observer.local, 'runtime', None)
        if context is None:
            return
        context['metadata'] = dict(layer_count=len(result),
            layers_with_metadata=sum(value is not None for value in result.values()),
            for_capture=bool(arguments.get('for_capture', False)),
            block_table_shapes=[list(value.shape) for value in arguments['block_tables']],
            slot_mapping_shape=list(arguments['slot_mappings'].shape))

    wrap(cls, 'prepare_attn', observer, after=prepared)


def judge(records, config, check):
    """Reconcile every observed serving forward with native dispatch and useful work."""
    starts, ends = defaultdict(list), defaultdict(list)
    useful, scheduled = Counter(), Counter()
    rank_key = lambda row: (row['role'], row['dp_rank'], row['tp_rank'])
    for row in records:
        rank = rank_key(row) if row.get('tp_rank') is not None else None
        if row['event'] == 'native_forward_start':
            starts[rank].append(row)
        elif row['event'] == 'native_forward_end':
            ends[rank].append(row)
        elif row['event'] in {'source_compute', 'target_compute'}:
            scheduled[rank, row['request_id']] += row['num_scheduled_tokens']
    engine = config['engine']
    ranks = {(role, dp, tp) for role in ('prefill', 'decode')
             for dp in range(engine['data_parallel_size'])
             for tp in range(engine['tensor_parallel_size'])}
    check.require(set(starts) == set(ends) == ranks, 'Incomplete MRv2 forward census across roles/DP/TP')
    signatures, totals = {}, []
    for rank, rows in starts.items():
        rows.sort(key=lambda row: row['forward_index'])
        indices = [row['forward_index'] for row in rows]
        completed = sorted(row['forward_index'] for row in ends[rank])
        check.expect(indices == list(range(1, len(rows)+1)) == completed,
                     f'{rank}: duplicate, missing or incomplete native model forward')
        signatures[rank] = []
        modes = Counter()
        for row in rows:
            queries, contexts, ids = row['query_lengths'], row['context_lengths'], row['native_request_ids']
            check.expect(len(queries) == len(contexts) == len(ids) == row['num_reqs'],
                         f'{rank}: incomplete per-request native input arrays')
            check.expect(sum(queries) == row['num_tokens'] <= row['num_tokens_after_padding'],
                         f'{rank}: native input token accounting differs')
            check.expect(row['descriptor']['num_tokens'] == row['num_tokens_after_padding'],
                         f'{rank}: input disagrees with post-DP execution descriptor')
            mode = row['descriptor']['cg_mode']
            check.expect(row['full_cudagraph'] == (mode == 'FULL'), f'{rank}: actual FULL dispatch mismatch')
            check.expect(row['observer_fence'] is False and row['is_profile'] is False,
                         f'{rank}: non-serving or fenced observation entered forward census')
            metadata = row['metadata']
            check.require(metadata and metadata['layers_with_metadata'] == metadata['layer_count'] > 0,
                          f'{rank}: actual serving attention metadata is missing')
            if row['dummy_run']:
                check.expect(not any(contexts), f'{rank}: native dummy unexpectedly owns token history')
            else:
                for req, count in zip(ids, queries):
                    useful[rank, req] += count
            modes[mode, row['dummy_run']] += 1
            signatures[rank].append((row['dummy_run'], row['descriptor'], queries, contexts,
                                     metadata, [] if row['dummy_run'] else ids))
        totals.append(dict(role=rank[0], dp_rank=rank[1], tp_rank=rank[2],
            forward_count=len(rows), useful_tokens=sum(row['num_tokens'] for row in rows if not row['dummy_run']),
            dummy_tokens=sum(row['num_tokens'] for row in rows if row['dummy_run']),
            modes=[dict(cg_mode=mode, dummy_run=dummy, forwards=count)
                   for (mode, dummy), count in sorted(modes.items())]))
    check.expect(useful == scheduled, 'Useful model inputs disagree with independent native scheduler witnesses')
    for role in ('prefill', 'decode'):
        for dp in range(engine['data_parallel_size']):
            baseline = signatures.get((role, dp, 0))
            for tp in range(1, engine['tensor_parallel_size']):
                check.require(baseline is not None and signatures.get((role, dp, tp)) == baseline,
                              f'{role}/DP{dp}: TP ranks executed different forward inputs or graph modes')
    check.details.update(ranks=totals, timing_eligible=False,
                         completion_scope='native host return; native runtime/client gates establish healthy completion')
