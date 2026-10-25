"""CPU check of native_scheduler_lifecycle against vLLM 0.28's AsyncScheduler.

Run in the vLLM 0.28 environment. The NATIVE vLLM checkout supplies the
KV-connector scheduler test helpers (``tests/v1/kv_connector/unit/utils.py``).
``vllm`` is imported from the installed package before the checkout enters
``sys.path``, so the helpers' ``vllm`` imports resolve to the installed package:

    python tests/comparison/calibration/check_native_scheduler_lifecycle.py \
        --vllm-checkout <NATIVE vLLM checkout> <new output directory>

The check drives the remote-prefill (D) and remote-decode (P) lifecycles of
the vLLM unit tests once on AsyncScheduler and once on the observer, requires
identical scheduler results, checks the recorded events, injects observer
failures, and times steady decode steps with and without the observer.
"""

import argparse
import copy
import dataclasses
import enum
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Results carry wall and monotonic times; they are not scheduler decisions.
TIME_FIELDS = {'timestamp', 'arrival_time', 'scheduler_clock'}
# Eight 8-token prompts fill the helpers' 64-token step budget, so every
# later step decodes all eight requests.
TIMING_REQUESTS, TIMING_PROMPT_TOKENS, TIMING_STEPS, TIMING_ROUNDS = 8, 8, 2000, 3
# facebook/opt-125m metadata; the scheduler reads only the model config.
OPT_125M_CONFIG = dict(architectures=['OPTForCausalLM'], model_type='opt',
    activation_function='relu', bos_token_id=2, eos_token_id=2, pad_token_id=1,
    do_layer_norm_before=True, ffn_dim=3072, hidden_size=768, word_embed_proj_dim=768,
    max_position_embeddings=2048, num_attention_heads=12, num_hidden_layers=12,
    vocab_size=50272, torch_dtype='float16')


def plain(value):
    """Comparable form of scheduler results, without time fields."""
    import msgspec
    if isinstance(value, msgspec.Struct):
        return {name: plain(getattr(value, name)) for name in value.__struct_fields__
                if name not in TIME_FIELDS}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: plain(getattr(value, field.name))
                for field in dataclasses.fields(value) if field.name not in TIME_FIELDS}
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted((plain(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, enum.Enum):
        return value.name
    if hasattr(value, '__dict__') and not isinstance(value, type):
        fields = {name: item for name, item in vars(value).items() if name not in TIME_FIELDS}
        # NIXL send deadlines are clock values; their request ids are compared.
        if 'reqs_to_send' in fields:
            fields['reqs_to_send'] = sorted(fields['reqs_to_send'])
        return {'type': type(value).__name__, **plain(fields)}
    return value


def build(scheduler_cls, helpers, model_dir):
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.structured_output import StructuredOutputManager
    config = helpers.create_vllm_config(model=str(model_dir))
    config.scheduler_config.async_scheduling = True
    config.kv_transfer_config.engine_id = 'check-engine'
    block_size, num_blocks = config.cache_config.block_size, 10000
    config.cache_config.num_gpu_blocks = num_blocks
    kv_cache_config = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(['layer'], FullAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32))])
    return scheduler_cls(vllm_config=config, kv_cache_config=kv_cache_config, log_stats=True,
        structured_output_manager=StructuredOutputManager(config), block_size=block_size)


def transfer_output(**fields):
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
    output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    output.kv_connector_output = KVConnectorOutput(**fields)
    return output


def remote_prefill(scheduler, helpers):
    """D side, as in test_remote_prefill_lifecycle.test_basic_lifecycle."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import NixlKVConnectorStats
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
    block_size = scheduler.cache_config.block_size
    request = helpers.create_request(request_id=1, block_size=block_size,
        num_tokens=int(block_size * 2.5), do_remote_prefill=True)
    stats = NixlKVConnectorStats()
    stats.record_transfer(SimpleNamespace(xferDuration=12300, postDuration=400,
        totalBytes=4096, descCount=3))
    scheduler.add_request(request)
    outputs = [request.num_prompt_tokens]
    for model_output in (lambda: EMPTY_MODEL_RUNNER_OUTPUT,
            lambda: transfer_output(finished_recving={request.request_id}, kv_connector_stats=stats),
            lambda: helpers.create_model_runner_output([request]),
            lambda: helpers.create_model_runner_output([request], use_eos=True)):
        step = scheduler.schedule()
        outputs.append((plain(step), plain(scheduler.update_from_output(step, model_output()))))
    outputs.append(plain(scheduler.schedule()))
    return outputs


def remote_decode(scheduler, helpers):
    """P side, as in test_remote_decode_lifecycle.test_basic_lifecycle."""
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
    block_size = scheduler.cache_config.block_size
    request = helpers.create_request(request_id=1, block_size=block_size, max_tokens=1,
        num_tokens=int(block_size * 2.5), do_remote_decode=True)
    scheduler.add_request(request)
    outputs = [request.num_prompt_tokens]
    for model_output in (lambda: helpers.create_model_runner_output(reqs=[request]),
            lambda: EMPTY_MODEL_RUNNER_OUTPUT,
            lambda: transfer_output(finished_sending={request.request_id})):
        step = scheduler.schedule()
        outputs.append((plain(step), plain(scheduler.update_from_output(step, model_output()))))
    return outputs


def records(directory):
    files = sorted((directory / 'scheduler_lifecycle').glob('dp*_pid*.jsonl'))
    if len(files) != 1:
        raise ValueError(f'Expected one lifecycle file in {directory}, found {len(files)}')
    return [json.loads(line) for line in files[0].read_text().splitlines()]


def run_drive(drive, scheduler_cls, helpers, model_dir, directory, role, failures=()):
    os.environ['FRONTIER_NATIVE_SERVING_EVIDENCE_DIR'] = str(directory)
    os.environ['FRONTIER_NATIVE_ROLE'] = role
    try:
        scheduler = build(scheduler_cls, helpers, model_dir)
        for name in failures:
            def fail(*args, name=name):
                raise RuntimeError(f'injected observer failure in {name}')
            setattr(scheduler, name, fail)
        result = drive(scheduler, helpers)
        scheduler.shutdown()
        return result
    finally:
        del os.environ['FRONTIER_NATIVE_SERVING_EVIDENCE_DIR'], os.environ['FRONTIER_NATIVE_ROLE']


def check_prefill_records(rows, prompt_tokens):
    by_event = lambda name: [row for row in rows if row['event'] == name]
    start, = by_event('start')
    add, = by_event('add')
    schedules, updates = by_event('schedule'), by_event('update')
    stop, = by_event('stop')
    rid = add['request_id']
    k1 = [row for row in schedules if row.get('new')]
    return dict(
        start=start['role'] == 'decode' and start['async_scheduling'] is True
            and start['connector'] == 'NixlConnector' and start['dp_rank'] == 0,
        add=add['do_remote_prefill'] is True and add['remote_request_id'] == 'prefill-1'
            and add['prompt_tokens'] == prompt_tokens,
        waiting_remote_kv=[row.get('waiting_remote_kv') for row in schedules[:2]] == [[rid], [rid]]
            and schedules[0].get('recv') == [rid] and 'tokens' not in schedules[0],
        finished_recving=[row['step'] for row in updates if row.get('finished_recving') == [rid]]
            == [schedules[1]['step']] and by_event('update')[1]['kv_stats']['transfer_duration'] == [0.0123],
        promotion_k1=len(k1) == 1 and k1[0]['new'] == [[rid, prompt_tokens - 1, 1]]
            and k1[0]['step'] == schedules[2]['step'],
        first_token=[row['step'] for row in updates if row.get('first_token') == [rid]]
            == [schedules[2]['step']],
        done=[row['step'] for row in updates if row.get('done') == [rid]] == [schedules[3]['step']],
        steps_matched=[row['step'] for row in updates] == [row['step'] for row in schedules[:4]],
        no_errors=not by_event('error') and stop['errors'] == 0
            and stop['records'] == len(rows) - 1)


def check_decode_records(rows, prompt_tokens):
    by_event = lambda name: [row for row in rows if row['event'] == name]
    start, = by_event('start')
    add, = by_event('add')
    schedules, updates = by_event('schedule'), by_event('update')
    rid = add['request_id']
    return dict(
        start=start['role'] == 'prefill',
        add=add['do_remote_decode'] is True and add['max_tokens'] == 1,
        prefill=schedules[0].get('new') == [[rid, 0, prompt_tokens]],
        first_token_and_done=updates[0].get('first_token') == [rid] and updates[0].get('done') == [rid],
        send=any(rid in row.get('send', ()) for row in schedules),
        finished=schedules[1].get('finished') == [rid],
        finished_sending=[row['step'] for row in updates if row.get('finished_sending') == [rid]]
            == [schedules[2]['step']],
        no_errors=not by_event('error'))


def steady_decode(scheduler, helpers, steps):
    block_size = scheduler.cache_config.block_size
    requests = [helpers.create_request(request_id=100 + index, block_size=block_size,
        num_tokens=TIMING_PROMPT_TOKENS, max_tokens=steps + 8) for index in range(TIMING_REQUESTS)]
    for request in requests:
        scheduler.add_request(request)
    begin = time.perf_counter()
    for _ in range(steps):
        step = scheduler.schedule()
        scheduler.update_from_output(step, helpers.create_model_runner_output(requests))
    return (time.perf_counter() - begin) / steps


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--vllm-checkout', type=Path, required=True)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import vllm
    checkout = str(args.vllm_checkout.resolve())
    sys.path.insert(0, checkout)
    try:
        import tests.v1.kv_connector.unit.utils as helpers
    finally:
        sys.path.remove(checkout)
    if (not Path(helpers.__file__).resolve().is_relative_to(checkout)
            or Path(vllm.__file__).resolve().is_relative_to(checkout)):
        raise ValueError(f'Unexpected sources: helpers {helpers.__file__}, vllm {vllm.__file__}')
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from native_scheduler_lifecycle import LifecycleObservedScheduler
    model_dir = args.output / 'model'
    model_dir.mkdir()
    (model_dir / 'config.json').write_text(json.dumps(OPT_125M_CONFIG))

    result = dict(vllm=vllm.__version__, vllm_file=vllm.__file__, helpers=helpers.__file__,
        python=sys.version, checks={})
    checks = result['checks']
    for name, drive, verify, role in (
            ('remote_prefill', remote_prefill, check_prefill_records, 'decode'),
            ('remote_decode', remote_decode, check_decode_records, 'prefill')):
        native = run_drive(drive, AsyncScheduler, helpers, model_dir, args.output / f'{name}_native', role)
        observed = run_drive(drive, LifecycleObservedScheduler, helpers, model_dir,
            args.output / f'{name}_observed', role)
        injected = run_drive(drive, LifecycleObservedScheduler, helpers, model_dir,
            args.output / f'{name}_injected', role, failures=('_record_add', '_transfer_fields'))
        checks[f'{name}.identical_results'] = observed == native
        checks[f'{name}.native_records_absent'] = not (args.output / f'{name}_native').exists()
        for item, passed in verify(records(args.output / f'{name}_observed'), native[0]).items():
            checks[f'{name}.records.{item}'] = passed
        rows = records(args.output / f'{name}_injected')
        errors = [row for row in rows if row['event'] == 'error']
        checks[f'{name}.injected.identical_results'] = injected == native
        checks[f'{name}.injected.errors_recorded'] = (
            sorted({row['where'] for row in errors}) == ['add', 'transfer']
            and len(errors) <= 5 and rows[-1]['event'] == 'stop' and rows[-1]['errors'] >= 2
            and any(row['event'] == 'update' for row in rows))

    timing = {'native': [], 'observed': []}
    for round_index in range(TIMING_ROUNDS):
        for label, scheduler_cls in (('native', AsyncScheduler), ('observed', LifecycleObservedScheduler)):
            directory = args.output / f'timing_{label}_{round_index}'
            seconds = run_drive(lambda scheduler, helpers: steady_decode(scheduler, helpers, TIMING_STEPS),
                scheduler_cls, helpers, model_dir, directory, 'decode')
            timing[label].append(seconds * 1e6)
    native_us, observed_us = statistics.median(timing['native']), statistics.median(timing['observed'])
    result['timing'] = dict(requests=TIMING_REQUESTS, steps=TIMING_STEPS, rounds=TIMING_ROUNDS,
        per_step_us=timing, median_native_us=native_us, median_observed_us=observed_us,
        median_added_us=observed_us - native_us)
    result['verdict'] = 'PASS' if all(checks.values()) else 'FAIL'
    (args.output / 'check_result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(verdict=result['verdict'],
        failed=[name for name, passed in checks.items() if not passed],
        timing_us=dict(native=native_us, observed=observed_us)), sort_keys=True))
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
