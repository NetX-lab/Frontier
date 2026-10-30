"""CPU check of DP wave hooks against the installed vLLM 0.28 engine loop.

Only scheduler/executor work and the existing collective are test doubles. The
busy loop, queue processing, add/wave handling and 32-step logic are native.
"""
import argparse
import contextlib
import hashlib
import json
import os
import queue
import statistics
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ITERATIONS = 4000
ROUNDS = 5
OVERHEAD_LIMIT_US = 50.0
RUN_SECONDS = 150
ENGINES_PER_ROLE = 4
DUMMY_SECONDS = .005


def drive(core, log=None, iterations=65, inject=False, blocked=True, rank=0, bootstrap=False):
    from vllm.config import ParallelConfig
    from vllm.v1.core.sched.scheduler import PauseState
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.core import EngineShutdownState
    from native_dp_wave import EngineWaveLog
    engine = core.__new__(core)
    requests, decisions = [], []
    scheduler = SimpleNamespace(_lifecycle=log, pause_state=PauseState.UNPAUSED,
        requests=requests, has_requests=lambda: bool(requests),
        has_unfinished_requests=lambda: bool(requests), add_request=requests.append)
    engine.scheduler = scheduler
    engine.input_queue, engine.output_queue, engine.aborts_queue = queue.Queue(), queue.Queue(), queue.Queue()
    engine.current_wave = engine.step_counter = 0
    engine.engine_index = engine.dp_rank = rank
    engine.dp_group = object()
    engine.engines_running = engine.pending_pause = engine.ignore_start_dp_wave = False
    engine.shutdown_state = EngineShutdownState.RUNNING
    engine.has_coordinator, engine.process_input_queue_block = True, False
    engine.publish_dp_lb_stats = False
    engine.eep_scaling_state = None
    engine.batch_queue = None
    engine.model_executor = SimpleNamespace(is_sleeping=False)
    engine._notify_idle_state_callbacks = lambda: None
    engine.post_step = lambda result: decisions.append(('step', result))
    engine.capture_iteration_details = lambda _: contextlib.nullcontext(None)
    engine.execute_dummy_batch = lambda: decisions.append(('dummy',))
    def step():
        executed = bool(requests)
        requests.clear()
        return {}, executed
    engine.step_fn = step
    count = 0
    input_wait = threading.Event()
    native_wait = engine.input_queue.not_empty.wait
    def wait(*args, **kwargs):
        input_wait.set()
        return native_wait(*args, **kwargs)
    engine.input_queue.not_empty.wait = wait
    def input_thread():
        if not input_wait.wait(10):
            raise RuntimeError('Driven idle loop did not block')
        engine.input_queue.put_nowait((EngineCoreRequestType.START_DP_WAVE, (0, -1)))
    thread = None
    def boundary():
        nonlocal count, thread
        if count == 1:
            engine.process_input_queue_block = blocked
            if blocked:
                thread = threading.Thread(target=input_thread)
                thread.start()
            else:
                engine.input_queue.put_nowait((EngineCoreRequestType.START_DP_WAVE, (0, -1)))
        if count == 33:
            request = SimpleNamespace(request_id='stale-wave', pooling_params=None,
                kv_transfer_params=None, ec_transfer_params=None, abort_immediately=False)
            engine.input_queue.put_nowait((EngineCoreRequestType.ADD, (request, 0)))
            engine.process_input_queue_block = False
        count += 1
        return count <= iterations
    engine._handle_shutdown = boundary
    observer = EngineWaveLog(engine) if log and not bootstrap else None
    native_sync = lambda *a, **k: (iterations > 65, False)
    with patch.object(ParallelConfig, 'sync_dp_state', staticmethod(native_sync)):
        if observer:
            observer.install()
            if inject:
                native_record = observer.record
                def record(event, *args, **kwargs):
                    if event in ('engine_dummy_batch', 'handle_client_request', 'engine_add'):
                        raise ValueError('Injected observation failure')
                    return native_record(event, *args, **kwargs)
                observer.record = record
        started = time.perf_counter()
        try:
            core.run_busy_loop(engine)
        except SystemExit as error:
            exit_value = error.code
        finally:
            if observer:
                observer.observe('finish', observer.finish)
        elapsed = time.perf_counter() - started
    if thread:
        thread.join(10)
        assert not thread.is_alive()
    outputs = []
    while not engine.output_queue.empty():
        client, output = engine.output_queue.get_nowait()
        outputs.append((client, output.wave_complete, output.start_wave))
    return dict(decisions=decisions, outputs=outputs, current_wave=engine.current_wave,
        step_counter=engine.step_counter, engines_running=engine.engines_running, exit=exit_value), elapsed


def worker_check(output):
    import torch
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu import dp_utils
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from native_dp_wave_worker import WorkerWaveLog
    class Worker:
        vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_rank=0))
        def execute_model(self, scheduler_output):
            dp_utils.sync_cudagraph_and_dp_padding(None, descriptor, 6144, 1, None, 4, 0)
            return object_result
        def execute_dummy_batch(self):
            dp_utils.sync_cudagraph_and_dp_padding(None, descriptor, 6144, 1, None, 4, 0)
            return object_result
        def sample_tokens(self, grammar):
            return object_result
        def compile_or_warm_up_model(self):
            return object_result
        def shutdown(self):
            return object_result
    object_result = object()
    worker = Worker()
    observer = WorkerWaveLog(worker, output)
    descriptor = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=6144, num_reqs=1)
    reduced = [[6144, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [-1, -1, -1, -1]]
    def reduce(tensor, **kwargs):
        tensor.copy_(torch.tensor(reduced, dtype=torch.int32))
        return object_result
    with patch.object(dp_utils, 'get_dp_group', return_value=SimpleNamespace(cpu_group=object())), \
         patch.object(dp_utils.dist, 'all_reduce', reduce), \
         patch.object(dp_utils, 'sync_cudagraph_and_dp_padding', dp_utils.sync_cudagraph_and_dp_padding):
        native = dp_utils.sync_cudagraph_and_dp_padding(None, descriptor, 6144, 1, None, 4, 0)
        observer.install(dp_utils)
        observed = dp_utils.sync_cudagraph_and_dp_padding(None, descriptor, 6144, 1, None, 4, 0)
        assert observed[0] == native[0] and torch.equal(observed[1], native[1])
        scheduled = SimpleNamespace(total_num_scheduled_tokens=6144, num_scheduled_tokens={'r': 6144})
        assert worker.execute_model(scheduled) is object_result
        assert worker.execute_dummy_batch() is object_result
        assert worker.sample_tokens(None) is object_result
        failure = RuntimeError('native exception identity')
        worker.native_failure = lambda: (_ for _ in ()).throw(failure)
        observer.span(worker, 'native_failure')
        try:
            worker.native_failure()
        except RuntimeError as error:
            assert error is failure
        else:
            raise AssertionError('Native exception suppressed')
        with patch.object(observer, 'record', side_effect=ValueError('Injected worker observation failure')):
            assert worker.execute_model(scheduled) is object_result
            assert dp_utils.sync_cudagraph_and_dp_padding(None, descriptor, 6144, 1, None, 4, 0)[0] == native[0]
        assert worker.compile_or_warm_up_model() is object_result
        assert worker.shutdown() is object_result
    path = next((output / 'worker_dp_wave').glob('*.jsonl'))
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert all({'dp_padding_sync','dp_cpu_all_reduce'} <= row['calls'].keys() for row in rows
        if row['event'] in ('worker_execute_model','worker_dummy_batch'))
    rows.extend(dict(event=event, **call) for row in list(rows) for event,call in row.get('calls',{}).items())
    kinds = {row['event'] for row in rows}
    assert {'dp_padding_sync', 'dp_cpu_all_reduce', 'worker_execute_model', 'worker_dummy_batch',
            'worker_sample_tokens', 'worker_warmup_complete', 'error', 'stop'} <= kinds
    record = next(r for r in rows if r['event'] == 'dp_cpu_all_reduce')
    assert record['reduced_values'] == reduced and record['end_mono'] >= record['mono']
    return len(rows)



def worker_timing(output):
    """Measure TP0 observer CPU cost separately from the engine process."""
    import torch
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu import dp_utils
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from native_dp_wave_worker import WorkerWaveLog
    descriptor = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=1, num_reqs=1)
    reduced = torch.tensor([[1] * ENGINES_PER_ROLE, [0] * ENGINES_PER_ROLE,
        [1] * ENGINES_PER_ROLE, [-1] * ENGINES_PER_ROLE], dtype=torch.int32)
    class Worker:
        vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_rank=0))
        def execute_model(self, output):
            return None
        def execute_dummy_batch(self):
            return dp_utils.sync_cudagraph_and_dp_padding(None, descriptor, 1, 1, 1, ENGINES_PER_ROLE, 0)
        def sample_tokens(self, grammar):
            return None
        def compile_or_warm_up_model(self):
            return None
        def shutdown(self):
            return None
    times = {'native': [], 'observed': []}
    for round_index in range(ROUNDS):
        for label in times:
            worker = Worker()
            with patch.object(dp_utils, 'get_dp_group', return_value=SimpleNamespace(cpu_group=object())), \
                 patch.object(dp_utils.dist, 'all_reduce', lambda tensor, **kwargs: tensor.copy_(reduced)), \
                 patch.object(dp_utils, 'sync_cudagraph_and_dp_padding', dp_utils.sync_cudagraph_and_dp_padding):
                observer = WorkerWaveLog(worker, output / f'{round_index}_{label}') if label=='observed' else None
                if observer:
                    observer.install(dp_utils)
                    worker.compile_or_warm_up_model()
                begin = time.perf_counter()
                for _ in range(ITERATIONS):
                    worker.execute_dummy_batch()
                times[label].append((time.perf_counter()-begin)*1e6/ITERATIONS)
                if observer:
                    observer.log.close()
    added = statistics.median(times['observed']) - statistics.median(times['native'])
    return dict(per_forward_us=times, added_median_us=added,
        limitation='Separate process CPU measurement with the existing reduction mocked; sum with engine cost is a serial estimate, not observed native wall time')

def worker_bootstrap_check(output):
    from native_dp_wave import SCHEDULER_CLASS
    from native_dp_wave_worker import install_worker_hooks
    from vllm.v1.worker.gpu import dp_utils
    result = object()
    for role in ('prefill','decode'):
        for rank in range(ENGINES_PER_ROLE):
            class Worker:
                def init_device(self):
                    return result
                def execute_model(self, output):
                    return result
                def execute_dummy_batch(self):
                    return result
                def sample_tokens(self, grammar):
                    return result
                def compile_or_warm_up_model(self):
                    return result
                def shutdown(self):
                    return result
            native = Worker.init_device
            install_worker_hooks(Worker)
            assert Worker.init_device is native
            directory = output / role / f'dp{rank}'
            with patch.dict(os.environ, FRONTIER_NATIVE_SERVING_EVIDENCE_DIR=str(directory), FRONTIER_NATIVE_ROLE=role), \
                 patch('vllm.distributed.get_tp_group', return_value=SimpleNamespace(rank_in_group=1), create=True) as tp, \
                 patch.object(dp_utils,'sync_cudagraph_and_dp_padding',dp_utils.sync_cudagraph_and_dp_padding), \
                 patch.object(dp_utils.dist,'all_reduce',dp_utils.dist.all_reduce):
                install_worker_hooks(Worker)
                worker = Worker()
                worker.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_rank=rank),
                    scheduler_config=SimpleNamespace(scheduler_cls='native_scheduler_lifecycle.LifecycleObservedScheduler'))
                assert worker.init_device() is result and not hasattr(worker,'_dp_wave_log')
                worker.vllm_config.scheduler_config.scheduler_cls = SCHEDULER_CLASS
                assert worker.init_device() is result and not hasattr(worker,'_dp_wave_log')
                tp.return_value.rank_in_group = 0
                assert worker.init_device() is result and hasattr(worker,'_dp_wave_log')
                assert worker.shutdown() is result
            rows = [json.loads(line) for path in directory.glob('worker_dp_wave/*.jsonl') for line in path.read_text().splitlines()]
            assert rows[0]['dp_rank']==rank and rows[0]['role']==role and rows[0]['tp_rank']==0
            assert not any(row['event']=='error' for row in rows)
    return ENGINES_PER_ROLE * 2


def volume(lifecycle):
    lanes = []
    for path in sorted((lifecycle / 'nodes').glob('*/scheduler_lifecycle/*.jsonl')):
        with path.open() as stream:
            rows = [json.loads(line) for line in stream]
        start = rows[0]
        schedules = [row for row in rows if row['event'] == 'schedule']
        tokens = [row for row in schedules if row.get('tokens')]
        # Existing lifecycle records omit dummy-only iterations. Successive
        # schedule calls on active D lanes measure the model/engine cadence.
        gaps = [b['mono']-a['mono'] for a,b in zip(tokens, tokens[1:]) if 0 < b['mono']-a['mono'] < .1]
        rate = 1 / statistics.median(gaps) if gaps and start['role']=='decode' else None
        lanes.append(dict(role=start['role'], dp_rank=start['dp_rank'], schedules=len(schedules),
            median_active_iterations_per_second=rate))
    decode_rates = [lane['median_active_iterations_per_second'] for lane in lanes if lane['role']=='decode']
    decode = statistics.median(decode_rates)
    # One engine iteration and one packed worker forward; real D adds sample_tokens.
    p_iterations = ENGINES_PER_ROLE * RUN_SECONDS / DUMMY_SECONDS
    d_iterations = ENGINES_PER_ROLE * RUN_SECONDS * decode
    records = p_iterations * 2 + d_iterations * 3
    return dict(seconds=RUN_SECONDS, lanes=lanes, prefill_dummy_iterations_upper_bound=p_iterations,
        decode_iterations_estimate=d_iterations, records_estimate=round(records),
        limitation='P dummy cadence is the 5 ms source/evidence estimate; lifecycle logs omit dummy-only iterations')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vllm-checkout', required=True, type=Path)
    parser.add_argument('--lifecycle-copy', required=True, type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source-only', action='store_true', help='Pinned source methods with CPU doubles; does not replace the installed-wheel check')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.source_only:
        from check_native_dp_wave_source import install
        install(args.vllm_checkout)
    import vllm
    from vllm.v1.engine import core
    # This check calls the native loop directly; bootstrap installation is
    # exercised separately below without permitting any executor/GPU init.
    os.environ.pop('FRONTIER_NATIVE_SERVING_EVIDENCE_DIR', None)
    from native_scheduler_lifecycle import LifecycleLog
    from native_dp_wave import install_engine_hooks
    native_loop = core.DPEngineCoreProc.run_busy_loop
    installed = Path(vllm.__file__).parent
    hashes = {}
    for name in ('v1/engine/core.py', 'v1/core/sched/async_scheduler.py'):
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        hashes[name] = digest(installed / name) == digest(args.vllm_checkout / 'vllm' / name)
    assert all(hashes.values()), hashes
    checks = dict(source_hashes=True, observer_disabled=core.DPEngineCoreProc.run_busy_loop is native_loop)
    native, _ = drive(core.DPEngineCoreProc)
    for label, inject in [('observed', False), ('injected', True)]:
        log = LifecycleLog(args.output / label / 'engine.jsonl')
        observed, _ = drive(core.DPEngineCoreProc, log, inject=inject)
        log.close()
        checks[label + '.native_results'] = native == observed
        rows = [json.loads(line) for line in (args.output / label / 'engine.jsonl').read_text().splitlines()]
        rows.extend(dict(event=event, **span) for row in list(rows) for event,span in row.get('calls',{}).items())
        kinds = {r['event'] for r in rows}
        if not inject:
            checks['every_engine_kind'] = {'engine_wave_start', 'busy_iteration', 'process_input_queue',
                'process_engine_step', 'engine_dummy_batch', 'global_unfinished', 'wave_complete',
                'start_dp_wave_enqueue', 'handle_client_request', 'add_enqueue', 'engine_add',
                'start_wave_sent', 'busy_loop_return', 'stop'} <= kinds
            checks['blocked_input'] = any(r.get('blocked') for r in rows if r['event']=='process_input_queue')
            syncs = [r for r in rows if r['event']=='global_unfinished' and r['all_reduce_ran']]
            checks['32_step_sync'] = len(syncs)==2 and all(r['step_counter_after']==32 and r['result'] is False for r in syncs)
            add = next(r for r in rows if r['event']=='engine_add')
            checks['stale_wave_add'] = add['request_wave']==0 and add['current_wave_before']==1 and add['current_wave_after']==1 and add['engines_running_before'] is False and add['engines_running_after'] is True and add['start_wave_sent'] is True
            checks['iteration_count'] = len([r for r in rows if r['event']=='busy_iteration'])==65
            checks['no_errors'] = 'error' not in kinds
            checks['span_timestamps'] = all(r['end_mono'] >= r['mono'] for r in rows
                if r['event'] in ('process_input_queue','process_engine_step','engine_dummy_batch','global_unfinished'))
            iterations = [r for r in rows if r['event']=='busy_iteration']
            checks['iteration_state_fields'] = all({'current_wave','engines_running','step_counter','calls'} <= r.keys() for r in iterations)
            handled = next(r for r in rows if r['event']=='handle_client_request' and r['request_type']=='START_DP_WAVE')
            checks['start_wave_fields'] = (handled['wave']==0 and handled['excluded_engine_index']==-1 and
                handled['engines_running_before'] is False and handled['engines_running_after'] is True)
            checks['model_execution_fields'] = any(r.get('model_executed') is True for r in rows if r['event']=='process_engine_step')
            checks['enqueue_fields'] = any(r.get('request_wave')==0 and r.get('request_id')=='stale-wave' for r in rows if r['event']=='add_enqueue')
        else:
            checks['failures_isolated'] = 'error' in kinds
    with patch.object(core.DPEngineCoreProc, 'run_busy_loop', native_loop):
        install_engine_hooks()
        checks['bootstrap_disabled_without_evidence'] = core.DPEngineCoreProc.run_busy_loop is native_loop
        with patch.dict(os.environ, FRONTIER_NATIVE_SERVING_EVIDENCE_DIR=str(args.output)):
            install_engine_hooks()
            checks['bootstrap_installed'] = core.DPEngineCoreProc.run_busy_loop is not native_loop
            for role in ('prefill', 'decode'):
                for rank in range(ENGINES_PER_ROLE):
                    with patch.object(core.DPEngineCoreProc, 'run_busy_loop', native_loop):
                        baseline, _ = drive(core.DPEngineCoreProc, rank=rank)
                    path = args.output / 'engine_bootstrap' / role / f'dp{rank}.jsonl'
                    log = LifecycleLog(path)
                    observed, _ = drive(core.DPEngineCoreProc, log, rank=rank, bootstrap=True)
                    log.close()
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    checks[f'bootstrap.{role}.dp{rank}'] = (observed == baseline and
                        rows[0]['event']=='engine_wave_start' and rows[0]['dp_rank']==rank and
                        sum(row['event']=='busy_iteration' for row in rows)==65 and
                        not any(row['event']=='error' for row in rows))
    worker_records = worker_check(args.output / 'worker')
    checks['worker_records_and_failure_isolation'] = True
    checks['all_8_worker_bootstraps_tp0_only'] = worker_bootstrap_check(args.output/'worker_bootstrap')==8
    timing = {'native': [], 'observed': []}
    for round_index in range(ROUNDS):
        for label in timing:
            log = LifecycleLog(args.output / f'timing_{round_index}_{label}.jsonl') if label=='observed' else None
            _, elapsed = drive(core.DPEngineCoreProc, log, ITERATIONS, blocked=False)
            timing[label].append(elapsed * 1e6 / ITERATIONS)
            if log:
                log.close()
    overhead = statistics.median(timing['observed']) - statistics.median(timing['native'])
    worker_times = worker_timing(args.output/'worker_timing')
    checks['engine_overhead_at_most_50us'] = overhead <= OVERHEAD_LIMIT_US
    checks['serial_overhead_target_at_most_50us'] = overhead + worker_times['added_median_us'] <= OVERHEAD_LIMIT_US
    estimated_volume = volume(args.lifecycle_copy)
    engine_path = args.output/'observed/engine.jsonl'
    worker_path = next((args.output/'worker/worker_dp_wave').glob('*.jsonl'))
    engine_lines = [json.loads(line) for line in engine_path.read_text().splitlines()]
    worker_lines = [json.loads(line) for line in worker_path.read_text().splitlines()]
    average_size = lambda rows,event: statistics.mean(len(json.dumps(row,separators=(',',':')))+1 for row in rows if row['event']==event)
    engine_size = average_size(engine_lines,'busy_iteration')
    worker_timing_path = next((args.output/'worker_timing/0_observed/worker_dp_wave').glob('*.jsonl'))
    worker_timing_rows = [json.loads(line) for line in worker_timing_path.read_text().splitlines()]
    worker_size = average_size(worker_timing_rows,'worker_dummy_batch')
    sample_size = average_size(worker_lines,'worker_sample_tokens')
    estimated_volume['engine_iteration_bytes'] = engine_size
    estimated_volume['worker_dummy_iteration_bytes'] = worker_size
    estimated_volume['observer_bytes_estimate'] = round((estimated_volume['prefill_dummy_iterations_upper_bound']+estimated_volume['decode_iterations_estimate'])*(engine_size+worker_size) +
        estimated_volume['decode_iterations_estimate']*sample_size)
    estimated_volume['bytes_exclude'] = 'existing lifecycle records, initialization and sparse messages; includes one estimated D sample_tokens per forward'
    result = dict(verdict='PASS' if all(checks.values()) else 'FAIL', checks=checks,
        python=sys.version, vllm=vllm.__version__, vllm_file=vllm.__file__,
        required_installed_wheel_check=not args.source_only,
        timing=dict(per_iteration_us=timing, added_median_us=overhead,
                    rounds=ROUNDS, iterations=ITERATIONS, worker=worker_times,
                    combined_serial_estimate_us=overhead+worker_times['added_median_us']), worker_records=worker_records,
        volume=estimated_volume)
    (args.output / 'check_result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
