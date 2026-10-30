"""CPU-only DP wave spans, installed through WaveObservedScheduler in each core.

A span carries both entry and return timestamps in one buffered record. Native
calls, queues, results and exceptions are retained; no loop is reimplemented.
"""
import os
import time
import traceback
import sys
from functools import wraps

from native_scheduler_lifecycle import LifecycleObservedScheduler, compact

SCHEMA = 'native_dp_wave.v1'
DP_SYNC_INTERVAL = 32
SCHEDULER_CLASS = 'native_dp_wave.WaveObservedScheduler'


class WaveLog:
    def __init__(self, log):
        self.log = log

    def observe(self, where, operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except Exception:
            self.log.error(where)
            return None

    def stamp(self):
        return time.time(), time.monotonic()

    def record(self, event, start, end=None, **fields):
        end_t, end_mono = end or self.stamp()
        t, mono = start
        self.log.write(compact(dict(event=event, t=t, mono=mono,
            end_t=end_t, end_mono=end_mono, **fields)), end_mono)

    def span(self, owner, name, event=None, before=None, after=None):
        native = getattr(owner, name)

        @wraps(native)
        def call(*args, **kwargs):
            start = self.observe(name + '.stamp', self.stamp)
            fields = self.observe(name + '.before', before, *args, **kwargs) if before else {}
            try:
                result = native(*args, **kwargs)
            except BaseException:
                self.observe(name + '.exception', self.record, event or name, start,
                    **(fields or {}), native_exception=True)
                raise
            end = self.observe(name + '.end', self.stamp)
            extra = self.observe(name + '.after', after, result, *args, **kwargs) if after else {}
            self.observe(name + '.record', self.record, event or name, start, end=end,
                **(fields or {}), **(extra or {}))
            return result
        setattr(owner, name, call)


class EngineWaveLog(WaveLog):
    def __init__(self, engine):
        super().__init__(engine.scheduler._lifecycle)
        self.engine = engine
        self.iteration = 0
        self.active = None
        self.blocked = False
        self.sync_ran = False
        self.start_wave_sent = False
        self.iteration_fields = {}

    def record(self, event, start, end=None, **fields):
        if event in ('process_input_queue', 'process_engine_step', 'engine_dummy_batch', 'global_unfinished'):
            _, end_mono = end or self.stamp()
            self.iteration_fields[event] = dict(mono=start[1], end_mono=end_mono, **fields)
        else:
            super().record(event, start, end=end, **fields)

    def state(self):
        engine = self.engine
        return dict(iteration=self.iteration, current_wave=engine.current_wave,
            engines_running=engine.engines_running, step_counter=engine.step_counter)

    def iteration_boundary(self):
        if self.active is not None:
            self.record('busy_iteration', self.active, calls=self.iteration_fields, **self.state())
            if not self.engine.engines_running:
                self.log.flush(time.monotonic())
        self.iteration_fields = {}
        self.iteration += 1
        self.active = self.stamp()

    def input_begin(self):
        self.blocked = False
        return {}

    def input_end(self, result):
        return dict(blocked=self.blocked, block_requested=self.engine.process_input_queue_block)

    def enqueue_fields(self, item):
        from vllm.v1.engine import EngineCoreRequestType
        kind, value = item
        if kind == EngineCoreRequestType.START_DP_WAVE:
            wave, excluded = value
            return dict(event='start_dp_wave_enqueue', wave=wave,
                excluded_engine_index=excluded)
        if kind == EngineCoreRequestType.ADD:
            request, wave = value
            return dict(event='add_enqueue', request_id=request.request_id,
                request_wave=wave)
        return None

    def queue_put(self, owner, fields):
        native = owner.put_nowait
        def put(item, *args, **kwargs):
            start = self.observe('queue.stamp', self.stamp)
            data = self.observe('queue.fields', fields, item)
            result = native(item, *args, **kwargs)
            if data is not None:
                self.observe('queue.record', self.emission, data, start)
            return result
        owner.put_nowait = put

    def emission(self, data, start):
        event = data['event']
        if event == 'start_wave_sent':
            self.start_wave_sent = True
        self.record(event, start, **{k:v for k,v in data.items() if k!='event'})
        if event == 'wave_complete':
            self.log.flush(time.monotonic())

    def handle_begin(self, kind, value):
        from vllm.v1.engine import EngineCoreRequestType
        if kind != EngineCoreRequestType.START_DP_WAVE:
            return dict(request_type=kind.name, **self.state())
        wave, excluded = value
        return dict(request_type=kind.name, wave=wave, excluded_engine_index=excluded,
            current_wave_before=self.engine.current_wave,
            engines_running_before=self.engine.engines_running)

    def handle_end(self, result, *args):
        return dict(current_wave_after=self.engine.current_wave,
            engines_running_after=self.engine.engines_running)

    def add_begin(self, request, request_wave=0):
        self.start_wave_sent = False
        return dict(request_id=request.request_id, request_wave=request_wave,
            current_wave_before=self.engine.current_wave,
            engines_running_before=self.engine.engines_running)

    def add_end(self, result, *args, **kwargs):
        return dict(current_wave_after=self.engine.current_wave,
            engines_running_after=self.engine.engines_running, start_wave_sent=self.start_wave_sent)

    def output_fields(self, item):
        if isinstance(item, tuple):
            _, output = item
            if output.wave_complete is not None:
                return dict(event='wave_complete', wave=output.wave_complete, **self.state())
            if output.start_wave is not None:
                return dict(event='start_wave_sent', wave=output.start_wave, **self.state())
        return None

    def global_begin(self, local_unfinished):
        self.sync_ran = False
        return dict(local_unfinished=local_unfinished, step_counter=self.engine.step_counter)

    def global_end(self, result, *args):
        return dict(step_counter_after=self.engine.step_counter,
            all_reduce_ran=self.sync_ran, result=result)

    def install(self):
        from vllm.config import ParallelConfig
        engine = self.engine
        self.record('engine_wave_start', self.stamp(), schema=SCHEMA,
            engine_index=engine.engine_index, dp_rank=engine.dp_rank,
            **self.state())
        self.queue_put(engine.input_queue, self.enqueue_fields)
        # Queue.get calls Condition.wait only when it actually blocks. An
        # empty() probe before get would race with the input-thread enqueue.
        native_wait = engine.input_queue.not_empty.wait
        def wait(*args, **kwargs):
            self.observe('input_wait', setattr, self, 'blocked', True)
            return native_wait(*args, **kwargs)
        engine.input_queue.not_empty.wait = wait
        self.queue_put(engine.output_queue, self.output_fields)
        self.span(engine, '_process_input_queue', 'process_input_queue', self.input_begin, self.input_end)
        self.span(engine, '_process_engine_step', 'process_engine_step', None,
            lambda result: dict(model_executed=result))
        self.span(engine, 'execute_dummy_batch', 'engine_dummy_batch')
        self.span(engine, '_handle_client_request', 'handle_client_request', self.handle_begin, self.handle_end)
        self.span(engine, 'add_request', 'engine_add', self.add_begin, self.add_end)
        self.span(engine, '_has_global_unfinished_reqs', 'global_unfinished', self.global_begin, self.global_end)
        native_sync = ParallelConfig.sync_dp_state
        def sync(*args, **kwargs):
            self.sync_ran = True
            return native_sync(*args, **kwargs)
        ParallelConfig.sync_dp_state = staticmethod(sync)
        native_shutdown = engine._handle_shutdown
        def shutdown():
            self.observe('iteration_boundary', self.iteration_boundary)
            return native_shutdown()
        engine._handle_shutdown = shutdown

    def finish(self):
        if self.active is not None:
            self.record('busy_loop_return', self.active, calls=self.iteration_fields, **self.state())
        self.log.flush(time.monotonic())


def install_engine_hooks():
    if os.environ.get('FRONTIER_NATIVE_SERVING_EVIDENCE_DIR') is None:
        return
    from vllm.v1.engine.core import DPEngineCoreProc
    native = DPEngineCoreProc.run_busy_loop
    if getattr(native, '_frontier_wave', False):
        return
    @wraps(native)
    def run(engine, *args, **kwargs):
        observer = None
        try:
            if engine.scheduler._lifecycle is not None:
                observer = EngineWaveLog(engine)
                observer.observe('engine.install', observer.install)
        except Exception:
            print('NATIVE_SCHEDULER_LIFECYCLE_UNVERIFIED', 'engine.install',
                traceback.format_exc(), file=sys.stderr, flush=True)
        try:
            return native(engine, *args, **kwargs)
        finally:
            if observer is not None:
                observer.observe('engine.finish', observer.finish)
    run._frontier_wave = True
    DPEngineCoreProc.run_busy_loop = run


class WaveObservedScheduler(LifecycleObservedScheduler):
    """Keep the lifecycle contract and install engine hooks in this process."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._lifecycle is not None:
            self._observe('engine.bootstrap', install_engine_hooks)


# Scheduler resolution occurs in each engine constructor, before run_busy_loop.
# Workers use native_full_structure's bootstrap instead of constructing a scheduler.
if os.environ.get('FRONTIER_NATIVE_SERVING_EVIDENCE_DIR') is not None:
    try:
        install_engine_hooks()
    except Exception:
        print('NATIVE_SCHEDULER_LIFECYCLE_UNVERIFIED', 'engine.bootstrap',
            traceback.format_exc(), file=sys.stderr, flush=True)
