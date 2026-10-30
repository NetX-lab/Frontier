"""Install TP0 CPU wave records through the FullModelInventory import path.

Only the WaveObservedScheduler configuration activates worker records. The
extension name and its inventory API stay unchanged, so existing launchers still
supply the evidence directory and runtime identity without replay-driver edits.
"""
import os
import socket
import sys
import threading
import time
from functools import wraps
from pathlib import Path

from native_dp_wave import DP_SYNC_INTERVAL, SCHEMA, SCHEDULER_CLASS, WaveLog
from native_scheduler_lifecycle import LifecycleLog


class WorkerWaveLog(WaveLog):
    def __init__(self, worker, directory):
        rank = worker.vllm_config.parallel_config.data_parallel_rank
        log = LifecycleLog(Path(directory) / 'worker_dp_wave' / f'dp{rank}_tp0_pid{os.getpid()}.jsonl')
        super().__init__(log)
        self.worker = worker
        self.local = threading.local()
        self.forwards = 0
        self.stage = 'initialization'
        self.record('worker_wave_start', self.stamp(), schema=SCHEMA,
            host=socket.gethostname(), pid=os.getpid(), dp_rank=rank, tp_rank=0,
            role=os.environ.get('FRONTIER_NATIVE_ROLE'))

    def record(self, event, start, end=None, **fields):
        calls = getattr(self.local, 'calls', None)
        if calls is not None and event in ('dp_padding_sync', 'dp_cpu_all_reduce'):
            _, end_mono = end or self.stamp()
            calls[event] = dict(mono=start[1], end_mono=end_mono, **fields)
            return
        if event in ('worker_execute_model', 'worker_dummy_batch'):
            fields['calls'] = calls or {}
            self.local.calls = None
        super().record(event, start, end=end, **fields)
        if event in ('worker_execute_model', 'worker_dummy_batch', 'worker_sample_tokens') and self.forwards % DP_SYNC_INTERVAL == 0:
            self.log.flush(time.monotonic())

    def begin_call(self):
        self.local.calls = {}
        return {}

    def execute_fields(self, scheduler_output):
        self.begin_call()
        ids = list(scheduler_output.num_scheduled_tokens)
        return dict(tokens=scheduler_output.total_num_scheduled_tokens, request_ids=ids)

    def sync_begin(self, cudagraph_manager, desired_batch_desc, num_tokens, num_reqs, uniform_token_count, dp_size, dp_rank,
                   max_query_len=None, num_active_loras=0):
        self.forwards += 1
        self.local.sync = self.forwards
        return dict(forward=self.forwards, stage=self.stage, num_tokens=num_tokens, num_reqs=num_reqs,
            dp_rank=dp_rank, dp_size=dp_size, desired_graph_mode=desired_batch_desc.cg_mode.name,
            uniform_token_count=uniform_token_count, max_query_len=max_query_len,
            num_active_loras=num_active_loras)

    def sync_end(self, result, *args, **kwargs):
        descriptor, tokens = result
        self.local.sync = None
        return dict(reduced_num_tokens=None if tokens is None else tokens.tolist(),
            graph_mode=descriptor.cg_mode.name, padded_num_tokens=descriptor.num_tokens)

    def all_reduce_fields(self, tensor, *args, **kwargs):
        # Called solely in the native CPU rendezvous; never read a device tensor.
        return dict(forward=self.local.sync, values=tensor.tolist())

    def install(self, dp_utils):
        worker = self.worker
        self.span(worker, 'execute_model', 'worker_execute_model', self.execute_fields)
        self.span(worker, 'execute_dummy_batch', 'worker_dummy_batch', self.begin_call)
        self.span(worker, 'sample_tokens', 'worker_sample_tokens')
        native_sync = dp_utils.sync_cudagraph_and_dp_padding
        def sync(*args, **kwargs):
            start = self.observe('sync.stamp', self.stamp)
            fields = self.observe('sync.begin', self.sync_begin, *args, **kwargs)
            try:
                result = native_sync(*args, **kwargs)
            except BaseException:
                self.local.sync = None
                self.observe('sync.exception', self.record, 'dp_padding_sync', start,
                    **(fields or {}), native_exception=True)
                raise
            end = self.observe('sync.return', self.stamp)
            extra = self.observe('sync.end', self.sync_end, result)
            self.observe('sync.record', self.record, 'dp_padding_sync', start, end=end,
                **(fields or {}), **(extra or {}))
            return result
        dp_utils.sync_cudagraph_and_dp_padding = sync
        native_reduce = dp_utils.dist.all_reduce
        @wraps(native_reduce)
        def reduce(tensor, *args, **kwargs):
            if getattr(self.local, 'sync', None) is None:
                return native_reduce(tensor, *args, **kwargs)
            start = self.observe('all_reduce.stamp', self.stamp)
            fields = self.observe('all_reduce.before', self.all_reduce_fields, tensor)
            try:
                result = native_reduce(tensor, *args, **kwargs)
            except BaseException:
                self.observe('all_reduce.exception', self.record, 'dp_cpu_all_reduce', start,
                    **(fields or {}), native_exception=True)
                raise
            end = self.observe('all_reduce.return', self.stamp)
            reduced = self.observe('all_reduce.after', tensor.tolist)
            self.observe('all_reduce.record', self.record, 'dp_cpu_all_reduce', start, end=end,
                **(fields or {}), reduced_values=reduced)
            return result
        dp_utils.dist.all_reduce = reduce
        native_warmup = worker.compile_or_warm_up_model
        def warmup(*args, **kwargs):
            result = native_warmup(*args, **kwargs)
            self.forwards = 0
            self.stage = 'serving'
            self.observe('worker.warmup', lambda: self.record('worker_warmup_complete', self.stamp()))
            self.observe('worker.flush', self.log.flush, time.monotonic())
            return result
        worker.compile_or_warm_up_model = warmup
        native_shutdown = worker.shutdown
        def shutdown(*args, **kwargs):
            try:
                return native_shutdown(*args, **kwargs)
            finally:
                self.observe('worker.close', self.log.close)
        worker.shutdown = shutdown


def install_worker_hooks(worker_class=None):
    directory = os.environ.get('FRONTIER_NATIVE_SERVING_EVIDENCE_DIR')
    if directory is None:
        return
    if worker_class is None:
        from vllm.v1.worker.gpu_worker import Worker
        worker_class = Worker
    native = worker_class.init_device
    if getattr(native, '_frontier_wave', False):
        return
    @wraps(native)
    def init(worker, *args, **kwargs):
        result = native(worker, *args, **kwargs)
        observer = None
        try:
            if worker.vllm_config.scheduler_config.scheduler_cls != SCHEDULER_CLASS:
                return result
            from vllm.distributed import get_tp_group
            if get_tp_group().rank_in_group != 0:
                return result
            from vllm.v1.worker.gpu import dp_utils
            observer = WorkerWaveLog(worker, directory)
            observer.observe('worker.install', observer.install, dp_utils)
            worker._dp_wave_log = observer
        except Exception:
            if observer is not None:
                observer.log.error('worker.start')
            else:
                import traceback
                print('NATIVE_SCHEDULER_LIFECYCLE_UNVERIFIED', 'worker.start',
                    traceback.format_exc(), file=sys.stderr, flush=True)
        return result
    init._frontier_wave = True
    worker_class.init_device = init
