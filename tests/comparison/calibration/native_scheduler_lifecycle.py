"""Record native vLLM engine-core request states for the D-side handoff measurement.

Selected through the engine's ``scheduler_cls``. Every scheduler call returns
the native ``AsyncScheduler`` result unchanged and native exceptions propagate
unchanged; observation failures are recorded and never reach the engine core.
Without FRONTIER_NATIVE_SERVING_EVIDENCE_DIR nothing is recorded.
"""

import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import RequestStatus

SCHEMA = 'native_scheduler_lifecycle.v1'
FLUSH_INTERVAL_S = 2.0
MAX_ERROR_RECORDS = 5


def compact(record):
    return {name: value for name, value in record.items() if value is not None and value != []}


class LifecycleLog:
    """Buffered JSONL file of one engine core, flushed at most every FLUSH_INTERVAL_S."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open('a')
        self.lines = []
        self.flushed = time.monotonic()
        self.records = self.errors = 0

    def write(self, record, now):
        self.lines.append(json.dumps(record, separators=(',', ':')) + '\n')
        self.records += 1
        if now - self.flushed >= FLUSH_INTERVAL_S:
            self.flush(now)

    def flush(self, now):
        if self.lines:
            self.file.write(''.join(self.lines))
            self.file.flush()
            self.lines.clear()
        self.flushed = now

    def error(self, where):
        # Even a broken formatter or file must not affect native work.
        try:
            self.errors += 1
            if self.errors <= MAX_ERROR_RECORDS:
                text = traceback.format_exc()
                self.lines.append(json.dumps(dict(event='error', where=where,
                    count=self.errors, traceback=text)) + '\n')
                print('NATIVE_SCHEDULER_LIFECYCLE_UNVERIFIED', where, text,
                    file=sys.stderr, flush=True)
        except Exception:
            pass

    def close(self):
        if self.file.closed:
            return
        now = time.monotonic()
        self.write(dict(event='stop', t=time.time(), mono=now,
            records=self.records, errors=self.errors), now)
        self.flush(now)
        self.file.close()


class LifecycleObservedScheduler(AsyncScheduler):
    """AsyncScheduler that records add, schedule and update events per request."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lifecycle = None
        directory = os.environ.get('FRONTIER_NATIVE_SERVING_EVIDENCE_DIR')
        if directory is None:
            return
        # Scheduler outputs by identity until their update; requests that emitted a token.
        self._lifecycle_steps, self._lifecycle_emitted = {}, set()
        try:
            rank, pid = self.parallel_config.data_parallel_rank, os.getpid()
            log = LifecycleLog(Path(directory) / 'scheduler_lifecycle' / f'dp{rank}_pid{pid}.jsonl')
            transfer = self.vllm_config.kv_transfer_config
            now = time.monotonic()
            log.write(compact(dict(event='start', schema=SCHEMA,
                role=os.environ.get('FRONTIER_NATIVE_ROLE'), dp_rank=rank,
                host=socket.gethostname(), pid=pid, t=time.time(), mono=now,
                base_scheduler=AsyncScheduler.__module__ + '.' + AsyncScheduler.__name__,
                async_scheduling=self.scheduler_config.async_scheduling,
                connector=None if transfer is None else transfer.kv_connector)), now)
            self._lifecycle = log
        except Exception:
            print('NATIVE_SCHEDULER_LIFECYCLE_UNVERIFIED', 'start', traceback.format_exc(),
                file=sys.stderr, flush=True)

    def _observe(self, where, operation, *args):
        try:
            return operation(*args)
        except Exception:
            self._lifecycle.error(where)
            return None

    def add_request(self, request):
        t, mono = time.time(), time.monotonic()
        super().add_request(request)
        if self._lifecycle is not None:
            self._observe('add', self._record_add, request, t, mono)

    def schedule(self, *args, **kwargs):
        t, mono = time.time(), time.monotonic()
        output = super().schedule(*args, **kwargs)
        if self._lifecycle is not None:
            self._observe('schedule', self._record_schedule, output, t, mono, time.monotonic())
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        t, mono = time.time(), time.monotonic()
        transfer = None
        if self._lifecycle is not None:
            # The native update aggregates scheduler-side connector stats into
            # the worker stats object, so the worker fields are copied first.
            transfer = self._observe('transfer', self._transfer_fields, model_runner_output)
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        if self._lifecycle is not None:
            self._observe('update', self._record_update, scheduler_output, outputs,
                transfer, t, mono, time.monotonic())
        return outputs

    def shutdown(self):
        try:
            super().shutdown()
        finally:
            if self._lifecycle is not None:
                self._observe('shutdown', self._lifecycle.close)

    def _record_add(self, request, t, mono):
        params = request.kv_transfer_params or {}
        self._lifecycle.write(compact(dict(event='add', t=t, mono=mono,
            request_id=request.request_id, arrival_time=request.arrival_time,
            prompt_tokens=request.num_prompt_tokens, max_tokens=request.max_tokens,
            do_remote_prefill=params.get('do_remote_prefill'),
            do_remote_decode=params.get('do_remote_decode'),
            remote_request_id=params.get('remote_request_id'))), mono)

    def _record_schedule(self, output, t, mono, end):
        step = self.current_step
        self._lifecycle_steps[id(output)] = step
        record = dict(event='schedule', step=step, t=t, mono=mono, dur_ms=(end - mono) * 1e3)
        if output.total_num_scheduled_tokens:
            scheduled = output.num_scheduled_tokens
            record.update(tokens=output.total_num_scheduled_tokens,
                new=[[new.req_id, new.num_computed_tokens, scheduled[new.req_id]]
                     for new in output.scheduled_new_reqs],
                cached=len(output.scheduled_cached_reqs.req_ids))
        metadata = output.kv_connector_metadata
        record.update(
            waiting_remote_kv=[request_id for request_id, request in self.requests.items()
                               if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS],
            recv=list(getattr(metadata, 'reqs_to_recv', ())),
            send=list(getattr(metadata, 'reqs_to_send', ())),
            finished=sorted(output.finished_req_ids),
            preempted=sorted(output.preempted_req_ids or ()))
        self._lifecycle.write(compact(record), end)

    def _transfer_fields(self, model_runner_output):
        transfer = model_runner_output.kv_connector_output
        if transfer is None:
            return None
        stats = transfer.kv_connector_stats
        return dict(finished_recving=sorted(transfer.finished_recving or ()),
            finished_sending=sorted(transfer.finished_sending or ()),
            kv_stats=None if stats is None else
                {name: list(value) if isinstance(value, list) else value
                 for name, value in stats.data.items()} or None,
            invalid_blocks=len(transfer.invalid_block_ids) or None)

    def _record_update(self, scheduler_output, outputs, transfer, t, mono, end):
        first, done = [], []
        for engine_outputs in outputs.values():
            for output in engine_outputs.outputs:
                if output.new_token_ids and output.request_id not in self._lifecycle_emitted:
                    self._lifecycle_emitted.add(output.request_id)
                    first.append(output.request_id)
                if output.finish_reason is not None:
                    done.append(output.request_id)
        self._lifecycle_emitted.difference_update(done, scheduler_output.finished_req_ids)
        record = dict(event='update', step=self._lifecycle_steps.pop(id(scheduler_output), None),
            t=t, mono=mono, dur_ms=(end - mono) * 1e3,
            tokens=scheduler_output.total_num_scheduled_tokens or None,
            first_token=first, done=done)
        record.update(transfer or {})
        self._lifecycle.write(compact(record), end)
