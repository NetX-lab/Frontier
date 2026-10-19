"""Observe prepared MRv2 CPU request metadata without qualifying execution."""
from dataclasses import dataclass, field
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

import role_capacity_observation as capacity
import role_model_input_observation as model_observation
import role_state_observation as state_observation
import role_profiler_observation as profiler_observation


CPU_ARRAY_FIELDS = (
    'idx_mapping_np', 'num_scheduled_tokens', 'query_start_loc_np',
    'num_computed_tokens_np', 'prefill_len_np', 'num_computed_prefill_tokens_np',
    'is_prefilling_np', 'prefill_runs_as_decode_np',
)
BATCH_COUNT_FIELDS = ('num_reqs', 'num_reqs_after_padding', 'num_tokens',
                      'num_tokens_after_padding')
UNMEASURED = dict.fromkeys((
    'model_input_arguments', 'device_inputs', 'content', 'finite', 'gpu_completion',
    'graph_replay', 'physical_scope_census', 'precision_and_exclusion_binding', 'G2',
), 'UNMEASURED')
# These identify the observer files read here, not installed native source or execution.
SOURCE = dict(observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              capacity_observer_sha256=hashlib.sha256(Path(capacity.__file__).read_bytes()).hexdigest(),
              model_input_observer_sha256=hashlib.sha256(
                  Path(model_observation.__file__).read_bytes()).hexdigest(),
              state_observer_sha256=hashlib.sha256(Path(state_observation.__file__).read_bytes()).hexdigest(),
              profiler_observer_sha256=profiler_observation.SOURCE_SHA256,
              native_source_status='UNVERIFIED')


@dataclass
class RequestObservation:
    invocations: int = 0
    records_written: int = 0
    metadata_snapshots: int = 0
    native_failures: int = 0
    capture_metadata_records: int = 0
    collection_stopped: bool = False
    unverified: bool = False
    last_invocation: dict = field(default_factory=dict)
    last_batch_counts: dict = field(default_factory=dict)
    observer_error: dict | None = None
    native_error: dict | None = None
    record_file: str | None = None
    rank_bytes_at_last_write: int | None = None
    rank_limit: int | None = None
    device_rpc_bytes: int | None = None
    device_owned_bytes: int = 0
    device_reservation_file: str | None = None
    device_prepared: bool = False
    profiler_reserved_bytes: int = 0
    profiler_label: str | None = None
    profiler: object | None = None


def class_name(value):
    return type(value).__module__+'.'+type(value).__qualname__


def observer_failure(state, kind, failure, *, stop=False):
    """Keep one compact error in RPC state and report the complete diagnostic locally."""
    state.unverified = True
    state.collection_stopped |= stop
    state.observer_error = dict(invocation=state.invocations, kind=kind,
                                error_type=class_name(failure))
    try:
        capacity.report_error(json.dumps(dict(
            request_metadata_status='UNVERIFIED', **state.observer_error,
            observer_error=traceback.format_exc())))
    except BaseException:
        pass


def scheduler_snapshot(args, kwargs):
    scheduler = args[0] if args else kwargs['scheduler_output']
    return dict(total_num_scheduled_tokens=int(scheduler.total_num_scheduled_tokens),
                num_scheduled_tokens=[dict(req_id=req_id, count=int(count))
                    for req_id, count in scheduler.num_scheduled_tokens.items()])


def batch_snapshot(batch):
    """Copy only retained InputBatch CPU fields; never read prepared device buffers."""
    row = {name: int(getattr(batch, name)) for name in BATCH_COUNT_FIELDS}
    row['req_ids'] = list(batch.req_ids)
    for name in CPU_ARRAY_FIELDS:
        value = getattr(batch, name)
        if value is None and name == 'prefill_runs_as_decode_np':
            # Native leaves this unset for no-prefill and some PCP prefill paths.
            row[name] = None
        else:
            if not isinstance(value, np.ndarray):
                raise TypeError(name+' must be a CPU numpy array')
            row[name] = value.copy().tolist()
    return row


def rank_record_bytes(output, rank):
    prefixes = (f'rank_{rank}_pid_', f'request_rank_{rank}_pid_',
                f'worker_{rank}_', f'device_reservation_rank_{rank}_pid_',
                f'profile_rank_{rank}_pid_')
    return sum(path.stat().st_size for path in output.iterdir()
               if path.is_file() and path.name.startswith(prefixes))


def reserve_device_bytes(worker, owned_bytes, *, profiler_bytes=None,
                         device_preparation=True, label=None):
    """Reserve tensors and the original trace allowance through one worker record."""
    state = worker._request_observation
    output, record_bytes = capacity.observation_output()
    pid = os.getpid()
    path = output/f'device_reservation_rank_{worker.rank}_pid_{pid}.json'
    reserved = state.profiler_reserved_bytes if profiler_bytes is None else profiler_bytes
    label = state.profiler_label if label is None else label
    if any(type(value) is not int or value < 0 for value in (owned_bytes, reserved)):
        raise ValueError('Native observer reservations need explicit nonnegative byte counts')
    if device_preparation and state.device_prepared:
        raise ValueError('Device observation may be prepared only once')
    if state.device_reservation_file is not None and state.device_reservation_file != str(path):
        raise ValueError('Native observer reservation changed its original worker file')
    identity = dict(getattr(worker, '_capacity_identity', {}))
    value = dict(kind='device_observation_reservation', rank=worker.rank,
        local_rank=worker.local_rank, pid=pid, worker_identity=identity, source=dict(SOURCE),
        status='RESERVED_BEFORE_ALLOCATION' if owned_bytes else 'NOT_ALLOCATED_PREPARATION_FAILED',
        owned_bytes=owned_bytes, allocation_scope='prepared_tensor_storage_bytes')
    if profiler_bytes is not None or state.profiler is not None:
        if label not in profiler_observation.WINDOWS:
            raise ValueError('Native trace reservation must identify the original diagnostic window')
        value.update(profiler_bytes=reserved, profiler_label=label,
            device_prepared=state.device_prepared or device_preparation,
            allocation_scope='prepared_tensor_storage_and_pending_native_trace_bytes')
    encoded = (json.dumps(value, allow_nan=False, separators=(',', ':'))+'\n').encode('utf-8')
    pending = pending_trace_bytes(output, worker.rank, pid, reserved, label)
    if rank_record_bytes(output, worker.rank)+len(encoded)+owned_bytes+pending > record_bytes//8:
        raise ValueError('Rank aggregate device reservation exceeds record budget')
    if state.device_reservation_file is None:
        with path.open('xb') as stream:
            stream.write(encoded)
    else:
        old = json.loads(path.read_text())
        if (old['worker_identity'] != identity or old['source'] != SOURCE or
                old['owned_bytes'] != state.device_owned_bytes or
                old.get('profiler_bytes', 0) != state.profiler_reserved_bytes or
                old.get('profiler_label') != state.profiler_label):
            raise ValueError('Native observer reservation no longer belongs to the original worker')
        temporary = path.with_name(path.name+'.pending')
        with temporary.open('xb') as stream:
            stream.write(encoded)
        # Atomic replacement keeps concurrent caller reads on a complete record.
        # Both copies are included in the allowance above until replacement.
        os.replace(temporary, path)
    state.device_owned_bytes, state.profiler_reserved_bytes = owned_bytes, reserved
    state.device_reservation_file, state.profiler_label = str(path), label
    state.device_prepared |= device_preparation


def pending_trace_bytes(output, rank, pid, reserved, label):
    """Charge retained trace bytes once and reserve only the unfinished remainder."""
    if not reserved:
        return 0
    path = Path(output)/f'profile_rank_{rank}_pid_{pid}_{label}.json'
    retained = path.stat().st_size if path.exists() else 0
    return max(0, reserved-retained)


def device_reservation_bytes(output, *, require_complete=False):
    """Read the same tensor/trace reservations for caller and node finalization."""
    output = Path(output)
    seen, used = set(), 0
    for path in output.glob('device_reservation_rank_*_pid_*.json'):
        row = json.loads(path.read_text())
        rank, local, pid, owned = (row[name] for name in ('rank', 'local_rank', 'pid', 'owned_bytes'))
        trace = row.get('profiler_bytes', 0)
        if (any(type(value) is not int for value in (rank, local, pid, owned, trace)) or
                rank < 0 or not 0 <= local < 8 or pid <= 0 or owned < 0 or trace < 0 or rank in seen or
                path.name != f'device_reservation_rank_{rank}_pid_{pid}.json' or
                row['kind'] != 'device_observation_reservation' or row['source'] != SOURCE or
                row['status'] not in ('RESERVED_BEFORE_ALLOCATION', 'NOT_ALLOCATED_PREPARATION_FAILED')):
            raise ValueError('Device reservation identity, source or byte count is unverified')
        identity = json.loads((output/f'worker_{rank}_{pid}.json').read_text())
        if identity != row['worker_identity'] or any(identity[name] != row[name] for name in ('rank', 'local_rank', 'pid')):
            raise ValueError('Device reservation differs from its original worker identity')
        if row['status'] == 'NOT_ALLOCATED_PREPARATION_FAILED' and owned != 0:
            raise ValueError('Failed preparation without allocation must reserve zero bytes')
        if 'profiler_bytes' in row and (row['profiler_label'] not in profiler_observation.WINDOWS or
                type(row['device_prepared']) is not bool):
            raise ValueError('Native trace reservation has no original diagnostic window')
        if require_complete and trace:
            raise ValueError('Final native trace reservation remains pending; completion is unverified')
        seen.add(rank)
        used += owned+pending_trace_bytes(output, rank, pid, trace, row.get('profiler_label'))
    if require_complete:
        workers = [json.loads(path.read_text()) for path in output.glob('worker_*.json')]
        if len(workers) != 8 or seen != {row['rank'] for row in workers}:
            raise ValueError('Final device reservations do not cover all eight node workers')
    return used


def pending_profiler_bytes(state):
    """Use the same trace reservation calculation as node and archive accounting."""
    if not state.profiler_reserved_bytes:
        return 0
    output, _ = capacity.observation_output()
    observed = state.profiler
    rank = observed.worker.rank if observed is not None else json.loads(
        Path(state.device_reservation_file).read_text())['rank']
    return pending_trace_bytes(output, rank, os.getpid(), state.profiler_reserved_bytes, state.profiler_label)


def device_response(worker, state, result, observed=None):
    """Cap the complete JSON response; native wire serialization is not measured."""
    cap = state.device_rpc_bytes
    if cap is None and observed is not None and observed.device is not None:
        cap = state.device_rpc_bytes = observed.device.limits.rpc_bytes
    if cap is None:
        return result
    try:
        encoded = (json.dumps(result, allow_nan=False, separators=(',', ':'))+'\n').encode('utf-8')
        if len(encoded) > cap:
            raise ValueError('Complete device observation RPC response exceeds its JSON byte cap')
        return result
    except BaseException as failure:
        observer_failure(state, 'device_rpc_response', failure, stop=True)
        if observed is not None:
            observed.stopped = True
            if observed.device is not None:
                observed.device.error('device_rpc_response', failure)
    compact = dict(status='UNVERIFIED', rank=worker.rank, collection_stopped=True,
                   error_kind='device_rpc_response', cap_scope='complete_worker_json_response')
    if len((json.dumps(compact, separators=(',', ':'))+'\n').encode('utf-8')) <= cap:
        return compact
    compact = dict(status='UNVERIFIED')
    if len((json.dumps(compact, separators=(',', ':'))+'\n').encode('utf-8')) <= cap:
        return compact
    return None


def write_record(worker, payload):
    """Share the original rank allowance and exclusively create a new request JSONL."""
    if 'STEP5_CAPACITY_NODE_FILE' not in os.environ:
        raise ValueError('Request records require a worker-local node observer output')
    output, record_bytes = capacity.observation_output()
    state = worker._request_observation
    pid = os.getpid()
    path = output/f'request_rank_{worker.rank}_pid_{pid}.jsonl'
    used = rank_record_bytes(output, worker.rank)
    encoded = payload.encode('utf-8')
    rank_limit = record_bytes//8
    if used+len(encoded)+state.device_owned_bytes+pending_profiler_bytes(state) > rank_limit:
        raise ValueError('Rank aggregate evidence exceeds record budget')
    # First write cannot append to an earlier session's evidence. No retry on failure.
    with path.open('xb' if state.record_file is None else 'ab') as stream:
        stream.write(encoded)
    state.record_file = str(path)
    state.rank_bytes_at_last_write = used+len(encoded)
    state.rank_limit = rank_limit


def record_invocation(worker, state, scheduler, scheduler_error, native_error, record):
    """Serialize before return/sampling so no CPU array or request-list aliases survive."""
    row = dict(scope='cpu_request_metadata_only_not_G2', invocation=state.invocations,
               time_ns=time.time_ns(), pid=os.getpid(), rank=worker.rank,
               local_rank=worker.local_rank,
               worker_identity=dict(getattr(worker, '_capacity_identity', {})),
               source=dict(SOURCE), scheduler=scheduler,
               native_outcome='raised' if native_error is not None else 'returned',
               native_error=native_error, runner=None, prepared_batch=None,
               prepared_batch_status='UNVERIFIED', evidence_status=dict(UNMEASURED),
               cpu_frontier_semantics='optimistic_upper_bounds_not_device_progress')
    try:
        runner = getattr(worker, 'model_runner', None)
        row['runner'] = class_name(runner) if runner is not None else None
        if native_error is not None:
            row['prepared_batch_status'] = 'NOT_OBSERVED_NATIVE_ERROR'
        elif scheduler_error is not None:
            row['observer_error'] = scheduler_error
        elif scheduler['total_num_scheduled_tokens'] == 0:
            row['prepared_batch_status'] = 'NOT_OBSERVED_ZERO_WORK'
        elif scheduler['total_num_scheduled_tokens'] > 0:
            native_state = getattr(runner, 'execute_model_state', None)
            if native_state is None:
                raise ValueError('Positive scheduler work has no observable MRv2 execute_model_state')
            row['prepared_batch'] = batch_snapshot(native_state.input_batch)
            row['prepared_batch_status'] = 'OBSERVED_CPU_METADATA'
        else:
            raise ValueError('Negative scheduler work cannot establish a prepared batch')
    except BaseException as failure:
        row['observer_error'] = traceback.format_exc()
        observer_failure(state, 'prepared_batch', failure)
    observed = getattr(runner, '_model_input_observation', None)
    if observed is not None and observed.last_record is not None:
        row['source_pinned_model_observation'] = observed.last_record
    if hasattr(worker, '_runner_observer_error'):
        row['runner_observer_error'] = dict(worker._runner_observer_error)
    if runner is not None and hasattr(runner, '_model_input_observation_error'):
        row['model_input_observer_error'] = copy.deepcopy(runner._model_input_observation_error)
    state.last_invocation = {name: row[name] for name in
        ('invocation', 'native_outcome', 'prepared_batch_status', 'runner')}
    try:
        # JSON text is an immutable, detached snapshot even for an injected recorder.
        payload = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n'
        record(worker, payload)
    except BaseException as failure:
        observer_failure(state, 'record', failure, stop=True)
        return
    state.records_written += 1
    if row['prepared_batch_status'] == 'OBSERVED_CPU_METADATA':
        state.metadata_snapshots += 1
        state.last_batch_counts = {name: row['prepared_batch'][name] for name in BATCH_COUNT_FIELDS}


def prepare_device_observation(worker, preparation):
    """Allocate diagnostic destinations through the existing native worker RPC."""
    state = getattr(worker, '_request_observation', None)
    if state is None:
        state = worker._request_observation = RequestObservation()
    observed = None
    try:
        limits = model_observation.DeviceObservationLimits(**preparation['limits'])
        if state.device_rpc_bytes is None:
            state.device_rpc_bytes = limits.rpc_bytes
        runner = worker.model_runner
        observed = runner._model_input_observation
        if observed.device is not None or state.device_prepared:
            raise ValueError('Device observation may be prepared only once per native resident runner')
        if state.collection_stopped:
            raise ValueError('Worker evidence collection is already stopped')
        output, record_bytes = capacity.observation_output()
        used = rank_record_bytes(output, worker.rank)
        candidate = model_observation.DeviceObservation(runner, observed.sites, limits,
            preparation['requests'], preparation['sampling_fields'],
            record_bytes//8-used-pending_profiler_bytes(state),
            metadata_limits=observed.limits, reserve=lambda owned: reserve_device_bytes(worker, owned))
        observed.device = candidate
        result = dict(rank=worker.rank, local_rank=worker.local_rank, pid=os.getpid(),
            worker_identity=dict(getattr(worker, '_capacity_identity', {})),
            status='PREPARED_DIAGNOSTIC_DEVICE_WITNESSES_NOT_G2', source=dict(SOURCE),
            preparation=copy.deepcopy(preparation), device_observation=candidate.summary())
    except BaseException as failure:
        observer_failure(state, 'device_preparation', failure, stop=True)
        if observed is not None:
            observed.stopped = True
            if observed.device is not None:
                observed.device.error('device_preparation', failure)
        if not state.device_prepared:
            try:
                reserve_device_bytes(worker, 0)
            except BaseException as reservation_error:
                observer_failure(state, 'device_reservation', reservation_error, stop=True)
        result = dict(rank=worker.rank, status='UNVERIFIED', source=dict(SOURCE),
            error_type=class_name(failure), error=str(failure), owned_bytes=state.device_owned_bytes)
    return device_response(worker, state, result, observed)


def request_snapshot(worker, record=write_record):
    """Return CPU counters and drain prepared diagnostic snapshots after generate."""
    state = getattr(worker, '_request_observation', None)
    if state is None:
        state = RequestObservation()
    status = ('UNVERIFIED' if state.unverified or state.native_failures else
              'OBSERVED_CPU_METADATA_ONLY_NOT_G2' if state.metadata_snapshots else 'UNMEASURED')
    result = dict(scope='cpu_request_metadata_only_not_G2', status=status,
                pid=os.getpid(), rank=worker.rank, local_rank=worker.local_rank,
                worker_identity=dict(getattr(worker, '_capacity_identity', {})), source=dict(SOURCE),
                request_metadata=dict(invocations=state.invocations,
                    records_written=state.records_written, metadata_snapshots=state.metadata_snapshots,
                    capture_metadata_records=state.capture_metadata_records,
                    collection_stopped=state.collection_stopped,
                    last_invocation=dict(state.last_invocation),
                    last_batch_counts=dict(state.last_batch_counts),
                    record_file=state.record_file,
                    rank_bytes_at_last_write=state.rank_bytes_at_last_write, rank_limit=state.rank_limit),
                errors=dict(observer=dict(state.observer_error) if state.observer_error else None,
                            native=dict(state.native_error) if state.native_error else None,
                            native_failures=state.native_failures),
                evidence_status=dict(UNMEASURED))
    runner = getattr(worker, 'model_runner', None)
    observed = getattr(runner, '_model_input_observation', None)
    if observed is not None:
        result['source_pinned_model_observation'] = observed.summary()
        if observed.device is not None:
            try:
                def retain_device(row):
                    payload = json.dumps(dict(pid=os.getpid(), rank=worker.rank, local_rank=worker.local_rank,
                        worker_identity=dict(getattr(worker, '_capacity_identity', {})),
                        source=dict(SOURCE), **row), allow_nan=False, separators=(',', ':'))+'\n'
                    record(worker, payload)
                    state.records_written += 1
                result['device_observation'] = observed.device.drain(retain_device)
                result['request_metadata']['records_written'] = state.records_written
                result['request_metadata']['rank_bytes_at_last_write'] = state.rank_bytes_at_last_write
                result['request_metadata']['rank_limit'] = state.rank_limit
            except BaseException as failure:
                observed.device.error('post_generate_device_drain', failure)
                observer_failure(state, 'device_drain', failure, stop=True)
                result['device_observation'] = observed.device.summary()
                result['status'] = 'UNVERIFIED'
                result['request_metadata']['collection_stopped'] = state.collection_stopped
                result['errors']['observer'] = dict(state.observer_error)
    if state.profiler is not None:
        result['profiler_observation'] = state.profiler.summary()
    if hasattr(worker, '_runner_observer_error'):
        result['runner_observer_error'] = copy.deepcopy(worker._runner_observer_error)
    if runner is not None and hasattr(runner, '_model_input_observation_error'):
        result['model_input_observer_error'] = copy.deepcopy(runner._model_input_observation_error)
    return device_response(worker, state, result, observed)


def observed_worker(base, record=write_record, *, prepare_runner=model_observation.installed_sites):
    """Compose capacity records, one native execution and the supported MRv2 factory seam."""
    capacity_worker = capacity.observed_worker(base)
    original = capacity_worker.execute_model

    def execute_model(self, *args, **kwargs):
        state = getattr(self, '_request_observation', None)
        if state is None:
            state = self._request_observation = RequestObservation()
        state.invocations += 1
        scheduler, scheduler_error = None, None
        if not state.collection_stopped:
            try:
                scheduler = scheduler_snapshot(args, kwargs)
            except BaseException as failure:
                scheduler_error = traceback.format_exc()
                observer_failure(state, 'scheduler', failure)
        try:
            runner = getattr(self, 'model_runner', None)
            observed = getattr(runner, '_model_input_observation', None)
            if observed is not None:
                observed.last_record = None
        except BaseException as failure:
            observer_failure(state, 'runner_observation_state', failure)
        try:
            try:
                result = original(self, *args, **kwargs)
            except BaseException as failure:
                state.native_failures += 1
                state.native_error = dict(invocation=state.invocations, error_type=class_name(failure))
                state.last_invocation = dict(invocation=state.invocations, native_outcome='raised')
                if not state.collection_stopped:
                    try:
                        record_invocation(self, state, scheduler, scheduler_error, traceback.format_exc(), record)
                    except BaseException as observer_error:
                        observer_failure(state, 'record', observer_error, stop=True)
                raise
            state.last_invocation = dict(invocation=state.invocations, native_outcome='returned')
            if not state.collection_stopped:
                try:
                    record_invocation(self, state, scheduler, scheduler_error, None, record)
                except BaseException as failure:
                    observer_failure(state, 'record', failure, stop=True)
            return result
        finally:
            if state.collection_stopped:
                try:
                    runner = getattr(self, 'model_runner', None)
                    observed = getattr(runner, '_model_input_observation', None)
                    if observed is not None:
                        observed.stopped = True
                except BaseException as failure:
                    observer_failure(state, 'runner_collection_stop', failure, stop=True)

    def physical_profile(self, label, is_start=True):
        state = getattr(self, '_request_observation', None)
        observed = state.profiler if state is not None else None
        if observed is None:
            return dict(status='UNVERIFIED', error='No prepared original native profiler', rank=self.rank)
        stop_window = False
        try:
            if type(is_start) is not bool or label not in profiler_observation.WINDOWS:
                raise ValueError('Native profiler control requires an original window and explicit operation')
            if is_start:
                observed.before_start(label)
                self.profile(is_start=True, profile_prefix=label, delay_iterations=0, max_iterations=0)
                observed.after_start()
            else:
                if observed.current is None or observed.current['label'] != label:
                    raise ValueError('Native profiler stop must identify the original diagnostic window')
                stop_window = True
                self.profile(is_start=False)
        except BaseException as failure:
            observed.error('native_profiler_control', failure)
            # A failed start may leave the original wrapper active. Follow its
            # normal stop once and keep the diagnostic window unverified.
            if is_start and observed.current is not None and observed.current['label'] == label:
                stop_window = True
                try:
                    self.profile(is_start=False)
                except BaseException as stop_failure:
                    observed.error('native_profiler_cleanup', stop_failure)
        finally:
            if stop_window:
                try:
                    observed.after_stop()
                except BaseException as failure:
                    observed.error('native_profiler_finalization', failure)
        try:
            result = dict(rank=self.rank, local_rank=self.local_rank, pid=os.getpid(),
                worker_identity=dict(getattr(self, '_capacity_identity', {})),
                source=dict(SOURCE), profiler_observation=observed.summary())
        except BaseException as failure:
            observed.error('native_profiler_response', failure)
            if stop_window and observed.windows and observed.windows[-1]['label'] == label:
                observed.windows[-1]['status'] = 'UNVERIFIED'
            result = dict(status='UNVERIFIED', rank=self.rank)
        return device_response(self, state, result, getattr(self.model_runner, '_model_input_observation', None))

    methods = dict(execute_model=execute_model, physical_profile=physical_profile,
                   request_snapshot=lambda self: request_snapshot(self, record),
                   prepare_device_observation=prepare_device_observation)
    if hasattr(base, '_get_cudagraph_capture_context'):
        from contextlib import contextmanager
        native_capture_context = base._get_cudagraph_capture_context

        def capture_context(self):
            context = native_capture_context(self)
            if 'STEP5_PHYSICAL_OBSERVATION' not in os.environ:
                return context
            state = getattr(self, '_request_observation', None)
            if state is None:
                state = self._request_observation = RequestObservation()
            try:
                if state.profiler is not None:
                    raise ValueError('The native capture profiler may be prepared only once')
                state.profiler = profiler_observation.create_observation(self, state, record)
            except BaseException as failure:
                observer_failure(state, 'native_profiler_preparation', failure)
                return context

            @contextmanager
            def observe_capture():
                physical_profile(self, 'capture', True)
                try:
                    with context:
                        yield
                finally:
                    physical_profile(self, 'capture', False)
            return observe_capture()
        methods['_get_cudagraph_capture_context'] = capture_context
    if hasattr(base, '_make_model_runner'):
        native_factory = base._make_model_runner

        def make_model_runner(self):
            def capture_record(row):
                state = getattr(self, '_request_observation', None)
                if state is None:
                    state = self._request_observation = RequestObservation()
                if state.collection_stopped:
                    raise ValueError('Worker evidence collection has already stopped')
                payload = json.dumps(dict(scope='capture_cpu_metadata_only_not_G2',
                    pid=os.getpid(), rank=self.rank, local_rank=self.local_rank,
                    worker_identity=dict(getattr(self, '_capacity_identity', {})),
                    source=dict(SOURCE), source_pinned_model_observation=row,
                    evidence_status=dict(UNMEASURED)), allow_nan=False, separators=(',', ':'))+'\n'
                try:
                    record(self, payload)
                except BaseException as failure:
                    observer_failure(state, 'capture_record', failure, stop=True)
                    raise
                state.records_written += 1
                state.capture_metadata_records += 1

            try:
                runner_type, sites = prepare_runner(self, native_factory)
                observed_type = model_observation.observed_runner(
                    runner_type, sites, batch_snapshot, capture_record)
            except BaseException as failure:
                self._runner_observer_error = dict(status='UNVERIFIED',
                    error_type=class_name(failure), error=str(failure),
                    native_compatibility='UNVERIFIED', evidence_status=dict(UNMEASURED))
                # Preserve the actual native branch on unavailable source/observer compatibility.
                return native_factory(self)
            # A native constructor failure is never retried through the factory.
            return observed_type(self.vllm_config, self.device)

        methods['_make_model_runner'] = make_model_runner
    return type('RequestObservedWorker', (capacity_worker,), methods)
