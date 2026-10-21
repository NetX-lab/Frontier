"""Observe the original native Torch profiler without qualifying physical nodes or G2."""
from dataclasses import dataclass, field
import copy
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import time

import role_model_input_observation as model_observation


SOURCE_PINS = {
    'wrapper': ('vllm/profiler/wrapper.py', 'ffc5424d87fdcd8ff8bc5da9eaabfb2c8eb9ac70f106b74442b7df6e6e921013'),
    'worker': ('vllm/v1/worker/gpu_worker.py', model_observation.SOURCE_PINS['worker']),
    'torch_graphs': ('torch/cuda/graphs.py', '15b535903f0832cbfbf83070c663ed899a9fd87b5499befde5087b4c2962ed81'),
    'cuda_runtime': ('cuda/bindings/runtime.cpython-312-x86_64-linux-gnu.so', 'bdd061294dd0d595df280aa7931e125d1356e80016751eaec009f5f2f2c57e59'),
    'cupti': ('nvidia/cu13/lib/libcupti.so.13', 'e2f9ed861fe27c492b8bb52b5e3220ef5120f3edcda36312e96b7fd8a186be3e'),
}
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
WINDOWS = ('capture', 'requests')
ANNOTATION_PREFIX = 'step5_native:'


@dataclass(frozen=True)
class ProfilerObservationLimits:
    trace_bytes: int
    trace_events: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError('Profiler trace limits must be explicit positive integers')


def native_config(output):
    """Use the original Torch wrapper with explicit unscheduled CPU/CUDA activities."""
    return dict(profiler='torch', torch_profiler_dir=str(output),
        torch_profiler_activities=['CPU', 'CUDA'], torch_profiler_with_stack=False,
        torch_profiler_with_flops=False, torch_profiler_use_gzip=False,
        torch_profiler_dump_cuda_time_total=False, torch_profiler_record_shapes=False,
        torch_profiler_with_memory=False, capture_torch_profiler=False,
        detailed_trace_annotation=False, ignore_frontend=True,
        delay_iterations=0, max_iterations=0, wait_iterations=0, warmup_iterations=0)


def annotation_name(lifecycle, sequence, event, kind):
    if lifecycle not in ('capture', 'execute') or kind not in ('scope', 'call', 'capture', 'replay'):
        raise ValueError('An annotation must refer to an existing native observation boundary')
    if any(type(value) is not int or value < 0 for value in (sequence, event)):
        raise ValueError('Annotation sequence and event must identify the original scope')
    return f'{ANNOTATION_PREFIX}{lifecycle}:{sequence}:{event}:{kind}'


def drain_trace(fifo, sink, byte_limit, deadline, *, clock=time.monotonic):
    """Consume a native FIFO; retention failures must still let its writer finish."""
    result = dict(bytes_seen=0, retained_bytes=0, error=None, deadline_exceeded=False)
    try:
        while True:
            # Keep the read endpoint until the exporter opens its writer, even
            # when a read failure is injected at the first available byte.
            select.select([fifo], [], [])
            chunk = os.read(fifo, 65536)
            if not chunk:
                break
            result['bytes_seen'] += len(chunk)
            if result['error'] is None:
                try:
                    result['deadline_exceeded'] |= clock() >= deadline
                    remaining = byte_limit-result['retained_bytes']
                    if not result['deadline_exceeded'] and remaining > 0:
                        kept = chunk[:remaining]
                        count = sink.write(kept)
                        if count != len(kept):
                            raise OSError('Trace output did not retain the complete write')
                        result['retained_bytes'] += count
                except BaseException as failure:
                    result['error'] = repr(failure)
    except BaseException as failure:
        result['error'] = repr(failure)
    finally:
        for close in (sink.close, lambda: os.close(fifo)):
            try:
                close()
            except BaseException as failure:
                result['error'] = result['error'] or repr(failure)
    return result


def export_trace(profiler, path, byte_limit, deadline, *, clock=time.monotonic):
    """Run the native exporter on its calling thread with a fresh CPU FIFO reader.

    Kineto can hold the GIL throughout export. A Python thread cannot drain a
    trace larger than the pipe, so the reader execs a separate interpreter with
    only its FIFO and output descriptors. No CUDA state or profiler is passed.
    Native export remains supervised by the original absolute workload deadline.
    """
    if type(byte_limit) is not int or byte_limit <= 0 or not math.isfinite(deadline):
        raise ValueError('Trace export needs a positive byte limit and finite absolute deadline')
    if clock() >= deadline:
        raise TimeoutError('Trace export must begin before the original workload deadline')
    path = Path(path)
    native_path = path.with_name(path.name+'.native')
    pipe = native_path.with_name(native_path.name+'.tmp')
    if native_path.exists() or native_path.is_symlink():
        raise FileExistsError('Native trace export must not overwrite an existing artifact')
    started = clock()
    result = dict(status='UNVERIFIED', path=str(path), bytes_seen=0, retained_bytes=0,
                  byte_limit=byte_limit, error=None, deadline_exceeded=False,
                  started=started, deadline=deadline, reader_complete=False, reader_reaped=False,
                  cleanup_errors=[])
    stream = path.open('xb')
    descriptor, keeper, reader, pipe_identity = None, None, None, None

    def cleanup_pipe():
        for candidate in (pipe, native_path):
            try:
                identity = candidate.lstat()
            except FileNotFoundError:
                continue
            if (identity.st_dev, identity.st_ino) == pipe_identity:
                candidate.unlink()

    def remaining_time():
        try:
            return max(0.0, deadline-clock())
        except BaseException as failure:
            result['error'] = result['error'] or repr(failure)
            return 0.0

    try:
        os.mkfifo(pipe, 0o600)
        identity = pipe.lstat()
        pipe_identity = (identity.st_dev, identity.st_ino)
        descriptor = os.open(pipe, os.O_RDONLY | os.O_NONBLOCK)
        # This writer prevents EOF until the original exporter returns. The
        # caller releases its read endpoint before export so reader failure
        # cannot leave a blocked writer with an unused read endpoint.
        keeper = os.open(pipe, os.O_WRONLY | os.O_NONBLOCK)
        os.set_blocking(descriptor, True)
        reader = subprocess.Popen([sys.executable, '-B', '-S', str(Path(__file__).resolve()),
            str(descriptor), str(stream.fileno()), str(byte_limit), str(deadline)],
            pass_fds=(descriptor, stream.fileno()), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result['reader_pid'] = reader.pid
        ready, _, _ = select.select([reader.stdout], [], [], remaining_time())
        if not ready or os.read(reader.stdout.fileno(), 6) != b'READY\n':
            raise RuntimeError('Fresh trace reader did not acknowledge its owned descriptors')
        os.close(descriptor)
        descriptor = None
        stream.close()
        if clock() >= deadline:
            raise TimeoutError('Fresh trace reader startup exhausted the original workload deadline')
        try:
            # The locked Kineto exporter writes <path>.tmp before renaming it.
            profiler.export_chrome_trace(str(native_path))
        except BaseException as failure:
            result['error'] = result['error'] or repr(failure)
        finally:
            os.close(keeper)
            keeper = None
        try:
            stdout, stderr = reader.communicate(timeout=remaining_time())
            result['reader_exit_code'] = reader.returncode
            if reader.returncode != 0 or stderr:
                raise RuntimeError('Fresh trace reader failed: '+stderr.decode(errors='replace'))
            drained = json.loads(stdout)
            result['bytes_seen'] = drained['bytes_seen']
            result['retained_bytes'] = drained['retained_bytes']
            result['deadline_exceeded'] |= drained['deadline_exceeded']
            result['error'] = result['error'] or drained['error']
            result['reader_complete'] = True
        except BaseException as failure:
            result['error'] = result['error'] or repr(failure)
    finally:
        try:
            stream.close()
        except BaseException as failure:
            result['error'] = result['error'] or repr(failure)
        for endpoint in (descriptor, keeper):
            if endpoint is not None:
                try:
                    os.close(endpoint)
                except BaseException as failure:
                    result['error'] = result['error'] or repr(failure)
        if reader is not None:
            killed = False
            try:
                if reader.poll() is None:
                    reader.kill()
                    killed = True
            except BaseException as failure:
                result['cleanup_errors'].append(repr(failure))
            try:
                reader.communicate(timeout=None if killed else remaining_time())
            except BaseException as failure:
                result['cleanup_errors'].append(repr(failure))
            try:
                result['reader_exit_code'] = reader.poll()
                result['reader_reaped'] = result['reader_exit_code'] is not None
            except BaseException as failure:
                result['cleanup_errors'].append(repr(failure))
            for stream in (reader.stdout, reader.stderr):
                try:
                    stream.close()
                except BaseException as failure:
                    result['cleanup_errors'].append(repr(failure))
        if pipe_identity is not None:
            try:
                cleanup_pipe()
            except BaseException as failure:
                result['cleanup_errors'].append(repr(failure))
        if result['cleanup_errors']:
            result['error'] = result['error'] or result['cleanup_errors'][0]
    try:
        result['finished'] = clock()
        result['deadline_exceeded'] |= result['finished'] >= deadline
        result['seconds'] = result['finished']-started
        if (result['reader_complete'] and result['reader_reaped'] and result['error'] is None and
                not result['deadline_exceeded'] and 0 < result['bytes_seen'] <= byte_limit and
                result['bytes_seen'] == result['retained_bytes'] == path.stat().st_size):
            result['status'] = 'EXPORTED_WITHIN_BYTE_AND_DEADLINE_LIMITS'
    except BaseException as failure:
        result['error'] = result['error'] or repr(failure)
    return result


def correlate_capture_nodes(capture, stored, kernels):
    """Join locked-runtime node IDs while preserving host-interval ownership limits.

    The installed Torch graph annotation remapper replaces the high 32-bit graph
    ID and preserves the low 32-bit node index. The H800 API smoke verifies this
    encoding for the locked CUPTI getters and Kineto's actual trace fields.
    Child graphs, absent intervals and unmatched nodes receive no attribution.
    """
    result = dict(status='UNVERIFIED', kernels=[], unmatched_kernel_ids=[],
                  ownership='HOST_CALL_INTERVAL_ONLY_NOT_EXCLUSIVE_GPU_OWNERSHIP')
    binding = stored.get('capture_binding_event')
    if stored['path'] != 'FULL' or type(binding) is not int or binding <= 0:
        result['reason'] = 'NO_FULL_CAPTURE_INTERVAL_BINDING'
        return result
    intervals = [row for row in capture['events'] if row['kind'] == 'capture_node_interval' and
                 row['capture_binding_event'] == binding]
    forwards = [row for row in capture['events'] if row['event'] == binding and
                row['kind'] == 'capture_forward' and row.get('native_capture_region') is True and
                row['owner_id'] == stored['owner_id']]
    if (len(forwards) != 1 or len(intervals) < 2 or intervals[0]['boundary'] != 'capture_begin' or
            intervals[-1]['boundary'] != 'capture_end'):
        result['reason'] = 'INCOMPLETE_NATIVE_CAPTURE_INTERVALS'
        return result
    calls = {row['event']: row for row in capture['events'] if row['kind'] == 'native_call'}
    nodes, previous_event = {}, binding
    identity = {key: intervals[0][key] for key in
                ('capture_id', 'graph_handle', 'stream_handle', 'cupti_graph_id')}
    exec_id = stored['cupti_graph_exec_id']
    if (type(exec_id) is not int or not 0 < exec_id < 2**32 or
            type(identity['cupti_graph_id']) is not int or not 0 < identity['cupti_graph_id'] < 2**32):
        raise ValueError('CUPTI graph IDs must fit the locked 32-bit identity fields')
    for index, row in enumerate(intervals):
        if (not previous_event < row['event'] < stored['event'] or
                any(row[key] != value for key, value in identity.items()) or
                row['baseline'] is not (index == 0)):
            raise ValueError('Capture node intervals lost their original sequence or graph identity')
        previous_event = row['event']
        owner = calls.get(row['call_event'])
        if row['call_event'] is not None and (owner is None or
                owner['capture_binding_event'] != binding or owner['event'] >= row['event']):
            raise ValueError('Capture node interval does not refer to its original native call')
        for node in row['nodes']:
            node_id = node['cupti_node_id']
            if (type(node_id) is not int or node_id <= 0 or type(node['node_type']) is not int or
                    node_id >> 32 != identity['cupti_graph_id'] or node_id in nodes):
                raise ValueError('Capture nodes have ambiguous or incompatible CUPTI identities')
            nodes[node_id] = dict(interval_event=row['event'], call_event=row['call_event'],
                node_type=node['node_type'], site=owner.get('site') if owner else None,
                layer_id=owner.get('layer_id') if owner else None,
                owner_id=owner.get('owner_id') if owner else None)
        if type(row['total_nodes']) is not int or row['total_nodes'] != len(nodes):
            raise ValueError('Capture node intervals do not cover the observed graph additions')
    remapped = {(exec_id << 32) | (node_id & 0xFFFFFFFF): (node_id, row)
                for node_id, row in nodes.items()}
    seen = set()
    for kernel in kernels:
        args = kernel['args']
        node_id = args.get('graph node id')
        match = remapped.get(node_id) if type(node_id) is int else None
        if (type(args.get('graph id')) is not int or args['graph id'] != exec_id or
                match is None or match[1]['node_type'] != 0 or node_id in seen):
            result['unmatched_kernel_ids'].append(node_id)
            continue
        seen.add(node_id)
        result['kernels'].append(dict(cupti_replay_node_id=node_id,
            cupti_capture_node_id=match[0], **match[1]))
    if result['kernels'] and not result['unmatched_kernel_ids']:
        result['status'] = 'OBSERVED_CAPTURE_INTERVAL_KERNEL_CORRELATION'
    return result


def correlate_trace(events, records):
    """Join actual replay annotations and launch correlations; never infer node owners."""
    from frontier.profiling.kernel_gap.main import GRAPH_LAUNCH_API_NAMES, graph_launch_chains

    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise ValueError('Native trace must contain activity objects')
    for event in events:
        if event.get('cat') in ('user_annotation', 'cuda_runtime', 'cuda_driver', 'kernel'):
            if type(event.get('name')) is not str:
                raise ValueError('Native trace activity names must identify the recorded activity')
            if 'args' in event and not isinstance(event['args'], dict):
                raise ValueError('Native trace activity arguments must be objects')
    metadata, captures = {}, []
    for record in records:
        lifecycle, sequence = record['lifecycle'], record['sequence']
        scope = annotation_name(lifecycle, sequence, 0, 'scope')
        metadata.setdefault(scope, []).append(dict(kind='scope', lifecycle=lifecycle, sequence=sequence))
        for row in record['events']:
            if row['kind'] == 'capture_entry_stored':
                captures.append(dict(row, lifecycle=lifecycle, sequence=sequence))
            name = row.get('trace_annotation')
            if name is not None:
                expected_kind = {'native_call': 'call', 'capture_forward': 'capture',
                    'wrapper_capture_forward': 'capture', 'replay_attempt': 'replay'}.get(row['kind'])
                if expected_kind is None or name != annotation_name(lifecycle, sequence, row['event'], expected_kind):
                    raise ValueError('Trace annotation differs from its original scope/event identity')
                outcomes = [event for event in record['events'] if event['kind'] == 'replay_outcome' and
                            event['replay_event'] == row['event']]
                metadata.setdefault(name, []).append(dict(row, lifecycle=lifecycle, sequence=sequence,
                    replay_outcomes=outcomes if expected_kind == 'replay' else []))
    annotations = [event for event in events if event.get('cat') == 'user_annotation' and
                   event.get('name', '').startswith(ANNOTATION_PREFIX)]
    launches = [event for event in events if event.get('cat') in ('cuda_runtime', 'cuda_driver') and
                event.get('name', '').startswith(GRAPH_LAUNCH_API_NAMES)]
    kernels = [event for event in events if event.get('cat') == 'kernel']
    result = dict(status='UNVERIFIED', annotations=len(annotations), graph_launches=len(launches),
        kernels=len(kernels), replay_launches=[], unmatched_launches=[], ambiguous_launches=[],
        unmatched_annotations=[event['name'] for event in annotations if event['name'] not in metadata],
        unmatched_kernels=len(kernels), physical_kernel_attribution='UNVERIFIED_NO_CAPTURE_NODE_BINDING',
        transient_scale_content='UNMEASURED', clean_performance='UNVERIFIED', G2='UNVERIFIED')
    for event in (*annotations, *launches, *kernels):
        if (event.get('ph') != 'X' or any(type(event.get(key)) not in (int, float) or
                not math.isfinite(event[key]) or event[key] < 0 for key in ('ts', 'dur'))):
            result['error'] = 'Native activity timestamps or complete durations are missing or invalid'
            return result
    if any(type(event.get('pid')) is not int or type(event.get('tid')) is not int
           for event in (*annotations, *launches)):
        result['error'] = 'Native host process/thread identities are missing'
        return result
    if not launches:
        return result
    if any(type(event.get('args', {}).get('correlation')) is not int or
           event['args']['correlation'] < 0 for event in (*launches, *kernels)):
        result['error'] = 'Native launch/kernel correlation IDs are missing or invalid'
        return result
    # Reuse the existing producer. A repeated correlation anywhere in this
    # worker trace is ambiguous, including across host processes or API layers.
    chains = graph_launch_chains(events)
    if len(chains) != len(launches):
        raise ValueError('The existing graph chain producer did not preserve every launch')
    launch_counts, annotation_counts = {}, {}
    for event in events:
        if event.get('cat') in ('cuda_runtime', 'cuda_driver'):
            correlation = event.get('args', {}).get('correlation')
            if type(correlation) is int:
                launch_counts[correlation] = launch_counts.get(correlation, 0)+1
    for event in annotations:
        annotation_counts[event['name']] = annotation_counts.get(event['name'], 0)+1
    indices = {id(event): index for index, event in enumerate(events)}
    matched = set()
    for launch, chain in zip(launches, chains):
        correlation = launch['args']['correlation']
        candidates = [event for event in annotations if event['name'].endswith(':replay') and
            (event['pid'], event['tid']) == (launch['pid'], launch['tid']) and
            event['ts'] <= launch['ts'] and launch['ts']+launch['dur'] <= event['ts']+event['dur']]
        row = dict(correlation=correlation, kernels=len(chain), pid=launch['pid'], tid=launch['tid'],
                   launch_trace_event=indices[id(launch)])
        if launch_counts[correlation] != 1 or len(candidates) > 1:
            result['ambiguous_launches'].append(row)
            continue
        owners = metadata.get(candidates[0]['name'], []) if candidates else []
        if len(owners) > 1 or candidates and annotation_counts[candidates[0]['name']] != 1:
            result['ambiguous_launches'].append(row)
            continue
        owner = owners[0] if owners else None
        if (owner is None or owner['kind'] != 'replay_attempt' or not chain or
                any(type(owner.get(key)) is not int or owner[key] <= 0
                    for key in ('owner_id', 'entry_id', 'graph_id')) or
                len(owner['replay_outcomes']) != 1 or
                owner['replay_outcomes'][0]['outcome'] != 'returned_submission_not_completion'):
            result['unmatched_launches'].append(row)
            continue
        stored = [entry for entry in captures if all(entry.get(key) == owner.get(key)
                    for key in ('path', 'owner_id', 'entry_id', 'graph_id', 'cuda_graph_exec_handle',
                                'cupti_graph_exec_id')) and
                  all(type(item.get('continuity', {}).get('serial')) is int and
                      item['continuity']['serial'] > 0 and
                      type(item.get('cuda_graph_exec_handle')) is int and
                      item['cuda_graph_exec_handle'] > 0 and
                      type(item.get('cupti_graph_exec_id')) is int and item['cupti_graph_exec_id'] > 0 and
                      item.get('cuda_graph_exec_identity') == 'OBSERVED_READ_ONLY_HANDLE_NOT_NODE_BINDING'
                      for item in (entry, owner)) and
                  entry['continuity']['serial'] == owner['continuity']['serial']]
        replay = dict(row, annotation=candidates[0]['name'],
            **{key: copy.deepcopy(owner.get(key)) for key in
               ('lifecycle', 'sequence', 'event', 'path', 'owner_id', 'entry_id', 'graph_id',
                'cuda_graph_exec_handle', 'cupti_graph_exec_id', 'cuda_graph_exec_identity', 'continuity')},
            capture_binding=({key: stored[0][key] for key in ('lifecycle', 'sequence', 'event')}
                             if len(stored) == 1 else 'UNVERIFIED_CAPTURE_BINDING'),
            kernel_trace_events=[indices[id(kernel)] for kernel in chain])
        if len(stored) == 1:
            capture_records = [record for record in records if
                (record['lifecycle'], record['sequence']) == (stored[0]['lifecycle'], stored[0]['sequence'])]
            if len(capture_records) == 1:
                replay['capture_node_correlation'] = correlate_capture_nodes(capture_records[0], stored[0], chain)
        result['replay_launches'].append(replay)
        matched.update(id(kernel) for kernel in chain)
    result['unmatched_kernels'] = len(kernels)-len(matched)
    if (result['replay_launches'] and not result['unmatched_launches'] and
            not result['ambiguous_launches'] and not result['unmatched_annotations']):
        result['status'] = 'OBSERVED_REPLAY_LAUNCH_CORRELATION_ONLY_NOT_NODE_ATTRIBUTION'
        if all(row.get('capture_node_correlation', {}).get('status') ==
               'OBSERVED_CAPTURE_INTERVAL_KERNEL_CORRELATION' for row in result['replay_launches']):
            result['status'] = 'OBSERVED_REPLAY_AND_CAPTURE_INTERVAL_CORRELATION'
            result['physical_kernel_attribution'] = 'UNVERIFIED_EXCLUSIVE_NODE_OWNERSHIP'
    return result


@dataclass
class NativeProfilerObservation:
    worker: object
    limits: ProfilerObservationLimits
    output: Path
    deadline: float
    record: object
    state: object
    sources: dict
    profiler: object = None
    current: dict | None = None
    windows: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    clock: object = time.monotonic
    runtime: object = None
    cupti: object = None

    def bind_graph_api(self):
        """Bind installed read-only APIs; do not initialize CUDA or enable CUPTI collection."""
        from cuda.bindings import runtime

        package = importlib.metadata.distribution('nvidia-cuda-cupti')
        for key, path in (('cuda_runtime', Path(runtime.__file__)),
                          ('cupti', Path(package.locate_file(SOURCE_PINS['cupti'][0])))):
            relative, digest = SOURCE_PINS[key]
            if (not path.as_posix().endswith('/'+relative) or
                    hashlib.sha256(path.read_bytes()).hexdigest() != digest):
                raise ValueError('Installed graph API differs from the locked source: '+key)
            self.sources[str(path)] = dict(sha256=digest, code_owners=[])
        cupti = ctypes.CDLL(str(package.locate_file(SOURCE_PINS['cupti'][0])))
        for name, output in (('cuptiGetGraphId', ctypes.c_uint32),
                             ('cuptiGetGraphNodeId', ctypes.c_uint64),
                             ('cuptiGetGraphExecId', ctypes.c_uint32)):
            function = getattr(cupti, name)
            function.argtypes = (ctypes.c_void_p, ctypes.POINTER(output))
            function.restype = ctypes.c_int
        self.runtime, self.cupti = runtime, cupti

    def graph_read(self, function, *args, **kwargs):
        """Every native metadata read shares the original absolute workload deadline."""
        if self.worker.profiler is not self.profiler or self.clock() >= self.deadline:
            raise RuntimeError('Graph observation lost its original profiler or workload deadline')
        result = function(*args, **kwargs)
        if self.clock() >= self.deadline:
            raise TimeoutError('Graph metadata read exceeded the original workload deadline')
        if int(result[0]) != 0:
            raise RuntimeError('Native graph metadata API failed: '+str(result[0]))
        return result[1:]

    def cupti_id(self, name, handle, output_type):
        output = output_type()
        def read():
            return getattr(self.cupti, name)(ctypes.c_void_p(handle), ctypes.byref(output)),
        self.graph_read(read)
        if output.value <= 0:
            raise ValueError('CUPTI did not return a positive graph or node identity')
        return output.value

    def capture_nodes(self, previous):
        """Read node additions while the original FULL capture is still active.

        Intervals associate nodes with host call boundaries. They do not prove
        exclusive ownership when other capture threads or streams add work.
        The caller discards the numeric handle set before native capture ends.
        """
        import torch

        if self.current is None or self.current['label'] != 'capture' or not self.profiler.is_running:
            raise ValueError('Capture nodes require the original active capture profiler window')
        runtime = self.runtime
        stream = torch.cuda.current_stream().cuda_stream
        status, capture, graph, _, _, _ = self.graph_read(runtime.cudaStreamGetCaptureInfo,
            runtime.cudaStream_t(init_value=stream))
        if status != runtime.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
            raise ValueError('Original stream is no longer in an active capture')
        identity = dict(capture_id=int(capture), graph_handle=int(graph), stream_handle=int(stream))
        if identity['capture_id'] <= 0 or identity['graph_handle'] <= 0:
            raise ValueError('Active capture must expose its original positive identities')
        if previous is not None and any(previous[key] != value for key, value in identity.items()):
            raise ValueError('Capture identity changed before the node interval ended')
        _, count = self.graph_read(runtime.cudaGraphGetNodes, graph)
        if type(count) is not int or not 0 <= count <= self.limits.trace_events:
            raise ValueError('Capture node count exceeds the original trace event allowance')
        nodes = []
        if count:
            nodes, filled = self.graph_read(runtime.cudaGraphGetNodes, graph, numNodes=count)
            _, final_count = self.graph_read(runtime.cudaGraphGetNodes, graph)
            if filled != count or final_count != count or len(nodes) != count:
                raise ValueError('Capture node enumeration changed during observation')
        handles = {int(node) for node in nodes}
        if len(handles) != count or any(handle <= 0 for handle in handles):
            raise ValueError('Native capture node handles must be positive and unique')
        old = previous['nodes'] if previous is not None else {}
        if not old.keys() <= handles:
            raise ValueError('Previously observed nodes disappeared during the same capture')
        graph_id = (previous['cupti_graph_id'] if previous is not None else
                    self.cupti_id('cuptiGetGraphId', int(graph), ctypes.c_uint32))
        additions = []
        ids = set(old.values())
        for node in nodes:
            handle = int(node)
            if handle in old:
                continue
            node_id = self.cupti_id('cuptiGetGraphNodeId', handle, ctypes.c_uint64)
            if node_id in ids:
                raise ValueError('CUPTI node identity is not unique within this capture')
            node_type, = self.graph_read(runtime.cudaGraphNodeGetType, node)
            additions.append(dict(cupti_node_id=node_id, node_type=int(node_type),
                                  node_type_name=node_type.name))
            ids.add(node_id)
        # Commit only after every read succeeds. No graph/node object escapes.
        current = previous if previous is not None else dict(identity, cupti_graph_id=graph_id, nodes={})
        new_handles = [int(node) for node in nodes if int(node) not in old]
        current['nodes'].update(zip(new_handles, (row['cupti_node_id'] for row in additions)))
        return current, dict(identity, cupti_graph_id=graph_id, total_nodes=count,
            baseline=previous is None, nodes=additions,
            association='NODE_ADDITIONS_BETWEEN_HOST_BOUNDARIES_NOT_EXCLUSIVE_OWNERSHIP',
            replay_node_mapping='UNVERIFIED', gpu_completion='UNMEASURED')

    def error(self, kind, failure):
        if not self.errors:
            self.errors.append(dict(kind=kind, error_type=model_observation.class_name(failure),
                                    error=str(failure)))
        self.state.unverified = True
        if self.current is not None:
            self.current['status'] = 'UNVERIFIED'

    def before_start(self, label):
        if (label not in WINDOWS or self.current is not None or self.errors or
                [row['label'] for row in self.windows] != list(WINDOWS[:WINDOWS.index(label)])):
            raise ValueError('Original capture/request windows must start once in their native order')
        if self.worker.profiler is not self.profiler or self.profiler.is_running or self.profiler._active:
            raise ValueError('The original idle native profiler must own the diagnostic window')
        if self.clock() >= self.deadline:
            raise TimeoutError('Native profiling exceeded the original workload deadline')
        from role_execution_observation import reserve_device_bytes
        reserve_device_bytes(self.worker, self.state.device_owned_bytes,
                             profiler_bytes=self.limits.trace_bytes, device_preparation=False, label=label)
        self.current = dict(label=label, status='START_REQUESTED_NOT_RUNNING',
            start=self.clock(), profiler_id=id(self.profiler), wrapper_source=dict(self.sources),
            handler_calls=0, trace=None, native_start='UNMEASURED', native_stop='UNMEASURED')

    def after_start(self):
        if self.current is None or self.worker.profiler is not self.profiler or not self.profiler.is_running:
            raise ValueError('Successful profile RPC did not establish a running original native profiler')
        if self.current['native_start'] != 'ORIGINAL_START_RETURNED':
            raise ValueError('The original native profiler start did not return successfully')
        self.current['native_start'] = 'ORIGINAL_PROFILER_RUNNING'
        self.current['status'] = 'COLLECTING_DIAGNOSTIC_TRACE_NOT_G2'

    def trace_ready(self, profiler):
        """Observe export completion independently of swallowed native stop failures."""
        try:
            window = self.current
            if (window is None or self.worker.profiler is not self.profiler or
                    profiler is not self.profiler.profiler or window['handler_calls'] != 0):
                raise ValueError('Trace handler must be delivered once by the original active profiler')
            window['handler_calls'] += 1
            path = self.output/f'profile_rank_{self.worker.rank}_pid_{os.getpid()}_{window["label"]}.json'
            exported = export_trace(profiler, path, self.limits.trace_bytes, self.deadline, clock=self.clock)
            window['trace'] = exported
            if exported['status'] != 'EXPORTED_WITHIN_BYTE_AND_DEADLINE_LIMITS':
                raise ValueError('Native trace export is incomplete, over budget or beyond the original deadline')
            exported['inspection_started'] = self.clock()
            payload = path.read_bytes()
            events = json.loads(payload)['traceEvents']
            if (not isinstance(events, list) or not events or len(events) > self.limits.trace_events or
                    any(not isinstance(row, dict) for row in events)):
                raise ValueError('Native trace has missing activity or exceeds its event limit')
            if self.clock() >= self.deadline:
                raise TimeoutError('Trace inspection exceeded the original workload deadline')
            exported.update(sha256=hashlib.sha256(payload).hexdigest(), events=len(events),
                status='COMPLETE_NATIVE_TRACE_NOT_G2',
                inspection_started=exported.get('inspection_started'),
                annotations=sum(row.get('cat') == 'user_annotation' and
                    row.get('name', '').startswith(ANNOTATION_PREFIX) for row in events),
                graph_launches=sum(row.get('cat') in ('cuda_runtime', 'cuda_driver') and
                    row.get('name', '').startswith(('cudaGraphLaunch', 'cuGraphLaunch')) for row in events),
                kernels=sum(row.get('cat') == 'kernel' for row in events), inspection_finished=self.clock())
        except BaseException as failure:
            self.error('native_trace_completion', failure)

    def after_stop(self):
        window = self.current
        try:
            if window is None:
                raise ValueError('Native profiler stop has no original active diagnostic window')
            if self.worker.profiler is not self.profiler or self.profiler.is_running or self.profiler._active:
                raise ValueError('Native profiler ownership or stop completion is unverified')
            if (window['native_start'] != 'ORIGINAL_PROFILER_RUNNING' or window['handler_calls'] != 1 or
                    window['native_stop'] != 'ORIGINAL_STOP_RETURNED' or window['trace'] is None or
                    window['trace']['status'] != 'COMPLETE_NATIVE_TRACE_NOT_G2' or self.errors):
                raise ValueError('Profiler stop returned without a complete verified trace handler')
            window['native_stop'] = 'ORIGINAL_PROFILER_STOPPED_AND_TRACE_COMPLETE'
            window['status'] = 'COMPLETE_DIAGNOSTIC_WINDOW_NOT_G2'
        except BaseException as failure:
            self.error('native_profiler_stop', failure)
        finally:
            if window is not None:
                # While an export reader is still alive it retains ownership of
                # the remaining allowance; its bytes are never charged twice.
                trace = window['trace']
                remaining = (self.limits.trace_bytes
                    if trace is not None and not trace.get('reader_reaped', False) else 0)
                try:
                    from role_execution_observation import reserve_device_bytes
                    reserve_device_bytes(self.worker, self.state.device_owned_bytes,
                        profiler_bytes=remaining, device_preparation=False, label=window['label'])
                except BaseException as failure:
                    self.error('native_profiler_reservation', failure)
                try:
                    window['stop'] = self.clock()
                except BaseException as failure:
                    window['stop'] = None
                    self.error('native_profiler_stop_clock', failure)
                self.windows.append(window)
                self.current = None
                try:
                    row = dict(scope='native_profiler_completion_only_not_G2', rank=self.worker.rank,
                        local_rank=self.worker.local_rank, pid=os.getpid(),
                        worker_identity=dict(getattr(self.worker, '_capacity_identity', {})),
                        observer_sha256=SOURCE_SHA256, window=window, errors=list(self.errors))
                    self.record(self.worker, json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
                    self.state.records_written += 1
                except BaseException as failure:
                    window['status'] = 'UNVERIFIED'
                    self.error('native_profiler_completion_record', failure)

    def bind_profiler(self, profiler_config):
        """Construct the original wrapper with only lifecycle and trace-handler observation."""
        from vllm.profiler.wrapper import TorchProfilerWrapper

        def native_boundary(name, field):
            original = getattr(TorchProfilerWrapper, name)
            def call(wrapper):
                try:
                    result = original(wrapper)
                except BaseException as failure:
                    self.error('native_profiler_'+name, failure)
                    if self.current is not None:
                        self.current[field] = 'ORIGINAL_METHOD_RAISED'
                    raise
                if self.current is not None:
                    self.current[field] = 'ORIGINAL_'+('START' if name == '_start' else 'STOP')+'_RETURNED'
                return result
            return call

        native_type = type('ObservedTorchProfiler', (TorchProfilerWrapper,), dict(
            _start=native_boundary('_start', 'native_start'), _stop=native_boundary('_stop', 'native_stop')))
        self.profiler = native_type(profiler_config,
            worker_name=f'step5_rank_{self.worker.rank}_pid_{os.getpid()}', local_rank=self.worker.local_rank,
            activities=('CPU', 'CUDA'), on_trace_ready=self.trace_ready)
        self.worker.profiler = self.profiler
        self.worker.model_runner._step5_profiler_observation = self
        return self.profiler

    def graph_identity(self, graph):
        """Read only an existing executable at a native capture/replay boundary."""
        if self.worker.profiler is not self.profiler:
            raise ValueError('Graph identity cannot use a substituted native profiler')
        import torch
        if not isinstance(graph, torch.cuda.CUDAGraph):
            return dict(cuda_graph_exec_handle=None,
                        cuda_graph_exec_identity='UNVERIFIED_NOT_TORCH_CUDAGRAPH')
        handle = torch.cuda.CUDAGraph.raw_cuda_graph_exec(graph)
        if type(handle) is not int or handle <= 0:
            raise ValueError('Original graph executable getter did not return an instantiated identity')
        return dict(cuda_graph_exec_handle=handle,
                    cupti_graph_exec_id=self.cupti_id('cuptiGetGraphExecId', handle, ctypes.c_uint32),
                    cuda_graph_exec_identity='OBSERVED_READ_ONLY_HANDLE_NOT_NODE_BINDING')

    def annotate(self, name):
        if self.worker.profiler is not self.profiler:
            raise ValueError('Native trace annotation cannot use a substituted profiler')
        if self.current is None or not self.profiler.is_running:
            return None
        return self.profiler.annotate_context_manager(name)

    def summary(self):
        return dict(status='UNVERIFIED' if self.errors else 'NATIVE_PROFILER_DIAGNOSTIC_ONLY_NOT_G2',
            observer_sha256=SOURCE_SHA256, sources=copy.deepcopy(self.sources), limits=vars(self.limits),
            worker_identity=dict(getattr(self.worker, '_capacity_identity', {})),
            windows=copy.deepcopy(self.windows), current=copy.deepcopy(self.current), errors=copy.deepcopy(self.errors),
            physical_kernel_attribution='UNVERIFIED_NO_CAPTURE_NODE_BINDING',
            transient_scale_content='UNMEASURED', clean_performance='UNVERIFIED', G2='UNVERIFIED')


def create_observation(worker, state, record):
    """Bind the existing native wrapper through its original trace-handler interface."""
    if 'STEP5_PHYSICAL_OBSERVATION' not in os.environ:
        return None
    if os.environ['STEP5_PROFILER_OBSERVER_SHA'] != SOURCE_SHA256:
        raise ValueError('Native profiler observer source identity mismatch')
    from role_capacity_observation import observation_output
    output, record_bytes = observation_output()
    node = json.loads(Path(os.environ['STEP5_CAPACITY_NODE_FILE']).read_text())
    deadline = node['workload_deadline']
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError('Native profiler requires the original local monotonic deadline')
    configured = json.loads(os.environ['STEP5_PHYSICAL_OBSERVATION'])
    limits = ProfilerObservationLimits(**configured)
    if (node['physical_observation'] != configured or
            node['profiler_observer_sha256'] != SOURCE_SHA256):
        raise ValueError('Native profiler worker environment differs from the original node configuration')
    if worker.profiler is not None:
        raise ValueError('An existing native profiler must not be replaced for a diagnostic witness')
    from vllm.profiler.wrapper import TorchProfilerWrapper
    from vllm.v1.worker.gpu_worker import Worker
    from torch.cuda import CUDAGraph
    sources = {}
    for function, owner, key in (
            (TorchProfilerWrapper.__init__, ('TorchProfilerWrapper', '__init__'), 'wrapper'),
            (TorchProfilerWrapper._start, ('TorchProfilerWrapper', '_start'), 'wrapper'),
            (TorchProfilerWrapper._stop, ('TorchProfilerWrapper', '_stop'), 'wrapper'),
            (TorchProfilerWrapper.annotate_context_manager, ('TorchProfilerWrapper', 'annotate_context_manager'), 'wrapper'),
            (Worker.profile, ('Worker', 'profile'), 'worker'),
            (Worker._get_cudagraph_capture_context, ('Worker', '_get_cudagraph_capture_context'), 'worker'),
            (CUDAGraph.raw_cuda_graph_exec, ('CUDAGraph', 'raw_cuda_graph_exec'), 'torch_graphs')):
        model_observation.bind_code(function, owner, SOURCE_PINS[key][1], sources)
    if any(getattr(worker.profiler_config, key) != value for key, value in native_config(output).items()):
        raise ValueError('Native profiler config differs from the unscheduled all-rank diagnostic policy')
    state.rank_limit = record_bytes//8
    observed = NativeProfilerObservation(worker, limits, output, deadline, record, state, sources)
    observed.bind_graph_api()
    observed.bind_profiler(worker.profiler_config)
    return observed


if __name__ == '__main__':
    fifo, output, limit, deadline = sys.argv[1:]
    sink = os.fdopen(int(output), 'wb', buffering=0)
    os.write(sys.stdout.fileno(), b'READY\n')
    drained = drain_trace(int(fifo), sink, int(limit), float(deadline))
    print(json.dumps(drained, allow_nan=False), flush=True)
