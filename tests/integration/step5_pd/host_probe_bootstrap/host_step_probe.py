"""Diagnostic host-step probe for native Step-5 vLLM role servers.

Each worker process writes one JSON line per model-runner step to
`<output_dir>/<role>_<host>_<pid>.jsonl`, after a header line with its host, TP and DP rank.
A step line holds host `perf_counter_ns` stamps around:

- `GPUModelRunner.execute_model` (`enter`, `exit`) and `sample_tokens` (`sample_exit`);
- `KVBlockZeroer.zero_block_ids` (`zero`), with the hook count, the block counts, and for
  the first side-storage hook its entry, the return of its `.tolist()` (`first_sync`), its
  exit, and whether the previous step's device work was still running at its entry;
- the DP padding sync `sync_cudagraph_and_dp_padding` (`sync`: the per-process call ordinal
  that joins the DP engines, and `reduce_enter`/`reduce_exit` around its gloo `all_reduce`,
  the only part that waits for the other engines);
- `StepTimingCollector.forward_start` (`launch`).

CUDA events at the launch and at the end of the step's main-stream work (`sample_tokens`
exit, or `_dummy_run` exit for a dummy step) give `gap_ms`, the main-stream time from the
previous step's end to this step's launch, and `span_ms`, launch to step end.

`GPUModelRunner.shutdown`, which every orderly worker exit calls and which synchronizes the
device first, writes the remaining steps and a final `end` line with the step count. A file
without it is truncated. Forked workers leave through `os._exit`, so `atexit` never runs.

The probe never changes arguments, return values or exceptions, and calls every original
exactly once. The first probe error disables the probe in that process, writes an `error`
line, and prints an UNVERIFIED line. The hooks perturb native timing, so a probe run is
never a reference run.
"""
import functools
import importlib.abc
import importlib.machinery
import inspect
import json
import os
import socket
import sys
import time

ENV = 'STEP5_HOST_STEP_PROBE'
RUNNER = 'vllm.v1.worker.gpu.model_runner'
ZEROER = 'vllm.v1.worker.utils'
DP_SYNC = 'vllm.v1.worker.gpu.dp_utils'
STEP_TIMING = 'vllm.v1.worker.gpu.async_utils'
SPARSE_SUMMARY = 'vllm.models.step5.nvidia.sparse_summary_cache'
MODULES = frozenset((RUNNER, ZEROER, DP_SYNC, STEP_TIMING, SPARSE_SUMMARY))
MARK = '_step5_host_probe'

_installed = None


def now():
    return time.perf_counter_ns()


def device_event():
    import torch

    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


class Probe:
    def __init__(self, settings):
        self.role = settings['role']
        self.output_dir = settings['output_dir']
        self.disabled = False
        self.pid = None
        self.file = None
        self.clear_steps()

    def clear_steps(self):
        self.sequence = 0
        self.ordinal = 0
        self.current = None
        self.last = None
        self.last_end = None
        self.zeroing = None
        self.sync = None
        self.pending = []

    def start_process(self, runner):
        """A forked worker inherits the parent's state; it starts its own file."""
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        self.pid = os.getpid()
        self.clear_steps()
        host = socket.gethostname()
        os.makedirs(self.output_dir, exist_ok=True)
        self.file = open(os.path.join(self.output_dir, f'{self.role}_{host}_{self.pid}.jsonl'),
                         'w', buffering=1)
        self.write(dict(kind='header', role=self.role, host=host, pid=self.pid,
                        tp_rank=get_tensor_model_parallel_rank(),
                        tp_size=get_tensor_model_parallel_world_size(),
                        dp_rank=runner.dp_rank, dp_size=runner.dp_size,
                        clock='perf_counter_ns', wall_ns=time.time_ns(), perf_ns=now()))

    def write(self, record):
        self.file.write(json.dumps(record, separators=(',', ':')) + '\n')

    def safe(self, where, callback):
        try:
            return callback()
        except Exception as error:
            self.disabled = True
            print(f'{ENV} UNVERIFIED {where}: {error!r}', file=sys.stderr, flush=True)
            try:
                self.write(dict(kind='error', where=where, error=repr(error)))
            except Exception:
                pass
            return None

    def resolve(self):
        """Write finished steps in order once their device events completed."""
        while self.pending:
            record, events = self.pending[0]
            if not all(event.query() for event in events.values()):
                return
            self.pending.pop(0)
            if 'previous_end' in events and 'launch' in events:
                record['gap_ms'] = events['previous_end'].elapsed_time(events['launch'])
            if 'launch' in events and 'end' in events:
                record['span_ms'] = events['launch'].elapsed_time(events['end'])
            self.write(record)

    def finish_last(self):
        if self.last is not None:
            self.pending.append(self.last)
            self.last = None

    def step_before(self, arguments):
        runner = arguments['self']
        if self.pid != os.getpid():
            self.start_process(runner)
        self.finish_last()
        self.resolve()
        scheduler_output = arguments['scheduler_output']
        self.sequence += 1
        record = dict(kind='step', seq=self.sequence, wall_ns=time.time_ns(), enter=now(),
                      dummy=arguments.get('dummy_run', False),
                      profile=arguments.get('is_profile', False),
                      reqs=len(scheduler_output.num_scheduled_tokens),
                      tokens=scheduler_output.total_num_scheduled_tokens)
        events = {} if self.last_end is None else {'previous_end': self.last_end}
        self.current = (record, events)
        return self.current

    def step_after(self, current):
        record, events = current
        record['exit'] = now()
        self.current = None
        self.last = current

    def dummy_finished(self, token):
        """`_dummy_run` queues more main-stream work after its `execute_model` returns."""
        if self.last is None:
            return
        record, events = self.last
        if record['dummy'] and 'launch' in events and 'end' not in events:
            events['end'] = self.last_end = device_event()

    def sampled(self, token):
        if self.last is None:
            return
        record, events = self.last
        if 'launch' in events and 'end' not in events:
            record['sample_exit'] = now()
            events['end'] = self.last_end = device_event()

    def zeroing_before(self, arguments):
        if self.current is None:
            return None
        groups = arguments.get('block_ids_by_group')
        self.zeroing = dict(enter=now(), blocks=len(arguments['block_ids']),
                            groups=None if groups is None else [len(ids) for ids in groups],
                            hooks=0)
        return self.zeroing

    def zeroing_after(self, zeroing):
        if zeroing is None:
            return
        zeroing['exit'] = now()
        self.current[0]['zero'] = zeroing
        self.zeroing = None

    def hook_before(self, arguments):
        zeroing = self.zeroing
        if zeroing is None:
            return
        zeroing['hooks'] += 1
        if zeroing['hooks'] == 1:
            zeroing['first_enter'] = now()
            zeroing['prior_pending'] = self.last_end is not None and not self.last_end.query()

    def hook_after(self, token):
        zeroing = self.zeroing
        if zeroing is not None and zeroing['hooks'] == 1:
            zeroing['first_exit'] = now()

    def ids_returned(self, token):
        zeroing = self.zeroing
        if zeroing is not None and zeroing['hooks'] == 1 and 'first_sync' not in zeroing:
            zeroing['first_sync'] = now()

    def sync_before(self, arguments):
        self.ordinal += 1
        self.sync = dict(ordinal=self.ordinal, enter=now())
        return self.sync

    def reduce_before(self, arguments):
        if self.sync is not None:
            self.sync['reduce_enter'] = now()

    def reduce_after(self, token):
        if self.sync is not None:
            self.sync['reduce_exit'] = now()

    def sync_after(self, sync):
        # Every call is one DP all-reduce, so the ordinal counts calls outside a step too.
        self.sync = None
        if self.current is not None:
            sync['exit'] = now()
            self.current[0]['sync'] = sync

    def launched(self, arguments):
        if self.current is not None:
            record, events = self.current
            record['launch'] = now()
            events['launch'] = device_event()

    def close(self, token):
        """Native shutdown has synchronized the device, so every step event has completed."""
        if self.pid != os.getpid() or self.file is None:
            return
        self.finish_last()
        self.resolve()
        self.write(dict(kind='end', steps=self.sequence))
        self.file.close()
        self.file = None


class DistributedSeam:
    """Stands in for `torch.distributed` inside the DP sync module only."""

    def __init__(self, distributed):
        self.distributed = distributed

    def __getattr__(self, name):
        return getattr(self.distributed, name)


def wrap(owner, name, probe, before=None, after=None, bind=False, static=False):
    """Only callbacks are guarded; the original runs exactly once."""
    original = getattr(owner, name)
    if getattr(original, MARK, False):
        return
    signature = inspect.signature(original) if bind else None

    @functools.wraps(original)
    def probed(*args, **kwargs):
        token = None
        if before is not None and not probe.disabled:
            token = probe.safe(name, lambda: before(
                None if signature is None else signature.bind(*args, **kwargs).arguments))
        result = original(*args, **kwargs)
        if after is not None and not probe.disabled:
            probe.safe(name, lambda: after(token))
        return result

    setattr(probed, MARK, True)
    setattr(owner, name, staticmethod(probed) if static else probed)


def instrument(module, probe):
    name = module.__name__
    if name == RUNNER:
        wrap(module.GPUModelRunner, 'execute_model', probe,
             before=probe.step_before, after=probe.step_after, bind=True)
        wrap(module.GPUModelRunner, 'sample_tokens', probe, after=probe.sampled)
        wrap(module.GPUModelRunner, '_dummy_run', probe, after=probe.dummy_finished)
        wrap(module.GPUModelRunner, 'shutdown', probe, after=probe.close)
    elif name == ZEROER:
        wrap(module.KVBlockZeroer, 'zero_block_ids', probe,
             before=probe.zeroing_before, after=probe.zeroing_after, bind=True)
    elif name == DP_SYNC:
        # dispatch_cg_and_sync_dp resolves this module global at each call.
        wrap(module, 'sync_cudagraph_and_dp_padding', probe,
             before=probe.sync_before, after=probe.sync_after)
        # Replacing the module's own `dist` name stamps no other caller's all_reduce.
        if not isinstance(module.dist, DistributedSeam):
            module.dist = DistributedSeam(module.dist)
        wrap(module.dist, 'all_reduce', probe, before=probe.reduce_before, after=probe.reduce_after)
    elif name == STEP_TIMING:
        wrap(module.StepTimingCollector, 'forward_start', probe, before=probe.launched)
    elif name == SPARSE_SUMMARY:
        # Hooks are bound at KV cache setup, after this import, so they bind the wrapper.
        layer = module.Step5SparseSummaryCacheLayer
        wrap(layer, 'zero_kv_cache_side_storage', probe,
             before=probe.hook_before, after=probe.hook_after)
        wrap(layer, '_ids', probe, after=probe.ids_returned, static=True)


class ProbeLoader(importlib.abc.Loader):
    def __init__(self, loader, probe):
        self.loader, self.probe = loader, probe

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        # A native import failure propagates once. Never retry module execution.
        self.loader.exec_module(module)
        self.probe.safe('instrument.' + module.__name__, lambda: instrument(module, self.probe))

    def __getattr__(self, name):
        return getattr(self.loader, name)


class ProbeFinder(importlib.abc.MetaPathFinder):
    def __init__(self, probe):
        self.probe = probe

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = ProbeLoader(spec.loader, self.probe)
        return spec


def install():
    """Activate only through the per-server JSON environment contract."""
    global _installed
    if not os.environ.get(ENV) or _installed is not None:
        return _installed
    probe = Probe(json.loads(os.environ[ENV]))
    _installed = probe
    sys.meta_path.insert(0, ProbeFinder(probe))
    for name in sorted(MODULES & sys.modules.keys()):
        probe.safe('instrument.' + name, lambda: instrument(sys.modules[name], probe))
    return probe
