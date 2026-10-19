"""Verify the original Torch wrapper and capped trace export without using GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from vllm.config import ProfilerConfig
from vllm.profiler.wrapper import TorchProfilerWrapper

import role_execution_observation as execution
import role_model_input_observation as model
import role_profiler_observation as observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    report = dict(scope='LOCKED_IMAGE_NATIVE_WRAPPER_CPU_API_ONLY_NOT_CUDA_OR_G2',
        torch_version=torch.__version__, cuda=torch.version.cuda,
        torch_git_revision=torch.version.git_version, sources={}, cases=[], G2='UNVERIFIED')
    for name in ('__init__', '_start', '_stop', 'annotate_context_manager'):
        model.bind_code(getattr(TorchProfilerWrapper, name), ('TorchProfilerWrapper', name),
            observation.SOURCE_PINS['wrapper'][1], report['sources'])
    model.bind_code(torch.cuda.CUDAGraph.raw_cuda_graph_exec, ('CUDAGraph', 'raw_cuda_graph_exec'),
        observation.SOURCE_PINS['torch_graphs'][1], report['sources'])
    for label, cap in (('complete', 262144), ('capped', 31)):
        records = output/label
        records.mkdir()
        node = output/(label+'_node.json')
        node.write_text(json.dumps(dict(observer_output=str(records), record_bytes=8*1024*1024)))
        os.environ['STEP5_CAPACITY_NODE_FILE'] = str(node)
        identity = dict(rank=0, local_rank=0, pid=os.getpid(), start_time=1, parent_pid=os.getppid(),
            parent_start_time=1, generation='locked-cpu-api-'+label, node_rank=0, boot_id='cpu-api',
            pid_namespace='cpu-api', observer_sha256=execution.SOURCE['capacity_observer_sha256'])
        (records/f'worker_0_{os.getpid()}.json').write_text(json.dumps(identity))
        state = execution.RequestObservation()
        worker = SimpleNamespace(rank=0, local_rank=0, _capacity_identity=identity,
                                 _request_observation=state, model_runner=SimpleNamespace(), profiler=None)
        observed = observation.NativeProfilerObservation(worker,
            observation.ProfilerObservationLimits(cap, 100000), records, time.monotonic()+20,
            execution.write_record, state, report['sources'])
        state.profiler = observed
        original = observed.bind_profiler(ProfilerConfig(**observation.native_config(records)))
        assert original is worker.profiler and isinstance(original, TorchProfilerWrapper)
        assert worker.model_runner._step5_profiler_observation is observed
        windows = observation.WINDOWS if label == 'complete' else ('capture',)
        for window in windows:
            observed.before_start(window)
            original.start(delay_iterations=0, max_iterations=0)
            observed.after_start()
            assert original.is_running and original._active
            assert observed.current['native_start'] == 'ORIGINAL_PROFILER_RUNNING'
            lifecycle = 'capture' if window == 'capture' else 'execute'
            with observed.annotate(observation.annotation_name(lifecycle, 1, 1, 'call')):
                torch.ones(2)+1
            original.step()
            original.stop()
            observed.after_stop()
            assert not original.is_running and not original._active
        summary = observed.summary()
        assert summary['current'] is None and summary['G2'] == 'UNVERIFIED'
        assert execution.device_reservation_bytes(records) == 0
        if label == 'complete':
            assert not summary['errors']
            assert all(row['status'] == 'COMPLETE_DIAGNOSTIC_WINDOW_NOT_G2' for row in summary['windows'])
            assert all(row['trace']['annotations'] == 1 for row in summary['windows'])
            assert all(row['native_stop'] == 'ORIGINAL_PROFILER_STOPPED_AND_TRACE_COMPLETE'
                       for row in summary['windows'])
        else:
            assert summary['errors'] and summary['windows'][0]['status'] == 'UNVERIFIED'
            trace = summary['windows'][0]['trace']
            assert trace['reader_complete'] and trace['retained_bytes'] <= cap < trace['bytes_seen']
        report['cases'].append(dict(label=label, summary=summary,
            files={path.name:dict(bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                   for path in records.iterdir() if path.is_file()}))
    report.update(cuda_initialized=torch.cuda.is_initialized(), status='PASS')
    assert not report['cuda_initialized']
    report['observer_sha256'] = observation.SOURCE_SHA256
    with (output/'api_report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], scope=report['scope'],
                         cases=[row['label'] for row in report['cases']],
                         cuda_initialized=report['cuda_initialized'])), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
