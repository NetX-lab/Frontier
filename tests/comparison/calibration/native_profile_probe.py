"""Probe native stream identities and capture attribution; never serving costs."""

import argparse
import hashlib
import inspect
import json
import logging
import os
import sys
import traceback
from pathlib import Path


def fingerprint(path):
    path = Path(path).resolve()
    with path.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    return dict(path=str(path), sha256=digest)


def joined(events, label):
    """Retain exact same-thread containment and correlation evidence, not names."""
    from forward_device_timeline import LAUNCH_CATEGORIES
    markers = [e for e in events if e.get('ph') == 'X'
               and e.get('cat') == 'user_annotation' and e.get('name') == label]
    launches, kernels = [], []
    if len(markers) == 1:
        marker = markers[0]
        launches = [e for e in events if e.get('ph') == 'X'
                    and e.get('cat') in LAUNCH_CATEGORIES
                    and (e.get('pid'), e.get('tid')) == (marker.get('pid'), marker.get('tid'))
                    and marker['ts'] <= e['ts']
                    and e['ts'] + e['dur'] <= marker['ts'] + marker['dur']]
        correlations = {e['args']['correlation'] for e in launches
                        if e.get('args', {}).get('correlation') is not None}
        kernels = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'kernel'
                   and e.get('args', {}).get('correlation') in correlations]
    return dict(annotations=markers, runtime=launches, kernels=kernels)


def graph_identity(event):
    args = event.get('args', {})
    pair = (args.get('graph id'), args.get('graph node id'))
    return pair if all(type(value) is int and value > 0 for value in pair) else None


def graph_correspondence(report):
    cycles = report.get('cycles', {})
    if cycles.get('1', {}).get('capture_returned') is not True:
        return dict(status='UNKNOWN', reason='Capture did not complete successfully')
    capture = cycles['1'].get('capture_ops', {})
    owners = {}
    for label, row in capture.items():
        identities = {graph_identity(e) for e in row.get('runtime', []) + row.get('kernels', [])}
        identities.discard(None)
        if len(row.get('annotations', [])) != 1 or not identities:
            return dict(status='UNKNOWN', reason='No direct captured per-op graph/node identities')
        for pair in identities:
            if pair in owners:
                return dict(status='UNKNOWN', reason='Captured graph/node identity has multiple owners')
            owners[pair] = label
    if len(capture) != 2:
        return dict(status='UNKNOWN', reason='Two captured CPU annotations are not retained')
    transfers = {}
    for cycle in ('1', '2'):
        state = cycles.get(cycle, {})
        if state.get('replay_returned') is not True:
            return dict(status='UNKNOWN', reason=f'Replay cycle {cycle} did not complete successfully')
        replay = state.get('replay', {})
        kernels = replay.get('kernels', [])
        pairs = [graph_identity(e) for e in kernels]
        if (len(replay.get('annotations', [])) != 1 or not kernels
                or None in pairs or len(set(pairs)) != len(pairs)
                or set(pairs) != set(owners)):
            return dict(status='UNKNOWN', reason='Replay does not exactly cover captured graph/node IDs')
        transfers[cycle] = [dict(graph_id=p[0], node_id=p[1], capture_annotation=owners[p]) for p in pairs]
    return dict(status='PASS', basis='Exact trace graph/node identities only', transfers=transfers)


def run(output, report, save):
    import torch
    report['runtime'] = dict(python=sys.version, executable=sys.executable,
                             platform=sys.platform, torch=torch.__version__, cuda=torch.version.cuda)
    report['sources'] = [fingerprint(p) for p in {torch.__file__, torch._C.__file__, inspect.getfile(torch.cuda.CUDAGraph), inspect.getfile(torch.profiler.profile)}]
    save()
    import native_serve_profile as collector
    from vllm.config import ProfilerConfig
    from vllm.profiler.wrapper import TorchProfilerWrapper
    from forward_device_timeline import load_trace

    callables = (collector.ServingProfile.stream, collector.ServingProfile.trace_stream,
                 TorchProfilerWrapper, ProfilerConfig, load_trace)
    report['sources'].extend(fingerprint(path) for path in sorted({inspect.getfile(item) for item in callables}))
    report['vllm_version'] = sys.modules['vllm.version'].__version__
    report['environment'] = {key: os.environ.get(key) for key in
                             ('LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'FRONTIER_NATIVE_PROFILE_LIMITS')}
    logging.getLogger('vllm.profiler.wrapper').addHandler(logging.FileHandler(output / 'native_profiler.log'))
    observer = collector.ServingProfile(None)
    report['collector_errors'] = observer.errors

    def attempt(stage, action):
        try:
            return action()
        except Exception:
            report['errors'].append(dict(stage=stage, traceback=traceback.format_exc()))
            save()

    def snapshot(stage):
        rows = {}
        report['snapshots'][stage] = rows
        for name, stream in streams.items():
            with torch.cuda.stream(stream):
                key = observer.stream(stream)
                rows[name] = dict(observer.streams[key])
                rows[name]['queried_cupti_id'] = attempt(stage + ':' + name,
                                                       lambda: observer.trace_stream(int(stream.cuda_stream)))
        save()

    def operation(label, name, multiply=False):
        with torch.cuda.stream(streams[name]), wrapper.annotate_context_manager(label):
            x, y = tensors[name]
            (torch.mul if multiply else torch.add)(x, x, out=y)

    device = torch.cuda.get_device_properties(0)
    report['runtime']['gpu'] = dict(properties=str(device), uuid=str(device.uuid))
    from cuda.bindings import runtime
    report['sources'].append(fingerprint(runtime.__file__))
    report['runtime']['driver_api_version'] = tuple(map(int, runtime.cudaDriverGetVersion()))
    report['runtime']['runtime_api_version'] = tuple(map(int, runtime.cudaRuntimeGetVersion()))
    streams = dict(default=torch.cuda.default_stream(), aux=torch.cuda.Stream())
    tensors = {name: (torch.ones(1024, device='cuda'), torch.empty(1024, device='cuda')) for name in streams}
    torch.cuda.synchronize()
    trace_dir = output / 'native_traces'
    trace_dir.mkdir()
    config = ProfilerConfig(profiler='torch', torch_profiler_dir=str(trace_dir),
                            torch_profiler_with_stack=False, torch_profiler_dump_cuda_time_total=False,
                            torch_profiler_use_gzip=True, delay_iterations=0, max_iterations=0,
                            warmup_iterations=0, wait_iterations=0)
    report['profiler_config'] = str(config)
    wrapper = TorchProfilerWrapper(config, 'native_stream_probe', 0, ['CPU', 'CUDA'])
    report['profiler_objects'] = dict(wrapper=id(wrapper), torch_profiler=id(wrapper.profiler))
    graph = None
    for cycle in (1, 2):
        state = dict(eager={}, priming={}, status='UNKNOWN')
        report['cycles'][str(cycle)] = state
        before = set(trace_dir.glob('*.pt.trace.json.gz'))
        attempt(f'before_cycle{cycle}', lambda: snapshot(f'before_cycle{cycle}'))
        try:
            wrapper.start()
            state.update(running_after_start=wrapper.is_running, wrapper=id(wrapper), torch_profiler=id(wrapper.profiler))
            save()
            if not wrapper.is_running:
                raise RuntimeError(f'Native profiler cycle {cycle} did not start; see native_profiler.log')
            attempt(f'during_cycle{cycle}', lambda: snapshot(f'during_cycle{cycle}'))
            for name in streams:
                # Match RecordFunctionTracer's four separately retained priming kernels.
                for index in range(4):
                    label = f'probe.c{cycle}.prime.{name}.{index}'
                    state['priming'][label] = {}
                    operation(label, name)
                for op in ('add', 'mul'):
                    label = f'probe.c{cycle}.eager.{name}.{op}'
                    state['eager'][label] = dict(stream=name)
                    operation(label, name, op == 'mul')
            torch.cuda.synchronize()
            attempt(f'after_eager_cycle{cycle}', lambda: snapshot(f'after_eager_cycle{cycle}'))
            if cycle == 1:
                def capture():
                    nonlocal graph
                    graph = torch.cuda.CUDAGraph()
                    state['capture_ops'] = {}
                    with wrapper.annotate_context_manager('probe.capture_and_instantiate'):
                        with torch.cuda.graph(graph, stream=streams['aux']):
                            for op in ('add', 'mul'):
                                label = 'probe.capture.' + op
                                state['capture_ops'][label] = {}
                                operation(label, 'aux', op == 'mul')
                    state.update(capture_returned=True, instantiate_basis='Native capture_end with keep_graph=False')
                attempt('capture_and_instantiate', capture)
            if graph is not None and report['cycles']['1'].get('capture_returned'):
                def replay():
                    with torch.cuda.stream(streams['aux']), wrapper.annotate_context_manager(f'probe.c{cycle}.replay'):
                        graph.replay()
                    torch.cuda.synchronize()
                    state['replay_returned'] = True
                attempt(f'replay_cycle{cycle}', replay)
        except Exception:
            report['errors'].append(dict(stage=f'cycle{cycle}', traceback=traceback.format_exc()))
        finally:
            attempt(f'flush_cycle{cycle}', torch.cuda.synchronize)
            attempt(f'stop_cycle{cycle}', wrapper.stop)
            state.update(running_after_stop=wrapper.is_running, wrapper_after_stop=id(wrapper), torch_profiler_after_stop=id(wrapper.profiler))
            state['traces'] = [fingerprint(path) for path in sorted(set(trace_dir.glob('*.pt.trace.json.gz')) - before)]
            attempt(f'after_cycle{cycle}', lambda: snapshot(f'after_cycle{cycle}'))
            save()
        def analyze():
            if len(state['traces']) != 1:
                raise ValueError(f'Expected one complete native trace in cycle {cycle}')
            events = load_trace(Path(state['traces'][0]['path']))['traceEvents']
            for label in state['eager']:
                row = state['eager'][label]
                row.update(joined(events, label))
                expected = report['snapshots'].get(f'during_cycle{cycle}', {}).get(row['stream'], {})
                identifier = expected.get('queried_cupti_id')
                row['status'] = ('PASS' if identifier is not None and len(row['annotations']) == 1
                                 and row['kernels'] and expected.get('trace_stream_id') == identifier
                                 and all(k.get('args', {}).get('stream') == identifier for k in row['kernels']) else 'FAIL')
            state['status'] = 'PASS' if len(state['eager']) == 4 and all(r['status'] == 'PASS' for r in state['eager'].values()) else 'FAIL'

            def analyze_priming():
                for label in state['priming']:
                    state['priming'][label] = joined(events, label)

            def analyze_graph():
                for label in state.get('capture_ops', {}):
                    state['capture_ops'][label] = joined(events, label)
                state['capture_envelope'] = joined(events, 'probe.capture_and_instantiate')
                state['replay'] = joined(events, f'probe.c{cycle}.replay')

            attempt(f'analyze_priming_cycle{cycle}', analyze_priming)
            attempt(f'analyze_graph_cycle{cycle}', analyze_graph)
        attempt(f'analyze_cycle{cycle}', analyze)
        save()
    report['graph_correspondence'] = attempt('graph_correspondence', lambda: graph_correspondence(report)) or dict(status='UNKNOWN')
    expected_stages = [f'{stage}_cycle{cycle}' for cycle in (1, 2)
                       for stage in ('before', 'during', 'after_eager', 'after')]
    stable = all(len(report['snapshots'].get(stage, {})) == 2 and
                 len({r.get('queried_cupti_id') for r in report['snapshots'][stage].values()}) == 2 and
                 all(r['queried_cupti_id'] is not None and r['trace_stream_id'] == r['queried_cupti_id']
                     for r in report['snapshots'][stage].values()) for stage in expected_stages)
    report['eager_stream_status'] = ('PASS' if stable and not observer.errors
                                    and all(r['status'] == 'PASS' for r in report['cycles'].values()) else 'FAIL')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(task_id='W128-NATIVE-STREAM-PROBE', script=fingerprint(__file__),
                  errors=[], snapshots={}, cycles={}, eager_stream_status='UNKNOWN',
                  graph_correspondence=dict(status='UNKNOWN'), serving_cost_claim=False)
    def save():
        (output / 'probe_report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        run(output, report, save)
    except Exception:
        report['errors'].append(dict(stage='probe', traceback=traceback.format_exc()))
    finally:
        maps = Path('/proc/self/maps').read_text()
        (output / 'proc_maps.txt').write_text(maps)
        report['libraries'] = [fingerprint(path) for path in sorted({line.split()[-1] for line in maps.splitlines()
                               if any(name in line for name in ('libcupti', 'libcuda', 'libtorch', 'libkineto'))})]
        save()
    print(json.dumps(dict(report=str(output / 'probe_report.json'), eager=report['eager_stream_status'],
                          graph=report['graph_correspondence']['status'])))
    return 0 if report['eager_stream_status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
