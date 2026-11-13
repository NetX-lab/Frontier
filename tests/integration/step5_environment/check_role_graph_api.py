"""Exercise capture metadata on one GPU; this is not model or E2E ground truth."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output, archive = Path(config['local_output']), Path(config['archive'])
    cloud = Path('/mnt/codesign-exp/ycfeng/step5-preview')
    if not archive.is_relative_to(cloud) or archive == cloud or archive.exists():
        raise ValueError('The API smoke requires a new personal cloud archive')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(scope='TORCH_GRAPH_API_SMOKE_NO_MODEL_OR_E2E_GROUND_TRUTH',
        generation=config['generation'], status='UNVERIFIED', config=config,
        intervals=[], graph_identity=[], replay_kernels=[], G2='UNVERIFIED',
        replay_node_mapping='UNVERIFIED', started=time.monotonic())
    try:
        for name, digest in config['sources'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise ValueError('API smoke source differs from its frozen input: '+name)
        import torch
        from vllm.config import ProfilerConfig
        import role_capacity_observation as capacity
        import role_execution_observation as execution
        import role_profiler_observation as observation
        from inspect_runtime_sources import load

        environment = load(Path(config['task_root'])/'native_g1/run_g1_cases.py')
        report['environment'] = environment.record_environment(output)
        if report['environment']['device_name'] != config['device'] or torch.cuda.device_count() != 1:
            raise ValueError('API smoke must use the single requested GPU')
        workers = output/'workers'
        workers.mkdir()
        deadline = report['started']+config['seconds']
        node = output/'node.json'
        node.write_text(json.dumps(dict(observer_output=str(workers), record_bytes=config['record_bytes'],
            generation=config['generation'], node_rank=0, observer_sha256=execution.SOURCE['capacity_observer_sha256'],
            workload_deadline=deadline, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            pid_namespace=os.readlink('/proc/self/ns/pid'))))
        os.environ['STEP5_CAPACITY_NODE_FILE'] = str(node)
        state = execution.RequestObservation()
        worker = SimpleNamespace(rank=0, local_rank=0, _request_observation=state,
                                 model_runner=SimpleNamespace(), profiler=None)
        capacity.identity_record(worker)
        report['identity'] = worker._capacity_identity
        observed = observation.NativeProfilerObservation(worker,
            observation.ProfilerObservationLimits(**config['physical_observation']),
            workers, deadline, execution.write_record, state, {})
        state.profiler = observed
        observed.bind_graph_api()
        profiler = observed.bind_profiler(ProfilerConfig(**observation.native_config(workers)))
        x = torch.ones(config['elements'], device='cuda')
        y = torch.empty_like(x)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            y.copy_(x)
            y.mul_(config['multiplier'])
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()

        def snapshot(previous):
            try:
                current, row = observed.capture_nodes(previous)
                report['intervals'].append(row)
                return current
            except Exception:
                report.setdefault('api_errors', []).append(traceback.format_exc())
                return previous

        for window in observation.WINDOWS:
            observed.before_start(window)
            profiler.start(delay_iterations=0, max_iterations=0)
            observed.after_start()
            try:
                if window == 'capture':
                    with torch.cuda.graph(graph):
                        previous = snapshot(None)
                        y.copy_(x)
                        previous = snapshot(previous)
                        y.mul_(config['multiplier'])
                        previous = snapshot(previous)
                    previous = None
                else:
                    x.fill_(config['changed_input'])
                    with observed.annotate(observation.annotation_name('execute', 1, 1, 'replay')):
                        graph.replay()
                    torch.cuda.synchronize()
                    actual = y.cpu().tolist()
                    expected = config['changed_input']*config['multiplier']
                    report['content'] = dict(actual=actual, expected=expected)
                    if actual != [expected]*config['elements']:
                        raise ValueError('The original graph did not compute the changed input')
                try:
                    report['graph_identity'].append(observed.graph_identity(graph))
                except Exception:
                    report.setdefault('api_errors', []).append(traceback.format_exc())
                profiler.step()
            finally:
                profiler.stop()
                observed.after_stop()
        report['profiler'] = observed.summary()
        if observed.errors or report.get('api_errors'):
            raise ValueError('Graph API or native profiler evidence is unverified')
        if len(report['intervals']) != 3 or not all(row['nodes'] for row in report['intervals'][1:]):
            raise ValueError('Both actual captured operations must expose node additions')
        if len(report['graph_identity']) != 2 or report['graph_identity'][0] != report['graph_identity'][1]:
            raise ValueError('Capture and replay must retain the same executable identity')
        trace = report['profiler']['windows'][1]['trace']
        events = json.loads(Path(trace['path']).read_bytes())['traceEvents']
        report['replay_kernels'] = [row for row in events if row.get('cat') == 'kernel']
        report['status'] = 'PASS_API_ONLY_NODE_MAPPING_UNVERIFIED'
    except BaseException:
        report['status'] = 'FAIL'
        report['error'] = traceback.format_exc()
    finally:
        report['finished'] = time.monotonic()
        (output/'api_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        size = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
        if size > config['archive_bytes']:
            raise ValueError('API smoke artifacts exceed the frozen archive allowance')
        shutil.copytree(output, archive)
        print(json.dumps(dict(status=report['status'], generation=config['generation'],
                              archive=str(archive), bytes=size, G2='UNVERIFIED')), flush=True)
    return int(report['status'] == 'FAIL')


if __name__ == '__main__':
    raise SystemExit(main())
