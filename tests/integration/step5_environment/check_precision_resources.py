#!/usr/bin/env python3
"""Measure a new unchanged native precision workload; retain original evidence."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

WORKTREE = Path(__file__).resolve().parents[3]
TASK = WORKTREE/'task_memory/task_2026-10-15_step5_preview_h800'
CLOUD = Path('/mnt/codesign-exp/ycfeng/step5-preview/g1_precision_resources_20261017_01')
PRECISION_SHA = '7cb9acf7b030fd61dd545abe9dc989c1934b148684c5f8775f7625c6c9cabeb7'
SAMPLE_SECONDS = 1.0


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def rss(pid):
    text = Path(f'/proc/{pid}/status').read_text()
    return {key: int(line.split()[1]) for line in text.splitlines()
            for key in ('VmRSS', 'VmHWM') if line.startswith(key+':')}


def process_tree(pid):
    pending, rows = [pid], []
    while pending:
        current = pending.pop()
        try:
            rows.append(dict(pid=current, rss_kib=rss(current)))
            children = set()
            for thread in Path(f'/proc/{current}/task').iterdir():
                children.update(int(child) for child in (thread/'children').read_text().split())
            pending.extend(children)
        except (FileNotFoundError, ProcessLookupError):
            # Compiler subprocesses can exit while their parent remains alive.
            continue
    return rows


def cgroup_memory():
    return read_cgroup_memory(Path('/proc/self/cgroup').read_text(), Path('/proc/self/mountinfo').read_text())


def read_cgroup_memory(cgroups, mounts):
    paths = []
    for group in cgroups.splitlines():
        _, controllers, relative = group.split(':', 2)
        for line in mounts.splitlines():
            before, after = line.split(' - ', 1)
            fields, filesystem = before.split(), after.split()
            matched = (not controllers and filesystem[0] == 'cgroup2') or (
                'memory' in controllers.split(',') and filesystem[0] == 'cgroup'
                and 'memory' in filesystem[2].split(','))
            if not matched:
                continue
            root, mount = [Path(value.replace('\\040', ' ').replace('\\134', '\\')) for value in fields[3:5]]
            # The process namespace can expose either a hierarchy path or '/'.
            suffix = Path(relative).relative_to(root) if Path(relative).is_relative_to(root) else Path(relative.lstrip('/'))
            current = mount/suffix
            assert current.is_relative_to(mount)
            paths.extend([current, *[parent for parent in current.parents if parent.is_relative_to(mount)]])
    values = {str(root/name): (root/name).read_text().strip() for root in set(paths)
              for name in ('memory.current', 'memory.peak', 'memory.max', 'memory.events',
                           'memory.usage_in_bytes', 'memory.max_usage_in_bytes', 'memory.limit_in_bytes', 'memory.failcnt')
              if (root/name).is_file()}
    assert values, 'No actual cgroup memory files resolved from mountinfo'
    return values


def record_boundary(record, phase, boundary):
    try:
        record(phase, boundary)
    except Exception:
        print(json.dumps(dict(phase=phase, boundary=boundary, resource_evidence_verified=False,
                              observer_error=traceback.format_exc())), file=sys.stderr, flush=True)


def snapshot(torch):
    cuda = dict(cuda_allocated_bytes=None, cuda_reserved_bytes=None,
                cuda_lifetime_peak_allocated_bytes=None, cuda_lifetime_peak_reserved_bytes=None)
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
        cuda = dict(cuda_allocated_bytes=torch.cuda.memory_allocated(), cuda_reserved_bytes=torch.cuda.memory_reserved(),
                    cuda_lifetime_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cuda_lifetime_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return dict(time_ns=time.time_ns(), pid=os.getpid(), host_current_and_peak_rss_kib=rss(os.getpid()),
                host_lifetime_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **cuda)


def observe(owner, name, label, record):
    original = getattr(owner, name)
    def call(*args, **kwargs):
        phase = label(*args, **kwargs) if callable(label) else label
        record_boundary(record, phase, 'before')
        try:
            return original(*args, **kwargs)
        finally:
            record_boundary(record, phase, 'after')
    setattr(owner, name, call)
    return original


def execute(precision, task, output, collector, case):
    import torch
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
    phases, reset_epoch, errors = [], 0, []
    def record(phase, boundary):
        row = dict(phase=phase, boundary=boundary, reset_epoch=reset_epoch)
        try:
            row.update(snapshot(torch), verified=True)
        except Exception:
            row.update(verified=False, error=traceback.format_exc())
            errors.append(row)
        phases.append(row)
        try:
            collector.dump_json(output/'resource_phases.json', dict(phases=phases, errors=errors))
        except Exception:
            errors.append(dict(phase=phase, error=traceback.format_exc()))
            raise
    reset = torch.cuda.reset_peak_memory_stats
    def reset_observed(*args, **kwargs):
        nonlocal reset_epoch
        record_boundary(record, 'native_peak_reset', 'before')
        result = reset(*args, **kwargs)
        reset_epoch += 1
        record_boundary(record, 'native_peak_reset', 'after')
        return result
    torch.cuda.reset_peak_memory_stats = reset_observed
    record_boundary(record, 'configured_entry', 'before')
    construct = precision.construct
    def construct_observed(*args, **kwargs):
        record_boundary(record, 'constructed', 'before')
        try:
            module = construct(*args, **kwargs)
        finally:
            record_boundary(record, 'constructed', 'after')
        module.register_forward_pre_hook(lambda m, a: record_boundary(record, 'forward', 'before'))
        module.register_forward_hook(lambda m, a, y: record_boundary(record, 'forward', 'after'))
        return module
    precision.construct = construct_observed
    # These functions are imported inside the original execute function.
    from vllm.model_executor.model_loader.reload import layerwise
    from vllm.model_executor.model_loader import utils
    observe(layerwise, 'record_metadata_for_reloading', 'reload_metadata', record)
    observe(DummyModelLoader, 'load_weights', 'dummy_loaded', record)
    observe(utils, 'process_weights_after_loading', 'postprocessed', record)
    observe(precision, 'retain', lambda t, tensors, directory, c: 'retained_'+directory.name, record)
    try:
        precision.execute(task, output, collector, case)
    finally:
        record_boundary(record, 'module_execution', 'after')
        def summary(phase, boundary):
            collector.dump_json(output/'resources.json', dict(phases=phases, errors=errors, cgroup=cgroup_memory(),
                original_precision_sha256=PRECISION_SHA, instrumentation_sha256=collector.sha256_file(Path(__file__)),
                peak_scope='Cumulative peaks within each native-reset epoch, not independent per-phase peaks',
                gpu_capacity_bytes=torch.cuda.get_device_properties(0).total_memory))
        record_boundary(summary, 'resource_summary', 'after')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['submit', 'run'])
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--case')
    args = parser.parse_args(); task, output = args.task_root.resolve(), args.output_dir.resolve()
    assert task == TASK and output.is_relative_to(TASK) and output != TASK and not output.exists()
    if args.mode == 'run':
        assert os.path.ismount('/mnt/codesign-exp'), 'Required cloud volume is not mounted'
    precision = load(Path(__file__).with_name('check_precision_ffn.py'))
    collector = load(task/'native_g1/run_g1_cases.py')
    assert collector.sha256_file(Path(precision.__file__)) == PRECISION_SHA
    assert args.case is None or args.case in precision.CASES
    precision.CLOUD_ROOT = CLOUD
    precision.__file__ = __file__
    if args.mode == 'submit':
        precision.main(); return
    original_execute = precision.execute
    def child_execute(*arguments):
        precision.execute = original_execute
        execute(precision, *arguments)
    precision.execute = child_execute
    native_run = subprocess.run
    started = time.monotonic()
    samples, child_statuses = [], []
    def sample(child=None):
        row = dict(time_ns=time.time_ns(), parent_pid=os.getpid(), parent_rss_kib=rss(os.getpid()), cgroup=cgroup_memory())
        if child is not None and child.poll() is None:
            row['child_pid'] = child.pid; row['processes'] = process_tree(child.pid)
            row['sampled_process_tree_rss_kib'] = row['parent_rss_kib'].get('VmRSS', 0) + sum(
                process['rss_kib'].get('VmRSS', 0) for process in row['processes'])
        try:
            gpu = native_run(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10)
            row['physical_gpu'] = dict(exit_code=gpu.returncode, stdout=gpu.stdout, stderr=gpu.stderr)
        except subprocess.TimeoutExpired as error:
            row['physical_gpu'] = dict(error=str(error), verified=False)
        samples.append(row)
        try:
            collector.dump_json(output/'parent_samples.json', dict(samples=samples))
        except Exception:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
    def sample_observed(child=None):
        record_boundary(lambda p, b: sample(child), 'parent_sample', 'periodic')
    def child_run(arguments, *a, **kw):
        if not (len(arguments) > 2 and arguments[2] == str(Path(__file__).resolve())):
            return native_run(arguments, *a, **kw)
        assert not a and not kw
        sample_observed()
        child_started = time.monotonic()
        child = subprocess.Popen(arguments)
        while child.poll() is None:
            sample_observed(child); time.sleep(SAMPLE_SECONDS)
        sample_observed()
        child_statuses.append(dict(case=arguments[-1], pid=child.pid, exit_code=child.returncode,
                                  wall_seconds=time.monotonic()-child_started))
        record_boundary(lambda p, b: collector.dump_json(output/'resource_case_statuses.json', dict(cases=child_statuses)),
                        'child_status', 'after')
        return subprocess.CompletedProcess(arguments, child.returncode)
    if args.case is None:
        subprocess.run = child_run
    try:
        precision.main()
    except Exception:
        if output.is_dir():
            error = traceback.format_exc()
            record_boundary(lambda p, b: collector.dump_json(output/'resource_failure.json', dict(error=error)),
                            'native_parent_failure', 'after')
        raise
    finally:
        if args.case is None and output.is_dir():
            subprocess.run = native_run
            sample_observed()
            def summary(phase, boundary):
                import torch
                collector.dump_json(output/'parent_resources.json', dict(samples=samples, cgroup=cgroup_memory(),
                    parent_cuda_and_host=snapshot(torch),
                    parent_lifetime_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    sample_interval_seconds=SAMPLE_SECONDS, sampling_limit='Periodic observations are not continuous physical peaks',
                    wall_seconds=time.monotonic()-started, original_precision_sha256=PRECISION_SHA,
                    instrumentation_sha256=collector.sha256_file(Path(__file__)), child_statuses=child_statuses,
                    generation='new_resource_measurement_not_historical_04e_reconstruction'))
            record_boundary(summary, 'parent_resources', 'after')


if __name__ == '__main__':
    main()
