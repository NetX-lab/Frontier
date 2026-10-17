#!/usr/bin/env python3
"""Run one unchanged native TP8 full-role capacity attempt, not G2 acceptance."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

WORKTREE = Path(__file__).resolve().parents[3]
TASK = WORKTREE/'task_memory/task_2026-10-15_step5_preview_h800'
CLOUD = Path('/mnt/codesign-exp/ycfeng/step5-preview/role_capacity_20261017_01')
SECONDS = 2400
SOURCE_MANIFESTS = {
    'native_role_sources/observed': 'cce94cb50e4354a38a47168bbf1bd14762c11aa1bed31d90e2dd62d956eb6251',
    'native_role_sources/observed_generation2': 'ea6cfa0a89d78b1fcdbf71c4996c67197c7021439e7f4cdb76bdcbd1295904bb',
    'native_source_export/observed': '4bcd69c5838bc9a52576e0fd518b47852d97181627127b4ea5fefa41903fdc46',
    'native_runtime_sources/observed': 'a4c4722e16f0da3d4968f06b8ffde9cc978e834fc28b865da329de9487b82308',
    'native_precision_sources/observed': '651f041ffff31f6d0b120edf35f31f600872f9d94beb2b9bc74812711eb2334f',
}


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def validate(task, output):
    if task != TASK or output == TASK or not output.is_relative_to(TASK) or output.exists():
        raise ValueError('Exact trusted task and exclusive new output child required')


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, default=str); stream.write('\n')


def diagnostic(write):
    try:
        write()
    except Exception:
        try:
            print('CAPACITY_EVIDENCE_UNVERIFIED '+traceback.format_exc(), flush=True)
        except Exception:
            pass


def execute(task, output, launcher):
    from vllm import LLM
    package = importlib.metadata.distribution('vllm')
    assert package.version == launcher.EXPECTED_VLLM_VERSION
    for relative, digest in SOURCE_MANIFESTS.items():
        path = task/relative/'source_manifest.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative
        manifest = json.loads(path.read_text())
        for row in manifest['files']:
            name = row['relative_path']; path = Path(package.locate_file(name))
            expected = row.get('sha256', row.get('source_sha256'))
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    observer = Path(__file__).with_name('role_capacity_observation.py')
    os.environ['STEP5_CAPACITY_TASK'] = str(task)
    os.environ['STEP5_CAPACITY_OUTPUT'] = str(CLOUD)
    os.environ['STEP5_CAPACITY_OBSERVER_SHA'] = hashlib.sha256(observer.read_bytes()).hexdigest()
    kwargs = dict(model=str(task/'reference_inputs'), load_format='dummy', skip_tokenizer_init=True,
        tensor_parallel_size=8, data_parallel_size=1, pipeline_parallel_size=1,
        max_model_len=8192, max_num_batched_tokens=512, max_num_seqs=1,
        enable_chunked_prefill=True, enable_prefix_caching=False, speculative_config=None,
        worker_cls='role_capacity_observation.CapacityWorker')
    save(output/'invocation.json', dict(kwargs=kwargs, absent_overrides=['dtype', 'quantization',
        'enable_expert_parallel', 'gpu_memory_utilization', 'enforce_eager', 'limit_mm_per_prompt',
        'compilation_config', 'attention_backend'], observer_sha256=os.environ['STEP5_CAPACITY_OBSERVER_SHA']))
    started = time.perf_counter()
    try:
        llm = LLM(**kwargs)
        diagnostic(lambda: save(output/'initialization.json', dict(status='native_initialization_returned_not_G2',
            wall_seconds=time.perf_counter()-started)))
        # No request or replay claim: first checkpoint answers native initialization capacity only.
        diagnostic(lambda: save(output/'completion.json', dict(status='capacity_attempt_completed_not_G2',
             runtime_class=type(llm).__module__+'.'+type(llm).__name__)))
    except BaseException:
        failure = dict(status='BLOCKED_ON_RUNTIME', wall_seconds=time.perf_counter()-started,
                       traceback=traceback.format_exc())
        diagnostic(lambda: save(output/'failure.json', failure))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('submit', 'run', 'child'))
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); task, output = args.task_root.resolve(), args.output_dir.resolve()
    validate(task, output)
    if args.mode != 'submit':
        assert os.path.ismount('/mnt/codesign-exp'), 'Required actual cloud mount missing'
    launcher = load(task/'native_source_export/submit_export.py')
    launcher.inspection.locked_inputs()
    argv = [sys.executable, '-B', str(Path(__file__).resolve()), 'run',
            '--task-root', str(task), '--output-dir', str(output)]
    if args.mode == 'submit':
        assert os.environ['STEPMIND_BACKEND'] == 'rjob'
        from steptron.exp.base_exp import ResourceConfig
        from steptron.utils.stepmind import spawn_tasks
        resources = launcher.resource_config_arguments()
        resources.update(cpu=32, gpu=8, mem_gb=128, replica=1)
        argv[0] = 'python3'
        command = ('export VLLM_ENABLE_CUDA_COMPATIBILITY=1 '
            'LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH '
            'PYTHONPATH='+shlex.quote(str(Path(__file__).parent)+':'+str(WORKTREE))+':$PYTHONPATH; exec '+shlex.join(argv))
        os.chdir(WORKTREE); os.environ['EXP_ID'] = 'step5-native-role-capacity'
        workers = spawn_tasks(ResourceConfig(**resources),
            **launcher.spawn_arguments('timeout '+str(SECONDS)+' bash -c '+shlex.quote(command)))
        print('JOB_ID', workers.rjob.meta.name, 'FINAL_STATUS', workers.poll(), flush=True)
        return
    if args.mode == 'child':
        output.mkdir(parents=True, exist_ok=False)
        execute(task, output, launcher)
        return
    assert not CLOUD.exists(), 'Cloud generation must be new'
    CLOUD.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    resources = load(Path(__file__).with_name('check_precision_resources.py'))
    argv[3] = 'child'; argv[-1] = str(output/'native')
    with (output/'native.stdout').open('xb') as out, (output/'native.stderr').open('xb') as err:
        child = subprocess.Popen(argv, stdout=out, stderr=err)
        def append(row):
            with (CLOUD/'parent_samples.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
        while True:
            row = dict(time_ns=time.time_ns(), child_pid=child.pid)
            try:
                row['processes'] = resources.process_tree(os.getpid())
                row['cgroup'] = resources.cgroup_memory()
                gpu = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.total',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
                row['physical_gpu'] = dict(exit_code=gpu.returncode, stdout=gpu.stdout, stderr=gpu.stderr)
            except Exception:
                row['observer_error'] = traceback.format_exc()
            diagnostic(lambda: append(row))
            status = child.poll()
            if status is not None: break
            time.sleep(1)
    diagnostic(lambda: save(output/'status.json', dict(native_exit=status, scope='capacity_only_not_G2', cloud=str(CLOUD))))
    raise SystemExit(status)


if __name__ == '__main__':
    main()
