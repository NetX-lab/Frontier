#!/usr/bin/env python3
"""Collect local eager SWA/DSA observations through the unchanged G1 runtime.

The controller imports no Torch, vLLM or platform client. Submission is an
operator entry point; importing this file or preparing its payload is inert.
Native return codes and auxiliary evidence failures have separate records.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

from check_role_capacity import load, save, diagnostic, TASK, WORKTREE
from check_precision_ffn import SOURCE_BATCHES

KINDS = ('swa', 'dsa')
WARMUPS, REPEATS, SECONDS = 3, 5, 2400
PERSONAL_CLOUD = Path('/mnt/codesign-exp/ycfeng')
SCRIPT = Path(__file__).resolve()
FROZEN_FILES = {
    'tests/integration/step5_environment/check_role_capacity.py': '48662f5dacae7356c8c470e2d7aa6ae0b38f4c6c5b3e6cc3375b16a97f65eb3d',
    'tests/integration/step5_environment/check_precision_ffn.py': '7cb9acf7b030fd61dd545abe9dc989c1934b148684c5f8775f7625c6c9cabeb7',
    'tests/integration/step5_environment/check_precision_resources.py': 'e6a282ddf02b5dac724fbf9356a51d24d7b4f175fc90907d9188e4948667a987',
    'tests/integration/step5_environment/local_attention_measurement.py': 'f35dbb431557222abb7894bd41371a7610676bab24ba390b5578c122eedfdcd9',
    'tests/integration/step5_environment/local_ffn_measurement.py': '6658c5ef7d48781bd1993eaf1f2d14478773a25385682f3502fc815ca9d19307',
    'tests/integration/step5_environment/local_scope_manifest.py': 'f5447e4f38e4ee5bf8f1489fc48cedb0bf636384447ad253e94cc7f8fa9b783b',
    'tests/integration/step5_environment/compare_precision_ffn.py': '163fde3bb19482a65e749552a9cbed56922ffbfacf7b81de8920642064acd7c8',
    'frontier/profiling/utils/record_function_tracer.py': 'c176903a390a102bb15b1e6495a6a988abb19b8cb39a5403e2ee0af9a555da51',
    'task_memory/task_2026-10-15_step5_preview_h800/native_g1/run_g1_cases.py': '5c56eeb1da369a8a707973690a2a1d3a16fcb1bf9dbb9534f205e3d52483345b',
    'task_memory/task_2026-10-15_step5_preview_h800/native_source_export/submit_export.py': '6fea41320bb9ecfcd3a49b81b5f3941bf3cc88f4ebb8dbc9e80dc85ca0779ece',
    'task_memory/task_2026-10-15_step5_preview_h800/native_reference_inspection/submit_inspection.py': 'a4434a5e01aa4780a8924c93a968e319098f81f65b3e0ae7ca70d015c41ab01a',
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen(driver_sha256):
    if sha256(SCRIPT) != driver_sha256:
        raise ValueError('Attention driver differs from the explicitly frozen launch input')
    for relative, expected in FROZEN_FILES.items():
        if sha256(WORKTREE/relative) != expected:
            raise ValueError('Changed attention dependency: '+relative)
    # Check actual imported local modules as well as the intended filesystem paths.
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path is None:
            continue
        path = Path(path).resolve()
        for relative, expected in FROZEN_FILES.items():
            target = WORKTREE/relative
            if module.__name__ == target.stem or path == target:
                if path != target or sha256(path) != expected:
                    raise ValueError('Loaded attention dependency drift: '+module.__name__)


def validate_paths(args):
    task, output, cloud, scratch = (getattr(args, name).resolve() for name in
                                  ('task_root', 'output_dir', 'cloud_root', 'worker_tmp_root'))
    if task != TASK or output == task or not output.is_relative_to(task) or output.exists():
        raise ValueError('Exact trusted task and exclusive new output child required')
    if cloud == PERSONAL_CLOUD or not cloud.is_relative_to(PERSONAL_CLOUD):
        raise ValueError('Cloud generation must be within /mnt/codesign-exp/ycfeng')
    if scratch.is_relative_to(Path('/mnt/codesign-exp')) or scratch.is_relative_to(WORKTREE):
        raise ValueError('Temporary observations and caches require worker-local storage')
    if scratch == Path('/') or any(a.is_relative_to(b) or b.is_relative_to(a)
                                  for a, b in ((scratch, task), (scratch, cloud))):
        raise ValueError('Worker scratch must not overlap task or cloud storage')
    if args.tensor_copy_bytes <= 0:
        raise ValueError('A positive tensor-copy byte budget is required')
    if type(args.evidence_bytes) is not int or args.evidence_bytes <= 0:
        raise ValueError('A positive integer generation evidence byte cap is required')
    if args.mode == 'child':
        if args.kind not in KINDS or not cloud.is_dir() or not scratch.is_dir():
            raise ValueError('Child requires an actual kind and parent-owned generation directories')
        if (cloud/args.kind).exists() or (scratch/'observations'/args.kind).exists():
            raise ValueError('Attention kind output must be exclusive and new')
    elif args.kind is not None or cloud.exists() or scratch.exists():
        raise ValueError('Parent requires exclusive new generations and no child kind')
    return task, output, cloud, scratch


def prepare(task, driver_sha256):
    verify_frozen(driver_sha256)
    launcher = load(task/'native_source_export/submit_export.py')
    launcher.inspection.locked_inputs()
    collector = load(task/'native_g1/run_g1_cases.py')
    _, manifest, _ = collector.verify_inputs(task/'g1_preparation.json')
    files = list(manifest['files'])
    for batch, expected in SOURCE_BATCHES.items():
        path = task/batch/'observed/source_manifest.json'
        if collector.sha256_file(path) != expected:
            raise ValueError('Native source manifest drift: '+batch)
        files += [dict(relative_path=row['relative_path'], source_sha256=row['sha256'],
                       source_size_bytes=row['size_bytes'])
                  for row in json.loads(path.read_text())['files'] if row['status'] == 'copied']
    unique = {}
    for row in files:
        relative = row['relative_path']
        if relative in unique and unique[relative] != row:
            raise ValueError('Conflicting native source identities: '+relative)
        unique[relative] = row
    collector.IMPORTED_FILES = tuple(unique)
    import local_attention_measurement as attention
    verify_frozen(driver_sha256)
    plan = json.loads((task/'local_scope_sampling_01.json').read_text())
    attention.admit(plan, task)
    selected = attention.samples(plan)
    if len(selected) != 30 or sum(row['partition'] == 'holdout' for row in selected) != 3:
        raise ValueError('The frozen eager selection must preserve 30 shapes including three heldouts')
    return launcher, collector, attention, list(unique.values()), plan


def invocation_argv(mode, task, output, cloud, scratch, args, kind=None):
    argv = [str(SCRIPT), mode, '--task-root', str(task), '--output-dir', str(output),
            '--cloud-root', str(cloud), '--worker-tmp-root', str(scratch),
            '--driver-sha256', args.driver_sha256, '--tensor-copy-bytes', str(args.tensor_copy_bytes),
            '--evidence-bytes', str(args.evidence_bytes)]
    if kind is not None:
        argv += ['--kind', kind]
    return argv


def submit_payload(launcher, task, output, cloud, scratch, args):
    argv = ['python3', '-B', *invocation_argv('run', task, output, cloud, scratch, args)]
    command = ('export VLLM_ENABLE_CUDA_COMPATIBILITY=1 '
               'LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH '
               'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '
               'PYTHONPYCACHEPREFIX='+shlex.quote(str(scratch/'cache'/'pycache'))+'; exec '+shlex.join(argv))
    return launcher.resource_config_arguments(), launcher.spawn_arguments(
        'timeout '+str(SECONDS)+' bash -c '+shlex.quote(command))


def evidence(label, write, errors):
    def attempt():
        try:
            write()
        except Exception:
            errors.append(dict(evidence=label, error=traceback.format_exc()))
            raise
    diagnostic(attempt)


def release_closed_stdout():
    """Discard log bytes that a closed pipe can no longer deliver.

    CPython's failed exit flush would replace the known exit with 120. This runs after
    the final status write, so no later print can refill the buffer.
    """
    try:
        sys.stdout.flush()
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        os.close(devnull)


def nic_counters():
    return {str(path): int(path.read_text()) for path in
            sorted(Path('/sys/class/infiniband').glob('*/ports/*/counters/port_*_data'))}


def record_environment(collector, output, errors):
    """Keep the collector's CUDA checks while isolating its diagnostic write."""
    original = collector.dump_json
    def write(path, payload):
        evidence('environment_write', lambda: original(path, payload), errors)
    collector.dump_json = write
    try:
        return collector.record_environment(output)
    finally:
        collector.dump_json = original


def loaded_sources(files, collector):
    """Verify measured-path loaded Python files, not just intended installation paths."""
    root = Path(importlib.util.find_spec('vllm').origin).resolve().parent.parent
    rows = []
    for row in files:
        name = row['relative_path'][:-3].replace('/', '.')
        if name.endswith('.__init__'):
            name = name[:-len('.__init__')]
        module = sys.modules.get(name)
        if module is None:
            continue
        path = Path(module.__file__).resolve()
        actual = dict(module=name, path=str(path), sha256=collector.sha256_file(path),
                      size_bytes=path.stat().st_size, verification='matched_frozen_export')
        if (path != root/row['relative_path'] or actual['sha256'] != row['source_sha256']
                or actual['size_bytes'] != row['source_size_bytes']):
            raise ValueError('Loaded native source differs from the frozen source: '+name)
        rows.append(actual)
    return rows


def loaded_extensions(collector):
    """Record actually loaded package/extension bytes; no compiled-build identity claim."""
    rows = []
    for name, module in sorted(tuple(sys.modules.items())):
        path = getattr(module, '__file__', None)
        if name.split('.')[0] not in ('torch', 'triton', 'vllm', 'flashinfer', 'flash_attn') or path is None:
            continue
        if name.split('.')[0] != 'vllm' and name not in ('torch', 'triton', 'flashinfer', 'flash_attn') and not path.endswith('.so'):
            continue
        path = Path(path).resolve()
        rows.append(dict(module=name, path=str(path), size_bytes=path.stat().st_size,
                         sha256=collector.sha256_file(path)))
    mapped = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                     if line.split() and line.split()[-1].startswith('/')
                     and Path(line.split()[-1]).name.startswith(('libcuda.', 'libcudart.', 'libnvrtc.', 'libnccl.'))})
    libraries = [dict(path=str(Path(path).resolve()), size_bytes=Path(path).stat().st_size,
                      sha256=collector.sha256_file(Path(path))) for path in mapped]
    return dict(modules=rows, mapped_gpu_libraries=libraries,
                scope='loaded_package_and_extension_files_not_compiled_build_or_numeric_oracle')


def execute(task, output, scratch, kind, args, collector, attention, files, plan):
    output.mkdir(parents=True, exist_ok=False)
    errors, state = [], dict(measurement_returned=False, workload_error=None, trust_error=None)
    stage = 'installed_source_verification'
    try:
        collector.verify_installed(dict(files=files))
        stage = 'runtime_imports'
        import torch
        from vllm.config import set_current_vllm_config
        from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
        from vllm.engine.arg_utils import EngineArgs
        from vllm.model_executor.model_loader.utils import configure_quant_config
        from vllm.models.step5.nvidia.model import Step5ForCausalLM
        from vllm.v1.worker.workspace import init_workspace_manager

        stage = 'runtime_identity'
        environment = record_environment(collector, output, errors)
        expected = plan['identity']
        for field, key in (('device_name', 'hardware'), ('torch', 'torch'), ('triton', 'triton'),
                           ('vllm', 'vllm'), ('python', 'native_python'), ('kernel_module_label', 'kernel_driver')):
            if environment[field] != expected[key]:
                raise ValueError('Runtime identity drift: '+field)
        if environment['mapped_libcuda'] != [expected['user_driver']] or torch.cuda.device_count() != 1:
            raise ValueError('Loaded CUDA driver or one-device allocation drift')
        verify_frozen(args.driver_sha256)
        identities = loaded_sources(files, collector)
        evidence('loaded_sources_before', lambda: save(output/'loaded_sources_before.json',
                 dict(files=identities)), errors)
        evidence('device', lambda: save(output/'device.json', dict(
            uuid=str(torch.cuda.get_device_properties(0).uuid),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))), errors)
        stage = 'native_configuration'
        config = EngineArgs(model=str(task/'reference_inputs'), load_format='dummy', skip_tokenizer_init=True,
            enable_prefix_caching=False, enable_chunked_prefill=True, max_num_seqs=4,
            max_num_batched_tokens=512, max_model_len=8192).create_engine_config()
        native = config.with_hf_config(config.model_config.hf_text_config)
        configure_quant_config(native.quant_config, Step5ForCausalLM)
        text = native.model_config.hf_config
        assert (text.num_hidden_layers, len(text.layer_types_with_mtp), text.moe_num_experts) == (92, 95, 352)
        assert len(native.quant_config.ignored_layers) == 1406
        torch.set_default_dtype(native.model_config.dtype); torch.cuda.set_device(0)
        with set_current_vllm_config(native), torch.device('cuda'), torch.no_grad():
            init_distributed_environment(world_size=1, rank=0, local_rank=0,
                distributed_init_method='file://'+str(scratch/('distributed_'+kind)), backend='nccl')
            initialize_model_parallel(1, 1); init_workspace_manager(torch.device('cuda'))
            evidence('invocation', lambda: save(output/'invocation.json', dict(kind=kind,
                warmups=WARMUPS, repeats=REPEATS, tensor_copy_bytes=args.tensor_copy_bytes,
                evidence_bytes=args.evidence_bytes,
                driver_sha256=args.driver_sha256, consumer_sha256=sha256(Path(attention.__file__)),
                plan_sha256=collector.sha256_file(task/'local_scope_sampling_01.json'),
                dtype=str(native.model_config.dtype), quantization=type(native.quant_config).__name__,
                compilation_cudagraph_mode=str(native.compilation_config.cudagraph_mode),
                mode='native_rank_local_eager_attention_not_full_role')), errors)
            counters = {}
            evidence('nic_before', lambda: counters.update(before=nic_counters()), errors)
            stage = 'native_measurement'
            rows = attention.measure(task, scratch/'observations'/kind, native, kind,
                warmups=WARMUPS, repeats=REPEATS, tensor_copy_bytes=args.tensor_copy_bytes)
            state['measurement_returned'] = True
            stage = 'source_verification_after'
            # Source identity is a trust prerequisite, not an auxiliary validator verdict.
            verify_frozen(args.driver_sha256)
            identities = loaded_sources(files, collector)
            evidence('loaded_sources_after', lambda: save(output/'loaded_sources_after.json',
                     dict(files=identities)), errors)
            evidence('nic_after', lambda: counters.update(after=nic_counters()), errors)
            evidence('measurements', lambda: save(output/'measurements.json', dict(rows=rows, nic=counters,
                expected_observations=30, status='raw_unadmitted',
                nic_scope='RDMA port counters in 4-byte words; node-wide samples do not establish continuous exclusivity',
                artifact_path_scope='Raw paths name worker-local files; parent archive manifest maps retained paths')), errors)
    except BaseException:
        field = 'trust_error' if stage in ('installed_source_verification', 'runtime_imports',
            'runtime_identity', 'source_verification_after') else 'workload_error'
        state[field] = dict(stage=stage, traceback=traceback.format_exc())
        evidence('failure', lambda: save(output/'failure.json', state), errors)
        raise
    finally:
        if stage not in ('installed_source_verification', 'runtime_imports'):
            if state['workload_error'] is not None or state['trust_error'] is not None:
                evidence('loaded_sources_failure', lambda: save(output/'loaded_sources_failure.json',
                         dict(files=loaded_sources(files, collector))), errors)
            evidence('loaded_extensions', lambda: save(output/'loaded_extensions.json', loaded_extensions(collector)), errors)
        evidence('child_status', lambda: save(output/'native_status.json', dict(state, evidence_errors=errors,
                 status='raw_unadmitted', kind=kind, evidence_bytes=args.evidence_bytes)), errors)


def evidence_files(root, excluded=None):
    """List raw files without following symlinks or entering the cache tree."""
    files = []
    if root.is_symlink():
        raise ValueError('Symlink evidence root cannot be archived: '+str(root))
    def unreadable(error):
        raise error
    for directory, subdirs, names in os.walk(root, followlinks=False, onerror=unreadable):
        directory = Path(directory)
        for name in tuple(subdirs):
            path = directory/name
            if path == excluded:
                subdirs.remove(name)
            elif path.is_symlink():
                raise ValueError('Symlink evidence cannot be archived: '+str(path))
        for name in names:
            path = directory/name
            if path.is_symlink():
                raise ValueError('Symlink evidence cannot be archived: '+str(path))
            if path.is_file():
                files.append(path)
    return sorted(files)


def archive(source, destination, collector):
    """Copy completed or failed partial artifacts without rewriting raw collector bytes."""
    files = evidence_files(source)
    shutil.copytree(source, destination)
    return [dict(worker_path=str(path), cloud_path=str(destination/path.relative_to(source)),
                 sha256=collector.sha256_file(destination/path.relative_to(source)), size_bytes=path.stat().st_size)
            for path in files]


def run_parent(task, output, cloud, scratch, args, collector):
    cloud.mkdir(parents=True, exist_ok=False)
    scratch.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    (scratch/'logs').mkdir(); (scratch/'observations').mkdir(); (scratch/'cache').mkdir()
    resources = load(SCRIPT.with_name('check_precision_resources.py'))
    statuses, errors, archives = [], [], []
    startup_error = None
    child_env = os.environ.copy()
    child_env.update(TMPDIR=str(scratch), XDG_CACHE_HOME=str(scratch/'cache'),
        HF_HOME=str(scratch/'cache'/'huggingface'), TRITON_CACHE_DIR=str(scratch/'cache'/'triton'),
        TORCHINDUCTOR_CACHE_DIR=str(scratch/'cache'/'torchinductor'),
        VLLM_CACHE_ROOT=str(scratch/'cache'/'vllm'), PYTHONPYCACHEPREFIX=str(scratch/'cache'/'pycache'),
        PYTHONDONTWRITEBYTECODE='1')

    def generation_bytes():
        files = evidence_files(output)+evidence_files(scratch, scratch/'cache')
        return sum(path.stat().st_size for path in files)

    def require_bytes(usage):
        if usage > args.evidence_bytes:
            raise ValueError(f'Generation evidence byte cap exceeded: {usage} > {args.evidence_bytes}')
        return usage

    def copy_evidence(source, destination):
        require_bytes(generation_bytes())
        archived = sum(path.stat().st_size for path in evidence_files(cloud))
        size = sum(path.stat().st_size for path in evidence_files(source))
        require_bytes(archived+size)
        archives.extend(archive(source, destination, collector))

    stage = 'child_logs'
    try:
        for kind in KINDS:
            argv = [sys.executable, '-B', *invocation_argv('child', task, output/kind, cloud, scratch, args, kind)]
            stage = 'child_logs'
            with (scratch/'logs'/(kind+'.stdout')).open('xb') as out, (scratch/'logs'/(kind+'.stderr')).open('xb') as err:
                stage = 'child_spawn'
                child = subprocess.Popen(argv, stdout=out, stderr=err, env=child_env)
                stage = 'child_wait'
                while True:
                    row = dict(time_ns=time.time_ns(), parent_pid=os.getpid(), child_pid=child.pid, kind=kind)
                    def observe():
                        row['processes'] = resources.process_tree(os.getpid())
                        row['cgroup'] = resources.cgroup_memory()
                        row['nic'] = nic_counters()
                        for key, query in (('physical_gpu', '--query-gpu=name,uuid,memory.used,memory.total'),
                                           ('compute_apps', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory')):
                            gpu = subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                                                 capture_output=True, text=True, timeout=10)
                            row[key] = dict(exit_code=gpu.returncode, stdout=gpu.stdout, stderr=gpu.stderr)
                            if gpu.returncode:
                                errors.append(dict(evidence=key, kind=kind, **row[key]))
                    evidence('parent_observation', observe, errors)
                    def append():
                        with (scratch/'logs'/'parent_samples.jsonl').open('a') as stream:
                            stream.write(json.dumps(row)+'\n')
                    evidence('parent_sample_write', append, errors)
                    status = child.poll()
                    if status is not None:
                        break
                    time.sleep(resources.SAMPLE_SECONDS)
            statuses.append(dict(kind=kind, pid=child.pid, native_exit=status,
                                 scope='raw_child_exit_not_evidence_acceptance'))
            evidence(kind+'_exit_print', lambda: print('ATTENTION_CHILD_EXIT '+json.dumps(statuses[-1]), flush=True), errors)
            evidence(kind+'_status', lambda: save(output/(kind+'_status.json'), statuses[-1]), errors)
            evidence(kind+'_evidence_bytes', lambda: require_bytes(generation_bytes()), errors)
            if (scratch/'observations'/kind).is_dir():
                evidence(kind+'_archive', lambda: copy_evidence(scratch/'observations'/kind, cloud/kind), errors)
    except OSError:
        if stage in ('child_logs', 'child_spawn'):
            startup_error = dict(kind=kind, stage=stage, traceback=traceback.format_exc())
        raise
    finally:
        evidence('logs_archive', lambda: copy_evidence(scratch/'logs', cloud/'diagnostics'), errors)
        archived = {}
        evidence('archive_bytes', lambda: archived.update(bytes=sum(
            path.stat().st_size for path in evidence_files(cloud))), errors)
        evidence('archive_manifest', lambda: save(output/'archive_manifest.json', dict(files=archives,
            raw_paths_preserved=True, cloud=str(cloud), worker_tmp=str(scratch),
            evidence_bytes=args.evidence_bytes, archived_bytes=archived.get('bytes'))), errors)
        payload = dict(cases=statuses, startup_error=startup_error,
            native_exit=next((row['native_exit'] for row in statuses if row['native_exit'] != 0), 0) if statuses else None,
            status='raw_unadmitted', evidence_errors=errors, cloud=str(cloud), worker_tmp=str(scratch),
            evidence_bytes=args.evidence_bytes, archived_bytes=archived.get('bytes'), evidence_usage_bytes=None,
            expected_observations=60, sample_scope='Periodic parent/GPU/NIC evidence is not continuous exclusivity')

        def account_status():
            usage = generation_bytes()
            # Include the exact pending status bytes, which also contain cap diagnostics.
            while True:
                payload['evidence_status'] = 'UNVERIFIED' if errors or startup_error else 'recorded_not_validated'
                total = usage+len((json.dumps(payload, indent=2, default=str)+'\n').encode())
                if payload['evidence_usage_bytes'] == total:
                    return total
                payload['evidence_usage_bytes'] = total

        evidence('parent_evidence_bytes', lambda: require_bytes(account_status()), errors)
        # A cap failure adds a traceback to the status; recount without writing twice.
        diagnostic(account_status)
        payload['evidence_status'] = 'UNVERIFIED' if errors or startup_error else 'recorded_not_validated'
        evidence('parent_status', lambda: save(output/'status.json', payload), errors)
        release_closed_stdout()
    return next((row['native_exit'] for row in statuses if row['native_exit'] != 0), 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('submit', 'run', 'child'))
    for flag in ('task-root', 'output-dir', 'cloud-root', 'worker-tmp-root'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--driver-sha256', required=True)
    parser.add_argument('--tensor-copy-bytes', type=int, required=True)
    parser.add_argument('--evidence-bytes', type=int, required=True)
    parser.add_argument('--kind', choices=KINDS)
    args = parser.parse_args()
    if (args.mode == 'child' and args.kind is None) or (args.mode != 'child' and args.kind is not None):
        parser.error('--kind is required only for child')
    task, output, cloud, scratch = validate_paths(args)
    if args.mode != 'submit' and not os.path.ismount('/mnt/codesign-exp'):
        raise ValueError('Required actual cloud volume is not mounted')
    launcher, collector, attention, files, plan = prepare(task, args.driver_sha256)
    if args.mode == 'submit':
        if os.environ.get('STEPMIND_BACKEND') != 'rjob':
            raise ValueError('STEPMIND_BACKEND must be rjob')
        config, kwargs = submit_payload(launcher, task, output, cloud, scratch, args)
        from steptron.exp.base_exp import ResourceConfig
        from steptron.utils.stepmind import spawn_tasks
        os.chdir(WORKTREE); os.environ['EXP_ID'] = 'step5-local-attention'
        workers = spawn_tasks(ResourceConfig(**config), **kwargs)
        print('JOB_ID', workers.rjob.meta.name, 'FINAL_STATUS', workers.poll(), flush=True)
        return 0
    if args.mode == 'child':
        execute(task, output, scratch, args.kind, args, collector, attention, files, plan)
        return 0
    return run_parent(task, output, cloud, scratch, args, collector)


if __name__ == '__main__':
    raise SystemExit(main())
