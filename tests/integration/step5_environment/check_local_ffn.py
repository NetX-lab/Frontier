#!/usr/bin/env python3
"""Run isolated preregistered FFN measurements, never a full-role qualifier."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from check_role_capacity import load, validate, save, diagnostic, TASK, WORKTREE
from check_precision_ffn import CASES, SOURCE_BATCHES, construct
from local_ffn_measurement import measure

CLOUD = Path('/mnt/codesign-exp/ycfeng/step5-preview/local_ffn_20261017_02')
FFN_CASES = CASES[:3]
WARMUPS, REPEATS, SECONDS = 3, 5, 2400


def nic_counters():
    paths = sorted(Path('/sys/class/infiniband').glob('*/ports/*/counters/port_*_data'))
    return {str(path): int(path.read_text()) for path in paths}


def execute(task, output, case, tensor_copy_bytes):
    collector = load(task/'native_g1/run_g1_cases.py')
    _, manifest, _ = collector.verify_inputs(task/'g1_preparation.json')
    files = list(manifest['files'])
    for batch, digest in SOURCE_BATCHES.items():
        path = task/batch/'observed/source_manifest.json'
        assert collector.sha256_file(path) == digest
        files += [dict(relative_path=r['relative_path'], source_sha256=r['sha256'],
                       source_size_bytes=r['size_bytes']) for r in json.loads(path.read_text())['files']
                  if r['status'] == 'copied']
    collector.IMPORTED_FILES = [r['relative_path'] for r in files]
    collector.verify_installed(dict(files=files))
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.models.step5.nvidia.model import Step5ForCausalLM
    from vllm.v1.worker.workspace import init_workspace_manager
    import torch
    assert CLOUD.is_dir() and not (CLOUD/case).exists(), 'Parent-owned exclusive cloud generation required'
    output.mkdir(parents=True, exist_ok=False)
    environment = collector.record_environment(output)
    assert torch.cuda.device_count() == 1 and environment['device_name'] == 'NVIDIA H800'
    save(output/'device.json', dict(uuid=str(torch.cuda.get_device_properties(0).uuid),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
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
            distributed_init_method='file:///tmp/step5_local_ffn_'+case, backend='nccl')
        initialize_model_parallel(1, 1); init_workspace_manager(torch.device('cuda'))
        save(output/'invocation.json', dict(case=case, warmups=WARMUPS, repeats=REPEATS,
            tensor_copy_bytes=tensor_copy_bytes,
            mode='native_rank_local_not_full_role', plan_sha256=collector.sha256_file(task/'local_scope_sampling_01.json'),
            consumer_sha256=collector.sha256_file(Path(__file__).with_name('local_ffn_measurement.py'))))
        before = nic_counters()
        rows = measure(task, CLOUD/case, native, construct, case, warmups=WARMUPS, repeats=REPEATS,
                       tensor_copy_bytes=tensor_copy_bytes)
        after = nic_counters()
        save(output/'measurements.json', dict(rows=rows, nic_before=before, nic_after=after,
            nic_scope='RDMA port counters in 4-byte words; node-wide activity and missing counters preclude exclusive NIC claim'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('submit', 'run', 'child'))
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--case', choices=FFN_CASES)
    parser.add_argument('--tensor-copy-bytes', type=int, required=True)
    args = parser.parse_args(); task, output = args.task_root.resolve(), args.output_dir.resolve()
    validate(task, output)
    if args.tensor_copy_bytes <= 0:
        parser.error('--tensor-copy-bytes must be positive')
    if args.mode == 'child' and args.case is None:
        parser.error('child requires --case')
    if args.mode != 'child' and args.case is not None:
        parser.error('--case is only supported by child')
    if args.mode != 'submit':
        assert os.path.ismount('/mnt/codesign-exp'), 'Required actual cloud mount missing'
    launcher = load(task/'native_source_export/submit_export.py'); launcher.inspection.locked_inputs()
    if args.mode == 'submit':
        assert os.environ['STEPMIND_BACKEND'] == 'rjob'
        from steptron.exp.base_exp import ResourceConfig
        from steptron.utils.stepmind import spawn_tasks
        argv = ['python3', '-B', str(Path(__file__).resolve()), 'run', '--task-root', str(task), '--output-dir', str(output),
                '--tensor-copy-bytes', str(args.tensor_copy_bytes)]
        command = ('export VLLM_ENABLE_CUDA_COMPATIBILITY=1 LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH '
            'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; exec '+shlex.join(argv))
        os.chdir(WORKTREE); os.environ['EXP_ID'] = 'step5-local-ffn'
        workers = spawn_tasks(ResourceConfig(**launcher.resource_config_arguments()),
            **launcher.spawn_arguments('timeout '+str(SECONDS)+' bash -c '+shlex.quote(command)))
        print('JOB_ID', workers.rjob.meta.name, 'FINAL_STATUS', workers.poll(), flush=True)
        return
    if args.mode == 'child':
        execute(task, output, args.case, args.tensor_copy_bytes)
        return
    assert not CLOUD.exists(), 'Cloud generation must be new'
    CLOUD.mkdir(parents=True, exist_ok=False); output.mkdir(parents=True, exist_ok=False)
    resources = load(Path(__file__).with_name('check_precision_resources.py'))
    statuses = []
    for case in FFN_CASES:
        argv = [sys.executable, '-B', str(Path(__file__).resolve()), 'child', '--task-root', str(task),
                '--output-dir', str(output/case), '--case', case, '--tensor-copy-bytes', str(args.tensor_copy_bytes)]
        with (output/(case+'.stdout')).open('xb') as out, (output/(case+'.stderr')).open('xb') as err:
            child = subprocess.Popen(argv, stdout=out, stderr=err)
            while True:
                row = dict(time_ns=time.time_ns(), case=case)
                try:
                    row['processes'] = resources.process_tree(os.getpid())
                    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
                    row['compute_apps'] = dict(exit_code=gpu.returncode, stdout=gpu.stdout, stderr=gpu.stderr)
                except Exception as error:
                    row['observer_error'] = repr(error)
                def append():
                    with (CLOUD/'compute_samples.jsonl').open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                diagnostic(append)
                status = child.poll()
                if status is not None: break
                time.sleep(1)
        statuses.append(dict(case=case, native_exit=status))
        diagnostic(lambda: save(output/(case+'_status.json'), statuses[-1]))
    diagnostic(lambda: save(output/'status.json', dict(cases=statuses, status='raw_unadmitted', cloud=str(CLOUD))))
    raise SystemExit(1 if any(r['native_exit'] != 0 for r in statuses) else 0)


if __name__ == '__main__':
    main()
