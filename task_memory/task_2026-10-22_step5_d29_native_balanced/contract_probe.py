"""Read-only frozen-method CPU probes and hypothetical assignment contract checks.

No strategy is installed or registered; hypothetical IDs only demonstrate a
proposed mathematical contract. This is not an implementation for serving.
"""

import ast
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import torch

ROOT = Path('/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm')
TASK = Path(__file__).resolve().parent
REPO = TASK.parents[1]


def extracted(relative, names, namespace, class_name=None):
    tree = ast.parse((ROOT / relative).read_text())
    nodes = tree.body
    if class_name:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef)
            and node.name == class_name).body
    selected = [node for node in nodes if getattr(node, 'name', None) in names]
    assert len(selected) == len(names)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[
        ast.alias(name='annotations')], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT / relative), 'exec'), namespace)
    return namespace


def main():
    sys.path.insert(0, str(REPO))
    from frontier.moe_ep_workload import materialize_expert_token_counts

    # Execute actual native DPMetadata methods with real CPU Torch only.
    space = dict(torch=torch, dataclass=dataclass, contextmanager=contextmanager)
    extracted('forward_context.py', {'_compute_sp_num_tokens', 'DPMetadata'}, space)
    dp = space['DPMetadata']
    checked = []
    for counts in ([6128, 1], [6144, 1], [8, 8], [1, 1], [4, 4], [256, 256], [0, 1]):
        meta = dp(torch.tensor(counts, dtype=torch.int32))
        with meta.sp_local_sizes(1) as sizes:
            assert sizes == counts
            assert meta.cu_tokens_across_sp(1).tolist() == [counts[0], sum(counts)]
        assert meta.local_sizes is None
        # Hypothetical global-prefix sequence, NOT a registered routing strategy.
        rows = [torch.arange(n * 8).reshape(n, 8).add(sum(counts[:rank]) * 8).remainder(352)
            for rank, n in enumerate(counts)]
        ids = torch.cat(rows)
        actual = torch.bincount(ids.flatten(), minlength=352).tolist()
        expected = materialize_expert_token_counts(routing_ratios={e: 1/352 for e in range(352)},
            total_routed_assignments=sum(counts)*8, total_expert_num=352)
        assert actual == [expected[e] for e in range(352)]
        assert sum(actual) == sum(counts)*8 and max(actual)-min(actual) <= 1
        assert all(len(set(row)) == 8 for row in ids.tolist())
        reset = torch.cat([torch.arange(n*8).reshape(n,8).remainder(352) for n in counts])
        reset_counts = torch.bincount(reset.flatten(), minlength=352).tolist()
        logits = torch.linspace(-2,2,352).repeat(sum(counts),1)
        weights = torch.sigmoid(logits.gather(1,ids))
        weights = weights/weights.sum(-1,keepdim=True)*3
        assert torch.allclose(weights.sum(-1),torch.full((sum(counts),),3.0),atol=1e-6)
        checked.append(dict(cpu_dp_counts=counts,physical_wave_tokens=sum(counts),
            global_prefix_assignments=sum(actual),per_expert_min=min(actual),per_expert_max=max(actual),
            frontier_exact_match=True,per_dp_reset_exact_match=reset_counts==actual,
            mismatched_experts=sum(a!=b for a,b in zip(actual,reset_counts)),topk_distinct=True,
            hypothesis_weight_sum3=True,first_ids_per_dp=[r[0].tolist() if len(r) else [] for r in rows]))

    # Run real native padding postprocess with hypothetical already-coordinated
    # CPU tensors. Never call _run_ar or any distributed function.
    space = dict(torch=torch)
    extracted('v1/worker/dp_utils.py', {'_post_process_dp_padding'}, space)
    tensor = torch.tensor([[1,0],[4,1],[0,0],[2,2]],dtype=torch.int32)
    assert space['_post_process_dp_padding'](tensor,True).tolist()==[4,4]
    assert space['_post_process_dp_padding'](tensor,False).tolist()==[4,1]

    # Execute frozen router factory branch with marker classes, no native imports.
    class Marker:
        def __init__(self, **kwargs): self.kwargs=kwargs
    factory = dict(envs=NS(VLLM_MOE_ROUTING_SIMULATION_STRATEGY='uniform_random'),
        RoutingSimulatorRouter=Marker)
    extracted('model_executor/layers/fused_moe/router/router_factory.py',
        {'create_fused_moe_router'},factory)
    selected=factory['create_fused_moe_router'](top_k=8,global_num_experts=352,
        renormalize=True,scoring_func='sigmoid',routed_scaling_factor=3.0,
        e_score_correction_bias=torch.zeros(352))
    assert selected.kwargs==dict(top_k=8,global_num_experts=352,eplb_state=None)

    # Frozen MoERunner method executes select_experts before forward_modular
    # for each actual precision's source-derived modular quant method.
    namespace={}
    extracted('model_executor/layers/fused_moe/runner/moe_runner.py',
        {'_apply_quant_method'},namespace,'MoERunner')
    calls=[]
    for precision,method in (('fp8','Fp8MoEMethod'),('bf16','UnquantizedFusedMoEMethod')):
        def select_experts(**kwargs):
            calls.append([precision,'select_experts'])
            return torch.ones(1,8),torch.arange(8).reshape(1,8)
        def forward_modular(**kwargs):
            calls.append([precision,'forward_modular'])
            assert kwargs['topk_ids'].tolist()==[list(range(8))]
            return torch.zeros(1,1)
        routed=NS(quant_method=NS(is_monolithic=False),forward_modular=forward_modular)
        runner=NS(routed_experts=routed,router=NS(select_experts=select_experts),
            _quant_method=NS(topk_indices_dtype=torch.int32),
            _maybe_apply_shared_experts=lambda *args:None,_shared_experts=NS(output=None))
        namespace['SharedExpertsOrder']=NS(NO_OVERLAP=0)
        namespace['_apply_quant_method'](runner,torch.zeros(1,1),torch.zeros(1,352),None)
    assert calls==[['fp8','select_experts'],['fp8','forward_modular'],
        ['bf16','select_experts'],['bf16','forward_modular']]

    # Read retained real H800 TP8/DP2/EP16 module witnesses from all16 ranks.
    retained=Path('/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/step5_comm_20261020/s2_20261020_01')
    manifest=json.loads((retained/'manifest.json').read_text())
    assert manifest['job']['engine']['moe_backend']=='auto'
    assert manifest['job']['engine']['tensor_parallel_size']==8 and manifest['job']['engine']['data_parallel_size']==2
    witnesses=[]
    for rank in range(16):
        path=retained/f'native_dispatch/rank_{rank}.json'
        rows=json.loads(path.read_text())['layers']
        assert [(r['layer_id'],r['oracle_backend'],r['experts_class']) for r in rows]==[
            (3,'DEEPGEMM','TritonOrDeepGemmExperts'),(88,'TRITON','TritonExperts')]
        assert all(r['prepare_finalize_class']=='MoEPrepareAndFinalizeNaiveDPEPModular'
            and r['sequence_parallel'] is False for r in rows)
        witnesses.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    source_files=['forward_context.py','config/parallel.py','config/kernel.py','envs.py',
        'models/step5/nvidia/model.py','model_executor/models/step3p7.py',
        'model_executor/layers/fused_moe/router/router_factory.py',
        'model_executor/layers/fused_moe/router/routing_simulator_router.py',
        'model_executor/layers/fused_moe/router/fused_topk_bias_router.py',
        'model_executor/layers/fused_moe/runner/moe_runner.py',
        'model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py',
        'distributed/device_communicators/all2all.py','model_executor/layers/fused_moe/modular_kernel.py',
        'model_executor/layers/fused_moe/oracle/fp8.py','model_executor/layers/fused_moe/oracle/unquantized.py',
        'model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py',
        'model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py',
        'model_executor/layers/fused_moe/experts/deep_gemm_moe.py',
        'model_executor/layers/fused_moe/experts/triton_deep_gemm_moe.py',
        'model_executor/layers/fused_moe/experts/triton_moe.py','v1/worker/dp_utils.py','v1/worker/gpu_model_runner.py']
    receipt=dict(task_id='D29-NATIVE-BALANCED',status='PASS_CPU_CONTRACT_PROPOSAL_ONLY',
        interpreter=sys.executable,python=sys.version,torch=torch.__version__,cases=checked,
        cpu_dp_padding=dict(unpadded=[1,0],local_padded=[4,1],synced_graph=[4,4]),
        frozen_factory_marker_kwargs=selected.kwargs,actual_runner_method_call_order=calls,
        retained_native_module_witnesses=witnesses,retained_manifest_sha256=hashlib.sha256((retained/'manifest.json').read_bytes()).hexdigest(),
        source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files},
        limits=['Hypothetical cyclic IDs/weights validate math only; no new native strategy implementation/registration',
            'Native factory/runner/metadata method bodies executed with CPU/mocked class seams, not full engine construction',
            'Retained16-rank receipts are historical native module evidence, not current full-serving/graph kernel attestation',
            'No source/native/image/package change, GPU/platform/network/auth/cloud/SSH or collective call'])
    with (TASK/'cpu_receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps([{k:r[k] for k in ('cpu_dp_counts','physical_wave_tokens','frontier_exact_match','per_dp_reset_exact_match','mismatched_experts')} for r in checked]))
    print(receipt['status'])


if __name__=='__main__': main()
