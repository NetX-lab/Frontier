"""Selected Step physical contracts; CPU fixtures do not qualify GPU prices."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from frontier.cc_backend.collective_plan import CollectivePlan, CollectiveTensor, CollectivePlanCost
from frontier.config.model_config import BaseModelConfig
from frontier.config.precision_type import PrecisionType
from frontier.execution_time_predictor.step5_cost import step5_operator_binding
from frontier.attention.step5 import describe_step5_layer
from frontier.operators.families import get_moe_profiling_operators


def context(quant_dtype=None):
    return dict(model_architecture_profile="step5", routing_runtime_path="position_cyclic",
                quant_dtype=quant_dtype, defer_input_quant=False)


@pytest.mark.parametrize("quant,names", [
    (None, ("moe_gating_linear", "moe_gating_routing_topk", "moe_grouped_gemm")),
    ("float8_e4m3fn", ("moe_gating_linear", "moe_gating_routing_topk", "moe_input_quantize", "moe_grouped_gemm")),
])
def test_selected_registry_and_real_routed_binding(quant, names):
    assert tuple(op.name for op in get_moe_profiling_operators(context(quant))) == names
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    profile = model.get_model_architecture_profile()
    prepare = next(op for op in get_moe_profiling_operators(context("float8_e4m3fn"))
                   if op.name == "moe_input_quantize")
    assert prepare.ep_agnostic
    contract = profile.resolve_layer_contract(model, layer_id=3, operator_name=prepare.name,
        attention_tp_size=8, ffn_tp_size=8, moe_tp_size=1, expert_parallel_size=16)
    assert contract.operator_family_id == "moe"
    metadata = describe_step5_layer(model, 3, attention_tp_size=8, moe_tp_size=1, expert_parallel_size=16)
    predictor = SimpleNamespace(_model_config=model, _get_model_architecture_profile=lambda: profile,
        _get_ffn_tp_key_for_linear_op=lambda: 8,
        _replica_config=SimpleNamespace(attn_tensor_parallel_size=8, moe_tensor_parallel_size=1,
                                      moe_expert_parallel_size=16))
    assert step5_operator_binding(predictor, metadata, prepare.name, context=context("float8_e4m3fn"))[0] is PrecisionType.FP8
    assert step5_operator_binding(predictor, metadata, "moe_gating_linear", context=context(quant))[0] is PrecisionType.FP32


def test_legacy_and_deferred_prepare_target_selection():
    assert tuple(op.name for op in get_moe_profiling_operators()) == (
        "moe_gating_linear", "moe_gating_routing_topk", "moe_shuffling", "moe_grouped_gemm")
    assert "moe_input_quantize" not in [op.name for op in get_moe_profiling_operators(
        {**context("float8_e4m3fn"), "defer_input_quant": True})]
    with pytest.raises(ValueError, match="quant_dtype"):
        get_moe_profiling_operators(context("int8"))


def plan(sizes=(16, 1), phase="dispatch"):
    return CollectivePlan(phase, sizes, (CollectiveTensor("activation", "bfloat16", 2, 4096, 4096),),
        tuple((t, 8 + t) for t in range(8)), "none", json.dumps(context()), '{"supplied_output": true}')


def test_lossless_plan_order_identity_and_count_separation():
    p = plan()
    assert p.stages == (("Broadcast", 0, 0), ("Broadcast", 0, 1))
    assert p.endpoint_bytes(8) == (8192, 17 * 8192)
    assert p.cache_key() != plan((1, 16)).cache_key()
    assert plan((1, 1)).stages == (("AllGather", 0, None),)
    with pytest.raises(ValueError, match="rank order"):
        replace(p, groups=((0, 8),))
    with pytest.raises(ValueError, match="uniform"):
        replace(p, graph_mode="full")
    cost = CollectivePlanCost(6., 1., (2.,) * 16, 3., (7,) * 16,
        json.dumps(dict(evidence_kind="cpu_fixture",kernel_counts=[7]*16,
                        per_rank_events=[["cpu_fixture"]]*16,source_receipt="cpu_fixture")),
        p.cache_key(), "cpu_fixture")
    assert cost.kernel_counts[0] == 7 and len(p.stages) == 2
    with pytest.raises(ValueError, match="composition"):
        replace(cost, duration_ms=5.)


def test_workload_and_result_freeze_caller_owned_sequences():
    sizes=[16,1];tensors=list(plan().tensors);groups=[list(g) for g in plan().groups]
    p=replace(plan(),sizes=sizes,tensors=tensors,groups=tuple(tuple(g) for g in groups))
    key,stages,endpoints=p.cache_key(),p.stages,p.endpoint_bytes(8)
    local=[2.]*16;counts=[7]*16
    cost=CollectivePlanCost(6.,1.,local,3.,counts,json.dumps(dict(kernel_counts=counts,
        per_rank_events=[["cpu_fixture"]]*16,source_receipt="cpu_fixture",evidence_kind="cpu_fixture")),key,"cpu_fixture")
    sizes[0]=8192;tensors.clear();local[0]=9.;counts[0]=99
    assert p.cache_key() == key and p.stages == stages and p.endpoint_bytes(8) == endpoints
    assert p.sizes == (16,1) and len(p.tensors) == 1
    assert cost.local_rank_ms == (2.,)*16 and cost.kernel_counts == (7,)*16
    nested=[list(g) for g in plan().groups]
    p=replace(plan(),groups=nested)
    nested[0][0]=15;nested.clear()
    assert p.groups == plan().groups and cost.duration_ms == 1.+max(cost.local_rank_ms)+3.


def test_native_inclusive_scope_calls_real_source_methods_with_prepare_excluded():
    torch = pytest.importorskip("torch")
    import ast
    from pathlib import Path
    from frontier.profiling.moe.fp8_block_experts import NativeRoutedExperts
    import os
    source = Path(os.environ["FRONTIER_STEP5_NATIVE_SOURCE_ROOT"])/"model_executor/layers/fused_moe" if os.environ.get("FRONTIER_STEP5_NATIVE_SOURCE_ROOT") else Path("unavailable_native_source")
    if not source.is_dir():
        pytest.skip("Retained native Python source is unavailable")
    import sys
    from types import ModuleType
    prepare=next(n for n in ast.parse((source/"prepare_finalize/naive_dp_ep.py").read_text()).body
        if isinstance(n,ast.FunctionDef) and n.name == "_quantize_and_setup_dispatch")
    prepare.returns=None
    for arg in (*prepare.args.args,*prepare.args.kwonlyargs): arg.annotation=None
    namespace={"moe_kernel_quantize_input":lambda *a,**kw:(a[0],torch.ones(a[0].shape[0],1))}
    exec(compile(ast.Module(body=[prepare],type_ignores=[]),str(source/"prepare_finalize/naive_dp_ep.py"),"exec"),namespace)
    module=ModuleType("vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep")
    module._quantize_and_setup_dispatch=namespace["_quantize_and_setup_dispatch"]
    def native_method(path, cls, method, namespace):
        module = ast.parse((source/path).read_text())
        definition = next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name == cls)
        fn = next(n for n in definition.body if isinstance(n,ast.FunctionDef) and n.name == method)
        fn.returns = None
        for arg in (*fn.args.posonlyargs,*fn.args.args,*fn.args.kwonlyargs):
            arg.annotation = None
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source/path),"exec"),namespace)
        return namespace[method]
    calls=[]
    class Experts:
        a2_scale=None
        expects_unquantized_inputs=False
        quant_config=SimpleNamespace(quant_dtype=None)
        def moe_problem_size(self,a,w1,w2,ids):
            return None,a.shape[0],w2.shape[-1],a.shape[-1],ids.shape[-1]
        def apply(self,**kwargs):
            assert kwargs["a1q_scale"] is scales and kwargs["topk_ids"] is ids
            calls.append(("inclusive",kwargs["hidden_states"].shape[0]))
            kwargs["output"].fill_(2.)
        def finalize_weight_and_reduce_impl(self):
            return self
        def apply_reduce(self,**kwargs):
            calls.append(("finalize","weight_and_reduce"));kwargs["output"].copy_(kwargs["fused_expert_output"])
    class WeightAndReduce:
        def apply(self,**kwargs):
            Experts().apply_reduce(**kwargs)
    class Delegate: pass
    class Finalize:
        finalize=native_method("prepare_finalize/no_dp_ep.py","MoEPrepareAndFinalizeNoDPEPModular",
            "finalize",dict(TopKWeightAndReduceDelegate=Delegate))
    class Kernel:
        fused_experts=Experts()
        prepare_finalize=Finalize()
        _fused_experts=native_method("modular_kernel.py","FusedMoEKernelModularImpl","_fused_experts",
            dict(torch=torch,UnfinalizedMoEOutput=type("UnfinalizedMoEOutput",(),{}),
                 current_platform=SimpleNamespace(is_rocm=lambda:False)))
        def _allocate_buffers(self,*args):
            assert args[8] == 22
            return None,None,torch.empty(17,4),False
        def apply(self,**kwargs):
            pytest.fail("Finalized apply would double charge source prepare")
    Kernel.fused_experts.finalize_weight_and_reduce_impl=lambda:WeightAndReduce()
    impl=Kernel()
    kernel=SimpleNamespace(impl=impl,fused_experts=impl.fused_experts,prepare_finalize=impl.prepare_finalize)
    native=NativeRoutedExperts("DEEPGEMM","cpu_fixture",None,kernel,
        torch.empty(22,8,4),torch.empty(22,4,4),"swiglustep",352)
    activation=torch.zeros(17,4);scales=torch.ones(17,1);ids=torch.full((17,8),-1,dtype=torch.int32)
    from unittest.mock import patch
    impl.fused_experts.quant_config=SimpleNamespace(quant_dtype=None,use_nvfp4_w4a4=False,a1_scale=None,
        per_act_token_quant=False,block_shape=None,mx_alignment=1)
    with patch.dict(sys.modules,{module.__name__:module}):
        prepared,scale=native.prepare_source(activation)
        assert prepared is activation and scale.shape == (17,1)
        impl.fused_experts.expects_unquantized_inputs=True
        assert native.prepare_source(activation) == (activation,None)
        impl.fused_experts.expects_unquantized_inputs=False
    output=native.apply_prepared(activation,scales,torch.zeros(17,8),ids,None,in_dtype=torch.bfloat16)
    assert output.shape == (17,4) and torch.all(output == 2.)
    assert calls == [("inclusive",17),("finalize","weight_and_reduce")]
    selector_scope=dict(is_deep_gemm_e8m0_used=lambda:True,_valid_deep_gemm=lambda *a:False)
    selector=native_method("experts/triton_deep_gemm_moe.py","TritonOrDeepGemmExperts",
        "_select_experts_impl",selector_scope)
    class DeepGemm: pass
    class Triton: pass
    impl.fused_experts.experts=DeepGemm();impl.fused_experts.fallback_experts=Triton()
    impl.fused_experts._select_experts_impl=selector.__get__(impl.fused_experts)
    assert native.prepared_context(activation,scales)["experts_class"] == "DeepGemm"
    selector_scope["is_deep_gemm_e8m0_used"]=lambda:False
    assert native.prepared_context(activation,scales)["experts_class"] == "Triton"
    selector_scope["_valid_deep_gemm"]=lambda *a:True
    assert native.prepared_context(activation,scales)["experts_class"] == "DeepGemm"
    coordinator=ast.parse((source.parents[2]/"distributed/parallel_state.py").read_text())
    cls=next(n for n in coordinator.body if isinstance(n,ast.ClassDef) and n.name == "GroupCoordinator")
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == "reduce_scatterv_into_output")
    method.returns=None
    for arg in (*method.args.args,*method.args.kwonlyargs): arg.annotation=None
    namespace={"torch":torch};exec(compile(ast.Module(body=[method],type_ignores=[]),"native_group","exec"),namespace)
    target=torch.empty_like(activation)
    def into_output(input_,output,dim,sizes):
        assert input_ is activation and output is target and sizes == [16,1]
        return output
    group=SimpleNamespace(device_communicator=SimpleNamespace(reduce_scatterv_into_output=into_output))
    assert namespace["reduce_scatterv_into_output"](group,activation,target,0,[16,1]) is target

    # Native capacity is physical M, not a sum rounded from local live routes.
    tree=ast.parse((source/"deep_gemm_utils.py").read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "compute_aligned_M_and_alignment")
    fn.returns=None
    for arg in fn.args.args: arg.annotation=None
    capacity_scope={"round_up":lambda x,a:((x+a-1)//a)*a}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source/"deep_gemm_utils.py"),"exec"),capacity_scope)
    align_module=ModuleType("vllm.utils.deep_gemm")
    align_module.get_theoretical_mk_alignment_for_contiguous_layout=lambda **kw:128
    with patch.dict(sys.modules,{align_module.__name__:align_module}):
        compute=capacity_scope["compute_aligned_M_and_alignment"]
        assert compute(17,8,22,128,None) == (2944,128)
        assert compute(1,8,22,128,None) == (1024,128)

    context_scope=native.prepared_context(activation,scales,source_tensors=(activation[:16],scales[:16]))
    assert context_scope["activation_layout"]["shape"][0] == 17
    assert context_scope["source_activation_layout"]["shape"][0] == 16
    seen=[]
    with patch.dict(os.environ,{"FRONTIER_STEP5_PROFILE_MANIFEST":"cpu_fixture"}), patch(
        "frontier.profiling.common.runtime_identity.verify_step5_profile_runtime",
        side_effect=lambda path,**kw:seen.append((path,kw["callables"])) or {"evidence_kind":"cpu_fixture"}):
        selected=impl.fused_experts.experts
        selected.apply=lambda:None
        assert native.prepared_context(activation,scales)["native_facts"]["evidence_kind"] == "cpu_fixture"
    assert seen == [("cpu_fixture",[selected.apply])]


@pytest.mark.parametrize("ep_id",[0,8,15])
def test_h04_calls_all_physical_scopes_including_empty_lanes(ep_id):
    from tests.unit.test_moe_position_routing import source,materialize
    from frontier.execution_time_predictor.moe_operator_times import MoeOperatorTimes
    lane=materialize(source()).lane(ep_id);ctx=context("float8_e4m3fn")
    class Predictor(MoeOperatorTimes):
        _replica_config=SimpleNamespace(attn_tensor_parallel_size=8)
        _model_config=SimpleNamespace(supports_share_expert=lambda:True)
        _step5_ffn_costs={}
        def _predict_step5_ffn_operator(self,name,features,*,context):
            self._step5_ffn_costs[name]=dict(kernel_count=3,exact_lookup=True,evidence_kind="cpu_fixture")
            return 1.
    predictor=Predictor()
    def price(p):
        return CollectivePlanCost(3.,1.,(1.,)*16,1.,(2,)*16,json.dumps(dict(kernel_counts=[2]*16,
            per_rank_events=[["cpu_fixture"]]*16,source_receipt="cpu_fixture",evidence_kind="cpu_fixture")),p.cache_key(),"cpu_fixture")
    predictor._cc_backend=SimpleNamespace(predict_collective_plan=price)
    ni=lane.routing_source.source_lanes[ep_id//8].physical_rows
    features={op.name:dict(num_tokens=17 if op.name == "moe_grouped_gemm" else ni,
        total_routed_tokens=lane.routed_token_count) for op in get_moe_profiling_operators(ctx)}
    plans={"step5_moe_dispatch":plan(),"step5_moe_combine":plan(phase="combine")}
    plans.update({name:replace(plan(),phase=phase,groups=(tuple(range(8)),tuple(range(8,16)))) for name,phase in (
        ("moe_tensor_parallel_allreduce","routed_tp"),("share_expert_tensor_parallel_allreduce","shared_tp"))})
    times,comm=predictor._predict_step5_position_operator_times(lane,context=ctx,operator_features=features,collective_plans=plans)
    assert set(times.op_times) == set(features) and "moe_shuffling" not in times.op_times
    assert len(times.physical_frame["operator_costs"]) == 4 and len(comm.costs) == 4
    predictor._predict_step5_ffn_operator=lambda *a,**kw:0.
    with pytest.raises(ValueError,match="positive exact"):
        predictor._predict_step5_position_operator_times(lane,context=ctx,operator_features=features,collective_plans=plans)
