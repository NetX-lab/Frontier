"""Observe two native Step5MoE layers under torchrun, without measuring cost.

Run after the collective benchmark in the official Step image::

    torchrun --standalone --nproc-per-node=8 -m \
        frontier.profiling.collectives.step5_dispatch_probe \
        --model-path /model/metadata --output-dir /output/native_dispatch

For multi-node torchrun, pass --tensor-parallel-size 8 and
--data-parallel-size 4 (S1, four nodes) or 2 (S2, two nodes). Each node runs
--gpus-per-node 8 processes; EP is derived from TP x DP, never set separately.

The metadata directory must contain config.json; no checkpoint is loaded. The
native dummy loader initializes one layer at a time. Native shared experts,
router, expert oracle, prepare/finalize, and communication remain unchanged.
Interceptors record before calling the original methods. A schema discrepancy
is a diagnostic verdict in dispatch_probe.json, not a fabricated pass or a
reason to discard the preceding benchmark. Native execution errors still raise.
Observer and artifact failures mark diagnostics unverified, retain in-memory
and best-effort stderr evidence, and do not block native calls or the next layer.
This verifies complete native MoE layers, not the full model/model-runner path.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from unittest.mock import patch


# Preregistered layer identities, not cost constants.
LAYERS = ((3, "fp8", 4288), (88, "bf16", 8256))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--data-parallel-size", type=int, default=2)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--tokens-per-dp-rank", type=int, nargs="+")
    parser.add_argument("--routing-input-file",type=Path)
    parser.add_argument("--moe-input-ids",nargs="+")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    bound=os.environ.get("FRONTIER_STEP5_PROFILE_MANIFEST")
    selected=os.environ.get("VLLM_MOE_ROUTING_SIMULATION_STRATEGY") == "position_cyclic"
    if bound or selected or args.routing_input_file or args.moe_input_ids:
        if not (bound and args.routing_input_file and args.moe_input_ids) or args.tokens_per_dp_rank is not None:
            parser.error("Bound position probe requires explicit input file/IDs, not a legacy token vector")
        if len(set(args.moe_input_ids)) != len(args.moe_input_ids):
            parser.error("Probe input IDs must be distinct")
    try:
        expected_topology(args.tensor_parallel_size, args.data_parallel_size, args.gpus_per_node)
    except ValueError as error:
        parser.error(str(error))
    if args.tokens_per_dp_rank is None:
        args.tokens_per_dp_rank = [1] * args.data_parallel_size
    if len(args.tokens_per_dp_rank) != args.data_parallel_size:
        parser.error("--tokens-per-dp-rank must contain one count per DP rank "
                     f"(expected {args.data_parallel_size})")
    if any(n <= 0 for n in args.tokens_per_dp_rank):
        parser.error("Each DP rank needs a positive token count (idle ranks use one)")
    return args


def expected_topology(tp, dp, gpus_per_node):
    """Plan native TP x DP ranks on full nodes with node-local TP groups."""
    if min(tp, dp, gpus_per_node) <= 0:
        raise ValueError("TP, DP and GPUs per node must be positive")
    world_size = tp * dp
    if gpus_per_node % tp:
        raise ValueError("GPUs per node must be divisible by TP for node-local TP groups")
    if world_size % gpus_per_node:
        raise ValueError("TP x DP must fill an integer number of GPU nodes")
    ranks = []
    for rank in range(world_size):
        tp_start = rank // tp * tp
        ranks.append({"rank": rank, "node_rank": rank // gpus_per_node,
                      "local_rank": rank % gpus_per_node, "dp_rank": rank // tp,
                      "group_ranks": {"tp": list(range(tp_start, tp_start + tp)),
                                      "dp": list(range(rank % tp, world_size, tp)),
                                      "ep": list(range(world_size))}})
    return {"world_size": world_size, "gpus_per_node": gpus_per_node,
            "node_count": world_size // gpus_per_node, "ranks": ranks}


def validate_launcher(topology, environ):
    """Reject an incompatible torchrun allocation before native initialization."""
    for name, expected in (("WORLD_SIZE", topology["world_size"]),
                           ("LOCAL_WORLD_SIZE", topology["gpus_per_node"])):
        actual = int(environ[name])
        if actual != expected:
            raise ValueError(f"{name}={actual}; expected {expected} from TP x DP "
                             "and --gpus-per-node")


def compare_placement(topology, ranks):
    """Compare gathered rank/host observations without raising on discrepancies."""
    differences = []

    def check(name, actual, expected):
        if actual != expected:
            differences.append({"field": name, "expected": expected, "actual": actual})

    expected_ranks = topology["ranks"]
    check("rank_order", [r["rank"] for r in ranks], [r["rank"] for r in expected_ranks])
    hosts = sorted({r["host"] for r in ranks})
    check("node_count", len(hosts), topology["node_count"])
    for host in hosts:
        members = [r for r in ranks if r["host"] == host]
        check(f"host[{host}].gpu_count", len(members), topology["gpus_per_node"])
        check(f"host[{host}].local_ranks", sorted(r["local_rank"] for r in members),
              list(range(topology["gpus_per_node"])))
    observed = {r["rank"]: r for r in ranks}
    for expected in expected_ranks:
        if expected["rank"] in observed:
            check(f"rank[{expected['rank']}].local_rank",
                  observed[expected["rank"]]["local_rank"], expected["local_rank"])
    for node in range(topology["node_count"]):
        node_ranks = [r["rank"] for r in expected_ranks if r["node_rank"] == node]
        members = [r for r in ranks if r["rank"] in node_ranks]
        check(f"node[{node}].ranks", sorted(r["rank"] for r in members), node_ranks)
        check(f"node[{node}].host_count", len({r["host"] for r in members}), 1)
    tp_groups = {tuple(r["group_ranks"]["tp"]) for r in expected_ranks}
    for group in sorted(tp_groups):
        members = [r for r in ranks if r["rank"] in group]
        check(f"tp_group{list(group)}.host_count", len({r["host"] for r in members}), 1)
    return {"status": "discrepancy" if differences else "match", "differences": differences}


def probe_plan(args):
    """Read the explicitly supplied model metadata; do not import torch/vLLM."""
    config_path = args.model_path.resolve() / "config.json"
    metadata = json.loads(config_path.read_text())
    text = metadata.get("text_config", metadata)
    if text["model_type"] != "step5":
        raise ValueError("The native dispatch probe requires Step-5 metadata")
    quant = metadata.get("quantization_config", text.get("quantization_config"))
    if quant is None or quant["quant_method"] != "fp8":
        raise ValueError("Step-5 FP8 checkpoint quantization metadata is required")
    hidden, topk = text["hidden_size"], text["moe_top_k"]
    block_k = quant["weight_block_size"][1]
    if hidden % block_k:
        raise ValueError("Hidden width must be divisible by the FP8 activation group")
    cases = []
    for layer, precision, design_bytes in LAYERS:
        tensors = [(hidden, "float8_e4m3fn" if precision == "fp8" else "bfloat16"),
                   (topk, "float32"), (topk, "int32")]
        if precision == "fp8":
            tensors.append((hidden // block_k, "float32"))
        cases.append({"layer_id": layer, "precision": precision,
                      "prefix": f"model.layers.{layer}.moe",
                      "expected_columns_dtypes": tensors,
                       "design_row_bytes": design_bytes})
    result = {"schema_version": 1, "purpose": "native_dispatch_identity",
            "cost_data": False, "timing_rows": 0,
            "verification_scope": "native_moe_layer",
            "full_model_verified": False, "model_runner_verified": False,
            "model_config_path": str(config_path),
            "parallelism": {"tp": args.tensor_parallel_size, "dp": args.data_parallel_size,
                            "ep": args.tensor_parallel_size * args.data_parallel_size},
            "topology": expected_topology(args.tensor_parallel_size, args.data_parallel_size,
                                          args.gpus_per_node),
            "tokens_per_dp_rank": list(args.tokens_per_dp_rank),
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
             "cases": cases}
    if args.routing_input_file:
        from dataclasses import asdict
        from frontier.profiling.moe.inputs import read_routing_profile_inputs
        inputs={item.moe_input_id:item for item in read_routing_profile_inputs(args.routing_input_file)}
        if not set(args.moe_input_ids) <= inputs.keys():
            raise ValueError("Selected probe input is missing from retained source file")
        if any(len(inputs[key].source_frames) != args.data_parallel_size for key in args.moe_input_ids):
            raise ValueError("Selected probe input DP world differs from allocation")
        result.pop("tokens_per_dp_rank")
        result.update(verification_scope="eager_native_moe_module",profile_manifest=os.environ["FRONTIER_STEP5_PROFILE_MANIFEST"],
            routing_input_file=str(args.routing_input_file),moe_input_ids=args.moe_input_ids,
            cases=[{**case,"routing_input":asdict(inputs[key])} for key in args.moe_input_ids for case in cases])
    return result


def tensor_schema(tensor):
    """Describe actual tensors without reading their values or synchronizing."""
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "element_size": tensor.element_size(), "numel": tensor.numel(),
            "nbytes": tensor.numel() * tensor.element_size(),
            "device": str(tensor.device)}


@contextmanager
def diagnostic_scope(errors, operation):
    """Isolate diagnostic work only; native execution must stay outside this scope."""
    try:
        yield
    except Exception as error:
        record = {"operation": operation, "error": repr(error)}
        errors.append(record)
        try:
            print("STEP5_DISPATCH_PROBE_DIAGNOSTIC_ERROR", json.dumps(record),
                  file=sys.stderr, flush=True)
        except Exception:
            # The in-memory evidence remains even when the diagnostic stream fails.
            pass


def write_report(path, report):
    """Retain write failures in memory without replacing a native exception."""
    errors = report["diagnostic_errors"]
    if errors and report["status"] != "runtime_error":
        report["status"] = "discrepancy"
    with diagnostic_scope(errors, f"write:{path.name}"):
        path.write_text(json.dumps(report, indent=2) + "\n")
    if errors and report["status"] != "runtime_error":
        report["status"] = "discrepancy"


class DispatchObserver:
    """Transparent, synchronous diagnostics at the GroupCoordinator boundary."""

    def __init__(self, emit):
        self.emit = emit
        self.calls = []
        self.expert_calls = []
        self.routing_calls = []
        self.collective_operands = []
        self.expert_operands = []
        self.errors = []

    @contextmanager
    def routing(self, router):
        """Retain native selection references only; numerical checks run afterwards."""
        original=router.select_experts
        @wraps(original)
        def selected(hidden_states,router_logits,topk_indices_dtype=None,*,input_ids=None):
            record=dict(hidden=hidden_states,logits=router_logits,indices_dtype=topk_indices_dtype,
                input_ids=input_ids)
            with diagnostic_scope(self.errors,"routing.enter"):
                from vllm.forward_context import get_forward_context
                from vllm import envs
                from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import RoutingSimulator
                context=get_forward_context()
                record.update(positions=context.routing_positions,is_padding=context.is_padding,
                    actual_config=dict(top_k=router.top_k,global_num_experts=router.global_num_experts,
                        renormalize=router.renormalize,routed_scaling_factor=router.routed_scaling_factor,
                        eplb_enabled=router.eplb_state is not None,routing_method_type=router.routing_method_type.name,
                        strategy=envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY,skip_padding=envs.VLLM_MOE_SKIP_PADDING),
                    strategy=RoutingSimulator._routing_strategies[envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY])
            value=original(hidden_states=hidden_states,router_logits=router_logits,
                topk_indices_dtype=topk_indices_dtype,input_ids=input_ids)
            with diagnostic_scope(self.errors,"routing.exit"):
                record["result"]=value
                self.routing_calls.append(record)
            return value
        with patch.object(router,"select_experts",selected): yield

    def compare_routing(self, source_tensors, text):
        """Verify observed outputs against explicit positions and those same logits."""
        import torch
        observations=[];differences=[]
        def check(field,actual,expected):
            if actual != expected: differences.append(dict(field=field,actual=actual,expected=expected))
        with diagnostic_scope(self.errors,"routing.compare"):
            positions,mask=source_tensors
            check("routing_count",len(self.routing_calls),1)
            for record in self.routing_calls:
                config=record["actual_config"];logits=record["logits"];weights,ids=record["result"]
                expected_config=dict(top_k=text.moe_top_k,global_num_experts=text.moe_num_experts,
                    renormalize=text.norm_expert_weight,routed_scaling_factor=text.moe_router_scaling_factor,
                    eplb_enabled=False,routing_method_type="Simulated",strategy="position_cyclic",skip_padding=True)
                check("router_config",config,expected_config)
                check("scoring_func",getattr(text,"moe_router_activation","sigmoid"),"sigmoid")
                check("strategy_class",type(record["strategy"]).__name__,"PositionCyclicRouting")
                check("logits_shape",list(logits.shape),[len(positions),text.moe_num_experts])
                check("positions",torch.equal(record["positions"],positions),True)
                check("is_padding",torch.equal(record["is_padding"],mask),True)
                wanted_ids=(positions.to(torch.int64)[:,None]*text.moe_top_k+
                    torch.arange(text.moe_top_k,device=positions.device))%text.moe_num_experts
                wanted_weights=logits.float().gather(1,wanted_ids).sigmoid()
                if text.norm_expert_weight: wanted_weights=wanted_weights/wanted_weights.sum(-1,keepdim=True)
                wanted_weights=wanted_weights*text.moe_router_scaling_factor
                wanted_ids=wanted_ids.masked_fill(mask[:,None],-1)
                wanted_weights=wanted_weights.masked_fill(mask[:,None],0)
                check("ids",torch.equal(ids,wanted_ids),True)
                check("weights",torch.equal(weights,wanted_weights),True)
                check("ids_dtype",str(ids.dtype),str(record["indices_dtype"] or torch.int32))
                check("weights_dtype",str(weights.dtype),str(torch.float32))
                observations.append(dict(actual_config=config,scoring_contract="selected_unbiased_sigmoid",
                    strategy_class=type(record["strategy"]).__name__,logits=tensor_schema(logits),
                    positions=positions.tolist(),is_padding=mask.tolist(),ids=ids.tolist(),weights=weights.tolist(),
                    physical_rows=len(positions),live_rows=int((~mask).sum()),live_assignments=int((ids>=0).sum())))
        return dict(status="unverified" if self.errors else "discrepancy" if differences else "match",
            differences=differences,observations=observations)

    @contextmanager
    def collectives(self, coordinator):
        with ExitStack() as stack:
            for name in ("all_gatherv", "reduce_scatterv", "reduce_scatterv_into_output",
                         "all_reduce"):
                original = getattr(coordinator, name)

                @wraps(original)
                def observed(group, input_, *args, _name=name, _original=original, **kwargs):
                    # Bind the native signature: into_output's second argument
                    # is a tensor, whereas all_gatherv's second argument is dim.
                    import inspect

                    call = None
                    with diagnostic_scope(self.errors, f"{_name}.enter"):
                        bound = inspect.signature(_original).bind(group, input_, *args, **kwargs)
                        bound.apply_defaults()
                        tensors = input_ if isinstance(input_, list) else [input_]
                        call = {"op": _name, "group_ranks": list(group.ranks),
                                "dim": bound.arguments.get("dim"),
                                "sizes": bound.arguments.get("sizes"),
                                "inputs": [tensor_schema(t) for t in tensors],
                                "completed": False}
                        self.calls.append(call)
                        self.emit({"event": "collective_enter", **call})
                    operation=lambda:_original(group, input_, *args, **kwargs)
                    if os.environ.get("FRONTIER_STEP5_PROFILE_MANIFEST") and call is not None:
                        observe=None
                        with diagnostic_scope(self.errors,"execution_policy.setup"):
                            from frontier.profiling.collectives.vllm_collective_benchmark import observe_collective_execution
                            observe=observe_collective_execution
                        if observe is None: result=operation()
                        else:
                            result,policy=observe(group,tensors,operation,output=bound.arguments.get("output"))
                            with diagnostic_scope(self.errors,"execution_policy.receipt"):
                                call["execution_policy"]=policy
                                policy["runtime"]=self.runtime_policy()
                                self.errors.extend(dict(operation="execution_policy",error=error) for error in policy["diagnostic_errors"])
                    else:
                        result=operation()
                    if call is not None:
                        with diagnostic_scope(self.errors, f"{_name}.exit"):
                            output = bound.arguments.get("output", result)
                            outputs = output if isinstance(output, list) else [output]
                            call["outputs"] = [tensor_schema(t) for t in outputs if t is not None]
                            call["completed"] = True
                            self.collective_operands.append(dict(op=_name,inputs=tensors,outputs=outputs))
                            self.emit({"event": "collective_exit", **call})
                    return result

                stack.enter_context(patch.object(coordinator, name, observed))
            yield

    @contextmanager
    def experts(self, experts):
        """Observe actual fallback execution too, without invoking its selector."""
        with ExitStack() as stack:
            def observe(instance):
                original = instance.apply

                @wraps(original)
                def applied(*args, **kwargs):
                    with diagnostic_scope(self.errors, "expert.enter"):
                        name = type(instance).__name__
                        self.expert_calls.append(name)
                        self.emit({"event": "expert_enter", "experts_class": name})
                    result=original(*args, **kwargs)
                    with diagnostic_scope(self.errors,"expert.operands"):
                        import inspect
                        bound=inspect.signature(original).bind(*args,**kwargs)
                        bound.apply_defaults()
                        self.expert_operands.append((instance,bound.arguments))
                    return result

                stack.enter_context(patch.object(instance, "apply", applied))
                # Native FallbackExperts owns precisely these two children.
                for attr in ("experts", "fallback_experts"):
                    child = getattr(instance, attr, None)
                    if child is not None:
                        observe(child)

            observe(experts)
            yield

    def compute_context(self, layer, text):
        """Associate the actual expert invocation with its native dispatch operands."""
        from frontier.profiling.moe.fp8_block_experts import NativeRoutedExperts
        from frontier.profiling.common.model_config import get_model_architecture_profile
        dispatch,=[c for c in self.collective_operands if c["op"] == "all_gatherv"]
        instance,args=self.expert_operands[0]
        source=dispatch["inputs"];gathered=dispatch["outputs"]
        weights,ids=self.routing_calls[0]["result"]
        if weights is not source[1] or ids is not source[2]:
            raise ValueError("Observed router outputs differ from actual dispatch operands")
        if (gathered[0] is not args["hidden_states"] or gathered[1] is not args["topk_weights"] or
                gathered[2] is not args["topk_ids"] or
                (gathered[3] if len(gathered) == 4 else None) is not args["a1q_scale"]):
            raise ValueError("Actual expert operands differ from completed native dispatch")
        method=layer.experts.routed_experts.quant_method
        backend=getattr(method,"fp8_backend",getattr(method,"unquantized_backend",None))
        native=NativeRoutedExperts(backend.value,type(instance).__name__,None,method.moe_kernel,
            args["w1"],args["w2"],args["activation"],args["global_num_experts"])
        context=native.prepared_context(gathered[0],args["a1q_scale"],topk_ids=args["topk_ids"],
            expert_map=args["expert_map"],source_tensors=(source[0],source[3] if len(source) == 4 else None))
        if context["experts_class"] != type(self.expert_operands[0][0]).__name__:
            raise ValueError("Observed expert implementation differs from inspected point")
        def tensor(name,value):
            return dict(name=name,dtype=str(value.dtype).removeprefix("torch."),
                element_size=value.element_size(),columns=value.shape[1],row_stride=value.stride(0))
        combine,=[c for c in self.collective_operands if c["op"] == "reduce_scatterv_into_output"]
        context.update(model_architecture_profile=get_model_architecture_profile(text).profile_id,
            routing_runtime_path="position_cyclic",collective_tensors=dict(
                dispatch=[tensor(n,t) for n,t in zip(("activation","weights","ids","scales"),source)],
                output=[tensor("output",combine["outputs"][0])]))
        return context


def compare_dispatch(case, calls, *, rows, sizes, dp_ranks, tp_ranks, hidden):
    """Compare observed schemas, retaining every mismatch rather than coercing it."""
    differences = []

    def check(name, actual, expected):
        if actual != expected:
            differences.append({"field": name, "expected": expected, "actual": actual})

    dispatches = [call for call in calls if call["op"] == "all_gatherv"]
    check("dispatch_count", len(dispatches), 1)
    observed_bytes = []
    for index, call in enumerate(dispatches):
        label = f"dispatch[{index}]"
        check(f"{label}.completed", call["completed"], True)
        check(f"{label}.group_ranks", call["group_ranks"], dp_ranks)
        check(f"{label}.dim", call["dim"], 0)
        check(f"{label}.sizes", call["sizes"], sizes)
        expected = [([rows, columns], dtype)
                    for columns, dtype in case["expected_columns_dtypes"]]
        actual = [(t["shape"], t["dtype"]) for t in call["inputs"]]
        check(f"{label}.input_schema", actual, expected)
        expected_out = [([sum(sizes), columns], dtype)
                        for columns, dtype in case["expected_columns_dtypes"]]
        actual_out = [(t["shape"], t["dtype"]) for t in call.get("outputs", [])]
        check(f"{label}.output_schema", actual_out, expected_out)
        row_bytes = sum(t["nbytes"] for t in call["inputs"]) / rows
        observed_bytes.append(row_bytes)
        check(f"{label}.row_bytes", row_bytes, case["design_row_bytes"])
    combines = [call for call in calls if call["op"].startswith("reduce_scatterv")]
    check("combine_count", len(combines), 1)
    for call in combines:
        check("combine.completed", call["completed"], True)
        check("combine.group_ranks", call["group_ranks"], dp_ranks)
        check("combine.dim", call["dim"], 0)
        check("combine.sizes", call["sizes"], sizes)
        check("combine.input_schema", [(t["shape"], t["dtype"]) for t in call["inputs"]],
              [([sum(sizes), hidden], "bfloat16")])
        check("combine.output_schema", [(t["shape"], t["dtype"])
                                        for t in call.get("outputs", [])],
              [([rows, hidden], "bfloat16")])
    reductions = [call for call in calls if call["op"] == "all_reduce"]
    check("tp_all_reduce_count", len(reductions), 2)
    for call in reductions:
        check("tp_all_reduce.completed", call["completed"], True)
        check("tp_all_reduce.group_ranks", call["group_ranks"], tp_ranks)
    return {"status": "discrepancy" if differences else "match",
            "expected_row_bytes": case["design_row_bytes"],
            "observed_row_bytes": observed_bytes, "differences": differences}


def create_native_config(args, *, engine_limits=None):
    """Use the same text-submodel and quantization mappings as native serving."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.models.step5.nvidia.model import Step5ForCausalLM

    config = EngineArgs(
        model=str(args.model_path.resolve()),
        load_format="dummy", skip_tokenizer_init=True, language_model_only=True,
        tensor_parallel_size=args.tensor_parallel_size, data_parallel_size=args.data_parallel_size,
        enable_expert_parallel=True, all2all_backend="allgather_reducescatter",
        distributed_executor_backend="external_launcher", enforce_eager=True,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        **({name: value for name, value in engine_limits.items()
            if name in ("max_model_len", "max_num_batched_tokens", "max_num_seqs")}
           if engine_limits is not None else {}),
    ).create_engine_config()
    config = config.with_hf_config(config.model_config.hf_text_config,
                                   architectures=["Step5ForCausalLM"])
    configure_quant_config(config.quant_config, Step5ForCausalLM)
    return config


def load_native_layer(config, case, device):
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
    from vllm.model_executor.model_loader.utils import (
        initialize_model,
        process_weights_after_loading,
    )
    from vllm.models.step5.nvidia.model import Step5MoE
    from vllm.utils.torch_utils import set_default_torch_dtype

    with set_default_torch_dtype(config.model_config.dtype), device:
        layer = initialize_model(config, model_class=Step5MoE, prefix=case["prefix"])
        DummyModelLoader(config.load_config).load_weights(layer, config.model_config)
        process_weights_after_loading(layer, config.model_config, device)
    return layer.eval()


def native_backend(layer):
    routed = layer.experts.routed_experts
    method = routed.quant_method
    backend = getattr(method, "fp8_backend", getattr(method, "unquantized_backend", None))
    kernel = method.moe_kernel
    quant=kernel.fused_experts.quant_config.quant_dtype
    return {"runner_class": type(layer.experts).__name__,
            "quant_method": type(method).__name__,
            "oracle_backend": getattr(backend, "name", None),
            "experts_class": type(kernel.fused_experts).__name__,
            "quant_dtype": None if quant is None else str(quant).removeprefix("torch."),
            "defer_input_quant": bool(kernel.fused_experts.expects_unquantized_inputs),
            "prepare_finalize_class": type(kernel.prepare_finalize).__name__,
            "w13": tensor_schema(routed.w13_weight),
            "w2": tensor_schema(routed.w2_weight),
             "sequence_parallel": layer.use_sequence_parallel}


@contextmanager
def source_forward_context(config, rows, sizes, device, routing_input, dp_rank):
    """Bind the one CORE materialization to the real native eager forward."""
    from vllm.forward_context import set_forward_context
    if routing_input is None:
        with set_forward_context(None,config,num_tokens=rows,num_tokens_across_dp=sizes):
            yield
    else:
        from frontier.profiling.moe.inputs import MoESourceFrame, source_frame_tensors
        from vllm.forward_context import set_routing_positions
        frame=MoESourceFrame(**routing_input["source_frames"][dp_rank])
        positions,is_padding=source_frame_tensors(frame,device=device)
        with set_forward_context(None,config,num_tokens=rows,num_tokens_across_dp=sizes,is_padding=is_padding):
            with set_routing_positions(positions):
                yield positions,is_padding


def run_probe(args, plan):
    topology = plan["topology"]
    validate_launcher(topology, os.environ)

    import torch
    import vllm
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state as ps
    from vllm.forward_context import set_forward_context
    from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
    from vllm.v1.worker.workspace import init_workspace_manager

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    config = create_native_config(args)
    rank_path = args.output_dir / f"rank_{rank}.json"
    report = {"rank": rank, "local_rank": local_rank, "host": socket.gethostname(),
              "status": "in_progress", "layers": [], "diagnostic_errors": [],
              "torch": torch.__version__, "vllm": vllm.__version__}
    errors = report["diagnostic_errors"]
    log = None
    with diagnostic_scope(errors, "output_directory"):
        args.output_dir.mkdir(parents=True, exist_ok=True)
    with diagnostic_scope(errors, "log_open"):
        log = (args.output_dir / f"rank_{rank}.jsonl").open("w", buffering=1)

    def emit(record):
        if log is not None:
            with diagnostic_scope(errors, f"log_write:{record['event']}:{record.get('layer_id')}"):
                log.write(json.dumps({"rank": rank, **record}) + "\n")

    with set_current_vllm_config(config):
        try:
            init_worker_distributed_environment(config, rank, "env://", local_rank)
            init_workspace_manager(device)
            groups = {name: getattr(ps, f"get_{name}_group")() for name in ("tp", "dp", "ep")}
            report["group_ranks"] = {name: list(group.ranks) for name, group in groups.items()}
            if args.routing_input_file:
                with diagnostic_scope(errors,"actual_communicator_policy"):
                    from frontier.profiling.collectives.vllm_collective_benchmark import native_collective_runtime,collective_runtime_policy
                    report["startup_runtime_policy"]=collective_runtime_policy(native_collective_runtime(torch,config,groups))
            expected_rank = topology["ranks"][rank]
            expected_groups = expected_rank["group_ranks"]
            expected_tp, expected_dp = expected_groups["tp"], expected_groups["dp"]
            dp_rank = config.parallel_config.data_parallel_rank
            report["dp_rank"] = dp_rank
            group_differences = []
            for field, expected in (("group_ranks", expected_groups),
                                    ("dp_rank", expected_rank["dp_rank"])):
                if report[field] != expected:
                    group_differences.append({"field": field, "expected": expected,
                                              "actual": report[field]})
            report["group_comparison"] = {
                "status": "discrepancy" if group_differences else "match",
                "differences": group_differences}
            emit({"event": "group_verdict", **report["group_comparison"]})
            rows = args.tokens_per_dp_rank[dp_rank]
            sizes = torch.tensor(args.tokens_per_dp_rank, dtype=torch.int32, device="cpu")
            hidden = config.model_config.hf_text_config.hidden_size
            for case in plan["cases"]:
                routing_input=case.get("routing_input")
                vector=(args.tokens_per_dp_rank if routing_input is None else
                    [frame["physical_rows"] for frame in routing_input["source_frames"]])
                rows=vector[dp_rank];sizes=torch.tensor(vector,dtype=torch.int32,device="cpu")
                error_start = len(errors)
                emit({"event": "layer_start", "layer_id": case["layer_id"]})
                layer = load_native_layer(config, case, device)
                observer = DispatchObserver(lambda event: emit({"layer_id": case["layer_id"], **event}))
                if args.routing_input_file:
                    observer.runtime_policy=lambda:collective_runtime_policy(native_collective_runtime(torch,config,groups))
                result = {"layer_id": case["layer_id"], "observer_errors": observer.errors,
                          "comparison": {"status": "unverified", "differences": []}}
                report["layers"].append(result)
                if routing_input is not None:
                    result.update(moe_input_id=routing_input["moe_input_id"],source_frame=routing_input["source_frames"][dp_rank])
                    with diagnostic_scope(observer.errors,"bound_native_source"):
                        from frontier.profiling.common.runtime_identity import verify_step5_profile_runtime
                        from vllm.forward_context import set_routing_positions
                        from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import RoutingSimulator
                        from vllm import envs
                        result["native_facts"]=verify_step5_profile_runtime(plan["profile_manifest"],
                            callables=[type(layer),layer.forward,set_forward_context,set_routing_positions,
                                 layer.experts.router.select_experts,
                                 layer.experts.router._select_experts,layer.experts.router._compute_routing,
                                 RoutingSimulator.simulate_routing,RoutingSimulator._routing_strategies[
                                     envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY].route_tokens,
                                layer.experts.routed_experts.quant_method.moe_kernel.apply])
                with diagnostic_scope(observer.errors, "native_backend"):
                    result.update(native_backend(layer))
                emit({"event": "native_backend", **result})
                inputs = torch.randn(rows, hidden, dtype=config.model_config.dtype, device=device)
                observation = ExitStack()
                with diagnostic_scope(observer.errors, "observe_collectives"):
                    observation.enter_context(observer.collectives(ps.GroupCoordinator))
                with diagnostic_scope(observer.errors, "observe_experts"):
                    observation.enter_context(observer.experts(
                        layer.experts.routed_experts.quant_method.moe_kernel.fused_experts))
                if routing_input is not None:
                    with diagnostic_scope(observer.errors,"observe_routing"):
                        observation.enter_context(observer.routing(layer.experts.router))
                try:
                    with torch.inference_mode(), source_forward_context(config,rows,sizes,device,routing_input,dp_rank) as source_tensors:
                        shared, routed = layer(inputs)
                        torch.cuda.synchronize(device)
                finally:
                    with diagnostic_scope(observer.errors, "observer_close"):
                        observation.close()
                result.update({"collectives": observer.calls,
                               "executed_experts_classes": observer.expert_calls})
                with diagnostic_scope(observer.errors, "output_schema"):
                    result.update({"shared_output": tensor_schema(shared),
                                   "routed_output": tensor_schema(routed)})
                with diagnostic_scope(observer.errors, "compare_dispatch"):
                    expected_case=case
                    if routing_input is not None:
                        quant=result["quant_dtype"]
                        if quant not in (None,"float8_e4m3fn"):
                            raise ValueError("Bound probe reached an unsupported activation quantization")
                        fp8=quant == "float8_e4m3fn" and not result["defer_input_quant"]
                        topk=config.model_config.hf_text_config.moe_top_k
                        columns=[(hidden,"float8_e4m3fn" if fp8 else "bfloat16"),(topk,"float32"),(topk,"int32")]
                        if fp8:
                            quant_config=layer.experts.routed_experts.quant_method.moe_kernel.fused_experts.quant_config
                            shape=quant_config.scale_shape(rows,hidden)
                            if shape != (1,1): columns.append((shape[1],"float32"))
                        from frontier.profiling.collectives.vllm_forward_collectives import DTYPE_BYTES
                        expected_case={**case,"expected_columns_dtypes":columns,
                            "design_row_bytes":sum(c*DTYPE_BYTES[d] for c,d in columns)}
                    result["comparison"] = compare_dispatch(
                        expected_case, observer.calls, rows=rows, sizes=vector,
                        dp_ranks=expected_dp, tp_ranks=expected_tp, hidden=hidden)
                if routing_input is not None:
                    result["routing_comparison"]=observer.compare_routing(source_tensors,config.model_config.hf_text_config)
                    if result["routing_comparison"]["status"] != "match":
                        result["comparison"]["status"]=result["routing_comparison"]["status"]
                    with diagnostic_scope(observer.errors,"observed_compute_context"):
                        result["compute_context"]=observer.compute_context(layer,config.model_config.hf_text_config)
                if observer.errors or len(errors) > error_start:
                    result["comparison"]["status"] = "unverified"
                emit({"event": "layer_verdict", "layer_id": case["layer_id"], **result["comparison"]})
                result["diagnostic_errors"] = errors[error_start:]
                if result["diagnostic_errors"]:
                    result["comparison"]["status"] = "unverified"
                write_report(rank_path, report)
                del inputs, shared, routed, layer
                torch.cuda.empty_cache()
            report["status"] = ("match" if not group_differences and not errors and all(
                layer["comparison"]["status"] == "match" for layer in report["layers"])
                else "discrepancy")
        except Exception as error:
            report.update(status="runtime_error", error=repr(error))
            raise
        finally:
            if log is not None:
                with diagnostic_scope(errors, "log_close"):
                    log.close()
            write_report(rank_path, report)
        ranks = [None] * topology["world_size"]
        torch.distributed.all_gather_object(ranks, report, group=ps.get_world_group().cpu_group)
        if rank == 0:
            aggregate = {**plan, "ranks": ranks, "status": "discrepancy", "diagnostic_errors": []}
            placement = {"status": "unverified", "differences": []}
            with diagnostic_scope(aggregate["diagnostic_errors"], "compare_placement"):
                placement = compare_placement(topology, ranks)
            aggregate.update(placement_comparison=placement,
                             status="match" if placement["status"] == "match" and all(
                                 r["status"] == "match" for r in ranks) else "discrepancy")
            path = args.output_dir / "dispatch_probe.json"
            write_report(path, aggregate)
            with diagnostic_scope(aggregate["diagnostic_errors"], "report_stdout"):
                print("STEP5_DISPATCH_PROBE_REPORT", json.dumps({"path": str(path),
                      "status": aggregate["status"], "cost_data": False}), flush=True)
            if aggregate["diagnostic_errors"] and aggregate["status"] == "match":
                aggregate["status"] = "discrepancy"
                write_report(path, aggregate)
            report = aggregate
        ps.destroy_model_parallel()
        ps.destroy_distributed_environment()
    return report


def main(argv=None):
    args = parse_args(argv)
    plan = probe_plan(args)
    if args.dry_run:
        print(json.dumps({**plan, "status": "not_run"}, indent=2))
        return
    run_probe(args, plan)


if __name__ == "__main__":
    main()
