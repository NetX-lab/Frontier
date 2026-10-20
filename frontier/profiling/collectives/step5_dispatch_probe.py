"""Observe two native Step5MoE layers under torchrun, without measuring cost.

Run after the collective benchmark in the official Step image::

    torchrun --standalone --nproc-per-node=8 -m \
        frontier.profiling.collectives.step5_dispatch_probe \
        --model-path /model/metadata --output-dir /output/native_dispatch

The metadata directory must contain config.json; no checkpoint is loaded. The
native dummy loader initializes one layer at a time. Native shared experts,
router, expert oracle, prepare/finalize, and communication remain unchanged.
Interceptors record before calling the original methods. A schema discrepancy
is a diagnostic verdict in dispatch_probe.json, not a fabricated pass or a
reason to discard the preceding benchmark. Native execution errors still raise.
This verifies complete native MoE layers, not the full model/model-runner path.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from unittest.mock import patch


# S0's preregistered layer identities and serving topology, not cost constants.
LAYERS = ((3, "fp8", 4288), (88, "bf16", 8256))
TP_SIZE, DP_SIZE, EP_SIZE = 4, 2, 8


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens-per-dp-rank", type=int, nargs=DP_SIZE, default=[1, 1])
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if any(n <= 0 for n in args.tokens_per_dp_rank):
        parser.error("Each DP rank needs a positive token count (idle ranks use one)")
    return args


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
    return {"schema_version": 1, "purpose": "native_dispatch_identity",
            "cost_data": False, "timing_rows": 0,
            "verification_scope": "native_moe_layer",
            "full_model_verified": False, "model_runner_verified": False,
            "model_config_path": str(config_path),
            "parallelism": {"tp": TP_SIZE, "dp": DP_SIZE, "ep": EP_SIZE},
            "tokens_per_dp_rank": list(args.tokens_per_dp_rank),
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
            "cases": cases}


def tensor_schema(tensor):
    """Describe actual tensors without reading their values or synchronizing."""
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "element_size": tensor.element_size(), "numel": tensor.numel(),
            "nbytes": tensor.numel() * tensor.element_size(),
            "device": str(tensor.device)}


class DispatchObserver:
    """Transparent, synchronous diagnostics at the GroupCoordinator boundary."""

    def __init__(self, emit):
        self.emit = emit
        self.calls = []
        self.expert_calls = []

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
                    result = _original(group, input_, *args, **kwargs)
                    output = bound.arguments.get("output", result)
                    outputs = output if isinstance(output, list) else [output]
                    call["outputs"] = [tensor_schema(t) for t in outputs if t is not None]
                    call["completed"] = True
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
                    name = type(instance).__name__
                    self.expert_calls.append(name)
                    self.emit({"event": "expert_enter", "experts_class": name})
                    return original(*args, **kwargs)

                stack.enter_context(patch.object(instance, "apply", applied))
                # Native FallbackExperts owns precisely these two children.
                for attr in ("experts", "fallback_experts"):
                    child = getattr(instance, attr, None)
                    if child is not None:
                        observe(child)

            observe(experts)
            yield


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


def create_native_config(args):
    """Use the same text-submodel and quantization mappings as native serving."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.models.step5.nvidia.model import Step5ForCausalLM

    config = EngineArgs(
        model=str(args.model_path.resolve()),
        load_format="dummy", skip_tokenizer_init=True, language_model_only=True,
        tensor_parallel_size=TP_SIZE, data_parallel_size=DP_SIZE,
        enable_expert_parallel=True, all2all_backend="allgather_reducescatter",
        distributed_executor_backend="external_launcher", enforce_eager=True,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
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
    return {"runner_class": type(layer.experts).__name__,
            "quant_method": type(method).__name__,
            "oracle_backend": getattr(backend, "name", None),
            "experts_class": type(kernel.fused_experts).__name__,
            "prepare_finalize_class": type(kernel.prepare_finalize).__name__,
            "w13": tensor_schema(routed.w13_weight),
            "w2": tensor_schema(routed.w2_weight),
            "sequence_parallel": layer.use_sequence_parallel}


def run_probe(args, plan):
    import torch
    import vllm
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state as ps
    from vllm.forward_context import set_forward_context
    from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
    from vllm.v1.worker.workspace import init_workspace_manager

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != EP_SIZE or int(os.environ["LOCAL_WORLD_SIZE"]) != EP_SIZE:
        raise ValueError("S0 requires torchrun --standalone --nproc-per-node=8 on one node")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    config = create_native_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_path = args.output_dir / f"rank_{rank}.json"
    report = {"rank": rank, "local_rank": local_rank, "host": socket.gethostname(),
              "status": "in_progress", "layers": [],
              "torch": torch.__version__, "vllm": vllm.__version__}
    with (args.output_dir / f"rank_{rank}.jsonl").open("w", buffering=1) as log:
        def emit(record):
            log.write(json.dumps({"rank": rank, **record}) + "\n")

        with set_current_vllm_config(config):
            init_worker_distributed_environment(config, rank, "env://", local_rank)
            init_workspace_manager(device)
            groups = {name: getattr(ps, f"get_{name}_group")() for name in ("tp", "dp", "ep")}
            report["group_ranks"] = {name: list(group.ranks) for name, group in groups.items()}
            tp_start = rank // TP_SIZE * TP_SIZE
            expected_tp = list(range(tp_start, tp_start + TP_SIZE))
            expected_dp = list(range(rank % TP_SIZE, EP_SIZE, TP_SIZE))
            expected_groups = {"tp": expected_tp, "dp": expected_dp, "ep": list(range(EP_SIZE))}
            if report["group_ranks"] != expected_groups:
                raise RuntimeError(f"Unexpected native serving groups: {report['group_ranks']}")
            dp_rank = config.parallel_config.data_parallel_rank
            rows = args.tokens_per_dp_rank[dp_rank]
            sizes = torch.tensor(args.tokens_per_dp_rank, dtype=torch.int32, device="cpu")
            hidden = config.model_config.hf_text_config.hidden_size
            try:
                for case in plan["cases"]:
                    emit({"event": "layer_start", "layer_id": case["layer_id"]})
                    layer = load_native_layer(config, case, device)
                    result = {"layer_id": case["layer_id"], **native_backend(layer)}
                    report["layers"].append(result)
                    emit({"event": "native_backend", **result})
                    observer = DispatchObserver(lambda event: emit({"layer_id": case["layer_id"], **event}))
                    inputs = torch.randn(rows, hidden, dtype=config.model_config.dtype, device=device)
                    with (torch.inference_mode(), set_forward_context(
                            None, config, num_tokens=rows, num_tokens_across_dp=sizes),
                          observer.collectives(ps.GroupCoordinator),
                          observer.experts(layer.experts.routed_experts.quant_method.moe_kernel.fused_experts)):
                        shared, routed = layer(inputs)
                        torch.cuda.synchronize(device)
                    result.update({"collectives": observer.calls,
                                   "executed_experts_classes": observer.expert_calls,
                                   "shared_output": tensor_schema(shared),
                                   "routed_output": tensor_schema(routed)})
                    result["comparison"] = compare_dispatch(
                        case, observer.calls, rows=rows, sizes=args.tokens_per_dp_rank,
                        dp_ranks=expected_dp, tp_ranks=expected_tp, hidden=hidden)
                    emit({"event": "layer_verdict", "layer_id": case["layer_id"],
                          **result["comparison"]})
                    rank_path.write_text(json.dumps(report, indent=2) + "\n")
                    del inputs, shared, routed, layer
                    torch.cuda.empty_cache()
                report["status"] = ("match" if all(layer["comparison"]["status"] == "match"
                                                   for layer in report["layers"]) else "discrepancy")
            except Exception as error:
                report.update(status="runtime_error", error=repr(error))
                raise
            finally:
                rank_path.write_text(json.dumps(report, indent=2) + "\n")
            ranks = [None] * EP_SIZE
            torch.distributed.all_gather_object(ranks, report, group=ps.get_world_group().cpu_group)
            if rank == 0:
                report = {**plan, "ranks": ranks,
                          "status": "match" if all(r["status"] == "match" for r in ranks) else "discrepancy"}
                if len({r["host"] for r in ranks}) != 1:
                    report["status"] = "discrepancy"
                    report["placement_error"] = "S0 requires one node"
                path = args.output_dir / "dispatch_probe.json"
                path.write_text(json.dumps(report, indent=2) + "\n")
                print("STEP5_DISPATCH_PROBE_REPORT", json.dumps({"path": str(path),
                      "status": report["status"], "cost_data": False}), flush=True)
            ps.destroy_model_parallel()
            ps.destroy_distributed_environment()


def main(argv=None):
    args = parse_args(argv)
    plan = probe_plan(args)
    if args.dry_run:
        print(json.dumps({**plan, "status": "not_run"}, indent=2))
        return
    run_probe(args, plan)


if __name__ == "__main__":
    main()
