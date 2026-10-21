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
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
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
    return {"schema_version": 1, "purpose": "native_dispatch_identity",
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
        self.errors = []

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
                    result = _original(group, input_, *args, **kwargs)
                    if call is not None:
                        with diagnostic_scope(self.errors, f"{_name}.exit"):
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
                    with diagnostic_scope(self.errors, "expert.enter"):
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
        tensor_parallel_size=args.tensor_parallel_size, data_parallel_size=args.data_parallel_size,
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
                error_start = len(errors)
                emit({"event": "layer_start", "layer_id": case["layer_id"]})
                layer = load_native_layer(config, case, device)
                observer = DispatchObserver(lambda event: emit({"layer_id": case["layer_id"], **event}))
                result = {"layer_id": case["layer_id"], "observer_errors": observer.errors,
                          "comparison": {"status": "unverified", "differences": []}}
                report["layers"].append(result)
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
                try:
                    with torch.inference_mode(), set_forward_context(
                            None, config, num_tokens=rows, num_tokens_across_dp=sizes):
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
                    result["comparison"] = compare_dispatch(
                        case, observer.calls, rows=rows, sizes=args.tokens_per_dp_rank,
                        dp_ranks=expected_dp, tp_ranks=expected_tp, hidden=hidden)
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
