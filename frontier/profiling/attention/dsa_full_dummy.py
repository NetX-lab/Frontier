"""Measure DSA scope CUDA-event wall time inside native whole-model FULL replay.

This output is intentionally separate from regular serving tables. CUDA events
are captured as external graph nodes; Python module hooks do not run on replay.
Kernel-only attribution is not inferred from these event intervals.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

from frontier.profiling.attention import dsa_vllm_runner as runner
from frontier.profiling.attention.dsa_main import _add_plan_arguments, build_plan
from frontier.profiling.attention.dsa_plan import INDEXER_SCOPE, MLA_SCOPE
from frontier.profiling.common.runtime_identity import (
    RuntimeIdentity, collect_device_identity, collect_runtime_versions, file_sha256,
)

SCHEMA = "dsa_full_dummy.v1"
NATIVE_REVISION = "2cf0a6915ce544dc493a0990f2ea38d81601128a"


def capture_sizes(values: list[int], max_seqs: int) -> tuple[int, ...]:
    """FULL_DECODE_ONLY only captures uniform query-one widths up to max_seqs."""
    if (not values or any(type(n) is not int or n <= 0 for n in values)
            or len(set(values)) != len(values) or max(values) > max_seqs):
        raise ValueError("Capture sizes must be unique positive ints <= max_num_seqs")
    return tuple(sorted(values))


def dummy_witness(runner_: Any, kwargs: dict, metadata: dict) -> dict:
    """Read native input buffers outside capture, after metadata construction."""
    width = kwargs["num_tokens_padded"]
    if kwargs["num_tokens"] != 1 or kwargs["num_reqs"] != width:
        raise ValueError("FULL dummy replay must have one actual query and padded requests")
    slots = {str(k): value.cpu().tolist() for k, value in kwargs["slot_mappings"].items()}
    seq_lens = runner_.optimistic_seq_lens_cpu[:width].tolist()
    query_starts = runner_.query_start_loc.np[:width + 1].tolist()
    if (not slots or any(v != [-1] * width for v in slots.values())
            or seq_lens != [1] + [0] * (width - 1)
            or query_starts != [0] + [1] * width):
        raise ValueError("Native FULL dummy slot/sequence/query padding witness disagrees")
    return {"capture_width": width, "actual_query_tokens": 1,
            "actual_requests": 1, "cached_context_tokens": 0,
            "slot_mappings": slots, "seq_lens": seq_lens,
            "query_start_loc": query_starts,
            "metadata": {name: runner._scalar_fields(value, runner._MLA_METADATA_FIELDS
                         + runner._INDEXER_METADATA_FIELDS) for name, value in metadata.items()}}


class _FullDummyState(runner._WorkerState):
    def __init__(self, torch_module: Any) -> None:
        super().__init__(0, torch_module)
        self.events: dict[int, dict[tuple[int, str], tuple[Any, Any]]] = {}
        self.hook_calls = 0

    def capture_metadata(self, key, read) -> None:
        # Metadata is recorded by the runner before entering the model graph.
        pass

    def timed_call(self, layer_id, scope, function, args, kwargs, annotate=True):
        self.hook_calls += 1
        if not self.torch.cuda.is_current_stream_capturing():
            return function(*args, **kwargs)
        from vllm.forward_context import get_forward_context

        width = get_forward_context().batch_descriptor.num_tokens
        key = layer_id, scope
        events = self.events.setdefault(width, {})
        if key in events:
            raise ValueError(f"Duplicate captured DSA scope {width}/{key}")
        start = self.torch.cuda.Event(enable_timing=True, external=True)
        end = self.torch.cuda.Event(enable_timing=True, external=True)
        start.record()
        result = function(*args, **kwargs)
        end.record()
        events[key] = start, end
        return result


def measure_worker(worker: Any, plan, sizes: tuple[int, ...], warmup: int, repeats: int,
                   expected_backend: str):
    import torch
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.config import CUDAGraphMode

    model_runner = worker.model_runner
    if (not isinstance(model_runner.model, CUDAGraphWrapper)
            or model_runner.uniform_decode_query_len != 1):
        raise ValueError("Expected native whole-model CUDAGraphWrapper without speculation")
    variants = {layer.global_layer_id: layer.attention_variant for layer in plan.shape.layers}
    inventory = runner.install_dsa_hooks(worker, 0, variants)
    if inventory["attention_backend"] != expected_backend:
        raise ValueError("Native attention backend differs from the requested backend")
    state = _FullDummyState(torch)
    runner._STATE = state
    model_runner.model.clear_graphs()
    # Native capture owns its stream, warmups, dispatcher keys and graph pool.
    with torch.inference_mode():
        model_runner.capture_model()
    if set(state.events) != set(sizes):
        raise ValueError(f"Captured widths {sorted(state.events)} differ from requested {sizes}")
    expected = {(layer.global_layer_id, scope) for layer in plan.shape.layers
                for scope in ((INDEXER_SCOPE, MLA_SCOPE) if layer.indexer_executed else (MLA_SCOPE,))}
    if any(set(events) != expected for events in state.events.values()):
        raise ValueError("Capture did not execute exactly the planned DSA scopes")
    determine = model_runner._determine_batch_execution_and_padding
    build_metadata = model_runner._build_attention_metadata
    records = []
    with torch.inference_mode():
        for width in sizes:
            mode, descriptor = model_runner.cudagraph_dispatcher.dispatch(width, uniform_decode=True)
            if mode != CUDAGraphMode.FULL or descriptor.num_tokens != width:
                raise ValueError(f"Native dispatcher has no exact FULL capture key for {width}")
            entry = model_runner.model.concrete_cudagraph_entries[descriptor]
            if entry.cudagraph is None:
                raise ValueError(f"Native key {descriptor} has no captured graph")
            witnesses = []

            def padded(**kwargs):
                result = determine(**kwargs)
                if kwargs["num_tokens"] != 1:
                    raise ValueError("Only the native actual-query-one dummy is supported")
                # A single GPU has no peer DP maximum. Substitute only the native
                # dispatcher result; all dummy buffers/metadata remain native.
                return mode, descriptor, False, result[3], result[4]

            def metadata(*args, **kwargs):
                result = build_metadata(*args, **kwargs)
                witnesses.append(dummy_witness(model_runner, kwargs, result[0]))
                return result

            with (patch.object(model_runner, "_determine_batch_execution_and_padding", padded),
                  patch.object(model_runner, "_build_attention_metadata", metadata)):
                for repeat in range(warmup + repeats):
                    before = state.hook_calls
                    witness_count = len(witnesses)
                    graph = entry.cudagraph
                    model_runner._dummy_run(1, cudagraph_runtime_mode=CUDAGraphMode.FULL,
                                            uniform_decode=True, allow_microbatching=False,
                                            skip_eplb=True)
                    torch.cuda.synchronize()
                    if state.hook_calls != before or entry.cudagraph is not graph:
                        raise ValueError("FULL replay unexpectedly executed Python scopes or recaptured")
                    if len(witnesses) != witness_count + 1:
                        raise ValueError("FULL replay did not restage native attention metadata")
                    if repeat < warmup:
                        continue
                    for (layer_id, scope), (start, end) in sorted(state.events[width].items()):
                        elapsed = start.elapsed_time(end)
                        if not math.isfinite(elapsed) or elapsed < 0:
                            raise ValueError(f"Invalid replay CUDA-event time {elapsed}")
                        records.append({"schema_version": SCHEMA, "measurement_type": "CUDA_EVENT",
                                        "execution_kind": "full_dummy", "graph_mode": "FULL",
                                        "global_layer_id": layer_id, "attention_variant": variants[layer_id],
                                        "scope": scope, "repeat": repeat - warmup,
                                        "time_ms": elapsed,
                                        "witness": witnesses[-1]})
    return {"inventory": inventory, "records": records,
            "kv_cache_dtype": str(model_runner.vllm_config.cache_config.cache_dtype),
            "resolved_capture_sizes": sorted(state.events),
            "runtime_versions": collect_runtime_versions(),
            "device": collect_device_identity()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    _add_plan_arguments(parser)
    parser.add_argument("--capture_sizes", required=True, type=int, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_dir")
    parser.add_argument("--gpu_memory_utilization", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--model_config_revision", required=True)
    parser.add_argument("--expected_attention_backend", required=True)
    args = parser.parse_args(argv)
    plan = build_plan(args)
    sizes = capture_sizes(args.capture_sizes, args.max_num_seqs)
    if args.query_one_context_max is not None:
        parser.error("Regular query-one serving plans cannot measure FULL dummy replay")
    if args.source_revision != NATIVE_REVISION:
        parser.error(f"This source contract requires native revision {NATIVE_REVISION}")
    payload = {"schema_version": SCHEMA, "native_revision": args.source_revision,
               "model_config_revision": args.model_config_revision,
               "model_config_sha256": file_sha256(args.model_config_path),
               "plan": plan.to_json(), "capture_sizes": sizes,
               "measurement_type": "CUDA_EVENT", "kernel_only_qualified": False,
               "consumer_admission": False, "gpu_execution_observed": False}
    if args.command == "run":
        if None in (args.model_dir, args.gpu_memory_utilization, args.seed):
            parser.error("run requires --model_dir, --gpu_memory_utilization and --seed")
        if file_sha256(Path(args.model_dir) / "config.json") != payload["model_config_sha256"]:
            raise ValueError("Engine model config differs from the planned config")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from vllm import LLM
        from vllm.config import CompilationMode, CUDAGraphMode

        llm = LLM(model=args.model_dir, load_format="dummy", skip_tokenizer_init=True,
                  hf_overrides=dict(plan.hf_overrides), tensor_parallel_size=1,
                  dtype=plan.shape.dtype, enforce_eager=False,
                  enable_prefix_caching=False, enable_chunked_prefill=False,
                  async_scheduling=False, max_num_batched_tokens=args.max_context_len,
                  max_num_seqs=args.max_num_seqs, max_model_len=args.max_context_len,
                  block_size=args.block_size, gpu_memory_utilization=args.gpu_memory_utilization,
                  seed=args.seed, compilation_config={"mode": CompilationMode.NONE,
                    "cudagraph_mode": CUDAGraphMode.FULL_DECODE_ONLY,
                    "cudagraph_capture_sizes": list(sizes)})
        result = llm.collective_rpc(measure_worker, args=(plan, sizes, args.decode_warmup_steps,
            args.decode_measured_steps, args.expected_attention_backend))[0]
        from frontier.profiling.common.model_config import ModelConfig

        device_name, device_capability = result.pop("device")
        payload["identity"] = RuntimeIdentity(
            runtime_versions=result.pop("runtime_versions"), source_revision=args.source_revision,
            model_config_revision=args.model_config_revision,
            model_config_sha256=payload["model_config_sha256"], device_name=device_name,
            device_capability=device_capability, schema_version=SCHEMA,
            quant_signature=ModelConfig.from_model_name(args.model_name).get_quant_signature(),
            parallel_groups={"target_tensor_parallel": plan.shape.target_tensor_parallel_size,
                             "engine_tensor_parallel": 1}, graph_mode="FULL").to_json()
        payload.update(result)
        payload["gpu_execution_observed"] = True
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
