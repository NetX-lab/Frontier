"""Import SGLang torch-profiler traces as Frontier GDN kernel profiles."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from frontier.gdn import GATED_DELTA_NET_FAMILY
from frontier.gdn.profiling_schema import validate_gdn_profiling_dataframe
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.timer_stats_store import TimerStatsStore
from frontier.profiling.utils import build_profile_method_output_path
from frontier.types import MeasurementType


_TRACE_FILENAME_RE = re.compile(
    r"_batch(?P<batch_size>\d+)_input(?P<input_len>\d+)_output(?P<output_len>\d+)"
    r"_(?P<phase>prefill|decode)\.trace\.json(?:\.gz)?$"
)
_INPUT_NORM_KERNELS = (
    "_gemma_fused_add_rmsnorm_kernel",
    "_gemma_rmsnorm_kernel",
)
_OUTPUT_NORM_KERNEL = "_layer_norm_fwd_1pass_kernel"
_TRANSFORM_KERNEL = "elementwise_kernel_manual_unroll"
_ALL_REDUCE_KERNELS = (
    "quickreduce::allreduce",
    "cross_device_reduce",
)
_PHASE_ANCHORS = {
    "prefill": "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "decode": "fused_recurrent_gated_delta_rule_packed_decode_kernel",
}


def _contains_any(name: str, fragments: Iterable[str]) -> bool:
    return any(fragment in name for fragment in fragments)


def _load_trace_events(path: str | Path) -> list[dict[str, Any]]:
    trace_path = Path(path)
    opener = gzip.open if trace_path.suffix == ".gz" else open
    with opener(trace_path, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    events = [
        event
        for event in trace.get("traceEvents", ())
        if event.get("cat") == "kernel"
        and event.get("ph") == "X"
        and float(event.get("dur", 0.0)) >= 0.0
    ]
    if not events:
        raise ValueError(f"SGLang trace contains no GPU kernel events: {trace_path}")
    return sorted(events, key=lambda event: float(event["ts"]))


def parse_sglang_trace_shape(path: str | Path) -> dict[str, int | str]:
    """Read the static-batch shape and phase encoded by one_batch.py."""

    match = _TRACE_FILENAME_RE.search(Path(path).name)
    if match is None:
        raise ValueError(
            "SGLang trace filename must end in "
            "_batchN_inputN_outputN_{prefill|decode}.trace.json[.gz]: "
            f"{path}"
        )
    values: dict[str, int | str] = match.groupdict()
    for name in ("batch_size", "input_len", "output_len"):
        values[name] = int(values[name])
    return values


def _find_previous_input_norm(events: list[dict[str, Any]], anchor: int) -> int:
    for index in range(anchor - 1, -1, -1):
        if _contains_any(str(events[index]["name"]), _INPUT_NORM_KERNELS):
            return index
    raise ValueError("Could not find the decoder-layer input norm before a GDN anchor")


def _find_next(
    events: list[dict[str, Any]],
    start: int,
    predicate,
    *,
    description: str,
) -> int:
    for index in range(start, len(events)):
        if predicate(str(events[index]["name"])):
            return index
    raise ValueError(f"Could not find {description} after a GDN anchor")


def extract_sglang_gdn_samples(
    path: str | Path,
    *,
    phase: str | None = None,
) -> dict[str, list[float]]:
    """Extract per-layer kernel milliseconds from an SGLang Kineto trace.

    SGLang's Qwen3.5 layer keeps its input norm and TP all-reduce outside
    ``Qwen3_5GatedDeltaNet``.  Those kernels are used as stable boundaries but
    deliberately excluded from the returned GDN samples.
    """

    shape = parse_sglang_trace_shape(path)
    trace_phase = str(shape["phase"])
    if phase is not None and str(phase) != trace_phase:
        raise ValueError(f"Trace phase {trace_phase!r} does not match {phase!r}")
    events = _load_trace_events(path)
    return extract_sglang_gdn_event_samples(events, phase=trace_phase)


def extract_sglang_gdn_event_samples(
    events: list[dict[str, Any]], *, phase: str
) -> dict[str, list[float]]:
    """Extract from one rank's ordered GPU kernels, including captured batches."""
    if phase not in _PHASE_ANCHORS:
        raise ValueError(f"Unsupported SGLang GDN trace phase: {phase}")
    trace_phase = phase
    anchor_fragment = _PHASE_ANCHORS[trace_phase]
    anchor_indices = [
        index
        for index, event in enumerate(events)
        if anchor_fragment in str(event["name"])
    ]
    if not anchor_indices:
        raise ValueError(
            f"SGLang {trace_phase} trace contains no GDN anchor {anchor_fragment!r}"
        )

    samples = {
        "gdn_input_projections": [],
        f"gdn_core_{trace_phase}": [],
        "gdn_output_projection": [],
        "gdn_layer_e2e": [],
    }
    for anchor in anchor_indices:
        input_norm = _find_previous_input_norm(events, anchor)
        layer_start = input_norm + 1
        transform_start = _find_next(
            events,
            layer_start,
            lambda name: _TRANSFORM_KERNEL in name,
            description="the first GDN projection-output transform",
        )
        output_start = _find_next(
            events,
            anchor + 1,
            lambda name: _OUTPUT_NORM_KERNEL in name,
            description="the GDN gated output norm",
        )
        all_reduce = _find_next(
            events,
            output_start + 1,
            lambda name: _contains_any(name, _ALL_REDUCE_KERNELS),
            description="the post-GDN TP all-reduce",
        )
        layer_end = all_reduce - 1
        if not (layer_start < transform_start <= anchor < output_start <= layer_end):
            raise ValueError(
                "Invalid SGLang GDN kernel ordering: "
                f"start={layer_start}, transform={transform_start}, anchor={anchor}, "
                f"output={output_start}, end={layer_end}"
            )

        def elapsed_ms(start: int, end: int) -> float:
            # Kineto records GPU durations in microseconds.
            return sum(float(events[index]["dur"]) for index in range(start, end + 1)) / 1000.0

        input_ms = elapsed_ms(layer_start, transform_start - 1)
        core_ms = elapsed_ms(transform_start, output_start - 1)
        output_ms = elapsed_ms(output_start, layer_end)
        samples["gdn_input_projections"].append(input_ms)
        samples[f"gdn_core_{trace_phase}"].append(core_ms)
        samples["gdn_output_projection"].append(output_ms)
        samples["gdn_layer_e2e"].append(input_ms + core_ms + output_ms)

    return samples


def build_sglang_profile_row(
    path: str | Path,
    *,
    model_config: ModelConfig,
    device: str,
    tensor_parallel_size: int,
    runtime_stack_signature: str,
) -> dict[str, Any]:
    shape = parse_sglang_trace_shape(path)
    return build_sglang_profile_row_from_samples(
        extract_sglang_gdn_samples(path, phase=str(shape["phase"])),
        shape=shape, trace_path=str(path), model_config=model_config, device=device,
        tensor_parallel_size=tensor_parallel_size,
        runtime_stack_signature=runtime_stack_signature,
    )


def build_sglang_profile_row_from_samples(
    samples: dict[str, list[float]], *, shape: dict[str, Any], trace_path: str,
    model_config: ModelConfig, device: str, tensor_parallel_size: int,
    runtime_stack_signature: str,
) -> dict[str, Any]:
    """Share profile metadata/units between filename and batch-ledger imports."""
    phase = str(shape["phase"])
    if phase not in _PHASE_ANCHORS:
        raise ValueError(f"Unsupported SGLang GDN profile phase: {phase}")
    batch_size = int(shape["batch_size"])
    input_len = int(shape["input_len"])
    expected_layers = model_config.get_num_gdn_layers()
    sample_count = len(samples["gdn_layer_e2e"])
    if expected_layers <= 0 or sample_count <= 0 or sample_count % expected_layers != 0:
        raise ValueError(
            "SGLang GDN sample count must be a positive multiple of the model's "
            f"GDN layer count: samples={sample_count}, layers={expected_layers}"
        )

    time_stats = TimerStatsStore.get_stats_from_times(samples)
    sample_iterations = sample_count // expected_layers
    for operator in GATED_DELTA_NET_FAMILY.operators:
        time_stats.setdefault(
            operator.profiling_name(),
            {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "count": sample_count,
            },
        )

    gdn_config = model_config.get_gdn_config()
    if gdn_config is None:
        raise ValueError("SGLang GDN trace import requires GDN model dimensions")
    is_prefill = phase == "prefill"
    return {
        "measurement_type": MeasurementType.KERNEL_ONLY.value,
        "model_architecture_profile": (
            model_config.get_model_architecture_profile().profile_id
        ),
        "quant_signature": model_config.get_quant_signature(),
        "device": str(device),
        "runtime_stack_signature": str(runtime_stack_signature),
        "gdn_runtime_backend": "sglang_rocm_aiter_trace",
        "gdn_rank_aggregation": "single_rank_kernel_trace",
        "gdn_prefill_backend": "triton_chunk_gated_delta_rule",
        "gdn_decode_backend": "packed_recurrent_triton",
        "gqa_interleaved_layout": False,
        "packed_recurrent_decode": True,
        "model_dtype": str(model_config.dtype),
        "conv_state_dtype": "bfloat16",
        "recurrent_state_dtype": "float32",
        "weight_source": "checkpoint_quark_mxfp4",
        "num_tensor_parallel_workers": int(tensor_parallel_size),
        "hidden_size": model_config.embedding_dim,
        "conv_kernel_size": gdn_config.conv_kernel_size,
        "key_head_dim": gdn_config.key_head_dim,
        "value_head_dim": gdn_config.value_head_dim,
        "num_key_heads": gdn_config.num_key_heads,
        "num_value_heads": gdn_config.num_value_heads,
        "batch_size": batch_size,
        "batch_num_tokens": batch_size * input_len if is_prefill else batch_size,
        "batch_num_prefill_tokens": batch_size * input_len if is_prefill else 0,
        "batch_num_decode_tokens": 0 if is_prefill else batch_size,
        "max_query_len": input_len if is_prefill else 1,
        "has_initial_state": not is_prefill,
        "query_lens": [input_len] * batch_size if is_prefill else [1] * batch_size,
        "context_lens": [0] * batch_size if is_prefill else [input_len] * batch_size,
        "trace_path": trace_path,
        "trace_profiled_iterations": sample_iterations,
        "time_stats": time_stats,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="mi355x")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--output-dir", default="data/profiling")
    parser.add_argument(
        "--runtime-stack-signature",
        required=True,
        help="Pinned SGLang/torch/ROCm/AITER revision signature",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_config = ModelConfig.from_model_name(args.model)
    rows = [
        build_sglang_profile_row(
            path,
            model_config=model_config,
            device=args.device,
            tensor_parallel_size=args.tensor_parallel_size,
            runtime_stack_signature=args.runtime_stack_signature,
        )
        for path in args.trace
    ]
    dataframe = pd.DataFrame(rows)
    dataframe = (
        pd.json_normalize(dataframe["time_stats"])
        .add_prefix("time_stats.")
        .join(dataframe.drop(columns=["time_stats"]))
    )
    validate_gdn_profiling_dataframe(dataframe)
    output_path = build_profile_method_output_path(
        output_root=args.output_dir,
        profiling_type="compute",
        hardware=args.device,
        model_name=args.model,
        op_name="gdn",
        profile_method="record_function",
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    print(f"Wrote {len(dataframe)} SGLang GDN trace rows to {output_path}")


if __name__ == "__main__":
    main()
