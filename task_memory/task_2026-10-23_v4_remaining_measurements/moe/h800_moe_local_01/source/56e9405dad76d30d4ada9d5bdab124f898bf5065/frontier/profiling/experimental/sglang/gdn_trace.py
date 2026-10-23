"""Import external SGLang Kineto traces as experimental GDN evidence.

The importer records known kernel identities and phase-qualified component
samples.  It does not turn trace durations into standard ``DEVICE_EVENT``
training rows, and it fails closed when the pinned SGLang/AITER anchors are
missing or the trace does not contain a complete GDN-layer pass.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from frontier.attention.families import GATED_DELTA_NET_FAMILY
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.timer_stats_store import TimerStatsStore
from frontier.types import MeasurementType


_TRACE_FILENAME_RE = re.compile(
    r"_batch(?P<batch_size>\d+)_input(?P<input_len>\d+)_output(?P<output_len>\d+)"
    r"_(?P<phase>prefill|decode)\.trace\.json(?:\.gz)?$"
)
_INPUT_NORM_KERNELS = ("_gemma_fused_add_rmsnorm_kernel", "_gemma_rmsnorm_kernel")
_OUTPUT_NORM_KERNEL = "_layer_norm_fwd_1pass_kernel"
_TRANSFORM_KERNEL = "elementwise_kernel_manual_unroll"
_ALL_REDUCE_KERNELS = ("quickreduce::allreduce", "cross_device_reduce")
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
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError(f"SGLang trace must contain a JSON object: {trace_path}")
    events = [
        event
        for event in payload.get("traceEvents", ())
        if isinstance(event, Mapping)
        and event.get("cat") == "kernel"
        and event.get("ph") == "X"
        and float(event.get("dur", 0.0)) >= 0.0
        and "name" in event
        and "ts" in event
    ]
    if not events:
        raise ValueError(f"SGLang trace contains no GPU kernel events: {trace_path}")
    return sorted((dict(event) for event in events), key=lambda event: float(event["ts"]))


def parse_sglang_trace_shape(path: str | Path) -> dict[str, int | str]:
    """Read static batch shape and phase encoded by SGLang one-batch traces."""

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
    events: list[dict[str, Any]], start: int, predicate, *, description: str
) -> int:
    for index in range(start, len(events)):
        if predicate(str(events[index]["name"])):
            return index
    raise ValueError(f"Could not find {description} after a GDN anchor")


def extract_sglang_gdn_event_samples(
    events: list[dict[str, Any]], *, phase: str, expected_layers: int | None = None
) -> dict[str, list[float]]:
    """Extract known GDN component intervals from ordered kernel events."""

    if phase not in _PHASE_ANCHORS:
        raise ValueError(f"Unsupported SGLang GDN trace phase: {phase}")
    anchor_fragment = _PHASE_ANCHORS[phase]
    anchor_indices = [
        index for index, event in enumerate(events) if anchor_fragment in str(event["name"])
    ]
    if not anchor_indices:
        raise ValueError(f"SGLang {phase} trace contains no GDN anchor {anchor_fragment!r}")
    if expected_layers is not None:
        if type(expected_layers) is not int or expected_layers <= 0:
            raise ValueError("expected_layers must be a positive integer")
        if len(anchor_indices) == 0 or len(anchor_indices) % expected_layers:
            raise ValueError(
                "SGLang trace anchor count is not a positive multiple of the GDN layer count: "
                f"anchors={len(anchor_indices)}, layers={expected_layers}"
            )

    samples = {
        "gdn_input_projections": [],
        f"gdn_core_{phase}": [],
        "gdn_output_projection": [],
        "gdn_layer_e2e": [],
    }
    for anchor in anchor_indices:
        input_norm = _find_previous_input_norm(events, anchor)
        layer_start = input_norm + 1
        transform_start = _find_next(
            events, layer_start, lambda name: _TRANSFORM_KERNEL in name,
            description="the first GDN projection-output transform",
        )
        output_start = _find_next(
            events, anchor + 1, lambda name: _OUTPUT_NORM_KERNEL in name,
            description="the GDN gated output norm",
        )
        all_reduce = _find_next(
            events, output_start + 1,
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
            return sum(float(events[index]["dur"]) for index in range(start, end + 1)) / 1000.0

        input_ms = elapsed_ms(layer_start, transform_start - 1)
        core_ms = elapsed_ms(transform_start, output_start - 1)
        output_ms = elapsed_ms(output_start, layer_end)
        samples["gdn_input_projections"].append(input_ms)
        samples[f"gdn_core_{phase}"].append(core_ms)
        samples["gdn_output_projection"].append(output_ms)
        samples["gdn_layer_e2e"].append(input_ms + core_ms + output_ms)
    return samples


def extract_sglang_gdn_samples(path: str | Path, *, phase: str | None = None) -> dict[str, list[float]]:
    shape = parse_sglang_trace_shape(path)
    trace_phase = str(shape["phase"])
    if phase is not None and str(phase) != trace_phase:
        raise ValueError(f"Trace phase {trace_phase!r} does not match {phase!r}")
    return extract_sglang_gdn_event_samples(_load_trace_events(path), phase=trace_phase)


def _zero_time_stat(count: int) -> dict[str, float | int]:
    return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0, "count": count}


def build_sglang_profile_row_from_samples(
    samples: Mapping[str, list[float]], *, shape: Mapping[str, Any], trace_path: str,
    model_config: ModelConfig, device: str, tensor_parallel_size: int,
    runtime_stack_signature: str,
) -> dict[str, Any]:
    """Build an experimental trace row with explicit runtime provenance."""

    phase = str(shape["phase"])
    if phase not in _PHASE_ANCHORS:
        raise ValueError(f"Unsupported SGLang GDN profile phase: {phase}")
    batch_size = int(shape["batch_size"])
    input_len = int(shape["input_len"])
    expected_layers = model_config.get_num_gdn_layers()
    sample_count = len(samples.get("gdn_layer_e2e", ()))
    if expected_layers <= 0 or sample_count <= 0 or sample_count % expected_layers != 0:
        raise ValueError(
            "SGLang GDN sample count must be a positive multiple of the model's "
            f"GDN layer count: samples={sample_count}, layers={expected_layers}"
        )
    for key in ("gdn_input_projections", f"gdn_core_{phase}", "gdn_output_projection"):
        if len(samples.get(key, ())) != sample_count:
            raise ValueError(f"SGLang trace component sample count mismatch for {key}")

    time_stats = TimerStatsStore.get_stats_from_times(dict(samples))
    for operator in GATED_DELTA_NET_FAMILY.profiling_ops():
        time_stats.setdefault(operator.name, _zero_time_stat(sample_count))
    gdn_config = model_config.get_gdn_config()
    if gdn_config is None:
        raise ValueError("SGLang GDN trace import requires GDN model dimensions")
    is_prefill = phase == "prefill"
    row = {
        "measurement_type": MeasurementType.KERNEL_ONLY.value,
        "measurement_source": "sglang_kineto_trace",
        "evidence_kind": "in_situ_kernel_sum",
        "experimental_trace": True,
        "model_architecture_profile": model_config.get_model_architecture_profile().profile_id,
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
        "trace_path": str(trace_path),
        "trace_profiled_iterations": sample_count // expected_layers,
        "time_stats": time_stats,
    }
    return row


def build_sglang_profile_row(
    path: str | Path, *, model_config: ModelConfig, device: str,
    tensor_parallel_size: int, runtime_stack_signature: str,
) -> dict[str, Any]:
    shape = parse_sglang_trace_shape(path)
    return build_sglang_profile_row_from_samples(
        extract_sglang_gdn_samples(path, phase=str(shape["phase"])),
        shape=shape, trace_path=str(path), model_config=model_config,
        device=device, tensor_parallel_size=tensor_parallel_size,
        runtime_stack_signature=runtime_stack_signature,
    )


def write_trace_summary(rows: Iterable[Mapping[str, Any]], output_dir: str | Path) -> tuple[Path, Path]:
    """Write experimental summary CSV and JSON beside the external traces."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in rows]
    flattened = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key != "time_stats"}
        for name, stats in row.get("time_stats", {}).items():
            for suffix, value in stats.items():
                flat[f"time_stats.{name}.{suffix}"] = value
        flattened.append(flat)
    csv_path = output / "gdn-trace-summary.csv"
    json_path = output / "gdn-trace-summary.json"
    pd.DataFrame(flattened).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "experimental_trace": True,
                "measurement_source": "sglang_kineto_trace",
                "evidence_kind": "in_situ_kernel_sum",
                "rows": rows,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    return csv_path, json_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="mi355x")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--output-dir", default="experimental-sglang-trace")
    parser.add_argument("--runtime-stack-signature", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_config = ModelConfig.from_model_name(args.model)
    rows = [
        build_sglang_profile_row(
            path, model_config=model_config, device=args.device,
            tensor_parallel_size=args.tensor_parallel_size,
            runtime_stack_signature=args.runtime_stack_signature,
        )
        for path in args.trace
    ]
    csv_path, _ = write_trace_summary(rows, args.output_dir)
    print(f"Wrote {len(rows)} experimental SGLang GDN trace rows to {csv_path}")


if __name__ == "__main__":
    main()
