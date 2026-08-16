"""Measure held-out and high-token attention point sets on one GPU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from frontier.profiling.attention.attention_input import AttentionInput
from frontier.profiling.attention.attention_wrapper import AttentionWrapper
from frontier.profiling.attention.backends import AttentionBackend
from frontier.profiling.attention.main import (
    _attach_attention_output_metadata,
    _get_available_gpus,
    _get_gpu_local_index_map,
    _get_physical_max_num_blocks_across_gpus,
    _required_max_num_blocks,
    _resolve_model_arch_for_metadata,
    _resolve_precision_for_model,
)
from frontier.profiling.attention.memory_budget import (
    get_attention_backend_workspace_reservation_bytes,
    resolve_requested_max_num_blocks,
)
from frontier.profiling.attention.mixed_attention_input import MixedAttentionInput
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_config import (
    ParallelConfig,
    validate_profile_tp_sizes,
)
from frontier.profiling.utils import profile_method_to_measurement_type


HELD_OUT_POINT_SET = "held-out"
HIGH_TOKEN_TRAINING_POINT_SET = "high-token-training"
MIXED_PREFILL_TAIL_TRAINING_POINT_SET = "mixed-prefill-tail-training"
POINT_SETS = (
    HELD_OUT_POINT_SET,
    HIGH_TOKEN_TRAINING_POINT_SET,
    MIXED_PREFILL_TAIL_TRAINING_POINT_SET,
)
SCHEMA_VERSION = "frontier.attention.cache_save_held_out_measurement/v1"
HIGH_TOKEN_TRAINING_SCHEMA_VERSION = (
    "frontier.attention.cache_save_high_token_training_measurement/v1"
)
MIXED_PREFILL_TAIL_TRAINING_SCHEMA_VERSION = (
    "frontier.attention.mixed_prefill_tail_training_measurement/v1"
)
MODEL_NAME = "Qwen3-30B-A3B-tiny"
TP_SIZES = (1, 2, 4, 8)
BLOCK_SIZE = 16
MAX_MODEL_LEN = 4096
PROFILE_MAX_SEQ_LEN = 4095
PROFILE_METHOD = "cuda_event"
ATTENTION_BACKEND = AttentionBackend.FLASHINFER


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--point-set",
        choices=POINT_SETS,
        default=HELD_OUT_POINT_SET,
    )
    return parser.parse_args()


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _measurement_profile_identity(
    *,
    model_config: ModelConfig,
    precision: str,
) -> dict[str, str]:
    """Return the profiling identity shared by measurement rows and sidecar."""

    return {
        "attention_backend": ATTENTION_BACKEND.value,
        "precision": precision,
        "quant_signature": model_config.get_quant_signature(),
        "model_architecture_profile": (
            model_config.get_model_architecture_profile().profile_id
        ),
    }


def _profile_inputs(
    point_set: str = HELD_OUT_POINT_SET,
) -> list[tuple[str, AttentionInput | MixedAttentionInput]]:
    if point_set == HELD_OUT_POINT_SET:
        return [
            (
                "single_prefill_gap",
                AttentionInput(
                    prefill_chunk_size=8,
                    kv_cache_size=0,
                    batch_size=1,
                    is_prefill=True,
                ),
            ),
            (
                "default_batch_extrapolation",
                MixedAttentionInput(
                    seq_lens=[4095, 4095, 4095, 4095],
                    kv_cache_size=0,
                    mode="even",
                ),
            ),
        ]
    if point_set == HIGH_TOKEN_TRAINING_POINT_SET:
        return [
            (
                f"batch4_seq{sequence_length}",
                MixedAttentionInput(
                    seq_lens=[sequence_length] * 4,
                    kv_cache_size=0,
                    mode="even",
                ),
            )
            for sequence_length in (3072, 3584, 3840, 4032)
        ]
    if point_set == MIXED_PREFILL_TAIL_TRAINING_POINT_SET:
        return [
            (
                f"batch4_seq{sequence_length}",
                MixedAttentionInput(
                    seq_lens=[sequence_length] * 4,
                    kv_cache_size=0,
                    mode="even",
                ),
            )
            for sequence_length in (4056, 4072, 4088)
        ]
    raise ValueError(f"unsupported cache-save measurement point set: {point_set!r}")


def _expected_keys(point_set: str) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (
            int(profile_input.to_dict()["total_tokens"]),
            int(profile_input.to_dict()["kv_cache_size"]),
            int(profile_input.to_dict()["batch_size"]),
        )
        for _, profile_input in _profile_inputs(point_set)
    )


def _measurement_schema_version(point_set: str) -> str:
    if point_set == HELD_OUT_POINT_SET:
        return SCHEMA_VERSION
    if point_set == HIGH_TOKEN_TRAINING_POINT_SET:
        return HIGH_TOKEN_TRAINING_SCHEMA_VERSION
    if point_set == MIXED_PREFILL_TAIL_TRAINING_POINT_SET:
        return MIXED_PREFILL_TAIL_TRAINING_SCHEMA_VERSION
    raise ValueError(f"unsupported cache-save measurement point set: {point_set!r}")


def _measurement_output_stem(point_set: str) -> str:
    if point_set == HELD_OUT_POINT_SET:
        return "attention_cache_save_held_out"
    if point_set == HIGH_TOKEN_TRAINING_POINT_SET:
        return "attention_cache_save_high_token_training"
    if point_set == MIXED_PREFILL_TAIL_TRAINING_POINT_SET:
        return "attention_mixed_prefill_tail_training"
    raise ValueError(f"unsupported cache-save measurement point set: {point_set!r}")


def _flatten_result(
    result: dict[str, object],
    *,
    scenario: str,
    allocation: dict[str, int],
    precision: str,
    model_config: ModelConfig,
) -> dict[str, object]:
    raw_time_stats = result.pop("time_stats")
    if not isinstance(raw_time_stats, dict):
        raise TypeError(f"time_stats must be a dictionary, got {type(raw_time_stats)}")

    row = dict(result)
    for key, value in pd.json_normalize(raw_time_stats).iloc[0].items():
        row[f"time_stats.{key}"] = value

    row["scenario"] = scenario
    row["physical_max_num_blocks"] = allocation["physical"]
    row["requested_max_num_blocks"] = allocation["requested"]
    row["required_max_num_blocks"] = allocation["required"]
    row["selected_max_num_blocks"] = allocation["selected"]
    row["allocated_max_num_blocks"] = allocation["selected"]
    row["allocated_kv_token_capacity"] = allocation["selected"] * BLOCK_SIZE
    row["backend_workspace_reservation_bytes"] = allocation["workspace_bytes"]
    row["profile_input_grid_max_seq_len"] = PROFILE_MAX_SEQ_LEN
    row["is_native_profile_allocation"] = True

    profile_identity = _measurement_profile_identity(
        model_config=model_config,
        precision=precision,
    )
    frame = _attach_attention_output_metadata(
        pd.DataFrame([row]),
        precision_str=profile_identity["precision"],
        model_arch=_resolve_model_arch_for_metadata(model_config),
        model_architecture_profile=profile_identity["model_architecture_profile"],
        quant_signature=profile_identity["quant_signature"],
        measurement_type=profile_method_to_measurement_type(PROFILE_METHOD).value,
    )
    return frame.iloc[0].to_dict()


def _validate_rows(
    rows: pd.DataFrame,
    point_set: str = HELD_OUT_POINT_SET,
) -> None:
    expected = {
        (tp, total_tokens, kv_cache_size, batch_size)
        for tp in TP_SIZES
        for total_tokens, kv_cache_size, batch_size in _expected_keys(point_set)
    }
    actual = {
        (
            int(row.num_tensor_parallel_workers),
            int(row.total_tokens),
            int(row.kv_cache_size),
            int(row.batch_size),
        )
        for row in rows.itertuples(index=False)
    }
    if actual != expected or len(rows) != len(expected):
        raise AssertionError(
            f"measured tuple lattice mismatch: expected={sorted(expected)}, "
            f"actual={sorted(actual)}, rows={len(rows)}"
        )

    target_columns = [
        "time_stats.attn_kv_cache_save.count",
        "time_stats.attn_kv_cache_save.median",
    ]
    missing = [column for column in target_columns if column not in rows.columns]
    if missing:
        raise AssertionError(f"missing cache-save timing columns: {missing}")

    for column in target_columns:
        values = pd.to_numeric(rows[column], errors="raise")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise AssertionError(f"{column} contains non-finite values")
        if (values < 0).any():
            raise AssertionError(f"{column} contains negative values")
    if (pd.to_numeric(rows[target_columns[0]], errors="raise") <= 0).any():
        raise AssertionError("cache-save timing sample count must be positive")

    timing_columns = [
        column for column in rows.columns if column.startswith("time_stats.")
    ]
    for column in timing_columns:
        values = pd.to_numeric(rows[column], errors="coerce").dropna()
        if not values.map(math.isfinite).all() or (values < 0).any():
            raise AssertionError(
                f"timing column contains non-finite or negative values: {column}"
            )


def main() -> None:
    args = _parse_args()
    if args.output_dir.exists():
        raise ValueError(f"output directory must be absent: {args.output_dir}")
    if not args.run_id.strip():
        raise ValueError("run-id must be non-empty")
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA-capable torch runtime is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"exactly one visible GPU is required, got {torch.cuda.device_count()}"
        )

    validate_profile_tp_sizes(TP_SIZES)
    profile_inputs = _profile_inputs(args.point_set)
    required_blocks = _required_max_num_blocks(
        [profile_input for _, profile_input in profile_inputs],
        block_size=BLOCK_SIZE,
    )
    expected_required_blocks_by_point_set = {
        HELD_OUT_POINT_SET: 1024,
        HIGH_TOKEN_TRAINING_POINT_SET: 1008,
        MIXED_PREFILL_TAIL_TRAINING_POINT_SET: 1024,
    }
    expected_required_blocks = expected_required_blocks_by_point_set[args.point_set]
    if required_blocks != expected_required_blocks:
        raise AssertionError(
            f"expected {expected_required_blocks} required KV blocks, "
            f"got {required_blocks}"
        )

    args.output_dir.mkdir(parents=True)
    model_config = ModelConfig.from_model_name(MODEL_NAME)
    dtype, precision = _resolve_precision_for_model(model_config, None, MODEL_NAME)
    profile_identity = _measurement_profile_identity(
        model_config=model_config,
        precision=precision,
    )
    gpu_ids = _get_available_gpus(1)
    gpu_local_idx_map = _get_gpu_local_index_map(gpu_ids)
    torch.cuda.set_device(gpu_local_idx_map[gpu_ids[0]])
    workspace_bytes = get_attention_backend_workspace_reservation_bytes(
        ATTENTION_BACKEND
    )

    rows: list[dict[str, object]] = []
    allocations: dict[str, dict[str, int]] = {}
    for tp_size in TP_SIZES:
        parallel_config = ParallelConfig(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=1,
        )
        physical_blocks = _get_physical_max_num_blocks_across_gpus(
            model_config=model_config,
            parallel_config=parallel_config,
            block_size=BLOCK_SIZE,
            dtype=dtype,
            gpu_ids=gpu_ids,
            max_pipeline_parallel_size=1,
            reserved_memory_bytes=workspace_bytes,
        )
        selected_blocks = resolve_requested_max_num_blocks(
            physical_max_num_blocks=physical_blocks,
            requested_max_num_blocks=required_blocks,
            required_max_num_blocks=required_blocks,
            profile_max_seq_len=PROFILE_MAX_SEQ_LEN,
            block_size=BLOCK_SIZE,
        )
        allocation = {
            "physical": int(physical_blocks),
            "requested": int(required_blocks),
            "required": int(required_blocks),
            "selected": int(selected_blocks),
            "workspace_bytes": int(workspace_bytes),
        }
        allocations[str(tp_size)] = allocation

        wrapper = AttentionWrapper(
            model_config=model_config,
            parallel_config=parallel_config,
            max_num_blocks=selected_blocks,
            max_model_len=MAX_MODEL_LEN,
            profile_max_seq_len=PROFILE_MAX_SEQ_LEN,
            block_size=BLOCK_SIZE,
            attention_backend=ATTENTION_BACKEND,
            dtype=dtype,
            profile_method=PROFILE_METHOD,
            output_dir=str(args.output_dir),
        )
        for scenario, profile_input in profile_inputs:
            if isinstance(profile_input, AttentionInput):
                result = wrapper.profile(profile_input)
            else:
                result = wrapper.profile_mixed(profile_input)
            rows.append(
                _flatten_result(
                    result,
                    scenario=scenario,
                    allocation=allocation,
                    precision=precision,
                    model_config=model_config,
                )
            )

        del wrapper
        gc.collect()
        torch.cuda.empty_cache()

    frame = pd.DataFrame(rows).sort_values(
        ["num_tensor_parallel_workers", "total_tokens", "kv_cache_size", "batch_size"]
    )
    _validate_rows(frame, args.point_set)

    output_stem = _measurement_output_stem(args.point_set)
    csv_path = args.output_dir / f"{output_stem}.csv"
    frame.to_csv(csv_path, index=False)
    csv_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    requested_tuples = [
        [tp_size, total_tokens, kv_cache_size, batch_size]
        for tp_size in TP_SIZES
        for total_tokens, kv_cache_size, batch_size in _expected_keys(args.point_set)
    ]
    requested_tuple_sha256 = hashlib.sha256(
        json.dumps(
            requested_tuples,
            separators=(",", ":"),
            sort_keys=False,
        ).encode("utf-8")
    ).hexdigest()

    sidecar = {
        "schema_version": _measurement_schema_version(args.point_set),
        "point_set": args.point_set,
        "run_id": args.run_id,
        "command": sys.argv,
        "model": MODEL_NAME,
        "device": "h800",
        "tp_sizes": list(TP_SIZES),
        "measurement_type": profile_method_to_measurement_type(PROFILE_METHOD).value,
        "profile_method": PROFILE_METHOD,
        **profile_identity,
        "max_model_len": MAX_MODEL_LEN,
        "profile_max_seq_len": PROFILE_MAX_SEQ_LEN,
        "block_size": BLOCK_SIZE,
        "requested_tuples": requested_tuples,
        "requested_tuple_sha256": requested_tuple_sha256,
        "allocations_by_tp": allocations,
        "csv": csv_path.name,
        "csv_rows": len(frame),
        "csv_sha256": csv_sha256,
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flashinfer": importlib.metadata.version("flashinfer-python"),
            "vllm": importlib.metadata.version("vllm"),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "git_head": _git_value("rev-parse", "HEAD"),
            "git_branch": _git_value("branch", "--show-current"),
            "git_dirty": bool(_git_value("status", "--short")),
        },
    }
    sidecar_path = args.output_dir / f"{output_stem}.provenance.json"
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = frame[
        [
            "num_tensor_parallel_workers",
            "scenario",
            "total_tokens",
            "kv_cache_size",
            "batch_size",
            "time_stats.attn_kv_cache_save.count",
            "time_stats.attn_kv_cache_save.median",
        ]
    ].to_dict(orient="records")
    print(
        json.dumps(
            {
                "status": "PASS",
                "csv": str(csv_path),
                "sidecar": str(sidecar_path),
                "csv_sha256": csv_sha256,
                "rows": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
