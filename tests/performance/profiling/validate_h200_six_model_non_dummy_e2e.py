"""Validate the six-model H200 profiling and non-dummy E2E contract."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from frontier.moe_routing_runtime import (
    resolve_moe_gating_routing_runtime_path,
)
from frontier.operators.families import MOE_FAMILY, get_family_profiling_names
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan


_MODEL_PIPELINE_STAGES = {
    "llama3.1-8b": 1,
    "llama3.3-70b": 2,
    "Qwen3-235B-A22B": 2,
    "qwen3-a3b-30b-moe": 1,
    "step3-moe-noquant": 1,
    "mixtral_8x7b_moe": 1,
}

SUPPORTED_MODELS = tuple(_MODEL_PIPELINE_STAGES)

_PROFILE_MEASUREMENT_TYPES = {
    "linear_op.csv": "CUDA_EVENT",
    "linear_op_kernel_only.csv": "KERNEL_ONLY",
    "attention.csv": "CUDA_EVENT",
    "attention_kernel_only.csv": "KERNEL_ONLY",
    "moe.csv": "CUDA_EVENT",
    "moe_kernel_only.csv": "KERNEL_ONLY",
}

_PROFILE_CONFIG_FIELDS = {
    "linear_op.csv": "linear_op_input_file",
    "linear_op_kernel_only.csv": "linear_op_kernel_only_input_file",
    "attention.csv": "atten_input_file",
    "attention_kernel_only.csv": "atten_kernel_only_input_file",
    "moe.csv": "moe_input_file",
    "moe_kernel_only.csv": "moe_kernel_only_input_file",
}


@dataclass(frozen=True)
class ModelContract:
    model_name: str
    is_moe: bool
    is_mixed_layer_moe: bool
    num_layers: int
    num_pipeline_stages: int
    dense_layer_count: int
    moe_layer_count: int
    profile_filenames: tuple[str, ...]
    linear_target_columns: tuple[str, ...]
    moe_target_columns: tuple[str, ...]
    moe_routing_mode: str
    routing_runtime_path: str
    profiling_precision: str
    quant_signature: str
    model_arch: str
    model_architecture_profile: str
    num_q_heads: int
    num_kv_heads: int
    embedding_dim: int
    mlp_hidden_dim: int
    vocab_size: int
    use_gated_mlp: bool
    use_qk_norm: bool
    attn_output_gate: bool
    share_expert_dim: int | None
    share_q_dim: int | None
    num_experts: int
    router_topk: int


def build_model_contract(
    model_name: str,
    *,
    moe_routing_mode: str = "simulation",
) -> ModelContract:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model {model_name!r}; expected one of {SUPPORTED_MODELS}."
        )

    model_config = ModelConfig.from_model_name(model_name)
    num_pipeline_stages = _MODEL_PIPELINE_STAGES[model_name]
    if int(model_config.num_layers) % num_pipeline_stages != 0:
        raise ValueError(
            f"Model runtime contract for {model_name!r} requires PP"
            f"{num_pipeline_stages}, but num_layers={model_config.num_layers} "
            "is not evenly divisible."
        )
    is_moe = bool(model_config.is_moe)
    moe_layer_count = len(model_config.get_moe_layer_ids())
    dense_layer_count = int(model_config.num_layers) - moe_layer_count
    is_mixed_layer_moe = is_moe and dense_layer_count > 0

    profiling_plan = build_profiling_plan(
        model_config=model_config,
        tp_size=1,
        attn_tp=(1,),
        ffn_tp=(1,),
        is_moe=is_moe and not is_mixed_layer_moe,
    )
    enabled_ops = list(profiling_plan["enabled_ops"])
    if model_config.uses_fused_add_norm:
        enabled_ops = [op_name for op_name in enabled_ops if op_name != "add"]

    profile_filenames = (
        "linear_op.csv",
        "linear_op_kernel_only.csv",
        "attention.csv",
        "attention_kernel_only.csv",
    )
    if is_moe:
        profile_filenames += ("moe.csv", "moe_kernel_only.csv")

    return ModelContract(
        model_name=model_name,
        is_moe=is_moe,
        is_mixed_layer_moe=is_mixed_layer_moe,
        num_layers=int(model_config.num_layers),
        num_pipeline_stages=num_pipeline_stages,
        dense_layer_count=dense_layer_count,
        moe_layer_count=moe_layer_count,
        profile_filenames=profile_filenames,
        linear_target_columns=tuple(
            f"time_stats.{op_name}.median" for op_name in enabled_ops
        ),
        moe_target_columns=tuple(
            f"time_stats.{op_name}.median"
            for op_name in get_family_profiling_names(MOE_FAMILY)
        ),
        moe_routing_mode=moe_routing_mode,
        routing_runtime_path=resolve_moe_gating_routing_runtime_path(
            moe_routing_mode
        ),
        profiling_precision=ModelConfig._dtype_to_str(model_config.dtype),
        quant_signature=model_config.get_quant_signature(),
        model_arch=str(model_config.model_arch),
        model_architecture_profile=(
            model_config.get_model_architecture_profile().profile_id
        ),
        num_q_heads=int(model_config.num_q_heads),
        num_kv_heads=int(model_config.num_kv_heads),
        embedding_dim=int(model_config.embedding_dim),
        mlp_hidden_dim=int(model_config.mlp_hidden_dim),
        vocab_size=int(model_config.vocab_size),
        use_gated_mlp=bool(model_config.use_gated_mlp),
        use_qk_norm=bool(model_config.use_qk_norm),
        attn_output_gate=bool(model_config.attn_output_gate),
        share_expert_dim=(
            None
            if model_config.share_expert_dim is None
            else int(model_config.share_expert_dim)
        ),
        share_q_dim=(
            None
            if model_config.share_q_dim is None
            else int(model_config.share_q_dim)
        ),
        num_experts=int(model_config.num_experts),
        router_topk=int(model_config.num_experts_per_tok),
    )


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required profiling CSV is missing: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Profiling CSV is empty: {path}")
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Profiling CSV is not parseable: {path}: {exc}") from exc
    if frame.empty:
        raise ValueError(f"Profiling CSV has no data rows: {path}")
    return frame


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        if value in (0, 1):
            return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}.")


def _require_exact_column(
    frame: pd.DataFrame,
    *,
    path: Path,
    column: str,
    expected: Any,
) -> None:
    if column not in frame.columns:
        raise ValueError(f"{path.name} is missing required column {column!r}.")
    series = frame[column]
    if expected is None:
        if not series.isna().all():
            values = sorted({str(value) for value in series.dropna().unique()})
            raise ValueError(
                f"{path.name} column {column!r} must be empty, got {values}."
            )
        return
    if isinstance(expected, bool):
        try:
            values = {_normalize_bool(value) for value in series}
        except ValueError as exc:
            raise ValueError(
                f"{path.name} column {column!r} contains an invalid boolean."
            ) from exc
        if values != {expected}:
            raise ValueError(
                f"{path.name} column {column!r} must equal {expected}, got {values}."
            )
        return
    if isinstance(expected, int):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any() or not (numeric == expected).all():
            values = sorted({str(value) for value in series.unique()})
            raise ValueError(
                f"{path.name} column {column!r} must equal {expected}, got {values}."
            )
        return
    values = {str(value).strip() for value in series}
    if values != {str(expected)}:
        raise ValueError(
            f"{path.name} column {column!r} must equal {expected!r}, got {values}."
        )


def _require_positive_target(
    frame: pd.DataFrame,
    *,
    path: Path,
    column: str,
    mask: pd.Series | None = None,
) -> float:
    if column not in frame.columns:
        raise ValueError(f"{path.name} is missing required target column {column!r}.")
    values = pd.to_numeric(frame[column], errors="coerce")
    if mask is not None:
        values = values[mask]
    finite_positive = values.map(
        lambda value: pd.notna(value) and math.isfinite(float(value)) and float(value) > 0
    )
    if values.empty or not finite_positive.any():
        raise ValueError(
            f"{path.name} target {column!r} has no finite positive training row."
        )
    return float(values[finite_positive].min())


def _validate_common_metadata(
    frame: pd.DataFrame,
    *,
    path: Path,
    contract: ModelContract,
    measurement_type: str,
) -> None:
    expected = {
        "measurement_type": measurement_type,
        "profiling_precision": contract.profiling_precision,
        "quant_signature": contract.quant_signature,
        "model_arch": contract.model_arch,
        "model_architecture_profile": contract.model_architecture_profile,
    }
    for column, value in expected.items():
        _require_exact_column(frame, path=path, column=column, expected=value)


def _validate_linear_profile(
    frame: pd.DataFrame,
    *,
    path: Path,
    contract: ModelContract,
) -> dict[str, Any]:
    expected = {
        "n_head": contract.num_q_heads,
        "n_kv_head": contract.num_kv_heads,
        "n_embd": contract.embedding_dim,
        "n_expanded_embd": contract.mlp_hidden_dim,
        "vocab_size": contract.vocab_size,
        "use_gated_mlp": contract.use_gated_mlp,
        "use_qk_norm": contract.use_qk_norm,
        "attn_output_gate": contract.attn_output_gate,
        "num_tensor_parallel_workers": 1,
        "padded_n_embd": contract.embedding_dim,
        "padded_n_expanded_embd": contract.mlp_hidden_dim,
        "share_expert_dim": contract.share_expert_dim,
        "share_q_dim": contract.share_q_dim,
    }
    for column, value in expected.items():
        _require_exact_column(frame, path=path, column=column, expected=value)
    target_minima = {
        column: _require_positive_target(frame, path=path, column=column)
        for column in contract.linear_target_columns
    }
    return {
        "target_count": len(target_minima),
        "minimum_positive_target_ms": min(target_minima.values()),
    }


def _validate_attention_profile(
    frame: pd.DataFrame,
    *,
    path: Path,
    contract: ModelContract,
    measurement_type: str,
) -> dict[str, Any]:
    expected = {
        "n_embd": contract.embedding_dim,
        "n_q_head": contract.num_q_heads,
        "n_kv_head": contract.num_kv_heads,
        "block_size": 16,
        "num_tensor_parallel_workers": 1,
    }
    for column, value in expected.items():
        _require_exact_column(frame, path=path, column=column, expected=value)
    if "is_prefill" not in frame.columns:
        raise ValueError(f"{path.name} is missing required column 'is_prefill'.")
    try:
        is_prefill = frame["is_prefill"].map(_normalize_bool)
    except ValueError as exc:
        raise ValueError(f"{path.name} has an invalid is_prefill value.") from exc
    decode_min = _require_positive_target(
        frame,
        path=path,
        column="time_stats.attn_decode.median",
        mask=~is_prefill,
    )
    report = {
        "decode_row_count": int((~is_prefill).sum()),
        "minimum_positive_decode_ms": decode_min,
    }
    if measurement_type == "CUDA_EVENT":
        prefill_min = _require_positive_target(
            frame,
            path=path,
            column="time_stats.attn_prefill.median",
            mask=is_prefill,
        )
        report.update(
            {
                "prefill_row_count": int(is_prefill.sum()),
                "minimum_positive_prefill_ms": prefill_min,
            }
        )
    return report


def _validate_moe_profile(
    frame: pd.DataFrame,
    *,
    path: Path,
    contract: ModelContract,
) -> dict[str, Any]:
    if contract.num_experts % 2:
        raise ValueError(
            f"{contract.model_name} num_experts={contract.num_experts} is not divisible by EP=2."
        )
    expected = {
        "num_experts": contract.num_experts,
        "num_experts_per_device": contract.num_experts // 2,
        "expert_parallel_size": 2,
        "routing_runtime_path": contract.routing_runtime_path,
        "gating_runtime_context": "direct",
        "router_topk": contract.router_topk,
        "hidden_dim": contract.embedding_dim,
        "expert_hidden_dim": contract.mlp_hidden_dim,
        "use_gated": contract.use_gated_mlp,
        "num_tensor_parallel_workers": 1,
    }
    for column, value in expected.items():
        _require_exact_column(frame, path=path, column=column, expected=value)
    target_minima = {
        column: _require_positive_target(frame, path=path, column=column)
        for column in contract.moe_target_columns
    }
    return {
        "routing_runtime_path": contract.routing_runtime_path,
        "target_count": len(target_minima),
        "minimum_positive_target_ms": min(target_minima.values()),
    }


def validate_profile_directory(
    profile_dir: str | Path,
    contract: ModelContract,
) -> dict[str, Any]:
    profile_path = Path(profile_dir).resolve()
    if not profile_path.is_dir():
        raise FileNotFoundError(
            f"Profiling directory is missing for {contract.model_name}: {profile_path}"
        )

    file_reports: dict[str, Any] = {}
    for filename in contract.profile_filenames:
        path = profile_path / filename
        measurement_type = _PROFILE_MEASUREMENT_TYPES[filename]
        frame = _load_csv(path)
        _validate_common_metadata(
            frame,
            path=path,
            contract=contract,
            measurement_type=measurement_type,
        )
        if filename.startswith("linear_op"):
            details = _validate_linear_profile(
                frame,
                path=path,
                contract=contract,
            )
        elif filename.startswith("attention"):
            details = _validate_attention_profile(
                frame,
                path=path,
                contract=contract,
                measurement_type=measurement_type,
            )
        else:
            details = _validate_moe_profile(
                frame,
                path=path,
                contract=contract,
            )
        file_reports[filename] = {
            "path": str(path),
            "row_count": int(len(frame)),
            "measurement_type": measurement_type,
            **details,
        }

    return {
        "status": "PASS",
        "model": contract.model_name,
        "profile_dir": str(profile_path),
        "files": file_reports,
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required runtime artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Runtime artifact is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Runtime JSON artifact must contain an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required runtime artifact is missing: {path}")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Runtime JSONL artifact is invalid at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"Runtime JSONL row must contain an object at {path}:{line_number}."
            )
        rows.append(value)
    if not rows:
        raise ValueError(f"Runtime JSONL artifact has no rows: {path}")
    return rows


def _finite_positive(value: Any, *, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}.") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{label} must be finite and positive, got {numeric}.")
    return numeric


def _validate_runtime_config(
    config: Mapping[str, Any],
    *,
    contract: ModelContract,
    profile_dir: Path,
) -> dict[str, Any]:
    expected_top_level = {
        "simulation_mode": "offline",
        "sys_arch": "co-location",
        "decode_cuda_graph_mode": "full_decode_only",
    }
    for field, expected in expected_top_level.items():
        actual = config.get(field)
        if actual != expected:
            raise ValueError(
                f"config.json field {field!r} must equal {expected!r}, got {actual!r}."
            )

    request_generator_config = config.get("request_generator_config")
    if not isinstance(request_generator_config, dict):
        raise ValueError("config.json must contain request_generator_config.")
    length_generator_config = request_generator_config.get("length_generator_config")
    if not isinstance(length_generator_config, dict):
        raise ValueError(
            "config.json request_generator_config must contain length_generator_config."
        )
    expected_request = {
        "generator_name": "synthetic",
        "num_requests": 1,
        "length_generator_name": "fixed",
        "prefill_tokens": 2,
        "decode_tokens": 2,
    }
    actual_request = {
        "generator_name": request_generator_config.get("name"),
        "num_requests": request_generator_config.get("num_requests"),
        "length_generator_name": length_generator_config.get("name"),
        "prefill_tokens": length_generator_config.get("prefill_tokens"),
        "decode_tokens": length_generator_config.get("decode_tokens"),
    }
    if actual_request != expected_request:
        raise ValueError(
            "config.json request generator must equal "
            f"{expected_request}, got {actual_request}."
        )

    cluster_config = config.get("cluster_config")
    if not isinstance(cluster_config, dict):
        raise ValueError("config.json must contain cluster_config.")
    replica_config = cluster_config.get("replica_config")
    predictor_config = cluster_config.get("execution_time_predictor_config")
    if not isinstance(replica_config, dict) or not isinstance(predictor_config, dict):
        raise ValueError(
            "config.json must contain replica_config and execution_time_predictor_config."
        )
    if predictor_config.get("enable_dummy_mode") is not False:
        raise ValueError(
            "config.json execution_time_predictor_config must set enable_dummy_mode=false."
        )

    expected_replica = {
        "model_name": contract.model_name,
        "device": "h200",
        "num_pipeline_stages": contract.num_pipeline_stages,
        "attn_tensor_parallel_size": 1,
        "attn_data_parallel_size": 2 if contract.is_moe else 1,
        "data_parallel_size": 2 if contract.is_moe else 1,
    }
    if contract.is_moe:
        expected_replica.update(
            {
                "moe_tensor_parallel_size": 1,
                "moe_expert_parallel_size": 2,
                "total_expert_num": contract.num_experts,
                "router_topk": contract.router_topk,
                "moe_routing_mode": contract.moe_routing_mode,
            }
        )
    for field, expected in expected_replica.items():
        actual = replica_config.get(field)
        if actual != expected:
            raise ValueError(
                f"config.json replica_config field {field!r} must equal "
                f"{expected!r}, got {actual!r}."
            )

    profile_files: dict[str, str] = {}
    for filename in contract.profile_filenames:
        config_field = _PROFILE_CONFIG_FIELDS[filename]
        actual_value = predictor_config.get(config_field)
        if not isinstance(actual_value, str) or not actual_value.strip():
            raise ValueError(
                f"config.json predictor field {config_field!r} must be explicit."
            )
        expected_path = (profile_dir / filename).resolve()
        actual_path = Path(actual_value).expanduser().resolve()
        if actual_path != expected_path:
            raise ValueError(
                f"config.json predictor field {config_field!r} must equal "
                f"{expected_path}, got {actual_path}."
            )
        profile_files[config_field] = str(actual_path)

    return {
        "simulation_mode": "offline",
        "sys_arch": "co-location",
        "decode_cuda_graph_mode": "full_decode_only",
        "request": expected_request,
        "replica": expected_replica,
        "profile_files": profile_files,
    }


def validate_runtime_artifacts(
    run_dir: str | Path,
    contract: ModelContract,
    *,
    profile_dir: str | Path,
) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(
            f"Runtime output directory is missing for {contract.model_name}: {run_path}"
        )
    profile_path = Path(profile_dir).resolve()

    system_metrics = _load_json(run_path / "system_metrics.json")
    simulation_metadata = system_metrics.get("simulation_metadata")
    if not isinstance(simulation_metadata, dict):
        raise ValueError("system_metrics.json is missing simulation_metadata.")
    total_requests = simulation_metadata.get("total_requests")
    completed_requests = simulation_metadata.get("completed_requests")
    system_architecture = simulation_metadata.get("system_architecture")
    if total_requests != 1 or completed_requests != 1:
        raise ValueError(
            "system_metrics.json must report total_requests=1 and "
            f"completed_requests=1, got {total_requests}/{completed_requests}."
        )
    if system_architecture != "co-location":
        raise ValueError(
            "system_metrics.json must report system_architecture='co-location', "
            f"got {system_architecture!r}."
        )

    request_metrics_path = run_path / "request_metrics.csv"
    request_metrics = _load_csv(request_metrics_path)
    if len(request_metrics) != 1:
        raise ValueError(
            f"request_metrics.csv must contain exactly one row, got {len(request_metrics)}."
        )
    row = request_metrics.iloc[0]
    for field, expected in (
        ("request_num_prefill_tokens", 2),
        ("request_num_decode_tokens", 2),
    ):
        if field not in row or int(row[field]) != expected:
            raise ValueError(
                f"request_metrics.csv field {field!r} must equal {expected}."
            )
    ttft_ms = _finite_positive(row.get("ttft"), label="request_metrics.csv ttft")
    tpot_ms = _finite_positive(row.get("tpot"), label="request_metrics.csv tpot")
    e2e_ms = _finite_positive(
        row.get("request_e2e_time"),
        label="request_metrics.csv request_e2e_time",
    )
    expected_e2e_ms = ttft_ms + tpot_ms
    if not math.isclose(e2e_ms, expected_e2e_ms, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            "request_metrics.csv must satisfy E2E == TTFT + TPOT for decode_tokens=2: "
            f"e2e={e2e_ms}, ttft={ttft_ms}, tpot={tpot_ms}."
        )

    ledger_rows = _load_jsonl(run_path / "frontier_stage_batch_ledger.jsonl")
    if len(ledger_rows) < 2:
        raise ValueError(
            "frontier_stage_batch_ledger.jsonl must contain at least two rows."
        )
    ledger_first_ms = _finite_positive(
        ledger_rows[0].get("execution_time", {}).get("model_time_ms"),
        label="first ledger execution_time.model_time_ms",
    )
    ledger_last_ms = _finite_positive(
        ledger_rows[-1].get("execution_time", {}).get("model_time_ms"),
        label="last ledger execution_time.model_time_ms",
    )

    trace_rows = _load_jsonl(run_path / "op_traces.jsonl")
    if "meta" not in trace_rows[0]:
        raise ValueError("op_traces.jsonl first row must contain metadata.")
    trace_events = trace_rows[1:]
    if not trace_events:
        raise ValueError("op_traces.jsonl must contain at least one event.")
    event_durations = [
        _finite_positive(
            event.get("duration_ms"),
            label=f"op_traces.jsonl event {index} duration_ms",
        )
        for index, event in enumerate(trace_events, start=1)
    ]

    config_report = _validate_runtime_config(
        _load_json(run_path / "config.json"),
        contract=contract,
        profile_dir=profile_path,
    )

    return {
        "status": "PASS",
        "model": contract.model_name,
        "run_dir": str(run_path),
        "system_metrics": {
            "total_requests": 1,
            "completed_requests": 1,
            "system_architecture": "co-location",
        },
        "request_metrics": {
            "row_count": 1,
            "prefill_tokens": 2,
            "decode_tokens": 2,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "e2e_ms": e2e_ms,
        },
        "ledger": {
            "row_count": len(ledger_rows),
            "first_model_time_ms": ledger_first_ms,
            "last_model_time_ms": ledger_last_ms,
        },
        "op_trace": {
            "event_count": len(trace_events),
            "minimum_duration_ms": min(event_durations),
            "maximum_duration_ms": max(event_durations),
        },
        "config": config_report,
    }


def _write_report(report: Mapping[str, Any], output_path: str | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate registry-derived H200 profiles and matching non-dummy E2E runs."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    contract_parser = subparsers.add_parser(
        "contract",
        help="Print one registry-derived model contract.",
    )
    contract_parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    contract_parser.add_argument("--moe-routing-mode", default="simulation")

    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Validate one model profile directory.",
    )
    preflight_parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    preflight_parser.add_argument("--profile-dir", required=True)
    preflight_parser.add_argument("--moe-routing-mode", default="simulation")
    preflight_parser.add_argument("--report-json")

    runtime_parser = subparsers.add_parser(
        "validate-run",
        help="Validate one model runtime output directory.",
    )
    runtime_parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    runtime_parser.add_argument("--run-dir", required=True)
    runtime_parser.add_argument("--profile-dir", required=True)
    runtime_parser.add_argument("--moe-routing-mode", default="simulation")
    runtime_parser.add_argument("--report-json")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    contract = build_model_contract(
        args.model,
        moe_routing_mode=args.moe_routing_mode,
    )
    if args.command == "contract":
        _write_report(asdict(contract), None)
        return 0
    if args.command == "preflight":
        report = validate_profile_directory(args.profile_dir, contract)
        _write_report(report, args.report_json)
        return 0
    if args.command == "validate-run":
        report = validate_runtime_artifacts(
            args.run_dir,
            contract,
            profile_dir=args.profile_dir,
        )
        _write_report(report, args.report_json)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
