"""Evaluate held-out KV-cache-save measurements against persisted predictors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd

from frontier.execution_time_predictor.model_cache_contract import (
    validate_cached_model,
)
from frontier.execution_time_predictor.prediction_cache_contract import (
    build_on_demand_prediction_record,
    classify_prediction_key,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.profiling.common.model_config import ModelConfig
from frontier.types import ClusterType, MeasurementType


MEASUREMENT_SCHEMA_VERSION = "frontier.attention.cache_save_held_out_measurement/v1"
MODEL_NAME = "Qwen3-30B-A3B-tiny"
OPERATOR_NAME = "attn_kv_cache_save"
TP_SIZES = (1, 2, 4, 8)
EXPECTED_KEYS = ((8, 0, 1), (16380, 0, 4))
TARGET_COLUMN = "time_stats.attn_kv_cache_save.median"


class _ConcretePredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):  # pragma: no cover - runtime-only evaluator
        raise AssertionError("the held-out evaluator must not train")

    def _get_grid_search_params(self):  # pragma: no cover
        raise AssertionError("the held-out evaluator must not train")


class _CountingEstimator:
    """Forward a persisted estimator while counting canonical model calls."""

    def __init__(self, estimator) -> None:
        self._estimator = estimator
        self.calls = 0
        self.__dict__.update(getattr(estimator, "__dict__", {}))

    def __getattr__(self, name):
        return getattr(self._estimator, name)

    def predict(self, frame):
        self.calls += 1
        return self._estimator.predict(frame)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-csv", type=Path, required=True)
    parser.add_argument("--measurement-sidecar", type=Path, required=True)
    parser.add_argument(
        "--artifact-root-template",
        required=True,
        help="Directory template containing one {tp} placeholder.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _requested_tuple_digest(tuples: list[list[int]]) -> str:
    return hashlib.sha256(
        json.dumps(tuples, separators=(",", ":"), sort_keys=False).encode("utf-8")
    ).hexdigest()


def _require_int(value: object, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {value!r}")
    return int(value)


def _validate_measurement_artifacts(
    csv_path: Path,
    sidecar_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"measurement CSV does not exist: {csv_path}")
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"measurement sidecar does not exist: {sidecar_path}"
        )

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("schema_version") != MEASUREMENT_SCHEMA_VERSION:
        raise ValueError(
            "held-out measurement schema mismatch: "
            f"{sidecar.get('schema_version')!r}"
        )
    expected_metadata = {
        "model": MODEL_NAME,
        "device": "h800",
        "measurement_type": "CUDA_EVENT",
        "max_model_len": 4096,
        "profile_max_seq_len": 4095,
        "block_size": 16,
    }
    for field, expected in expected_metadata.items():
        if sidecar.get(field) != expected:
            raise ValueError(
                f"held-out measurement {field} mismatch: "
                f"expected={expected!r}, actual={sidecar.get(field)!r}"
            )

    if sidecar.get("csv") != csv_path.name:
        raise ValueError(
            f"sidecar CSV name mismatch: {sidecar.get('csv')!r} != {csv_path.name!r}"
        )
    actual_csv_sha256 = _sha256(csv_path)
    if sidecar.get("csv_sha256") != actual_csv_sha256:
        raise ValueError(
            "held-out measurement CSV digest mismatch: "
            f"sidecar={sidecar.get('csv_sha256')!r}, actual={actual_csv_sha256!r}"
        )

    expected_tuples = [
        [tp, total_tokens, kv_cache_size, batch_size]
        for tp in TP_SIZES
        for total_tokens, kv_cache_size, batch_size in EXPECTED_KEYS
    ]
    if sidecar.get("tp_sizes") != list(TP_SIZES):
        raise ValueError(f"unexpected TP set: {sidecar.get('tp_sizes')!r}")
    if sidecar.get("requested_tuples") != expected_tuples:
        raise ValueError(
            "held-out requested tuple lattice mismatch: "
            f"expected={expected_tuples!r}, actual={sidecar.get('requested_tuples')!r}"
        )
    expected_tuple_digest = _requested_tuple_digest(expected_tuples)
    if sidecar.get("requested_tuple_sha256") != expected_tuple_digest:
        raise ValueError(
            "held-out requested tuple digest mismatch: "
            f"expected={expected_tuple_digest!r}, "
            f"actual={sidecar.get('requested_tuple_sha256')!r}"
        )

    frame = pd.read_csv(csv_path, low_memory=False)
    if sidecar.get("csv_rows") != len(frame) or len(frame) != len(expected_tuples):
        raise ValueError(
            "held-out row count mismatch: "
            f"sidecar={sidecar.get('csv_rows')!r}, actual={len(frame)}, "
            f"expected={len(expected_tuples)}"
        )
    required_columns = {
        "num_tensor_parallel_workers",
        "total_tokens",
        "kv_cache_size",
        "batch_size",
        TARGET_COLUMN,
        "time_stats.attn_kv_cache_save.count",
        "physical_max_num_blocks",
        "requested_max_num_blocks",
        "required_max_num_blocks",
        "selected_max_num_blocks",
        "allocated_max_num_blocks",
        "allocated_kv_token_capacity",
        "measurement_type",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"held-out measurement CSV is missing columns: {missing_columns!r}"
        )

    actual_tuples = sorted(
        [
            [
                int(row.num_tensor_parallel_workers),
                int(row.total_tokens),
                int(row.kv_cache_size),
                int(row.batch_size),
            ]
            for row in frame.itertuples(index=False)
        ]
    )
    if actual_tuples != sorted(expected_tuples):
        raise ValueError(
            "held-out emitted tuple lattice mismatch: "
            f"expected={sorted(expected_tuples)!r}, actual={actual_tuples!r}"
        )
    if set(frame["measurement_type"].astype(str)) != {"CUDA_EVENT"}:
        raise ValueError("held-out CSV mixes measurement families")

    allocations = sidecar.get("allocations_by_tp")
    if not isinstance(allocations, dict):
        raise ValueError("held-out sidecar allocations_by_tp must be a mapping")
    block_size = int(sidecar["block_size"])
    for tp in TP_SIZES:
        raw = allocations.get(str(tp))
        if not isinstance(raw, dict):
            raise ValueError(f"missing held-out allocation for TP={tp}")
        physical = _require_int(raw.get("physical"), label=f"TP={tp} physical", minimum=1)
        requested = _require_int(
            raw.get("requested"), label=f"TP={tp} requested", minimum=1
        )
        required = _require_int(
            raw.get("required"), label=f"TP={tp} required", minimum=1
        )
        selected = _require_int(
            raw.get("selected"), label=f"TP={tp} selected", minimum=1
        )
        _require_int(
            raw.get("workspace_bytes"),
            label=f"TP={tp} workspace_bytes",
            minimum=0,
        )
        if requested != selected or not required <= selected <= physical:
            raise ValueError(
                f"unsafe held-out allocation for TP={tp}: "
                f"required={required}, requested={requested}, "
                f"selected={selected}, physical={physical}"
            )

        tp_rows = frame[frame["num_tensor_parallel_workers"] == tp]
        if len(tp_rows) != len(EXPECTED_KEYS):
            raise ValueError(f"expected two held-out rows for TP={tp}")
        expected_row_values = {
            "physical_max_num_blocks": physical,
            "requested_max_num_blocks": requested,
            "required_max_num_blocks": required,
            "selected_max_num_blocks": selected,
            "allocated_max_num_blocks": selected,
            "allocated_kv_token_capacity": selected * block_size,
        }
        for column, expected in expected_row_values.items():
            values = set(pd.to_numeric(tp_rows[column], errors="raise").astype(int))
            if values != {expected}:
                raise ValueError(
                    f"TP={tp} {column} mismatch: expected={expected}, actual={values!r}"
                )

    for column in (TARGET_COLUMN, "time_stats.attn_kv_cache_save.count"):
        values = pd.to_numeric(frame[column], errors="raise")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise ValueError(f"held-out measurement contains non-finite {column}")
        if (values <= 0).any():
            raise ValueError(f"held-out measurement requires positive {column}")
    return frame, sidecar


def _artifact_path(root: Path) -> Path:
    matches = sorted(root.glob(f"{OPERATOR_NAME}_*.pkl"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {OPERATOR_NAME} artifact in {root}, got {matches}"
        )
    return matches[0]


def _runtime_predictor(model) -> tuple[_ConcretePredictor, _CountingEstimator]:
    counted = _CountingEstimator(model)
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._config = SimpleNamespace(
        enable_prediction_domain_diagnostics=True,
        prediction_max_tokens_per_request=4096,
        prediction_max_batch_size=10,
        prediction_max_prefill_chunk_size=16384,
        kv_cache_prediction_granularity=64,
        prediction_min_kv_cache_size=0,
    )
    predictor._model_config = ModelConfig.from_model_name(MODEL_NAME)
    predictor._max_tokens = 16384
    predictor._serving_max_tokens_per_request = 4096
    predictor._runtime_cache = defaultdict(lambda: defaultdict(dict))
    predictor._prediction_domain_diagnostics = defaultdict(
        lambda: defaultdict(dict)
    )
    predictor._models = {OPERATOR_NAME: counted}
    predictor._predictions = {
        OPERATOR_NAME: build_on_demand_prediction_record(
            OPERATOR_NAME,
            counted,
            counted._frontier_feature_names,
            exact_lookup=getattr(counted, "_frontier_exact_lookup", {}),
        )
    }
    return predictor, counted


def _evaluate_tp(
    *,
    tp: int,
    artifact_root: Path,
    measurement_rows: pd.DataFrame,
) -> dict[str, object]:
    path = _artifact_path(artifact_root)
    model = joblib.load(path)
    validate_cached_model(
        OPERATOR_NAME,
        model,
        expected_model_hash=model._frontier_model_hash,
        feature_names=model._frontier_feature_names,
        target_col=model._frontier_target_col,
        operator_binding=model._frontier_operator_binding,
    )
    binding = model._frontier_operator_binding
    profile_structure = binding["profile_structure"]
    expected_identity = {
        "model_name": MODEL_NAME,
        "device": "h800",
        "operator_family": "attention",
        "operator_name": OPERATOR_NAME,
        "measurement_type": "CUDA_EVENT",
        "tp": tp,
    }
    actual_identity = {
        "model_name": binding.get("model_name"),
        "device": binding.get("device"),
        "operator_family": binding.get("operator_family"),
        "operator_name": binding.get("operator_name"),
        "measurement_type": profile_structure.get("measurement_type"),
        "tp": profile_structure.get("num_tensor_parallel_workers"),
    }
    if actual_identity != expected_identity:
        raise ValueError(
            f"TP={tp} artifact identity mismatch: "
            f"expected={expected_identity!r}, actual={actual_identity!r}"
        )

    feature_names = tuple(str(name) for name in model._frontier_feature_names)
    if feature_names != ("total_tokens", "kv_cache_size", "batch_size"):
        raise ValueError(
            f"TP={tp} cache-save feature schema mismatch: {feature_names!r}"
        )
    predictor, counted = _runtime_predictor(model)
    results: list[dict[str, object]] = []
    for key in EXPECTED_KEYS:
        selected_rows = measurement_rows[
            (measurement_rows["total_tokens"] == key[0])
            & (measurement_rows["kv_cache_size"] == key[1])
            & (measurement_rows["batch_size"] == key[2])
        ]
        if len(selected_rows) != 1:
            raise ValueError(
                f"TP={tp} expected one held-out measurement for key={key!r}, "
                f"got {len(selected_rows)}"
            )
        actual_ms = float(selected_rows.iloc[0][TARGET_COLUMN])
        before = counted.calls
        prediction_ms = predictor._get_on_demand_prediction(
            OPERATOR_NAME,
            dict(zip(feature_names, key, strict=True)),
        )
        model_calls = counted.calls - before
        before_repeat = counted.calls
        repeated_ms = predictor._get_on_demand_prediction(
            OPERATOR_NAME,
            dict(zip(feature_names, key, strict=True)),
        )
        repeat_model_calls = counted.calls - before_repeat
        if model_calls != 1 or repeat_model_calls != 0:
            raise AssertionError(
                f"TP={tp} key={key!r} must call the model once then reuse runtime cache"
            )
        if float(repeated_ms) != float(prediction_ms):
            raise AssertionError(
                f"TP={tp} key={key!r} runtime-cache value changed on repeat"
            )

        classification = classify_prediction_key(
            model._frontier_feature_domain,
            key,
        )
        expected_classification = (
            "interpolation" if key == (8, 0, 1) else "extrapolation"
        )
        if classification["classification"] != expected_classification:
            raise AssertionError(
                f"TP={tp} key={key!r} classification mismatch: {classification!r}"
            )
        if not classification["sparse_gap"]:
            raise AssertionError(
                f"TP={tp} key={key!r} must be marked as a sparse gap"
            )

        absolute_error_ms = abs(float(prediction_ms) - actual_ms)
        relative_error_pct = absolute_error_ms / actual_ms * 100.0
        results.append(
            {
                "key": list(key),
                "prediction_ms": float(prediction_ms),
                "actual_median_ms": actual_ms,
                "absolute_error_ms": absolute_error_ms,
                "relative_error_pct": relative_error_pct,
                "classification": classification["classification"],
                "sparse_gap": bool(classification["sparse_gap"]),
                "outside_axes": classification["outside_axes"],
                "axis_gap": classification["axis_gap"],
                "first_model_calls": model_calls,
                "repeat_model_calls": repeat_model_calls,
                "runtime_cache_entries": len(
                    predictor._runtime_cache["eager"][OPERATOR_NAME]
                ),
            }
        )

    return {
        "artifact": str(path),
        "artifact_root_resolved": str(artifact_root.resolve()),
        "model_hash": model._frontier_model_hash,
        "identity": actual_identity,
        "feature_names": list(feature_names),
        "results": results,
        "diagnostics": predictor.get_prediction_domain_diagnostics(),
    }


def main() -> None:
    args = _parse_args()
    if "{tp}" not in args.artifact_root_template:
        raise ValueError("artifact-root-template must contain one {tp} placeholder")
    if args.output_json.exists():
        raise ValueError(f"output JSON must be absent: {args.output_json}")

    frame, sidecar = _validate_measurement_artifacts(
        args.measurement_csv,
        args.measurement_sidecar,
    )
    tp_results: dict[str, object] = {}
    for tp in TP_SIZES:
        tp_rows = frame[frame["num_tensor_parallel_workers"] == tp].copy()
        tp_results[str(tp)] = _evaluate_tp(
            tp=tp,
            artifact_root=Path(args.artifact_root_template.format(tp=tp)),
            measurement_rows=tp_rows,
        )

    result = {
        "status": "PASS",
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "pandas": pd.__version__,
        },
        "measurement": {
            "csv": str(args.measurement_csv),
            "sidecar": str(args.measurement_sidecar),
            "csv_sha256": sidecar["csv_sha256"],
            "rows": len(frame),
            "requested_tuple_sha256": sidecar["requested_tuple_sha256"],
            "allocations_by_tp": sidecar["allocations_by_tp"],
        },
        "tp_results": tp_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
