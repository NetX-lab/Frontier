"""Artifact-backed CPU-safe prediction for GDN attention operators."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any, Mapping

import numpy as np
import pandas as pd

from frontier.attention.gdn.features import (
    GDN_ARTIFACT_SCHEMA_VERSION,
    GDNBatchFeatures,
    GDN_FEATURE_COLUMNS,
    GDN_TASKS,
    validate_gdn_artifact_identity,
)
from frontier.entities.time_components import AttentionOperatorTimes, AttentionTime
from frontier.execution_time_predictor.cache_io import dataset_fingerprint
from frontier.logger import init_logger
from frontier.types import MeasurementType


logger = init_logger(__name__)


@dataclass(frozen=True)
class _TaskModel:
    """A fitted task whose persisted metadata has passed boundary validation."""

    estimator: Any
    feature_names: tuple[str, ...]
    exact_lookup: Mapping[tuple[float, ...], float]
    feature_bounds: Mapping[str, tuple[float, float]]


def _coerce_measurement_type(value: str | MeasurementType) -> MeasurementType:
    return value if isinstance(value, MeasurementType) else MeasurementType.from_string(value)


class GDNPredictor:
    """Load six pre-trained phase-qualified GDN estimators.

    Prediction never fits, clips, rescales across TP, or substitutes another
    phase. Runtime identity is checked before any estimator is made available.
    """

    def __init__(
        self,
        models: Mapping[tuple[str, str], Any],
        *,
        identity: Mapping[str, Any],
    ) -> None:
        if set(models) != set(GDN_TASKS):
            raise ValueError("GDNPredictor requires the complete phase-qualified task set")
        self.identity = validate_gdn_artifact_identity(identity)
        self.measurement_type = _coerce_measurement_type(self.identity["measurement_type"])
        self.runtime_stack_signature = self.identity["runtime_stack_signature"]
        self._models = {
            key: self._validate_task_model(key, estimator)
            for key, estimator in models.items()
        }

    def _validate_task_model(self, key: tuple[str, str], estimator: Any) -> _TaskModel:
        operator, phase = key
        task_name = f"{operator}_{phase}"
        try:
            artifact_identity = estimator._frontier_gdn_identity
            artifact_task = estimator._frontier_gdn_task
            features = tuple(estimator._frontier_gdn_feature_names)
            target = estimator._frontier_gdn_target_col
            exact = estimator._frontier_gdn_exact_lookup
            bounds = estimator._frontier_gdn_feature_bounds
            fitted_features = tuple(estimator.feature_names_in_)
        except (AttributeError, TypeError) as exc:
            raise ValueError(f"GDN artifact has incomplete metadata for {task_name}") from exc
        if artifact_identity != self.identity:
            raise ValueError(f"GDN artifact identity mismatch for {task_name}")
        if (
            artifact_task != task_name
            or features != GDN_FEATURE_COLUMNS
            or fitted_features != features
            or target != f"time_stats.{operator}.median"
        ):
            raise ValueError(f"GDN artifact task/feature/target metadata mismatch for {task_name}")
        if not isinstance(exact, Mapping) or not isinstance(bounds, Mapping):
            raise ValueError(f"GDN artifact has invalid lookup metadata for {task_name}")
        if set(bounds) != set(features):
            raise ValueError(f"GDN artifact has incomplete feature bounds for {task_name}")
        try:
            normalized_bounds = {name: tuple(map(float, bounds[name])) for name in features}
            normalized_exact = {tuple(map(float, key)): float(value) for key, value in exact.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"GDN artifact has invalid numeric metadata for {task_name}") from exc
        if any(
            len(bound) != 2 or not np.isfinite(bound).all() or bound[0] > bound[1]
            for bound in normalized_bounds.values()
        ) or any(
            len(vector) != len(features) or not np.isfinite(vector).all()
            or not np.isfinite(value) or value < 0
            for vector, value in normalized_exact.items()
        ):
            raise ValueError(f"GDN artifact has invalid numeric metadata for {task_name}")
        return _TaskModel(estimator, features, normalized_exact, normalized_bounds)

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        model_config: Any | None = None,
        device: str | None = None,
        tensor_parallel_size: int | None = None,
        measurement_type: str | MeasurementType | None = None,
        runtime_stack_signature: str | None = None,
        model_architecture_profile: str | None = None,
        quant_signature: str | None = None,
        dataset_path: str | Path | None = None,
    ) -> "GDNPredictor":
        directory = Path(directory)
        manifest_path = directory / "gdn_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"GDN manifest does not exist: {manifest_path}")
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict) or manifest.get("schema_version") != GDN_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("Unsupported GDN manifest schema_version")
        if not isinstance(manifest.get("identity"), dict):
            raise ValueError("GDN manifest requires an identity object")
        identity = validate_gdn_artifact_identity(manifest["identity"])
        cls._validate_requested_identity(
            identity,
            model_config=model_config,
            device=device,
            tensor_parallel_size=tensor_parallel_size,
            measurement_type=measurement_type,
            runtime_stack_signature=runtime_stack_signature,
            model_architecture_profile=model_architecture_profile,
            quant_signature=quant_signature,
            dataset_path=dataset_path,
        )
        tasks = manifest.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("GDN manifest must contain a non-empty tasks list")
        models: dict[tuple[str, str], Any] = {}
        for task in tasks:
            if not isinstance(task, dict):
                raise ValueError("GDN manifest task must be an object")
            operator, phase = task.get("operator"), task.get("phase")
            if not isinstance(operator, str) or not isinstance(phase, str):
                raise ValueError("GDN manifest requires task operator and phase")
            key = (operator, phase)
            if key not in GDN_TASKS or key in models:
                raise ValueError(f"GDN manifest has unknown or duplicate task {key}")
            if (
                task.get("task") != f"{operator}_{phase}"
                or task.get("feature_names") != list(GDN_FEATURE_COLUMNS)
                or task.get("target_column") != f"time_stats.{operator}.median"
                or not isinstance(task.get("artifact"), str)
                or not task["artifact"]
            ):
                raise ValueError(f"GDN manifest task metadata mismatch for {key}")
            artifact = directory / task["artifact"]
            if not artifact.is_file():
                raise FileNotFoundError(f"GDN estimator artifact does not exist: {artifact}")
            with artifact.open("rb") as handle:
                estimator = pickle.load(handle)
            models[key] = estimator
        predictor = cls(models, identity=identity)
        return predictor

    @staticmethod
    def _validate_requested_identity(
        identity: Mapping[str, Any],
        *,
        model_config: Any | None,
        device: str | None,
        tensor_parallel_size: int | None,
        measurement_type: str | MeasurementType | None,
        runtime_stack_signature: str | None,
        model_architecture_profile: str | None,
        quant_signature: str | None,
        dataset_path: str | Path | None,
    ) -> None:
        expected: dict[str, Any] = {}
        if model_config is not None:
            profile_getter = getattr(model_config, "get_model_architecture_profile", None)
            if callable(profile_getter):
                expected["model_architecture_profile"] = profile_getter().profile_id
            quant_getter = getattr(model_config, "get_quant_signature", None)
            if callable(quant_getter):
                expected["quant_signature"] = quant_getter()
            embedding_dim = getattr(model_config, "embedding_dim", None)
            gdn_getter = getattr(model_config, "get_gdn_config", None)
            gdn_config = gdn_getter() if callable(gdn_getter) else None
            if embedding_dim is not None:
                expected["hidden_size"] = int(embedding_dim)
            if gdn_config is not None:
                expected.update(
                    {
                        "conv_kernel_size": int(gdn_config.conv_kernel_size),
                        "key_head_dim": int(gdn_config.key_head_dim),
                        "value_head_dim": int(gdn_config.value_head_dim),
                        "num_key_heads": int(gdn_config.num_key_heads),
                        "num_value_heads": int(gdn_config.num_value_heads),
                    }
                )
        if device is not None:
            expected["device"] = str(device)
        if tensor_parallel_size is not None:
            expected["tensor_parallel_size"] = int(tensor_parallel_size)
        if measurement_type is not None:
            expected["measurement_type"] = _coerce_measurement_type(measurement_type).value
        if runtime_stack_signature is not None:
            expected["runtime_stack_signature"] = runtime_stack_signature
        if model_architecture_profile is not None:
            expected["model_architecture_profile"] = model_architecture_profile
        if quant_signature is not None:
            expected["quant_signature"] = quant_signature
        if dataset_path is not None:
            expected["dataset_fingerprint"] = dataset_fingerprint(dataset_path)
        mismatches = {
            key: (identity.get(key), value)
            for key, value in expected.items()
            if str(identity.get(key)) != str(value)
        }
        if mismatches:
            raise ValueError(f"GDN model identity mismatch: {mismatches}")

    def _get_model(self, operator_name: str, phase: str) -> _TaskModel:
        try:
            return self._models[(operator_name, phase)]
        except KeyError as exc:
            raise ValueError(
                "GDN profile does not support the requested phase/operator: "
                f"operator={operator_name!r}, phase={phase!r}"
            ) from exc

    @staticmethod
    def _predict_one(model: _TaskModel, features: GDNBatchFeatures) -> float:
        feature_names = model.feature_names
        vector = features.as_vector(feature_names)
        key = tuple(vector)
        if key in model.exact_lookup:
            prediction = model.exact_lookup[key]
        else:
            bounds = model.feature_bounds
            out_of_range = {
                name: (value, bounds[name])
                for name, value in zip(feature_names, vector)
                if value < bounds[name][0] or value > bounds[name][1]
            }
            if out_of_range:
                logger.warning(
                    "GDN numeric prediction is outside profiled feature bounds; "
                    "using estimator extrapolation without clipping: %s",
                    out_of_range,
                )
            prediction = float(
                model.estimator.predict(pd.DataFrame([vector], columns=list(feature_names)))[0]
            )
        if not np.isfinite(prediction) or prediction < 0.0:
            raise ValueError(f"GDN estimator produced invalid timing {prediction!r}")
        return prediction

    def predict_operator_times(
        self, features: GDNBatchFeatures | Any
    ) -> dict[str, float]:
        if not isinstance(features, GDNBatchFeatures):
            features = GDNBatchFeatures.from_batch(features)
        phase = features.phase
        return {
            name: self._predict_one(self._get_model(name, phase), features)
            for name, task_phase in GDN_TASKS
            if task_phase == phase
        }

    def predict_attention_time(
        self,
        batch: Any,
        *,
        norm_time_ms: float = 0.0,
    ) -> AttentionTime:
        features = GDNBatchFeatures.from_batch(batch)
        operator_times = self.predict_operator_times(features)
        # Structured attention consumers enumerate the complete family schema.
        # The inactive phase contributes no work and must remain explicitly zero.
        operator_times.setdefault("gdn_core_prefill", 0.0)
        operator_times.setdefault("gdn_core_decode", 0.0)
        core_name = "gdn_core_prefill" if features.phase == "prefill" else "gdn_core_decode"
        return AttentionTime(
            attention_prefill_execution_time=(
                operator_times[core_name] if features.phase == "prefill" else 0.0
            ),
            attention_decode_execution_time=(
                operator_times[core_name] if features.phase == "decode" else 0.0
            ),
            attention_layer_pre_proj_execution_time=operator_times[
                "gdn_input_projections"
            ],
            attention_layer_post_proj_execution_time=operator_times[
                "gdn_output_projection"
            ],
            attn_norm_time=float(norm_time_ms),
            operator_times=AttentionOperatorTimes(operator_times),
        )


ProfiledGDNPredictor = GDNPredictor

__all__ = ["GDNPredictor", "ProfiledGDNPredictor"]
