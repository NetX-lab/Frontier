"""Artifact-backed CPU-safe prediction for GDN attention operators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Mapping

import numpy as np
import pandas as pd

from frontier.attention.gdn.features import GDNBatchFeatures
from frontier.entities.time_components import AttentionOperatorTimes, AttentionTime
from frontier.logger import init_logger
from frontier.types import MeasurementType


logger = init_logger(__name__)


def _dataset_fingerprint(path: str | Path) -> str:
    """Return the content identity used by the training manifest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        if not models:
            raise ValueError("GDNPredictor requires at least one trained estimator")
        self._models = dict(models)
        self.identity = dict(identity)
        self.measurement_type = _coerce_measurement_type(self.identity.get("measurement_type"))
        stack = str(self.identity.get("runtime_stack_signature", "")).strip()
        if not stack:
            raise ValueError("GDN model identity requires runtime_stack_signature")
        self.runtime_stack_signature = stack

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
        identity = dict(manifest.get("identity") or {})
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
            operator = str(task["operator"])
            phase = str(task["phase"])
            artifact = directory / str(task["artifact"])
            if not artifact.is_file():
                raise FileNotFoundError(f"GDN estimator artifact does not exist: {artifact}")
            with artifact.open("rb") as handle:
                estimator = pickle.load(handle)
            artifact_identity = getattr(estimator, "_frontier_gdn_identity", None)
            if artifact_identity != identity:
                raise ValueError(
                    f"GDN artifact identity mismatch for {task.get('task', artifact.name)}"
                )
            models[(operator, phase)] = estimator
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
            expected["dataset_fingerprint"] = _dataset_fingerprint(dataset_path)
        mismatches = {
            key: (identity.get(key), value)
            for key, value in expected.items()
            if str(identity.get(key)) != str(value)
        }
        if mismatches:
            raise ValueError(f"GDN model identity mismatch: {mismatches}")

    def _get_model(self, operator_name: str, phase: str) -> Any:
        try:
            return self._models[(operator_name, phase)]
        except KeyError as exc:
            raise ValueError(
                "GDN profile does not support the requested phase/operator: "
                f"operator={operator_name!r}, phase={phase!r}"
            ) from exc

    @staticmethod
    def _predict_one(estimator: Any, features: GDNBatchFeatures) -> float:
        feature_names = tuple(
            getattr(estimator, "_frontier_gdn_feature_names", ())
        )
        if not feature_names:
            raise ValueError("GDN estimator is missing feature identity metadata")
        values = features.feature_values
        vector = [float(values[name]) for name in feature_names]
        exact_lookup = getattr(estimator, "_frontier_gdn_exact_lookup", {})
        key = tuple(vector)
        if key in exact_lookup:
            return float(exact_lookup[key])
        bounds = getattr(estimator, "_frontier_gdn_feature_bounds", {})
        out_of_range = {
            name: (value, bounds[name])
            for name, value in zip(feature_names, vector)
            if name in bounds and (value < bounds[name][0] or value > bounds[name][1])
        }
        if out_of_range:
            logger.warning(
                "GDN numeric prediction is outside profiled feature bounds; "
                "using estimator extrapolation without clipping: %s",
                out_of_range,
            )
        prediction = float(
            estimator.predict(pd.DataFrame([vector], columns=list(feature_names)))[0]
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
        names = (
            "gdn_input_projections",
            "gdn_core_prefill" if phase == "prefill" else "gdn_core_decode",
            "gdn_output_projection",
        )
        return {
            name: self._predict_one(self._get_model(name, phase), features)
            for name in names
        }

    def predict_attention_time(
        self,
        batch: Any,
        *,
        norm_time_ms: float = 0.0,
    ) -> AttentionTime:
        features = GDNBatchFeatures.from_batch(batch)
        operator_times = self.predict_operator_times(features)
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
