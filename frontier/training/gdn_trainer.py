"""CPU-safe training for standard GDN profiling rows."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
from typing import Any, Dict, List

import pandas as pd

from frontier.attention.families import GATED_DELTA_NET_ATTENTION_FAMILY
from frontier.attention.gdn.features import GDNBatchFeatures, GDN_FEATURE_COLUMNS
from frontier.attention.profiling_mapping import validate_attention_profiling_dataframe
from frontier.logger import init_logger
from frontier.training.base_trainer import BaseTrainer
from frontier.types import MeasurementType


logger = init_logger(__name__)


GDN_TASKS = (
    ("gdn_input_projections", "prefill"),
    ("gdn_core_prefill", "prefill"),
    ("gdn_output_projection", "prefill"),
    ("gdn_input_projections", "decode"),
    ("gdn_core_decode", "decode"),
    ("gdn_output_projection", "decode"),
)


def _task_name(operator_name: str, phase: str) -> str:
    return f"{operator_name}_{phase}"


class GDNTrainer(BaseTrainer):
    """Train six phase-qualified GDN operator estimators.

    The trainer filters one identity-homogeneous CSV scope before fitting. It
    deliberately ignores ``gdn_layer_e2e`` and never merges CUDA_EVENT and
    DEVICE_EVENT rows or runtime stack signatures.
    """

    TASKS = GDN_TASKS

    def __init__(
        self,
        dataset_path: str,
        output_dir: str,
        *,
        model_name: str | None = None,
        model_config: Any | None = None,
        model_architecture_profile: str | None = None,
        quant_signature: str | None = None,
        device: str | None = None,
        tensor_parallel_size: int = 1,
        measurement_type: str | MeasurementType = MeasurementType.DEVICE_EVENT,
        runtime_stack_signature: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dataset_path,
            output_dir,
            predictor_type=kwargs.pop("predictor_type", "random_forest"),
            measurement_type=(
                measurement_type.value
                if isinstance(measurement_type, MeasurementType)
                else measurement_type
            ),
            **kwargs,
        )
        self.model_name = model_name
        self.model_architecture_profile = model_architecture_profile
        self.quant_signature = quant_signature
        self.device = device
        if model_config is not None:
            profile_getter = getattr(model_config, "get_model_architecture_profile", None)
            if self.model_architecture_profile is None and callable(profile_getter):
                self.model_architecture_profile = profile_getter().profile_id
            quant_getter = getattr(model_config, "get_quant_signature", None)
            if self.quant_signature is None and callable(quant_getter):
                self.quant_signature = quant_getter()
        self.tensor_parallel_size = int(tensor_parallel_size)
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        self.expected_measurement_type = (
            measurement_type
            if isinstance(measurement_type, MeasurementType)
            else MeasurementType.from_string(measurement_type)
        )
        self.runtime_stack_signature = runtime_stack_signature
        self.df: pd.DataFrame | None = None
        self.identity: dict[str, Any] = {}

    def _load_dataset(self) -> pd.DataFrame:
        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f"GDN dataset not found: {self.dataset_path}")
        frame = pd.read_csv(self.dataset_path)
        validate_attention_profiling_dataframe(
            frame,
            GATED_DELTA_NET_ATTENTION_FAMILY,
            measurement_type=self.expected_measurement_type,
        )
        frame = frame.copy()
        frame["measurement_type"] = frame["measurement_type"].map(
            lambda value: MeasurementType.from_string(value).value
        )
        selectors = {
            "measurement_type": self.expected_measurement_type.value,
            "device": self.device,
            "model_architecture_profile": self.model_architecture_profile,
            "quant_signature": self.quant_signature,
            "num_tensor_parallel_workers": self.tensor_parallel_size,
        }
        for column, expected in selectors.items():
            if expected is not None:
                if column not in frame.columns:
                    raise ValueError(f"GDN dataset is missing identity column {column!r}")
                frame = frame[frame[column].astype(str) == str(expected)]
        if frame.empty:
            raise ValueError(f"No GDN rows match requested identity: {selectors}")

        stack_values = tuple(sorted(str(value) for value in frame["runtime_stack_signature"].dropna().unique()))
        if len(stack_values) != 1:
            raise ValueError(
                "GDN profiling rows mix incompatible runtime contracts: "
                f"runtime_stack_signature={stack_values}"
            )
        if self.runtime_stack_signature is not None and stack_values[0] != self.runtime_stack_signature:
            raise ValueError(
                "GDN runtime_stack_signature mismatch: "
                f"expected={self.runtime_stack_signature!r}, actual={stack_values[0]!r}"
            )
        runtime_identity_columns = (
            "gdn_runtime_backend",
            "gdn_rank_aggregation",
            "gdn_prefill_backend",
            "gdn_decode_backend",
            "gqa_interleaved_layout",
            "packed_recurrent_decode",
            "model_dtype",
            "conv_state_dtype",
            "recurrent_state_dtype",
            "hidden_size",
            "conv_kernel_size",
            "key_head_dim",
            "value_head_dim",
            "num_key_heads",
            "num_value_heads",
        )
        for column in runtime_identity_columns:
            values = tuple(sorted(str(value) for value in frame[column].dropna().unique()))
            if len(values) != 1:
                raise ValueError(
                    "GDN profiling rows mix incompatible runtime contracts: "
                    f"{column}={values}"
                )
        self.identity = {
            "model_name": self.model_name,
            "model_architecture_profile": str(frame.iloc[0]["model_architecture_profile"]),
            "quant_signature": str(frame.iloc[0]["quant_signature"]),
            "device": str(frame.iloc[0]["device"]),
            "tensor_parallel_size": int(frame.iloc[0]["num_tensor_parallel_workers"]),
            "measurement_type": self.expected_measurement_type.value,
            "runtime_stack_signature": stack_values[0],
            "gdn_runtime_backend": str(frame.iloc[0]["gdn_runtime_backend"]),
            "gdn_rank_aggregation": str(frame.iloc[0]["gdn_rank_aggregation"]),
            "gdn_prefill_backend": str(frame.iloc[0]["gdn_prefill_backend"]),
            "gdn_decode_backend": str(frame.iloc[0]["gdn_decode_backend"]),
            **{
                column: (
                    int(frame.iloc[0][column])
                    if column in {
                        "hidden_size",
                        "conv_kernel_size",
                        "key_head_dim",
                        "value_head_dim",
                        "num_key_heads",
                        "num_value_heads",
                    }
                    else str(frame.iloc[0][column])
                )
                for column in runtime_identity_columns
            },
        }
        def materialize_features(row: pd.Series) -> pd.Series:
            features = GDNBatchFeatures.from_row(row)
            for name, value in features.feature_values.items():
                row[name] = value
            row["__gdn_phase"] = features.phase
            return row

        frame = frame.apply(materialize_features, axis=1)
        self.df = frame
        return frame

    def _get_model_names(self) -> List[str]:
        return [_task_name(operator, phase) for operator, phase in self.TASKS]

    def _get_feature_cols(self, model_name: str) -> List[str]:
        return list(GDN_FEATURE_COLUMNS)

    def _get_target_col(self, model_name: str) -> str:
        operator_name, phase = model_name.rsplit("_", 1)
        if phase not in {"prefill", "decode"}:
            raise ValueError(f"Invalid GDN task name: {model_name}")
        return f"time_stats.{operator_name}.median"

    def _create_estimator_and_params(self):
        estimator, grid = super()._create_estimator_and_params()
        if self.predictor_type == "random_forest":
            estimator.set_params(random_state=0, n_jobs=1)
        return estimator, grid

    def _attach_metadata(self, estimator: Any, frame: pd.DataFrame, *, task_name: str, feature_cols: list[str], target_col: str) -> Any:
        exact_lookup: dict[tuple[float, ...], float] = {}
        for _, row in frame.iterrows():
            features = GDNBatchFeatures.from_row(row)
            key = features.exact_key(tuple(feature_cols))
            target = float(row[target_col])
            previous = exact_lookup.get(key)
            if previous is not None and previous != target:
                raise ValueError(
                    f"Conflicting duplicate GDN rows for task={task_name}, key={key}"
                )
            exact_lookup[key] = target
        setattr(estimator, "_frontier_gdn_identity", dict(self.identity))
        setattr(estimator, "_frontier_gdn_task", task_name)
        setattr(estimator, "_frontier_gdn_feature_names", list(feature_cols))
        setattr(estimator, "_frontier_gdn_target_col", target_col)
        setattr(estimator, "_frontier_gdn_exact_lookup", exact_lookup)
        setattr(
            estimator,
            "_frontier_gdn_feature_bounds",
            {
                name: (float(frame[name].min()), float(frame[name].max()))
                for name in feature_cols
            },
        )
        return estimator

    def _train_task_estimator(
        self,
        task: str,
        frame: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> Any:
        """Train one task, including the one-row fixture case.

        A one-row CPU fixture has no meaningful cross-validation split. Fit the
        deterministic estimator directly with its first configured parameter
        value; normal multi-row datasets continue through ``BaseTrainer``'s
        cache/grid-search path.
        """

        if len(frame) != 1:
            return self._train_single_model(
                model_name=task,
                df=frame,
                feature_cols=feature_cols,
                target_col=target_col,
            )
        estimator, grid = self._create_estimator_and_params()
        selected = {
            name: values[0]
            for name, values in grid.items()
            if isinstance(values, (list, tuple)) and values
        }
        if selected:
            estimator.set_params(**selected)
        estimator.fit(frame[feature_cols], frame[target_col])
        return estimator

    def train(self) -> Dict[str, Any]:
        frame = self._load_dataset()
        models: Dict[str, Any] = {}
        manifest_tasks: list[dict[str, Any]] = []
        for operator_name, phase in self.TASKS:
            task = _task_name(operator_name, phase)
            task_frame = frame[frame["__gdn_phase"] == phase].copy()
            if task_frame.empty:
                raise ValueError(f"GDN dataset has no {phase} rows for task {task}")
            target_col = f"time_stats.{operator_name}.median"
            if target_col not in task_frame.columns:
                raise ValueError(f"GDN dataset is missing target {target_col}")
            # Each phase-qualified task receives only its physical phase rows;
            # gdn_layer_e2e is intentionally never referenced.
            estimator = self._train_task_estimator(
                task,
                task_frame,
                list(GDN_FEATURE_COLUMNS),
                target_col,
            )
            estimator = self._attach_metadata(
                estimator,
                task_frame,
                task_name=task,
                feature_cols=list(GDN_FEATURE_COLUMNS),
                target_col=target_col,
            )
            artifact_name = f"{task}.pkl"
            artifact_path = Path(self.output_dir) / artifact_name
            with artifact_path.open("wb") as handle:
                pickle.dump(estimator, handle, protocol=pickle.HIGHEST_PROTOCOL)
            models[task] = estimator
            manifest_tasks.append(
                {
                    "task": task,
                    "operator": operator_name,
                    "phase": phase,
                    "artifact": artifact_name,
                    "feature_names": list(GDN_FEATURE_COLUMNS),
                    "target_column": target_col,
                }
            )
        manifest = {
            "schema_version": 1,
            "identity": self.identity,
            "tasks": manifest_tasks,
        }
        with (Path(self.output_dir) / "gdn_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return models


__all__ = ["GDNTrainer", "GDN_TASKS"]
