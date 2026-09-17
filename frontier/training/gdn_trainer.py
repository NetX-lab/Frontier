"""CPU-safe training for standard GDN profiling rows."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from frontier.attention.families import GATED_DELTA_NET_ATTENTION_FAMILY
from frontier.attention.gdn.features import (
    GDN_ARTIFACT_SCHEMA_VERSION,
    GDNBatchFeatures,
    GDN_FEATURE_COLUMNS,
    GDN_IDENTITY_COLUMNS,
    GDN_TASKS,
    validate_gdn_artifact_identity,
)
from frontier.attention.profiling_mapping import validate_attention_profiling_dataframe
from frontier.execution_time_predictor.cache_io import (
    atomic_json_dump,
    atomic_pickle_dump,
    dataset_fingerprint,
)
from frontier.logger import init_logger
from frontier.training.base_trainer import BaseTrainer
from frontier.types import MeasurementType


logger = init_logger(__name__)


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
            if self.model_architecture_profile is None:
                self.model_architecture_profile = (
                    model_config.get_model_architecture_profile().profile_id
                )
            if self.quant_signature is None:
                self.quant_signature = model_config.get_quant_signature()
        self.tensor_parallel_size = int(tensor_parallel_size)
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        self.expected_measurement_type = (
            measurement_type
            if isinstance(measurement_type, MeasurementType)
            else MeasurementType.from_string(measurement_type)
        )
        self.runtime_stack_signature = runtime_stack_signature
        self.identity: dict[str, Any] = {}

    def _load_dataset(self) -> pd.DataFrame:
        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f"GDN dataset not found: {self.dataset_path}")
        frame = pd.read_csv(self.dataset_path)
        validate_attention_profiling_dataframe(
            frame,
            GATED_DELTA_NET_ATTENTION_FAMILY,
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
            "runtime_stack_signature": self.runtime_stack_signature,
        }
        for column, expected in selectors.items():
            if expected is not None:
                if column not in frame.columns:
                    raise ValueError(f"GDN dataset is missing identity column {column!r}")
                frame = frame[frame[column].astype(str) == str(expected)]
        if frame.empty:
            raise ValueError(f"No GDN rows match requested identity: {selectors}")

        validate_attention_profiling_dataframe(
            frame,
            GATED_DELTA_NET_ATTENTION_FAMILY,
            measurement_type=self.expected_measurement_type,
            identity_columns=GDN_IDENTITY_COLUMNS,
        )
        selected_identity = {
            ("tensor_parallel_size" if column == "num_tensor_parallel_workers" else column):
            frame.iloc[0][column]
            for column in GDN_IDENTITY_COLUMNS
        }
        selected_identity.update(
            model_name=self.model_name,
            dataset_fingerprint=dataset_fingerprint(self.dataset_path),
        )
        self.identity = validate_gdn_artifact_identity(selected_identity)

        def materialize_features(row: pd.Series) -> pd.Series:
            features = GDNBatchFeatures.from_row(row)
            for name, value in features.feature_values.items():
                row[name] = value
            row["__gdn_phase"] = features.phase
            return row

        frame = frame.apply(materialize_features, axis=1)
        return frame

    def _get_model_names(self) -> List[str]:
        return [_task_name(operator, phase) for operator, phase in self.TASKS]

    def _get_feature_cols(self, model_name: str) -> List[str]:
        return list(GDN_FEATURE_COLUMNS)

    def _get_target_col(self, model_name: str) -> str:
        operator_name, phase = model_name.rsplit("_", 1)
        if (operator_name, phase) not in self.TASKS:
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
        """Fit one task using the BaseTrainer estimator and cache contract.

        One sample cannot support cross-validation. It is supported only when
        the caller specifies exactly one parameter configuration; choosing
        among multiple configurations requires at least two samples.
        """

        if len(frame) != 1:
            return self._train_single_model(
                model_name=task,
                df=frame,
                feature_cols=feature_cols,
                target_col=target_col,
            )
        estimator, grid = self._create_estimator_and_params()
        candidates = ParameterGrid(grid)
        if len(candidates) != 1:
            raise ValueError(
                "One-row GDN training requires exactly one parameter configuration; "
                "provide at least two rows for cross-validation"
            )
        model_hash = self._get_model_hash(task, frame)
        cached = self._load_model_from_cache(task, model_hash)
        if cached is not None:
            return cached
        estimator.set_params(**candidates[0])
        estimator.fit(frame[feature_cols], frame[target_col])
        self._store_model_in_cache(task, model_hash, estimator)
        return estimator

    def _store_model_in_cache(self, model_name: str, model_hash: str, model: Any) -> None:
        """Use complete-file publication for this trainer's estimator caches."""

        from fasteners import InterProcessReaderWriterLock

        with InterProcessReaderWriterLock(
            f"{self.output_dir}/{model_hash}_model_lock.file"
        ).write_lock():
            atomic_pickle_dump(model, Path(self.output_dir) / f"{model_name}_{model_hash}.pkl")

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
            targets = task_frame[target_col].to_numpy(dtype=float)
            if not np.isfinite(targets).all() or (targets < 0).any():
                raise ValueError(f"GDN dataset has invalid timings for {task}")
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
        # Fit and validate every task before exposing any new final artifact.
        for task in manifest_tasks:
            atomic_pickle_dump(models[task["task"]], Path(self.output_dir) / task["artifact"])
        manifest = {
            "schema_version": GDN_ARTIFACT_SCHEMA_VERSION,
            "identity": self.identity,
            "tasks": manifest_tasks,
        }
        atomic_json_dump(manifest, Path(self.output_dir) / "gdn_manifest.json")
        return models


__all__ = ["GDNTrainer", "GDN_TASKS"]
