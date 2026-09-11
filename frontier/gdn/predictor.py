"""Profile-backed execution-time prediction for gated-delta-network layers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from frontier.entities.time_components import AttentionOperatorTimes, AttentionTime
from frontier.gdn.family import GATED_DELTA_NET_FAMILY
from frontier.gdn.profiling_schema import validate_gdn_profiling_dataframe
from frontier.types import MeasurementType


_FEATURE_COLUMNS = (
    "batch_size",
    "batch_num_tokens",
    "batch_num_prefill_tokens",
    "batch_num_decode_tokens",
    "max_query_len",
    "has_initial_state",
)


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{field_name} must contain boolean values, got {value!r}")


@dataclass(frozen=True)
class GDNBatchFeatures:
    """Runtime features shared by the profiler and prediction path."""

    batch_size: int
    batch_num_tokens: int
    batch_num_prefill_tokens: int
    batch_num_decode_tokens: int
    max_query_len: int
    has_initial_state: bool

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("GDN prediction batch_size must be positive")
        if self.batch_num_tokens <= 0:
            raise ValueError("GDN prediction batch_num_tokens must be positive")
        if self.batch_num_prefill_tokens < 0 or self.batch_num_decode_tokens < 0:
            raise ValueError("GDN prediction token counts must be non-negative")
        if (
            self.batch_num_prefill_tokens + self.batch_num_decode_tokens
            != self.batch_num_tokens
        ):
            raise ValueError("GDN prefill/decode token counts must sum to total tokens")
        if self.max_query_len <= 0:
            raise ValueError("GDN prediction max_query_len must be positive")

    @property
    def phase(self) -> str:
        if self.batch_num_prefill_tokens and self.batch_num_decode_tokens:
            return "mixed"
        if self.batch_num_prefill_tokens:
            return "prefill"
        return "decode"

    def as_vector(self) -> np.ndarray:
        # Log scaling keeps token and batch axes useful on the wide profiling
        # grids used by inference workloads while preserving exact row lookup.
        return np.asarray(
            [
                np.log1p(self.batch_size),
                np.log1p(self.batch_num_tokens),
                np.log1p(self.batch_num_prefill_tokens),
                np.log1p(self.batch_num_decode_tokens),
                np.log1p(self.max_query_len),
                float(self.has_initial_state),
            ],
            dtype=np.float64,
        )

    def exact_key(self) -> tuple[int, int, int, int, int, bool]:
        return (
            self.batch_size,
            self.batch_num_tokens,
            self.batch_num_prefill_tokens,
            self.batch_num_decode_tokens,
            self.max_query_len,
            self.has_initial_state,
        )

    @classmethod
    def from_batch(cls, batch: Any) -> "GDNBatchFeatures":
        query_lens = tuple(int(value) for value in batch.num_tokens)
        requests = tuple(batch.requests)
        if not requests or len(query_lens) != len(requests):
            raise ValueError(
                "GDN prediction requires one scheduled query length per request"
            )
        batch_size = len(requests)
        prefill_tokens = int(batch.num_prefill_tokens)
        decode_tokens = int(batch.num_decode_tokens)
        metadata = getattr(batch, "decode_cuda_graph_metadata", None)
        if metadata is not None and metadata.runtime_mode == "FULL":
            # Packed recurrent kernels execute every captured graph lane, even
            # when only a subset represents real requests. Match the profiled
            # physical shape while leaving request progress/padding untouched.
            if prefill_tokens:
                raise ValueError("Full decode GDN graphs cannot contain prefill tokens")
            batch_size = int(metadata.padded_decode_batch_size)
            decode_tokens = int(metadata.padded_total_tokens)
            if batch_size < len(requests) or decode_tokens != batch_size:
                raise ValueError("Invalid padded shape for non-speculative GDN decode")
        return cls(
            batch_size=batch_size,
            batch_num_tokens=prefill_tokens + decode_tokens,
            batch_num_prefill_tokens=prefill_tokens,
            batch_num_decode_tokens=decode_tokens,
            max_query_len=max(query_lens),
            has_initial_state=any(
                int(getattr(request, "num_processed_tokens", 0)) > 0
                or bool(getattr(request, "is_prefill_complete", False))
                for request in requests
            ),
        )

    def batch_scaling_key(self) -> tuple[int, bool, float, float]:
        """Only interpolate batches with identical per-request work/state."""
        return (self.max_query_len, self.has_initial_state,
                self.batch_num_prefill_tokens / self.batch_size,
                self.batch_num_decode_tokens / self.batch_size)


@dataclass
class _OperatorModel:
    estimator: RandomForestRegressor
    exact_lookup: Mapping[tuple[int, int, int, int, int, bool], float]
    batch_scaling_curves: Mapping[tuple[int, bool, float, float], tuple[np.ndarray, np.ndarray]]

    def predict(self, features: GDNBatchFeatures) -> float:
        exact = self.exact_lookup.get(features.exact_key())
        if exact is not None:
            return float(exact)
        curve = self.batch_scaling_curves.get(features.batch_scaling_key())
        if curve is not None:
            sizes, times = curve
            if len(sizes) >= 2 and sizes[0] <= features.batch_size <= sizes[-1]:
                # Bounded interpolation preserves the measured endpoints on a
                # one-dimensional batch sweep. RF bootstrap averages can be
                # strongly biased on the two-row grids used during bring-up.
                return float(np.interp(features.batch_size, sizes, times))
        prediction = float(self.estimator.predict(features.as_vector()[None, :])[0])
        if not np.isfinite(prediction) or prediction < 0.0:
            raise ValueError(f"GDN prediction produced invalid time {prediction!r}")
        return prediction


class ProfiledGDNPredictor:
    """Fit per-operator regressors over one runtime-homogeneous GDN CSV."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        model_config: Any,
        device: str,
        tensor_parallel_size: int,
        measurement_type: MeasurementType,
    ) -> None:
        validate_gdn_profiling_dataframe(dataframe)
        frame = dataframe.copy()
        frame["measurement_type"] = frame["measurement_type"].map(
            lambda value: MeasurementType.from_string(value).value
        )
        frame["has_initial_state"] = frame["has_initial_state"].map(
            lambda value: _coerce_bool(value, field_name="has_initial_state")
        )

        architecture_profile = (
            model_config.get_model_architecture_profile().profile_id
        )
        expected = {
            "measurement_type": measurement_type.value,
            "model_architecture_profile": architecture_profile,
            "quant_signature": model_config.get_quant_signature(),
            "device": str(device),
            "num_tensor_parallel_workers": int(tensor_parallel_size),
        }
        for column, expected_value in expected.items():
            frame = frame[frame[column] == expected_value]
        if frame.empty:
            raise ValueError(
                "GDN profiling data has no rows matching runtime contract: "
                f"{expected}"
            )

        gdn_config = model_config.get_gdn_config()
        if gdn_config is None:
            raise ValueError("ProfiledGDNPredictor requires GDN model dimensions")
        dimension_contract = {
            "hidden_size": int(model_config.embedding_dim),
            "conv_kernel_size": int(gdn_config.conv_kernel_size),
            "key_head_dim": int(gdn_config.key_head_dim),
            "value_head_dim": int(gdn_config.value_head_dim),
            "num_key_heads": int(gdn_config.num_key_heads),
            "num_value_heads": int(gdn_config.num_value_heads),
        }
        for column, expected_value in dimension_contract.items():
            mismatches = frame[frame[column].astype(int) != expected_value]
            if not mismatches.empty:
                raise ValueError(
                    f"GDN profiling {column} does not match model: "
                    f"expected {expected_value}"
                )

        runtime_columns = (
            "runtime_stack_signature",
            "gdn_runtime_backend",
            "gdn_rank_aggregation",
            "gdn_prefill_backend",
            "gdn_decode_backend",
            "gqa_interleaved_layout",
            "packed_recurrent_decode",
            "model_dtype",
            "conv_state_dtype",
            "recurrent_state_dtype",
            "weight_source",
        )
        ambiguous = {
            column: sorted(str(value) for value in frame[column].dropna().unique())
            for column in runtime_columns
            if frame[column].nunique(dropna=False) != 1
        }
        if ambiguous:
            raise ValueError(
                "GDN profiling rows mix incompatible runtime contracts: "
                f"{ambiguous}"
            )
        self.runtime_contract = {
            column: frame.iloc[0][column] for column in runtime_columns
        }
        self.measurement_type = measurement_type
        self._models: dict[tuple[str, str], _OperatorModel] = {}
        phases = frame.apply(self._row_phase, axis=1)
        for phase in ("prefill", "decode", "mixed"):
            operator_frame = frame[phases == phase]
            if operator_frame.empty:
                continue
            for operator in GATED_DELTA_NET_FAMILY.predictor_ops():
                if operator.name.startswith("gdn_core_") and operator.name != f"gdn_core_{phase}":
                    continue
                # Even shared projection names need phase-specific fits. With
                # sparse grids, a bootstrap tree can contain only prefill rows
                # and inject millisecond projections into microsecond decode.
                # Exact rows retain their existing lookup values.
                target_column = f"time_stats.{operator.profiling_name()}.median"
                self._models[(operator.name, phase)] = self._fit_operator_model(
                    operator_frame, target_column,
                )

    @staticmethod
    def _row_phase(row: pd.Series) -> str:
        if int(row["batch_num_prefill_tokens"]) > 0 and int(
            row["batch_num_decode_tokens"]
        ) > 0:
            return "mixed"
        if int(row["batch_num_prefill_tokens"]) > 0:
            return "prefill"
        return "decode"

    @staticmethod
    def _row_features(row: pd.Series) -> GDNBatchFeatures:
        return GDNBatchFeatures(
            batch_size=int(row["batch_size"]),
            batch_num_tokens=int(row["batch_num_tokens"]),
            batch_num_prefill_tokens=int(row["batch_num_prefill_tokens"]),
            batch_num_decode_tokens=int(row["batch_num_decode_tokens"]),
            max_query_len=int(row["max_query_len"]),
            has_initial_state=bool(row["has_initial_state"]),
        )

    @classmethod
    def _fit_operator_model(
        cls,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> _OperatorModel:
        if target_column not in dataframe.columns:
            raise ValueError(f"GDN profiling data is missing target {target_column}")
        features = [cls._row_features(row) for _, row in dataframe.iterrows()]
        targets = dataframe[target_column].astype(float).to_numpy()
        if not np.isfinite(targets).all() or (targets < 0.0).any():
            raise ValueError(f"GDN target {target_column} contains invalid timings")
        estimator = RandomForestRegressor(
            n_estimators=128,
            max_depth=16,
            random_state=0,
            n_jobs=1,
        )
        estimator.fit(np.vstack([feature.as_vector() for feature in features]), targets)
        exact_lookup: dict[tuple[int, int, int, int, int, bool], float] = {}
        for feature, target in zip(features, targets):
            key = feature.exact_key()
            target_value = float(target)
            previous = exact_lookup.get(key)
            if previous is not None and not np.isclose(previous, target_value):
                raise ValueError(
                    "GDN profiling data has conflicting duplicate feature rows: "
                    f"key={key}, values=({previous}, {target_value})"
                )
            exact_lookup[key] = target_value
        curves: dict[tuple[int, bool, float, float], dict[int, float]] = {}
        for feature, target in zip(features, targets):
            curves.setdefault(feature.batch_scaling_key(), {})[feature.batch_size] = float(target)
        return _OperatorModel(
            estimator=estimator, exact_lookup=exact_lookup,
            batch_scaling_curves={key: (np.asarray(sorted(values)),
                                        np.asarray([values[size] for size in sorted(values)]))
                                  for key, values in curves.items()},
        )

    @classmethod
    def from_csv(
        cls,
        path: str,
        **kwargs: Any,
    ) -> "ProfiledGDNPredictor":
        profile_path = Path(path)
        if not profile_path.is_file():
            raise FileNotFoundError(
                f"GDN profiling file does not exist: {profile_path}"
            )
        return cls(pd.read_csv(profile_path), **kwargs)

    def predict_operator_times(self, features: GDNBatchFeatures) -> dict[str, float]:
        required_names = (
            "gdn_input_projections",
            f"gdn_core_{features.phase}",
            "gdn_output_projection",
        )
        missing = [name for name in required_names if (name, features.phase) not in self._models]
        if missing:
            raise ValueError(
                f"GDN profile does not support {features.phase} prediction: {missing}"
            )
        return {
            name: self._models[(name, features.phase)].predict(features) for name in required_names
        }

    def predict_attention_time(self, batch: Any, *, norm_time_ms: float) -> AttentionTime:
        features = GDNBatchFeatures.from_batch(batch)
        op_times = self.predict_operator_times(features)
        core_name = f"gdn_core_{features.phase}"
        return AttentionTime(
            attention_prefill_execution_time=(
                op_times[core_name] if features.phase != "decode" else 0.0
            ),
            attention_decode_execution_time=(
                op_times[core_name] if features.phase == "decode" else 0.0
            ),
            attention_layer_pre_proj_execution_time=op_times[
                "gdn_input_projections"
            ],
            attention_layer_post_proj_execution_time=op_times[
                "gdn_output_projection"
            ],
            attn_norm_time=float(norm_time_ms),
            operator_times=AttentionOperatorTimes(op_times),
        )


__all__ = ["GDNBatchFeatures", "ProfiledGDNPredictor"]
