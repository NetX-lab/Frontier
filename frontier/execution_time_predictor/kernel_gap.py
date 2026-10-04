"""Device idle time between the kernels of a kernel-only forward step.

A kernel-only operator table times each operator's kernels and nothing between
them. kernel_gap.csv (frontier.profiling.kernel_gap) holds, per execution mode,
the mean device idle time before a kernel; a step priced from kernel-only
tables pays it once per kernel it launches. Each kernel-only operator table
records the kernels an operator launches in ``time_stats.<op>.kernel_count``,
and a kernel-count model trained beside the operator's time model prices it.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

import pandas as pd

from frontier.profiling.kernel_gap.schema import EXECUTION_MODE_COLUMN, EXECUTION_MODES, KERNEL_GAP_US_COLUMN
from frontier.types import MeasurementType

KERNEL_COUNT_STAT = "kernel_count"


def kernel_count_column(target_column: str) -> str | None:
    """The kernel-count column of an operator time column ``time_stats.<op>.<stat>``."""
    parts = target_column.split(".")
    if len(parts) != 3 or parts[0] != "time_stats" or parts[2] == KERNEL_COUNT_STAT:
        return None
    return f"time_stats.{parts[1]}.{KERNEL_COUNT_STAT}"


def read_kernel_gap_ms(path: str) -> dict[str, float]:
    """Mean device gap before a kernel, in milliseconds, by execution mode."""
    table = pd.read_csv(path)
    gaps_ms = {}
    for mode in EXECUTION_MODES:
        gaps_us = table.loc[table[EXECUTION_MODE_COLUMN] == mode, KERNEL_GAP_US_COLUMN]
        if len(gaps_us) != 1 or not math.isfinite(gaps_us.iloc[0]) or gaps_us.iloc[0] < 0:
            raise ValueError(
                f"{path} needs one finite, non-negative {KERNEL_GAP_US_COLUMN} row for "
                f"{EXECUTION_MODE_COLUMN}={mode}"
            )
        gaps_ms[mode] = float(gaps_us.iloc[0]) * 1e-3
    return gaps_ms


def load_kernel_gap_ms(
    config: Any, kernel_gap_input_file: str, replica_config: Any, *, sys_arch: str,
) -> dict[str, float] | None:
    """The kernel gap that prices kernel-only steps, or None when kernel_gap.csv is absent."""
    if config.enable_dummy_mode or not os.path.exists(kernel_gap_input_file):
        return None
    # These paths price layers outside the per-layer kernel counting.
    if sys_arch == "pd-af-disaggregation":
        unsupported = "PD-AF"
    elif replica_config.model_config.get_num_gdn_layers() > 0:
        unsupported = "GDN layers"
    elif replica_config.speculative_decoding_config.enabled:
        unsupported = "speculative decoding"
    else:
        return read_kernel_gap_ms(kernel_gap_input_file)
    raise ValueError(
        f"{kernel_gap_input_file} prices the gap before every kernel of a kernel-only step, "
        f"which {unsupported} does not support; remove the file to price steps without it."
    )


@dataclass
class LayerKernelCounts:
    """Kernels one layer launches: its attention operators' in total, the others' by operator."""

    attention: float = 0.0
    operators: dict[str, float] = field(default_factory=dict)


class KernelCountTraining:
    """Trains each kernel-only operator time model with a model of the kernels it launches."""

    _kernel_gap_ms: dict[str, float] | None = None
    _kernel_gap_input_file: str = ""

    def _paired_with_kernel_count_model(
        self,
        model: Any,
        model_name: str,
        rows: pd.DataFrame,
        feature_columns: Sequence[str],
        target_column: str,
        train: Callable[[str, pd.DataFrame, list[str], str], Any],
    ) -> Any:
        """Attach the kernel-count model trained on the rows that trained ``model``.

        The count model is attached after ``model`` is cached, so cached time
        models stay the same with and without kernel_gap.csv.
        """
        count_column = kernel_count_column(target_column)
        if (
            self._kernel_gap_ms is None
            or self._active_measurement_type is not MeasurementType.KERNEL_ONLY
            or count_column is None
        ):
            return model
        measured = rows.dropna(subset=[*feature_columns, target_column])
        if count_column not in measured or measured[count_column].isna().any():
            raise ValueError(
                f"{self._kernel_gap_input_file} prices the gap before every kernel of a "
                f"kernel-only step, so every kernel-only row that trains {model_name} needs "
                f"{count_column}."
            )
        count_model = train(f"{model_name}_kernel_count", measured, list(feature_columns), count_column)
        count_model._frontier_operator_name = count_column.split(".")[1]
        model._frontier_kernel_count_model = count_model
        return model


class KernelGapPricing:
    """Adds the device gap before each kernel to the layers of kernel-only steps."""

    _kernel_count_scopes: tuple[LayerKernelCounts, ...] = ()

    @contextmanager
    def _counting_kernels(self) -> Iterator[LayerKernelCounts]:
        counts = LayerKernelCounts()
        self._kernel_count_scopes = (*self._kernel_count_scopes, counts)
        try:
            yield counts
        finally:
            self._kernel_count_scopes = self._kernel_count_scopes[:-1]

    @contextmanager
    def _counting_launch(self) -> Iterator[None]:
        """Add the kernels of one more launch of the operators looked up in this block.

        Unlike a repeated lookup, such as one KV extract per request, each
        launch runs its kernels again.
        """
        with self._counting_kernels() as launch:
            yield
        if self._kernel_count_scopes:
            operators = self._kernel_count_scopes[-1].operators
            for name, count in launch.operators.items():
                operators[name] = operators.get(name, 0.0) + count

    def _record_kernel_count(
        self,
        model_name: str,
        model: Any,
        feature_key: tuple[float, ...],
        feature_names: Sequence[str],
    ) -> None:
        """Record the kernels of the operator that ``model`` prices at ``feature_key``.

        An operator looked up twice in one layer launches its kernels once, so
        a later lookup replaces the earlier count.
        """
        count_model = getattr(model, "_frontier_kernel_count_model", None)
        if not self._kernel_count_scopes or count_model is None:
            return
        count = count_model._frontier_exact_lookup.get(feature_key)
        if count is None:
            family_name = self._measurement_family_name(self._active_measurement_type)
            predicted = self._runtime_cache[family_name][f"{model_name}_kernel_count"]
            if feature_key not in predicted:
                features = pd.DataFrame([feature_key], columns=list(feature_names))
                predicted[feature_key] = float(count_model.predict(features)[0])
            count = predicted[feature_key]
        self._kernel_count_scopes[-1].operators[count_model._frontier_operator_name] = float(count)

    def _record_attention_kernels(self, kernel_count: float) -> None:
        if self._kernel_count_scopes:
            self._kernel_count_scopes[-1].attention = kernel_count

    def _predict_layer_with_kernel_gap(self, predict_layer: Callable[..., Any], *args, **kwargs) -> Any:
        """Predict one layer; on a kernel-only step, add the gap before each of its kernels."""
        if self._kernel_gap_ms is None:
            return predict_layer(*args, **kwargs)
        with self._counting_kernels() as counts:
            layer = predict_layer(*args, **kwargs)
        if self._active_measurement_type is MeasurementType.KERNEL_ONLY:
            # An eager step's device stream runs eager launches; a graph step replays a CUDA graph.
            execution_mode = (
                "eager" if self._is_event_measurement_type(self._step_measurement_type) else "cuda_graph"
            )
            layer.add_kernel_gap(self._kernel_gap_ms[execution_mode], counts.attention, counts.operators)
        return layer
