"""Metrics collection, output taxonomy and cache locations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import os
from typing import Optional

from frontier.utils.output_paths import (
    validate_output_filename,
    validate_run_id,
)


@dataclass
class MetricsConfig:
    """Metric configuration."""

    write_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to write metrics."},
    )
    write_json_trace: bool = field(
        default=False,
        metadata={"help": "Whether to write json trace."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases project name."},
    )
    wandb_group: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases group name."},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run name."},
    )
    wandb_sweep_id: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases sweep id."},
    )
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run id."},
    )
    enable_chrome_trace: bool = field(
        default=True,
        metadata={"help": "Enable Chrome tracing."},
    )

    # Op-Level Tracing
    enable_op_level_tracing: bool = field(
        default=False,
        metadata={"help": "Enable detailed op-level tracing (output to JSONL)."},
    )
    trace_output_file: str = field(
        default="op_traces.jsonl",
        metadata={"help": "Output filename for op-level traces."},
    )
    enable_metrics_ground_truth_trace: bool = field(
        default=False,
        metadata={
            "help": "Enable explicit request-level metrics ground-truth JSONL output."
        },
    )
    metrics_ground_truth_trace_file: str = field(
        default="metrics_ground_truth.jsonl",
        metadata={"help": "Output filename for metrics ground-truth request traces."},
    )
    enable_per_layer_expansion: bool = field(
        default=False,
        metadata={
            "help": "Enable per-layer trace expansion. When enabled, traces show "
            "individual layer operations instead of aggregated spans."
        },
    )
    num_requests_to_trace_per_layer: int = field(
        default=5,
        metadata={
            "help": "Number of requests to capture with per-layer expansion. "
            "Only applies when enable_per_layer_expansion is True."
        },
    )

    save_table_to_wandb: bool = field(
        default=False,
        metadata={"help": "Whether to save table to wandb."},
    )
    store_plots: bool = field(
        default=True,
        metadata={"help": "Whether to store plots."},
    )
    enable_memory_time_series: bool = field(
        default=False,
        metadata={
            "help": "Enable memory usage time series output. "
            "Only valid when log_level is 'debug'."
        },
    )
    store_operation_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to store operation metrics."},
    )
    store_token_completion_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to store token completion metrics."},
    )
    store_request_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store request metrics."},
    )
    store_batch_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store batch metrics."},
    )
    store_utilization_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store utilization metrics."},
    )
    keep_individual_batch_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to keep individual batch metrics."},
    )
    store_frontier_stage_batch_ledger: bool = field(
        default=True,
        metadata={"help": "Whether to write the full Frontier stage-batch ledger."},
    )
    store_frontier_stage_batch_ledger_summary: bool = field(
        default=False,
        metadata={
            "help": "Whether to write a bounded Frontier stage-batch ledger summary."
        },
    )
    subsamples: Optional[int] = field(
        default=None,
        metadata={"help": "Subsamples."},
    )
    min_batch_index: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum batch index."},
    )
    max_batch_index: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum batch index."},
    )
    output_dir: str = field(
        default="outputs/metrics",
        metadata={"help": "Metrics output root directory."},
    )
    cache_dir: str = field(
        default="cache",
        metadata={"help": "Cache directory."},
    )
    run_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "Metrics run id used under outputs/metrics/<model>/<workload>/<run_id>."
        },
    )

    def __post_init__(self):
        if self.run_id is None:
            self.run_id = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
        self.run_id = validate_run_id(self.run_id)
        self.trace_output_file = validate_output_filename(
            self.trace_output_file, "trace_output_file"
        )
        self.metrics_ground_truth_trace_file = validate_output_filename(
            self.metrics_ground_truth_trace_file, "metrics_ground_truth_trace_file"
        )
        os.makedirs(self.output_dir, exist_ok=True)
