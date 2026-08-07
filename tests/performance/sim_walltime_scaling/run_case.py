"""Run one Frontier wall-clock scaling case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from frontier.errors import FrontierMemoryOOMError


SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_NAMES = {
    "dense": "llama3.3-70b",
    "moe": "Qwen3-235B-A22B",
}
MODEL_EXPERT_CONFIG = {
    "dense": (1, 1),
    "moe": (128, 8),
}
PRIMARY_SHAPES = {
    "dense": {
        "attn_tp": 4,
        "attn_dp": 1,
        "moe_tp": 1,
        "moe_ep": 1,
        "pp": 2,
    },
    "moe": {
        "attn_tp": 4,
        "attn_dp": 2,
        "moe_tp": 1,
        "moe_ep": 8,
        "pp": 2,
    },
}
MEASUREMENT_DISABLED_METRICS_FIELDS = (
    "write_metrics",
    "write_json_trace",
    "enable_chrome_trace",
    "enable_op_level_tracing",
    "enable_metrics_ground_truth_trace",
    "enable_per_layer_expansion",
    "save_table_to_wandb",
    "store_plots",
    "enable_memory_time_series",
    "store_operation_metrics",
    "store_token_completion_metrics",
    "store_request_metrics",
    "store_batch_metrics",
    "store_utilization_metrics",
    "store_frontier_stage_batch_ledger",
    "store_frontier_stage_batch_ledger_summary",
)
MEASUREMENT_DISABLED_RUNTIME_FIELDS = (
    "enable_cluster_event_logging",
    "enable_performance_profiling",
)
REQUIRED_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "case_id",
        "attempt_id",
        "attempt_index",
        "case_fingerprint",
        "git_sha",
        "python_executable",
        "seed",
        "model",
        "model_name",
        "total_gpus",
        "simulation_mode",
        "shape",
        "replicas_per_cluster",
        "mode",
        "host",
        "worker_job_id",
        "status",
        "sim_wallclock_s",
        "init_s",
        "total_proc_s",
        "peak_rss_mb",
        "requests",
        "qps",
        "prefill_tokens",
        "decode_tokens",
        "expected_requests",
        "completed_requests",
        "event_count",
        "events_per_s",
        "command",
        "started_at",
        "completed_at",
        "exit_code",
        "signal",
        "failure_reason",
        "oom_evidence",
        "notes",
        "stderr_tail",
    }
)


@dataclass(frozen=True)
class ParallelShape:
    """Per-replica parallel shape shared by the PREFILL and DECODE clusters."""

    attn_tp: int
    attn_dp: int
    moe_tp: int
    moe_ep: int
    pp: int

    def __post_init__(self) -> None:
        values = {
            "attn_tp": self.attn_tp,
            "attn_dp": self.attn_dp,
            "moe_tp": self.moe_tp,
            "moe_ep": self.moe_ep,
            "pp": self.pp,
        }
        for name, value in values.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.attn_tp > 8:
            raise ValueError(f"attn_tp must be <= 8, got {self.attn_tp}")
        if self.moe_tp > 8:
            raise ValueError(f"moe_tp must be <= 8, got {self.moe_tp}")
        if self.pp > 16:
            raise ValueError(f"pp must be <= 16, got {self.pp}")

    @property
    def attention_domain_size(self) -> int:
        return self.attn_tp * self.attn_dp

    @property
    def moe_domain_size(self) -> int:
        return self.moe_tp * self.moe_ep

    @property
    def replica_world_size(self) -> int:
        return self.pp * self.attention_domain_size

    @property
    def slug(self) -> str:
        return (
            f"atp{self.attn_tp}-adp{self.attn_dp}-"
            f"mtp{self.moe_tp}-mep{self.moe_ep}-pp{self.pp}"
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParallelShape":
        return cls(
            attn_tp=int(payload["attn_tp"]),
            attn_dp=int(payload["attn_dp"]),
            moe_tp=int(payload["moe_tp"]),
            moe_ep=int(payload["moe_ep"]),
            pp=int(payload["pp"]),
        )


@dataclass(frozen=True)
class CaseSpec:
    """Serializable contract for one immutable simulation attempt."""

    model: str
    total_gpus: int
    mode: str
    attempt_index: int
    shape: ParallelShape
    num_requests: int
    qps: float
    prefill_tokens: int
    decode_tokens: int
    seed: int = 42
    simulation_mode: str = "online"
    dummy_execution_time_ms: float = 1.0
    device: str = "h800"
    network_device: str = "h800_dgx"

    def __post_init__(self) -> None:
        if self.model not in MODEL_NAMES:
            raise ValueError(
                f"model must be one of {sorted(MODEL_NAMES)}, got {self.model!r}"
            )
        if self.mode not in {"parallel", "sequential"}:
            raise ValueError(
                f"mode must be 'parallel' or 'sequential', got {self.mode!r}"
            )
        if self.simulation_mode != "online":
            raise ValueError(
                "wall-clock scaling cases require simulation_mode='online' "
                "so Poisson QPS controls request arrivals"
            )
        if not isinstance(self.total_gpus, int) or self.total_gpus <= 0:
            raise ValueError(
                f"total_gpus must be a positive integer, got {self.total_gpus!r}"
            )
        if not isinstance(self.attempt_index, int) or self.attempt_index < 0:
            raise ValueError(
                f"attempt_index must be a non-negative integer, got {self.attempt_index!r}"
            )
        if not isinstance(self.num_requests, int) or self.num_requests <= 0:
            raise ValueError(
                f"num_requests must be a positive integer, got {self.num_requests!r}"
            )
        if self.qps <= 0:
            raise ValueError(f"qps must be > 0, got {self.qps}")
        if self.prefill_tokens < 2:
            raise ValueError(
                f"prefill_tokens must be >= 2, got {self.prefill_tokens}"
            )
        if self.decode_tokens < 1:
            raise ValueError(
                f"decode_tokens must be >= 1, got {self.decode_tokens}"
            )
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.dummy_execution_time_ms <= 0:
            raise ValueError(
                "dummy_execution_time_ms must be > 0, got "
                f"{self.dummy_execution_time_ms}"
            )

        if self.model == "dense":
            if self.shape.attn_dp != 1:
                raise ValueError("dense cases require attn_dp=1")
            if (self.shape.moe_tp, self.shape.moe_ep) != (1, 1):
                raise ValueError("dense cases require moe_tp=1 and moe_ep=1")
        elif self.shape.attention_domain_size != self.shape.moe_domain_size:
            raise ValueError(
                "MoE shared-domain shape requires attn_tp*attn_dp == "
                "moe_tp*moe_ep"
            )

        cluster_world_size_numerator = self.total_gpus
        cluster_world_size_denominator = 2 * self.shape.replica_world_size
        if (
            self.total_gpus % 2 != 0
            or cluster_world_size_numerator % cluster_world_size_denominator != 0
        ):
            raise ValueError(
                "total_gpus and replica shape must produce integral replicas per "
                f"cluster, got total_gpus={self.total_gpus}, "
                f"replica_world_size={self.shape.replica_world_size}"
            )
        if self.replicas_per_cluster < 1:
            raise ValueError("replicas per cluster must be >= 1")

    @classmethod
    def for_scale(
        cls,
        *,
        model: str,
        total_gpus: int,
        mode: str,
        attempt_index: int,
        shape: ParallelShape | None = None,
        num_requests: int | None = None,
        qps: float | None = None,
        prefill_tokens: int = 512,
        decode_tokens: int = 128,
        seed: int = 42,
        dummy_execution_time_ms: float = 1.0,
        device: str = "h800",
        network_device: str = "h800_dgx",
    ) -> "CaseSpec":
        if model not in PRIMARY_SHAPES:
            raise ValueError(
                f"model must be one of {sorted(PRIMARY_SHAPES)}, got {model!r}"
            )
        resolved_shape = shape or ParallelShape(**PRIMARY_SHAPES[model])
        return cls(
            model=model,
            total_gpus=total_gpus,
            mode=mode,
            attempt_index=attempt_index,
            shape=resolved_shape,
            num_requests=(
                int(num_requests) if num_requests is not None else 2 * total_gpus
            ),
            qps=float(qps) if qps is not None else 0.25 * total_gpus,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            seed=seed,
            dummy_execution_time_ms=dummy_execution_time_ms,
            device=device,
            network_device=network_device,
        )

    @property
    def model_name(self) -> str:
        return MODEL_NAMES[self.model]

    @property
    def total_experts(self) -> int:
        return MODEL_EXPERT_CONFIG[self.model][0]

    @property
    def router_topk(self) -> int:
        return MODEL_EXPERT_CONFIG[self.model][1]

    @property
    def replicas_per_cluster(self) -> int:
        return self.total_gpus // (2 * self.shape.replica_world_size)

    @property
    def case_id(self) -> str:
        return f"{self.model}-g{self.total_gpus}-{self.mode}-{self.shape.slug}"

    @property
    def attempt_id(self) -> str:
        return f"{self.case_id}-attempt-{self.attempt_index:02d}"

    @property
    def case_fingerprint(self) -> str:
        payload = self.to_dict()
        payload.pop("attempt_index")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = asdict(self.shape)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CaseSpec":
        allowed = {
            "model",
            "total_gpus",
            "mode",
            "attempt_index",
            "shape",
            "num_requests",
            "qps",
            "prefill_tokens",
            "decode_tokens",
            "seed",
            "simulation_mode",
            "dummy_execution_time_ms",
            "device",
            "network_device",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown CaseSpec fields: {sorted(unknown)}")
        required = {
            "model",
            "total_gpus",
            "mode",
            "attempt_index",
            "shape",
            "num_requests",
            "qps",
            "prefill_tokens",
            "decode_tokens",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"Missing CaseSpec fields: {sorted(missing)}")
        values = dict(payload)
        values["shape"] = ParallelShape.from_dict(values["shape"])
        return cls(**values)


def _append_option(argv: list[str], option: str, value: Any) -> None:
    argv.extend([option, str(value)])


def build_frontier_argv(case: CaseSpec, output_root: Path) -> list[str]:
    """Build the exact Frontier CLI represented by a CaseSpec."""

    shape = case.shape
    replicas = case.replicas_per_cluster
    argv: list[str] = [
        "--simulation_mode",
        case.simulation_mode,
        "--sys_arch",
        "pd-disaggregation",
        "--enable_parallel_clusters"
        if case.mode == "parallel"
        else "--no-enable_parallel_clusters",
        "--seed",
        str(case.seed),
        "--log_level",
        "WARNING",
    ]

    for cluster_name in ("prefill", "decode"):
        _append_option(
            argv,
            f"--cluster_config_{cluster_name}_cluster_num_replicas",
            replicas,
        )
        prefix = f"--cluster_config_{cluster_name}_replica_config"
        _append_option(argv, f"{prefix}_num_pipeline_stages", shape.pp)
        _append_option(argv, f"{prefix}_attn_tensor_parallel_size", shape.attn_tp)
        _append_option(argv, f"{prefix}_attn_data_parallel_size", shape.attn_dp)
        _append_option(argv, f"{prefix}_moe_tensor_parallel_size", shape.moe_tp)
        _append_option(argv, f"{prefix}_moe_expert_parallel_size", shape.moe_ep)
        _append_option(argv, f"{prefix}_total_expert_num", case.total_experts)
        _append_option(argv, f"{prefix}_router_topk", case.router_topk)
        _append_option(argv, f"{prefix}_device", case.device)
        _append_option(argv, f"{prefix}_network_device", case.network_device)

    replica_values = {
        "--replica_config_model_name": case.model_name,
        "--replica_config_device": case.device,
        "--replica_config_network_device": case.network_device,
        "--replica_config_num_pipeline_stages": shape.pp,
        "--replica_config_attn_tensor_parallel_size": shape.attn_tp,
        "--replica_config_attn_data_parallel_size": shape.attn_dp,
        "--replica_config_moe_tensor_parallel_size": shape.moe_tp,
        "--replica_config_moe_expert_parallel_size": shape.moe_ep,
        "--replica_config_total_expert_num": case.total_experts,
        "--replica_config_router_topk": case.router_topk,
        "--replica_config_moe_routing_mode": "simulation",
        "--replica_config_moe_routing_seed": case.seed,
    }
    for option, value in replica_values.items():
        _append_option(argv, option, value)

    argv.extend(
        [
            "--cc_backend_config_type",
            "analytical",
            "--replica_scheduler_config_type",
            "vllm_v1",
            "--decode_cuda_graph_mode",
            "none",
            "--vllm_v1_scheduler_config_num_blocks",
            "0",
            "--vllm_v1_scheduler_config_num_blocks_mode",
            "memory_planner",
            "--vllm_v1_scheduler_config_block_size",
            "16",
            "--vllm_v1_scheduler_config_max_tokens_in_batch",
            "1024",
            "--vllm_v1_scheduler_config_long_prefill_token_threshold",
            "64",
            "--vllm_v1_scheduler_config_enable_chunked_prefill",
            "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
            "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms",
            str(case.dummy_execution_time_ms),
            "--request_generator_config_type",
            "synthetic",
            "--synthetic_request_generator_config_num_requests",
            str(case.num_requests),
            "--length_generator_config_type",
            "fixed",
            "--fixed_request_length_generator_config_prefill_tokens",
            str(case.prefill_tokens),
            "--fixed_request_length_generator_config_decode_tokens",
            str(case.decode_tokens),
            "--interval_generator_config_type",
            "poisson",
            "--poisson_request_interval_generator_config_qps",
            str(case.qps),
            "--analytical_kv_cache_transfer_config_network_bandwidth_gbps",
            "200.0",
            "--analytical_kv_cache_transfer_config_network_latency_ms",
            "0.5",
            "--metrics_config_output_dir",
            str(output_root),
            "--metrics_config_run_id",
            case.attempt_id,
            "--no-metrics_config_write_metrics",
            "--no-metrics_config_write_json_trace",
            "--no-metrics_config_enable_chrome_trace",
            "--no-metrics_config_enable_op_level_tracing",
            "--no-metrics_config_enable_metrics_ground_truth_trace",
            "--no-metrics_config_enable_per_layer_expansion",
            "--no-metrics_config_save_table_to_wandb",
            "--no-metrics_config_store_plots",
            "--no-metrics_config_enable_memory_time_series",
            "--no-metrics_config_store_operation_metrics",
            "--no-metrics_config_store_token_completion_metrics",
            "--no-metrics_config_store_request_metrics",
            "--no-metrics_config_store_batch_metrics",
            "--no-metrics_config_store_utilization_metrics",
            "--no-metrics_config_store_frontier_stage_batch_ledger",
            "--no-metrics_config_store_frontier_stage_batch_ledger_summary",
            "--no-enable_cluster_event_logging",
            "--no-enable_performance_profiling",
        ]
    )
    return argv


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Create one JSON artifact atomically without replacing an existing attempt."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable result: {path}")

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_case(path: Path) -> CaseSpec:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Case JSON must contain an object")
    return CaseSpec.from_dict(payload)


def _default_config_factory(argv: Sequence[str]) -> Any:
    from frontier.config import SimulationConfig

    original_argv = sys.argv
    sys.argv = ["frontier.main", *argv]
    try:
        return SimulationConfig.create_from_cli_args()
    finally:
        sys.argv = original_argv


def _default_simulator_factory(config: Any) -> Any:
    from frontier.simulator import Simulator

    return Simulator(config)


def validate_measurement_config(config: Any) -> None:
    """Fail when effective configuration would contaminate event-loop timing."""

    enabled_fields = [
        f"metrics_config.{field_name}"
        for field_name in MEASUREMENT_DISABLED_METRICS_FIELDS
        if getattr(config.metrics_config, field_name) is not False
    ]
    enabled_fields.extend(
        field_name
        for field_name in MEASUREMENT_DISABLED_RUNTIME_FIELDS
        if getattr(config, field_name) is not False
    )
    if enabled_fields:
        raise ValueError(
            "Wall-clock measurement requires output and profiling flags to be "
            f"disabled; enabled fields: {', '.join(enabled_fields)}"
        )


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    sha = result.stdout.strip()
    if len(sha) != 40:
        raise RuntimeError(f"Unexpected git SHA from rev-parse: {sha!r}")
    return sha


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class _SequentialEventCounter:
    """Minimal counter installed at the simulator's existing profiler call site."""

    enabled = True

    def __init__(self) -> None:
        self.event_count = 0
        self.phase_times: dict[str, float] = {}

    @contextmanager
    def profile(
        self,
        _component_name: str,
        _metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        yield

    def record_event_processing(self, _event_type: str, _duration: float) -> None:
        self.event_count += 1

    def record_phase(self, phase_name: str, duration: float) -> None:
        self.phase_times[phase_name] = duration


def _collect_event_count(simulator: Any, sequential_counter: Any) -> int:
    if getattr(simulator, "_parallel_mode", False):
        cluster_simulators = getattr(simulator, "_cluster_simulators", {})
        return sum(
            int(getattr(cluster_simulator, "_events_processed", 0))
            for cluster_simulator in cluster_simulators.values()
        )
    return int(sequential_counter.event_count)


def _collect_request_counts(simulator: Any, case: CaseSpec) -> tuple[int, int | None]:
    metric_store = getattr(simulator, "metric_store", None)
    if metric_store is None:
        return case.num_requests, None
    expected = int(metric_store.get_total_requests())
    completed = int(metric_store.get_completed_requests())
    return expected, completed


def _base_result(
    *,
    case: CaseSpec,
    command: list[str],
    started_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.case_id,
        "attempt_id": case.attempt_id,
        "attempt_index": case.attempt_index,
        "case_fingerprint": case.case_fingerprint,
        "git_sha": _git_sha(),
        "python_executable": sys.executable,
        "seed": case.seed,
        "model": case.model,
        "model_name": case.model_name,
        "total_gpus": case.total_gpus,
        "simulation_mode": case.simulation_mode,
        "shape": asdict(case.shape),
        "replicas_per_cluster": case.replicas_per_cluster,
        "mode": case.mode,
        "host": socket.gethostname(),
        "worker_job_id": os.environ.get("FRONTIER_WORKER_JOB_ID"),
        "status": None,
        "sim_wallclock_s": None,
        "init_s": None,
        "total_proc_s": None,
        "peak_rss_mb": None,
        "requests": case.num_requests,
        "qps": case.qps,
        "prefill_tokens": case.prefill_tokens,
        "decode_tokens": case.decode_tokens,
        "expected_requests": case.num_requests,
        "completed_requests": None,
        "event_count": None,
        "events_per_s": None,
        "command": command,
        "started_at": started_at,
        "completed_at": None,
        "exit_code": None,
        "signal": None,
        "failure_reason": None,
        "oom_evidence": None,
        "notes": [],
        "stderr_tail": "",
    }


def run_case(
    case: CaseSpec,
    result_path: Path,
    *,
    config_factory: Callable[[Sequence[str]], Any] | None = None,
    simulator_factory: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Initialize, run, classify, and persist one simulation attempt."""

    result_path = Path(result_path)
    if result_path.exists():
        raise FileExistsError(
            f"Refusing to run an already-recorded immutable attempt: {result_path}"
        )

    config_factory = (
        _default_config_factory if config_factory is None else config_factory
    )
    simulator_factory = (
        _default_simulator_factory
        if simulator_factory is None
        else simulator_factory
    )
    output_root = result_path.parent / "simulator-configs"
    argv = build_frontier_argv(case, output_root)
    command = [sys.executable, "-m", "frontier.main", *argv]
    started_at = _utc_now()
    process_start = time.perf_counter()
    result = _base_result(case=case, command=command, started_at=started_at)

    simulator = None
    sequential_counter = None
    run_start = None
    previous_log_level = os.environ.get("FRONTIER_LOG_LEVEL")
    os.environ["FRONTIER_LOG_LEVEL"] = "WARNING"

    try:
        init_start = time.perf_counter()
        config = config_factory(argv)
        validate_measurement_config(config)
        simulator = simulator_factory(config)
        result["init_s"] = time.perf_counter() - init_start

        if not getattr(simulator, "_parallel_mode", False):
            sequential_counter = _SequentialEventCounter()
            simulator._profiler = sequential_counter

        run_start = time.perf_counter()
        simulator.run()
        result["sim_wallclock_s"] = time.perf_counter() - run_start

        expected, completed = _collect_request_counts(simulator, case)
        result["expected_requests"] = expected
        result["completed_requests"] = completed
        event_count = _collect_event_count(simulator, sequential_counter)
        result["event_count"] = event_count
        result["events_per_s"] = event_count / result["sim_wallclock_s"]

        if expected != case.num_requests:
            result["status"] = "bug"
            result["exit_code"] = 1
            result["failure_reason"] = "request_accounting_mismatch"
            result["stderr_tail"] = (
                "Simulator request accounting does not match the CaseSpec: "
                f"requests={case.num_requests}, expected_requests={expected}, "
                f"completed_requests={completed}"
            )
        elif completed != expected:
            result["status"] = "bug"
            result["exit_code"] = 1
            result["failure_reason"] = "completed_requests_mismatch"
            result["stderr_tail"] = (
                "Simulator.run() returned normally with incomplete requests: "
                f"completed_requests={completed}/{expected}"
            )
        elif event_count <= 0:
            result["status"] = "bug"
            result["exit_code"] = 1
            result["failure_reason"] = "non_positive_event_count"
            result["stderr_tail"] = (
                "Simulator.run() returned normally without processing events: "
                f"event_count={event_count}"
            )
        else:
            result["status"] = "success"
            result["exit_code"] = 0

    except FrontierMemoryOOMError as exc:
        if run_start is not None:
            result["sim_wallclock_s"] = time.perf_counter() - run_start
        result["status"] = "simulated-oom"
        result["exit_code"] = 2
        result["failure_reason"] = exc.reason
        result["oom_evidence"] = exc.details
        result["stderr_tail"] = str(exc)

    except Exception as exc:
        if run_start is not None:
            result["sim_wallclock_s"] = time.perf_counter() - run_start
        result["status"] = "bug"
        result["exit_code"] = 1
        result["failure_reason"] = type(exc).__name__
        result["stderr_tail"] = traceback.format_exc()[-20000:]

    finally:
        if previous_log_level is None:
            os.environ.pop("FRONTIER_LOG_LEVEL", None)
        else:
            os.environ["FRONTIER_LOG_LEVEL"] = previous_log_level

    result["total_proc_s"] = time.perf_counter() - process_start
    result["peak_rss_mb"] = _peak_rss_mb()
    result["completed_at"] = _utc_now()

    missing_fields = REQUIRED_RESULT_FIELDS - set(result)
    if missing_fields:
        raise RuntimeError(f"Result schema missing fields: {sorted(missing_fields)}")
    write_json_atomic(result_path, result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one Frontier wall-clock scaling case."
    )
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    case = load_case(args.case_json)
    result = run_case(case, args.result_json)
    if result["status"] == "simulated-oom":
        print(f"FRONTIER_MEMORY_OOM: {result['stderr_tail']}", file=sys.stderr)
    elif result["status"] == "bug":
        print(result["stderr_tail"], file=sys.stderr)
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
