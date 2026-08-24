"""Unit and tiny integration tests for the wall-clock case runner."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from frontier.errors import FrontierMemoryOOMError
from tests.performance.sim_walltime_scaling import run_case as run_case_module
from tests.performance.sim_walltime_scaling.run_case import (
    REQUIRED_RESULT_FIELDS,
    SCHEMA_VERSION,
    CaseSpec,
    ParallelShape,
    build_frontier_argv,
    run_case,
    write_json_atomic,
)


DISABLED_METRICS_FIELDS = (
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


def _measurement_config(
    *,
    enabled_metrics_field: str | None = None,
    enable_cluster_event_logging: bool = False,
    enable_performance_profiling: bool = False,
) -> SimpleNamespace:
    metrics_values = {field: False for field in DISABLED_METRICS_FIELDS}
    if enabled_metrics_field is not None:
        metrics_values[enabled_metrics_field] = True
    return SimpleNamespace(
        metrics_config=SimpleNamespace(**metrics_values),
        enable_cluster_event_logging=enable_cluster_event_logging,
        enable_performance_profiling=enable_performance_profiling,
    )


def test_dense_case_derives_formal_weak_scaling_and_cli(tmp_path) -> None:
    case = CaseSpec.for_scale(
        model="dense",
        total_gpus=32,
        mode="sequential",
        attempt_index=0,
    )

    assert case.model_name == "llama3.3-70b"
    assert case.shape == ParallelShape(
        attn_tp=4,
        attn_dp=1,
        moe_tp=1,
        moe_ep=1,
        pp=2,
    )
    assert case.replicas_per_cluster == 2
    assert case.num_requests == 64
    assert case.qps == 8.0
    assert case.prefill_tokens == 512
    assert case.decode_tokens == 128

    argv = build_frontier_argv(case, tmp_path / "simulator-output")
    joined = " ".join(argv)
    assert "--simulation_mode online" in joined
    assert "--sys_arch pd-disaggregation" in joined
    assert "--no-enable_parallel_clusters" in argv
    assert "--replica_config_model_name llama3.3-70b" in joined
    assert "--cluster_config_prefill_cluster_num_replicas 2" in joined
    assert "--cluster_config_decode_cluster_num_replicas 2" in joined
    assert "--vllm_v1_scheduler_config_num_blocks_mode memory_planner" in joined
    assert "--vllm_v1_scheduler_config_num_blocks 0" in joined
    assert "--synthetic_request_generator_config_num_requests 64" in joined
    assert "--poisson_request_interval_generator_config_qps 8.0" in joined
    assert "--no-metrics_config_write_metrics" in argv
    assert "--no-metrics_config_enable_chrome_trace" in argv
    assert "--no-metrics_config_store_request_metrics" in argv
    assert "--no-metrics_config_store_batch_metrics" in argv
    assert "--no-metrics_config_store_utilization_metrics" in argv
    assert "--no-metrics_config_store_frontier_stage_batch_ledger" in argv
    assert "--no-enable_cluster_event_logging" in argv
    assert "--no-enable_performance_profiling" in argv


def test_moe_case_derives_shared_domain_shape() -> None:
    case = CaseSpec.for_scale(
        model="moe",
        total_gpus=32,
        mode="parallel",
        attempt_index=1,
    )

    assert case.model_name == "Qwen3-235B-A22B"
    assert case.shape == ParallelShape(
        attn_tp=8,
        attn_dp=1,
        moe_tp=1,
        moe_ep=8,
        pp=2,
    )
    assert case.replicas_per_cluster == 1
    assert case.shape.attn_tp * case.shape.attn_dp == (
        case.shape.moe_tp * case.shape.moe_ep
    )
    assert case.attempt_id.endswith("attempt-01")

    argv = build_frontier_argv(case, Path("simulator-output"))
    assert "--enable_parallel_clusters" in argv


@pytest.mark.parametrize(
    ("total_gpus", "shape"),
    [
        (
            31,
            ParallelShape(
                attn_tp=4,
                attn_dp=1,
                moe_tp=1,
                moe_ep=1,
                pp=2,
            ),
        ),
        (
            32,
            ParallelShape(
                attn_tp=8,
                attn_dp=1,
                moe_tp=1,
                moe_ep=1,
                pp=4,
            ),
        ),
    ],
)
def test_case_rejects_non_integral_pd_replica_layout(
    total_gpus: int,
    shape: ParallelShape,
) -> None:
    with pytest.raises(ValueError, match="replicas per cluster"):
        CaseSpec.for_scale(
            model="dense",
            total_gpus=total_gpus,
            mode="sequential",
            attempt_index=0,
            shape=shape,
        )


def test_atomic_json_write_refuses_to_overwrite(tmp_path) -> None:
    result_path = tmp_path / "result.json"
    write_json_atomic(result_path, {"status": "success"})

    with pytest.raises(FileExistsError):
        write_json_atomic(result_path, {"status": "bug"})

    assert json.loads(result_path.read_text(encoding="utf-8")) == {
        "status": "success"
    }


def test_atomic_json_write_has_one_winner_under_concurrency(tmp_path) -> None:
    result_path = tmp_path / "concurrent-result.json"
    start_barrier = Barrier(2)

    def publish(status: str) -> str:
        start_barrier.wait()
        try:
            write_json_atomic(result_path, {"status": status})
        except FileExistsError:
            return "exists"
        return "created"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, ("first", "second")))

    assert sorted(outcomes) == ["created", "exists"]
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] in {
        "first",
        "second",
    }


@pytest.mark.parametrize("enabled_field", DISABLED_METRICS_FIELDS)
def test_measurement_config_rejects_enabled_metrics_output(enabled_field: str) -> None:
    config = _measurement_config(enabled_metrics_field=enabled_field)

    with pytest.raises(ValueError, match=enabled_field):
        run_case_module.validate_measurement_config(config)


@pytest.mark.parametrize(
    "enabled_field",
    ("enable_cluster_event_logging", "enable_performance_profiling"),
)
def test_measurement_config_rejects_enabled_runtime_output(enabled_field: str) -> None:
    config = _measurement_config(**{enabled_field: True})

    with pytest.raises(ValueError, match=enabled_field):
        run_case_module.validate_measurement_config(config)


def test_run_case_validates_output_flags_before_simulator_init(tmp_path) -> None:
    case = CaseSpec.for_scale(
        model="dense",
        total_gpus=32,
        mode="sequential",
        attempt_index=0,
        num_requests=1,
        prefill_tokens=16,
        decode_tokens=2,
        qps=1.0,
    )
    simulator_factory_called = False

    def simulator_factory(_config):
        nonlocal simulator_factory_called
        simulator_factory_called = True
        raise AssertionError("simulator must not be initialized")

    result = run_case(
        case,
        tmp_path / "invalid-output-config.json",
        config_factory=lambda _argv: _measurement_config(
            enabled_metrics_field="write_metrics"
        ),
        simulator_factory=simulator_factory,
    )

    assert simulator_factory_called is False
    assert result["status"] == "bug"
    assert result["failure_reason"] == "ValueError"
    assert "write_metrics" in result["stderr_tail"]


def test_run_case_classifies_simulated_oom(tmp_path) -> None:
    case = CaseSpec.for_scale(
        model="dense",
        total_gpus=32,
        mode="sequential",
        attempt_index=0,
        num_requests=1,
        prefill_tokens=16,
        decode_tokens=2,
        qps=1.0,
    )
    result_path = tmp_path / "oom.json"

    def raise_oom(_config):
        raise FrontierMemoryOOMError(
            "test memory admission",
            reason="insufficient_kv_cache_budget",
            details={"required_bytes": 17},
        )

    result = run_case(
        case,
        result_path,
        config_factory=lambda _argv: _measurement_config(),
        simulator_factory=raise_oom,
    )

    assert result["status"] == "simulated-oom"
    assert result["exit_code"] == 2
    assert result["failure_reason"] == "insufficient_kv_cache_budget"
    assert "required_bytes=17" in result["stderr_tail"]
    assert json.loads(result_path.read_text(encoding="utf-8")) == result


def test_run_case_classifies_unexpected_bug(tmp_path) -> None:
    case = CaseSpec.for_scale(
        model="dense",
        total_gpus=32,
        mode="sequential",
        attempt_index=0,
        num_requests=1,
        prefill_tokens=16,
        decode_tokens=2,
        qps=1.0,
    )
    result_path = tmp_path / "bug.json"

    class BugSimulator:
        _parallel_mode = False
        metric_store = SimpleNamespace(
            get_total_requests=lambda: 1,
            get_completed_requests=lambda: 0,
        )

        def __init__(self, _config) -> None:
            self._profiler = SimpleNamespace()

        def run(self) -> None:
            raise RuntimeError("intentional runner defect")

    result = run_case(
        case,
        result_path,
        config_factory=lambda _argv: _measurement_config(),
        simulator_factory=BugSimulator,
    )

    assert result["status"] == "bug"
    assert result["exit_code"] == 1
    assert result["failure_reason"] == "RuntimeError"
    assert "RuntimeError: intentional runner defect" in result["stderr_tail"]


def test_run_case_rejects_incomplete_normal_return(tmp_path) -> None:
    case = CaseSpec.for_scale(
        model="dense",
        total_gpus=32,
        mode="sequential",
        attempt_index=0,
        num_requests=1,
        prefill_tokens=16,
        decode_tokens=2,
        qps=1.0,
    )
    result_path = tmp_path / "incomplete.json"

    class IncompleteSimulator:
        _parallel_mode = False
        metric_store = SimpleNamespace(
            get_total_requests=lambda: 1,
            get_completed_requests=lambda: 0,
        )

        def __init__(self, _config) -> None:
            self._profiler = SimpleNamespace()

        def run(self) -> None:
            return None

    result = run_case(
        case,
        result_path,
        config_factory=lambda _argv: _measurement_config(),
        simulator_factory=IncompleteSimulator,
    )

    assert result["status"] == "bug"
    assert result["failure_reason"] == "completed_requests_mismatch"
    assert result["expected_requests"] == 1
    assert result["completed_requests"] == 0


def test_run_case_rejects_generated_request_count_mismatch(tmp_path) -> None:
    case = CaseSpec.for_scale(
        model="dense",
        total_gpus=32,
        mode="sequential",
        attempt_index=0,
        num_requests=1,
        prefill_tokens=16,
        decode_tokens=2,
        qps=1.0,
    )
    result_path = tmp_path / "request-count-mismatch.json"

    class MissingRequestSimulator:
        _parallel_mode = False
        metric_store = SimpleNamespace(
            get_total_requests=lambda: 0,
            get_completed_requests=lambda: 0,
        )

        def __init__(self, _config) -> None:
            self._profiler = SimpleNamespace()

        def run(self) -> None:
            return None

    result = run_case(
        case,
        result_path,
        config_factory=lambda _argv: _measurement_config(),
        simulator_factory=MissingRequestSimulator,
    )

    assert result["status"] == "bug"
    assert result["failure_reason"] == "request_accounting_mismatch"
    assert result["requests"] == 1
    assert result["expected_requests"] == 0
    assert result["completed_requests"] == 0


def test_run_case_rejects_success_without_processed_events(tmp_path) -> None:
    case = CaseSpec.for_scale(
        model="dense",
        total_gpus=32,
        mode="sequential",
        attempt_index=0,
        num_requests=1,
        prefill_tokens=16,
        decode_tokens=2,
        qps=1.0,
    )
    result_path = tmp_path / "zero-events.json"

    class ZeroEventSimulator:
        _parallel_mode = False
        metric_store = SimpleNamespace(
            get_total_requests=lambda: 1,
            get_completed_requests=lambda: 1,
        )

        def __init__(self, _config) -> None:
            self._profiler = SimpleNamespace()

        def run(self) -> None:
            return None

    result = run_case(
        case,
        result_path,
        config_factory=lambda _argv: _measurement_config(),
        simulator_factory=ZeroEventSimulator,
    )

    assert result["status"] == "bug"
    assert result["failure_reason"] == "non_positive_event_count"
    assert result["event_count"] == 0
    assert result["events_per_s"] == 0.0


def test_run_case_counts_parallel_cluster_events(tmp_path) -> None:
    case = CaseSpec.for_scale(
        model="moe",
        total_gpus=32,
        mode="parallel",
        attempt_index=0,
        num_requests=1,
        prefill_tokens=16,
        decode_tokens=2,
        qps=1.0,
    )

    class ParallelSimulator:
        _parallel_mode = True
        _cluster_simulators = {
            "prefill": SimpleNamespace(_events_processed=3),
            "decode": SimpleNamespace(_events_processed=4),
        }
        metric_store = SimpleNamespace(
            get_total_requests=lambda: 1,
            get_completed_requests=lambda: 1,
        )

        def __init__(self, _config) -> None:
            self._profiler = SimpleNamespace()

        def run(self) -> None:
            return None

    result = run_case(
        case,
        tmp_path / "parallel-success.json",
        config_factory=lambda _argv: _measurement_config(),
        simulator_factory=ParallelSimulator,
    )

    assert result["status"] == "success"
    assert result["mode"] == "parallel"
    assert result["event_count"] == 7
    assert result["events_per_s"] > 0
    assert result["effective_parallel_mode"] is True
    assert len(result["runner_sha256"]) == 64
    int(result["runner_sha256"], 16)


def test_run_case_rejects_missing_parallel_cluster_event_counter(
    tmp_path: Path,
) -> None:
    case = CaseSpec.for_scale(
        model="moe",
        total_gpus=32,
        mode="parallel",
        attempt_index=0,
        num_requests=1,
        prefill_tokens=16,
        decode_tokens=2,
        qps=1.0,
    )

    class IncompleteParallelSimulator:
        _parallel_mode = True
        _cluster_simulators = {
            "prefill": SimpleNamespace(_events_processed=3),
            "decode": SimpleNamespace(),
        }
        metric_store = SimpleNamespace(
            get_total_requests=lambda: 1,
            get_completed_requests=lambda: 1,
        )

        def __init__(self, _config) -> None:
            self._profiler = SimpleNamespace()

        def run(self) -> None:
            return None

    result = run_case(
        case,
        tmp_path / "missing-parallel-event-counter.json",
        config_factory=lambda _argv: _measurement_config(),
        simulator_factory=IncompleteParallelSimulator,
    )

    assert result["status"] == "bug"
    assert result["failure_reason"] == "AttributeError"
    assert "_events_processed" in result["stderr_tail"]


@pytest.mark.parametrize(
    ("requested_mode", "actual_parallel_mode"),
    [
        ("parallel", False),
        ("sequential", True),
    ],
)
def test_run_case_rejects_effective_mode_mismatch_before_run(
    tmp_path: Path,
    requested_mode: str,
    actual_parallel_mode: bool,
) -> None:
    case = CaseSpec.for_scale(
        model="dense",
        total_gpus=32,
        mode=requested_mode,
        attempt_index=0,
        num_requests=1,
        prefill_tokens=16,
        decode_tokens=2,
        qps=1.0,
    )
    run_called = False

    class WrongModeSimulator:
        _parallel_mode = actual_parallel_mode
        metric_store = SimpleNamespace(
            get_total_requests=lambda: 1,
            get_completed_requests=lambda: 0,
        )

        def __init__(self, _config) -> None:
            self._profiler = SimpleNamespace()

        def run(self) -> None:
            nonlocal run_called
            run_called = True

    result = run_case(
        case,
        tmp_path / f"mode-mismatch-{requested_mode}.json",
        config_factory=lambda _argv: _measurement_config(),
        simulator_factory=WrongModeSimulator,
    )

    assert run_called is False
    assert result["status"] == "bug"
    assert result["failure_reason"] == "RuntimeError"
    assert (
        "Effective simulator mode does not match CaseSpec: "
        f"requested_mode={requested_mode}, "
        f"simulator_parallel_mode={actual_parallel_mode}"
    ) in result["stderr_tail"]


def test_run_case_rejects_non_boolean_effective_mode_before_run(
    tmp_path: Path,
) -> None:
    case = CaseSpec.for_scale(
        model="dense",
        total_gpus=32,
        mode="parallel",
        attempt_index=0,
        num_requests=1,
        prefill_tokens=16,
        decode_tokens=2,
        qps=1.0,
    )
    run_called = False

    class InvalidModeSimulator:
        _parallel_mode = "parallel"
        metric_store = SimpleNamespace(
            get_total_requests=lambda: 1,
            get_completed_requests=lambda: 0,
        )

        def __init__(self, _config) -> None:
            self._profiler = SimpleNamespace()

        def run(self) -> None:
            nonlocal run_called
            run_called = True

    result = run_case(
        case,
        tmp_path / "non-boolean-mode.json",
        config_factory=lambda _argv: _measurement_config(),
        simulator_factory=InvalidModeSimulator,
    )

    assert run_called is False
    assert result["status"] == "bug"
    assert result["failure_reason"] == "RuntimeError"
    assert (
        "Simulator must expose a boolean _parallel_mode for wall-clock "
        "measurement, got 'parallel'"
    ) in result["stderr_tail"]


def test_run_case_rejects_missing_effective_mode_before_run(
    tmp_path: Path,
) -> None:
    case = CaseSpec.for_scale(
        model="dense",
        total_gpus=32,
        mode="parallel",
        attempt_index=0,
        num_requests=1,
        prefill_tokens=16,
        decode_tokens=2,
        qps=1.0,
    )
    run_called = False

    class MissingModeSimulator:
        metric_store = SimpleNamespace(
            get_total_requests=lambda: 1,
            get_completed_requests=lambda: 0,
        )

        def __init__(self, _config) -> None:
            self._profiler = SimpleNamespace()

        def run(self) -> None:
            nonlocal run_called
            run_called = True

    result = run_case(
        case,
        tmp_path / "missing-mode.json",
        config_factory=lambda _argv: _measurement_config(),
        simulator_factory=MissingModeSimulator,
    )

    assert run_called is False
    assert result["status"] == "bug"
    assert result["failure_reason"] == "RuntimeError"
    assert (
        "Simulator must expose a boolean _parallel_mode for wall-clock "
        "measurement, got None"
    ) in result["stderr_tail"]


def test_run_case_honors_falsey_factory_callables(tmp_path) -> None:
    case = CaseSpec.for_scale(
        model="dense",
        total_gpus=32,
        mode="parallel",
        attempt_index=0,
        num_requests=1,
        prefill_tokens=16,
        decode_tokens=2,
        qps=1.0,
    )

    class FalseyConfigFactory:
        called = False

        def __bool__(self) -> bool:
            return False

        def __call__(self, _argv):
            self.called = True
            return _measurement_config()

    class TinyParallelSimulator:
        _parallel_mode = True
        _cluster_simulators = {"dense": SimpleNamespace(_events_processed=1)}
        metric_store = SimpleNamespace(
            get_total_requests=lambda: 1,
            get_completed_requests=lambda: 1,
        )

        def __init__(self, _config) -> None:
            self._profiler = SimpleNamespace()

        def run(self) -> None:
            return None

    class FalseySimulatorFactory:
        called = False

        def __bool__(self) -> bool:
            return False

        def __call__(self, config):
            self.called = True
            return TinyParallelSimulator(config)

    config_factory = FalseyConfigFactory()
    simulator_factory = FalseySimulatorFactory()
    result = run_case(
        case,
        tmp_path / "falsey-factories.json",
        config_factory=config_factory,
        simulator_factory=simulator_factory,
    )

    assert config_factory.called is True
    assert simulator_factory.called is True
    assert result["status"] == "success"
    assert result["event_count"] == 1


def test_run_case_executes_real_tiny_sequential_simulation(tmp_path) -> None:
    case = CaseSpec.for_scale(
        model="dense",
        total_gpus=32,
        mode="sequential",
        attempt_index=0,
        num_requests=1,
        prefill_tokens=16,
        decode_tokens=2,
        qps=1.0,
    )
    result_path = tmp_path / "success.json"

    result = run_case(case, result_path)

    assert set(REQUIRED_RESULT_FIELDS).issubset(result)
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["status"] == "success"
    assert result["exit_code"] == 0
    assert result["expected_requests"] == 1
    assert result["completed_requests"] == 1
    assert result["sim_wallclock_s"] > 0
    assert result["init_s"] > 0
    assert result["total_proc_s"] >= result["sim_wallclock_s"]
    assert result["event_count"] > 0
    assert result["events_per_s"] > 0
    assert result["peak_rss_mb"] > 0
    assert result["python_executable"]
    assert len(result["git_sha"]) == 40
    assert json.loads(result_path.read_text(encoding="utf-8")) == result
