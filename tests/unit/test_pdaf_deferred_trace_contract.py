"""Deferred StepFun trace-replay configuration contract tests."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from frontier.config.config import (
    ClusterConfig,
    MetricsConfig,
    OrcaSchedulerConfig,
    ReplicaConfig,
    SimulationConfig,
)
from frontier.config.flat_dataclass import create_flat_dataclass


DEFERRED_TRACE_FIELDS = (
    "moe_routing_trace_path",
    "decode_attn_initial_lane_trace_path",
    "decode_attn_steady_state_snapshot_path",
    "decode_attn_steady_state_measurement_report_path",
)
DEFERRED_ERROR_MARKER = "pd-af-disaggregation v0.3 trace-replay is deferred"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_pdaf_cluster(replica_config: ReplicaConfig) -> ClusterConfig:
    return ClusterConfig(
        num_replicas=None,
        replica_config=replica_config,
        replica_scheduler_config=OrcaSchedulerConfig(),
        prefill_cluster_num_replicas=1,
        decode_attn_cluster_num_replicas=1,
        decode_ffn_cluster_num_replicas=1,
        decode_attn_af_pipeline_num_micro_batch=1,
        decode_ffn_af_pipeline_num_micro_batch=1,
    )


def _build_config(
    tmp_path: Path,
    *,
    replica_config: ReplicaConfig,
    sys_arch: str = "pd-af-disaggregation",
) -> SimulationConfig:
    return SimulationConfig(
        sys_arch=sys_arch,
        enable_parallel_clusters=False,
        cluster_config=_build_pdaf_cluster(replica_config),
        metrics_config=MetricsConfig(
            output_dir=str(tmp_path / "metrics"),
            cache_dir=str(tmp_path / "cache"),
            run_id="deferred-trace-contract",
        ),
    )


def test_flat_cli_exposes_reference_complete_deferred_trace_fields() -> None:
    flat_config = create_flat_dataclass(SimulationConfig)
    flat_field_names = {field.name for field in fields(flat_config)}

    expected_cli_fields = {
        f"replica_config_{field_name}" for field_name in DEFERRED_TRACE_FIELDS
    }
    assert expected_cli_fields <= flat_field_names, (
        "Missing deferred trace CLI fields: "
        f"{sorted(expected_cli_fields - flat_field_names)}"
    )


@pytest.mark.parametrize("field_name", DEFERRED_TRACE_FIELDS)
def test_nonempty_deferred_trace_field_fails_at_pdaf_config_boundary(
    tmp_path: Path,
    field_name: str,
) -> None:
    replica_config = ReplicaConfig(model_name="llama2_7b_dense_example")
    setattr(replica_config, field_name, "/data/ycfeng/tmp/deferred-trace.jsonl")

    with pytest.raises(ValueError, match=DEFERRED_ERROR_MARKER):
        _build_config(tmp_path, replica_config=replica_config)


def test_all_deferred_trace_fields_are_reported_together(tmp_path: Path) -> None:
    replica_config = ReplicaConfig(model_name="llama2_7b_dense_example")
    for field_name in DEFERRED_TRACE_FIELDS:
        setattr(replica_config, field_name, f"/data/ycfeng/tmp/{field_name}.json")

    with pytest.raises(ValueError) as exc_info:
        _build_config(tmp_path, replica_config=replica_config)

    message = str(exc_info.value)
    assert DEFERRED_ERROR_MARKER in message
    for field_name in DEFERRED_TRACE_FIELDS:
        assert field_name in message


def test_empty_deferred_trace_fields_preserve_pdaf_config_construction(
    tmp_path: Path,
) -> None:
    config = _build_config(
        tmp_path,
        replica_config=ReplicaConfig(model_name="llama2_7b_dense_example"),
    )

    for field_name in DEFERRED_TRACE_FIELDS:
        assert getattr(config.cluster_config.replica_config, field_name, None) == ""
        for role_config_name in (
            "prefill_replica_config",
            "decode_attn_replica_config",
            "decode_ffn_replica_config",
        ):
            assert (
                getattr(getattr(config.cluster_config, role_config_name), field_name, None)
                == ""
            )


def test_deferred_trace_fields_are_preserved_by_replica_config_copy() -> None:
    source = ReplicaConfig(model_name="llama2_7b_dense_example")
    expected = {
        field_name: f"/data/ycfeng/tmp/{field_name}.json"
        for field_name in DEFERRED_TRACE_FIELDS
    }
    for field_name, value in expected.items():
        setattr(source, field_name, value)

    cluster_config = ClusterConfig(
        num_replicas=1,
        replica_config=source,
        replica_scheduler_config=OrcaSchedulerConfig(),
    )
    copied = cluster_config._create_replica_config_copy()

    assert {
        field_name: getattr(copied, field_name) for field_name in DEFERRED_TRACE_FIELDS
    } == expected


def test_trace_driven_moe_routing_fails_without_silent_fallback(tmp_path: Path) -> None:
    replica_config = ReplicaConfig(
        model_name="llama2_7b_dense_example",
        moe_routing_trace_path="/data/ycfeng/tmp/deferred-trace.jsonl",
    )

    with pytest.raises(ValueError, match=DEFERRED_ERROR_MARKER):
        _build_config(tmp_path, replica_config=replica_config)


def test_pdaf_deferred_trace_cli_exits_without_traceback() -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "frontier.main",
            "--sys_arch",
            "pd-af-disaggregation",
            "--no-enable_parallel_clusters",
            "--replica_config_moe_routing_trace_path",
            "/data/ycfeng/tmp/deferred-trace.jsonl",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 1
    assert DEFERRED_ERROR_MARKER in result.stderr
    assert "Configured fields: moe_routing_trace_path" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("sys_arch", ("co-location", "pd-disaggregation"))
def test_non_pdaf_default_config_behavior_is_unchanged(
    tmp_path: Path,
    sys_arch: str,
) -> None:
    replica_config = ReplicaConfig(model_name="llama2_7b_dense_example")
    if sys_arch == "co-location":
        cluster_config = ClusterConfig(
            num_replicas=1,
            replica_config=replica_config,
            replica_scheduler_config=OrcaSchedulerConfig(),
        )
    else:
        cluster_config = ClusterConfig(
            num_replicas=None,
            replica_config=replica_config,
            replica_scheduler_config=OrcaSchedulerConfig(),
            prefill_cluster_num_replicas=1,
            decode_cluster_num_replicas=1,
        )

    config = SimulationConfig(
        sys_arch=sys_arch,
        enable_parallel_clusters=False,
        cluster_config=cluster_config,
        metrics_config=MetricsConfig(
            output_dir=str(tmp_path / "metrics"),
            cache_dir=str(tmp_path / "cache"),
            run_id=f"default-{sys_arch.replace('-', '_')}",
        ),
    )

    assert config.sys_arch == sys_arch
