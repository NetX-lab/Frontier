"""PD+AF configuration and fail-fast contract tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import frontier.main as frontier_main
from frontier.config.config import (
    ClusterConfig,
    OrcaSchedulerConfig,
    ReplicaConfig,
    SimulationConfig,
    SyntheticRequestGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
)
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.types import ClusterType


REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_pdaf_cluster_config(
    *,
    model_name: str,
    decode_ffn_replicas: int = 1,
    **overrides,
) -> ClusterConfig:
    config_kwargs = dict(
        num_replicas=None,
        replica_config=ReplicaConfig(model_name=model_name),
        replica_scheduler_config=OrcaSchedulerConfig(),
        prefill_cluster_num_replicas=1,
        decode_attn_cluster_num_replicas=2,
        decode_ffn_cluster_num_replicas=decode_ffn_replicas,
        decode_attn_af_pipeline_num_micro_batch=1,
        decode_ffn_af_pipeline_num_micro_batch=1,
    )
    config_kwargs.update(overrides)
    return ClusterConfig(**config_kwargs)


@pytest.mark.parametrize(
    ("field_name", "decode_role", "invalid_value"),
    [
        (
            "decode_attn_replica_config_num_pipeline_stages",
            "decode_attn",
            0,
        ),
        (
            "decode_ffn_replica_config_num_pipeline_stages",
            "decode_ffn",
            0,
        ),
        (
            "decode_attn_replica_config_num_pipeline_stages",
            "decode_attn",
            2,
        ),
        (
            "decode_ffn_replica_config_num_pipeline_stages",
            "decode_ffn",
            2,
        ),
    ],
)
def test_pdaf_rejects_unsupported_decode_pipeline_parallelism(
    field_name: str,
    decode_role: str,
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{field_name}.*must be 1.*{decode_role}",
    ):
        _build_pdaf_cluster_config(
            model_name="llama2_7b_dense_example",
            **{field_name: invalid_value},
        )


@pytest.mark.parametrize(
    ("field_name", "effective_config_name"),
    [
        (
            "decode_attn_replica_config_num_pipeline_stages",
            "decode_attn_replica_config",
        ),
        (
            "decode_ffn_replica_config_num_pipeline_stages",
            "decode_ffn_replica_config",
        ),
    ],
)
def test_pdaf_accepts_explicit_unit_decode_pipeline_parallelism(
    field_name: str,
    effective_config_name: str,
) -> None:
    config = _build_pdaf_cluster_config(
        model_name="llama2_7b_dense_example",
        **{field_name: 1},
    )

    assert getattr(config, effective_config_name).num_pipeline_stages == 1


def test_pdaf_decode_pipeline_defaults_do_not_restrict_prefill_pipeline_parallelism() -> None:
    config = _build_pdaf_cluster_config(
        model_name="llama2_7b_dense_example",
        replica_config=ReplicaConfig(
            model_name="llama2_7b_dense_example",
            num_pipeline_stages=2,
        ),
    )

    assert config.prefill_replica_config.num_pipeline_stages == 2
    assert config.decode_attn_replica_config.num_pipeline_stages == 1
    assert config.decode_ffn_replica_config.num_pipeline_stages == 1


def test_pdaf_prefill_vllm_scheduler_receives_role_specific_chunked_prefill_controls() -> None:
    config = _build_pdaf_cluster_config(
        model_name="step-moe-noquant",
        prefill_replica_scheduler_config_type="vllm_v1",
        prefill_replica_scheduler_config_max_tokens_in_batch=64,
        prefill_replica_scheduler_config_enable_chunked_prefill=True,
        prefill_replica_scheduler_config_long_prefill_token_threshold=16,
    )
    prefill_config = config.get_cluster_configs_for_disaggregation()[
        ClusterType.PREFILL
    ]

    scheduler_config = (
        BaseClusterScheduler._get_cluster_specific_replica_scheduler_config(
            None,
            prefill_config,
            ClusterType.PREFILL,
        )
    )

    assert isinstance(scheduler_config, VllmV1SchedulerConfig)
    assert scheduler_config.max_tokens_in_batch == 64
    assert scheduler_config.enable_chunked_prefill is True
    assert scheduler_config.long_prefill_token_threshold == 16


def _build_decode_ffn_scheduler(
    cluster_config: ClusterConfig,
) -> RoundRobinClusterScheduler:
    decode_ffn_config = cluster_config.get_cluster_configs_for_disaggregation()[
        ClusterType.DECODE_FFN
    ]
    first_decode_ffn_replica_id = (
        int(cluster_config.prefill_cluster_num_replicas)
        + int(cluster_config.decode_attn_cluster_num_replicas)
    )
    replicas = {
        first_decode_ffn_replica_id + replica_ordinal: object()
        for replica_ordinal in range(
            int(cluster_config.decode_ffn_cluster_num_replicas)
        )
    }
    cluster = SimpleNamespace(
        cluster_type=ClusterType.DECODE_FFN,
        replicas=replicas,
    )
    with patch(
        "frontier.scheduler.cluster_scheduler.base_cluster_scheduler."
        "ReplicaSchedulerRegistry.get",
        return_value=SimpleNamespace(),
    ):
        return RoundRobinClusterScheduler(
            config=decode_ffn_config,
            cluster=cluster,
            request_generator_config=SyntheticRequestGeneratorConfig(),
        )


def test_pdaf_rejects_prefix_caching_at_architecture_boundary() -> None:
    config = object.__new__(SimulationConfig)
    config.sys_arch = "pd-af-disaggregation"
    config.enable_parallel_clusters = False
    config.cluster_config = SimpleNamespace(
        replica_scheduler_config=VllmV1SchedulerConfig(
            enable_prefix_caching=True
        )
    )

    with pytest.raises(ValueError, match="Prefix caching.*pd-af-disaggregation"):
        config._validate_open_source_release_architecture_guard()


def test_pdaf_cli_prefix_caching_guard_exits_without_traceback() -> None:
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
            "--replica_scheduler_config_type",
            "vllm_v1",
            "--vllm_v1_scheduler_config_enable_prefix_caching",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 1
    assert "Prefix caching is excluded" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_reraises_unrelated_value_error() -> None:
    unrelated_error = ValueError("unrelated configuration defect")

    with patch.object(
        frontier_main.SimulationConfig,
        "create_from_cli_args",
        side_effect=unrelated_error,
    ):
        with pytest.raises(ValueError) as exc_info:
            frontier_main.main()

    assert exc_info.value is unrelated_error


def test_pdaf_rejects_parallel_clusters_at_architecture_boundary() -> None:
    config = object.__new__(SimulationConfig)
    config.sys_arch = "pd-af-disaggregation"
    config.enable_parallel_clusters = True
    config.cluster_config = SimpleNamespace(
        replica_scheduler_config=VllmV1SchedulerConfig(
            enable_prefix_caching=False
        )
    )

    with pytest.raises(ValueError, match="--no-enable_parallel_clusters"):
        config._validate_open_source_release_architecture_guard()


def test_pdaf_cli_parallel_guard_exits_without_traceback() -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "frontier.main", "--sys_arch", "pd-af-disaggregation"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 1
    assert "--no-enable_parallel_clusters" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("sys_arch", "enable_parallel_clusters"),
    [
        ("co-location", True),
        ("pd-disaggregation", False),
    ],
)
def test_non_pdaf_architectures_keep_prefix_caching_surface(
    sys_arch: str,
    enable_parallel_clusters: bool,
) -> None:
    config = object.__new__(SimulationConfig)
    config.sys_arch = sys_arch
    config.enable_parallel_clusters = enable_parallel_clusters
    config.cluster_config = SimpleNamespace(
        replica_scheduler_config=VllmV1SchedulerConfig(
            enable_prefix_caching=True
        )
    )

    config._validate_open_source_release_architecture_guard()


def test_dense_pdaf_decode_attn_rejects_data_parallelism() -> None:
    with pytest.raises(ValueError, match="attn_dp.*fixed to 1"):
        ReplicaConfig(
            model_name="llama2_7b_dense_example",
            cluster_prefix="decode_attn",
            attn_dp=2,
            moe_tensor_parallel_size=0,
            moe_expert_parallel_size=0,
            total_expert_num=0,
            local_expert_num=0,
        )


@pytest.mark.parametrize("cluster_name", ["prefill", "decode", "monolithic"])
def test_shared_moe_roles_reject_attention_data_parallelism(
    cluster_name: str,
) -> None:
    with pytest.raises(ValueError, match="attn_dp.*fixed to 1"):
        ReplicaConfig(
            model_name="step-moe-noquant",
            cluster_prefix=cluster_name,
            attn_tensor_parallel_size=2,
            attn_dp=2,
            moe_tensor_parallel_size=2,
            moe_expert_parallel_size=2,
        )


@pytest.mark.parametrize(
    ("replica_kwargs", "expected_field"),
    [
        ({"moe_expert_parallel_size": 2}, "moe_expert_parallel_size=1"),
        ({"router_topk": 2}, "router_topk=1"),
    ],
)
def test_dense_pdaf_decode_ffn_rejects_moe_only_parallelism(
    replica_kwargs: dict[str, int],
    expected_field: str,
) -> None:
    replica_config = ReplicaConfig(
        model_name="llama2_7b_dense_example",
        cluster_prefix="decode_ffn",
        **replica_kwargs,
    )
    cluster_config = object.__new__(ClusterConfig)

    with pytest.raises(ValueError, match=rf"{expected_field}.*decode_ffn"):
        cluster_config._validate_replica_config(replica_config, "decode_ffn")


def test_dense_pdaf_decode_role_invariants_accept_unit_parallelism() -> None:
    cluster_config = object.__new__(ClusterConfig)
    decode_attn_config = ReplicaConfig(
        model_name="llama2_7b_dense_example",
        cluster_prefix="decode_attn",
        attn_dp=1,
        moe_tensor_parallel_size=0,
        moe_expert_parallel_size=0,
        total_expert_num=0,
        local_expert_num=0,
    )
    decode_ffn_config = ReplicaConfig(
        model_name="llama2_7b_dense_example",
        cluster_prefix="decode_ffn",
        moe_expert_parallel_size=1,
        router_topk=1,
    )

    cluster_config._validate_replica_config(decode_attn_config, "decode_attn")
    cluster_config._validate_replica_config(decode_ffn_config, "decode_ffn")


def test_moe_pdaf_config_accepts_multiple_decode_ffn_replicas_as_capacity() -> None:
    config = _build_pdaf_cluster_config(
        model_name="step-moe-noquant",
        decode_ffn_replicas=2,
    )
    assert config.decode_ffn_cluster_num_replicas == 2


def test_removed_multi_replica_opt_in_is_not_a_config_field() -> None:
    assert "allow_experiment_multi_decode_ffn_replicas" not in (
        ClusterConfig.__dataclass_fields__
    )


def test_dense_pdaf_runtime_allows_multiple_decode_ffn_replicas() -> None:
    config = _build_pdaf_cluster_config(
        model_name="llama2_7b_dense_example",
        decode_ffn_replicas=2,
    )

    scheduler = _build_decode_ffn_scheduler(config)

    assert scheduler._ffn_replica_ids == [3, 4]
    assert scheduler._ffn_expected_lanes_by_target == {
        3: [(1, None)],
        4: [(2, None)],
    }


def test_moe_pdaf_runtime_accepts_multiple_decode_ffn_replicas() -> None:
    config = _build_pdaf_cluster_config(
        model_name="step-moe-noquant",
        decode_ffn_replicas=2,
    )
    scheduler = _build_decode_ffn_scheduler(config)
    assert scheduler._ffn_replica_ids == [3, 4]


def test_moe_pdaf_runtime_preserves_multi_replica_capacity() -> None:
    config = _build_pdaf_cluster_config(
        model_name="step-moe-noquant",
        decode_ffn_replicas=2,
    )

    scheduler = _build_decode_ffn_scheduler(config)

    assert scheduler._ffn_replica_ids == [3, 4]


def test_decode_ffn_replica_help_distinguishes_dense_and_moe_contracts() -> None:
    help_text = ClusterConfig.__dataclass_fields__[
        "decode_ffn_cluster_num_replicas"
    ].metadata["help"]
    assert "independent FFN serving copy" in help_text
