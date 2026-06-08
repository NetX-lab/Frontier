"""Unit tests for the open-source release architecture guard."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.config.config import (
    AICONFIGURATOR_BACKEND_RELEASE_ERROR,
    ClusterConfig,
    DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR,
    ReplicaConfig,
    SimulationConfig,
)
from frontier.config.kv_cache_transfer_config import AnalyticalKVCacheTransferConfig
from frontier.config.m2n_transfer_config import AnalyticalM2NTransferConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _git_ls_files(paths: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--", *paths],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(line for line in result.stdout.splitlines() if line)


def _assert_release_paths_are_tracked(paths: list[str]) -> None:
    assert _git_ls_files(paths) == sorted(paths)


def test_readme_referenced_figures_are_present_and_tracked() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    figure_paths = sorted(set(re.findall(r'<img src="(figs/[^"]+)"', readme)))

    assert figure_paths == [
        "figs/arch.png",
        "figs/icon.png",
        "figs/use_case_heterogeneous_gpu_allocation.png",
        "figs/use_case_pareto_frontier.png",
        "figs/use_case_rl_rollout_reconfiguration.png",
        "figs/use_case_stateful_reasoning_scheduler.png",
    ]
    for figure_path in figure_paths:
        full_path = PROJECT_ROOT / figure_path
        assert full_path.is_file(), figure_path
        assert full_path.stat().st_size > 0, figure_path
    _assert_release_paths_are_tracked(figure_paths)


def test_release_keeps_current_public_figures_tracked() -> None:
    current_public_figure_paths = [
        "figs/arch.png",
        "figs/icon.png",
        "figs/use_case_heterogeneous_gpu_allocation.png",
        "figs/use_case_pareto_frontier.png",
        "figs/use_case_rl_rollout_reconfiguration.png",
        "figs/use_case_stateful_reasoning_scheduler.png",
    ]

    for figure_path in current_public_figure_paths:
        full_path = PROJECT_ROOT / figure_path
        assert full_path.is_file(), figure_path
        assert full_path.stat().st_size > 0, figure_path
    _assert_release_paths_are_tracked(current_public_figure_paths)


def test_release_keeps_public_data_showcase_artifacts_tracked() -> None:
    required_paths = [
        "data/processed_traces/splitwise_code.csv",
        "data/processed_traces/splitwise_conv.csv",
        "data/profiling/compute/a100/config.yaml",
        "data/profiling/compute/rtx_pro_6000/attention_config.yaml",
        "data/profiling/compute/rtx_pro_6000/linear_op_config.yaml",
        "data/profiling/compute/rtx_pro_6000/moe_config.yaml",
        "data/profiling/compute/a800/qwen3-a3b-30b-moe/moe.csv",
        "data/profiling/network/a100_dgx/all_reduce.csv",
    ]

    for required_path in required_paths:
        full_path = PROJECT_ROOT / required_path
        assert full_path.is_file(), required_path
        assert full_path.stat().st_size > 0, required_path
    _assert_release_paths_are_tracked(required_paths)


def test_release_excludes_raw_session_aware_or_profiler_trace_artifacts() -> None:
    excluded_paths = [
        "data/processed_traces/mooncake_conversation_trace.csv",
        "data/profiling/profiler_traces",
        "data/profiling/compute/rtx_pro_6000/qwen2_dense_test/profiler_traces",
    ]

    for excluded_path in excluded_paths:
        assert not (PROJECT_ROOT / excluded_path).exists(), excluded_path


def test_release_keeps_public_log_and_output_case_examples_tracked() -> None:
    required_paths = [
        "logs/cluster_events/monolithic_20260606_110717.log",
        "outputs/metrics/meta_llama_llama_2_7b_hf/offline_batch/run_e2e_cross_validation_current/config.json",
        "outputs/metrics/meta_llama_llama_2_7b_hf/offline_batch/run_e2e_cross_validation_current/op_precision_metadata.csv",
        "outputs/metrics/meta_llama_llama_2_7b_hf/offline_batch/run_e2e_cross_validation_current/request_metrics.csv",
        "outputs/metrics/meta_llama_llama_2_7b_hf/offline_batch/run_e2e_cross_validation_current/system_metrics.json",
    ]

    for required_path in required_paths:
        full_path = PROJECT_ROOT / required_path
        assert full_path.is_file(), required_path
        assert full_path.stat().st_size > 0, required_path
    _assert_release_paths_are_tracked(required_paths)


def test_release_excludes_generated_output_trace_plot_and_ledger_artifacts() -> None:
    tracked_outputs = _git_ls_files(["outputs"])
    forbidden_fragments = [
        "/chrome_trace.json",
        "/frontier_stage_batch_ledger.jsonl",
        "/metrics_ground_truth.jsonl",
        "/plots/",
    ]

    assert tracked_outputs
    for tracked_output in tracked_outputs:
        for forbidden_fragment in forbidden_fragments:
            assert forbidden_fragment not in tracked_output, tracked_output


def test_readme_documents_public_case_artifact_examples() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    required_snippets = [
        "pre-release-v0.1",
        "models co-location only",
        "PDD",
        "AFD",
        "ASTRA-Sim analytical",
        "--cc_backend_config_type astra_sim_analytical",
        "collective_sim is optional",
        "--cc_backend_config_type collective_sim",
        "examples/",
        "fixtures/",
    ]

    for snippet in required_snippets:
        assert snippet in readme


def test_release_case_assets_do_not_embed_local_paths_or_secret_literals() -> None:
    marker_fragments = [
        "/" + "local/" + "yc" + "feng",
        "/" + "workspace/" + "yc" + "feng",
        "/" + "uac/gds/" + "yc" + "feng",
        "/" + "home/" + "yc" + "feng",
        "/" + "data/d0/gds/" + "yc" + "feng",
        "935" + "953" + "068",
        "TODO-" + "YC",
        "sk-",
        "ghp_",
        "github_pat_",
        "AKIA",
    ]
    pattern = "|".join(re.escape(fragment) for fragment in marker_fragments)
    result = subprocess.run(
        ["git", "grep", "-n", "-I", "-E", pattern, "--", "figs", "data", "logs", "outputs"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode in (0, 1)
    assert result.stdout == ""


def test_public_figures_have_no_personal_metadata_markers() -> None:
    figure_paths = [
        "figs/arch.png",
        "figs/icon.png",
        "figs/use_case_heterogeneous_gpu_allocation.png",
        "figs/use_case_pareto_frontier.png",
        "figs/use_case_rl_rollout_reconfiguration.png",
        "figs/use_case_stateful_reasoning_scheduler.png",
    ]
    forbidden_markers = [
        b"yichengfeng",
        b"yc" + b"feng",
        b"/local/",
        b"/home/",
        b"/uac/",
        b"/workspace/",
        b"935" + b"953" + b"068",
        b"TODO-" + b"YC",
    ]

    for figure_path in figure_paths:
        figure_bytes = (PROJECT_ROOT / figure_path).read_bytes()
        for forbidden_marker in forbidden_markers:
            assert forbidden_marker not in figure_bytes, figure_path


def test_release_tree_has_no_personal_yc_markers() -> None:
    personal_marker = "Y" + "C:"
    result = subprocess.run(
        ["git", "grep", "-n", "-I", personal_marker, "--", ".", ":!.git"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode in (0, 1)
    assert result.stdout == ""


def _make_config(sys_arch: str) -> SimulationConfig:
    config = SimulationConfig.__new__(SimulationConfig)
    config.sys_arch = sys_arch
    return config


@pytest.mark.parametrize("sys_arch", ["pd-disaggregation", "pd-af-disaggregation"])
def test_open_source_release_guard_rejects_disaggregated_architectures(
    sys_arch: str,
) -> None:
    config = _make_config(sys_arch)

    with pytest.raises(ValueError) as exc_info:
        SimulationConfig._validate_open_source_release_architecture_guard(config)

    assert str(exc_info.value) == DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR


def test_open_source_release_guard_allows_co_location() -> None:
    config = _make_config("co-location")

    SimulationConfig._validate_open_source_release_architecture_guard(config)


def test_open_source_release_guard_rejects_pd_af_cuda_graph_surface() -> None:
    config = _make_config("co-location")
    config.use_cuda_graph = True

    with pytest.raises(ValueError) as exc_info:
        SimulationConfig._validate_open_source_release_architecture_guard(config)

    assert str(exc_info.value) == DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR


@pytest.mark.parametrize("sys_arch", ["pd-disaggregation", "pd-af-disaggregation"])
def test_simulator_rejects_disaggregated_config_even_if_post_init_was_bypassed(
    sys_arch: str,
) -> None:
    from frontier.simulator import Simulator

    config = _make_config(sys_arch)

    with pytest.raises(ValueError) as exc_info:
        Simulator(config)

    assert str(exc_info.value) == DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR


def test_simulation_config_defaults_to_co_location() -> None:
    field = SimulationConfig.__dataclass_fields__["sys_arch"]

    assert field.default == "co-location"


def test_default_simulation_config_constructs_co_location() -> None:
    config = SimulationConfig()

    assert config.sys_arch == "co-location"
    assert config.cluster_config.cluster_type.name == "MONOLITHIC"


def test_replica_config_uses_release_default_model_name() -> None:
    config = ReplicaConfig()

    assert config.model_name == "meta-llama/Llama-2-7b-hf"
    assert config.model_config.get_name() == "meta-llama/Llama-2-7b-hf"


def test_cli_reconstructs_co_location_with_release_default_model_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "frontier.main",
            "--metrics_config_output_dir",
            str(tmp_path / "metrics"),
        ],
    )

    config = SimulationConfig.create_from_cli_args()

    assert config.sys_arch == "co-location"
    assert config.cluster_config.cluster_type.name == "MONOLITHIC"
    assert config.cluster_config.replica_config.model_name == "meta-llama/Llama-2-7b-hf"


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "kv_cache_transfer_config": AnalyticalKVCacheTransferConfig(
                network_bandwidth_gbps=101.0
            )
        },
        {
            "m2n_transfer_config": AnalyticalM2NTransferConfig(
                memory_bandwidth_gbps=201.0
            )
        },
    ],
)
def test_simulation_config_rejects_non_default_transfer_config_surfaces(
    kwargs: dict,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        SimulationConfig(**kwargs)

    assert str(exc_info.value) == DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR


def test_simulation_config_rejects_aiconfigurator_backend_surface() -> None:
    from frontier.cc_backend.cc_backend_config import AiconfiguratorCCBackendConfig

    with pytest.raises(ValueError) as exc_info:
        SimulationConfig(
            cluster_config=ClusterConfig(
                cc_backend_config=AiconfiguratorCCBackendConfig()
            )
        )

    assert str(exc_info.value) == AICONFIGURATOR_BACKEND_RELEASE_ERROR


def test_cc_backend_factory_rejects_aiconfigurator_backend_surface() -> None:
    from frontier.cc_backend.cc_backend_config import AiconfiguratorCCBackendConfig
    from frontier.cc_backend.cc_backend_factory import CCBackendFactory
    from frontier.types import CCBackendType, ClusterType

    with pytest.raises(ValueError) as exc_info:
        CCBackendFactory.create(
            backend_type=CCBackendType.AICONFIGURATOR,
            config=AiconfiguratorCCBackendConfig(),
            cluster_type=ClusterType.MONOLITHIC,
            device_type="h100",
            network_device="h100",
            num_devices=1,
        )

    assert str(exc_info.value) == AICONFIGURATOR_BACKEND_RELEASE_ERROR


def test_aiconfigurator_backend_constructor_rejects_release_surface_directly() -> None:
    from frontier.cc_backend.backends.aiconfigurator_cc_backend import (
        AiconfiguratorCCBackend,
    )
    from frontier.cc_backend.cc_backend_config import AiconfiguratorCCBackendConfig
    from frontier.types import ClusterType

    with pytest.raises(ValueError) as exc_info:
        AiconfiguratorCCBackend(
            config=AiconfiguratorCCBackendConfig(),
            cluster_type=ClusterType.MONOLITHIC,
            device_type="h100",
            network_device="h100",
            num_devices=1,
        )

    assert str(exc_info.value) == AICONFIGURATOR_BACKEND_RELEASE_ERROR


def test_cluster_config_rejects_disaggregated_replica_counts() -> None:
    with pytest.raises(ValueError) as exc_info:
        ClusterConfig(prefill_cluster_num_replicas=1, decode_cluster_num_replicas=1)

    assert str(exc_info.value) == DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR


def test_cluster_config_rejects_disaggregated_cluster_specific_fields() -> None:
    with pytest.raises(ValueError) as exc_info:
        ClusterConfig(prefill_replica_config_attn_tensor_parallel_size=2)

    assert str(exc_info.value) == DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR


def test_cluster_config_rejects_disaggregated_periodic_scheduling_clusters() -> None:
    from frontier.types import ClusterType

    with pytest.raises(ValueError) as exc_info:
        ClusterConfig(periodic_scheduling_clusters=[ClusterType.DECODE_ATTN])

    assert str(exc_info.value) == DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR


def test_cluster_config_rejects_af_pipeline_surface() -> None:
    with pytest.raises(ValueError) as exc_info:
        ClusterConfig(af_pipeline_num_micro_batch=1)

    assert str(exc_info.value) == DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR


def test_cluster_config_rejects_disaggregated_fields_even_with_monolithic_cluster_type() -> None:
    from frontier.types import ClusterType

    with pytest.raises(ValueError) as exc_info:
        ClusterConfig(
            cluster_type=ClusterType.MONOLITHIC,
            prefill_cluster_num_replicas=1,
        )

    assert str(exc_info.value) == DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR


@pytest.mark.parametrize(
    "cluster_type_name",
    ["PREFILL", "DECODE", "DECODE_ATTN", "DECODE_FFN"],
)
def test_cluster_config_rejects_direct_non_monolithic_cluster_type(
    cluster_type_name: str,
) -> None:
    from frontier.types import ClusterType

    with pytest.raises(ValueError) as exc_info:
        ClusterConfig(cluster_type=getattr(ClusterType, cluster_type_name))

    assert str(exc_info.value) == DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR


@pytest.mark.parametrize("sys_arch", ["pd-disaggregation", "pd-af-disaggregation"])
def test_cli_rejects_disaggregated_architecture_before_nested_config_validation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sys_arch: str,
) -> None:
    from frontier import main as frontier_main

    monkeypatch.setattr(sys, "argv", ["frontier.main", "--sys_arch", sys_arch])

    with pytest.raises(SystemExit) as exc_info:
        frontier_main.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert captured.out == ""
    assert captured.err.strip() == DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR


def test_cli_rejects_disaggregated_cluster_parameters_before_config_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from frontier import main as frontier_main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "frontier.main",
            "--cluster_config_prefill_cluster_num_replicas",
            "1",
            "--cluster_config_decode_cluster_num_replicas",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        frontier_main.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert captured.out == ""
    assert captured.err.strip() == DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR


def test_cli_rejects_disaggregated_cluster_specific_parameters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from frontier import main as frontier_main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "frontier.main",
            "--cluster_config_decode_replica_config_attn_tensor_parallel_size",
            "2",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        frontier_main.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert captured.out == ""
    assert captured.err.strip() == DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR


def test_cli_rejects_negated_disaggregated_cluster_bool_parameter(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from frontier import main as frontier_main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "frontier.main",
            "--no-cluster_config_decode_attn_use_cuda_graph",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        frontier_main.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert captured.out == ""
    assert captured.err.strip() == DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR


@pytest.mark.parametrize(
    "argv",
    [
        ["--no-cluster_config_prefill_replica_config_extend_ep_across_dp"],
        ["--no-cluster_config_decode_replica_config_extend_ep_across_dp"],
        ["--no-cluster_config_decode_attn_use_cuda_graph"],
        ["--no-cluster_config_decode_ffn_replica_config_extend_ep_across_dp"],
    ],
)
def test_cli_disaggregated_cluster_guard_detects_negated_bool_prefixes(
    argv: list[str],
) -> None:
    from frontier import main as frontier_main

    assert frontier_main._has_disaggregated_cluster_option(argv) is True


def test_cli_rejects_af_pipeline_parameter_before_config_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from frontier import main as frontier_main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "frontier.main",
            "--cluster_config_af_pipeline_num_micro_batch",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        frontier_main.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert captured.out == ""
    assert captured.err.strip() == DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR


def test_cli_disaggregated_cluster_guard_does_not_reject_decode_cuda_graph_mode() -> None:
    from frontier import main as frontier_main

    assert (
        frontier_main._has_disaggregated_cluster_option(
            ["--decode_cuda_graph_mode", "piecewise"]
        )
        is False
    )


def test_cli_rejects_pd_af_cuda_graph_surface_before_config_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from frontier import main as frontier_main

    monkeypatch.setattr(sys, "argv", ["frontier.main", "--use_cuda_graph"])

    with pytest.raises(SystemExit) as exc_info:
        frontier_main.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert captured.out == ""
    assert captured.err.strip() == DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR


def test_cli_rejects_pd_af_cuda_graph_equals_true_surface_before_argparse_usage(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from frontier import main as frontier_main

    monkeypatch.setattr(sys, "argv", ["frontier.main", "--use_cuda_graph=true"])

    with pytest.raises(SystemExit) as exc_info:
        frontier_main.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert captured.out == ""
    assert captured.err.strip() == DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR


@pytest.mark.parametrize(
    "argv",
    [
        ["frontier.main", "--kv_cache_transfer_config_type", "analytical"],
        ["frontier.main", "--m2n_transfer_config_type", "analytical"],
        [
            "frontier.main",
            "--analytical_kv_cache_transfer_config_network_bandwidth_gbps",
            "101",
        ],
        [
            "frontier.main",
            "--analytical_m2n_transfer_config_memory_bandwidth_gbps",
            "201",
        ],
    ],
)
def test_cli_rejects_transfer_config_surfaces_before_config_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    from frontier import main as frontier_main

    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        frontier_main.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert captured.out == ""
    assert captured.err.strip() == DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR


def test_cli_rejects_aiconfigurator_backend_before_missing_submodule_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from frontier import main as frontier_main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "frontier.main",
            "--cc_backend_config_type",
            "aiconfigurator",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        frontier_main.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert captured.out == ""
    assert captured.err.strip() == AICONFIGURATOR_BACKEND_RELEASE_ERROR


def test_public_cli_help_does_not_advertise_aiconfigurator_backend() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "frontier.main", "-h"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "aiconfigurator" not in result.stdout.lower()
    assert "aiconfigurator" not in result.stderr.lower()


def test_cluster_config_public_help_does_not_advertise_aiconfigurator_backend() -> None:
    for field_name, field_info in ClusterConfig.__dataclass_fields__.items():
        help_text = str(field_info.metadata.get("help", ""))
        assert "aiconfigurator" not in help_text.lower(), field_name


def test_cc_backend_public_api_does_not_advertise_aiconfigurator_backend() -> None:
    import frontier.cc_backend as cc_backend
    import frontier.cc_backend.backends as cc_backends
    from frontier.types import CCBackendType

    assert "AiconfiguratorCCBackendConfig" not in cc_backend.__all__
    assert "AiconfiguratorCCBackend" not in cc_backend.__all__
    assert "AiconfiguratorCCBackend" not in cc_backends.__all__
    assert "available" not in (CCBackendType.AICONFIGURATOR.__doc__ or "").lower()


@pytest.mark.parametrize(
    "argv",
    [
        ["--cc_backend_config_type", "aiconfigurator"],
        ["--cc_backend_config_type", "AICONFIGURATOR"],
        ["--cc_backend_config_type=aiconfigurator"],
        ["--cluster_config_cc_backend_config_type", "aiconfigurator"],
        ["--cluster_config_cc_backend_config_type=aiconfigurator"],
        ["--cluster_config_prefill_cc_backend_config_type", "aiconfigurator"],
        ["--cluster_config_decode_cc_backend_config_type=aiconfigurator"],
        ["--cluster_config_decode_attn_cc_backend_config_type", "aiconfigurator"],
        ["--cluster_config_decode_ffn_cc_backend_config_type=aiconfigurator"],
        ["--aiconfigurator_cc_backend_config_repo_root", "sota-infer-engine/aiconfigurator"],
    ],
)
def test_cli_aiconfigurator_backend_guard_detects_public_backend_surfaces(
    argv: list[str],
) -> None:
    from frontier import main as frontier_main

    assert frontier_main._has_aiconfigurator_backend_option(argv) is True


@pytest.mark.parametrize(
    "argv",
    [
        ["--cc_backend_config_type", "collective_sim"],
        ["--cc_backend_config_type=vidur"],
        ["--cluster_config_cc_backend_config_type", "astra_sim_analytical"],
        ["--foo_cc_backend_config_type", "aiconfigurator"],
        ["--not_aiconfigurator_cc_backend_config_repo_root", "ignored"],
    ],
)
def test_cli_aiconfigurator_backend_guard_does_not_reject_other_surfaces(
    argv: list[str],
) -> None:
    from frontier import main as frontier_main

    assert frontier_main._has_aiconfigurator_backend_option(argv) is False


@pytest.mark.parametrize(
    "cluster_type_name",
    ["PREFILL", "DECODE", "DECODE_ATTN", "DECODE_FFN"],
)
def test_cluster_scheduler_constructor_rejects_non_monolithic_clusters(
    cluster_type_name: str,
) -> None:
    from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
        RoundRobinClusterScheduler,
    )
    from frontier.types import ClusterType

    cluster = SimpleNamespace(
        cluster_type=getattr(ClusterType, cluster_type_name),
        replicas={},
    )
    config = SimpleNamespace(
        replica_config=SimpleNamespace(data_parallel_size=1),
    )

    with pytest.raises(ValueError) as exc_info:
        RoundRobinClusterScheduler(
            config=config,
            cluster=cluster,
            request_generator_config=SimpleNamespace(),
            predictor=None,
        )

    assert str(exc_info.value) == DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR
