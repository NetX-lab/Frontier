"""Release-readiness regression tests for the co-location review findings."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_BASE = REPO_ROOT / "tests/debug/e2e-level/monolith_mode/scripts/test_base.sh"
DEBUG_E2E_SCRIPTS_DIR = REPO_ROOT / "tests/debug/e2e-level/monolith_mode/scripts"
COLOCATION_SCRIPTS = [
    REPO_ROOT / "examples/architecture/co-location/offline/dense_model_basic.sh",
    REPO_ROOT / "examples/architecture/co-location/offline/moe_model_basic.sh",
    REPO_ROOT / "examples/architecture/co-location/offline/thinking_mode_basic.sh",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_shell_int(content: str, name: str) -> int:
    match = re.search(rf"^{name}=([0-9]+)$", content, re.MULTILINE)
    assert match is not None, f"Missing integer shell variable: {name}"
    return int(match.group(1))


def test_config_optimizer_help_prints_without_argparse_percent_crash() -> None:
    result = subprocess.run(
        [
            "python",
            "-m",
            "frontier.config_optimizer.config_explorer.main",
            "-h",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--min-search-granularity" in result.stdout
    assert "Minimum search granularity for capacity (%)" in result.stdout


def test_debug_e2e_base_does_not_default_to_private_conda_path() -> None:
    content = _read(TEST_BASE)

    assert "vidur_te" not in content
    assert 'CONDA_ENV="${CONDA_ENV:-frontier}"' in content


def test_debug_e2e_conda_activation_temporarily_disables_nounset() -> None:
    content = _read(TEST_BASE)

    assert "local nounset_was_enabled=0" in content
    assert "set +u" in content
    assert 'conda activate "$env_path"' in content
    assert "set -u" in content


def test_debug_e2e_base_uses_safe_pythonpath_expansion() -> None:
    content = _read(TEST_BASE)

    assert 'export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"' not in content
    assert 'prepend_project_root_to_pythonpath()' in content
    assert 'export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"' in content


@pytest.mark.parametrize("script", COLOCATION_SCRIPTS, ids=lambda p: p.name)
def test_colocation_examples_prepend_repo_root_to_existing_pythonpath(script: Path) -> None:
    content = _read(script)

    assert 'REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"' in content
    assert 'export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"' in content
    assert 'export PYTHONPATH="${PYTHONPATH:-' not in content



def test_readme_debug_scripts_do_not_use_post_increment_under_set_e() -> None:
    for script in [
        REPO_ROOT / "tests/debug/e2e-level/monolith_mode/scripts/test_dense_tp2_pp2_dummy.sh",
        REPO_ROOT / "tests/debug/e2e-level/monolith_mode/scripts/test_moe_tp2_ep2_pp2_dummy.sh",
    ]:
        content = _read(script)
        assert "((validation_passed++))" not in content
        assert "((validation_failed++))" not in content


def test_debug_e2e_base_resolves_latest_canonical_metrics_run_dir(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    older_metrics = (
        output_root
        / "llama_3_2_1b_instruct"
        / "offline_batch"
        / "run_001"
        / "request_metrics.csv"
    )
    newer_metrics = (
        output_root
        / "llama_3_2_1b_instruct"
        / "offline_batch"
        / "run_002"
        / "request_metrics.csv"
    )
    older_metrics.parent.mkdir(parents=True)
    newer_metrics.parent.mkdir(parents=True)
    older_metrics.write_text("request_id,request_e2e_time\n0,1.0\n", encoding="utf-8")
    newer_metrics.write_text("request_id,request_e2e_time\n0,2.0\n", encoding="utf-8")
    os.utime(older_metrics, (1_700_000_000, 1_700_000_000))
    os.utime(newer_metrics, (1_700_000_100, 1_700_000_100))

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{TEST_BASE}"; find_latest_request_metrics_csv "{output_root}"',
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(newer_metrics)


def test_release_debug_scripts_use_canonical_metrics_resolver() -> None:
    for script in [
        REPO_ROOT / "tests/debug/e2e-level/monolith_mode/scripts/test_dense_tp2_pp2_dummy.sh",
        REPO_ROOT / "tests/debug/e2e-level/monolith_mode/scripts/test_moe_tp2_ep2_pp2_dummy.sh",
    ]:
        content = _read(script)

        assert 'find_latest_request_metrics_csv "$OUTPUT_SUBDIR"' in content
        assert 'ls -td "$OUTPUT_SUBDIR"/*/' not in content
        assert '"$OUTPUT_SUBDIR/request_metrics.csv"' not in content
        assert '${latest_output}request_metrics.csv' not in content
        assert '${latest_output_dir}request_metrics.csv' not in content


def test_all_debug_e2e_scripts_use_canonical_metrics_artifact_resolvers() -> None:
    for script in sorted(DEBUG_E2E_SCRIPTS_DIR.glob("test_*.sh")):
        if script == TEST_BASE:
            continue

        content = _read(script)
        if "request_metrics.csv" not in content:
            continue

        assert (
            "find_latest_request_metrics_csv" in content
            or "find_latest_metrics_run_dir" in content
        ), script
        assert 'ls -td "$OUTPUT_SUBDIR"/*/' not in content, script
        assert '"$OUTPUT_SUBDIR/request_metrics.csv"' not in content, script
        assert '${latest_output}request_metrics.csv' not in content, script
        assert '${latest_output_dir}request_metrics.csv' not in content, script
        assert '${latest_output_dir}system_metrics.json' not in content, script
        assert '${latest_output_dir}batch_metrics.csv' not in content, script


def test_all_debug_e2e_scripts_use_prefix_validation_counter_increment() -> None:
    for script in sorted(DEBUG_E2E_SCRIPTS_DIR.glob("test_*.sh")):
        content = _read(script)

        assert "((validation_passed++))" not in content, script
        assert "((validation_failed++))" not in content, script


def test_readme_moe_debug_script_uses_valid_shared_parallel_domain() -> None:
    content = _read(
        REPO_ROOT / "tests/debug/e2e-level/monolith_mode/scripts/test_moe_tp2_ep2_pp2_dummy.sh"
    )

    assert "ATTN_TP_SIZE=4" in content
    assert "MOE_TP_SIZE=2" in content
    assert '"--replica_config_attn_tensor_parallel_size" "$ATTN_TP_SIZE"' in content
    assert '"--replica_config_moe_tensor_parallel_size" "$MOE_TP_SIZE"' in content


def test_readme_moe_debug_script_header_uses_current_tp_terms() -> None:
    content = _read(
        REPO_ROOT / "tests/debug/e2e-level/monolith_mode/scripts/test_moe_tp2_ep2_pp2_dummy.sh"
    )
    header = "\n".join(content.splitlines()[:25])

    assert "Mode (TP=2, EP=2, PP=2, DP=1)" not in header
    assert "Attn_TP=4, MoE_TP=2, EP=2, PP=2, DP=1" in header


def test_readme_moe_debug_script_satisfies_shared_parallel_domain() -> None:
    content = _read(
        REPO_ROOT / "tests/debug/e2e-level/monolith_mode/scripts/test_moe_tp2_ep2_pp2_dummy.sh"
    )

    attn_tp = _extract_shell_int(content, "ATTN_TP_SIZE")
    moe_tp = _extract_shell_int(content, "MOE_TP_SIZE")
    ep = _extract_shell_int(content, "EP_SIZE")
    dp = _extract_shell_int(content, "DP_SIZE")

    assert attn_tp * dp == moe_tp * ep


def test_readme_quick_start_calls_out_collective_sim_prerequisite() -> None:
    content = _read(REPO_ROOT / "README.md")

    assert "--cc_backend_config_type analytical" in content
    assert "--cc_backend_config_type collective_sim" in content
    assert "frontier/cc_backend/backends/collective-sim/sim/datacenter/htsim_ndp" in content


def test_docs_explain_kaleido_is_png_only_optional_dependency() -> None:
    content = _read(REPO_ROOT / "README.md") + "\n" + _read(REPO_ROOT / "examples/README.md")

    assert "kaleido" in content.lower()
    assert "PNG" in content
    assert "CSV/JSON" in content


def test_dummy_mode_does_not_load_profiling_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from frontier.config import (
        MetricsConfig,
        RandomForrestExecutionTimePredictorConfig,
        ReplicaConfig,
        VllmV1SchedulerConfig,
    )
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )
    from frontier.types import ClusterType

    class _DummyPredictor(SklearnExecutionTimePredictor):
        def _get_estimator(self):
            return None

        def _get_grid_search_params(self):
            return {}

    def fail_if_metadata_is_loaded(self) -> None:  # noqa: ANN001
        raise AssertionError("dummy mode must not load profiling metadata files")

    for method_name in [
        "_initialize_pp_stage_boundary_lookup",
        "_initialize_pp_receiver_head_lookup",
        "_initialize_pp_producer_send_path_lookup",
        "_initialize_pp_prefill_consumer_active_lookup",
    ]:
        monkeypatch.setattr(_DummyPredictor, method_name, lambda self: None)
    monkeypatch.setattr(
        _DummyPredictor,
        "_register_profiling_metadata_from_files",
        fail_if_metadata_is_loaded,
    )

    predictor_config = RandomForrestExecutionTimePredictorConfig(enable_dummy_mode=True)
    replica_config = ReplicaConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        device="a100",
        network_device="a100_pairwise_nvlink",
        attn_tensor_parallel_size=2,
    )
    metrics_config = MetricsConfig(output_dir=str(tmp_path / "sim_out"))
    scheduler_config = VllmV1SchedulerConfig()

    _DummyPredictor(
        predictor_config=predictor_config,
        replica_config=replica_config,
        replica_scheduler_config=scheduler_config,
        metrics_config=metrics_config,
        cluster_type=ClusterType.MONOLITHIC,
        training_file_paths=None,
    )


def test_non_dummy_shared_model_manager_registers_profiling_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from frontier.config import (
        MetricsConfig,
        RandomForrestExecutionTimePredictorConfig,
        ReplicaConfig,
        VllmV1SchedulerConfig,
    )
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )
    from frontier.types import ClusterType

    class _DummyPredictor(SklearnExecutionTimePredictor):
        def _get_estimator(self):
            return None

        def _get_grid_search_params(self):
            return {}

    class _FakeModelManager:
        def get_models(self) -> dict[str, dict[str, object]]:
            return {"eager": {}, "kernel_only": {}}

        def get_models_for_cluster(self, cluster_type: ClusterType) -> dict[str, dict[str, object]]:
            return {"eager": {}, "kernel_only": {}}

    metadata_calls = []

    for method_name in [
        "_initialize_pp_stage_boundary_lookup",
        "_initialize_pp_receiver_head_lookup",
        "_initialize_pp_producer_send_path_lookup",
        "_initialize_pp_prefill_consumer_active_lookup",
    ]:
        monkeypatch.setattr(_DummyPredictor, method_name, lambda self: None)
    monkeypatch.setattr(
        _DummyPredictor,
        "_predict_from_models_for_family",
        lambda self, measurement_type, models: {},
    )
    monkeypatch.setattr(
        _DummyPredictor,
        "_register_profiling_metadata_from_files",
        lambda self: metadata_calls.append(True),
    )

    predictor_config = RandomForrestExecutionTimePredictorConfig(enable_dummy_mode=False)
    replica_config = ReplicaConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        device="a100",
        network_device="a100_pairwise_nvlink",
        attn_tensor_parallel_size=2,
    )
    metrics_config = MetricsConfig(output_dir=str(tmp_path / "sim_out"))
    scheduler_config = VllmV1SchedulerConfig()

    _DummyPredictor(
        predictor_config=predictor_config,
        replica_config=replica_config,
        replica_scheduler_config=scheduler_config,
        metrics_config=metrics_config,
        cluster_type=ClusterType.MONOLITHIC,
        training_file_paths=None,
        model_manager=_FakeModelManager(),
    )

    assert metadata_calls == [True]
