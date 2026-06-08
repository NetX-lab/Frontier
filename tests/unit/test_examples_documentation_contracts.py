from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_readme_documents_current_examples_surface() -> None:
    readme = _read("README.md")

    for script in (
        "examples/architecture/co-location/dense_model_basic.sh",
        "examples/architecture/co-location/moe_model_basic.sh",
        "examples/architecture/co-location/thinking_mode_basic.sh",
        "examples/architecture/co-location/moe_spec_dec.sh",
        "examples/architecture/co-location/moe_prefix_caching.sh",
        "examples/profiling/profile_linear_op.sh",
        "examples/profiling/profile_attention_chunked_prefill.sh",
        "examples/profiling/profile_moe.sh",
        "examples/profiling/smoke_simulator_dense_csv.sh",
        "examples/profiling/smoke_simulator_moe_csv.sh",
    ):
        assert script in readme

    assert "decode_cuda_graph_mode" in readme
    assert "Chunked Prefill" in readme
    assert "Speculative Decoding / MTP" in readme
    assert "Prefix Caching" in readme
    assert "data/profiling/compute" in readme
    assert "├── fixtures/" in readme
    assert "examples/fixtures/prefix_cache_shared_session_trace.csv" in readme
    assert "tests/integration/fixtures/prefix_cache_shared_session_trace.csv" not in readme


def test_examples_docs_list_all_colocation_scripts_and_metrics_behavior() -> None:
    examples_readme = _read("examples/README.md")
    architecture_readme = _read("examples/architecture/README.md")
    combined = examples_readme + "\n" + architecture_readme

    for script in (
        "dense_model_basic.sh",
        "moe_model_basic.sh",
        "thinking_mode_basic.sh",
        "moe_spec_dec.sh",
        "moe_prefix_caching.sh",
    ):
        assert script in combined

    assert "CSV/JSON metrics" in combined
    assert "metrics/traces/plot outputs disabled by default" not in combined
    assert "--no-metrics_config_*" not in combined
    assert "full_decode_only" in combined
    assert "decode_cuda_graph_mode=none" in combined
    assert "├── fixtures/" in examples_readme
    assert "examples/fixtures/prefix_cache_shared_session_trace.csv" in combined


def test_examples_docs_default_to_astra_sim_and_mark_collective_sim_optional() -> None:
    examples_readme = _read("examples/README.md")
    architecture_readme = _read("examples/architecture/README.md")
    combined = examples_readme + "\n" + architecture_readme

    assert "--cc_backend_config_type astra_sim_analytical" in combined
    assert "default public example backend" in combined
    assert "collective_sim" in combined
    assert "optional" in combined.lower()
    assert "explicitly pass `--cc_backend_config_type collective_sim`" in combined
    assert "default `collective_sim`" not in combined
    assert "baseline examples use Frontier's default `collective_sim` backend" not in combined


def test_examples_docs_link_profiling_entrypoints_and_downstream_smokes() -> None:
    examples_readme = _read("examples/README.md")
    architecture_readme = _read("examples/architecture/README.md")
    profiling_readme = _read("examples/profiling/README.md")
    combined = examples_readme + "\n" + architecture_readme + "\n" + profiling_readme

    assert "examples/profiling/" in combined
    assert "profile_attention_chunked_prefill.sh" in combined
    assert "smoke_simulator_dense_csv.sh" in combined
    assert "smoke_simulator_moe_csv.sh" in combined
    assert "uniform_random" in combined
    assert "outputs/examples/profiling-simulator" in combined
    assert "task_memory/task_2026-06-07_examples_expansion_e2e_validation" not in combined
    assert "PROFILE_METHOD=cuda_event" in combined
    assert "wrapper default" in combined
    assert "record_function" in combined


def test_internal_profiling_docs_point_release_users_to_examples_profiling() -> None:
    profiling_readme = _read("frontier/profiling/README.md")
    legacy_readme = _read("frontier/profiling/example/README.md")
    combined = profiling_readme + "\n" + legacy_readme

    assert "examples/profiling/" in combined
    assert "legacy/internal" in combined
    assert "frontier/profiling/example/" in combined


def test_frontier_profiling_readme_prioritizes_release_wrappers() -> None:
    profiling_readme = _read("frontier/profiling/README.md")

    assert "bash examples/profiling/profile_linear_op.sh --dry-run" in profiling_readme
    assert (
        "bash examples/profiling/profile_attention_chunked_prefill.sh --dry-run"
        in profiling_readme
    )
    assert "bash examples/profiling/profile_moe.sh --dry-run" in profiling_readme
    assert "bash examples/profiling/smoke_simulator_dense_csv.sh" in profiling_readme
    assert "bash examples/profiling/smoke_simulator_moe_csv.sh" in profiling_readme
    assert "Historical / guarded architecture notes" in profiling_readme
    assert "current release-facing examples support co-location only" in profiling_readme


def test_frontier_profiling_readme_has_no_known_typo_in_key_principles() -> None:
    profiling_readme = _read("frontier/profiling/README.md")

    assert "Pa_projameter" not in profiling_readme
    assert "**EP as Distribution Parameter**" in profiling_readme
