"""Release documentation contract tests for setup and Docker usability."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _readme_text() -> str:
    return (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")


def _profiling_readme_text() -> str:
    return (PROJECT_ROOT / "frontier/profiling/README.md").read_text(encoding="utf-8")


def _assert_contains_all(text: str, snippets: tuple[str, ...]) -> None:
    missing = [snippet for snippet in snippets if snippet not in text]
    assert not missing, f"Missing README snippets: {missing}"


def test_readme_first_screen_documents_release_scope_and_backend_default() -> None:
    readme = _readme_text()
    first_screen = "\n".join(readme.splitlines()[:45])

    _assert_contains_all(
        first_screen,
        (
            "pre-release-v0.1",
            "co-location only",
            "PDD and AFD are upcoming roadmap items",
            "--cc_backend_config_type astra_sim_analytical",
            "collective_sim",
            "optional",
        ),
    )
    assert "models co-location, Prefill-Decode Disaggregation (PDD), and Attention-FFN Disaggregation (AFD)" not in first_screen


def test_readme_documents_optional_collective_sim_target_runtime_build() -> None:
    readme = _readme_text()

    _assert_contains_all(
        readme,
        (
            "collective_sim is optional",
            "explicitly select `--cc_backend_config_type collective_sim`",
            "git submodule update --init --recursive frontier/cc_backend/backends/collective-sim",
            "frontier/cc_backend/backends/collective-sim/sim",
            "make",
            "$(nproc)",
            "make -B",
            "GLIBC",
        ),
    )


def test_readme_documents_frontier_env_docker_python_entrypoint() -> None:
    readme = _readme_text()

    _assert_contains_all(
        readme,
        (
            "fengyicheng/frontier-env",
            "FRONTIER_DOCKER_PYTHON",
            "image-specific",
            "replace this path",
            "*/envs/vidur_te/bin/python",
            "-type l",
            "import pytest",
            "Python executable not found",
            "--tmpfs /workspace/frontier/outputs",
            "--tmpfs /workspace/frontier/cache",
            "-p no:cacheprovider",
        ),
    )

    local_user_path = f"/local/{Path.home().name}/"
    assert local_user_path not in readme
    assert "/research/d1/gds/" not in readme


def test_environment_yml_uses_conda_forge_nodefaults_and_pip_ddsketch() -> None:
    environment_yml = (PROJECT_ROOT / "environment.yml").read_text(encoding="utf-8")

    assert "- conda-forge" in environment_yml
    assert "- nodefaults" in environment_yml
    assert "- pip:" in environment_yml
    assert "- ddsketch>=3,<4" in environment_yml


def test_readme_documents_dedicated_profiling_environment() -> None:
    readme = _readme_text()

    _assert_contains_all(
        readme,
        (
            "environment_profiling.yml",
            "conda env create -f environment_profiling.yml",
            "vllm",
            "flashinfer",
            "existing environment",
        ),
    )


def test_profiling_readme_uses_environment_profiling_conda_name() -> None:
    profiling_readme = _profiling_readme_text()
    environment_name = None
    for line in (PROJECT_ROOT / "environment_profiling.yml").read_text(encoding="utf-8").splitlines():
        if line.startswith("name:"):
            environment_name = line.split(":", 1)[1].strip()
            break

    assert environment_name == "frontier-profiling"
    assert f"conda activate {environment_name}" in profiling_readme
    assert "conda activate frontier_profiling" not in profiling_readme




def test_readme_documents_flashinfer_jit_nvcc_requirement() -> None:
    readme = _readme_text()

    _assert_contains_all(
        readme,
        (
            "FlashInfer JIT",
            "nvcc",
            "cuda-nvcc",
            "CUDA_HOME",
        ),
    )

def test_readme_documents_production_docker_troubleshooting() -> None:
    readme = _readme_text()

    _assert_contains_all(
        readme,
        (
            "docker pull fengyicheng/frontier-env",
            "NVIDIA Container Toolkit",
            "--gpus all",
            "--shm-size",
            "driver compatibility",
            "nvidia-smi",
        ),
    )
