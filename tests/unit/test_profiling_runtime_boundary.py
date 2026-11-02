"""Tests for the staged profiling/runtime import boundary."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTOR_ROOT = REPO_ROOT / "frontier/execution_time_predictor"
# Simulator dependencies that the vLLM serving image the profilers run in lacks.
SIMULATOR_ONLY_PACKAGES = ("fasteners", "sklearn")
# Modules a GPU worker imports directly, besides each profiling package's main.
WORKER_ENTRY_MODULES = (
    "frontier.profiling.attention.dsv4_graph_replay_runner",
    "frontier.profiling.collectives.vllm_collective_benchmark",
)
_BLOCKED_IMPORT = """
import importlib, sys
blocked = set(sys.argv[1].split(","))
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in blocked:
            raise ModuleNotFoundError(f"No module named {name!r}")
sys.meta_path.insert(0, Block())
for module in sys.argv[2:]:
    importlib.import_module(module)
"""
# Profiling helpers each predictor module may import; any module not listed
# imports none.
RUNTIME_IMPORTS = {
    # Attention composition retains the shared segment-class names for other scopes.
    "dsv4_composition.py": {"frontier.profiling.kernel_gap.schema"},
    # Replay admission reads the aggregate schema and balanced routing identity.
    "dsv4_predictor.py": {
        "frontier.profiling.attention.dsv4_replay_rows",
        "frontier.profiling.attention.dsv4_attention_rows",
    },
    # V4 non-attention receipts retain only the CPU provenance format.
    "deepseek_v4.py": {"frontier.profiling.common.runtime_identity"},
    "sklearn_execution_time_predictor.py": {
        "frontier.profiling.cpu_overhead.schema",
        "frontier.profiling.cpu_overhead.validation",
        "frontier.profiling.other_overhead.validation",
    },
    "profiling_dataframe_loaders.py": {
        "frontier.profiling.cpu_overhead.validation",
    },
    "prediction_family_trainers.py": {
        "frontier.profiling.cpu_overhead.validation",
    },
    "measurement_input_paths.py": {
        "frontier.profiling.cpu_overhead.schema",
    },
    "kernel_gap.py": {
        "frontier.profiling.kernel_gap.schema",
    },
}


def _absolute_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module)
    return imports


def test_runtime_predictors_use_only_allowlisted_profiling_helpers() -> None:
    for path in sorted(PREDICTOR_ROOT.glob("*.py")):
        profiling_imports = {
            module
            for module in _absolute_imports(path)
            if module.startswith("frontier.profiling")
        }
        assert profiling_imports == RUNTIME_IMPORTS.get(path.name, set()), path.name
        assert not profiling_imports & {
            "frontier.profiling.attention.main",
            "frontier.profiling.collectives.main",
            "frontier.profiling.cpu_overhead.benchmark_runner",
            "frontier.profiling.kernel_gap.main",
            "frontier.profiling.linear_op.main",
            "frontier.profiling.moe.main",
        }


def test_training_python_modules_do_not_import_profiling_implementation() -> None:
    training_root = REPO_ROOT / "frontier/training"
    for path in training_root.rglob("*.py"):
        profiling_imports = {
            module
            for module in _absolute_imports(path)
            if module.startswith("frontier.profiling")
        }
        assert not profiling_imports, f"{path}: {sorted(profiling_imports)}"


def test_moe_load_imbalance_features_have_a_runtime_owned_definition() -> None:
    from frontier.moe_load_imbalance import MoELoadImbalanceInput
    from frontier.profiling.moe.moe_input import (
        MoELoadImbalanceInput as ProfilingMoELoadImbalanceInput,
    )

    assert ProfilingMoELoadImbalanceInput is MoELoadImbalanceInput


def test_profiler_entry_points_import_without_simulator_only_packages() -> None:
    profiling_root = REPO_ROOT / "frontier/profiling"
    entries = sorted(
        ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
        for pattern in ("*/main.py", "*/*_main.py")
        for path in profiling_root.glob(pattern)
    )
    result = subprocess.run(
        [sys.executable, "-c", _BLOCKED_IMPORT, ",".join(SIMULATOR_ONLY_PACKAGES),
         *entries, *WORKER_ENTRY_MODULES],
        cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
