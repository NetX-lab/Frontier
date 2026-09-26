"""Tests for the staged profiling/runtime import boundary."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTOR_ROOT = REPO_ROOT / "frontier/execution_time_predictor"
# Profiling helpers each predictor module may import; any module not listed
# imports none.
RUNTIME_IMPORTS = {
    "sklearn_execution_time_predictor.py": {
        "frontier.profiling.cpu_overhead.schema",
        "frontier.profiling.cpu_overhead.validation",
        "frontier.profiling.other_overhead.validation",
    },
    "profiling_dataframe_loaders.py": {
        "frontier.profiling.cpu_overhead.validation",
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
