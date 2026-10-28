"""Reuse the P1h package snapshot mechanism after G1h coverage is resolved.

This module imports no GPU library and never stages a partial grid. Package
creation belongs to the lead after committing B11 and replacing source_revision.
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def package_builder():
    source = HERE.parent / "p1h_bridge/prepare_package.py"
    spec = importlib.util.spec_from_file_location("g1h_p1h_package_builder", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    return module


def main():
    spec = json.loads((HERE / "package_plan.json").read_text())
    grid = json.loads((HERE / spec["coverage_grid"]).read_text())
    if not grid["complete"]:
        raise RuntimeError("G1h planning is incomplete: resolve coverage_grid.json infeasible windows before packaging")
    package_builder().main()


if __name__ == "__main__":
    main()
