"""Lead entry point that reuses G1h's P1h snapshot builder; never run by B22."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def package_builder():
    spec = importlib.util.spec_from_file_location("supplement_g1h_prepare", HERE.parent / "g1h/prepare_package.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = module.package_builder()
    builder.HERE = HERE
    return builder


def main():
    spec = json.loads((HERE / "package_plan.json").read_text())
    if not json.loads((HERE / spec["coverage_grid"]).read_text())["complete"]:
        raise RuntimeError("G1h supplement planning is incomplete")
    package_builder().main()


if __name__ == "__main__":
    main()
