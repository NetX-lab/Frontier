"""Reproduce the accepted H200 V4 tables with the existing CPU reducers."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
from pathlib import Path

from frontier.profiling.attention.v4_attention_windows import assign_windows, repeat_windows
from frontier.profiling.attention.dsv4_attention_main import main as aggregate_attention
from frontier.profiling.common.runtime_identity import file_sha256
from frontier.profiling.kernel_gap.dsv4_replay_gaps import main as aggregate_gaps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
GPU_INPUT_ROOT = Path("/data/ycfeng/tmp/glm53_dsv4/h200")
MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
MODEL_DIRECTORY = Path("compute/h200") / MODEL_NAME
RAW_DIRECTORY = (GPU_INPUT_ROOT / "g1h_02/download/G1H_ATTENTION/bridge/FULL_AND_PIECEWISE/raw")


def union_csv(inputs: list[Path], output: Path) -> int:
    """Keep serialized measurements intact and fill absent input columns with blanks."""
    columns, rows = [], []
    for path in inputs:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns.extend(column for column in reader.fieldnames if column not in columns)
            rows.extend(reader)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def recorded_model_revision(launcher: Path) -> str:
    """Read the staged producer's revision constant without importing its job launcher."""
    for node in ast.parse(launcher.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MODEL_CONFIG_REVISION"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"{launcher} does not record MODEL_CONFIG_REVISION")


def row_count(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def copy_supplement_raw(raw_dir: Path, output: Path) -> tuple[list[Path], list[str]]:
    """Exclude repeat cases using their native steps and the shared repeat grid."""
    files = sorted(raw_dir.glob("dsv4_raw_e*.jsonl"))
    if not files:
        raise ValueError(f"No supplement dsv4_raw_e*.jsonl files under {raw_dir}")
    repeats, excluded = set(repeat_windows()), set()
    for path in files:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("kind") == "step" and repeats.intersection(assign_windows(row)):
                    excluded.add(row["case_id"])
    output.mkdir(parents=True, exist_ok=True)
    for path in files:
        with path.open() as source, (output / path.name).open("w") as target:
            for line in source:
                if json.loads(line).get("case_id") not in excluded:
                    target.write(line)
    return files, sorted(excluded)


def promote_tables(*, scratch_dir: Path, raw_dir: Path = RAW_DIRECTORY,
                   package_dir: Path | None = None,
                   supplement_raw_dir: Path | None = None,
                   supplement_plan_path: Path = HERE.parent / "g1h_supplement/package_plan.json",
                   output_root: Path = ROOT / "data/profiling",
                   manifest_path: Path = HERE / "manifest.json") -> dict:
    plan_path = HERE.parent / "g1h/package_plan.json"
    package = package_dir if package_dir is not None else HERE.parent / "g1h/package_02"
    package_manifest = package / "manifest.json"
    plan = json.loads(plan_path.read_text())
    snapshot = json.loads(package_manifest.read_text())
    if plan["source_revision"] != snapshot["source_revision"]:
        raise ValueError("G1h plan and accepted package source revisions disagree")
    config, pilot = package / "model/config.json", package / plan["pilot_cases"]
    launcher = package / "launch/glm_producer_jobs.py"
    model_revision = recorded_model_revision(launcher)
    raw_files = sorted(raw_dir.glob("dsv4_raw_e*.jsonl"))
    if {int(path.stem.rsplit("e", 1)[1]) for path in raw_files} != {
        run["engine_index"] for run in plan["runs"]
    }:
        raise ValueError("G1h raw engine identities differ from the accepted runs")
    copied_raw = scratch_dir / "raw"
    copied_raw.mkdir(parents=True, exist_ok=True)
    for source in raw_files:
        shutil.copyfile(source, copied_raw / source.name)
    generated_root = scratch_dir / "profiling"
    aggregate_attention([
        "aggregate", *plan["plan_args"], "--model_config_path", str(config),
        "--pilot_cases", str(pilot), "--engine_policy", "graph_replay",
        "--cudagraph_mode", plan["runs"][0]["mode"],
        "--cudagraph_capture_sizes", *map(str, plan["cudagraph_capture_sizes"]),
        "--raw_dir", str(copied_raw), "--output_root", str(generated_root),
        "--device", "h200", "--source_revision", plan["source_revision"],
        "--model_config_revision", model_revision,
    ])
    supplement_inputs, excluded_repeats = [], []
    if supplement_raw_dir is not None:
        supplement = json.loads(supplement_plan_path.read_text())
        supplement_pilot = supplement_plan_path.parent / supplement["pilot_cases"]
        supplement_files, excluded_repeats = copy_supplement_raw(
            supplement_raw_dir, scratch_dir / "supplement_raw")
        supplement_root = scratch_dir / "supplement_profiling"
        aggregate_attention([
            "aggregate", *supplement["plan_args"], "--model_config_path", str(config),
            "--pilot_cases", str(supplement_pilot), "--engine_policy", "graph_replay",
            "--cudagraph_mode", supplement["runs"][0]["mode"],
            "--cudagraph_capture_sizes", *map(str, supplement["cudagraph_capture_sizes"]),
            "--raw_dir", str(scratch_dir / "supplement_raw"), "--output_root", str(supplement_root),
            "--device", "h200", "--source_revision", supplement["source_revision"],
            "--model_config_revision", model_revision,
        ])
        filename = "attention_graph_replay_kernel_only.csv"
        merged = scratch_dir / "merged" / filename
        union_csv([generated_root / MODEL_DIRECTORY / filename,
                   supplement_root / MODEL_DIRECTORY / filename], merged)
        shutil.copyfile(merged, generated_root / MODEL_DIRECTORY / filename)
        supplement_inputs = [supplement_plan_path, supplement_pilot, *supplement_files]
    result = aggregate_gaps([
        "--raw_files", *map(str, (copied_raw / path.name for path in raw_files)),
        "--output_dir", str(generated_root), "--device", "h200", "--model", MODEL_NAME,
        "--diagnostics_path", str(scratch_dir / "gap_diagnostics.json"),
    ])
    if result:
        raise ValueError(f"V4 kernel-gap reducer failed with status {result}")
    table_dir = output_root / MODEL_DIRECTORY
    table_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for path in sorted((generated_root / MODEL_DIRECTORY).glob("*.csv")):
        target = table_dir / path.name
        shutil.copyfile(path, target)
        outputs.append(target)
    linear_inputs = [GPU_INPUT_ROOT / "g2h_01/download" / directory / MODEL_DIRECTORY /
                     "linear_op_kernel_only.csv" for directory in (
                         "G2H_OPERATORS_RECORD_FUNCTION", "G2H_SHARED_EXPERT_EP16_RECORD_FUNCTION")]
    moe_inputs = [GPU_INPUT_ROOT / f"g3h_01/download/G3H_MOE_RECORD_FUNCTION/ep{ep}" /
                  MODEL_DIRECTORY / "moe_kernel_only.csv" for ep in (16, 32)]
    for paths, filename in ((linear_inputs, "linear_op_kernel_only.csv"),
                            (moe_inputs, "moe_kernel_only.csv")):
        target = table_dir / filename
        union_csv(paths, target)
        outputs.append(target)
    inputs = [plan_path, package_manifest, config, pilot, launcher,
              *raw_files, *linear_inputs, *moe_inputs, *supplement_inputs]
    revisions = {"attention": plan["source_revision"]}
    if supplement_raw_dir is not None:
        revisions["attention_supplement"] = supplement["source_revision"]
    for path in (*linear_inputs, *moe_inputs):
        with path.open(newline="") as handle:
            revisions[str(path)] = sorted({row["source_revision"] for row in csv.DictReader(handle)})
    manifest = {
        "inputs": [{"path": str(path), "sha256": file_sha256(path)} for path in inputs],
        "outputs": [{"path": str(path), "sha256": file_sha256(path), "row_count": row_count(path)}
                    for path in sorted(outputs)],
        "source_revisions": revisions, "model_config_revision": model_revision,
        "native_revision": snapshot["native_revision"],
    }
    if supplement_raw_dir is not None:
        manifest["supplement_excluded_repeat_cases"] = excluded_repeats
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-dir", type=Path, required=True,
                        help="Writable scratch directory for the copied raw records and reducer outputs")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIRECTORY)
    parser.add_argument("--package-dir", type=Path,
                        help="Accepted G1h-02 package when stored outside this worktree")
    parser.add_argument("--supplement-raw-dir", type=Path)
    parser.add_argument("--supplement-plan", type=Path, default=HERE.parent / "g1h_supplement/package_plan.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/profiling")
    parser.add_argument("--manifest-path", type=Path, default=HERE / "manifest.json")
    args = parser.parse_args()
    print(json.dumps(promote_tables(scratch_dir=args.scratch_dir, raw_dir=args.raw_dir,
                                   package_dir=args.package_dir,
                                   supplement_raw_dir=args.supplement_raw_dir,
                                   supplement_plan_path=args.supplement_plan,
                                   output_root=args.output_root, manifest_path=args.manifest_path)["outputs"], indent=2))


if __name__ == "__main__":
    main()
