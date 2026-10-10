"""Entry point of the E2E accuracy gate: Frontier against vLLM on the 22 scenario-matrix cells.

Subcommands, by the evidence they need (README.md):

  cells       list the matrix's cells, their gate kind, coverage and routing status
  gate        gate a cell from normalized request rows (shipped fixtures or `normalize` output)
  normalize   normalized request rows of a cell from raw vLLM and Frontier run directories
  frontier    trained Frontier replay: rerun a cell's recorded Frontier command on pinned inputs
  cpu-table   CPU artifact replay: rebuild a case's published CPU-overhead CSVs (the current X2
              table, or the earlier k1d/k1e ones) from pinned probe logs
  admit       admission reading of one CPU-probe run (I-6, T43-C4PROBE or T43-ISOPREFILL)

Pinned inputs live under --artifacts-root, the calibration bundle's root; every path in
cases/scenario_matrix.json and artifacts.json is relative to it and carries its SHA-256, which
is checked before a file is used.

  PYTHONPATH=$PWD python tests/comparison/e2e_gate/run_gate.py gate \\
      --cell e2e_c1_dense_coloc/qps2 --rows tests/comparison/e2e_gate/fixtures/c1_qps2/rows
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from tests.comparison.e2e_gate import admission
from tests.comparison.e2e_gate.gate import gate_reading
from tests.comparison.e2e_gate.normalize import normalize_cell

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
MATRIX = HERE / "cases" / "scenario_matrix.json"
ARTIFACTS = HERE / "artifacts.json"


def load_matrix() -> dict:
    return json.loads(MATRIX.read_text())


def find_cell(matrix: dict, cell_id: str) -> tuple[dict, dict]:
    cells = {cell["cell_id"]: cell for cell in matrix["cells"]}
    if cell_id not in cells:
        raise SystemExit(f"unknown cell {cell_id}; `run_gate.py cells` lists them")
    cell = cells[cell_id]
    if cell["status"] != "claimed":
        raise SystemExit(f"{cell_id} is {cell['status']}: {cell['reason']}")
    return cell, matrix["cases"][cell["case_id"]]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned(root: Path, record: dict) -> Path:
    """The artifact a {path, sha256} record names, after its hash is checked."""
    path = root / record["path"]
    if not path.is_file():
        raise SystemExit(f"pinned artifact missing: {path}")
    if sha256(path) != record["sha256"]:
        raise SystemExit(f"{path}: SHA-256 differs from the pinned {record['sha256']}")
    return path


def expand(arguments: list[str], values: dict[str, str]) -> list[str]:
    return [argument.format(**values) for argument in arguments]


def command_cells(args) -> None:
    matrix = load_matrix()
    for cell in matrix["cells"]:
        if cell["status"] != "claimed":
            print(f"{cell['cell_id']:48s} {cell['status']}: {cell['reason']}")
            continue
        case = matrix["cases"][cell["case_id"]]
        print(f"{cell['cell_id']:48s} {cell['gate']:12s} routing {case['routing']['status']:15s} "
              f"coverage {cell['coverage']['class']}")


def command_gate(args) -> None:
    cell, case = find_cell(load_matrix(), args.cell)
    reading = gate_reading(cell, case, args.rows)
    text = json.dumps(reading, indent=1, default=str) + "\n"
    if args.output:
        args.output.write_text(text)
    for row in reading.get("metrics", []):
        print(f"{row['metric']:24s} vllm {row['vllm']:.4f} frontier {row['frontier']:.4f} "
              f"rel {row['relative_error']:.4f} {row['status']}")
    modes = reading.get("dp_pp_modes")
    if modes:
        for mode, metrics in modes["modes"].items():
            for metric, entry in metrics.items():
                print(f"{mode}.{metric:20s} requests {entry['requests_n']:4d} rel {entry['relative_error']:.4f} "
                      f"{entry.get('status', 'reported')}")
        print(f"mode outcome {modes['mode_outcome']}, reference modes {modes['reference_modes']}, "
              f"without gated pairs {modes['reference_modes_without_gated_pairs']}")
        for metric, entry in reading["cell_metrics_reported"].items():
            print(f"whole cell {metric:24s} rel {entry['relative_error']:.4f} reported, not gated")
    print(f"{args.cell}: {reading['verdict']} (routing {reading['routing_status']})")


def command_normalize(args) -> None:
    cell, case = find_cell(load_matrix(), args.cell)
    request_ids = json.loads(pinned(args.artifacts_root, cell["trace"]["request_ids"]).read_text())
    vllm_runs = {path.name: path for path in args.vllm_run_dir}
    frontier_runs = dict(args.frontier_run)
    normalize_cell(cell, case, request_ids, vllm_runs, frontier_runs, args.output_dir)
    print(f"wrote {args.output_dir}")


def command_frontier(args) -> None:
    cell, case = find_cell(load_matrix(), args.cell)
    run = cell["frontier_runs"][args.label]
    run_dir, cache_dir = args.output_dir.resolve(), (args.cache_dir or args.output_dir / "cache").resolve()
    for record in run["inputs"]:
        pinned(args.artifacts_root, record)
    if run_dir.exists():
        raise SystemExit(f"output already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    if run["route_delay_ms"] is not None:
        # The ensemble member's arrivals: the cell's trace delayed by the member's drawn route delays.
        trace = pd.read_csv(pinned(args.artifacts_root, cell["trace"]["trace_csv"]))
        trace["arrived_at"] += pd.Series(run["route_delay_ms"]) * 1e-3
        trace.to_csv(run_dir / "trace.csv", index=False)
        if sha256(run_dir / "trace.csv") != run["member_trace_sha256"]:
            raise SystemExit(f"{run_dir / 'trace.csv'} differs from the member trace the published run replayed")
    values = {"python": sys.executable, "artifacts": str(args.artifacts_root.resolve()),
              "run_dir": str(run_dir), "cache_dir": str(cache_dir)}
    command = expand(run["command"], values)
    print(" ".join(command))
    if args.dry_run:
        return
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), **run["environment"])
    with (run_dir / "simulator.log").open("w") as log:
        returncode = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"{args.cell} {args.label}: returncode {returncode}, log {run_dir / 'simulator.log'}")
    sys.exit(returncode)


def command_cpu_table(args) -> None:
    recipe = json.loads(ARTIFACTS.read_text())["cases"][args.case]["cpu_overhead"][args.table]
    if recipe is None:
        raise SystemExit(f"{args.case} has no {args.table} CPU-overhead table")
    producer = recipe["producer"]
    tree = (args.producer_tree or REPO_ROOT).resolve()
    if sha256(tree / producer["module"]) != producer["sha256"]:
        raise SystemExit(f"{tree / producer['module']} is not the producer of this table; pass --producer-tree "
                         f"with a checkout of Frontier {producer['commit']}")
    for record in recipe["cpu_probe_logs"] + recipe["dp_placement_logs"]:
        pinned(args.artifacts_root, record)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    values = {"artifacts": str(args.artifacts_root.resolve()), "output_dir": str(args.output_dir.resolve())}
    command = [sys.executable, "-m", "frontier.profiling.cpu_overhead.vllm_cpu_probe",
               *expand(recipe["producer_arguments"], values)]
    print(" ".join(command))
    subprocess.run(command, cwd=tree, check=True, env=dict(os.environ, PYTHONPATH=str(tree)))
    differs = [name for name, record in recipe["files"].items()
               if sha256(args.output_dir / record["path"]) != record["sha256"]]
    if differs:
        raise SystemExit(f"rebuilt CSVs differ from the published ones: {differs}")
    print(f"{args.case} {args.table}: rebuilt {sorted(recipe['files'])} equal the published SHA-256")


def command_admit(args) -> None:
    reading = admission.RULES[args.rule](args.probe, args.reference)
    print(json.dumps(reading, indent=1))


def frontier_run_argument(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError(f"expected LABEL=DIR, got {value!r}")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("cells", help="list the cells").set_defaults(run=command_cells)

    gate = commands.add_parser("gate", help="gate a cell from normalized rows")
    gate.add_argument("--cell", required=True, help="<case_id>/<cell>")
    gate.add_argument("--rows", type=Path, required=True)
    gate.add_argument("--output", type=Path, help="write the reading as JSON")
    gate.set_defaults(run=command_gate)

    normalize = commands.add_parser("normalize", help="normalized rows from raw runs")
    normalize.add_argument("--cell", required=True)
    normalize.add_argument("--artifacts-root", type=Path, required=True)
    normalize.add_argument("--vllm-run-dir", type=Path, action="append", required=True,
                           help="ground-truth run directory; repeat for each run of the cell")
    normalize.add_argument("--frontier-run", type=frontier_run_argument, action="append", required=True,
                           help="LABEL=DIR, a Frontier run directory holding metrics/ (and trace.csv for a member)")
    normalize.add_argument("--output-dir", type=Path, required=True)
    normalize.set_defaults(run=command_normalize)

    frontier = commands.add_parser("frontier", help="rerun a cell's recorded Frontier command")
    frontier.add_argument("--cell", required=True)
    frontier.add_argument("--label", required=True, help="a key of the cell's frontier_runs")
    frontier.add_argument("--artifacts-root", type=Path, required=True)
    frontier.add_argument("--output-dir", type=Path, required=True)
    frontier.add_argument("--cache-dir", type=Path, help="predictor cache (default: <output-dir>/cache)")
    frontier.add_argument("--dry-run", action="store_true", help="check the inputs and print the command")
    frontier.set_defaults(run=command_frontier)

    cpu_table = commands.add_parser("cpu-table", help="rebuild a case's published CPU-overhead CSVs")
    cpu_table.add_argument("--case", required=True)
    cpu_table.add_argument("--table", choices=("current", "k1d", "k1e"), default="current")
    cpu_table.add_argument("--producer-tree", type=Path,
                           help="Frontier checkout at the table's producer commit (default: this checkout)")
    cpu_table.add_argument("--artifacts-root", type=Path, required=True)
    cpu_table.add_argument("--output-dir", type=Path, required=True)
    cpu_table.set_defaults(run=command_cpu_table)

    admit = commands.add_parser("admit", help="admission reading of a CPU-probe run")
    admit.add_argument("--rule", choices=sorted(admission.RULES), required=True)
    admit.add_argument("--probe", type=Path, required=True)
    admit.add_argument("--reference", type=Path, action="append", required=True,
                       help="clean run (client_e2e: one; send_span: the job's clean runs of the cell) "
                            "or, for isolated_prefill, every CPU-probe run of the job, the probe included")
    admit.set_defaults(run=command_admit)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
