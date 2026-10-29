"""Predict C1 attention windows and stream-check native graph replay records.

Examples:
  python tests/analysis/v4_attention_coverage.py cases --package-plan SPEC --output STEPS.jsonl
  python tests/analysis/v4_attention_coverage.py check --package-plan SPEC --raw RAW_DIR --output REPORT.json
The check reports missing windows without failing a partial pilot; use
--require-complete-grid to make missing target windows an error.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

from frontier.profiling.attention.dsa_case_prediction import predict_case
from frontier.profiling.attention.dsv4_attention_main import build_parser, build_plan
from frontier.profiling.attention.v4_attention_windows import (
    C1, assign_windows, describe_step, native_class_in_grid, required_windows,
)


def plan_from_spec(path: Path):
    spec = json.loads(path.read_text())
    args = ["plan", *spec["plan_args"], "--model_config_path",
            str(Path(spec["base_package"]) / "model/config.json"),
            "--pilot_cases", str(path.parent / spec["pilot_cases"]),
            "--engine_policy", "graph_replay", "--cudagraph_mode", spec["runs"][0]["mode"],
            "--cudagraph_capture_sizes", *map(str, spec["cudagraph_capture_sizes"])]
    return build_plan(build_parser().parse_args(args))


def prediction_steps(plan) -> Iterable[dict[str, Any]]:
    for case in plan.cases:
        for step in predict_case(case, plan.engine(case.engine_index), plan.limits):
            yield {**step, "prediction": describe_step(step)}


def read_steps(paths: Iterable[Path]) -> Iterable[tuple[str, int, dict[str, Any]]]:
    """Never load call records or an entire raw JSONL file into memory."""
    for path in paths:
        files = sorted(path.glob("dsv4_raw_e*.jsonl")) if path.is_dir() else [path]
        if not files:
            raise ValueError(f"No dsv4_raw_e*.jsonl records under {path}")
        for file in files:
            with file.open() as stream:
                for line_number, line in enumerate(stream, 1):
                    record = json.loads(line)
                    if record.get("kind") == "step":
                        yield str(file), line_number, record


def native_coverage(paths: Iterable[Path]) -> dict[str, Any]:
    results = []
    for path in paths:
        source = json.loads(path.read_text())
        classes = {}
        outside = []
        for record in source["attention"]:
            if record["role"] != "decode" or record["kind"] != "serving":
                continue
            key = tuple(record[k] for k in ("mode", "W", "R", "own_tokens", "own_reqs", "has_prefill"))
            classes[key] = record
        for record in classes.values():
            if not native_class_in_grid(record):
                outside.append(record)
        multi = set()
        for lane, records in source["lifecycle"].items():
            if lane.startswith("decode/"):
                multi.update((own, tails) for own, tails, *_ in records["b_first_tails"] if tails > 1)
        missing_multi = sorted(multi - set(C1["multi_tail"]))
        results.append({"case": source["case"], "source": str(path), "classes": len(classes),
                        "inside": len(classes) - len(outside), "outside": outside,
                        "multi_tail": sorted(multi), "outside_multi_tail": missing_multi})
    return {"cases": results, "classes": sum(r["classes"] for r in results),
            "outside": sum(len(r["outside"]) + len(r["outside_multi_tail"]) for r in results)}


def check_steps(plan, records: Iterable[tuple[str, int, dict[str, Any]]], *,
                predicted: bool = False,
                c1_paths: Iterable[Path] = ()) -> dict[str, Any]:
    expected = {case.case_id: predict_case(case, plan.engine(case.engine_index), plan.limits)
                for case in plan.cases}
    cases = {case.case_id: case for case in plan.cases}
    counts: Counter = Counter()
    per_case: Counter = Counter()
    windows = set()
    outside, mismatches = [], []
    categories: Counter = Counter()
    for path, line, step in records:
        case_id = step["case_id"]
        key = (step["engine_index"], case_id, step.get("measurement"))
        ordinal = counts[key]
        counts[key] += 1
        per_case[case_id] += 1
        location = {"file": path, "line": line, "case_id": case_id,
                    "step_index": step.get("step_index", step.get("case_step")), "case_step": ordinal}
        predicted_step = expected.get(case_id, [])
        if (case_id not in cases or step["engine_index"] != cases[case_id].engine_index
                or ordinal >= len(predicted_step)):
            mismatches.append({**location, "reason": "unknown case/engine or extra measured step"})
        else:
            wanted = predicted_step[ordinal]
            for field in ("phase", "requests", "cudagraph_dispatch"):
                if step.get(field) != wanted[field]:
                    mismatches.append({**location, "field": field,
                                       "expected": wanted[field], "observed": step.get(field)})
        assigned = assign_windows(step)
        point = describe_step(step)
        categories[point["category"]] += 1
        if not assigned:
            outside.append({**location, "point": point,
                            "reason": "class, context profile or compression mix outside C1 targets"})
        windows.update(assigned)
    missing_steps = []
    for case_id, steps in expected.items():
        streams = [(key, n) for key, n in counts.items() if key[1] == case_id]
        if not streams:
            missing_steps.append({"case_id": case_id, "expected": len(steps), "observed": 0})
        for key, count in streams:
            if count < len(steps):
                missing_steps.append({"case_id": case_id, "measurement": key[2],
                                      "expected": len(steps), "observed": count})
    status = "predicted" if predicted else "covered"
    target_windows = [{**w.to_json(), "status": status if w in windows else "missing"}
                      for w in required_windows()]
    native = native_coverage(c1_paths)
    class_status = defaultdict(list)
    for w in required_windows():
        class_status[(w.category, w.mode, w.width, w.own, w.tails)].append(w in windows)
    return {"schema_version": "v4_attention_coverage.v1", "context_tolerance": C1["context_tolerance"],
            "measured_steps": sum(counts.values()), "cases_seen": len(per_case),
            "steps_by_category": dict(categories), "steps_by_case": dict(per_case),
            "windows": target_windows, "window_counts": dict(Counter(w["status"] for w in target_windows)),
            "classes": [{"category": key[0], "mode": key[1], "width": key[2], "own": key[3],
                         "tails": key[4], "covered_windows": sum(values), "required_windows": len(values),
                         "status": status if all(values) else "missing"}
                        for key, values in class_status.items()],
            "outside_steps": outside, "mismatches": mismatches, "missing_steps": missing_steps,
            "native": native,
            "prediction_matches": not mismatches and not missing_steps,
            "grid_complete": all(w["status"] != "missing" for w in target_windows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("cases", "check"):
        command = commands.add_parser(name)
        command.add_argument("--package-plan", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "check":
            command.add_argument("--raw", type=Path, nargs="+", required=True)
            command.add_argument("--c1-json", type=Path, nargs="*", default=[])
            command.add_argument("--predicted", action="store_true")
            command.add_argument("--require-complete-grid", action="store_true")
    args = parser.parse_args(argv)
    plan = plan_from_spec(args.package_plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "cases":
        count = 0
        with args.output.open("w") as output:
            for step in prediction_steps(plan):
                output.write(json.dumps(step, sort_keys=True) + "\n")
                count += 1
        print(json.dumps({"output": str(args.output), "cases": len(plan.cases), "predicted_steps": count}))
        return 0
    report = check_steps(plan, read_steps(args.raw), predicted=args.predicted, c1_paths=args.c1_json)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "measured_steps": report["measured_steps"],
                      "windows": report["window_counts"], "mismatches": len(report["mismatches"]),
                      "missing_cases_or_steps": len(report["missing_steps"]),
                      "outside_steps": len(report["outside_steps"]), "native": report["native"]}))
    if not report["prediction_matches"] or report["native"]["outside"]:
        return 1
    return 2 if args.require_complete_grid and not report["grid_complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
