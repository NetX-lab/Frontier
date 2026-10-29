"""Plan only G1h-02 coverage differences and fixed package comparison repeats.

CPU generation only. Package creation remains a separate lead-owned action.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path

from frontier.profiling.attention.dsa_plan import build_pilot_cases
from frontier.profiling.attention.v4_attention_cases import build_attention_cases
from frontier.profiling.attention.v4_attention_windows import (
    assign_windows, repeat_windows, required_windows,
)
from v4_attention_coverage import check_steps, plan_from_spec, prediction_steps, read_steps

HERE = Path(__file__).resolve().parent
RAW = Path("/data/ycfeng/tmp/glm53_dsv4/h200/g1h_02/download/G1H_ATTENTION/bridge/FULL_AND_PIECEWISE/raw")

# B22 assignment: approximately 3.93 GPU-hours already charged to G1h-G4h.
G1H_G4H_USED_GPU_HOURS = 3.93


def g1h_builder():
    spec = importlib.util.spec_from_file_location("supplement_g1h_builder", HERE.parent / "g1h/build_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    builder = g1h_builder()
    base_spec = HERE.parent / "g1h/package_plan.json"
    report = check_steps(plan_from_spec(base_spec), read_steps([args.raw]))
    if not report["prediction_matches"] or report["outside_steps"]:
        raise ValueError("G1h-02 raw records do not match the original case predictions/grid")
    missing = [window for window, result in zip(required_windows(), report["windows"])
               if result["status"] == "missing"]
    repeats = repeat_windows()
    if set(missing) & set(repeats):
        raise ValueError("A repeat window was not measured by G1h-02")
    selected = tuple(dict.fromkeys((*missing, *repeats)))
    spec = json.loads(base_spec.read_text())
    spec.update(destination="package_01", name="G1H_SUPPLEMENT",
                job_tag="v4-h200-g1h-supplement-20261029-01",
                output_dir="/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_attention/g1h_s01",
                source_revision=args.source_revision, task_id="H3O-STAGE4-G1H-SUPPLEMENT")
    limits = builder.limits_from_spec(spec)
    payload, grid = build_attention_cases(limits, windows=selected, case_prefix="g1hs")
    engines, cases = build_pilot_cases(limits, payload)
    repeat_set = set(repeats)
    for window, target in zip(selected, grid["windows"]):
        target["purpose"] = "repeat" if window in repeat_set else "supplement"
    spec["repeat_case_ids"] = sorted({target["case_id"] for target in grid["windows"]
                                      if target["purpose"] == "repeat" and target["status"] == "predicted"})
    spec["runs"] = [{"mode": limits.cudagraph_mode, "engine_index": engine.engine_index,
                     "case_ids": [case.case_id for case in cases if case.engine_index == engine.engine_index]}
                    for engine in engines]
    timing = builder.estimate_time(engines, cases, grid, limits.step_hold_us)
    timing.update(g1h_g4h_used_gpu_hours=G1H_G4H_USED_GPU_HOURS, g1h_g4h_remaining_gpu_hours=timing["g1h_g4h_cap_gpu_hours"] - G1H_G4H_USED_GPU_HOURS,
                  reduction_required=timing["gpu_hours"] > timing["c1_g1h_share_gpu_hours"])
    grid.update(base_windows_covered=report["window_counts"].get("covered", 0),
                added_windows=len(missing), repeat_windows=[w.to_json() for w in repeats],
                cases=len(cases), measured_steps=sum(c.measured for c in cases))
    categories = {}
    for name in dict.fromkeys(w.category for w in selected):
        windows = [t for t in grid["windows"] if t["category"] == name]
        ids = {t["case_id"] for t in windows if t["status"] == "predicted"}
        categories[name] = dict(windows=len(windows), cases=len(ids), measured_steps=len(ids),
                                supplement_windows=sum(t["purpose"] == "supplement" for t in windows),
                                repeat_windows=sum(t["purpose"] == "repeat" for t in windows))
    grid["counts_by_category"] = categories
    for name, value in (("pilot_cases.json", payload), ("coverage_grid.json", grid),
                        ("gpu_time_estimate.json", timing), ("package_plan.json", spec)):
        builder.write(HERE / name, value)
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    builder.write(args.evidence_dir / "g1h02_coverage.json", report)
    # Use the ordinary plan loader and prediction for the examples the lead reviews.
    plan = plan_from_spec(HERE / "package_plan.json")
    examples = {}
    for step in prediction_steps(plan):
        for window in assign_windows(step):
            if window in selected:
                examples.setdefault((window.category, window.width),
                                    dict(window=window.to_json(), case_id=step["case_id"],
                                         requests=step["requests"], prediction=step["prediction"]))
    builder.write(args.evidence_dir / "examples.json", list(examples.values()))
    builder.write(HERE / "placement_examples.json", list(examples.values()))
    print(json.dumps(dict(complete=grid["complete"], cases=len(cases),
                          windows=Counter(t["status"] for t in grid["windows"]),
                          categories=categories, gpu_hours=timing["gpu_hours"],
                          envelope=timing["full_grid_gpu_hours_envelope"])))
    return 0 if grid["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
