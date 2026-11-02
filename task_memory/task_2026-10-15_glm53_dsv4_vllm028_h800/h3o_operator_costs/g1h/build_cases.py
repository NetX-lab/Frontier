"""Generate the G1h CPU pilot list and full coverage target manifest.

Run with the lead's committed --source-revision after reviewing B11. This writes
planning inputs only; prepare_package.py remains a separate, lead-owned action.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from frontier.profiling.attention.dsa_plan import (
    DsaMeasurementLimits, build_pilot_cases, PHASE_DUMMY, PHASE_PREFILL,
)
from frontier.profiling.attention.dsv4_attention_main import build_parser
from frontier.profiling.attention.v4_attention_cases import build_attention_cases
from frontier.profiling.attention.v4_attention_windows import C1

HERE = Path(__file__).resolve().parent
# Measured P1h package 03 timing inputs; seconds include profiler handling and
# context construction. The cold start includes process import and recapture.
P1H_TIMING = {
    "engine0_total_seconds": 576.1036419617012,
    "engine1_total_seconds": 90.29085552506149,
    "engine0_case_seconds": 90.0,
    "engine0_startup_compile_autotune_seconds": 431.78,
    "engine1_startup_compile_autotune_seconds": 25.23,
    "engine0_estimated_steps": 280,
    "engine0_measured_steps": 163,
    "measured_hold_seconds": 0.2,
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def limits_from_spec(spec):
    args = build_parser().parse_args(["plan", *spec["plan_args"], "--model_config_path",
                                     str(Path(spec["base_package"]) / "model/config.json"),
                                     "--engine_policy", "graph_replay",
                                     "--cudagraph_mode", "FULL_AND_PIECEWISE",
                                     "--cudagraph_capture_sizes", *map(str, spec["cudagraph_capture_sizes"])])
    return DsaMeasurementLimits(args.max_context_len, args.max_kv_tokens_per_lane,
                                args.max_num_seqs, tuple(args.chunk_budgets), args.block_size,
                                args.decode_warmup_steps, args.decode_measured_steps,
                                args.prefill_warmup_repeats, args.prefill_measured_repeats,
                                "graph_replay", "FULL_AND_PIECEWISE",
                                tuple(spec["cudagraph_capture_sizes"]), args.step_hold_us)


def estimate_time(engines, cases, grid, hold_us):
    cold = P1H_TIMING["engine0_total_seconds"] - P1H_TIMING["engine0_case_seconds"]
    unheld = (P1H_TIMING["engine0_case_seconds"] - P1H_TIMING["engine0_measured_steps"]
              * P1H_TIMING["measured_hold_seconds"]) / P1H_TIMING["engine0_estimated_steps"]
    hold = hold_us / 1e6
    runs = []
    for engine in engines:
        selected = [case for case in cases if case.engine_index == engine.engine_index]
        steps = sum(case.estimated_steps for case in selected)
        measured = sum(case.measured for case in selected)
        total = cold + steps * unheld + measured * hold
        runs.append({"engine_index": engine.engine_index, "chunk_budget": engine.chunk_budget,
                     "cases": len(selected), "estimated_steps": steps, "measured_steps": measured,
                     "startup_seconds": cold, "step_seconds": steps * unheld + measured * hold,
                     "total_seconds": total, "gpu_hours": total / 3600})
    # An envelope for all requested windows, even those that have no valid case:
    # use the high context and conservative FCFS construction allowance per class.
    full_steps = defaultdict(int)
    full_windows = Counter()
    for window in grid["windows"]:
        tails, own = window["tails"], window["own"]
        index = 2 if tails > 1 else 1 if tails else 0
        engine = engines[index]
        if window["category"] in (PHASE_DUMMY, PHASE_PREFILL):
            steps = 1
        else:
            continuing = own - tails
            construction = (0 if not continuing else
                            (continuing * C1["cached_high"] + engine.chunk_budget - continuing)
                            // (engine.chunk_budget - continuing + 1) + 1)
            steps = construction + (2 if tails else 1)
        full_steps[index] += steps
        full_windows[index] += 1
    full = [{"engine_index": i, "estimated_steps_envelope": full_steps[i],
             "windows": full_windows[i], "startup_seconds": cold,
             "gpu_hours_envelope": (cold + full_steps[i] * unheld + full_windows[i] * hold) / 3600}
            for i in range(len(engines))]
    return {"basis": P1H_TIMING, "unheld_seconds_per_estimated_step": unheld,
            "seconds_per_measured_step_including_hold": unheld + hold,
            "engine_start_policy": "charge the observed cold process/start/recapture time to every engine",
            "engines": runs, "gpu_hours": sum(r["gpu_hours"] for r in runs),
            "full_grid_envelope": full, "full_grid_gpu_hours_envelope": sum(r["gpu_hours_envelope"] for r in full),
            "c1_g1h_share_gpu_hours": 1.5, "g1h_g4h_cap_gpu_hours": 7,
            "limitations": ["Full-grid envelope charges one case per requested window; the admitted run shares identical cases.",
                            "One-step cases provide no repeated-window variance; source warm-up/capture still occurs.",
                            "Profiler per-case overhead and larger-budget autotuning have not been GPU-qualified."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision")
    args = parser.parse_args(argv)
    spec = json.loads((HERE / "package_plan.json").read_text())
    if args.source_revision:
        spec["source_revision"] = args.source_revision
    limits = limits_from_spec(spec)
    payload, grid = build_attention_cases(limits)
    engines, cases = build_pilot_cases(limits, payload)
    spec.pop("coverage_context_tolerance", None)
    spec.pop("execution_blockers", None)
    spec["runs"] = [{"mode": limits.cudagraph_mode, "engine_index": engine.engine_index,
                     "case_ids": [c.case_id for c in cases if c.engine_index == engine.engine_index]}
                    for engine in engines]
    timing = estimate_time(engines, cases, grid, limits.step_hold_us)
    categories = defaultdict(Counter)
    shared_cases = defaultdict(set)
    for window in grid["windows"]:
        category = categories[window["category"]]
        category["requested_windows"] += 1
        category[window["status"]] += 1
        if window.get("shared_case"):
            category["shared_windows"] += 1
            shared_cases[window["category"]].add(window["case_id"])
    for case in cases:
        category = categories[next(w["category"] for w in grid["windows"] if w["case_id"] == case.case_id)]
        category["cases"] += 1
        category["measured_steps"] += case.measured
    for name, category in categories.items():
        category["shared_cases"] = len(shared_cases[name])
        category.setdefault("shared_windows", 0)
    grid["counts_by_category"] = {k: dict(v) for k, v in categories.items()}
    grid["cases"] = len(cases)
    grid["measured_steps"] = sum(c.measured for c in cases)
    grid["gpu_time_estimate"] = timing
    write(HERE / "pilot_cases.json", payload)
    write(HERE / "coverage_grid.json", grid)
    write(HERE / "gpu_time_estimate.json", timing)
    write(HERE / "package_plan.json", spec)
    print(json.dumps({"cases": len(cases), "grid_complete": grid["complete"],
                      "windows": dict(Counter(w["status"] for w in grid["windows"])),
                      "categories": grid["counts_by_category"], "gpu_hours": timing["gpu_hours"],
                      "full_grid_gpu_hours_envelope": timing["full_grid_gpu_hours_envelope"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
