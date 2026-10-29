"""Rebuild interval rows from the retained P1h bridge traces on CPU."""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "tests/comparison/calibration"))
from dsv4_graph_bridge_check import load_config, graph_mode_flags
from frontier.profiling.attention import dsv4_attention_main
from frontier.profiling.attention.dsa_vllm_runner import graph_replay_calls
from frontier.profiling.attention.dsv4_graph_replay_runner import CONTAINER_SCOPES
from frontier.profiling.attention.graph_capture import CaptureRecord


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    summary = {}
    for mode in config.modes:
        source = args.source / mode
        raw = source / "raw" / f"dsv4_raw_e{config.engine_index}.jsonl"
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        engine, = [row for row in rows if row["kind"] == "engine"]
        plan = dsv4_attention_main.build_plan(dsv4_attention_main.build_parser().parse_args(
            ["plan", *config.plan_args, *graph_mode_flags(config, mode)]))
        engine = dict(engine, target_tensor_parallel_size=plan.shape.target_tensor_parallel_size,
                      driver_version="not recorded in the bridge run")
        capture_path = source / "raw" / f"dsv4_capture_e{config.engine_index}.json"
        record = CaptureRecord.from_json(json.loads(capture_path.read_text())["record"])
        target = args.output / mode / "raw"
        target.mkdir(parents=True, exist_ok=True)
        counts, cases = Counter(), {}
        with (target / raw.name).open("w") as handle:
            handle.write(json.dumps(engine) + "\n")
            for case in config.case_ids:
                steps = [row for row in rows if row["kind"] == "step" and row["case_id"] == case]
                rebuilt = graph_replay_calls(source / "traces" / f"{case}.json", steps, config.engine_index,
                                              record=record, container_scopes=CONTAINER_SCOPES)
                old_calls = [row for row in rows if row["kind"] == "call" and row["case_id"] == case]
                new_calls = [row for row in rebuilt if row["kind"] == "call"]
                assert new_calls == old_calls, f"{mode}/{case}: call rows changed"
                intervals = [row for row in rebuilt if row["kind"] == "interval"]
                counts.update(row["activity"] for row in intervals)
                cases[case] = {"call_rows_equal": True, "call_count": len(new_calls), "interval_count": len(intervals)}
                for row in (*steps, *rebuilt):
                    handle.write(json.dumps(row) + "\n")
                print(f"{mode}/{case}: {len(new_calls)} call rows equal; {len(intervals)} intervals", flush=True)
        shutil.copyfile(capture_path, target / capture_path.name)
        (args.output / mode / "traces").symlink_to((source / "traces").resolve(), target_is_directory=True)
        summary[mode] = {"activity_counts": dict(counts), "cases": cases}
        print(f"{mode}: interval activities {dict(counts)}", flush=True)
    shutil.copyfile(args.source / "bridge_run.json", args.output / "bridge_run.json")
    payload = json.loads(args.config.read_text())
    payload["output_root"] = str(args.output.resolve())
    (args.output / "bridge_config.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output / "rebuild_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

if __name__ == "__main__":
    main()
