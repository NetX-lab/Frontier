#!/usr/bin/env python3
"""One-cell causal event-boundary experiment; never changes production timing."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

from frontier.profiling.step5.inputs import Step5ProfileInput, read_profile_inputs
from frontier.profiling.step5.job import PROBE_INPUTS, REPO_ROOT
from frontier.profiling.step5.timing import _event_duration_ms, _prime_events, _window_duration_ms


CONFIG_PATH = Path(__file__).with_name("boundary_probe_config.json")
CLOCK_FIELDS = ("timestamp", "uuid", "name", "clocks.current.sm", "clocks.current.graphics",
                "clocks.current.memory", "pstate", "power.draw", "temperature.gpu")


def schedule(windows):
    if type(windows) is not int or windows < 2 or windows % 2:
        raise ValueError("windows_per_cohort must be a positive even count >=2 for balanced AB/BA")
    return [{"block": block, "order": order, "position": position, "cohort": cohort}
            for block in range(windows) for order in ("AB" if block % 2 == 0 else "BA",)
            for position, cohort in enumerate(order)]


def describe(settings):
    from frontier.attention.step5 import describe_step5_layer
    from frontier.profiling.common.model_config import ModelConfig

    native, measurement = settings["native"], settings["measurement"]
    count = measurement["replays"]
    if type(count) is not int or count < 2:
        raise ValueError("replays must be an integer >=2")
    if not math.isfinite(measurement["min_window_ms"]) or measurement["min_window_ms"] <= 0:
        raise ValueError("min_window_ms must be positive and finite")
    if type(measurement["warmup_iterations"]) is not int or measurement["warmup_iterations"] < 1:
        raise ValueError("warmup_iterations must be a positive integer")
    cells = read_profile_inputs(REPO_ROOT / PROBE_INPUTS,
                               cudagraph_capture_sizes=native["cudagraph_capture_sizes"])
    selected = [cell for cell in cells if cell == Step5ProfileInput(
        **{**settings["cell"], "query_lens": tuple(settings["cell"]["query_lens"]),
           "context_lens": tuple(settings["cell"]["context_lens"]), "graph_mode": cell.graph_mode})]
    if len(selected) != 1:
        raise ValueError("The diagnostic must select exactly one existing fixed-probe input")
    cell = selected[0]
    config = ModelConfig.from_model_name(settings["model"])
    cell.workload(config, block_size=native["block_size"], max_model_len=native["max_model_len"])
    layer = describe_step5_layer(config, cell.global_layer_id,
                                attention_tp_size=native["tensor_parallel_size"],
                                moe_tp_size=1, expert_parallel_size=1)
    if layer.attention_spec.variant_id != "swa" or cell.batch_size != 1 or cell.graph_mode != "full":
        raise ValueError("This experiment supports one SWA FULL decode request only")
    order = schedule(measurement["windows_per_cohort"])
    return {"cell": asdict(cell), "native": native, "measurement": measurement,
            "schedule": order, "measured_replays": len(order) * count,
            "trace_replays": 2 * count, "event_handles": count + 3,
            "initial_validation_replays": 1, "total_graph_replays": (len(order) + 2) * count + 1,
            "native_graphs": 1, "admitted": False}


def events_for(torch, replays):
    return tuple(torch.cuda.Event(enable_timing=True) for _ in range(replays + 3))


def measure_window(run, events, cohort, replays):
    """The experimental intervention is only the N+1 internal event records."""
    start, end, *boundaries = events
    start.record()
    if cohort == "B":
        boundaries[0].record()
    for index in range(replays):
        output = run()
        if cohort == "B":
            boundaries[index + 1].record()
    end.record()
    outer = _window_duration_ms(start, end)
    inner = ([_event_duration_ms(left, right) for left, right in zip(boundaries, boundaries[1:])]
             if cohort == "B" else [])
    return {"outer_duration_ms": outer, "outer_per_replay_us": outer * 1000 / replays,
            "inner_durations_ms": inner, "replays": replays,
            "internal_boundary_records": replays + 1 if cohort == "B" else 0}, output


def collect(run, prepare, events, measurement, *, observe_clock, validate, record):
    rows = []
    for item in schedule(measurement["windows_per_cohort"]):
        before = observe_clock()
        _prime_events(events)
        prepare()
        values, output = measure_window(run, events, item["cohort"], measurement["replays"])
        after = observe_clock()
        validate(output)
        row = {**item, **values, "replay_callable_id": id(run),
               "minimum_window_met": values["outer_duration_ms"] >= measurement["min_window_ms"],
               "clock_before": before, "clock_after": after, "output_state_validation": "PASS"}
        record(row)
        rows.append(row)
    return rows


def observe_clocks():
    command = ["nvidia-smi", "--id=0", "--query-gpu=" + ",".join(CLOCK_FIELDS),
               "--format=csv,noheader,nounits"]
    observation = {"utc": datetime.now(timezone.utc).isoformat(), "monotonic_ns": time.monotonic_ns(),
                   "command": command}
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        observation.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if result.returncode == 0:
            values = next(csv.reader(result.stdout.splitlines()))
            observation["fields"] = dict(zip(CLOCK_FIELDS, (v.strip() for v in values), strict=True))
    except subprocess.TimeoutExpired as error:
        observation.update(error="clock query timed out", stdout=str(error.stdout), stderr=str(error.stderr))
    return observation


def evaluate(rows, kernels, settings):
    measurement, reference = settings["measurement"], settings["reference"]
    by_cohort = {c: [r for r in rows if r["cohort"] == c] for c in "AB"}
    paired = []
    for block in range(measurement["windows_per_cohort"]):
        pair = {r["cohort"]: r for r in rows if r["block"] == block}
        paired.append({"block": block, "order": pair["A"]["order"],
                       "outer_delta_per_replay_us": pair["B"]["outer_per_replay_us"]
                                                    - pair["A"]["outer_per_replay_us"]})
    effects = {order: statistics.fmean(r["outer_delta_per_replay_us"] for r in paired if r["order"] == order)
               for order in ("AB", "BA")}
    # This prediction follows event density, not a fitted latency adjustment.
    density = (measurement["replays"] + 1) / measurement["replays"]
    predicted = [gap * density for gap in reference["observed_swa_gap_range_us"]]
    same_kernels = all(kernels["A"][key] == kernels["B"][key]
                       for key in ("kernel_names", "kernel_count", "memory_copy_names", "memory_copy_count"))
    protocol = {"counterbalance": [{k: r[k] for k in ("block", "order", "position", "cohort")} for r in rows]
                                    == schedule(measurement["windows_per_cohort"]),
                "fixed_replay_counts": all(r["replays"] == measurement["replays"] for r in rows),
                "one_replay_callable": len({r["replay_callable_id"] for r in rows}) == 1,
                "minimum_windows": all(r["minimum_window_met"] for r in rows),
                "same_correlated_kernel_work": same_kernels}
    clocks = {}
    for cohort, entries in by_cohort.items():
        pairs = []
        for row in entries:
            value = row["clock_before"].get("fields", {}).get("clocks.current.sm")
            if value is not None and value.replace(".", "", 1).isdigit():
                pairs.append((float(value), row["outer_per_replay_us"]))
        clocks[cohort] = {"available_samples": len(pairs), "sm_mhz": [p[0] for p in pairs],
                          "duration_pearson_r": statistics.correlation(*zip(*pairs))
                          if len(pairs) > 1 and len({p[0] for p in pairs}) > 1
                          and len({p[1] for p in pairs}) > 1 else None}
    return {"admitted": False, "protocol_checks": protocol, "paired_deltas": paired,
            "outer_delta_us_by_order": effects,
            "outer_mean_us_by_cohort": {c: statistics.fmean(r["outer_per_replay_us"] for r in entries)
                                        for c, entries in by_cohort.items()},
            "inner_B_median_us": statistics.median(v * 1000 for r in by_cohort["B"] for v in r["inner_durations_ms"]),
            "historical_single_replay_gap_us": reference["cell_independent_median_us"] - reference["cell_mean_us"],
            "predicted_outer_delta_us_from_historical_range": predicted,
            "H1_compatible_with_prior_gap_in_both_orders": all(protocol.values())
                 and all(predicted[0] <= value <= predicted[1] for value in effects.values()),
            "H2_order_reversal_signature": effects["AB"] * effects["BA"] < 0,
            "H2_clock_observations": clocks,
            "H3_kernel_identity_difference": not same_kernels,
            "interpretation": "Compatibility is diagnostic evidence, not proof or admission; clock samples are outside timing; no outliers removed or overhead subtracted"}


def native_run(settings, model_path, output):
    from frontier.profiling.common.model_config import ModelConfig
    from frontier.profiling.step5.device_activity import parse_device_activity
    from frontier.profiling.step5.replay import capture_native_replay
    from frontier.profiling.step5.vllm_wrapper import VllmStep5AttentionWrapper
    from frontier.profiling.utils.record_function_tracer import RecordFunctionTracer

    plan = describe(settings)
    cell_data = plan["cell"]
    cell = Step5ProfileInput(**{**cell_data, "query_lens": tuple(cell_data["query_lens"]),
                               "context_lens": tuple(cell_data["context_lens"])})
    config = ModelConfig.from_model_name(settings["model"])
    native, measurement = settings["native"], settings["measurement"]
    output.mkdir(parents=True, exist_ok=True)
    with VllmStep5AttentionWrapper(config=config, model_path=model_path,
            layer_id=cell.global_layer_id, graph_mode=cell.graph_mode, trace_dir=output / "traces", **native) as wrapper:
        torch = wrapper.torch
        with torch.no_grad(), wrapper._set_current(wrapper.native):
            torch.manual_seed(native["seed"])
            wrapper._restore(wrapper.empty_state)
            prefix = Step5ProfileInput(cell.global_layer_id, cell.context_lens, (0,), "prefill")
            prefix_hidden = torch.randn((prefix.num_tokens, config.embedding_dim), dtype=torch.bfloat16, device="cuda")
            wrapper._forward(prefix_hidden, wrapper._metadata(prefix))
            prepared = wrapper._metadata(cell)
            initial = wrapper._snapshot()
            hidden = torch.randn((cell.num_tokens, config.embedding_dim), dtype=torch.bfloat16, device="cuda")
            for _ in range(measurement["warmup_iterations"]):
                wrapper._restore(initial)
                wrapper._forward(hidden, prepared)
            torch.cuda.synchronize()
            wrapper._restore(initial)
            expected = wrapper._forward(hidden, prepared).clone()
            expected_state = wrapper._snapshot()
            wrapper._restore(initial)
            run = capture_native_replay(torch, wrapper.native, wrapper.layer,
                                       lambda **kwargs: wrapper._forward(hidden, prepared, **kwargs), cell.graph_mode)
            prepare = lambda: wrapper._restore(initial)
            validate = lambda actual: wrapper._validate_output_state(actual, expected, expected_state)
            prepare()
            validate(run())
            pointers = lambda: [hidden.data_ptr(), prepared[0].data_ptr(), prepared[1].data_ptr(),
                                *(value.data_ptr() for value in wrapper.state)]
            original_pointers = pointers()
            events = events_for(torch, measurement["replays"])
            with (output / "windows.jsonl").open("x") as stream:
                def record(row):
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                rows = collect(run, prepare, events, measurement, observe_clock=observe_clocks,
                               validate=validate, record=record)
            kernels = {}
            for cohort in "AB":
                _prime_events(events)
                prepare()
                torch.cuda.synchronize()
                tracer = RecordFunctionTracer(str(output))
                tracer.trace_path = str(output / "traces" / f"cohort_{cohort}.json")
                with tracer, torch.profiler.record_function(f"boundary_probe/{cohort}"), \
                        torch.profiler.record_function("native_replay"):
                    _, actual = measure_window(run, events, cohort, measurement["replays"])
                validate(actual)
                kernels[cohort] = parse_device_activity(tracer.trace_path, ("native_replay",),
                                                        iteration_prefix="boundary_probe/")
                kernels[cohort]["trace_path"] = tracer.trace_path
            if pointers() != original_pointers:
                raise AssertionError("Captured native input/cache addresses changed")
            result = {"plan": plan, "runtime_signature": wrapper.runtime_signature,
                      "native_impl": type(wrapper.layer.attn.impl).__name__,
                      "buffer_addresses": original_pointers, "windows": rows, "kernels": kernels,
                      "evaluation": evaluate(rows, kernels, settings), "settings": settings}
    with (output / "results.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("BOUNDARY_PROBE_COMPLETE", json.dumps(result["evaluation"], allow_nan=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--model-path")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    settings = json.loads(args.config.read_text())
    if args.dry_run:
        print(json.dumps(describe(settings), indent=2))
        return
    if not args.model_path or args.output_dir is None:
        parser.error("native execution requires --model-path and --output-dir")
    native_run(settings, args.model_path, args.output_dir)


if __name__ == "__main__":
    main()
