"""Extract the 300-second B20 profile, residual costs and rough replay estimates.

Usage: profile_report.py <window-run-root> <base-run-root> <candidate-run-root> <output.json>
"""
import json
from pathlib import Path
import pstats
import sys


CAUSES = {
    "run": "Top-level DES run, including the profiled sequential event loop.",
    "_run_sequential": "Processes the ordered event queue and invokes handlers.",
    "enter_layer_sync": "Collects attention-DP participants and starts each ready EP layer wave.",
    "schedule_layer_wave": "Schedules each layer's EP plan, phases and completion events.",
    "_prepare_moe_ep_wave_plan": "Delegates immutable workload and per-rank phase planning for each layer wave.",
    "prepare_moe_wave_from_inputs": "Materializes the scheduler workload and all lane timings for each wave.",
    "predict_ep_wave_phase_times": "Builds and prices every EP rank, including shared wrapper and communication work.",
    "_predict_layer_with_kernel_gap": "Prices each layer then validates/applies per-operator kernel gaps.",
    "_predict_dsv4_structural_execution": "V4 prediction entry for each attention lane and every routed EP rank.",
    "predict_structural_execution": "Constructs complete layer timing, operator workloads, runner scopes and communication payloads.",
    "predict_source_execution": "Validates each source batch and builds the per-call source components/workloads.",
    "handle_event": "Runs the event-specific synchronization, admission or completion handler.",
    "on_decode_sync": "Passes a DECODE lane's layer completion to the shared synchronization boundary.",
    "enter_decode_sync": "Registers DECODE lane completion and starts a ready cohort's next layer wave.",
    "_on_decode_ep_wave_ready": "Schedules ready DECODE EP waves and their continuation events.",
    "on_prefill_sync": "Passes a PREFILL lane's layer completion to the shared synchronization boundary.",
    "enter_prefill_sync": "Registers PREFILL lane completion and starts a ready cohort's next layer wave.",
    "_on_prefill_ep_wave_ready": "Schedules ready PREFILL EP waves and their continuation events.",
    "_build_prefill_ep_lane_batch": "Copies source requests and constructs a lane batch for each EP participant.",
    "build_ep_lane_batch": "Copies source requests into one EP lane batch and binds its immutable workload.",
    "materialize_layer_ep_workload": "Validates source/routing rows, lane descriptors and all expert ownership for each scheduler wave.",
    "scope_times": "Repeats attention model selection, coverage/schema/exact/finite/sign checks; interpolation hits use the instance memo.",
    "build_payload_bytes": "Builds and validates each communication operator payload using runtime source/routing rows.",
    "_dsv4_ep_payload": "Validates the V4 EP routing and transfer tensor contract per payload request.",
    "_predict_comm_operator_with_context": "Validates communication metadata/payload then looks up the measured backend latency.",
    "build_prefill_ep_lane_batch": "Builds each EP participant's lane batch and request copies.",
    "_predict_comm_operator": "Resolves communication ownership and validates each runtime payload before backend lookup.",
    "_frontier_comm_payload_signature": "Validates and normalizes the communication payload contract per call.",
}


def function_record(key, value):
    file, line, name = key
    primitive, calls, own, cumulative, _ = value
    cause = CAUSES.get(name)
    if name == "__init__" and file.endswith("execution_time.py"):
        cause = "ExecutionTime constructs and validates a finalized layer's many fields and operator contracts."
    elif name == "__init__" and file.endswith("request.py"):
        cause = "Request copies made by EP lane batching initialize full request lifecycle state."
    return dict(file=file, line=line, function=name, primitive_calls=primitive,
                calls=calls, own_s=own, cumulative_s=cumulative, cause=cause)


def main(window, base, candidate):
    receipt = json.loads((window / "receipt.json").read_text())
    pair = [json.loads((root / "receipt.json").read_text()) for root in (base, candidate)]
    stats = pstats.Stats(str(window / "profile.pstats"))
    ordered = sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)
    residuals = [function_record(key, value) for key, value in ordered if (
        key[0].endswith("/entities/execution_time.py") and key[2] == "__init__"
        or key[0].endswith("/entities/request.py") and key[2] == "__init__"
        or key[2] in ("_build_prefill_ep_lane_batch", "build_prefill_ep_lane_batch", "_predict_comm_operator",
                     "_frontier_comm_payload_signature", "materialize_layer_ep_workload", "scope_times",
                     "build_ep_lane_batch", "build_payload_bytes", "_dsv4_ep_payload", "_predict_comm_operator_with_context"))]
    rate = receipt["simulated_time_s"] / receipt["run_wall_s"]
    p32_s = 146.1887597443391 / rate
    p64_s = 2 * 152.99305567235012 / rate
    rss_growth = (pair[1]["peak_rss_kib"] - pair[0]["peak_rss_kib"]) * 1024
    probe_rss_growth = (receipt["peak_rss_kib"] - pair[0]["peak_rss_kib"]) * 1024
    return dict(status="PASS" if receipt["simulated_time_s"] >= 1.5 and max(rss_growth, probe_rss_growth) < 1_000_000_000 else "FAIL",
                window_receipt={k: receipt[k] for k in ("status", "run_wall_s", "simulated_time_s", "peak_rss_kib", "init_wall_s")},
                simulated_seconds_per_wall_second=rate, top15=[function_record(*item) for item in ordered[:15]],
                residual_costs=residuals, peak_rss_growth_bytes=rss_growth,
                probe_peak_rss_growth_bytes=probe_rss_growth,
                identity_run_speedup=pair[0]["run_wall_s"] / pair[1]["run_wall_s"],
                estimated_p2_32_wall_s=p32_s, estimated_p2_32_wall_h=p32_s / 3600,
                estimated_p2_64_wall_s=p64_s, estimated_p2_64_wall_h=p64_s / 3600,
                estimate_assumptions="Linear extrapolation of the profiled initial P2-32 simulated-time rate to diagnostic durations; "
                                     "P2-64 assumed to cost twice as much per simulated second because it has twice the EP ranks. "
                                     "No full replay or P2-64 performance measurement was run; late-run behavior and cProfile overhead can change the rate.")


if __name__ == "__main__":
    result = main(*(Path(arg) for arg in sys.argv[1:4]))
    Path(sys.argv[4]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(result["status"] != "PASS")
