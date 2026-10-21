#!/usr/bin/env python3
"""Check one finished E2E gate ground-truth run (groundtruth-run-check, after the run).

  python check_groundtruth_run.py <case_dir> <run_tag> <mode>

Reads the run directory `<case_dir>/runs/groundtruth_<mode>/<run_tag>/` (run
manifest, launcher receipt, worker evidence) against the case manifest and
inputs, and writes `<case_dir>/analysis/run_check_status.groundtruth_<mode>.json`
with one entry per run tag. Each check is PASS or FAIL with its evidence; the
run's status is PASS only when every check passes. For a co-location case the
request rows come from request_metrics.jsonl, for a pd-disaggregation case from
the pd_metrics.py join and both instances' connector checks. A schedule_timing,
op_timing or kernel_timing run takes its request rows from the scheduler's own
log, schedule.jsonl (one per instance for a pd-disaggregation case), which must
also account for every scheduled step and token. An instrumented or cpu run
(request metrics off) takes its request rows from the replay client's records,
client_requests.jsonl; an instrumented run's MoE routing records are checked by
the case's routing check, and a cpu run's engine probe log (cpu_probe.jsonl,
one per instance) must hold every engine step, its stamps in execution order,
and every trace request, and its stage records (below) every forward. A
kv_save_timing or kv_save_device_ids run (diagnostic, pd-disaggregation, a
vLLM-BS diagnostic commit with the CPU-probe record format of 8604683bf) passes
the cpu run's checks, and its prefill instance's KV-save probe
(kv_save_probe.jsonl) must hold, inside each forward of the CPU probe's stage
records, one layer save per layer for every scheduled request and one
wait_for_save, and a send for every layer save; the decode instance saves
nothing. A device_timeline run (diagnostic,
co-location) passes the cpu run's checks; its worker reduced the traces of
every rank to run/device_timeline.json (forward_device_timeline.py, records of
global rank 0), which must name rank 0 of the server's world size and hold
eager forwards, and the
replay must have opened and closed the profiler.

The engine file is the one the run manifest names (manifests written before
launch.json engine_by_mode existed name none; those runs used inputs/engine.json).
Likewise the trace is the run manifest's trace_dir, a scenario-matrix cell's
trace for ticket 41 runs; earlier manifests name none and used inputs/trace.
The server environment must carry VLLM_MOE_UNIFORM_ROUTING=1 exactly when the
engine file sets moe_uniform_routing. When the engine file sets
compilation_config.cudagraph_mode FULL_DECODE_ONLY,
every server log must show the decode graphs: the engine config with that
level and cudagraph_mode [2, 0], the capture sizes vLLM derives from
max_num_seqs, the decode FULL capture complete for each size up to
max_num_seqs, "Graph capturing finished", no mixed prefill-decode capture, and
neither the eager-mode message nor the empty-dispatcher warning. The derived KV
block counts (GPU KV cache tokens / block_size, minimum over engines) are
reported for inputs/frontier.json.

With the engine file's data_parallel_size > 1, the schedule log's steps are
counted per DP engine core, each engine core writes its own CPU probe log
(cpu_probe_dp<d>.jsonl), and the trace requests are split over the engine
cores, none served by two. TP rank 0 of each PP stage, a single stage included,
writes stage records (cpu_probe[_dp<d>]_pp<p>.jsonl, vLLM-BS 8604683bf): each
stage holds every forward of its engine core in order (joined on the scheduled
tokens), stamps in execution order, a PP receive on every stage but the first
and a send on every stage but the last, sampling stamps on the last stage only,
DP all-reduce stamps inside the forward's input preparation and context entry
exactly when data_parallel_size > 1, a positive forward device time, and the
same padded token count and CUDA graph mode on every stage of a batch. The
server log shows graph capture only from the global first rank, so with
FULL_DECODE_ONLY every stage of every engine core that decodes must replay
decode graphs (cudagraph_runtime_mode FULL); a PD prefill instance runs no
decode step.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import yaml

CPU_PROBE = "VLLM_FRONTIER_CPU_PROBE_LOG_PATH"
KV_SAVE_PROBE = "VLLM_FRONTIER_KV_SAVE_PROBE_LOG_PATH"
KV_SAVE_DEVICE_IDS = "VLLM_FRONTIER_KV_SAVE_DEVICE_BLOCK_IDS"
KV_SAVE_MODES = ("kv_save_timing", "kv_save_device_ids")
MODE_SWITCHES = {
    "clean": ({"VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH"},
              {"VLLM_FRONTIER_INSTRUMENTATION", "VLLM_FRONTIER_MOE_ROUTING_LOG_PATH", CPU_PROBE}),
    "instrumented": ({"VLLM_FRONTIER_INSTRUMENTATION", "VLLM_FRONTIER_MOE_ROUTING_LOG_PATH"},
                     {"VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH", "VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH",
                      "VLLM_FRONTIER_BATCH_LOG_PATH", "VLLM_FRONTIER_PP_BOUNDARY_LOG_PATH",
                      "VLLM_FRONTIER_SCHED_LOG_PATH", CPU_PROBE}),
    # op_timing and kernel_timing differ only in VLLM_FRONTIER_CUDA_EVENT_SCOPE_MODE.
    **{timing: ({"VLLM_FRONTIER_INSTRUMENTATION", "VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH",
                 "VLLM_FRONTIER_BATCH_LOG_PATH", "VLLM_FRONTIER_SCHED_LOG_PATH"},
                {"VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH", "VLLM_FRONTIER_MOE_ROUTING_LOG_PATH", CPU_PROBE})
       for timing in ("op_timing", "kernel_timing")},
    "schedule_timing": ({"VLLM_FRONTIER_SCHED_LOG_PATH"},
                        {"VLLM_FRONTIER_INSTRUMENTATION", "VLLM_FRONTIER_MOE_ROUTING_LOG_PATH",
                         "VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH", "VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH",
                         "VLLM_FRONTIER_BATCH_LOG_PATH", "VLLM_FRONTIER_PP_BOUNDARY_LOG_PATH", CPU_PROBE}),
    "cpu": ({CPU_PROBE},
            {"VLLM_FRONTIER_INSTRUMENTATION", "VLLM_FRONTIER_MOE_ROUTING_LOG_PATH",
             "VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH", "VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH",
             "VLLM_FRONTIER_BATCH_LOG_PATH", "VLLM_FRONTIER_PP_BOUNDARY_LOG_PATH",
             "VLLM_FRONTIER_SCHED_LOG_PATH"}),
    "kv_save_timing": ({CPU_PROBE, KV_SAVE_PROBE},
                       {"VLLM_FRONTIER_INSTRUMENTATION", "VLLM_FRONTIER_MOE_ROUTING_LOG_PATH",
                        "VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH", "VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH",
                        "VLLM_FRONTIER_BATCH_LOG_PATH", "VLLM_FRONTIER_PP_BOUNDARY_LOG_PATH",
                        "VLLM_FRONTIER_SCHED_LOG_PATH", KV_SAVE_DEVICE_IDS}),
}
MODE_SWITCHES["kv_save_device_ids"] = (MODE_SWITCHES["kv_save_timing"][0] | {KV_SAVE_DEVICE_IDS},
                                       MODE_SWITCHES["kv_save_timing"][1] - {KV_SAVE_DEVICE_IDS})
# The KV-save switches are off in every gate mode.
for gate_mode in ("clean", "instrumented", "op_timing", "kernel_timing", "schedule_timing", "cpu"):
    MODE_SWITCHES[gate_mode][1].update({KV_SAVE_PROBE, KV_SAVE_DEVICE_IDS})
# vLLM's torch profiler with the model runner's Forward ranges, next to the CPU probe (vllm_replay.py).
MODE_SWITCHES["device_timeline"] = (
    MODE_SWITCHES["cpu"][0] | {"VLLM_TORCH_PROFILER_DIR", "VLLM_CUSTOM_SCOPES_FOR_PROFILING"},
    MODE_SWITCHES["cpu"][1])
# Stamps of one CPU-probe engine step in execution order (vLLM-BS frontier_trace.py, EngineCore.step
# and step_with_batch_queue).
CPU_PROBE_STAMPS = ("step_start", "schedule_end", "execute_end", "update_end")
# Stamps of one PP stage record in execution order (vLLM-BS 8604683bf frontier_trace.py,
# Worker.execute_model around GPUModelRunner.execute_model); the first stage receives nothing, the
# last sends nothing, and only the last samples.
CPU_PROBE_STAGE_STAMPS = ("recv_start", "recv_end", "execute_start", "preprocess_end", "forward_start",
                          "forward_context_entered", "forward_end", "sample_end", "bookkeep_end",
                          "execute_return", "send_start", "send_end")
# CUDAGraphMode.FULL_DECODE_ONLY = (FULL, NONE) as the engine config line prints it
# (vllm/config/compilation.py of the fork).
FULL_DECODE_ONLY = [2, 0]
DECODE_CAPTURE = re.compile(r"Capturing CUDA graphs \(decode, FULL\).*?(\d+)/(\d+)")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verdict(passed: bool, evidence: str) -> str:
    return f"{'PASS' if passed else 'FAIL'}: {evidence}"


def capture_sizes(engine: dict) -> tuple[list[int], list[int]]:
    """Graph sizes vLLM 0.10.2 captures without explicit sizes, and the decode subset.

    VllmConfig._set_cudagraph_sizes takes 1, 2, 4 and the multiples of 8 up to
    min(2 * max_num_seqs, 512), at most max_num_batched_tokens; the model runner
    captures a decode graph for each size up to max_num_seqs (one query token
    per request without speculative decoding).
    """
    largest = min(2 * engine["max_num_seqs"], 512)
    sizes = [size for size in (1, 2, 4, *range(8, largest + 1, 8))
             if size <= engine["max_num_batched_tokens"]]
    return sizes, [size for size in sizes if size <= engine["max_num_seqs"]]


def decode_graph_sizes(engine: dict) -> list[int]:
    """Batch sizes whose pure decode steps replay a decode CUDA graph; an eager engine has none."""
    return [] if engine["enforce_eager"] else capture_sizes(engine)[1]


def decode_graph_evidence(server_log: Path, engine: dict) -> dict:
    text = server_log.read_text(errors="replace")
    config_lines = [line for line in text.splitlines() if "Initializing a V1 LLM engine" in line]
    configs = [json.JSONDecoder().raw_decode(line, line.index("compilation_config=") + 19)[0]
               for line in config_lines]
    sizes, decode_sizes = capture_sizes(engine)
    captured = [(int(done), int(total)) for segment in re.split(r"[\r\n]", text)
                for done, total in DECODE_CAPTURE.findall(segment)]
    evidence = {
        "engine_configs": [{"level": c.get("level"), "cudagraph_mode": c.get("cudagraph_mode"),
                            "cudagraph_capture_sizes": c.get("cudagraph_capture_sizes")} for c in configs],
        "expected_capture_sizes": sorted(sizes, reverse=True),
        "decode_capture_last": list(captured[-1]) if captured else None,
        "expected_decode_captures": len(decode_sizes),
        "graph_capturing_finished": "Graph capturing finished" in text,
        "mixed_capture": "Capturing CUDA graphs (mixed prefill-decode" in text,
        "eager_message": "Cudagraph is disabled under eager mode" in text,
        "dispatcher_empty": "cudagraph dispatching keys are not initialized" in text,
    }
    evidence["passed"] = (
        bool(configs)
        and all(c.get("level") == engine["compilation_config"]["level"]
                and c.get("cudagraph_mode") == FULL_DECODE_ONLY
                and c.get("cudagraph_capture_sizes") == evidence["expected_capture_sizes"] for c in configs)
        and evidence["decode_capture_last"] == [len(decode_sizes), len(decode_sizes)]
        and evidence["graph_capturing_finished"]
        and not evidence["mixed_capture"]
        and not evidence["eager_message"]
        and not evidence["dispatcher_empty"]
    )
    return evidence


def schedule_log_rows(schedule_log: Path, remote_prefix: bool) -> tuple[list[dict], str, bool]:
    """Request rows of the scheduler log and whether its steps account for every token.

    remote_prefix marks a PD decode instance: P2pNcclConnector.get_num_new_matched_tokens
    loads every prompt token but the last from the prefill instance (no prefix caching).
    """
    events = read_jsonl(schedule_log)
    steps = [event for event in events if event["event"] == "schedule"]
    completed = [event for event in events if event["event"] == "request_completed"]
    # Each DP engine core numbers its own steps.
    engine_steps: dict[int, list[dict]] = {}
    for step in steps:
        engine_steps.setdefault(step["dp_rank"], []).append(step)
    scheduled, preempted = {}, set()
    for step in steps:
        preempted.update(step["preempted_req_ids"])
        for request_id, tokens in step["num_scheduled_tokens"].items():
            scheduled[request_id] = scheduled.get(request_id, 0) + tokens
    # Without preemption a request is scheduled for its locally computed prompt
    # tokens and every output token but the last.
    unaccounted = sorted(
        event["request_id"] for event in completed if event["request_id"] not in preempted
        and scheduled.get(event["request_id"]) != event["num_output_tokens"] + (
            0 if remote_prefix else event["num_prompt_tokens"] - 1))
    consecutive = all([step["step"] for step in dp_steps] == list(range(len(dp_steps)))
                      for dp_steps in engine_steps.values())
    ordered = all(a["timestamp_monotonic"] <= b["timestamp_monotonic"]
                  for dp_steps in engine_steps.values() for a, b in pairwise(dp_steps))
    rows = [{"request_id": event["request_id"], "request_num_decode_tokens": event["num_output_tokens"]}
            for event in completed]
    passed = bool(steps) and consecutive and ordered and not unaccounted
    return rows, (f"{len(steps)} steps over DP ranks {sorted(engine_steps)}, consecutive={consecutive}, "
                  f"monotonic={ordered}, "
                  f"{len(completed)} completed, preempted {sorted(preempted)[:5]}, "
                  f"token count differs for {unaccounted[:5]}"), passed


def cpu_probe_evidence(probe_log: Path) -> tuple[set[str], str, bool]:
    """Engine request ids in the CPU probe's engine log and whether its steps are complete."""
    steps = read_jsonl(probe_log) if probe_log.exists() else []
    forward_steps = [step for step in steps if step["num_scheduled_tokens"]]
    unordered = [step["step"] for step in steps if any(step[a] > step[b] for a, b in pairwise(CPU_PROBE_STAMPS))]
    consecutive = [step["step"] for step in steps] == list(range(len(steps)))
    seen = {request_id for step in steps for request_id in step["num_scheduled_tokens"]}
    passed = bool(forward_steps) and consecutive and not unordered
    return seen, (f"{len(steps)} steps ({len(forward_steps)} with tokens), consecutive={consecutive}, "
                  f"stamps out of order {unordered[:5]}"), passed


def cpu_probe_log(role_dir: Path, dp_rank: int, dp_size: int, pipeline_stage_id: int | None = None) -> Path:
    """The probe log one engine core (or one of its PP stages) writes (vLLM-BS frontier_trace._cpu_probe_log_path)."""
    name = "cpu_probe" + (f"_dp{dp_rank}" if dp_size > 1 else "")
    if pipeline_stage_id is not None:
        name += f"_pp{pipeline_stage_id}"
    return role_dir / f"{name}.jsonl"


def cpu_probe_stage_evidence(engine_log: Path, stage_logs: list[Path], decode_graphs: bool,
                             data_parallel: bool) -> tuple[str, bool]:
    """Whether the PP stage records of one engine core hold its every forward, in order.

    Every stage runs the batches the engine core submits in submission order,
    so the n-th record of each stage and the engine core's n-th step with tokens
    are the same batch; the join is checked on the scheduled tokens.
    """
    forwards = [step["num_scheduled_tokens"] for step in read_jsonl(engine_log) if step["num_scheduled_tokens"]]
    stages = [read_jsonl(path) if path.exists() else [] for path in stage_logs]
    last = len(stages) - 1
    evidence, passed = {}, True
    for stage, records in enumerate(stages):
        unordered = [r["step"] for r in records
                     if any(a > b for a, b in pairwise(r[f] for f in CPU_PROBE_STAGE_STAMPS if r[f] is not None))]
        wrong_transfers = [r["step"] for r in records
                           if (r["recv_start"] is None) != (stage == 0) or (r["send_start"] is None) != (stage == last)]
        wrong_sampling = [r["step"] for r in records if (r["sample_end"] is None) == (stage == last)]
        # The all-reduce runs in the input preparation with CUDA graphs and at context entry eager.
        wrong_dp_allreduce = [
            r["step"] for r in records
            if (r["dp_allreduce_start"] is not None) != data_parallel
            or (data_parallel and not r["execute_start"] <= r["dp_allreduce_start"] <= r["dp_allreduce_end"]
                <= r["forward_context_entered"])]
        no_device = [r["step"] for r in records if not r["forward_device_ms"] > 0]
        joined = [r["num_scheduled_tokens"] for r in records] == forwards
        modes = Counter(r["cudagraph_runtime_mode"] for r in records)
        eager_decode = sum(r["cudagraph_runtime_mode"] == "NONE" and set(r["num_scheduled_tokens"].values()) == {1}
                           for r in records)
        consecutive = [r["step"] for r in records] == list(range(len(records)))
        evidence[f"pp{stage}"] = (
            f"{len(records)} records for {len(forwards)} engine forwards, joined={joined}, "
            f"consecutive={consecutive}, stamps out of order {unordered[:5]}, PP transfer stamps wrong "
            f"{wrong_transfers[:5]}, sampling stamps wrong {wrong_sampling[:5]}, DP all-reduce stamps wrong "
            f"{wrong_dp_allreduce[:5]}, without device time {no_device[:5]}, graph modes {dict(modes)}, "
            f"one-token-per-request batches run eager {eager_decode}")
        passed &= (bool(records) and joined and consecutive and not unordered and not wrong_transfers
                   and not wrong_sampling and not wrong_dp_allreduce and not no_device
                   and (modes["FULL"] > 0 or not decode_graphs))
    batches = [[(r["num_input_tokens"], r["cudagraph_runtime_mode"]) for r in records] for records in stages]
    same_batches = all(stage_batches == batches[0] for stage_batches in batches)
    evidence["stages_agree"] = f"padded token count and graph mode equal on every stage: {same_batches}"
    return json.dumps(evidence), passed and same_batches


def kv_save_probe_evidence(probe_log: Path, cpu_probe_log: Path) -> tuple[str, bool]:
    """Whether the prefill instance's KV-save probe covers every forward.

    The connector saves each layer's KV of each scheduled request inside the
    model forward and waits for the sends before the forward returns, so each
    forward of the CPU probe's stage record, execute_start to forward_end, holds
    those records.
    """
    records = read_jsonl(probe_log) if probe_log.exists() else []
    saves = [r for r in records if r["event"] == "layer_save"]
    waits = [r for r in records if r["event"] == "wait_for_save"]
    sent = {r["tensor_id"] for r in records if r["event"] == "send"}
    layers = {r["tensor_id"].split("#", 1)[1] for r in saves}
    steps = read_jsonl(cpu_probe_log)
    incomplete, in_window = [], 0
    for step in steps:
        window = (step["execute_start"], step["forward_end"])
        step_saves = [r for r in saves if window[0] <= r["start"] <= window[1]]
        in_window += len(step_saves)
        layers_by_request: dict[str, set[str]] = {}
        for r in step_saves:
            request_id, layer = r["tensor_id"].split("#", 1)
            layers_by_request.setdefault(request_id, set()).add(layer)
        step_waits = sum(window[0] <= r["start"] <= window[1] for r in waits)
        if (set(layers_by_request) != set(step["num_scheduled_tokens"])
                or any(v != layers for v in layers_by_request.values()) or step_waits != 1):
            incomplete.append(step["step"])
    unsent = sorted({r["tensor_id"] for r in saves} - sent)
    passed = bool(steps) and bool(layers) and not incomplete and not unsent
    return (f"{len(saves)} layer saves over {len(layers)} layers, {len(waits)} wait_for_save, {len(sent)} sends; "
            f"{len(steps)} forward steps, incomplete {incomplete[:5]}, saves outside a forward step "
            f"{len(saves) - in_window}, saves without a send {unsent[:3]}"), passed


def native_v4_source_binding(manifest: dict, engine: dict, config: dict) -> None:
    """Bind the single approved C26 exception, not an arbitrary native overlay."""
    base = "2cf0a6915ce544dc493a0990f2ea38d81601128a"
    leaf = "vllm/models/deepseek_v4/nvidia/model.py"
    original = "c01be6d313f494eb38982c8f56c314267bbf67310e7f13ca53b596b31aef4e44"
    approved = "d3556a47056bd52680b295f1585906149310397ec53ab90aa0008ff7d57e86ee"
    stage = Path(config["script"]).parent
    declaration = config["runtime_source_overlay"]
    if (manifest["groundtruth_commit"] != base or engine["native_reference"]["source_sha"] != base
            or declaration["base_source_sha"] != base or declaration["approved_c26_model_sha256"] != approved
            or not declaration["effective_runtime_source"]):
        raise ValueError("V4 runtime must declare the exact approved base2cf plus C26 source identity")
    seal_path, lock_path = stage / "manifest.json", stage / "source_lock.json"
    seal, lock = json.loads(seal_path.read_text()), json.loads(lock_path.read_text())
    if lock["native_commit"] != base or lock["harness_commit"] != manifest["harness_commit"]:
        raise ValueError("V4 source_lock does not bind the actual base and harness commits")
    overlay_path = Path(config["overlay_manifest"])
    if overlay_path != stage / "overlay.json":
        raise ValueError("V4 overlay manifest is outside the sealed execution package")
    overlay = json.loads(overlay_path.read_text())
    if lock["runtime_overlay"] != overlay:
        raise ValueError("V4 source_lock runtime overlay differs from the sealed manifest")
    row = overlay["files"]
    if set(row) != {leaf.removeprefix("vllm/")}:
        raise ValueError("V4 C26 is exactly one approved model file, not a general overlay")
    row = row[leaf.removeprefix("vllm/")]
    source = Path(row["source"])
    if (source != stage / "overlay_model.py" or row["base_sha256"] != original
            or row["sha256"] != approved or sha(source) != approved
            or sha(Path(manifest["groundtruth_checkout_path"]) / leaf) != original
            or config["expected_source_hashes"][leaf] != approved):
        raise ValueError("V4 approved C26 model/base/observed-source mapping mismatch")
    for name in ("overlay.json", "overlay_model.py"):
        path = stage / name
        actual = sha(path)
        if (seal["files"][name] != actual or config["retained_files"][name] != actual
                or manifest["inputs_sha256"][str(path)] != actual
                or (name == "overlay_model.py" and lock["retained_files"][name]["sha256"] != actual)):
            raise ValueError(f"V4 overlay is not sealed, source-locked and prelaunch-bound: {name}")
    if (seal["files"]["source_lock.json"] != sha(lock_path)
            or seal["files"]["config.json"] != manifest["inputs_sha256"][manifest["execution_config"]]
            or json.loads((stage / "config.json").read_text()) != config
            or seal_path.stat().st_mtime_ns > (Path(config["case_dir"]) / "runs" /
                "groundtruth_clean" / config["run_tag"] / "run_manifest.json").stat().st_mtime_ns):
        raise ValueError("V4 sealed package/source lock differs from its actual prelaunch config")
    for name, expected in config["expected_source_hashes"].items():
        if name != leaf and sha(Path(manifest["groundtruth_checkout_path"]) / name) != expected:
            raise ValueError(f"Unapproved V4 native source change: {name}")


def native_v2_graph_dispatch(manifest: dict, engine: dict):
    """Execute pinned CPU-only native sizing/candidate/dispatch functions without importing CUDA."""
    import ast
    import dataclasses
    import enum
    from itertools import product
    from types import MethodType, SimpleNamespace

    source = Path(manifest["groundtruth_checkout_path"])
    compilation_tree = ast.parse((source / "vllm/config/compilation.py").read_text())
    tree = ast.parse((source / "vllm/config/vllm.py").read_text())
    manager_tree = ast.parse((source / "vllm/v1/worker/gpu/cudagraph_utils.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VllmConfig")
    fields = {n.target.id: n.value for n in cls.body if isinstance(n, ast.AnnAssign)}
    level = fields["optimization_level"].attr
    if (engine.get("optimization_level", level) != level or engine.get("speculative_config") is not None
            or engine.get("enable_lora", False) or engine["enforce_eager"]):
        raise ValueError("Unsupported V4 graph-resolution inputs; no speculative, LoRA or eager substitute")
    defaults = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "OPTIMIZATION_LEVEL_0" + level[1:] for t in n.targets))
    values = lambda node: {ast.literal_eval(k): v for k, v in zip(node.keys, node.values)}
    configured = values(values(defaults)["compilation_config"])["cudagraph_mode"].attr
    requested = engine.get("compilation_config", {})
    configured = requested.get("cudagraph_mode", configured)
    namespace = dict(enum=enum, dataclass=dataclasses.dataclass, defaultdict=__import__("collections").defaultdict,
                     product=product, round_up=lambda a, b: (a + b - 1) // b * b)
    nodes = [next(n for n in compilation_tree.body if isinstance(n, ast.ClassDef) and n.name == "CUDAGraphMode"),
             next(n for n in manager_tree.body if isinstance(n, ast.ClassDef) and n.name == "BatchExecutionDescriptor"),
             next(n for n in manager_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_is_compatible")]
    manager_cls = next(n for n in manager_tree.body if isinstance(n, ast.ClassDef) and n.name == "CudaGraphManager")
    methods = ("_init_candidates", "_resolve_effective_loras", "dispatch")
    nodes += [n for n in manager_cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    sizing = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_set_cudagraph_sizes")
    # Replace only the CUDA-platform import with the locked Hopper capability;
    # every sizing/candidate/dispatch branch remains the pinned native function.
    for n in ast.walk(sizing):
        if isinstance(n, ast.If):
            n.body[:] = [ast.copy_location(ast.Pass(), child) if isinstance(child, ast.ImportFrom)
                         and child.module == "vllm.platforms" else child for child in n.body]
    nodes.append(sizing)
    namespace.update(current_platform=SimpleNamespace(is_device_capability_family=lambda family: False),
                     logger=SimpleNamespace(warning=lambda *args: None))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(source / "vllm/config/vllm.py") + ":native_graph_functions", "exec", dont_inherit=True), namespace)
    mode = namespace["CUDAGraphMode"][configured]
    if not mode or requested.get("pass_config", {}).get("enable_sp", False):
        raise ValueError("V4 supported native graph dispatch requires graphs and MoE sequence-parallelism disabled")
    compilation = SimpleNamespace(cudagraph_mode=mode,
        cudagraph_capture_sizes=requested.get("cudagraph_capture_sizes"),
        max_cudagraph_capture_size=requested.get("max_cudagraph_capture_size"),
        pass_config=SimpleNamespace(enable_sp=False), post_init_cudagraph_sizes=lambda: None)
    cfg = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False), compilation_config=compilation,
        scheduler_config=SimpleNamespace(max_num_seqs=engine["max_num_seqs"],
                                        max_num_batched_tokens=engine["max_num_batched_tokens"]),
        parallel_config=SimpleNamespace(tensor_parallel_size=engine["tensor_parallel_size"]),
        performance_mode=engine.get("performance_mode", ast.literal_eval(fields["performance_mode"])),
        num_speculative_tokens=0)
    namespace["_set_cudagraph_sizes"](cfg)
    manager = SimpleNamespace(max_num_reqs=engine["max_num_seqs"], compilation_config=compilation,
        cudagraph_mode=mode, decode_query_len=1, varlen_decode=False,
        vllm_config=SimpleNamespace(speculative_config=None), lora_capture_cases=[0],
        _lora_dispatch_map={}, _candidates={}, _capture_descs={}, _graphs_captured=True)
    for name in methods:
        setattr(manager, name, MethodType(namespace[name], manager))
    manager._init_candidates()
    captured = [dataclasses.asdict(desc) | {"cg_mode": desc.cg_mode.name}
                for desc in manager._capture_descs.get(namespace["CUDAGraphMode"].FULL, [])]
    return manager, captured


def native_rank_graph_evidence(nodes: dict, manifest: dict, engine: dict, run_dir: Path,
                               workload: dict, records: list[dict], config: dict) -> tuple[bool, str]:
    """Bind successful serving host calls to every physical rank and canonical phase."""
    try:
        manifest_time = (run_dir / "run_manifest.json").stat().st_mtime_ns
        generation = manifest["run_generation"]
        tp, dp = engine["tensor_parallel_size"], engine["data_parallel_size"]
        if (not generation or engine["pipeline_parallel_size"] != 1
                or config["data_parallel_size_local"] != 1
                or manifest["resources"]["gpus"] != 2 * tp * dp or len(nodes) != 2 * dp):
            raise ValueError("Incomplete P/D physical-rank configuration")
        source_files = ("vllm/v1/cudagraph_dispatcher.py", "vllm/forward_context.py",
                        "vllm/distributed/parallel_state.py", "vllm/compilation/cuda_graph.py",
                        "vllm/config/vllm.py", "vllm/config/compilation.py",
                        "vllm/v1/worker/gpu_model_runner.py", "vllm/v1/worker/gpu_worker.py",
                        "vllm/entrypoints/openai/completion/serving.py")
        source_hashes = config["expected_source_hashes"]
        architecture = json.loads((Path(config["model_dir"]) / "config.json").read_text())["architectures"]
        runners = {("GlmMoeDsaForCausalLM",): "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
                   ("DeepseekV4ForCausalLM",): "vllm.v1.worker.gpu.model_runner.GPUModelRunner"}
        runner = runners[tuple(architecture)]
        v2 = architecture == ["DeepseekV4ForCausalLM"]
        if v2:
            source_files += ("vllm/v1/worker/gpu/model_runner.py", "vllm/v1/worker/gpu/cudagraph_utils.py",
                             "vllm/compilation/breakable_cudagraph.py")
            native_v4_source_binding(manifest, engine, config)
        for name in source_files:
            if source_hashes[name] != sha(Path(manifest["groundtruth_checkout_path"]) / name):
                raise ValueError(f"Native counter/dispatch source pin mismatch: {name}")
        module = "tests/comparison/calibration/native_full_structure.py"
        staged = Path(config["script"]).parent / config["harness_files"][module]
        module_hash = manifest["harness_sha256"][module]
        if (sha(staged) != module_hash or manifest["inputs_sha256"][str(staged)] != module_hash
                or sha(Path(manifest["harness_worktree"]) / module) != module_hash
                or Path(config["harness_dir"]) / "native_full_structure.py" != staged):
            raise ValueError("Serving collector module is not bound to the pinned harness and staged inputs")
        phases = {(repeat, phase) for repeat in workload["repetitions"]["repeat_indices"]
                  for phase in ("warmup", "formal")}
        client_ids = {phase: {r["request_id"] for r in records
                             if (r["repeat"], r["segment"]) == phase} for phase in phases}
        if any(not ids for ids in client_ids.values()):
            raise ValueError("Missing canonical client phase IDs")
        hooks = {"vllm.v1.worker.gpu_worker.Worker." + name for name in
                 ("execute_model", "shutdown", "compile_or_warm_up_model")}
        hooks |= {"vllm.v1.worker.gpu_model_runner.GPUModelRunner." + name for name in
                  ("_model_forward", "_dummy_run")}
        hooks.add("torch.cuda.graphs.CUDAGraph.replay")
        import ast
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "install"
               and len(n.args) > 1 and isinstance(n.args[0], ast.Name) and n.args[0].id == "Worker"
               and isinstance(n.args[1], ast.Constant) and n.args[1].value == "load_model"
               for n in ast.walk(ast.parse(staged.read_text()))):
            hooks.add("vllm.v1.worker.gpu_worker.Worker.load_model")
        if v2:
            hooks.add(runner + ".execute_model")
            hooks |= {"vllm.v1.worker.gpu.cudagraph_utils.ModelCudaGraphManager." + name
                      for name in ("capture", "run_fullgraph", "run_pw_graph")}
            dispatch, captured = native_v2_graph_dispatch(manifest, engine)
            configured = dispatch.cudagraph_mode.name
            sizes = dispatch.compilation_config.cudagraph_capture_sizes
        else:
            compilation = engine["compilation_config"]
            configured = compilation["cudagraph_mode"]
            sizes = sorted(set(compilation["cudagraph_capture_sizes"]))
        modes = {"FULL_AND_PIECEWISE": {"FULL", "PIECEWISE"},
                 "FULL_DECODE_ONLY": {"FULL"}, "FULL": {"FULL"}, "PIECEWISE": {"PIECEWISE"}}
        if engine["enforce_eager"] or configured not in modes:
            raise ValueError("Locked native configuration does not establish serving graph support")
        if (not sizes or any(type(n) is not int or n <= 0 for n in sizes)
                or max(sizes) > engine["max_num_batched_tokens"]
                or (dispatch.compilation_config.max_cudagraph_capture_size if v2 else
                    compilation["max_cudagraph_capture_size"]) != max(sizes)):
            raise ValueError("Unsupported or inconsistent locked native capture sizes")
        owner_ids, identities, totals, rank_counts = {}, set(), {}, {}
        physical = set()
        for host, node in sorted(nodes.items()):
            plan, env = node["plan"], node["server_env_set"]
            instrumentation = node["serving_instrumentation"]
            role, owner = plan["role"], plan["lane"]
            node_dir = run_dir / "nodes" / host
            if (role not in ("prefill", "decode") or owner not in range(dp)
                    or node["native_source_hashes"] != source_hashes
                    or env["FRONTIER_NATIVE_RUN_GENERATION"] != generation
                    or env["FRONTIER_NATIVE_ROLE"] != role
                    or Path(env["FRONTIER_NATIVE_SERVING_EVIDENCE_DIR"]) != Path(config["output_dir"]) / "nodes" / host
                    or instrumentation["module_sha256"] != module_hash
                    or instrumentation["extension"] != "native_full_structure.FullModelInventory"
                    or instrumentation["native_python_callables_wrapped"] is not True
                    or instrumentation["device_completion_observed"] is not False
                    or instrumentation["overhead_measured"] is not False
                    or plan["engine"]["worker_extension_cls"] != "native_full_structure.FullModelInventory"):
                raise ValueError(f"Serving collector/source/environment provenance mismatch: {host}")
            command = node["server_command"]
            extension = command.index("--worker-extension-cls")
            if command[extension + 1] != "native_full_structure.FullModelInventory":
                raise ValueError(f"Native serving extension command mismatch: {host}")
            devices = {}
            for line in node["nvidia_smi"].splitlines():
                match = re.fullmatch(r"GPU (\d+): (.+) \(UUID: ([^)]+)\)", line)
                if match:
                    index, name, uuid = match.groups()
                    devices[index] = (name, uuid.removeprefix("GPU-").lower())
            files = list((node_dir / "serving").glob("rank*.json"))
            if {p.name for p in files} != {f"rank{rank}.json" for rank in range(tp)}:
                raise ValueError(f"Missing or unexpected physical-rank serving files: {host}")
            for rank in range(tp):
                path = node_dir / "serving" / f"rank{rank}.json"
                if path.stat().st_mtime_ns < manifest_time:
                    raise ValueError(f"Stale serving rank output: {path}")
                payload = json.loads(path.read_text())
                inventory, counters = payload["inventory"], payload["counters"]
                if (payload["schema"] != "native_serving_v1" or payload["stage"] != "pre_shutdown"
                        or payload["run_generation"] != generation or payload["role"] != role
                        or payload["host"] != host or payload["observation_errors"]
                        or set(payload["hooks"]) != hooks or counters["dropped_rows"] != 0
                        or inventory["host"] != host or inventory["worker_rank"] != rank
                        or inventory["request_owner"] != owner
                        or inventory["model_runner"] != runner):
                    raise ValueError(f"Incomplete or mismatched pre-shutdown serving inventory: {host}/{rank}")
                global_rank = owner * tp + rank
                groups = {"world": list(range(tp * dp)), "ep": list(range(tp * dp)),
                          "tp": list(range(owner * tp, (owner + 1) * tp)),
                          "dp": list(range(rank, tp * dp, tp))}
                for group, ranks in groups.items():
                    observed = inventory["groups"][group]
                    if (observed["rank"] != global_rank or observed["ranks"] != ranks
                            or observed["world_size"] != len(ranks)
                            or observed["rank_in_group"] != ranks.index(global_rank)):
                        raise ValueError(f"Native {group} membership mismatch: {host}/{rank}")
                device = inventory["device_identity"]
                visible = plan["engine"]["role_devices"][role][rank]
                normalized = str(visible).removeprefix("GPU-").lower()
                actual = (devices[str(visible)] if str(visible).isdigit() else
                          next(value for value in devices.values() if value[1] == normalized))
                identity = (host, device["uuid"].removeprefix("GPU-").lower())
                if (identity in identities or identity[1] != actual[1]
                        or manifest["resources"]["gpu_type"] not in actual[0]
                        or device["name"] != actual[0] or type(device["pid"]) is not int or device["pid"] <= 0
                        or device["device"] != inventory["device"] or device["device"] != f"cuda:{rank}"
                        or len(device["capability"]) != 2
                        or (v2 and device["capability"] != [9, 0])
                        or any(type(n) is not int or n < 0 for n in device["capability"])):
                    raise ValueError(f"Physical device/UUID mismatch: {host}/{rank}")
                identities.add(identity)
                physical.add((role, global_rank))
                effective = inventory["effective_graph"]
                expected_graph = dict(configured_mode=configured, dispatcher_mode=configured,
                                      keys_initialized=True, capture_sizes=sizes, max_capture_size=max(sizes))
                if v2:
                    expected_graph["captured_descriptors"] = captured
                if effective != expected_graph:
                    raise ValueError(f"Native effective graph configuration mismatch: {host}/{rank}")
                requests = {}
                for row in counters["request_ids"]:
                    phase = (row["repeat"], row["phase"])
                    if phase not in phases or phase in requests or not row["request_ids"]:
                        raise ValueError(f"Missing, repeated or unknown serving ID phase: {host}/{rank}")
                    ids = []
                    for name in row["request_ids"]:
                        match = re.fullmatch(r"cmpl-(.+)-0(?:-[0-9a-f]{8})?", name)
                        if match is None:
                            raise ValueError(f"Unrecognized pinned native completion ID: {name}")
                        ids.append(match.group(1))
                    if len(ids) != len(set(ids)) or not set(ids) <= client_ids[phase]:
                        raise ValueError(f"Serving IDs do not belong to the canonical client phase: {host}/{rank}")
                    requests[phase] = set(ids)
                if set(requests) != phases:
                    raise ValueError(f"Serving IDs do not cover every repetition/phase: {host}/{rank}")
                key = (role, owner)
                if key in owner_ids and owner_ids[key] != requests:
                    raise ValueError(f"TP ranks disagree on DP-owner serving IDs: {host}/{rank}")
                owner_ids[key] = requests
                sums = Counter()
                phase_counts = {phase: Counter() for phase in phases}
                for row in counters["rows"]:
                    forwards, replays = row["successful_forwards"], row["successful_replays"]
                    if (any(type(n) is not int or n < 0 for n in (forwards, replays))
                            or not forwards + replays or row["kind"] not in ("serving", "profile", "capture", "dummy")):
                        raise ValueError(f"Invalid or unattributed native counters: {host}/{rank}")
                    sums.update(successful_forwards=forwards, successful_replays=replays)
                    if row["kind"] != "serving":
                        if row["repeat"] is not None or row["phase"] is not None:
                            raise ValueError(f"Startup call mislabeled with a serving phase: {host}/{rank}")
                        continue
                    phase = (row["repeat"], row["phase"])
                    descriptor, shapes = row["batch_descriptor"], row["input_shapes"]
                    scheduled, num_requests = row["scheduled_num_tokens"], row["scheduled_num_requests"]
                    if (phase not in phases or not forwards or type(scheduled) is not int
                            or type(num_requests) is not int or not 0 < num_requests <= len(requests[phase])
                            or not num_requests <= scheduled <= engine["max_num_batched_tokens"]
                            or shapes["inputs_embeds"] is not None
                            or type(shapes["input_ids"]) is not list or len(shapes["input_ids"]) != 1
                            or any(type(n) is not int or n <= 0 for n in shapes["input_ids"])
                            or shapes["positions"] != shapes["input_ids"]):
                        raise ValueError(f"Unknown serving shape, phase or successful-forward count: {host}/{rank}")
                    tokens = shapes["input_ids"][0]
                    if (descriptor["num_tokens"] != tokens or not scheduled <= tokens <= engine["max_num_batched_tokens"]
                            or descriptor["has_lora"] is not False or descriptor["num_active_loras"] != 0):
                        raise ValueError(f"Native concrete shape/descriptor mismatch: {host}/{rank}")
                    mode = row["graph_mode"]
                    if v2:
                        native_desc = dispatch.dispatch(num_requests, scheduled,
                            1 if scheduled == num_requests else None, 0,
                            max_query_len=1 if scheduled == num_requests else None)
                        expected_descriptor = dict(num_tokens=native_desc.num_tokens,
                            num_reqs=native_desc.num_reqs if mode == "FULL" else None,
                            uniform=None if mode == "FULL" else False, has_lora=False, num_active_loras=0)
                        if (mode != native_desc.cg_mode.name or descriptor != expected_descriptor
                                or (mode == "NONE" and replays) or (mode != "NONE" and not replays)):
                            raise ValueError(f"V2 actual serving replay/descriptor differs from pinned native dispatch: {host}/{rank}")
                    elif mode == "NONE":
                        unsupported = tokens > max(sizes) or (configured == "FULL_DECODE_ONLY" and scheduled != num_requests)
                        if (not unsupported or replays or descriptor["num_reqs"] is not None
                                or descriptor["uniform"] is not False):
                            raise ValueError(f"NONE is not established by the native dispatch/config and shape: {host}/{rank}")
                    elif (mode not in modes[configured] or tokens not in sizes or not replays
                          or (mode == "FULL" and ((configured != "FULL" and scheduled != num_requests)
                              or descriptor["num_reqs"] != min(tokens, engine["max_num_seqs"])
                              or descriptor["uniform"] is not (configured != "FULL")))
                          or (mode == "PIECEWISE" and (descriptor["num_reqs"] is not None or descriptor["uniform"] is not False))):
                        raise ValueError(f"Missing successful supported graph replay or invalid capture descriptor: {host}/{rank}")
                    phase_counts[phase].update(forwards=forwards, replays=replays)
                    totals.setdefault(role, Counter()).update({mode + "_forwards": forwards, mode + "_replays": replays})
                if any(type(counters[name]) is not int or counters[name] != sums[name]
                       for name in ("successful_forwards", "successful_replays")):
                    raise ValueError(f"Successful native counters disagree with concrete rows: {host}/{rank}")
                if any(not counts["forwards"] for counts in phase_counts.values()):
                    raise ValueError(f"No successful serving forward in a required repetition/phase: {host}/{rank}")
                rank_counts[f"{role}/{host}/rank{rank}"] = {
                    f"repeat{repeat:02d}_{phase}": dict(phase_counts[repeat, phase])
                    for repeat, phase in sorted(phases)}
        if physical != {(role, rank) for role in ("prefill", "decode") for rank in range(tp * dp)}:
            raise ValueError("Physical-rank coverage is not complete for both roles")
        for role in ("prefill", "decode"):
            for phase in phases:
                per_owner = [owner_ids[role, owner][phase] for owner in range(dp)]
                if set().union(*per_owner) != client_ids[phase] or sum(map(len, per_owner)) != len(client_ids[phase]):
                    raise ValueError(f"DP serving IDs are missing or shared: {role}/{phase}")
        return True, json.dumps(dict(physical_ranks=len(physical), phases_per_rank=len(phases),
            serving_counts=totals, per_rank_phase_counts=rank_counts, collector_sha256=module_hash,
            coverage="Successful serving host returns and CUDAGraph.replay returns only; no device completion/timing claim"))
    except (Exception, SystemExit) as error:
        return False, f"UNVERIFIED: {type(error).__name__}: {error}"


def native_idle_evidence(repetitions: dict, records: list[dict], workload: dict, dp: int,
                         dedicated_services: dict | None = None) -> tuple[bool, str]:
    """Recheck the retained observe_idle snapshots, not their verified labels."""
    repeats = workload["repetitions"]["repeat_indices"]
    phases = [(repeat, phase) for repeat in repeats for phase in ("warmup", "formal")]
    names = ["before_repeat01_warmup"] + [f"after_repeat{r:02d}_{p}" for r, p in phases]
    boundaries = repetitions["idle_boundaries"]
    if [b["boundary"] for b in boundaries] != names:
        return False, "Missing or reordered canonical idle boundaries"
    gauges = {"num_requests_running", "num_requests_waiting", "kv_cache_usage_perc"}
    completed, previous_time, previous_end = None, -math.inf, -math.inf
    for index, boundary in enumerate(boundaries):
        samples = boundary["observations"]
        if not boundary["verified"] or boundary.get("error") or not samples:
            return False, f"Unverified idle observations: {boundary['boundary']}"
        for sample in samples:
            stamp = sample["monotonic"]
            if not math.isfinite(stamp) or stamp <= previous_time or sample["errors"]:
                return False, f"Unordered or erroneous idle observation: {boundary['boundary']}"
            previous_time = stamp
        sample = samples[-1]
        if set(sample["roles"]) != {"prefill", "decode"}:
            return False, f"Idle observation does not cover both roles: {boundary['boundary']}"
        counts = {}
        cold = index == 0 and boundary.get("proof") == "dedicated_services_ready_client_zero_native_counters_zero"
        if cold and (not dedicated_services or boundary.get("fresh_services") != dedicated_services
                     or boundary.get("client_dispatched_requests") != 0
                     or any(r["dispatch_monotonic"] is None or r["dispatch_monotonic"] <= sample["monotonic"]
                            for r in records)):
            return False, "Unbound dedicated service or existing-client cold-start evidence"
        for role, observed in sample["roles"].items():
            lanes = observed["lanes"]
            unpublished = cold and all(not values for values in lanes.values())
            if (set(lanes) != {str(i) for i in range(dp)} or not unpublished and (not observed["idle"]
                    or any(set(values) != gauges or any(type(v) not in (int, float)
                            or not math.isfinite(v) or v != 0 for v in values.values())
                            for values in lanes.values()))):
                return False, f"Non-idle, missing or invalid {role} DP lane gauges: {boundary['boundary']}"
            count = observed["completed_requests"]
            if cold:
                counters = observed["completion_counters"]
                reasons = {"stop", "length", "abort", "error", "repetition"}
                if (count != 0 or set(counters) != {str(i) for i in range(dp)}
                        or any(set(values) != reasons or any(type(v) not in (int, float) or v != 0
                               for v in values.values()) for values in counters.values())):
                    return False, "Missing or nonzero cold-start native DP lane completion counters"
            if type(count) not in (int, float) or not math.isfinite(count) or count < 0 or int(count) != count:
                return False, f"Invalid native completion count: {role}/{boundary['boundary']}"
            counts[role] = count
        if index:
            repeat, phase = phases[index - 1]
            rows = [r for r in records if r["repeat"] == repeat and r["segment"] == phase]
            expected = (workload["selection"]["num_formal_requests"] if phase == "formal"
                        else workload["warmup"]["num_requests_per_repetition"])
            if (len(rows) != expected or min(r["dispatch_monotonic"] for r in rows) <= previous_end
                    or max(r["last_token_monotonic"] for r in rows) > sample["monotonic"]
                    or any(counts[role] < completed[role] + expected for role in counts)):
                return False, f"Idle boundary is not after the completed client phase: {boundary['boundary']}"
        previous_end, completed = sample["monotonic"], counts
    return True, f"{len(boundaries)} boundaries; both roles, all {dp} DP lanes and fresh phase completion counts"


def check_native_canonical(case_dir: Path, run_dir: Path, run_tag: str, mode: str,
                           case: dict, manifest: dict, request_ids: dict, engine: dict) -> dict:
    """Admit native artifacts at the existing case-bound groundtruth-run-check boundary."""
    checks, nodes, summary, config, workload, records = {}, {}, {}, {}, None, []
    manifest_path = run_dir / "run_manifest.json"

    def record(name, operation):
        try:
            passed, evidence = operation()
            checks[name] = verdict(passed, evidence)
            return passed
        except (Exception, SystemExit) as error:
            checks[name] = verdict(False, f"UNVERIFIED: {type(error).__name__}: {error}")
            return False

    def read_output(path):
        if path.stat().st_mtime_ns < manifest_path.stat().st_mtime_ns:
            raise ValueError(f"Stale output predates run manifest: {path}")
        return json.loads(path.read_text())

    def provenance():
        written = datetime.fromisoformat(manifest["written_before_launch_utc"].replace("Z", "+00:00"))
        if written.tzinfo is None or written > datetime.now(timezone.utc):
            raise ValueError("Missing timezone or future pre-launch timestamp")
        source = manifest["groundtruth_checkout_path"]
        commit = git(source, "rev-parse", "HEAD")
        branch = git(source, "branch", "--show-current")
        dirty = git(source, "status", "--porcelain", "--untracked-files=all")
        passed = (mode == manifest["mode"] == "clean" and manifest["case_id"] == case["case_id"]
                  and manifest["run_tag"] == run_tag
                  and manifest["run_generation"] == case["run_generation"] and bool(manifest["run_generation"])
                  and Path(source).resolve() == Path(case["groundtruth_checkout_path"]).resolve()
                  and bool(manifest["authorization"]) and manifest["groundtruth_tree_dirty"] is False
                  and commit == manifest["groundtruth_commit"] == case["groundtruth_commit"]
                  == engine["native_reference"]["source_sha"]
                  and branch == manifest["groundtruth_branch"] == case["groundtruth_branch"] and not dirty)
        return passed, f"{source}: {branch} {commit}, actual dirty={dirty!r}; case/run/mode/generation bound"

    record("checkout", provenance)

    def inputs():
        nonlocal config
        required = {"manifest.yaml", manifest["engine_config"], manifest["execution_config"]} | {
            str(Path(manifest["trace_dir"]) / name) for name in
            ("manifest.json", "trace.csv", "request_ids.json", "warmup_pool.csv", "SHA256SUMS")}
        hashes = manifest["inputs_sha256"]
        if not required <= set(hashes):
            raise ValueError(f"Missing input hashes: {sorted(required - set(hashes))}")
        stale = [name for name, digest in hashes.items() if sha(case_dir / name) != digest]
        config = json.loads((case_dir / manifest["execution_config"]).read_text())
        model = Path(config["model_dir"])
        if (not model.is_absolute() or not config["model_hashes"] or not config["expected_source_hashes"]
                or config["mode"] != "clean" or config["engine"] != engine or config["run_tag"] != run_tag):
            raise ValueError("Missing native model/source hashes or execution engine/mode mismatch")
        for name, digest in config["model_hashes"].items():
            if hashes.get(str(model / name)) != digest:
                raise ValueError(f"Missing model input hash: {model / name}")
        return not stale, f"changed since launch: {stale}"

    record("inputs_unchanged", inputs)

    def harness_readers():
        nonlocal workload
        harness = manifest["harness_worktree"]
        commit = manifest["harness_commit"]
        dirty = git(harness, "status", "--porcelain", "--untracked-files=all")
        if (git(harness, "rev-parse", "HEAD") != commit or dirty or manifest["harness_tree_dirty_paths"]
                or git(harness, "branch", "--show-current") != manifest["harness_branch"]):
            raise ValueError("Harness HEAD/branch/clean status differs from recorded clean provenance")
        required = {"tests/comparison/calibration/" + name + ".py" for name in
                    ("client_metrics", "pd_replay", "pd_metrics", "pd_proxy", "e2e_metrics_gap", "native_pd_worker")}
        required |= {"tests/comparison/dp_placement_pp/vllm_replay.py",
                     "tests/comparison/stage_admission_pp/vllm_burst_driver.py"}
        if not required <= set(manifest["harness_sha256"]):
            raise ValueError(f"Missing reader dependency hashes: {sorted(required - set(manifest['harness_sha256']))}")
        for name, digest in manifest["harness_sha256"].items():
            path = Path(harness) / name
            if Path(name).is_absolute() or not path.resolve().is_relative_to(Path(harness).resolve()):
                raise ValueError(f"Harness module outside checkout: {name}")
            content = subprocess.run(["git", "-C", harness, "show", f"{commit}:{name}"],
                                     check=True, capture_output=True).stdout
            if sha(path) != digest or hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Harness tracked module mismatch: {name}")
        # A fresh interpreter prevents preloaded modules from a different worktree being consumed.
        code = '''import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
from tests.comparison.calibration.pd_replay import canonical_workload, instance_env, kv_transfer_config
from tests.comparison.calibration.client_metrics import native_client_rows
from vllm_replay import server_command
run, trace = map(pathlib.Path, sys.argv[2:4])
workload = canonical_workload(trace)
rows = native_client_rows(run, workload, json.loads((trace / "request_ids.json").read_text()))
manifest = json.loads((run / "run_manifest.json").read_text())
case = run.parents[2]
config = json.loads((case / manifest["execution_config"]).read_text())
commands = {}
for node in sorted((run / "nodes").glob("*/result.json")):
    result = json.loads(node.read_text())
    plan = result["plan"]
    role, engine = plan["role"], plan["engine"]
    port = config["http_ports"][role]
    command = server_command(engine, pathlib.Path(config["model_dir"]), port)
    command += ["--kv-transfer-config", json.dumps(kv_transfer_config(engine, role, port))]
    _, env = instance_env("clean", engine, node.parent, {},
                          ",".join(engine["role_devices"][role]),
                          observer=config["serve_observers"][role], role=role)
    env["VLLM_ENGINE_READY_TIMEOUT_S"] = str(config["startup_timeout_s"])
    actual_env = dict(result["server_env_set"])
    if engine.get("worker_extension_cls") == "native_full_structure.FullModelInventory":
        # The rank predicate separately binds these approved collector values.
        for name in ("FRONTIER_NATIVE_SERVING_EVIDENCE_DIR", "FRONTIER_NATIVE_RUN_GENERATION",
                     "FRONTIER_NATIVE_ROLE"):
            actual_env.pop(name)
    # The image's Python executable differs from this CPU checker interpreter.
    commands[node.parent.name] = result["server_command"][1:] == command[1:] and actual_env == env
print(json.dumps({"workload": workload[0], "rows": {k: len(v) for k, v in rows.items()}, "commands": commands}))
'''
        read = subprocess.run([sys.executable, "-I", "-B", "-c", code, harness, str(run_dir),
                               str(case_dir / manifest["trace_dir"])],
                              check=True, capture_output=True, text=True)
        parsed = json.loads(read.stdout)
        workload = parsed["workload"]
        return (bool(parsed["commands"]) and all(parsed["commands"].values()),
                f"Actual clean pinned canonical_workload/native_client_rows: {parsed['rows']}; "
                f"native server commands/environments: {parsed['commands']}")

    record("harness_pinned_and_client_rows", harness_readers)

    def replay():
        nonlocal summary, records
        summary = read_output(run_dir / "run/replay_summary.json")
        repetitions = read_output(run_dir / "run/canonical_repetitions.json")
        finished = read_output(run_dir / "replay_finished.json")
        records = read_jsonl(run_dir / "run/client_requests.jsonl")
        if (summary["canonical_repetitions"] != repetitions or summary["status"] != 0
                or summary.get("error") or summary["engine_config"] != engine
                or summary["num_requests"] != len(records) or summary["failed_request_ids"]
                or finished["returncode"] != 0 or repetitions["observation_errors"]):
            raise ValueError("Failed/inconsistent replay, repetition artifact, or client population")
        if workload is None:
            raise ValueError("Pinned raw reader/workload admission failed")
        dedicated = None
        if repetitions["idle_boundaries"][0].get("proof") == "dedicated_services_ready_client_zero_native_counters_zero":
            # Only the new proof asks the already pinned harness to rebind actual node receipts.
            code = '''import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
from tests.comparison.calibration.pd_replay import dedicated_services
print(json.dumps(dedicated_services(pathlib.Path(sys.argv[2]), json.loads(sys.argv[3]), int(sys.argv[4]))))
'''
            urls = {role: summary["instances"][role]["http_url"] for role in ("prefill", "decode")}
            read = subprocess.run([sys.executable, "-I", "-B", "-c", code, manifest["harness_worktree"],
                                   str(run_dir), json.dumps(urls), str(engine["data_parallel_size"])],
                                  check=True, capture_output=True, text=True)
            dedicated = json.loads(read.stdout)
        return native_idle_evidence(repetitions, records, workload, engine["data_parallel_size"], dedicated)

    record("requests_and_idle_lanes", replay)

    def workers():
        resources = manifest["resources"]
        replica, gpu = config["replica"], config["gpu"]
        dp, tp, local = engine["data_parallel_size"], engine["tensor_parallel_size"], config["data_parallel_size_local"]
        if (any(resources[name] != config[name] for name in ("replica", "gpu", "gpu_type", "image", "charged_group"))
                or resources["image"] != engine["native_reference"]["image"] or resources["gpus"] != replica * gpu
                or replica != 2 * dp // local or gpu != tp * local or engine["pipeline_parallel_size"] != 1):
            raise ValueError("Resource and native P/D TP/DP node partition mismatch")
        paths = sorted((run_dir / "nodes").iterdir())
        if len(paths) != replica:
            raise ValueError(f"Expected {replica} nodes, found {len(paths)}")
        peers = sorted([read_output(path / "ready.json") for path in paths], key=lambda p: p["host"])
        if (len({p["host"] for p in peers}) != replica or len({p["ip"] for p in peers}) != replica
                or {p["host"] for p in peers} != {p.name for p in paths}):
            raise ValueError("Node host/IP provenance is not distinct or ready markers disagree")
        for position, peer in enumerate(peers):
            host = peer["host"]
            node_dir = run_dir / "nodes" / host
            node = read_output(node_dir / "result.json")
            nodes[host] = node
            plan = node["plan"]
            role = ("prefill", "decode")[position // (dp // local)]
            lane = position % (dp // local) * local
            devices = plan["engine"]["role_devices"][role]
            if (node["host"] != host or node["stage"] != "finished" or node["workload_returncode"] != 0
                    or node.get("exception") or (node_dir / "failed.json").exists()
                    or node["observation_errors"] or node["weight_audit"]
                    or node["versions"]["vllm"] != engine["vllm_version"]
                    or node["native_source_hashes"] != config["expected_source_hashes"]
                    or node["model_hashes"] != config["model_hashes"]):
                raise ValueError(f"Native failure, observation error, weight files or runtime/source mismatch: {host}")
            devices_seen = [line for line in node["nvidia_smi"].splitlines() if line.startswith("GPU ")]
            if len(devices_seen) != gpu or any(resources["gpu_type"] not in line for line in devices_seen):
                raise ValueError(f"Actual GPU count/type mismatch: {host}")
            heads = {name: peers[i * (dp // local)] for i, name in enumerate(("prefill", "decode"))}
            urls = {name: f"http://{p['ip']}:{config['http_ports'][name]}" for name, p in heads.items()}
            expected_engine = engine | {"host": peer["ip"], "data_parallel_size_local": local,
                "data_parallel_address": heads[role]["ip"], "data_parallel_rpc_port": config["dp_rpc_ports"][role],
                "role_devices": {role: devices}, "headless": lane != 0,
                "kv_transfer": engine["kv_transfer"] | {"side_channel_host": peer["ip"]}}
            if role in config.get("role_num_gpu_blocks_override", {}):
                expected_engine["num_gpu_blocks_override"] = config["role_num_gpu_blocks_override"][role]
            if lane:
                expected_engine["data_parallel_start_rank"] = lane
            if (plan != dict(position=position, role=role, lane=lane, engine=expected_engine,
                              heads=heads, peers=peers, urls=urls)
                    or len(devices) != gpu or len(set(devices)) != gpu):
                raise ValueError(f"Native node plan/lane mismatch: {host}")
            if lane == 0:
                ready = read_output(node_dir / "server_ready.json")
                if ready != dict(role=role, url=urls[role]):
                    raise ValueError(f"Head readiness endpoint mismatch: {host}")
            elif (node_dir / "server_ready.json").exists():
                raise ValueError(f"Headless node unexpectedly claims API head readiness: {host}")
            if summary["instances"][role]["http_url"] != urls[role]:
                raise ValueError(f"Replay endpoint does not match native head: {host}")
        return True, f"{len(nodes)} distinct native nodes; both roles and every configured DP lane; source/model/GPU verified"

    record("native_workers", workers)

    def lifecycle():
        if len(nodes) != config["replica"]:
            raise ValueError("Incomplete native node evidence")
        for host, node in nodes.items():
            directory = run_dir / "nodes" / host
            plan, order, stop = node["plan"], node["shutdown_order"], node["server_stop"]
            drained = read_output(directory / "drained.json")
            stopped = read_output(directory / "native_stopped.json")
            if (drained != dict(role=plan["role"], lane=plan["lane"])
                    or stopped != stop | {"shutdown_order": order}
                    or stop["returncode"] != 0 or stop["signal"] not in (None, "SIGINT")
                    or order["drained_before_stop"] is not True
                    or type(order["native_returned_before_requested_stop"]) is not bool
                    or (directory / "drained.json").stat().st_mtime_ns > (directory / "native_stopped.json").stat().st_mtime_ns):
                raise ValueError(f"Native drain/stop failure or missing ordering evidence: {host}")
            if plan["lane"] == 0:
                if order["native_returned_before_requested_stop"]:
                    raise ValueError(f"Head returned before its requested stop: {host}")
                if sorted(order["all_nodes_drained_before_head_stop"]) != sorted(nodes):
                    raise ValueError(f"Head stopped before every node drained: {host}")
            else:
                if order["head_stopped_before_headless_stop"] is not True:
                    raise ValueError(f"Headless node stopped before its head: {host}")
                if order["native_returned_before_requested_stop"] and stop["signal"] is not None:
                    raise ValueError(f"Returned headless node unexpectedly required a stop signal: {host}")
        return True, "Every node drained, heads stopped after all drains, headless stopped after heads, native return codes 0"

    record("native_lifecycle", lifecycle)
    record("rank_runtime_graph", lambda: native_rank_graph_evidence(nodes, manifest, engine, run_dir, workload, records, config))

    def platform():
        receipt = read_output(run_dir / "platform_terminal_receipt.json")
        launcher = read_output(run_dir / "launcher_receipt.json")
        if (launcher["rjob_name"] != receipt["job"]
                or launcher["run_generation"] != manifest["run_generation"]
                or launcher["run_manifest_sha256"] != sha(manifest_path)
                or launcher["execution_config_sha256"] != manifest["inputs_sha256"][manifest["execution_config"]]):
            raise ValueError("Launcher receipt does not bind the platform job to this run generation, manifest and execution config")
        resources = manifest["resources"]
        replicas = receipt["replicas"]
        task = receipt["tasks"]["default"]
        finished = datetime.fromisoformat(receipt["status"]["finishTime"].replace("Z", "+00:00"))
        written = datetime.fromisoformat(manifest["written_before_launch_utc"].replace("Z", "+00:00"))
        if (not receipt["job"] or receipt["creator"] != "i-fengyicheng"
                or receipt["charged_group"] != resources["charged_group"]
                or receipt["status"]["phase"] != "Succeeded" or task["replicas"] != resources["replica"]
                or len(replicas) != resources["replica"]
                or {r["node"] for r in replicas} != set(nodes) or finished < written
                or not task["resources"]
                or any(int(r["limits"]["nvidia.com/gpu"]) != resources["gpu"] for r in task["resources"])):
            raise ValueError("Platform terminal status, creator or resources disagree")
        for replica in replicas:
            if replica["phase"] != "Succeeded" or not replica["containers"]:
                raise ValueError("Platform replica did not succeed")
            for container in replica["containers"]:
                if (container["state"]["terminated"]["exitCode"] != 0 or container["restartCount"] != 0
                        or container["imageID"] != resources["image"]):
                    raise ValueError("Platform native container failure/restart/image mismatch")
        return True, f"{receipt['job']}: Succeeded, all containers exit 0, no restarts; launcher-bound retained platform receipt"

    record("terminal_status", platform)
    return {
        "checked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "check_record": "written after the run; native pre-launch evidence is the run manifest "
                        f"(written {manifest.get('written_before_launch_utc', 'UNVERIFIED')})",
        "run_manifest": str(run_dir.relative_to(case_dir) / "run_manifest.json"),
        "rjob_name": (json.loads((run_dir / "platform_terminal_receipt.json").read_text()).get("job")
                      if checks["terminal_status"].startswith("PASS") else None),
        "authorization": manifest.get("authorization"), "checks": checks,
        "kv_cache_tokens_by_engine": {}, "derived_num_blocks": {},
        "status": "PASS" if all(v.startswith("PASS") for v in checks.values()) else "FAIL",
    }


def check(case_dir: Path, run_tag: str, mode: str) -> dict:
    case = yaml.safe_load((case_dir / "manifest.yaml").read_text())
    run_dir = case_dir / "runs" / f"groundtruth_{mode}" / run_tag
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    request_ids = json.loads((case_dir / manifest.get("trace_dir", "inputs/trace") / "request_ids.json").read_text())
    engine = json.loads((case_dir / manifest.get("engine_config", "inputs/engine.json")).read_text())
    workload_manifest = case_dir / manifest.get("trace_dir", "inputs/trace") / "manifest.json"
    if (engine.get("vllm_version") == "0.28.0" and workload_manifest.exists()
            and json.loads(workload_manifest.read_text()).get("schema") == "frontier.canonical_fixed_workload.v1"):
        return check_native_canonical(case_dir, run_dir, run_tag, mode, case, manifest, request_ids, engine)
    receipt = json.loads((run_dir / "launcher_receipt.json").read_text())
    summary = json.loads((run_dir / "run/replay_summary.json").read_text())
    worker_env = json.loads((run_dir / "worker_env.json").read_text())
    overlay = json.loads((run_dir / "overlay_report.json").read_text())
    selected = manifest["selected_checkout"]
    pd = case["topology"]["frontier"]["sys_arch"] == "pd-disaggregation"
    checks = {}

    checks["checkout"] = verdict(
        selected["groundtruth_commit"] == case["groundtruth_commit"]
        and selected["groundtruth_branch"] == case["groundtruth_branch"]
        and not selected["groundtruth_tree_dirty"],
        f"{selected['groundtruth_branch']} {selected['groundtruth_commit']}, "
        f"dirty={selected['groundtruth_tree_dirty']}")
    checks["overlay"] = verdict(
        overlay["accepted"] and not overlay["unexpected"] and not overlay["missing"]
        and overlay["patch"]["sha256"] == case["groundtruth_overlay_patch_sha256"],
        f"accepted={overlay['accepted']}, {len(overlay['differing_py_files'])} differing .py files, "
        f"patch {overlay['patch']['sha256'][:8]}")
    stale = [name for name, digest in manifest["inputs_sha256"].items()
             if sha(case_dir / name if not name.startswith("/") else Path(name)) != digest]
    checks["inputs_unchanged"] = verdict(not stale, f"changed since launch: {stale}")
    harness = selected["harness_commit"]
    harness_changed = [
        name for name, digest in manifest["harness_sha256"].items()
        if hashlib.sha256(subprocess.run(
            ["git", "-C", selected["harness_worktree"], "show", f"{harness}:{name}"],
            check=True, capture_output=True).stdout).hexdigest() != digest
    ]
    checks["harness_pinned"] = verdict(
        not selected["harness_tree_dirty_paths"] and not harness_changed,
        f"{selected['harness_branch']} {harness[:7]}, dirty paths "
        f"{selected['harness_tree_dirty_paths']}, files differing from the commit {harness_changed}")
    gpus = manifest["resources"]["gpus"]
    checks["worker_gpus"] = verdict(
        worker_env["device_count"] == gpus
        and all(manifest["resources"]["gpu_type"] in d for d in worker_env["devices"]),
        f"{worker_env['device_count']} x {sorted(set(worker_env['devices']))}, torch {worker_env['torch']}")
    status_line = (run_dir / "COMPLETE").read_text().strip()
    # The launcher exit code is the RJob's, shared by every run of the job; a failed later run
    # explains it for a run whose own worker status is 0. A run after a failed one never starts
    # and leaves no COMPLETE.
    other_runs = [(case_dir / "runs" / f"groundtruth_{run_mode}" / tag / "COMPLETE", tag)
                  for tag, run_mode, *_ in (run.split(":") for run in receipt["job_runs"]) if tag != run_tag]
    failed_job_runs = [tag for complete, tag in other_runs
                       if not complete.exists() or complete.read_text().strip() != "status=0"]
    checks["terminal_status"] = verdict(
        status_line == "status=0" and (receipt["launcher_exit_code"] == 0 or bool(failed_job_runs)),
        f"COMPLETE {status_line!r}, launcher exit {receipt['launcher_exit_code']}, "
        f"failed runs of the same job {failed_job_runs}, RJob {receipt['rjob_name']}")

    required_on, required_off = MODE_SWITCHES[mode]
    instance_envs = ({role: summary["instances"][role]["server_env_set"] for role in ("prefill", "decode")}
                     if pd else {"server": summary["server_env_set"]})
    switch_problems = {
        role: {"missing": sorted(required_on - set(env)), "unexpected": sorted(required_off & set(env)),
               "attention_backend": env.get("VLLM_ATTENTION_BACKEND"),
               "moe_uniform_routing": env.get("VLLM_MOE_UNIFORM_ROUTING")}
        for role, env in instance_envs.items()
    }
    moe_uniform_routing = "1" if engine.get("moe_uniform_routing") else None
    checks["mode_predicate"] = verdict(
        all(not p["missing"] and not p["unexpected"]
            and p["attention_backend"] == engine["attention_backend"]
            and p["moe_uniform_routing"] == moe_uniform_routing for p in switch_problems.values()),
        json.dumps(switch_problems))
    if engine.get("compilation_config", {}).get("cudagraph_mode") == "FULL_DECODE_ONLY":
        server_logs = ({role: run_dir / "run" / role / "server.log" for role in ("prefill", "decode")}
                       if pd else {"server": run_dir / "run/server.log"})
        graphs = {role: decode_graph_evidence(log, engine) for role, log in server_logs.items()}
        checks["decode_graphs"] = verdict(all(g["passed"] for g in graphs.values()), json.dumps(graphs))

    trace_ids = [row["request_id"] for row in request_ids["rows"]]
    warmups = request_ids["warmup_request_ids"]
    checks["requests_served"] = verdict(
        summary["num_requests"] == len(trace_ids) and not summary["failed_request_ids"],
        f"{summary['num_requests']} of {len(trace_ids)} requests, failed {summary['failed_request_ids'][:5]}")
    if pd:
        connector = {role: summary["instances"][role]["connector_check"] for role in ("prefill", "decode")}
        checks["connector"] = verdict(
            all(c["passed"] for c in connector.values()),
            json.dumps({role: {"init_lines": len(c["init_lines"]), "faults": c["faults"]}
                        for role, c in connector.items()}))
        kv_tokens = {role: summary["instances"][role]["kv_cache_tokens_by_engine"] for role in ("prefill", "decode")}
        # Both instances receive the same proxied engine request id (pd_replay.py).
        trace_id = {record["engine_request_id"]: record["request_id"]
                    for record in read_jsonl(run_dir / "run/client_requests.jsonl")}
    else:
        kv_tokens = {"server": summary["kv_cache_tokens_by_engine"]}

    def trace_row(row: dict) -> dict:
        engine_id = row["request_id"]
        return dict(row, request_id=trace_id[engine_id] if pd else engine_id.removeprefix("cmpl-").removesuffix("-0"))

    if pd and mode == "clean":
        join = json.loads((run_dir / "run/pd_join_report.json").read_text())
        rows = read_jsonl(run_dir / "run/pd_request_metrics.jsonl")
        checks["request_rows"] = verdict(
            join["complete"] and sorted(row["request_id"] for row in rows) == sorted(trace_ids),
            f"joined {join['joined']} of {len(trace_ids)}, complete={join['complete']}")
    elif mode in ("schedule_timing", "op_timing", "kernel_timing"):
        schedule_logs = ({role: run_dir / "run" / role / "schedule.jsonl" for role in ("prefill", "decode")}
                         if pd else {"server": run_dir / "run/schedule.jsonl"})
        role_rows, evidence, accounted = {}, {}, True
        for role, schedule_log in schedule_logs.items():
            parsed, evidence[role], passed = schedule_log_rows(schedule_log, remote_prefix=role == "decode")
            role_rows[role] = [trace_row(row) for row in parsed]
            accounted &= passed
        served = {role: sorted(row["request_id"] for row in values) for role, values in role_rows.items()}
        checks["request_rows"] = verdict(
            all(ids == sorted(trace_ids) for ids in served.values()),
            json.dumps({role: f"{len(ids)} completed requests in schedule.jsonl for {len(trace_ids)} trace requests"
                        for role, ids in served.items()}))
        checks["schedule_log"] = verdict(accounted, json.dumps(evidence))
        if pd:
            # The proxy asks the prefill instance for exactly one output token.
            extra = sorted(row["request_id"] for row in role_rows["prefill"] if row["request_num_decode_tokens"] != 1)
            checks["prefill_output_tokens"] = verdict(not extra, f"prefill requests with other than one output token: {extra[:5]}")
        rows = role_rows["decode" if pd else "server"]
    elif mode in ("instrumented", "cpu", "device_timeline", *KV_SAVE_MODES):
        # Request metrics are off; the replay client records each request's status and output length.
        rows = [{"request_id": record["request_id"], "request_num_decode_tokens": record["usage"]["completion_tokens"]}
                for record in read_jsonl(run_dir / "run/client_requests.jsonl") if record["http_status"] == 200]
        checks["request_rows"] = verdict(
            sorted(row["request_id"] for row in rows) == sorted(trace_ids),
            f"{len(rows)} successful client records for {len(trace_ids)} trace requests")
        if mode != "instrumented":
            dp_size, pp_size = engine["data_parallel_size"], engine["pipeline_parallel_size"]
            role_dirs = ({role: run_dir / "run" / role for role in ("prefill", "decode")}
                         if pd else {"server": run_dir / "run"})
            evidence, complete = {}, True
            for role, role_dir in role_dirs.items():
                engines_seen = []
                for dp_rank in range(dp_size):
                    engine_log = cpu_probe_log(role_dir, dp_rank, dp_size)
                    seen, evidence[f"{role}/{engine_log.name}"], passed = cpu_probe_evidence(engine_log)
                    engines_seen.append({trace_row({"request_id": i})["request_id"] for i in seen})
                    complete &= passed
                    stage_logs = [cpu_probe_log(role_dir, dp_rank, dp_size, stage) for stage in range(pp_size)]
                    evidence[f"{role}/{engine_log.name} stages"], passed = cpu_probe_stage_evidence(
                        engine_log, stage_logs,
                        engine.get("compilation_config", {}).get("cudagraph_mode") == "FULL_DECODE_ONLY"
                        and role != "prefill",
                        dp_size > 1)
                    complete &= passed
                uncovered = sorted(set(trace_ids) - set().union(*engines_seen))
                engines_per_request = Counter(request for seen in engines_seen for request in seen)
                shared = sorted(request for request, count in engines_per_request.items() if count > 1)
                evidence[role] = f"trace requests absent {uncovered[:5]}, on more than one DP engine {shared[:5]}"
                complete &= not uncovered and not shared
            checks["cpu_probe"] = verdict(complete, json.dumps(evidence))
        if mode in KV_SAVE_MODES:
            prefill_evidence, prefill_passed = kv_save_probe_evidence(
                run_dir / "run/prefill/kv_save_probe.jsonl", cpu_probe_log(role_dirs["prefill"], 0, 1, 0))
            decode_records = read_jsonl(run_dir / "run/decode/kv_save_probe.jsonl")
            checks["kv_save_probe"] = verdict(
                prefill_passed and not decode_records,
                json.dumps({"prefill": prefill_evidence, "decode": f"{len(decode_records)} records"}))
        if mode == "device_timeline":
            timeline = json.loads((run_dir / "run/device_timeline.json").read_text())
            marks = summary["profile_marks"]
            checks["device_timeline"] = verdict(
                timeline["distributed_info"]["rank"] == 0
                and timeline["distributed_info"]["world_size"] == gpus
                and bool(timeline["forwards"]) and set(marks) == {"start_profile", "stop_profile"},
                f"{timeline['trace_file']} ({timeline['distributed_info']}): {len(timeline['forwards'])} eager "
                f"and {len(timeline['graph_forwards'])} graph forwards, "
                f"{sum(record['all_reduce_min_us'] is not None for kind in ('forwards', 'graph_forwards') for record in timeline[kind])} "
                f"with an all-reduce minimum over every rank; profiler calls {sorted(marks)}")
    else:
        rows = [trace_row(row) for row in read_jsonl(run_dir / "run/request_metrics.jsonl")]
        checks["request_rows"] = verdict(
            sorted(row["request_id"] for row in rows) == sorted(trace_ids),
            f"{len(rows)} rows for {len(trace_ids)} trace requests")
    row_ids = [row["request_id"] for row in rows]
    checks["warmups"] = verdict(
        len(warmups) >= 3 and set(warmups) <= set(row_ids),
        f"{len(warmups)} warmups served before the formal rows")
    decode_tokens = {row["request_id"]: row["request_num_decode_tokens"] for row in rows}
    expected = {row["request_id"]: row["num_decode_tokens"] for row in request_ids["rows"]}
    short = sorted(i for i in expected if decode_tokens.get(i) != expected[i])
    checks["output_lengths"] = verdict(not short, f"requests whose output length differs from the trace: {short[:5]}")

    num_blocks = {
        role: min(value for values in engines.values() for value in values) // engine["block_size"]
        for role, engines in kv_tokens.items() if engines
    }
    checks["kv_cache_logged"] = verdict(len(num_blocks) == len(kv_tokens), json.dumps(kv_tokens))
    return {
        "checked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "check_record": "written after the run; the pre-launch evidence is the run manifest "
                        f"(written {manifest['written_before_launch_utc']})",
        "run_manifest": str(run_dir.relative_to(case_dir) / "run_manifest.json"),
        "rjob_name": receipt["rjob_name"],
        "authorization": manifest["authorization"],
        "checks": checks,
        "kv_cache_tokens_by_engine": kv_tokens,
        "derived_num_blocks": num_blocks,
        "status": "PASS" if all(v.startswith("PASS") for v in checks.values()) else "FAIL",
    }


def main() -> int:
    case_dir, run_tag, mode = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3]
    result = check(case_dir, run_tag, mode)
    status_path = case_dir / "analysis" / f"run_check_status.groundtruth_{mode}.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {
        "entry": "groundtruth-run-check",
        "case_id": yaml.safe_load((case_dir / "manifest.yaml").read_text())["case_id"],
        "mode": f"groundtruth_{mode}",
        "runs": {},
    }
    status["runs"][run_tag] = result
    status_path.parent.mkdir(exist_ok=True)
    status_path.write_text(json.dumps(status, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"run_tag": run_tag, "status": result["status"],
                      "failed": [k for k, v in result["checks"].items() if v.startswith("FAIL")],
                      "derived_num_blocks": result["derived_num_blocks"]}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
