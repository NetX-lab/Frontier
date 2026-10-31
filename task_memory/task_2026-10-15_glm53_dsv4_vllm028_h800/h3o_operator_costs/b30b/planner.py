"""CPU dry run of both B30b jobs; held-out timings are used only for budgets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from frontier.profiling.collectives.cuda_launch_depth import probe_cases
from frontier.profiling.collectives.nixl_kv_read_benchmark import KVRegionLayout, read_cases
from frontier.profiling.collectives.vllm_collective_benchmark import plan_from_config
from frontier.profiling.collectives.vllm_forward_collectives import ServingParallelism, call_bytes, group_members

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
COLLECTIVE = "frontier.profiling.collectives.vllm_collective_benchmark"
NIXL = "frontier.profiling.collectives.nixl_kv_read_benchmark"
DEPTH = "frontier.profiling.collectives.cuda_launch_depth"
DEFAULT_G5H = [Path(f"/data/ycfeng/tmp/glm53_dsv4/h200/{run}/download/summary.csv")
               for run in ("g5h16_02", "g5h32_01")]
DEFAULT_K5 = Path("/data/ycfeng/tmp/glm53_dsv4/w5_comm_k5/nixl_summary.csv")


def establish_layout(config, repo=REPO):
    evidence = json.loads((repo / config["kv_layout_evidence"]).read_text())
    for key in ("native_source", "logical_groups_source"):
        source = (Path(evidence[key]) if key == "native_source" else
                  repo / evidence[key].split(":")[0])
        if hashlib.sha256(source.read_bytes()).hexdigest() != evidence[key + "_sha256"]:
            raise ValueError(f"Native layout source changed: {source}")
    signatures = []
    for item in evidence["logs"]:
        lines = Path(item["path"]).read_text().splitlines()
        if item["text"] not in lines[item["line"] - 1]:
            raise ValueError("Native layout evidence line changed")
        match = re.search(r"total_size=(\d+), block_stride=(\d+), num_blocks=(\d+), num_regions=(\d+)",
                          item["text"])
        if not match:
            raise ValueError("Cannot establish native packed KV region layout")
        total, stride, blocks, regions = map(int, match.groups())
        if total != stride * blocks or regions != 1:
            raise ValueError("Native packed layout is inconsistent")
        signatures.append((regions, stride))
    if not signatures or len(set(signatures)) != 1:
        raise ValueError("Native layout is missing or disagrees between nodes")
    phase = next(p for p in config["phases"] if p["benchmark_module"] == NIXL)
    layout = KVRegionLayout.from_config(phase["config"]["kv_cache"])
    if (layout.num_regions, layout.block_bytes) != signatures[0]:
        raise ValueError("Planned KV regions differ from native layout")
    # Bind the retained source contract, rather than accepting an arbitrary GLM group list.
    expected = ((256, None), (64, 128), (64, 128), (4, 8), (8, 128))
    if layout.logical_groups != expected:
        raise ValueError("V4 logical transfer groups differ from retained native contract")
    return layout


def phase_cases(phase):
    config = phase["config"]
    module = phase["benchmark_module"]
    if module == COLLECTIVE:
        parallel = ServingParallelism.from_engine(config["engine"])
        plan = plan_from_config(json.loads(Path(config["model_config"]).read_text()), parallel, config["grid"],
                                allow_link_probe_parallelism=config.get("allow_link_probe_parallelism", False))
        return [dict(scenario=s.name, call=c.name, op=c.op, group=c.group,
                     group_size=len(group_members(parallel, c.group, 0)),
                     cuda_graph=c.captured,
                     max_input_bytes=max(call_bytes(c, parallel, r)[0] for r in range(parallel.world_size)),
                     calls_per_forward=c.calls_per_forward)
                for s, c in plan]
    if module == NIXL:
        layout = KVRegionLayout.from_config(config["kv_cache"])
        return [dict(case=c.name, prompt_tokens=c.prompt_tokens,
                     blocks_per_request=c.blocks_per_request, active_pairs=c.active_pairs,
                     requests_in_flight=c.requests_in_flight,
                     bytes_per_reader=c.blocks_per_request * c.requests_in_flight * layout.block_bytes)
                for c in read_cases(config["grid"], layout, config["tensor_parallel_size"])]
    if module == DEPTH:
        return probe_cases(config)
    raise ValueError(f"Unknown benchmark {module}")


def read_timings(paths):
    return [dict(r, source=str(p)) for p in paths for r in csv.DictReader(p.open(newline=""))]


def collective_us(case, rows):
    selected = [r for r in rows if r["op"] == case["op"] and
                str(r["cuda_graph"]).lower() == str(case["cuda_graph"]).lower()]
    if not selected:
        raise ValueError(f"No measured timing proxy for {case['op']}")
    # Conservative timing envelope from the measured curve, scaled only beyond
    # its largest measured payload. This is a budget proxy, never a fit sample.
    maximum_size = max(int(r["max_input_bytes"]) for r in selected)
    return max(float(r["median_us"]) for r in selected) * max(1, case["max_input_bytes"] / maximum_size)


def estimate_phase(phase, cases, g5h, k5, estimates):
    config = phase["config"]
    module = phase["benchmark_module"]
    if module == COLLECTIVE:
        # One launch-inclusive repeat plus one queued repeat; the latter sleeps
        # for 2x the launch repeat. Native kernel evidence adds one untimed call.
        multiplier = config["passes"] * (config["warmup_calls"] + 4 * config["repeats"]) * config["calls_per_repeat"] + 1
        service = sum(collective_us(c, g5h) for c in cases) * multiplier / 1e6
        overhead = len(cases) * config["passes"] * estimates["collective_case_overhead_seconds"]
        arithmetic = f"sum(envelope_us) * {multiplier}/1e6 + {len(cases)}*{config['passes']}*{estimates['collective_case_overhead_seconds']} + process startup"
    elif module == NIXL:
        per_byte = max(float(r["complete_median_us"]) / int(r["bytes_per_reader"]) for r in k5)
        service = sum(c["bytes_per_reader"] * per_byte for c in cases) * config["passes"] * (config["warmup"] + config["repeats"]) / 1e6
        overhead = len(cases) * config["passes"] * estimates["nixl_case_overhead_seconds"]
        arithmetic = f"sum(reader_bytes)*max(K5_us/reader_bytes) * {config['passes']}*{config['warmup']+config['repeats']}/1e6 + {len(cases)}*{config['passes']}*{estimates['nixl_case_overhead_seconds']} + process startup"
    else:
        # A small collective's measured device time is a conservative service
        # proxy for one synthetic add kernel; actual F2 service time is unknown.
        kernel_us = min(float(r["median_us"]) for r in g5h)
        enqueued = sum(min(config["max_launches"], config["max_enqueued_kernels"] // c["kernels_per_graph"]) * c["kernels_per_graph"] for c in cases)
        service = sum(c["sleep_ms"] for c in cases) / 1000 + enqueued * kernel_us / 1e6
        overhead = sum(config["kernels_per_graph"]) * estimates["capture_seconds_per_kernel"]
        arithmetic = f"sleep sum + worst-case {enqueued} enqueued kernels * G5h min call {kernel_us:.6g}us/1e6 + capture + process startup"
    startup = estimates["process_startup_seconds"]
    return dict(service_seconds=service, overhead_seconds=overhead, process_startup_seconds=startup,
                wall_seconds=service + overhead + startup, arithmetic=arithmetic)


def dry_run(config, g5h, k5, repo=REPO):
    if (config["gpu_type"], config["charged_group"]) != ("H200", "step_main"):
        raise ValueError("B30b requires H200 / step_main")
    if config["runtime_check_gate"] != "soft":
        raise ValueError("B30b runtime hash checks must preserve completed measurements")
    if any(p["benchmark_module"] == NIXL for p in config["phases"]):
        establish_layout(config, repo)
    phases = []
    for phase in config["phases"]:
        if len(phase["gpu_indices"]) != config["replica"]:
            raise ValueError("Phase placement must specify every node")
        for indices in phase["gpu_indices"]:
            if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= config["gpu"]:
                raise ValueError("Invalid phase GPU placement")
        if not phase.get("independent_nodes") and len({len(i) for i in phase["gpu_indices"]}) != 1:
            raise ValueError("Distributed torchrun needs equal ranks per node")
        cases = phase_cases(phase)
        copies = config["replica"] if phase.get("independent_nodes") else 1
        estimate = estimate_phase(phase, cases, g5h, k5, config["estimate"])
        phases.append(dict(name=phase["name"], copies=copies, cases_per_copy=len(cases),
                           total_cases=len(cases) * copies, cases=cases, estimate=estimate,
                           gpu_indices=phase["gpu_indices"], output_dir=f"{config['output_dir']}/{phase['name']}"))
    unpadded = config["estimate"]["startup_seconds"] + sum(p["estimate"]["wall_seconds"] for p in phases)
    wall = unpadded * config["estimate"]["safety_factor"]
    allocated = config["gpu"] * config["replica"]
    return dict(stage=config["stage"], total_cases=sum(p["total_cases"] for p in phases),
                phases=phases, startup_seconds=config["estimate"]["startup_seconds"],
                unpadded_wall_seconds=unpadded, safety_factor=config["estimate"]["safety_factor"],
                wall_seconds=wall, allocated_gpus=allocated, gpu_hours=wall * allocated / 3600,
                timeout_seconds=config["timeout_seconds"],
                timeout_gpu_hours=config["timeout_seconds"] * allocated / 3600,
                arithmetic=f"({config['estimate']['startup_seconds']} + sum(phase wall seconds))*{config['estimate']['safety_factor']}; GPUh=wall*{allocated}/3600",
                timings_use="G5h and K5 are wall-time proxies only; excluded from F1b/F3 parameter fits",
                timing_sources=[dict(path=p, sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
                                for p in sorted({r['source'] for r in g5h + k5})])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=Path)
    parser.add_argument("--g5h", action="append", type=Path)
    parser.add_argument("--k5", type=Path, default=DEFAULT_K5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    configs = args.config or [ROOT / name for name in ("stage_b30b_links_kv.json", "stage_b30b_launch_depth.json")]
    g5h, k5 = read_timings(args.g5h or DEFAULT_G5H), read_timings([args.k5])
    args.output.mkdir(parents=True, exist_ok=True)
    for path in configs:
        config = json.loads(path.read_text())
        report = dry_run(config, g5h, k5)
        lines = []
        for phase in report["phases"]:
            lines.append(f"PHASE {phase['name']} copies={phase['copies']} cases/copy={phase['cases_per_copy']} placements={phase['gpu_indices']}")
            for case in phase["cases"]:
                line = json.dumps(case, sort_keys=True)
                lines.append(line)
                print(phase["name"], line)
            lines.append(json.dumps(phase["estimate"], sort_keys=True))
        compact = {k:v for k,v in report.items() if k != "phases"}
        lines.append(json.dumps(compact, indent=2))
        (args.output / f"{config['stage']}.txt").write_text("\n".join(lines) + "\n")
        (args.output / f"{config['stage']}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
