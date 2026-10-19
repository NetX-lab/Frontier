"""Measure the collectives of a vLLM v0.28 forward pass on serving process groups.

Run one process per GPU under ``torchrun`` with ``WORLD_SIZE`` equal to
TP x DP. Each process initializes vLLM's distributed state the way a vLLM
worker does (``init_worker_distributed_environment`` with the external-launcher
backend) and calls the vLLM ``GroupCoordinator`` APIs, so vLLM itself selects
the communicator: custom all-reduce, symmetric memory, PyNccl, or grouped
broadcast/reduce for uneven ``all_gatherv``/``reduce_scatterv`` sizes.

Each repeat runs ``calls_per_repeat`` back-to-back calls between CUDA events
after a CPU barrier; the per-call time of a repeat is the maximum over ranks,
which is the time the slowest participant spends in the collective.
Rank 0 appends one JSON line per (scenario, collective) to
``<output_dir>/measurements.jsonl`` and writes ``summary.csv``,
``forward_totals.csv`` and, for TP all-reduce, ``all_reduce.csv`` in the
network-profile format read by the Vidur CC backend.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from frontier.profiling.collectives.vllm_forward_collectives import (
    CollectiveCall,
    ModelShape,
    ServingParallelism,
    call_bytes,
    forward_collectives,
    scenarios_from_grid,
)

RUN_SETTINGS = ("warmup_calls", "repeats", "calls_per_repeat")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def summarize(record: dict) -> dict:
    """Reduce per-rank repeat timings of one measurement to critical-path stats."""
    per_rank = [r for r in record["per_rank_call_us"] if r is not None]
    critical = [max(values) for values in zip(*per_rank)]
    return {
        "scenario": record["scenario"], "call": record["call"], "op": record["op"],
        "group": record["group"], "group_size": record["group_size"],
        "cuda_graph": record["cuda_graph"], "calls_per_forward": record["calls_per_forward"],
        "tokens_per_dp_rank": "/".join(map(str, record["tokens_per_dp_rank"])),
        "max_input_bytes": record["max_input_bytes"],
        "max_output_bytes": record["max_output_bytes"],
        "median_us": statistics.median(critical), "mean_us": statistics.fmean(critical),
        "std_us": statistics.pstdev(critical), "min_us": min(critical),
        "max_us": max(critical), "p10_us": _percentile(critical, 0.1),
        "p90_us": _percentile(critical, 0.9),
        "kernels": ";".join(record["kernels"]),
    }


def forward_totals(summary_rows: list[dict]) -> list[dict]:
    """Per-scenario communication time of one forward pass (median-based)."""
    totals: dict[str, dict] = {}
    for row in summary_rows:
        total = totals.setdefault(row["scenario"], {
            "scenario": row["scenario"], "cuda_graph": row["cuda_graph"],
            "tokens_per_dp_rank": row["tokens_per_dp_rank"], "forward_comm_us": 0.0})
        total["forward_comm_us"] += row["calls_per_forward"] * row["median_us"]
        total[f"{row['call']}_us"] = row["calls_per_forward"] * row["median_us"]
    return list(totals.values())


def network_all_reduce_rows(summary_rows: list[dict], gpus_per_node: int) -> list[dict]:
    """TP all-reduce rows in the ``data/profiling/network/<device>/all_reduce.csv`` schema.

    ``size`` is in bytes and times are in milliseconds, as the Vidur CC backend
    reads them. Only eager TP groups within one node qualify, because the
    backend filters ``devices_per_node == num_workers``. Every measurement is
    kept; the backend fits a regression over size.
    """
    rows = [{
        "time_stats.all_reduce.min": row["min_us"] / 1e3,
        "time_stats.all_reduce.max": row["max_us"] / 1e3,
        "time_stats.all_reduce.mean": row["mean_us"] / 1e3,
        "time_stats.all_reduce.median": row["median_us"] / 1e3,
        "time_stats.all_reduce.std": row["std_us"] / 1e3,
        "rank": 0, "num_workers": row["group_size"], "size": row["max_input_bytes"],
        "collective": "all_reduce", "devices_per_node": row["group_size"],
        "max_devices_per_node": gpus_per_node,
    } for row in summary_rows
        if row["op"] == "all_reduce" and row["group"] == "tp" and not row["cuda_graph"]
        and row["group_size"] <= gpus_per_node]
    return sorted(rows, key=lambda row: row["size"])


class _Runner:
    def __init__(self, parallel: ServingParallelism, settings: dict):
        import torch
        from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
        from vllm.distributed import parallel_state
        from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

        self.torch = torch
        self.parallel = parallel
        self.settings = settings
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        if int(os.environ["WORLD_SIZE"]) != parallel.world_size:
            raise ValueError("torchrun WORLD_SIZE must equal TP x DP")
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.device)
        self.vllm_config = VllmConfig(parallel_config=ParallelConfig(
            tensor_parallel_size=parallel.tensor_parallel_size,
            data_parallel_size=parallel.data_parallel_size,
            pipeline_parallel_size=parallel.pipeline_parallel_size,
            enable_expert_parallel=parallel.enable_expert_parallel,
            all2all_backend=parallel.all2all_backend,
            distributed_executor_backend="external_launcher"))
        self._config_context = set_current_vllm_config(self.vllm_config)
        self._config_context.__enter__()
        init_worker_distributed_environment(self.vllm_config, self.rank, "env://",
                                            self.local_rank)
        self.ps = parallel_state
        self.groups = {"tp": parallel_state.get_tp_group(),
                       "ep": parallel_state.get_ep_group(),
                       "dp": parallel_state.get_dp_group()}
        self.world_cpu = parallel_state.get_world_group().cpu_group

    def communicators(self) -> dict:
        """Which all-reduce communicators vLLM enabled on each group of this rank."""
        import vllm

        enabled = {}
        for name, group in self.groups.items():
            comm = group.device_communicator
            enabled[name] = {
                attr: getattr(comm, attr) is not None
                and not getattr(getattr(comm, attr), "disabled", False)
                for attr in ("pynccl_comm", "ca_comm", "symm_mem_comm", "fi_ar_comm",
                             "qr_comm", "all2all_manager")}
        return {"torch": self.torch.__version__, "vllm": vllm.__version__,
                "communicators": enabled}

    def _tensors(self, call: CollectiveCall, rows: int, device):
        torch = self.torch
        return [torch.ones((rows, columns), dtype=getattr(torch, dtype), device=device)
                for columns, dtype in zip(call.columns, call.dtypes)]

    def _issue(self, call: CollectiveCall):
        """Return a zero-argument function that issues ``call`` once on this rank."""
        torch = self.torch
        group = self.groups[call.group]
        rows = call.shard_rows[self.rank]
        members = group.ranks
        sizes = [call.shard_rows[m] for m in members]
        if call.op == "cpu_all_reduce":
            (tensor,) = self._tensors(call, rows, "cpu")
            return lambda: torch.distributed.all_reduce(tensor, group=group.cpu_group)
        if call.op == "device_group_all_reduce":
            (tensor,) = self._tensors(call, rows, self.device)
            return lambda: torch.distributed.all_reduce(tensor, group=group.device_group)
        if call.op == "all_reduce":
            (tensor,) = self._tensors(call, rows, self.device)
            return lambda: group.all_reduce(tensor)
        if call.op == "all_gather":
            (tensor,) = self._tensors(call, rows, self.device)
            return lambda: group.all_gather(tensor, dim=call.dim)
        if call.op == "reduce_scatter":
            (tensor,) = self._tensors(call, rows * len(members), self.device)
            return lambda: group.reduce_scatter(tensor, dim=call.dim)
        if call.op == "all_gatherv":
            tensors = self._tensors(call, rows, self.device)
            return lambda: group.all_gatherv(tensors, dim=call.dim, sizes=sizes)
        if call.op == "reduce_scatterv":
            (tensor,) = self._tensors(call, sum(sizes), self.device)
            return lambda: group.reduce_scatterv(tensor, dim=call.dim, sizes=sizes)
        raise ValueError(f"Unknown op {call.op!r}")

    def _kernels(self, issue) -> list[str]:
        torch = self.torch
        activities = [torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities) as profile:
            issue()
            torch.cuda.synchronize()
        return sorted({event.name for event in profile.events()
                       if event.device_type == torch.autograd.DeviceType.CUDA})

    def _time(self, issue, barrier_group, on_cpu: bool, cuda_graph: bool) -> list[float]:
        """Per-call microseconds of each repeat; ranks align on ``barrier_group``."""
        torch = self.torch
        repeats, per_repeat = self.settings["repeats"], self.settings["calls_per_repeat"]
        for _ in range(self.settings["warmup_calls"]):
            issue()
        torch.cuda.synchronize()
        replay = issue
        if cuda_graph:
            graph = torch.cuda.CUDAGraph()
            with self.ps.graph_capture(self.device) as context:
                with torch.cuda.graph(graph, stream=context.stream):
                    for _ in range(per_repeat):
                        issue()
            torch.cuda.synchronize()
            replay = graph.replay
        timings = []
        for _ in range(repeats):
            torch.distributed.barrier(group=barrier_group)
            if on_cpu:
                start = time.perf_counter()
                for _ in range(per_repeat):
                    issue()
                timings.append((time.perf_counter() - start) * 1e6 / per_repeat)
                continue
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            if cuda_graph:
                replay()
            else:
                for _ in range(per_repeat):
                    issue()
            end.record()
            end.synchronize()
            timings.append(begin.elapsed_time(end) * 1e3 / per_repeat)
        return timings

    def measure(self, scenario, call: CollectiveCall) -> dict | None:
        """Measure one call; return the merged record on rank 0."""
        torch = self.torch
        torch.distributed.barrier(group=self.world_cpu)
        active = call.shard_rows[self.rank] > 0
        local = None
        if active:
            issue = self._issue(call)
            on_cpu = call.op == "cpu_all_reduce"
            # Only the call's participants time it; ranks with no rows (logits
            # on idle DP ranks) skip to the world gather below.
            barrier_group = self.groups[call.group].cpu_group
            local = {"timings": self._time(issue, barrier_group, on_cpu, call.captured),
                     "kernels": [] if on_cpu else self._kernels(issue),
                     "bytes": call_bytes(call, self.parallel, self.rank)}
        gathered = [None] * self.parallel.world_size
        torch.distributed.all_gather_object(gathered, local, group=self.world_cpu)
        if self.rank != 0:
            return None
        present = [g for g in gathered if g is not None]
        return {
            "scenario": scenario.name, "call": call.name, "op": call.op,
            "group": call.group, "group_size": self.groups[call.group].world_size,
            "cuda_graph": call.captured,
            "calls_per_forward": call.calls_per_forward,
            "tokens_per_dp_rank": list(scenario.tokens_per_dp_rank),
            "shard_rows": list(call.shard_rows), "source": call.source,
            "max_input_bytes": max(g["bytes"][0] for g in present),
            "max_output_bytes": max(g["bytes"][1] for g in present),
            "kernels": sorted({k for g in present for k in g["kernels"]}),
            "per_rank_call_us": [g["timings"] if g else None for g in gathered],
        }


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--list-cases", action="store_true",
                        help="Print the planned cases without initializing CUDA")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    parallel = ServingParallelism.from_engine(config["engine"])
    model = ModelShape.from_hf_config(json.loads(Path(config["model_config"]).read_text()))
    scenarios = scenarios_from_grid(parallel, config["grid"])
    plan = [(s, c) for s in scenarios for c in forward_collectives(model, parallel, s)]
    if args.list_cases:
        for scenario, call in plan:
            print(scenario.name, call.name, call.op, call.group,
                  call_bytes(call, parallel, 0), call.calls_per_forward)
        print("cases", len(plan))
        return
    settings = {key: int(config[key]) for key in RUN_SETTINGS}
    runner = _Runner(parallel, settings)
    output = Path(config["output_dir"])
    summary = []
    log = None
    if runner.rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "plan.json").write_text(json.dumps(
            {"config": config, "cases": len(plan), "runtime": runner.communicators()},
            indent=2))
        log = (output / "measurements.jsonl").open("a")
    for scenario, call in plan:
        record = runner.measure(scenario, call)
        if record is None:
            continue
        log.write(json.dumps(record) + "\n")
        log.flush()
        summary.append(summarize(record))
    if runner.rank == 0:
        log.close()
        _write_csv(output / "summary.csv", summary)
        _write_csv(output / "forward_totals.csv", forward_totals(summary))
        _write_csv(output / "all_reduce.csv",
                   network_all_reduce_rows(summary, int(config["gpus_per_node"])))
    runner.torch.distributed.barrier(group=runner.world_cpu)


if __name__ == "__main__":
    main()
