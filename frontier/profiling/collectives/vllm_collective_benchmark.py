"""Measure the collectives of a vLLM v0.28 forward pass on serving process groups.

Run one process per GPU under ``torchrun`` with ``WORLD_SIZE`` equal to
TP x DP. Each process initializes vLLM's distributed state the way a vLLM
worker does (``init_worker_distributed_environment`` with the external-launcher
backend) and calls the vLLM ``GroupCoordinator`` APIs, so vLLM itself selects
the communicator: FlashInfer, custom all-reduce, symmetric memory, PyNccl, or grouped
broadcast/reduce for uneven ``all_gatherv``/``reduce_scatterv`` sizes.

Each repeat runs ``calls_per_repeat`` back-to-back calls between CUDA events
after a CPU barrier on the call's group; the per-call time of a repeat is the
maximum over ranks, which is the time the slowest participant spends in the
collective. The primary time queues the launches behind a GPU sleep that starts
on every rank when a one-element all-reduce on the group's device group
completes, so it measures device work from aligned device start times; a
second, launch-inclusive time is kept alongside (``CollectiveRunner.time_call``).
Kernel names are recorded in a separate pass after all timings, so the profiler
is never active before a timed repeat.

The plan runs ``passes`` times, each pass in an order shuffled with
``shuffle_seed + pass`` and identical on every rank. Rank 0 appends one JSON
line per (pass, scenario, collective) to ``<output_dir>/measurements.jsonl``,
one line of kernel names per case to ``kernels.jsonl``, and writes
``summary.csv`` and ``forward_totals.csv`` with one row per pass and, for TP
all-reduce, ``all_reduce.csv`` in the network-profile format read by the Vidur
CC backend.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import statistics
import time
from dataclasses import replace
from pathlib import Path

from frontier.profiling.collectives.vllm_forward_collectives import (
    CollectiveCall,
    ModelShape,
    ServingParallelism,
    call_bytes,
    forward_collectives,
    group_members,
    scenarios_from_grid,
)

RUN_SETTINGS = ("warmup_calls", "repeats", "calls_per_repeat", "passes", "shuffle_seed")
# The queued timing budgets this multiple of the measured CPU enqueue window.
# The begin-event query verifies the ordering for every measured repeat.
QUEUE_SLEEP_MARGIN = 2.0
# Length of the GPU sleep that converts microseconds to torch.cuda._sleep cycles.
SLEEP_CALIBRATION_CYCLES = 10_000_000
# Kernel namespaces from FlashInfer's trtllm_allreduce headers, torch's
# symmetric-memory kernels, vLLM custom all-reduce, and NCCL respectively.
COMMUNICATOR_KERNEL_MARKERS = {
    "flashinfer_trtllm": ("flashinfer::trtllm_allreduce",),
    "symmetric_memory": ("multimem_all_reduce_kernel", "two_shot_all_reduce_kernel"),
    "custom_all_reduce": ("vllm::cross_device_reduce_",),
    "nccl": ("nccldevkernel", "ncclkernel"),
}
VLLM_ENVIRONMENT = (
    "VLLM_ALLREDUCE_USE_FLASHINFER", "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC",
    "VLLM_ALLREDUCE_USE_SYMM_MEM", "VLLM_USE_NCCL_SYMM_MEM",
    "VLLM_FLASHINFER_ALLREDUCE_BACKEND", "VLLM_BATCH_INVARIANT", "VLLM_USE_V2_MODEL_RUNNER",
)


def actual_communicator(op: str, kernels: list[str]) -> str:
    """Classify observed kernels, never the enabled backends or payload size.

    Uneven TP lanes may use different communicators in the same case, so keep
    every observed communicator. Copy kernels do not identify a communicator.
    """
    if op == "cpu_all_reduce":
        return "host_gloo"
    names = [name.lower() for name in kernels]
    observed = [comm for comm, markers in COMMUNICATOR_KERNEL_MARKERS.items()
                if any(marker in name for marker in markers for name in names)]
    return ";".join(observed) if observed else "unknown"


def runtime_environment(vllm_envs) -> dict:
    """Record image-resolved defaults and overrides without changing any env var."""
    return {
        "vllm_environment": {
            name: {"value": getattr(vllm_envs, name),
                   "environment_override": os.environ.get(name),
                   "source": "environment" if name in os.environ else "image_default"}
            for name in VLLM_ENVIRONMENT if name in vllm_envs.environment_variables
        },
        "nccl_environment": {name: value for name, value in sorted(os.environ.items())
                             if name.startswith("NCCL_")},
        "nccl_log_destination": os.environ.get("NCCL_DEBUG_FILE", "stderr (stage wrapper log)"),
    }


def active_dp_rails(parallel: ServingParallelism, call: CollectiveCall,
                    scenario=None, isolated: bool = False) -> tuple[int, ...]:
    """Active DP groups: native device rails, or the one host-sync rail."""
    if call.op == "cpu_all_reduce" or isolated:
        return (0,)
    rails = getattr(scenario, "active_dp_rails", None)
    return tuple(range(parallel.tensor_parallel_size)) if rails is None else rails


def pass_order(items: list, shuffle_seed: int, pass_index: int) -> list:
    """``items`` in the order of one pass; every rank derives the same order."""
    order = list(items)
    random.Random(shuffle_seed + pass_index).shuffle(order)
    return order


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def critical_path(per_rank: list) -> list[float]:
    """Slowest participating rank of each repeat."""
    return [max(values) for values in zip(*(r for r in per_rank if r is not None))]


def summarize(record: dict) -> dict:
    """Reduce per-rank repeat timings of one measurement to critical-path stats."""
    critical = critical_path(record["per_rank_call_us"])
    launch = critical_path(record["per_rank_launch_call_us"])
    held = [h for rank in record["per_rank_queue_held"] if rank is not None for h in rank]
    return {
        "pass": record["pass"],
        "scenario": record["scenario"], "call": record["call"], "op": record["op"],
        "group": record["group"], "group_size": record["group_size"],
        "cuda_graph": record["cuda_graph"], "calls_per_forward": record["calls_per_forward"],
        "tokens_per_dp_rank": "/".join(map(str, record["tokens_per_dp_rank"])),
        "max_input_bytes": record["max_input_bytes"],
        "max_output_bytes": record["max_output_bytes"],
        "useful_tokens_per_dp_rank": "/".join(map(str, record["useful_tokens_per_dp_rank"])),
        "padded_tokens_per_dp_rank": "/".join(map(str, record["padded_tokens_per_dp_rank"])),
        "dtypes": ";".join(record["dtypes"]),
        "columns": "/".join(map(str, record["columns"])),
        "concurrent_dp_rails": record["concurrent_dp_rails"],
        "participating_ranks": "/".join(map(str, record["participating_ranks"])),
        "group_ranks": json.dumps(record["group_ranks"]),
        "timing_domain": "host" if record["op"] == "cpu_all_reduce" else "device",
        "communicator": actual_communicator(record["op"], record["kernels"]),
        "median_us": statistics.median(critical), "mean_us": statistics.fmean(critical),
        "std_us": statistics.pstdev(critical), "min_us": min(critical),
        "max_us": max(critical), "p10_us": _percentile(critical, 0.1),
        "p90_us": _percentile(critical, 0.9),
        "launch_median_us": statistics.median(launch) if launch else None,
        "queue_held_fraction": sum(held) / len(held) if held else None,
        "kernels": ";".join(record["kernels"]),
    }


def forward_totals(summary_rows: list[dict]) -> list[dict]:
    """Per-pass, per-scenario communication time of one forward pass (median-based)."""
    totals: dict[tuple, dict] = {}
    for row in summary_rows:
        total = totals.setdefault((row["pass"], row["scenario"]), {
            "pass": row["pass"], "scenario": row["scenario"], "cuda_graph": row["cuda_graph"],
            "tokens_per_dp_rank": row["tokens_per_dp_rank"],
            "forward_device_comm_us": 0.0, "forward_host_comm_us": 0.0})
        total["cuda_graph"] = total["cuda_graph"] or row["cuda_graph"]
        total[f"forward_{row['timing_domain']}_comm_us"] += (
            row["calls_per_forward"] * row["median_us"])
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


class CollectiveRunner:
    """One rank of ``instances`` vLLM serving instances that share the torch world.

    vLLM lays the world out as ExternalDP x DP x PP x TP
    (parallel_state.py ``initialize_model_parallel``), so instance ``i`` holds
    ranks ``[i * W, (i + 1) * W)`` with ``W = TP x DP``. Calls are planned on
    instance-local ranks; each instance measures on its own ranks.
    """

    def __init__(self, parallel: ServingParallelism, settings: dict, instances: int = 1,
                 *, engine: dict | None = None, dp_rail_isolation: bool = False):
        import torch
        from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
        from vllm.distributed import parallel_state
        from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

        self.torch = torch
        self.parallel = parallel
        self.settings = settings
        self.dp_rail_isolation = dp_rail_isolation
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        if int(os.environ["WORLD_SIZE"]) != instances * parallel.world_size:
            raise ValueError(f"torchrun WORLD_SIZE must equal {instances} x TP x DP")
        self.instance, self.instance_rank = divmod(self.rank, parallel.world_size)
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.device)
        # Under the external launcher ParallelConfig derives the DP rank from
        # RANK (config/parallel.py ``__post_init__``), so it reads the
        # instance-local rank, as it would in an instance with its own world.
        os.environ["RANK"] = str(self.instance_rank)
        try:
            self.vllm_config = VllmConfig(parallel_config=ParallelConfig(
                tensor_parallel_size=parallel.tensor_parallel_size,
                data_parallel_size=parallel.data_parallel_size,
                pipeline_parallel_size=parallel.pipeline_parallel_size,
                enable_expert_parallel=parallel.enable_expert_parallel,
                all2all_backend=parallel.all2all_backend,
                disable_custom_all_reduce=bool((engine or {}).get("disable_custom_all_reduce", False)),
                distributed_executor_backend="external_launcher"))
        finally:
            os.environ["RANK"] = str(self.rank)
        self._config_context = set_current_vllm_config(self.vllm_config)
        self._config_context.__enter__()
        # The native worker calls set_custom_all_reduce from this ParallelConfig
        # before creating groups; do not override FlashInfer/symmetric-memory envs.
        init_worker_distributed_environment(self.vllm_config, self.rank, "env://",
                                            self.local_rank)
        self.ps = parallel_state
        self.groups = {"tp": parallel_state.get_tp_group(),
                       "ep": parallel_state.get_ep_group(),
                       "dp": parallel_state.get_dp_group()}
        # Every rank creates every instance's group, as new_group requires.
        instance_groups = [torch.distributed.new_group(
            list(range(i * parallel.world_size, (i + 1) * parallel.world_size)),
            backend="gloo") for i in range(instances)]
        self.instance_cpu = instance_groups[self.instance]
        self._check_layout()
        self.hosts = [None] * parallel.world_size
        torch.distributed.all_gather_object(self.hosts, socket.gethostname(),
                                            group=self.instance_cpu)
        self._start_flag = torch.zeros(1, device=self.device)
        for group in self.groups.values():
            self._align_start(group)
        torch.cuda.synchronize()
        self.sleep_cycles_per_us = self._sleep_cycles_per_us()

    def _check_layout(self) -> None:
        """Fail when vLLM's groups differ from the plan's."""
        offset = self.instance * self.parallel.world_size
        for name, group in self.groups.items():
            expected = [offset + member for member in
                        group_members(self.parallel, name, self.instance_rank)]
            if list(group.ranks) != expected:
                raise RuntimeError(f"vLLM {name} group of rank {self.rank} is "
                                   f"{group.ranks}; the plan assumes {expected}")
        # Each planned sync op explicitly selects cpu_group or device_group.
        # Step's V2 host sync does not consult the V1 runner's NCCL-sync flag.

    def _align_start(self, group) -> None:
        """Queue a one-element all-reduce on ``group``'s device group.

        Work queued after it starts on every member when the all-reduce
        completes, instead of when each rank's CPU leaves the barrier.
        """
        self.torch.distributed.all_reduce(self._start_flag, group=group.device_group)

    def _sleep_cycles_per_us(self) -> float:
        torch = self.torch
        torch.cuda._sleep(SLEEP_CALIBRATION_CYCLES)
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        torch.cuda._sleep(SLEEP_CALIBRATION_CYCLES)
        end.record()
        end.synchronize()
        return SLEEP_CALIBRATION_CYCLES / (begin.elapsed_time(end) * 1e3)

    def communicators(self) -> dict:
        """Which all-reduce communicators vLLM enabled on each group of this rank."""
        import vllm
        from vllm import envs

        enabled = {}
        pynccl_versions = {}
        for name, group in self.groups.items():
            comm = group.device_communicator
            enabled[name] = {
                attr: getattr(comm, attr, None) is not None
                and not getattr(getattr(comm, attr), "disabled", False)
                 for attr in ("pynccl_comm", "ca_comm", "symm_mem_comm", "fi_ar_comm",
                              "qr_comm", "all2all_manager")}
            pynccl_versions[name] = (comm.pynccl_comm.nccl_version
                                     if enabled[name]["pynccl_comm"] else None)
        return {"torch": self.torch.__version__, "vllm": vllm.__version__,
                 "enabled_communicators": enabled,
                 "pynccl_versions": pynccl_versions,
                "disable_custom_all_reduce": self.vllm_config.parallel_config.disable_custom_all_reduce,
                "disable_nccl_for_dp_synchronization": (
                    self.vllm_config.parallel_config.disable_nccl_for_dp_synchronization),
                "dp_rail_isolation": self.dp_rail_isolation,
                **runtime_environment(envs),
                "group_ranks": {name: list(g.ranks) for name, g in self.groups.items()},
                "hosts": self.hosts,
                "sleep_cycles_per_us": self.sleep_cycles_per_us}

    def _tensors(self, call: CollectiveCall, rows: int, device):
        torch = self.torch
        tensors = []
        for columns, dtype in zip(call.columns, call.dtypes):
            # Casting BF16 avoids requiring an FP8 fill kernel during allocation.
            allocation_dtype = torch.bfloat16 if dtype == "float8_e4m3fn" else getattr(torch, dtype)
            tensor = torch.ones((rows, columns), dtype=allocation_dtype, device=device)
            tensors.append(tensor.to(dtype=getattr(torch, dtype)))
        return tensors

    def _issue(self, call: CollectiveCall):
        """Return a zero-argument function that issues ``call`` once on this rank."""
        torch = self.torch
        group = self.groups[call.group]
        rows = call.shard_rows[self.instance_rank]
        offset = self.instance * self.parallel.world_size
        sizes = [call.shard_rows[m - offset] for m in group.ranks]
        if call.op in ("local_copy", "local_add"):
            (tensor,) = self._tensors(call, rows, self.device)
            output = torch.empty_like(tensor)
            if call.op == "local_copy":
                if tensor.dtype == torch.float8_e4m3fn:
                    tensor, output = tensor.view(torch.uint8), output.view(torch.uint8)
                return lambda: output.copy_(tensor)
            if tensor.dtype != torch.bfloat16:
                raise ValueError("The measured reduction proxy requires actual BF16 operands")
            peer = torch.empty_like(tensor)
            return lambda: torch.add(tensor, peer, out=output)
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
            (tensor,) = self._tensors(call, rows * len(sizes), self.device)
            return lambda: group.reduce_scatter(tensor, dim=call.dim)
        if call.op == "all_gatherv":
            tensors = self._tensors(call, rows, self.device)
            def issue():
                return group.all_gatherv(tensors, dim=call.dim, sizes=sizes)
            outputs = issue()
            issue.buffer_receipt = dict(entrypoint="all_gatherv",sizes=sizes,dim=call.dim,
                inputs=[dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),
                             element_size=t.element_size()) for t in tensors],
                outputs=[dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),
                              element_size=t.element_size()) for t in outputs],
                lifetime="captured" if call.captured else "eager_native_allocation")
            return issue
        if call.op == "reduce_scatterv":
            (tensor,) = self._tensors(call, sum(sizes), self.device)
            return lambda: group.reduce_scatterv(tensor, dim=call.dim, sizes=sizes)
        if call.op == "reduce_scatterv_into_output":
            (tensor,) = self._tensors(call, sum(sizes), self.device)
            (output,) = self._tensors(call, rows, self.device)
            from vllm.distributed.device_communicators.pynccl_allocator import is_symmetric_memory_tensor
            from vllm import envs
            def layout(value):
                return dict(shape=list(value.shape),stride=list(value.stride()),dtype=str(value.dtype),
                            element_size=value.element_size(),registered=is_symmetric_memory_tensor(value))
            def issue():
                return group.reduce_scatterv_into_output(output, tensor, dim=call.dim, sizes=sizes)
            returned = issue()
            issue.buffer_receipt = dict(entrypoint="reduce_scatterv_into_output",sizes=sizes,dim=call.dim,
                input=layout(tensor),output=layout(output),supplied_output=True,
                input_output_alias=tensor.data_ptr() == output.data_ptr(),
                return_output_alias=returned.data_ptr() == output.data_ptr(),
                explicit_size_branch="uniform" if len(set(sizes)) == 1 else "unequal",
                lifetime="captured" if call.captured else "eager_preallocated",
                timed_allocation=False,pynccl_version=group.device_communicator.pynccl_comm.nccl_version,
                nccl_symm_enabled=envs.VLLM_USE_NCCL_SYMM_MEM,batch_invariant=envs.VLLM_BATCH_INVARIANT)
            if not issue.buffer_receipt["return_output_alias"]:
                raise ValueError("Native into-output combine did not retain the supplied output storage")
            return issue
        raise ValueError(f"Unknown op {call.op!r}")

    def prepare(self, call: CollectiveCall):
        """Warm ``call`` up; return ``(issue, run)``, ``run`` issuing ``calls_per_repeat`` calls.

        Captured calls run as one CUDA-graph replay. The caller keeps ``issue``
        alive because the graph reads the tensors it owns.
        """
        torch = self.torch
        issue = self._issue(call)
        per_repeat = self.settings["calls_per_repeat"]
        for _ in range(self.settings["warmup_calls"]):
            issue()
        if call.op != "cpu_all_reduce":
            torch.cuda.synchronize()
        if not call.captured:
            def run():
                for _ in range(per_repeat):
                    issue()
            return issue, run
        graph = torch.cuda.CUDAGraph()
        # Only this group participates. A global context would register TP
        # buffers on inactive ranks during an isolated DP-rail capture.
        with self.groups[call.group].graph_capture() as context:
            with torch.cuda.graph(graph, stream=context.stream):
                for _ in range(per_repeat):
                    issue()
        torch.cuda.synchronize()
        return issue, graph.replay

    def gpu_repeat(self, run, group, sleep_cycles: int) -> tuple[float, bool, float]:
        """Return device call time, whether its begin event is pending, and CPU enqueue time."""
        torch = self.torch
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        torch.distributed.barrier(group=group.cpu_group)
        if sleep_cycles:
            self._align_start(group)
            torch.cuda._sleep(sleep_cycles)
        start = time.perf_counter()
        begin.record()
        run()
        end.record()
        enqueue_us = (time.perf_counter() - start) * 1e6
        # Both events and every launch must be queued before the GPU reaches
        # begin. Alignment latency is outside this enqueue window.
        held = bool(sleep_cycles) and not begin.query()
        end.synchronize()
        per_call_us = begin.elapsed_time(end) * 1e3 / self.settings["calls_per_repeat"]
        return per_call_us, held, enqueue_us

    def time_call(self, call: CollectiveCall, run) -> dict:
        """Time ``run`` on the ranks of the call's group.

        GPU calls are timed twice. ``launch_call_us`` starts the CUDA events
        when the CPU starts launching, so small calls measure the launch path.
        ``call_us`` first queues a GPU sleep, of equal length on every rank and
        started by a device-side all-reduce on the group, that outlasts the
        launches, so the events time the device work alone from aligned start
        times; ``queue_held`` checks that begin was still pending after enqueue.
        """
        torch = self.torch
        group = self.groups[call.group]
        repeats, per_repeat = self.settings["repeats"], self.settings["calls_per_repeat"]
        if call.op == "cpu_all_reduce":
            timings = []
            for _ in range(repeats):
                torch.distributed.barrier(group=group.cpu_group)
                start = time.perf_counter()
                run()
                timings.append((time.perf_counter() - start) * 1e6 / per_repeat)
            return {"call_us": timings, "launch_call_us": None, "queue_held": None,
                    "launch_enqueue_us": None, "enqueue_us": None, "queue_sleep_us": None}
        launch = [self.gpu_repeat(run, group, 0) for _ in range(repeats)]
        enqueue_us = [item[2] for item in launch]
        sleep_cycles = self.queue_sleep_cycles(enqueue_us, group)
        queued = [self.gpu_repeat(run, group, sleep_cycles) for _ in range(repeats)]
        return {"call_us": [q[0] for q in queued], "launch_call_us": [item[0] for item in launch],
                "queue_held": [q[1] for q in queued], "launch_enqueue_us": enqueue_us,
                "enqueue_us": [q[2] for q in queued],
                "queue_sleep_us": sleep_cycles / self.sleep_cycles_per_us}

    def queue_sleep_cycles(self, launch_enqueue_us: list[float], group) -> int:
        """Budget the GPU sleep from the measured CPU enqueue window on every member."""
        torch = self.torch
        sleep_us = torch.tensor(
            [QUEUE_SLEEP_MARGIN * max(launch_enqueue_us)],
            dtype=torch.float64)
        torch.distributed.all_reduce(sleep_us, op=torch.distributed.ReduceOp.MAX,
                                     group=group.cpu_group)
        return int(sleep_us.item() * self.sleep_cycles_per_us)

    def participates(self, call: CollectiveCall, scenario=None) -> bool:
        """Whether this rank issues ``call``; ranks without rows, such as idle DP ranks
        for logits, skip it."""
        if call.shard_rows[self.instance_rank] == 0:
            return False
        if call.group == "dp":
            rails = active_dp_rails(self.parallel, call, scenario, self.dp_rail_isolation)
            return self.instance_rank % self.parallel.tensor_parallel_size in rails
        return True

    def gather(self, local) -> list | None:
        """``local`` of every rank of this instance, on its first rank; None elsewhere."""
        gathered = [None] * self.parallel.world_size
        self.torch.distributed.all_gather_object(gathered, local, group=self.instance_cpu)
        return gathered if self.instance_rank == 0 else None

    def call_record(self, scenario, call: CollectiveCall, gathered: list) -> dict:
        """Identity and byte counts of ``call``, merged over the gathered ranks."""
        present = [g for g in gathered if g is not None]
        ranks = [rank for rank, value in enumerate(gathered) if value is not None]
        offset = self.instance * self.parallel.world_size
        groups = sorted({tuple(offset + m for m in group_members(self.parallel, call.group, rank))
                         for rank in ranks})
        return {
            "scenario": scenario.name, "call": call.name, "op": call.op,
            "group": call.group, "group_size": self.groups[call.group].world_size,
            "cuda_graph": call.captured,
            "calls_per_forward": call.calls_per_forward,
            "tokens_per_dp_rank": list(scenario.tokens_per_dp_rank),
            "useful_tokens_per_dp_rank": list(
                getattr(scenario, "useful_tokens_per_dp_rank", None) or scenario.tokens_per_dp_rank),
            "padded_tokens_per_dp_rank": list(scenario.tokens_per_dp_rank),
            "dtypes": list(call.dtypes), "columns": list(call.columns),
            "concurrent_dp_rails": len(groups) if call.group == "dp" else None,
            "participating_ranks": [offset + rank for rank in ranks],
            "group_ranks": [list(members) for members in groups],
            "shard_rows": list(call.shard_rows), "source": call.source,
            "max_input_bytes": max(g["bytes"][0] for g in present),
            "max_output_bytes": max(g["bytes"][1] for g in present),
        }

    def measure(self, scenario, call: CollectiveCall) -> dict | None:
        """Time one call; return the merged record on the instance's first rank."""
        self.torch.distributed.barrier(group=self.instance_cpu)
        local = None
        if self.participates(call, scenario):
            prepared = getattr(self, "_prepared_calls", None)
            if prepared is None:
                prepared = self._prepared_calls = {}
            key = scenario.name, call.name
            if key not in prepared:
                prepared[key] = self.prepare(call)
            issue, run = prepared[key]
            local = {**self.time_call(call, run),
                      "bytes": call_bytes(call, self.parallel, self.instance_rank)}
            if hasattr(issue,"buffer_receipt"):
                local["buffer_receipt"] = issue.buffer_receipt
            del issue, run
        gathered = self.gather(local)
        if gathered is None:
            return None
        record = {
            **self.call_record(scenario, call, gathered),
            "per_rank_call_us": [g["call_us"] if g else None for g in gathered],
            "per_rank_launch_call_us": [g["launch_call_us"] if g else None for g in gathered],
            "per_rank_queue_held": [g["queue_held"] if g else None for g in gathered],
            "per_rank_launch_enqueue_us": [g["launch_enqueue_us"] if g else None for g in gathered],
            "per_rank_enqueue_us": [g["enqueue_us"] if g else None for g in gathered],
            "per_rank_queue_sleep_us": [g["queue_sleep_us"] if g else None for g in gathered],
        }
        if any(g and "buffer_receipt" in g for g in gathered):
            record["buffer_receipt"] = {str(rank):g["buffer_receipt"] for rank,g in enumerate(gathered) if g}
        return record

    def kernel_names(self, call: CollectiveCall, scenario=None, *, prepared=None) -> list[str] | None:
        """CUDA kernels of one ``run`` of ``call``, merged over ranks on the first rank."""
        torch = self.torch
        torch.distributed.barrier(group=self.instance_cpu)
        names, graph_events = [], []
        if self.participates(call, scenario) and call.op != "cpu_all_reduce":
            if prepared is None:
                prepared = getattr(self, "_prepared_calls", {}).get((getattr(scenario,"name",None),call.name))
            issue, run = self.prepare(call) if prepared is None else prepared
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                run()
                torch.cuda.synchronize()
            names = [event.name for event in profile.events()
                      if event.device_type == torch.autograd.DeviceType.CUDA]
            graph_events = [event.name for event in profile.events() if event.name == "cudaGraphLaunch"]
            del issue, run
        evidence = self.gather({"kernels":names, "graph_events":graph_events})
        gathered = None if evidence is None else [rank["kernels"] for rank in evidence]
        if gathered is None:
            return None
        graphs = [rank["graph_events"] for rank in evidence]
        per_repeat = self.settings["calls_per_repeat"]
        self._last_kernel_evidence = {
            "per_rank_events": gathered, "per_rank_graph_events": graphs,
            "profiled_calls": per_repeat,
            "per_rank_kernel_counts": [len(events)//per_repeat if events and len(events)%per_repeat == 0
                                       else None for events in gathered],
            "kernel_count_scope": "observed_device_events_per_native_call",
            "graph_launch_count_scope": "observed_cpu_cudaGraphLaunch_events_per_profiled_run"}
        self._last_kernel_evidence["per_rank_copy_events"] = [
            [name for name in events if "memcpy" in name.lower() or "copy" in name.lower()]
            for events in gathered]
        return sorted({name for rank_names in gathered for name in rank_names})

    def measure_local_work(self, definition):
        """Independent preallocated layout-specific CUDA probes, not native durations."""
        kind, dtype, columns = definition["kind"], definition["dtype"], definition["columns"]
        if kind not in ("copy", "add") or dtype not in ("bfloat16", "float8_e4m3fn", "float32", "int32"):
            raise ValueError("Unsupported reached local-work proxy")
        from frontier.profiling.collectives.vllm_forward_collectives import ForwardScenario
        rows = definition["rows"]
        scenario = ForwardScenario(definition["name"], (rows,)*self.parallel.data_parallel_size,
                                   (0,)*self.parallel.data_parallel_size, definition["cuda_graph"])
        call = CollectiveCall(definition["name"], "local_"+kind, "tp", 0, (columns,), (dtype,),
                              (rows,)*self.parallel.world_size, 1, "independent_local_proxy", scenario.cuda_graph)
        issue, run = self.prepare(call)
        local = self.time_call(call, run)
        gathered = self.gather(local)
        kernels = self.kernel_names(call, scenario, prepared=(issue,run))
        del issue, run
        if gathered is None:
            return None
        return {**definition, "scope":"independent_preallocated_operand", "row_stride":columns,
            "fp8_copy_byte_view":dtype == "float8_e4m3fn", "kernels":kernels,
            "per_rank_call_us":[r["call_us"] for r in gathered],
            "per_rank_launch_call_us":[r["launch_call_us"] for r in gathered],
            "per_rank_queue_held":[r["queue_held"] for r in gathered],
            "device":str(self.device), **self._last_kernel_evidence}


def write_csv(path: Path, rows: list[dict]) -> None:
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
    parser.add_argument("--dp-rail-isolation", action="store_true",
                        help="Issue DP device calls only on the TP-index-0 group")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    parallel = ServingParallelism.from_engine(config["engine"])
    model = ModelShape.from_hf_config(json.loads(Path(config["model_config"]).read_text()))
    scenarios = scenarios_from_grid(parallel, config["grid"])
    if args.dp_rail_isolation:
        scenarios = [replace(scenario, active_dp_rails=(0,)) for scenario in scenarios]
    plan = [(s, c) for s in scenarios for c in forward_collectives(
        model, parallel, s, native_call_names=config["grid"].get("native_call_names"))]
    settings = {key: int(config[key]) for key in RUN_SETTINGS}
    if args.list_cases:
        for scenario, call in plan:
            print(scenario.name, call.name, call.op, call.group,
                  call_bytes(call, parallel, 0), call.calls_per_forward,
                  "dp_rails", active_dp_rails(parallel, call, scenario, args.dp_rail_isolation)
                  if call.group == "dp" else ())
        print("cases", len(plan), "passes", settings["passes"],
              "measurements", len(plan) * settings["passes"])
        return
    runner = CollectiveRunner(parallel, settings, engine=config["engine"],
                              dp_rail_isolation=args.dp_rail_isolation)
    output = Path(config["output_dir"])
    log = None
    if runner.rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "plan.json").write_text(json.dumps(
            {"config": config, "cases": len(plan), "runtime": runner.communicators()},
            indent=2))
        log = (output / "measurements.jsonl").open("a")
    records = []
    for pass_index in range(settings["passes"]):
        for scenario, call in pass_order(plan, settings["shuffle_seed"], pass_index):
            record = runner.measure(scenario, call)
            if record is None:
                continue
            record = {"pass": pass_index, **record}
            log.write(json.dumps(record) + "\n")
            log.flush()
            records.append(record)
    kernels, evidence = {}, {}
    for s,c in plan:
        kernels[s.name,c.name] = runner.kernel_names(c,s)
        if runner.rank == 0:
            evidence[s.name,c.name] = runner._last_kernel_evidence
    local_records = []
    for definition in config["grid"].get("local_work", []):
        for p in range(settings["passes"]):
            record = runner.measure_local_work(definition)
            if record is not None:
                local_records.append({"pass":p, **record})
    if runner.rank == 0:
        log.close()
        with (output / "kernels.jsonl").open("a") as kernel_log:
            for scenario, call in plan:
                names = kernels[(scenario.name, call.name)]
                kernel_log.write(json.dumps({"scenario": scenario.name, "call": call.name,
                                              "communicator": actual_communicator(call.op, names),
                                              "kernels": names, **evidence[scenario.name,call.name]}) + "\n")
        if local_records:
            (output/"local_work.jsonl").write_text("".join(json.dumps(r)+"\n" for r in local_records))
        summary = [summarize({**r, "kernels": kernels[(r["scenario"], r["call"])]})
                   for r in records]
        write_csv(output / "summary.csv", summary)
        write_csv(output / "forward_totals.csv", forward_totals(summary))
        write_csv(output / "all_reduce.csv",
                   network_all_reduce_rows(summary, int(config["gpus_per_node"])))
    runner.torch.distributed.barrier(group=runner.instance_cpu)


if __name__ == "__main__":
    main()
