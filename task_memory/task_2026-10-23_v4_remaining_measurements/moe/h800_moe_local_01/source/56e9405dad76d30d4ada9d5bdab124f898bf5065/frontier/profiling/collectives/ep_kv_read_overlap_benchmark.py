"""Measure EP dispatch/combine and NIXL KV READs on decode nodes, alone and at the same time.

Run under ``torchrun`` over two serving instances of the configured engine,
``WORLD_SIZE = 2 x TP x DP``. vLLM lays the world out as ExternalDP x DP x PP x
TP (``CollectiveRunner``), so each instance has its own TP, DP and EP groups.
The first instance is the prefill (P) side and holds the KV blocks; the second
is the decode (D) side. D rank ``r`` reads from P rank ``r - TP x DP``
(``KVReadPeer``), and every D rank reads at once, as an MLA transfer does when
P and D use the same TP size. The READs therefore arrive through the NICs that
carry the D instance's cross-node EP traffic.

For each EP call of the plan (dispatch ``all_gatherv`` and combine
``reduce_scatterv`` at the configured step sizes), the D ranks run four phases:

1. ``ep_alone``: the call, timed as ``CollectiveRunner.time_call`` times it.
2. ``read_alone``: the configured READ, timed as the NIXL benchmark times it.
3. ``ep_loaded``: queued EP repeats with the same GPU sleep as phase 1, each
   started while its own READ is in flight. ``ep_loaded_covered`` records
   whether that READ was still in flight when the repeat ended.
4. ``read_loaded``: READs timed while the D ranks issue the call back to back
   with at most ``ep_load_depth`` runs queued. The run count is
   ``ep_load_cover`` times the median READ-alone time over the median EP run
   time, maximized over the EP group so every member issues the same calls.
   ``read_loaded_covered`` records whether the READ finished before the EP
   runs did.

P ranks wait at the end of each case and count the READ notifications. The
plan runs ``passes`` times, each pass in an order shuffled with
``shuffle_seed + pass``. Rank 0 appends one JSON line per (pass, case) to
``<output_dir>/overlap.jsonl`` and writes ``overlap_summary.csv``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import deque
from pathlib import Path

from frontier.profiling.collectives.nixl_kv_read_benchmark import (
    KVReadPeer,
    KVRegionLayout,
    ReadCase,
    init_process_group,
    read_cases,
)
from frontier.profiling.collectives.vllm_collective_benchmark import (
    RUN_SETTINGS,
    CollectiveRunner,
    critical_path,
    pass_order,
    write_csv,
)
from frontier.profiling.collectives.vllm_forward_collectives import (
    ModelShape,
    ServingParallelism,
    call_bytes,
    forward_collectives,
    scenarios_from_grid,
)

INSTANCES = 2
OVERLAP_SETTINGS = ("read_warmup", "notification_wait_seconds", "ep_load_depth")


def ep_plan(model: ModelShape, parallel: ServingParallelism, grid: dict) -> list[tuple]:
    """(scenario, call) of every EP-group call of the grid's forward passes."""
    return [(s, c) for s in scenarios_from_grid(parallel, grid)
            for c in forward_collectives(model, parallel, s) if c.group == "ep"]


def overlap_read(config: dict, layout: KVRegionLayout, pairs: int) -> ReadCase:
    """The READ of every D rank in each phase; all pairs read at once."""
    read = config["read"]
    (case,) = read_cases([{"prompt_tokens": [read["prompt_tokens"]], "active_pairs": [pairs],
                           "requests_in_flight": [read["requests_in_flight"]]}], layout, pairs)
    return case


def expected_notifications(read: ReadCase, settings: dict) -> int:
    """READ notifications one P rank receives per case: phases 2-4."""
    return read.requests_in_flight * (settings["read_warmup"] + 3 * settings["repeats"])


def load_runs(read_alone_us: float, ep_run_us: float, cover: float) -> int:
    """EP runs whose total time is ``cover`` times the READ-alone time."""
    return max(1, math.ceil(cover * read_alone_us / ep_run_us))


def issue_load(torch, run, runs: int, depth: int, poll) -> bool:
    """Issue ``run`` ``runs`` times, at most ``depth`` queued, calling ``poll`` while waiting.

    Return the last ``poll`` taken before the final run completed: True when
    the polled work had finished while EP work was still running.
    """
    pending = deque()
    finished = False
    for _ in range(runs):
        while len(pending) >= depth:
            finished = poll()
            if pending[0].query():
                pending.popleft()
        run()
        event = torch.cuda.Event()
        event.record()
        pending.append(event)
    while pending:
        finished = poll()
        if pending[0].query():
            pending.popleft()
    return finished


def measure_case(runner: CollectiveRunner, peer: KVReadPeer, call, read: ReadCase,
                 settings: dict, tag: str) -> dict:
    """The four phases of one EP call on a D rank."""
    torch = runner.torch
    group = runner.groups[call.group]
    repeats = settings["repeats"]
    issue, run = runner.prepare(call)
    ep_alone = runner.time_call(call, run)

    peer.clear()
    read_alone, verified = [], None
    for repeat in range(settings["read_warmup"] + repeats):
        torch.distributed.barrier(group=group.cpu_group)
        timing = peer.finish_reads(peer.post_reads(read, f"{tag}:alone:{repeat}"))
        if verified is None:
            verified = peer.verify(read)
        if repeat >= settings["read_warmup"]:
            read_alone.append(timing)

    sleep_cycles = runner.queue_sleep_cycles(ep_alone["launch_call_us"], group)
    ep_loaded, ep_held, ep_covered = [], [], []
    for repeat in range(repeats):
        reads = peer.post_reads(read, f"{tag}:ep_loaded:{repeat}")
        call_us, held = runner.gpu_repeat(run, group, sleep_cycles)
        ep_covered.append(not peer.reads_done(reads))
        peer.finish_reads(reads)
        ep_loaded.append(call_us)
        ep_held.append(held)

    runs = torch.tensor([load_runs(
        statistics.median(t["complete_us"] for t in read_alone),
        statistics.median(ep_alone["call_us"]) * settings["calls_per_repeat"],
        settings["ep_load_cover"])], dtype=torch.int64)
    torch.distributed.all_reduce(runs, op=torch.distributed.ReduceOp.MAX, group=group.cpu_group)
    runs = int(runs.item())
    read_loaded, read_covered = [], []
    for repeat in range(repeats):
        torch.distributed.barrier(group=group.cpu_group)
        reads = peer.post_reads(read, f"{tag}:read_loaded:{repeat}")
        read_covered.append(issue_load(torch, run, runs, settings["ep_load_depth"],
                                       lambda: peer.reads_done(reads)))
        read_loaded.append(peer.finish_reads(reads))
    torch.cuda.synchronize()
    del issue, run
    return {
        "ep_alone_call_us": ep_alone["call_us"], "ep_alone_queue_held": ep_alone["queue_held"],
        "ep_loaded_call_us": ep_loaded, "ep_loaded_queue_held": ep_held,
        "ep_loaded_covered": ep_covered, "ep_load_runs": runs,
        "read_alone": read_alone, "read_loaded": read_loaded,
        "read_loaded_covered": read_covered, "data_verified": verified,
    }


def summarize_overlap(record: dict) -> dict:
    """Critical-path medians of each phase of one (pass, case) record."""
    ranks = record["per_rank"]

    def times(key: str) -> list[float]:
        return critical_path([r[key] for r in ranks])

    def read_times(key: str) -> list[float]:
        return critical_path([[t["complete_us"] for t in r[key]] for r in ranks])

    def fraction(key: str) -> float:
        flags = [flag for r in ranks for flag in r[key]]
        return sum(flags) / len(flags)

    ep_alone, ep_loaded = (statistics.median(times(k))
                           for k in ("ep_alone_call_us", "ep_loaded_call_us"))
    read_alone, read_loaded = (statistics.median(read_times(k))
                               for k in ("read_alone", "read_loaded"))
    return {
        "pass": record["pass"], "scenario": record["scenario"], "call": record["call"],
        "op": record["op"], "cuda_graph": record["cuda_graph"],
        "tokens_per_dp_rank": "/".join(map(str, record["tokens_per_dp_rank"])),
        "max_input_bytes": record["max_input_bytes"],
        "max_output_bytes": record["max_output_bytes"],
        "read_case": record["read_case"], "bytes_per_reader": record["bytes_per_reader"],
        "ep_load_runs": ranks[0]["ep_load_runs"],
        "ep_alone_median_us": ep_alone, "ep_loaded_median_us": ep_loaded,
        "ep_loaded_over_alone": ep_loaded / ep_alone,
        "ep_alone_queue_held_fraction": fraction("ep_alone_queue_held"),
        "ep_loaded_queue_held_fraction": fraction("ep_loaded_queue_held"),
        "ep_loaded_covered_fraction": fraction("ep_loaded_covered"),
        "read_alone_median_us": read_alone, "read_loaded_median_us": read_loaded,
        "read_loaded_over_alone": read_loaded / read_alone,
        "read_alone_per_reader_gigabytes_per_s": record["bytes_per_reader"] / read_alone / 1e3,
        "read_loaded_per_reader_gigabytes_per_s": record["bytes_per_reader"] / read_loaded / 1e3,
        "read_loaded_covered_fraction": fraction("read_loaded_covered"),
        "data_verified": all(r["data_verified"] for r in ranks),
        "notifications_received": record["notifications_received"],
        "notifications_expected": record["notifications_expected"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--list-cases", action="store_true",
                        help="Print the planned cases without initializing CUDA")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    parallel = ServingParallelism.from_engine(config["engine"])
    model = ModelShape.from_hf_config(json.loads(Path(config["model_config"]).read_text()))
    plan = ep_plan(model, parallel, config["grid"])
    layout = KVRegionLayout.from_config(config["kv_cache"])
    read = overlap_read(config, layout, parallel.world_size)
    settings = {key: int(config[key]) for key in RUN_SETTINGS + OVERLAP_SETTINGS}
    settings["ep_load_cover"] = float(config["ep_load_cover"])
    if args.list_cases:
        for scenario, call in plan:
            print(scenario.name, call.name, call.op, call_bytes(call, parallel, 0))
        print("read", read.name, read.blocks_per_request * read.requests_in_flight
              * layout.block_bytes)
        print("cases", len(plan), "passes", settings["passes"],
              "measurements", len(plan) * settings["passes"])
        return
    init_process_group(config)
    runner = CollectiveRunner(parallel, settings, instances=INSTANCES)
    peer = KVReadPeer(config, layout, parallel.world_size)
    dist = runner.torch.distributed
    world = dist.get_world_size()
    runtime = [None] * world
    dist.all_gather_object(runtime, {**runner.communicators(), "nixl": peer.environment})
    output = Path(config["output_dir"])
    if runner.rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "overlap_plan.json").write_text(json.dumps(
            {"config": config, "cases": len(plan), "read_case": read.name,
             "instances": [runtime[i * parallel.world_size] for i in range(INSTANCES)]},
            indent=2, default=str))
    expected = expected_notifications(read, settings)
    rows = []
    for pass_index in range(settings["passes"]):
        for scenario, call in pass_order(plan, settings["shuffle_seed"], pass_index):
            local = None
            if peer.is_reader:
                local = {"rank": runner.rank,
                         "bytes": call_bytes(call, parallel, runner.instance_rank),
                         **measure_case(runner, peer, call, read, settings,
                                        f"{pass_index}:{scenario.name}:{call.name}")}
            dist.barrier()
            received = (0 if peer.is_reader else
                        peer.drain_notifications(expected, settings["notification_wait_seconds"]))
            gathered = [None] * world
            dist.all_gather_object(gathered, {"received": received, "reader": local})
            if runner.rank != 0:
                continue
            readers = [g["reader"] for g in gathered if g["reader"] is not None]
            record = {
                "pass": pass_index, **runner.call_record(scenario, call, readers),
                "read_case": read.name,
                "bytes_per_reader": read.blocks_per_request * read.requests_in_flight
                * layout.block_bytes,
                "per_rank": [{k: v for k, v in r.items() if k != "bytes"} for r in readers],
                "notifications_received": sum(g["received"] for g in gathered),
                "notifications_expected": expected * len(readers),
            }
            with (output / "overlap.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")
            rows.append(summarize_overlap(record))
    if runner.rank == 0:
        write_csv(output / "overlap_summary.csv", rows)
    peer.close()


if __name__ == "__main__":
    main()
