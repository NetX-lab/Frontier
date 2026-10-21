"""Measure NIXL KV-cache READs between prefill and decode ranks as vLLM v0.28 issues them.

Run under ``torchrun`` on two nodes with ``nproc_per_node`` equal to the TP
size: node rank 0 holds the prefill (P) ranks, node rank 1 the decode (D)
ranks. Each D rank pairs with the P rank of the same TP index, which is the
only source an MLA model reads from when P and D use the same TP size
(nixl/tp_mapping.py, nixl/pull_worker.py ``_read_blocks_for_req``).

Each rank registers one NIXL region per KV-cache owner layer with
``num_gpu_blocks`` blocks (nixl/base_worker.py ``register_kv_caches``) and
preps one descriptor list of ``num_regions x num_gpu_blocks`` block
descriptors in region-major order (``_build_fa_local``). A request READ is one
``make_prepped_xfer`` over the descriptor ids of its blocks in every region
(``_compute_desc_ids``) followed by ``transfer``, with a notification to the
P rank (``_read_blocks``).

A case sets the prompt length, how many TP pairs read at once (1, or all of
them as in a real MLA transfer) and how many requests each D rank has in
flight. ``grid`` is a list of grid blocks whose cross products are joined, so
long prompts can use fewer requests in flight than the D block pool rejects.
The cases run ``passes`` times, each pass in an order shuffled with
``shuffle_seed + pass``. Each repeat starts after a CPU barrier; a D rank times
the descriptor preparation, the post, and the wait until every READ reports
``DONE``. Rank 0 appends one JSON line per (pass, case) to
``<output_dir>/nixl_reads.jsonl`` and writes ``nixl_summary.csv``, whose
bandwidth columns are in gigabytes per second.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from frontier.profiling.collectives.vllm_collective_benchmark import pass_order, write_csv

NIXL_MEMORY_TYPE = "VRAM"
LOCAL_AGENT = "NIXL_INIT_AGENT"


@dataclass(frozen=True)
class KVRegionLayout:
    """Per-region bytes of one KV block, in vLLM's region registration order."""

    region_block_bytes: tuple[int, ...]
    block_size: int
    num_gpu_blocks: int

    @classmethod
    def from_config(cls, config: dict) -> "KVRegionLayout":
        regions = tuple(int(group["block_bytes"]) for group in config["regions"]
                        for _ in range(int(group["count"])))
        layout = cls(regions, int(config["block_size"]), int(config["num_gpu_blocks"]))
        expected = int(config["resident_bytes_per_block"])
        if layout.block_bytes != expected:
            raise ValueError(f"Regions sum to {layout.block_bytes} bytes per block, "
                             f"the cache contract states {expected}")
        return layout

    @property
    def num_regions(self) -> int:
        return len(self.region_block_bytes)

    @property
    def block_bytes(self) -> int:
        return sum(self.region_block_bytes)


@dataclass(frozen=True)
class ReadCase:
    name: str
    prompt_tokens: int
    blocks_per_request: int
    active_pairs: int
    requests_in_flight: int


def descriptor_ids(num_regions: int, num_blocks: int, block_ids: list[int]) -> list[int]:
    """Region-major descriptor ids of ``block_ids`` (base_worker ``_compute_desc_ids``)."""
    return [region * num_blocks + block for region in range(num_regions)
            for block in block_ids]


def block_descriptors(base_addresses: list[int], layout: KVRegionLayout,
                      device: int) -> list[tuple[int, int, int]]:
    """(address, length, device) of every block of every region (``_build_fa_local``)."""
    return [(base + block * length, length, device)
            for base, length in zip(base_addresses, layout.region_block_bytes)
            for block in range(layout.num_gpu_blocks)]


def request_block_ids(case: ReadCase) -> list[list[int]]:
    """Distinct block ranges for the requests a D rank keeps in flight."""
    n = case.blocks_per_request
    return [list(range(i * n, (i + 1) * n)) for i in range(case.requests_in_flight)]


def read_cases(grids: list[dict], layout: KVRegionLayout, pairs: int) -> list[ReadCase]:
    """Cases of every grid block; ``pairs`` is the number of P/D rank pairs."""
    cases = []
    for grid in grids:
        for tokens in grid["prompt_tokens"]:
            blocks = -(-int(tokens) // layout.block_size)
            for active in grid["active_pairs"]:
                if not 0 < active <= pairs:
                    raise ValueError(f"active_pairs={active} outside 1..{pairs}")
                for in_flight in grid["requests_in_flight"]:
                    if blocks * in_flight > layout.num_gpu_blocks:
                        raise ValueError(f"{in_flight} requests of {blocks} blocks exceed "
                                         f"{layout.num_gpu_blocks} GPU blocks")
                    cases.append(ReadCase(f"t{tokens}_pairs{active}_inflight{in_flight}",
                                          int(tokens), blocks, int(active), int(in_flight)))
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("Grid blocks repeat a case")
    return cases


def summarize_case(record: dict) -> dict:
    """Critical-path statistics of one case from the active D ranks' repeats."""
    per_rank = [r["timings"] for r in record["readers"]]
    complete = [max(t["complete_us"] for t in repeat) for repeat in zip(*per_rank)]
    prep = [max(t["prep_us"] for t in repeat) for repeat in zip(*per_rank)]
    median = statistics.median(complete)
    total_bytes = record["bytes_per_reader"] * len(per_rank)
    return {
        "pass": record["pass"],
        "case": record["case"], "prompt_tokens": record["prompt_tokens"],
        "blocks_per_request": record["blocks_per_request"],
        "active_pairs": record["active_pairs"],
        "requests_in_flight": record["requests_in_flight"],
        "descriptors_per_request": record["descriptors_per_request"],
        "bytes_per_reader": record["bytes_per_reader"],
        "complete_median_us": median, "complete_min_us": min(complete),
        "complete_max_us": max(complete),
        "complete_p90_us": sorted(complete)[min(len(complete) - 1, int(0.9 * len(complete)))],
        "prep_median_us": statistics.median(prep),
        "per_reader_gigabytes_per_s": record["bytes_per_reader"] / median / 1e3,
        "aggregate_gigabytes_per_s": total_bytes / median / 1e3,
        "data_verified": all(r["data_verified"] for r in record["readers"]),
        "notifications_received": record["notifications_received"],
        "notifications_expected": record["notifications_expected"],
    }


class KVReadPeer:
    """One rank of ``pairs`` P/D rank pairs.

    Ranks below ``pairs`` are P; rank ``p + pairs`` reads from P rank ``p``.
    The caller initializes the torch default process group first.
    """

    def __init__(self, config: dict, layout: KVRegionLayout, pairs: int):
        import torch
        import torch.distributed as dist
        from nixl._api import nixl_agent, nixl_agent_config

        self.torch, self.dist = torch, dist
        self.layout = layout
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        if dist.get_world_size() != 2 * pairs:
            raise ValueError(f"WORLD_SIZE must be twice the {pairs} P/D pairs")
        self.is_reader = self.rank >= pairs
        self.pair_index = self.rank % pairs
        torch.cuda.set_device(self.local_rank)
        # One tensor per region, as each owner layer has its own cache tensor.
        self.regions = [torch.empty((layout.num_gpu_blocks, length), dtype=torch.uint8,
                                    device="cuda") for length in layout.region_block_bytes]
        for index, region in enumerate(self.regions):
            region.fill_(0 if self.is_reader else self._pattern(index))
        torch.cuda.synchronize()
        role = "decode" if self.is_reader else "prefill"
        self.agent = nixl_agent(f"{role}-{self.pair_index}-{os.getpid()}", nixl_agent_config(
            num_threads=int(config["nixl_num_threads"]), capture_telemetry=True))
        self.registration = self.agent.get_reg_descs(
            [(r.data_ptr(), r.nbytes, self.local_rank, "") for r in self.regions],
            NIXL_MEMORY_TYPE)
        self.agent.register_memory(self.registration, backends=["UCX"])
        own = {"metadata": self.agent.get_agent_metadata(),
               "bases": [r.data_ptr() for r in self.regions], "device": self.local_rank}
        everyone = [None] * dist.get_world_size()
        dist.all_gather_object(everyone, own)
        if self.is_reader:
            source = everyone[self.pair_index]
            self.remote_agent = self.agent.add_remote_agent(source["metadata"])
            self.local_dlist = self._prep(LOCAL_AGENT, own["bases"], self.local_rank)
            self.remote_dlist = self._prep(self.remote_agent, source["bases"],
                                           source["device"])
        self.environment = {
            "ucx_net_devices": os.environ.get("UCX_NET_DEVICES"),
            "ucx_backend_params": self.agent.get_backend_params("UCX"),
        }

    @staticmethod
    def _pattern(region_index: int) -> int:
        return region_index % 251 + 1

    def _prep(self, agent_name, bases, device):
        descs = self.agent.get_xfer_descs(block_descriptors(bases, self.layout, device),
                                          NIXL_MEMORY_TYPE)
        return self.agent.prep_xfer_dlist(agent_name, descs)

    def post_reads(self, case: ReadCase, tag: str) -> dict:
        """Prepare and post the case's READs; return the in-flight reads."""
        ids = [descriptor_ids(self.layout.num_regions, self.layout.num_gpu_blocks, blocks)
               for blocks in request_block_ids(case)]
        start = time.perf_counter()
        requests = [self.agent.make_prepped_xfer(
            "READ", self.local_dlist, desc_ids, self.remote_dlist, desc_ids,
            notif_msg=f"{tag}:{i}".encode()) for i, desc_ids in enumerate(ids)]
        prepared = time.perf_counter()
        states = [self.agent.transfer(request) for request in requests]
        return {"case": case, "requests": requests, "start": start, "prepared": prepared,
                "posted": time.perf_counter(), "done": None,
                "pending": {i for i, state in enumerate(states) if state != "DONE"}}

    def reads_done(self, reads: dict) -> bool:
        """Poll the in-flight reads once; True once every READ reported ``DONE``.

        The first poll that sees them all done stamps ``reads["done"]``.
        """
        for i in list(reads["pending"]):
            state = self.agent.check_xfer_state(reads["requests"][i])
            if state == "ERR":
                raise RuntimeError(f"NIXL READ failed in {reads['case'].name}")
            if state == "DONE":
                reads["pending"].discard(i)
        if not reads["pending"] and reads["done"] is None:
            reads["done"] = time.perf_counter()
        return reads["done"] is not None

    def finish_reads(self, reads: dict) -> dict:
        """Wait for the reads, release them, and return their timings in microseconds."""
        while not self.reads_done(reads):
            pass
        for request in reads["requests"]:
            self.agent.release_xfer_handle(request)
        start = reads["start"]
        return {"prep_us": (reads["prepared"] - start) * 1e6,
                "post_us": (reads["posted"] - reads["prepared"]) * 1e6,
                "complete_us": (reads["done"] - start) * 1e6}

    def verify(self, case: ReadCase) -> bool:
        used = case.blocks_per_request * case.requests_in_flight
        return all(bool((region[:used] == self._pattern(i)).all())
                   for i, region in enumerate(self.regions))

    def clear(self) -> None:
        """Zero the reader's regions so ``verify`` sees only data the READs bring."""
        for region in self.regions:
            region.zero_()
        self.torch.cuda.synchronize()

    def drain_notifications(self, expected: int, seconds: float) -> int:
        """Count READ notifications on a P rank until ``expected`` arrive or time runs out."""
        received = 0
        deadline = time.monotonic() + seconds
        while received < expected and time.monotonic() < deadline:
            received += sum(len(m) for m in self.agent.get_new_notifs().values())
            time.sleep(0.001)
        return received

    def run_case(self, case: ReadCase, settings: dict, pass_index: int) -> dict | None:
        dist = self.dist
        active = self.is_reader and self.pair_index < case.active_pairs
        if active:
            self.clear()
        timings, verified, notifications = [], None, 0
        for repeat in range(settings["warmup"] + settings["repeats"]):
            dist.barrier()
            if active:
                timing = self.finish_reads(self.post_reads(
                    case, f"{pass_index}:{case.name}:{repeat}"))
                if repeat >= settings["warmup"]:
                    timings.append(timing)
                if verified is None:
                    verified = self.verify(case)
        dist.barrier()
        if not self.is_reader:
            expected = (case.requests_in_flight * (settings["warmup"] + settings["repeats"])
                        if self.pair_index < case.active_pairs else 0)
            notifications = self.drain_notifications(
                expected, settings["notification_wait_seconds"])
        local = {"rank": self.rank, "reader": active, "timings": timings,
                 "data_verified": verified, "notifications": notifications}
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        if self.rank != 0:
            return None
        readers = [g for g in gathered if g["reader"]]
        per_reader = case.blocks_per_request * case.requests_in_flight * self.layout.block_bytes
        return {
            "pass": pass_index, "case": case.name, "prompt_tokens": case.prompt_tokens,
            "blocks_per_request": case.blocks_per_request,
            "active_pairs": case.active_pairs,
            "requests_in_flight": case.requests_in_flight,
            "descriptors_per_request": case.blocks_per_request * self.layout.num_regions,
            "bytes_per_reader": per_reader, "readers": readers,
            "notifications_received": sum(g["notifications"] for g in gathered),
            "notifications_expected": case.active_pairs * case.requests_in_flight
            * (settings["warmup"] + settings["repeats"]),
        }

    def close(self):
        if self.is_reader:
            self.agent.release_dlist_handle(self.local_dlist)
            self.agent.release_dlist_handle(self.remote_dlist)
        self.agent.deregister_memory(self.registration)
        self.dist.barrier()


def init_process_group(config: dict) -> None:
    """Gloo default process group, which carries barriers and metadata exchange."""
    import torch.distributed as dist

    dist.init_process_group("gloo", timeout=datetime.timedelta(
        seconds=int(config["process_group_timeout_seconds"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    layout = KVRegionLayout.from_config(config["kv_cache"])
    pairs = int(config["tensor_parallel_size"])
    cases = read_cases(config["grid"], layout, pairs)
    settings = {key: int(config[key]) for key in
                ("warmup", "repeats", "notification_wait_seconds", "passes", "shuffle_seed")}
    if args.list_cases:
        for case in cases:
            print(case.name, case.blocks_per_request * layout.num_regions,
                  case.blocks_per_request * case.requests_in_flight * layout.block_bytes)
        print("cases", len(cases), "passes", settings["passes"],
              "measurements", len(cases) * settings["passes"])
        return
    init_process_group(config)
    peer = KVReadPeer(config, layout, pairs)
    output = Path(config["output_dir"])
    rows = []
    if peer.rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "nixl_plan.json").write_text(json.dumps(
            {"config": config, "cases": len(cases), "environment": peer.environment},
            indent=2, default=str))
    for pass_index in range(settings["passes"]):
        for case in pass_order(cases, settings["shuffle_seed"], pass_index):
            record = peer.run_case(case, settings, pass_index)
            if record is not None:
                with (output / "nixl_reads.jsonl").open("a") as log:
                    log.write(json.dumps(record) + "\n")
                rows.append(summarize_case(record))
    if peer.rank == 0:
        write_csv(output / "nixl_summary.csv", rows)
    peer.close()


if __name__ == "__main__":
    main()
