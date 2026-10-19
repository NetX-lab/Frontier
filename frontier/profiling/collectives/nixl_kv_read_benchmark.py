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
flight. Each repeat starts after a CPU barrier; a D rank times the descriptor
preparation, the post, and the wait until every READ reports ``DONE``. Rank 0
appends one JSON line per case to ``<output_dir>/nixl_reads.jsonl`` and writes
``nixl_summary.csv``.
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


def read_cases(grid: dict, layout: KVRegionLayout, tensor_parallel_size: int) -> list[ReadCase]:
    cases = []
    for tokens in grid["prompt_tokens"]:
        blocks = -(-int(tokens) // layout.block_size)
        for pairs in grid["active_pairs"]:
            if not 0 < pairs <= tensor_parallel_size:
                raise ValueError(f"active_pairs={pairs} outside 1..{tensor_parallel_size}")
            for in_flight in grid["requests_in_flight"]:
                if blocks * in_flight > layout.num_gpu_blocks:
                    raise ValueError(f"{in_flight} requests of {blocks} blocks exceed "
                                     f"{layout.num_gpu_blocks} GPU blocks")
                cases.append(ReadCase(f"t{tokens}_pairs{pairs}_inflight{in_flight}",
                                      int(tokens), blocks, int(pairs), int(in_flight)))
    return cases


def summarize_case(record: dict) -> dict:
    """Critical-path statistics of one case from the active D ranks' repeats."""
    per_rank = [r["timings"] for r in record["readers"]]
    complete = [max(t["complete_us"] for t in repeat) for repeat in zip(*per_rank)]
    prep = [max(t["prep_us"] for t in repeat) for repeat in zip(*per_rank)]
    median = statistics.median(complete)
    total_bytes = record["bytes_per_reader"] * len(per_rank)
    return {
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
        "per_reader_gbps": record["bytes_per_reader"] / median / 1e3,
        "aggregate_gbps": total_bytes / median / 1e3,
        "data_verified": all(r["data_verified"] for r in record["readers"]),
        "notifications_received": record["notifications_received"],
        "notifications_expected": record["notifications_expected"],
    }


class _Peer:
    def __init__(self, config: dict, layout: KVRegionLayout):
        import torch
        import torch.distributed as dist
        from nixl._api import nixl_agent, nixl_agent_config

        self.torch, self.dist = torch, dist
        self.layout = layout
        self.tp = int(config["tensor_parallel_size"])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        if int(os.environ["WORLD_SIZE"]) != 2 * self.tp:
            raise ValueError("WORLD_SIZE must be twice the TP size (one P and one D node)")
        self.is_reader = self.rank >= self.tp
        self.tp_rank = self.rank % self.tp
        dist.init_process_group("gloo", timeout=datetime.timedelta(
            seconds=int(config["process_group_timeout_seconds"])))
        torch.cuda.set_device(self.local_rank)
        # One tensor per region, as each owner layer has its own cache tensor.
        self.regions = [torch.empty((layout.num_gpu_blocks, length), dtype=torch.uint8,
                                    device="cuda") for length in layout.region_block_bytes]
        for index, region in enumerate(self.regions):
            region.fill_(0 if self.is_reader else self._pattern(index))
        torch.cuda.synchronize()
        role = "decode" if self.is_reader else "prefill"
        self.agent = nixl_agent(f"{role}-{self.tp_rank}-{os.getpid()}", nixl_agent_config(
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
            source = everyone[self.tp_rank]
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

    def _read_once(self, case: ReadCase, repeat: int) -> dict:
        ids = [descriptor_ids(self.layout.num_regions, self.layout.num_gpu_blocks, blocks)
               for blocks in request_block_ids(case)]
        start = time.perf_counter()
        requests = [self.agent.make_prepped_xfer(
            "READ", self.local_dlist, desc_ids, self.remote_dlist, desc_ids,
            notif_msg=f"{case.name}:{repeat}:{i}".encode()) for i, desc_ids in enumerate(ids)]
        prepared = time.perf_counter()
        states = [self.agent.transfer(request) for request in requests]
        posted = time.perf_counter()
        pending = {i for i, state in enumerate(states) if state != "DONE"}
        while pending:
            for i in list(pending):
                state = self.agent.check_xfer_state(requests[i])
                if state == "ERR":
                    raise RuntimeError(f"NIXL READ failed in {case.name}")
                if state == "DONE":
                    pending.discard(i)
        done = time.perf_counter()
        for request in requests:
            self.agent.release_xfer_handle(request)
        return {"prep_us": (prepared - start) * 1e6, "post_us": (posted - prepared) * 1e6,
                "complete_us": (done - start) * 1e6}

    def _verify(self, case: ReadCase) -> bool:
        used = case.blocks_per_request * case.requests_in_flight
        return all(bool((region[:used] == self._pattern(i)).all())
                   for i, region in enumerate(self.regions))

    def run_case(self, case: ReadCase, settings: dict) -> dict | None:
        torch, dist = self.torch, self.dist
        active = self.is_reader and self.tp_rank < case.active_pairs
        if active:
            for region in self.regions:
                region.zero_()
            torch.cuda.synchronize()
        timings, verified, notifications = [], None, 0
        for repeat in range(settings["warmup"] + settings["repeats"]):
            dist.barrier()
            if active:
                timing = self._read_once(case, repeat)
                if repeat >= settings["warmup"]:
                    timings.append(timing)
                if verified is None:
                    verified = self._verify(case)
        dist.barrier()
        if not self.is_reader:
            deadline = time.monotonic() + settings["notification_wait_seconds"]
            expected = (case.requests_in_flight * (settings["warmup"] + settings["repeats"])
                        if self.tp_rank < case.active_pairs else 0)
            while notifications < expected and time.monotonic() < deadline:
                notifications += sum(len(m) for m in self.agent.get_new_notifs().values())
                time.sleep(0.001)
        local = {"rank": self.rank, "reader": active, "timings": timings,
                 "data_verified": verified, "notifications": notifications}
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        if self.rank != 0:
            return None
        readers = [g for g in gathered if g["reader"]]
        per_reader = case.blocks_per_request * case.requests_in_flight * self.layout.block_bytes
        return {
            "case": case.name, "prompt_tokens": case.prompt_tokens,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    layout = KVRegionLayout.from_config(config["kv_cache"])
    cases = read_cases(config["grid"], layout, int(config["tensor_parallel_size"]))
    if args.list_cases:
        for case in cases:
            print(case.name, case.blocks_per_request * layout.num_regions,
                  case.blocks_per_request * case.requests_in_flight * layout.block_bytes)
        print("cases", len(cases))
        return
    settings = {key: int(config[key]) for key in
                ("warmup", "repeats", "notification_wait_seconds")}
    peer = _Peer(config, layout)
    output = Path(config["output_dir"])
    rows = []
    if peer.rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "nixl_plan.json").write_text(json.dumps(
            {"config": config, "cases": len(cases), "environment": peer.environment},
            indent=2, default=str))
    for case in cases:
        record = peer.run_case(case, settings)
        if record is not None:
            with (output / "nixl_reads.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")
            rows.append(summarize_case(record))
    if peer.rank == 0:
        import csv

        with (output / "nixl_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    peer.close()


if __name__ == "__main__":
    main()
