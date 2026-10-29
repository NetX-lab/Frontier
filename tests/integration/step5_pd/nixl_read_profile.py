"""Write the Step NIXL READ cost profile from the KV transfer metrics of a native P/D run.

vLLM's NIXL connector logs, once per logging interval, the number of per-rank READs that
finished in the interval, their mean NIXL transfer duration and their mean payload. The
profile time is the mean duration over every READ of the run, weighted by the READ count of
each interval, so it includes the contention the native workload had. Every logged payload
must equal the per-rank payload Frontier reserves for the transferred prompt prefix.

Usage: nixl_read_profile.py --case h200_32 --server-logs <dir>/nodes --output <json>
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from frontier.attention.memory import AttentionKVCacheSelection, get_attention_kv_transfer_payload
from frontier.attention.step5_memory import resolve_step5_kv_cache_capacity
from frontier.config import ReplicaConfig
from tests.integration.step5_pd.frontier_workload import CASES, DEVICES, MODEL

ENGINE = json.loads((Path(__file__).with_name("canonical_run_h200.json")).read_text())["engine"]
METRICS_LINE = re.compile(r"\[metrics\.py:\d+\] KV Transfer metrics: (.*)$")


def read_intervals(server_logs: Path) -> list[dict[str, float]]:
    intervals = []
    for log in sorted(server_logs.glob("*/server.log")):
        for line in log.read_text(errors="replace").splitlines():
            match = METRICS_LINE.search(line)
            if match:
                fields = dict(field.split("=", 1) for field in match.group(1).split(", "))
                intervals.append({name: float(value) for name, value in fields.items()})
    if not intervals:
        raise ValueError(f"No NIXL KV transfer metrics under {server_logs}")
    return intervals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--server-logs", type=Path, required=True,
                        help="Directory with one <node>/server.log per native node")
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--profile-identity", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    topology = CASES[args.case]
    device, network_device = DEVICES[topology["gpu_type"]]
    tp, dp, block = ENGINE["tensor_parallel_size"], topology["data_parallel_size"], ENGINE["block_size"]
    replica = ReplicaConfig(model_name=MODEL, device=device, network_device=network_device,
                            attn_tensor_parallel_size=tp, attn_dp=dp, moe_tensor_parallel_size=1,
                            moe_expert_parallel_size=tp * dp, attention_backend="STEP5_DSA",
                            kv_cache_dtype="auto")
    capacity = resolve_step5_kv_cache_capacity(
        replica.model_config, block, AttentionKVCacheSelection.from_replica(replica, block_size=block))
    # P computes every full block before the last prompt token; D recomputes the rest.
    prefix = (args.prompt_tokens - 1) // block * block
    payload = get_attention_kv_transfer_payload(
        replica.model_config, capacity, prompt_tokens=prefix, tensor_parallel_size=tp,
        target_reserved_blocks=len(capacity.groups) * (prefix // block))

    intervals = read_intervals(args.server_logs)
    if any(row["Num failed transfers"] for row in intervals):
        raise ValueError("The native run logged failed NIXL transfers")
    logged_mib = {row["Avg MB per transfer"] for row in intervals}
    # vLLM reports the payload in MiB, rounded to three decimals.
    if logged_mib != {round(payload.transfer_bytes_per_rank / 2**20, 3)}:
        raise ValueError(f"Logged payloads {sorted(logged_mib)} MiB differ from the reserved "
                         f"{payload.transfer_bytes_per_rank} B per rank")
    reads = sum(row["Num successful transfers"] for row in intervals)
    time_ms = sum(row["Num successful transfers"] * row["Avg xfer time (ms)"] for row in intervals) / reads
    document = dict(
        schema="nixl_read_cost.v1", model_architecture_profile="step5",
        profile_identity=args.profile_identity, source_device=device, target_device=device,
        source_attention_tp=tp, target_attention_tp=tp, duration_scope="read_to_gpu_ready",
        cache_capacity=asdict(capacity),
        measurements=[dict(num_prefill_tokens=args.prompt_tokens, num_computed_tokens=prefix,
                           transfer_bytes_per_rank=payload.transfer_bytes_per_rank, time_ms=time_ms)],
    )
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(dict(intervals=len(intervals), reads=int(reads), time_ms=time_ms,
                          transfer_bytes_per_rank=payload.transfer_bytes_per_rank,
                          num_computed_tokens=prefix)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
