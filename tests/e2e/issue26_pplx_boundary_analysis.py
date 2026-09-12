"""Validate PPLX first-formal batch-boundary artifacts and summarize CUDA spans."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


FORMAL_ROW = "pf4096_dc1024"
TP_RANKS = (0, 1, 2, 3)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalized_request_id(value: str) -> str:
    """Map vLLM's server request id back to the client identity namespace."""
    value = value[5:] if value.startswith("cmpl-") else value
    return value[:-2] if value.endswith("-0") else value


def p90(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute P90 for an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    return statistics.quantiles(ordered, n=10, method="inclusive")[8]


def analyze(run: Path, output: Path, warmups: int) -> dict[str, Any]:
    client_path = run / "client.jsonl"
    clients = read_jsonl(client_path)
    expected_total = (warmups + 1) * 100
    expected_ids = {
        f"{FORMAL_ROW}:{index}" for index in range(100)
    } | {
        f"warmup:{FORMAL_ROW}:r{replay}:{index}"
        for replay in range(warmups)
        for index in range(100)
    }
    observed_ids = {str(row["request_id"]) for row in clients}
    if len(clients) != expected_total or observed_ids != expected_ids:
        raise AssertionError(
            f"client identity mismatch: expected rows={expected_total}, "
            f"observed rows={len(clients)}, unique IDs={len(observed_ids)}")
    if any(
            int(row["prompt_tokens"]) != 4096
            or int(row["max_tokens"]) != 1024
            or int(row["completion_tokens_observed"]) != 1024
            for row in clients):
        raise AssertionError("client prompt/completion predicates failed")

    formal = [
        row for row in clients
        if not str(row["request_id"]).startswith("warmup:")
    ]
    formal_by_id = {str(row["request_id"]): row for row in formal}
    if len(formal) != 100 or len(formal_by_id) != 100:
        raise AssertionError("formal request identity cardinality failed")
    first_formal_client_id = f"{FORMAL_ROW}:0"
    first_formal_response_id = formal_by_id[first_formal_client_id].get(
        "response_id")
    if not first_formal_response_id:
        raise AssertionError("first formal request has no response identity")
    expected_server_request_id = f"{first_formal_response_id}-0"

    paths = sorted(run.glob("server.boundary.dp*.tp*.pp*.jsonl"))
    if not paths:
        raise AssertionError("no PPLX boundary files were produced")

    rank_rows: dict[int, dict[str, Any]] = {}
    for path in paths:
        rows = read_jsonl(path)
        # The source opens one per-rank file during worker initialization. DP1
        # therefore normally leaves an empty file because the selected formal
        # batch belongs to DP0. Any non-empty non-DP0 file is unexpected.
        if not rows:
            continue
        if len(rows) != 1:
            raise AssertionError(f"{path} must contain exactly one row")
        row = rows[0]
        identity = (int(row["dp_rank"]), int(row["tp_rank"]),
                    int(row["pp_rank"]))
        if identity[0] != 0 or identity[1] not in TP_RANKS or identity[2] != 0:
            raise AssertionError(f"unexpected boundary worker {identity}")
        if identity[1] in rank_rows:
            raise AssertionError(f"duplicate TP rank {identity[1]}")
        rank_rows[identity[1]] = row

        request_ids = row.get("request_ids")
        request_tokens = row.get("request_num_tokens")
        if request_ids != [expected_server_request_id]:
            raise AssertionError(
                f"{identity} is not the first formal request: {request_ids} "
                f"!= {[expected_server_request_id]}")
        if request_tokens != [4096]:
            raise AssertionError(f"{identity} request token predicate failed")
        predicates = {
            "batch_size": int(row["batch_size"]) == 1,
            "batch_num_tokens": int(row["batch_num_tokens"]) == 4096,
            "batch_num_prefill_tokens": int(
                row["batch_num_prefill_tokens"]) == 4096,
            "batch_num_decode_tokens": int(row["batch_num_decode_tokens"]) == 0,
            "boundary_mode": row.get("boundary_mode")
            == "first_formal_4096_prefill",
            "event_duration_positive": math.isfinite(
                float(row["cuda_event_elapsed_ms"]))
            and float(row["cuda_event_elapsed_ms"]) > 0.0,
        }
        if not all(predicates.values()):
            raise AssertionError(f"{identity} first-formal predicates failed: {predicates}")
        row["validated_predicates"] = predicates

    if set(rank_rows) != set(TP_RANKS):
        raise AssertionError(f"missing TP ranks: {sorted(set(TP_RANKS) - set(rank_rows))}")
    durations = {
        tp: float(rank_rows[tp]["cuda_event_elapsed_ms"]) for tp in TP_RANKS
    }
    for value in durations.values():
        if not math.isfinite(value) or value <= 0.0:
            raise AssertionError(f"invalid CUDA boundary duration: {value}")
    ordered = list(durations.values())
    median_ms = statistics.median(ordered)
    rank_max_ms = max(ordered)
    rank_min_ms = min(ordered)
    rank_spread_ms = rank_max_ms - rank_min_ms
    report: dict[str, Any] = {
        "status": "PASS",
        "run": str(run),
        "backend": rank_rows[0].get("all2all_backend"),
        "warmup_rounds": warmups,
        "formal_requests": len(formal),
        "client_rows": len(clients),
        "workers": [
            {"dp_rank": 0, "tp_rank": tp, "pp_rank": 0,
             "cuda_event_elapsed_ms": durations[tp],
             "batch_dp_token_counts": rank_rows[tp].get("batch_dp_token_counts"),
             "source_commit": rank_rows[tp].get("source_commit")}
            for tp in TP_RANKS
        ],
        "first_formal_request_id": first_formal_client_id,
        "first_formal_server_request_id": expected_server_request_id,
        "first_formal_predicates": {
            "request_id": expected_server_request_id,
            "batch_size": 1,
            "prefill_tokens": 4096,
            "decode_tokens": 0,
            "dp_rank": 0,
            "tp_ranks": list(TP_RANKS),
            "pp_rank": 0,
        },
        "duration_ms_by_tp": {str(tp): durations[tp] for tp in TP_RANKS},
        "median_ms": median_ms,
        "p90_ms": p90(ordered),
        "rank_max_ms": rank_max_ms,
        "rank_spread_ms": rank_spread_ms,
        "outer_span_delta_ms": rank_spread_ms,
        "event_semantics": rank_rows[0].get("event_semantics"),
        "limits": (
            "CUDA-event envelope from model-forward begin to end on each rank; "
            "it can include queued device work and excludes host wall-clock "
            "gaps. This artifact has no per-op, scheduler, routing, or full "
            "diagnostic outer-span measurements."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=10)
    args = parser.parse_args()
    if args.warmups < 10:
        raise ValueError("--warmups must be at least 10")
    analyze(args.run, args.output, args.warmups)


if __name__ == "__main__":
    main()
