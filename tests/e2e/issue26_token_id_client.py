"""Send fixed-length token-ID requests for the Issue 26 calibration case."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import aiohttp


def build_token_ids(length: int, seed: int, request_index: int) -> list[int]:
    """Build deterministic ordinary-token IDs without tokenizer round-trips."""
    vocab_size = 151643
    offset = (seed * 104729 + request_index * 15485863) % vocab_size
    return [(offset + position) % vocab_size for position in range(length)]


async def send_request(
    session: aiohttp.ClientSession,
    *,
    url: str,
    model: str,
    request_id: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    scheduled_at: float,
    output_path: Path,
    semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
) -> None:
    delay = scheduled_at - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)

    wall_start_ns = time.time_ns()
    monotonic_start_ns = time.perf_counter_ns()
    dispatch_monotonic_s = time.monotonic()
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    headers = {"X-Request-Id": request_id}

    async with semaphore:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(
                    f"request {request_id} failed with HTTP {response.status}: {body[:500]}"
                )

            first_token_ns: int | None = None
            response_id: str | None = None
            completion_tokens = 0
            async for raw_line in response.content:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                chunk: dict[str, Any] = json.loads(data)
                response_id = response_id or chunk.get("id")
                choices = chunk.get("choices") or []
                if choices and any(
                    choice.get("text") or choice.get("token_ids")
                    for choice in choices
                ) and first_token_ns is None:
                    first_token_ns = time.perf_counter_ns()
                for choice in choices:
                    token_ids = choice.get("token_ids") or []
                    completion_tokens += len(token_ids)

            completed_ns = time.perf_counter_ns()

    if first_token_ns is None:
        raise RuntimeError(f"request {request_id} produced no streamed token")

    record = {
        "request_id": request_id,
        "response_id": response_id,
        "request_arrival_wall_time_ns": wall_start_ns,
        "planned_arrival_monotonic_s": scheduled_at,
        "dispatch_monotonic_s": dispatch_monotonic_s,
        "dispatch_lag_ms": (dispatch_monotonic_s - scheduled_at) * 1000.0,
        "client_first_token_time_ns": wall_start_ns + (first_token_ns - monotonic_start_ns),
        "client_completion_time_ns": wall_start_ns + (completed_ns - monotonic_start_ns),
        "client_ttft_ms": (first_token_ns - monotonic_start_ns) / 1e6,
        "client_e2e_ms": (completed_ns - monotonic_start_ns) / 1e6,
        "prompt_tokens": len(prompt_token_ids),
        "max_tokens": max_tokens,
        "completion_tokens_observed": completion_tokens,
    }
    async with write_lock:
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


async def run(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")

    if args.qps <= 0 or not math.isfinite(args.qps):
        raise ValueError("qps must be finite and positive")
    if args.requests <= 0 or args.warmups < 10:
        raise ValueError("requests must be positive and warmups must be at least ten")

    url = args.base_url.rstrip("/") + "/v1/completions"
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    rng = random.Random(args.seed)
    offsets = [0.0]
    for _ in range(1, args.requests):
        offsets.append(offsets[-1] + rng.expovariate(args.qps))
    trace_path = output_path.with_suffix(".planned_arrivals.csv")
    with trace_path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["request_id", "arrived_at", "num_prefill_tokens", "num_decode_tokens"])
        writer.writerows((f"{args.row}:{i}", offset, args.prefill_tokens, args.decode_tokens)
                         for i, offset in enumerate(offsets))
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=None)
    # Streaming requests may outlive idle pooled connections between replays.
    connector = aiohttp.TCPConnector(force_close=True, limit=args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for replay in range(args.warmups + 1):
            # Drain every complete replay before starting the next phase.
            phase_start = time.monotonic()
            jobs = []
            for index, offset in enumerate(offsets):
                request_id = (f"warmup:{args.row}:r{replay}:{index}"
                              if replay < args.warmups else f"{args.row}:{index}")
                jobs.append(asyncio.create_task(send_request(
                    session,
                    url=url,
                    model=args.model,
                    request_id=request_id,
                    prompt_token_ids=build_token_ids(
                        args.prefill_tokens, args.seed, replay * args.requests + index),
                    max_tokens=args.decode_tokens,
                    scheduled_at=phase_start + offset,
                    output_path=output_path,
                    semaphore=semaphore,
                    write_lock=write_lock,
                )))
            await asyncio.gather(*jobs)
            print(json.dumps({"replay": replay, "completed_requests": len(jobs),
                              "phase_start_monotonic_s": phase_start,
                              "phase_end_monotonic_s": time.monotonic()}), flush=True)

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    formal = [record for record in records if record["request_id"].startswith(f"{args.row}:")]
    expected_ids = {f"{args.row}:{index}" for index in range(args.requests)}
    observed_ids = {record["request_id"] for record in formal}
    if observed_ids != expected_ids or len(formal) != args.requests:
        raise AssertionError(
            f"formal ID join mismatch: expected={len(expected_ids)} observed={len(formal)} unique={len(observed_ids)}"
        )
    if any(record["prompt_tokens"] != args.prefill_tokens for record in formal):
        raise AssertionError("formal prompt token lengths do not match the manifest")
    print(json.dumps({"formal_requests": len(formal), "formal_unique_ids": len(observed_ids)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--row", required=True)
    parser.add_argument("--prefill-tokens", type=int, required=True)
    parser.add_argument("--decode-tokens", type=int, required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--qps", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
