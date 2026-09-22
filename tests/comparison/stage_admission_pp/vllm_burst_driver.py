#!/usr/bin/env python3
"""vLLM ground truth for stage admission of DP ranks under pipeline parallelism.

Runs inside the ``vllm/vllm-openai:v0.10.2`` image on one 4-GPU worker.  Two
subcommands:

``overlay``
    Build the instrumented vLLM package: copy the image's installed ``vllm``
    package (which carries the compiled extensions) and copy every
    ``vllm/**/*.py`` of the ground-truth checkout over it.  The overlay is
    accepted only when the files where the image and the checkout differ are
    exactly the checkout's own changes over its upstream base, listed in
    ``--expected-changes``.

``run``
    Start one ``AsyncLLM`` with DP=2, PP=2, TP=1 (EP for the MoE model), run
    warmup requests, then bursts of prefill-only requests pinned to rank
    ``i mod 2``.  Every request of a burst is added before any output is
    awaited.  The engines are idle between rounds.  Per-forward traces come
    from the checkout's own instrumentation; this script records each
    request's rank, submit and finish times, and the wall/monotonic clock
    offset around each round.

The script imports vLLM only inside ``run`` so that ``overlay`` never loads
the package it is building.
"""

from __future__ import annotations

import argparse
import asyncio
import filecmp
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path


def build_overlay(site_vllm: Path, checkout: Path, destination: Path, expected_changes: Path) -> dict:
    target = destination / "vllm"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(site_vllm, target, symlinks=True)
    differing = []
    for source in sorted((checkout / "vllm").rglob("*.py")):
        relative = source.relative_to(checkout)
        installed = site_vllm.parent / relative
        if not installed.exists() or not filecmp.cmp(source, installed, shallow=False):
            differing.append(str(relative))
        copy_target = destination / relative
        copy_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copy_target)
    expected = sorted(
        line.strip() for line in expected_changes.read_text().splitlines()
        if line.strip().endswith(".py")
    )
    return {
        "site_vllm": str(site_vllm),
        "checkout": str(checkout),
        "overlay": str(target),
        "differing_py_files": differing,
        "expected_py_changes": expected,
        "unexpected": sorted(set(differing) - set(expected)),
        "missing": sorted(set(expected) - set(differing)),
        "accepted": differing == expected,
    }


def write_model_dir(model_config: Path, model_dir: Path) -> dict:
    config = json.loads(model_config.read_text())
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config, indent=1))
    return config


def prompt_token_ids(request_id: str, length: int, vocab_size: int) -> list[int]:
    generator = random.Random(request_id)
    return [generator.randrange(100, vocab_size - 100) for _ in range(length)]


async def run_bursts(args: argparse.Namespace) -> dict:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    output_dir = Path(args.output_dir)
    model_config = write_model_dir(Path(args.model_config), output_dir / "model")
    engine_args = AsyncEngineArgs(
        model=str(output_dir / "model"),
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=args.data_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.prompt_tokens,
        max_num_seqs=args.max_num_seqs,
        block_size=16,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
        disable_log_stats=True,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    sampling = SamplingParams(
        max_tokens=1, ignore_eos=True, temperature=0.0, detokenize=False,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    vocab_size = int(model_config["vocab_size"])
    records: list[dict] = []

    async def wait_until_idle() -> float:
        started = time.monotonic()
        while engine.engine_core.dp_engines_running():
            if time.monotonic() - started > args.idle_timeout_s:
                raise RuntimeError("DP engines did not pause between rounds")
            await asyncio.sleep(0.05)
        await asyncio.sleep(args.idle_gap_s)
        return time.monotonic() - started

    async def burst(label: str, round_index: int, num_requests: int) -> dict:
        offset_before = time.time() - time.monotonic()
        queues = []
        for index in range(num_requests):
            request_id = f"{label}-q{index}"
            rank = index % args.data_parallel_size
            prompt = TokensPrompt(
                prompt_token_ids=prompt_token_ids(request_id, args.prompt_tokens, vocab_size)
            )
            submitted = time.monotonic()
            queue = await engine.add_request(request_id, prompt, sampling, data_parallel_rank=rank)
            queues.append((request_id, index, rank, submitted, queue))
        for request_id, index, rank, submitted, queue in queues:
            output = await queue.get()
            while not output.finished:
                output = await queue.get()
            records.append({
                "request_id": request_id, "burst": label, "round": round_index,
                "index": index, "rank": rank, "submit_monotonic": submitted,
                "finish_monotonic": time.monotonic(),
                "num_prompt_tokens": len(output.prompt_token_ids),
                "num_output_tokens": len(output.outputs[0].token_ids),
                "finish_reason": output.outputs[0].finish_reason,
            })
        offset_after = time.time() - time.monotonic()
        return {"label": label, "round": round_index, "num_requests": num_requests,
                "wall_minus_monotonic_before": offset_before,
                "wall_minus_monotonic_after": offset_after,
                "idle_wait_s": await wait_until_idle()}

    rounds = []
    try:
        rounds.append(await burst("warmup", 0, args.warmups))
        for num_requests in args.bursts:
            for round_index in range(args.rounds):
                rounds.append(await burst(f"b{num_requests}-r{round_index}", round_index, num_requests))
        cache_config = engine.vllm_config.cache_config
        summary = {
            "model_config": args.model_config,
            "num_gpu_blocks": cache_config.num_gpu_blocks,
            "block_size": cache_config.block_size,
            "engine_args": {key: value for key, value in vars(engine_args).items()
                            if isinstance(value, (bool, int, float, str, type(None)))},
            "rounds": rounds,
        }
    finally:
        engine.shutdown()
    (output_dir / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    overlay = commands.add_parser("overlay")
    overlay.add_argument("--site-vllm", type=Path, required=True)
    overlay.add_argument("--checkout", type=Path, required=True)
    overlay.add_argument("--destination", type=Path, required=True)
    overlay.add_argument("--expected-changes", type=Path, required=True)
    overlay.add_argument("--report", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--model-config", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--enable-expert-parallel", action="store_true")
    run.add_argument("--data-parallel-size", type=int, default=2)
    run.add_argument("--pipeline-parallel-size", type=int, default=2)
    run.add_argument("--prompt-tokens", type=int, default=256)
    run.add_argument("--max-num-seqs", type=int, default=4)
    run.add_argument("--max-model-len", type=int, default=512)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    run.add_argument("--bursts", type=int, nargs="+", default=[8, 16])
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--warmups", type=int, default=4)
    run.add_argument("--idle-gap-s", type=float, default=1.0)
    run.add_argument("--idle-timeout-s", type=float, default=60.0)
    args = parser.parse_args(argv)

    if args.command == "overlay":
        report = build_overlay(args.site_vllm, args.checkout, args.destination, args.expected_changes)
        args.report.write_text(json.dumps(report, indent=1))
        print(json.dumps({key: report[key] for key in ("accepted", "unexpected", "missing")}))
        return 0 if report["accepted"] else 3
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    summary = asyncio.run(run_bursts(args))
    Path(args.output_dir, "summary.json").write_text(json.dumps(summary, indent=1))
    print("DRIVER_DONE", json.dumps({"rounds": len(summary["rounds"]),
                                     "num_gpu_blocks": summary["num_gpu_blocks"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
