"""Measure the reached vLLM TP/DP primitives without loading model weights.

Run with torchrun --nproc-per-node=8 in the approved vLLM image. Each sample
times an eager block; synchronization and correctness checks are outside it.
The result is an event span including submission gaps, not pure wire latency.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


def describe(args):
    tokens = args.dp_tokens
    return {
        "tp_size": args.tp_size,
        "dp_tokens": tokens,
        "world_size": args.tp_size * len(tokens),
        "tp_groups": [list(range(i * args.tp_size, (i + 1) * args.tp_size))
                      for i in range(len(tokens))],
        "dp_groups": [[i * args.tp_size + t for i in range(len(tokens))]
                      for t in range(args.tp_size)],
        "dtype": "bfloat16",
        "hidden_size": args.hidden_size,
        "router_experts": args.experts,
        "local_hidden_bytes": [t * args.hidden_size * 2 for t in tokens],
        "local_router_bytes": [t * args.experts * 2 for t in tokens],
        "global_hidden_bytes": sum(tokens) * args.hidden_size * 2,
        "global_router_bytes": sum(tokens) * args.experts * 2,
        "timing": "CUDA events around eager blocks; no per-call synchronize",
        "warmup_calls": args.warmup,
        "calls_per_sample": args.calls,
        "samples": args.repeats,
    }


def measure(name, operation, args, torch, dist, cpu_group):
    for _ in range(args.warmup):
        output = operation()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True),
               torch.cuda.Event(enable_timing=True)) for _ in range(args.repeats)]
    rows = []
    for start, end in events:
        dist.barrier(group=cpu_group)
        submit_start = time.monotonic_ns()
        start.record()
        for _ in range(args.calls):
            output = operation()
        end.record()
        submit_end = time.monotonic_ns()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        if not 0 < elapsed < float("inf"):
            raise ValueError(f"Invalid event span for {name}: {elapsed}")
        rows.append({"block_ms": elapsed, "per_call_ms": elapsed / args.calls,
                     "submit_start_monotonic_ns": submit_start,
                     "submit_end_monotonic_ns": submit_end})
    return {"operation": name, "samples": rows,
            "median_per_call_ms": statistics.median(r["per_call_ms"] for r in rows),
            "min_per_call_ms": min(r["per_call_ms"] for r in rows),
            "max_per_call_ms": max(r["per_call_ms"] for r in rows)}


def run(args):
    import torch
    import torch.distributed as dist
    import vllm
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment, destroy_model_parallel, get_dp_group,
        get_ep_group, get_tp_group, get_world_group, init_distributed_environment,
        initialize_model_parallel)
    from vllm.forward_context import set_forward_context

    contract = describe(args)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != contract["world_size"]:
        raise ValueError("torchrun world size differs from the declared TP/DP layout")
    torch.cuda.set_device(local_rank)
    # Initialize the global torchrun group before a DP config can rebase ranks.
    init_distributed_environment(world_size=contract["world_size"], rank=rank,
                                 local_rank=local_rank)
    config = VllmConfig(parallel_config=ParallelConfig(
        tensor_parallel_size=args.tp_size, data_parallel_size=len(args.dp_tokens),
        data_parallel_rank=rank // args.tp_size, enable_expert_parallel=True,
        disable_custom_all_reduce=False))
    with set_current_vllm_config(config):
        initialize_model_parallel(tensor_model_parallel_size=args.tp_size)
        tp, dp, ep = get_tp_group(), get_dp_group(), get_ep_group()
        if tp.ranks != contract["tp_groups"][rank // args.tp_size]:
            raise ValueError("TP group mismatch")
        if dp.ranks != contract["dp_groups"][rank % args.tp_size]:
            raise ValueError("DP group mismatch")
        manager = ep.device_communicator.all2all_manager
        if type(manager).__name__ != "NaiveAll2AllManager":
            raise ValueError("This case requires VLLM_ALL2ALL_BACKEND=naive")
        device = torch.device("cuda", local_rank)
        local_tokens = args.dp_tokens[dp.rank_in_group]
        hidden = torch.full((local_tokens, args.hidden_size), dp.rank_in_group + 1,
                            device=device, dtype=torch.bfloat16)
        router = torch.full((local_tokens, args.experts), dp.rank_in_group + 1,
                            device=device, dtype=torch.bfloat16)
        partial = torch.full((sum(args.dp_tokens), args.hidden_size), rank + 1,
                             device=device, dtype=torch.bfloat16)
        tp_input = torch.full_like(hidden, rank + 1)
        counts = torch.tensor(args.dp_tokens, dtype=torch.int32)
        cumulative = counts.cumsum(0)
        communicator = tp.device_communicator
        custom = communicator.ca_comm
        custom_eligible = custom is not None and custom.should_custom_ar(tp_input)
        implementation = {
            "tp_custom_eligible": custom_eligible,
            "tp_custom_max_bytes": getattr(custom, "max_size", None),
            "tp_custom_disabled": getattr(custom, "disabled", None),
            "tp_pynccl_disabled": communicator.pynccl_comm.disabled,
            "dp_pynccl_disabled": dp.device_communicator.pynccl_comm.disabled,
            "tp_symm_mem_present": communicator.symm_mem_comm is not None,
            "naive_manager": type(manager).__name__,
        }
        if communicator.pynccl_comm.disabled or dp.device_communicator.pynccl_comm.disabled:
            raise ValueError("PyNCCL is disabled; the benchmark would use a different primitive")
        if communicator.symm_mem_comm is not None:
            raise ValueError("The frozen case does not enable symmetric-memory allreduce")
        with torch.inference_mode(), set_forward_context(None, config, num_tokens=local_tokens,
                                 num_tokens_across_dp=counts):
            operations = {
                "tp_runtime_local": lambda: tp.all_reduce(tp_input),
                "tp_pynccl_local": lambda: communicator.pynccl_comm.all_reduce(tp_input),
                "dp_runtime_global": lambda: dp.all_reduce(partial),
                "naive_hidden_multicast": lambda: manager.naive_multicast(hidden, cumulative),
                "naive_router_multicast": lambda: manager.naive_multicast(router, cumulative),
                "naive_dispatch": lambda: ep.dispatch(hidden, router),
                "naive_combine": lambda: ep.combine(partial),
                "naive_combine_post_tp": lambda: tp.all_reduce(ep.combine(partial)),
            }
            if custom_eligible:
                # All members of a TP group take the same branch. Other TP
                # groups still participate in the common case sequence below.
                implementation["tp_runtime_selected"] = "custom_all_reduce"
            else:
                implementation["tp_runtime_selected"] = "pynccl_all_reduce"
            result = operations["tp_runtime_local"]()
            torch.testing.assert_close(result, torch.full_like(result, sum(r + 1 for r in tp.ranks)))
            dispatched, dispatched_router = operations["naive_dispatch"]()
            start = 0
            for source, count in enumerate(args.dp_tokens):
                torch.testing.assert_close(dispatched[start:start + count],
                                           torch.full_like(dispatched[start:start + count], source + 1))
                torch.testing.assert_close(dispatched_router[start:start + count],
                                           torch.full_like(dispatched_router[start:start + count], source + 1))
                start += count
            result = operations["naive_combine"]()
            torch.testing.assert_close(result, torch.full_like(result, sum(r + 1 for r in dp.ranks)))
            result = operations["naive_combine_post_tp"]()
            expected = sum(range(1, contract["world_size"] + 1))
            torch.testing.assert_close(result, torch.full_like(result, expected))
            rows = [measure(name, operation, args, torch, dist,
                            get_world_group().cpu_group)
                    for name, operation in operations.items()]
        version = subprocess.check_output(
            ["git", "-C", str(Path(vllm.__file__).resolve().parent.parent), "rev-parse", "HEAD"],
            text=True).strip()
        record = {"rank": rank, "local_rank": local_rank, "hostname": platform.node(),
                  "tp_ranks": tp.ranks, "dp_ranks": dp.ranks, "ep_ranks": ep.ranks,
                  "gpu": torch.cuda.get_device_name(device), "python": platform.python_version(),
                  "torch": torch.__version__, "cuda": torch.version.cuda,
                  "vllm_commit": version, "contract": contract,
                  "implementation": implementation, "correctness": "PASS",
                  "measurements": rows,
                  "environment": {key: value for key, value in os.environ.items()
                                  if key.startswith("NCCL_") or key in (
                                      "VLLM_ALL2ALL_BACKEND", "VLLM_ALLREDUCE_USE_SYMM_MEM")}}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank_{rank}.json").write_text(json.dumps(record, indent=2) + "\n")
        dist.barrier(group=get_world_group().cpu_group)
        destroy_model_parallel()
    destroy_distributed_environment()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--dp-tokens", type=int, nargs="+", default=[4096, 1])
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--calls", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.tp_size, *args.dp_tokens, args.hidden_size, args.experts,
           args.warmup, args.calls, args.repeats) < 1:
        parser.error("Dimensions and measurement counts must be positive")
    if args.describe:
        print(json.dumps(describe(args), indent=2))
    elif args.output is None:
        parser.error("--output is required for GPU execution")
    else:
        run(args)


if __name__ == "__main__":
    main()
