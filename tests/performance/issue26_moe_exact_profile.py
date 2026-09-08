"""Measure the current case's exact uniform MoE layouts and expert components."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from frontier.operators.typed_contracts import serialize_typed_operator_contract_column
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.moe import moe_vllm_kernel as kernel
from frontier.profiling.moe.moe_impl import uniform_topk
from frontier.profiling.moe.moe_wrapper import MoEWrapper
from frontier.profiling.moe.main import (
    _attach_moe_output_metadata,
    _resolve_model_arch_for_metadata,
    _resolve_quant_signature_for_metadata,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts


def measure(step, samples, prefix=None):
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(samples)]
    times = []
    for start, end in pairs:
        if prefix is not None:
            prefix()
        start.record()
        step()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return {"samples_ms": times, "median_ms": float(np.median(times)),
            "min_ms": min(times), "max_ms": max(times)}


def expert_components(wrapper, routing, tokens, samples):
    hidden, width = wrapper.hidden_dim, wrapper.expert_hidden_dim
    top_k, experts = wrapper.router_topk, wrapper.num_experts_per_device
    a = torch.randn(tokens, hidden, device="cuda", dtype=wrapper._dtype) * 0.1
    w1 = torch.randn(experts, 2 * width, hidden, device="cuda", dtype=a.dtype) * 0.1
    w2 = torch.randn(experts, hidden, width, device="cuda", dtype=a.dtype) * 0.1
    config = kernel.try_get_optimal_moe_config(
        w1.shape, w2.shape, top_k, kernel.get_config_dtype_str(a.dtype), tokens
    )
    sorted_ids, block_experts, padded = kernel.moe_align_block_size(
        routing["topk_ids"], config["BLOCK_SIZE_M"], wrapper.num_experts,
        expert_map=routing["expert_map"],
    )
    workspace = torch.empty(tokens * top_k * max(2 * width, hidden), device="cuda", dtype=a.dtype)
    cache1 = workspace[:tokens * top_k * 2 * width].view(tokens, top_k, 2 * width)
    cache2 = workspace[:tokens * top_k * hidden].view(tokens, top_k, hidden)
    activated = torch.empty(tokens * top_k, width, device="cuda", dtype=a.dtype)
    output = torch.empty_like(a)
    common = dict(topk_weights=routing["topk_weights"], sorted_token_ids=sorted_ids,
                  expert_ids=block_experts, num_tokens_post_padded=padded, config=config)

    def full():
        kernel._run_fused_moe_iteration(
            a, w1, w2, cache1, cache2, activated, output,
            routing["topk_weights"], sorted_ids, block_experts, padded,
            top_k, config, None,
        )

    full()
    reference = fused_experts(a, w1, w2, routing["topk_weights"], routing["topk_ids"],
                             inplace=False, global_num_experts=wrapper.num_experts,
                             expert_map=routing["expert_map"])
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    block_size = config["BLOCK_SIZE_M"]
    block_count = int(padded.item()) // block_size
    local_blocks = block_experts[:block_count]
    valid_blocks = local_blocks[local_blocks >= 0]
    receipt = {
        "config": config, "output_bitwise_equal": True,
        "global_assignments": tokens * top_k,
        "local_expert_counts": routing["expert_token_counts"],
        "global_padded_assignments": int(padded.item()),
        "local_padded_assignments": int(valid_blocks.numel()) * block_size,
        "local_blocks_per_expert": torch.bincount(valid_blocks.long(), minlength=experts).cpu().tolist(),
        "input_shape": list(a.shape), "w1_shape": list(w1.shape), "w2_shape": list(w2.shape),
        "weight_std": 0.1, "dtype": str(a.dtype),
    }
    # These independent loops explain coverage; their sum is not a full-span estimate.
    # Components use distinct W1/W2 destinations so repeated reduction cannot overwrite activation input.
    component_cache1 = cache1.clone()
    component_cache2 = cache2.clone()
    steps = {
        "full_expert_path": full,
        "w1": lambda: kernel._invoke_kernel(A=a, B=w1, C=component_cache1,
                                             mul_routed_weight=False, top_k=top_k, **common),
        "activation": lambda: torch.ops._C.silu_and_mul(activated, component_cache1.view(-1, 2 * width)),
        "w2": lambda: kernel._invoke_kernel(A=activated, B=w2, C=component_cache2,
                                             mul_routed_weight=True, top_k=1, **common),
        "reduction": lambda: kernel.ops.moe_sum(component_cache2, output),
    }
    receipt["timings"] = {}
    for context, prefix in [("standalone", None),
                            ("prefill_hot", lambda: wrapper._run_prefill_hot_gating_prefix(a))]:
        receipt["timings"][context] = {
            name: measure(step, samples, prefix) for name, step in steps.items()
        }
    return receipt


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen3-a3b-30b-moe")
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--tokens", type=int, nargs="+", default=[4095, 4096, 4097])
    parser.add_argument("--ep-ranks", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = ModelConfig.from_model_name(args.model)
    wrapper = MoEWrapper(model, 1, args.ep_size, "cuda_event", 0, str(args.output_dir),
                         routing_runtime_path="uniform_topk", gating_runtime_context="prefill_hot")
    rows, receipts = [], []
    for tokens in args.tokens:
        hidden = torch.empty(tokens, wrapper.hidden_dim, device="cuda", dtype=wrapper._dtype)
        logits = torch.empty(tokens, wrapper.num_experts, device="cuda", dtype=wrapper._dtype)
        weights, ids, _ = uniform_topk(hidden, logits, wrapper.router_topk)
        global_counts = torch.bincount(ids.flatten().long(), minlength=wrapper.num_experts)
        for rank in args.ep_ranks:
            start = rank * wrapper.num_experts_per_device
            expert_map = torch.full((wrapper.num_experts,), -1, device="cuda", dtype=torch.int32)
            expert_map[start:start + wrapper.num_experts_per_device] = torch.arange(
                wrapper.num_experts_per_device, device="cuda")
            routing = dict(topk_weights=weights, topk_ids=ids,
                           expert_token_counts=global_counts[start:start + wrapper.num_experts_per_device].cpu().tolist(),
                           global_num_experts=wrapper.num_experts, expert_map=expert_map)
            for seed in args.seeds:
                torch.manual_seed(seed)
                gating = wrapper.profile_gating(tokens)
                shuffle = wrapper.profile_shuffling(tokens, seed=seed, routing_inputs=routing)
                gemm = wrapper.profile_grouped_gemm(tokens, seed=seed, routing_inputs=routing)
                times = {**gating["time_stats"], **shuffle["time_stats"], **gemm["time_stats"]}
                row = wrapper._build_profile_result(times, num_tokens=tokens)
                row.update({key: value for key, value in gemm.items() if key not in row})
                row.update(measurement_type="cuda_event", profiling_ep_rank=rank,
                           calibration_split="holdout" if tokens not in [4096, 4097] else "anchor")
                rows.append(row)
                receipt = expert_components(wrapper, routing, tokens, args.samples)
                receipt.update(num_tokens=tokens, ep_rank=rank, seed=seed,
                               calibration_split=row["calibration_split"])
                receipts.append(receipt)
                print(json.dumps({"num_tokens": tokens, "ep_rank": rank, "seed": seed,
                                  "local_counts": routing["expert_token_counts"],
                                  "full_path_ms": receipt["timings"]["standalone"]["full_expert_path"]["median_ms"]}), flush=True)
                (args.output_dir / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
                frame = pd.DataFrame(rows)
                frame = pd.json_normalize(frame["time_stats"]).add_prefix("time_stats.").join(frame.drop(columns="time_stats"))
                frame = _attach_moe_output_metadata(
                    serialize_typed_operator_contract_column(frame),
                    precision_str=ModelConfig._dtype_to_str(model.dtype),
                    model_arch=_resolve_model_arch_for_metadata(model),
                    model_architecture_profile=model.get_model_architecture_profile().profile_id,
                    quant_signature=_resolve_quant_signature_for_metadata(model),
                    measurement_type="cuda_event",
                )
                frame.to_csv(args.output_dir / "moe.csv", index=False)


if __name__ == "__main__":
    main()
