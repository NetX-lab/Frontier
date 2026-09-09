"""Strict decoder decomposition for SGLang Qwen3.5-family ROCm traces.

These are *in-situ kernel sums*, not CUDA-event microbenchmarks. Keep fused
runtime boundaries intact; do not relabel them as unfused Frontier operators.
Only the single-stream, unfused-collective, separate-shared-expert path is
supported. An unknown kernel/order fails admission instead of disappearing.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

from frontier.validation.batch_record import read_batches, validate_rank_cohorts
from frontier.validation.trace import extract_batch_kernels, interval_union_us
from frontier.validation.report import summarize_profile_perturbation
from frontier.operators.sglang_family import SGLANG_RUNTIME_CONTRACT


CONTRACT = SGLANG_RUNTIME_CONTRACT
_NORMS = {"_gemma_rmsnorm_kernel", "_gemma_fused_add_rmsnorm_kernel"}
_REDUCTIONS = ("quickreduce::allreduce", "cross_device_reduce")
_GEMMS = ("Cijk_Alik_", "_gemm_a16_w16_kernel_")
_EPILOGUE = "Cijk_SB_"
_CORE = {
    "prefill": "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "decode": "fused_recurrent_gated_delta_rule_packed_decode_kernel",
}
_GDN_AUX = (
    "at::native::", "rocprim::", "_causal_conv1d_", "fused_qkv_split_gdn_",
    "fused_gdn_gating_kernel", "l2norm_fwd_kernel", "chunk_local_cumsum_",
    "chunk_gated_delta_rule_", "recompute_w_u_fwd_kernel", "chunk_fwd_kernel_o",
    "fused_recurrent_gated_delta_rule_",
)


def _contains(name, fragments):
    return any(fragment in name for fragment in fragments)


class _Cursor:
    def __init__(self, events):
        self.events = events
        self.index = 0
        self.layer = -1
        self.rows = []

    def name(self):
        return str(self.events[self.index]["name"]) if self.index < len(self.events) else "<end>"

    def take(self, predicate, description):
        if not predicate(self.name()):
            raise ValueError(f"Layer {self.layer}, kernel {self.index}: expected {description}, "
                             f"found {self.name()}")
        self.index += 1

    def gemm(self):
        self.take(lambda n: _contains(n, _GEMMS), "dense GEMM")
        while self.name().startswith(_EPILOGUE):
            self.index += 1

    def emit(self, component, start):
        selected = self.events[start:self.index]
        if not selected:
            raise ValueError(f"Empty component {component}")
        self.rows.append({
            "layer_id": self.layer, "component": component,
            "first_kernel": start, "end_kernel_exclusive": self.index,
            "kernel_count": len(selected),
            "kernel_sum_ms": sum(float(e["dur"]) for e in selected) / 1000,
            "kernel_span_ms": (float(selected[-1]["ts"]) + float(selected[-1]["dur"])
                               - float(selected[0]["ts"])) / 1000,
        })

    def single(self, component, predicate):
        start = self.index
        self.take(predicate, component)
        self.emit(component, start)

    def projection(self, component):
        start = self.index
        self.gemm()
        self.emit(component, start)


def decompose_decoder(events: list[dict], *, phase: str, layer_kinds: list[str]) -> dict:
    """Partition every decoder kernel exactly once, preserving layer identity.

    ``layer_kinds`` must come from the model config, never inferred by counting
    the same trace being validated. Preparation/embedding/logits/sampling are
    explicitly outside this decoder boundary and are not treated as residual
    model overhead or used to correct a prediction.
    """
    if phase not in _CORE or not layer_kinds or set(layer_kinds) - {"gdn", "attention"}:
        raise ValueError("Expected prefill/decode and an explicit hybrid layer schedule")
    if not events:
        raise ValueError("No kernel events")
    streams = {(e.get("args", {}).get("device", e.get("pid")),
                e.get("args", {}).get("stream")) for e in events}
    if len(streams) != 1 or next(iter(streams))[1] is None:
        raise ValueError("Operator decomposition requires one identified GPU stream")
    previous_start = -math.inf
    for event in events:
        ts, dur = float(event["ts"]), float(event["dur"])
        if (event.get("cat") != "kernel" or event.get("ph") != "X" or
                not math.isfinite(ts) or not math.isfinite(dur) or dur <= 0):
            raise ValueError("Invalid GPU kernel event")
        if ts < previous_start:
            raise ValueError("Out-of-order kernels cannot use serial decomposition")
        previous_start = ts

    cursor = _Cursor(events)
    # The initial embedding and its all-reduce must precede the first norm.
    embedding = [i for i, e in enumerate(events)
                 if e["name"] == "_vocab_parallel_embedding_kernel"]
    if len(embedding) != 1:
        raise ValueError("Expected exactly one embedding kernel")
    cursor.index = embedding[0] + 1
    cursor.take(lambda n: _contains(n, _REDUCTIONS), "embedding TP all-reduce")
    decoder_start = cursor.index
    for layer, kind in enumerate(layer_kinds):
        cursor.layer = layer
        cursor.single("input_layernorm", lambda n: n in _NORMS)
        if kind == "gdn":
            start = cursor.index
            cursor.gemm()
            cursor.gemm()
            cursor.emit("gdn_input_projections", start)
            start = cursor.index
            anchors = 0
            while cursor.name() != "_layer_norm_fwd_1pass_kernel":
                anchors += _CORE[phase] in cursor.name()
                cursor.take(lambda n: _contains(n, _GDN_AUX), "GDN core/transform kernel")
            if anchors != 1:
                raise ValueError(f"Layer {layer}: expected one {phase} GDN core, found {anchors}")
            cursor.emit(f"gdn_core_{phase}", start)
            start = cursor.index
            cursor.take(lambda n: n == "_layer_norm_fwd_1pass_kernel", "GDN output norm")
            cursor.gemm()
            cursor.emit("gdn_output_projection", start)
        else:
            start = cursor.index
            cursor.gemm()
            cursor.take(lambda n: n == "_fused_qk_gemma_rmsnorm_gate_kernel", "fused QK norm/gate")
            cursor.emit("attn_pre_proj_qknorm", start)
            cursor.single("attn_rope", lambda n: n == "_triton_mrope_forward_fused")
            cursor.single("attn_kv_cache_write", lambda n: "sglang::store_kvcache<" in n)
            start = cursor.index
            if phase == "prefill":
                cursor.take(lambda n: "FmhaBatchPrefillWithPagedKVCacheKernel" in n,
                            "AITER prefill attention")
            else:
                cursor.take(lambda n: "paged_attention_ll4mi_QKV_mfma16_kernel" in n,
                            "AITER decode attention")
                cursor.take(lambda n: "paged_attention_ll4mi_reduce_kernel" in n,
                            "AITER decode split reduction")
            cursor.emit(f"attn_{phase}", start)
            start = cursor.index
            cursor.take(lambda n: n == "_fused_sigmoid_mul_kernel", "attention output gate")
            cursor.gemm()
            cursor.emit("attn_post_proj_gate", start)
        cursor.single("attention_tp_allreduce", lambda n: _contains(n, _REDUCTIONS))
        cursor.single("post_attention_layernorm", lambda n: n == "_gemma_fused_add_rmsnorm_kernel")
        cursor.projection("shared_expert_gate_up")
        cursor.single("shared_expert_activation", lambda n: "act_and_mul_kernel" in n)
        cursor.projection("shared_expert_down")
        cursor.projection("moe_router_linear")
        cursor.single("moe_routing_topk", lambda n: "topkGatingSoftmax" in n)
        start = cursor.index
        while "aiter::opus_moe_sorting_entry" in cursor.name():
            cursor.index += 1
        cursor.emit("moe_sorting", start)
        start = cursor.index
        for stage in (1, 2):
            quant_start = cursor.index
            while _contains(cursor.name(), ("dynamic_per_group_scaled_quant_kernel",
                                            "mxfp4_moe_sort_kernel", "fused_mx_quant_moe_sort_kernel")):
                cursor.index += 1
            if cursor.index == quant_start:
                raise ValueError(f"Layer {layer}: missing stage {stage} MXFP4 activation quantization")
            cursor.take(lambda n: n.startswith(f"mfma_moe{stage}_"), f"MXFP4 MoE GEMM {stage}")
        cursor.emit("moe_experts_quant_gemm_combine", start)
        cursor.single("shared_gate_sigmoid_mul_add", lambda n: n == "_fused_gate_sigmoid_mul_add_kernel")
        cursor.single("mlp_tp_allreduce", lambda n: _contains(n, _REDUCTIONS))

    decoder_end = cursor.index
    cursor.take(lambda n: n == "_gemma_fused_add_rmsnorm_kernel", "final model norm")
    # Additional decoder work beyond the configured schedule must not silently
    # become 'outside decoder' work.
    if any(e["name"] in _NORMS or "topkGatingSoftmax" in e["name"] or
           _contains(e["name"], tuple(_CORE.values())) for e in events[cursor.index:]):
        raise ValueError("Unexpected decoder kernels after the configured last layer")
    selected = events[decoder_start:decoder_end]
    totals = defaultdict(float)
    for row in cursor.rows:
        totals[row["component"]] += row["kernel_sum_ms"]
    decoder_sum = sum(totals.values())
    busy_ms = interval_union_us(selected) / 1000
    return {
        "contract": CONTRACT, "measurement_type": "KERNEL_ONLY",
        "rank_aggregation": "single_rank_in_situ", "phase": phase,
        "num_layers": len(layer_kinds), "components": cursor.rows,
        "component_totals_ms": dict(totals),
        "decoder_kernel_sum_ms": decoder_sum,
        "decoder_gpu_busy_ms": busy_ms,
        "decoder_kernel_overlap_ms": max(0, decoder_sum - busy_ms),
        "decoder_gpu_span_ms": (float(selected[-1]["ts"]) + float(selected[-1]["dur"])
                                - float(selected[0]["ts"])) / 1000,
        "decoder_kernel_count": len(selected),
        "outside_decoder_kernel_count": len(events) - len(selected),
        "outside_decoder_kernel_sum_ms": sum(float(e["dur"]) for e in
            events[:decoder_start] + events[decoder_end:]) / 1000,
        "boundary_note": "Outside decoder includes prepare, embedding, final norm, logits and "
                         "sampling; it is not a measured full-forward residual.",
    }


def model_layer_kinds(model_config) -> list[str]:
    if model_config.model_type != "qwen3_5_moe_text":
        raise ValueError("Trace contract requires the Qwen3.5-family hybrid MoE architecture")
    return ["gdn" if model_config.is_gdn_layer(layer) else "attention"
            for layer in range(model_config.num_layers)]


def import_capture(capture_dir: Path, *, model_config, rank: int) -> dict:
    manifest = json.loads((capture_dir / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("engine") != "sglang":
        raise ValueError("Requires a complete SGLang capture")
    topology = manifest["topology"]
    if (topology != {"tp": 8, "ep": 1, "pp": 1, "nodes": 1} or rank not in range(8)):
        raise ValueError("This trace contract requires one-node TP=8, EP=1, PP=1")
    if Path(manifest["model_path"]).name != Path(model_config.name).name:
        raise ValueError("Model configuration does not match capture checkpoint")
    records = read_batches(capture_dir / "batches.jsonl")
    validate_rank_cohorts(records, tensor_parallel_size=topology["tp"])
    ledger = {r.batch_id: r for r in records if r.rank == rank and r.profiled}
    rows, seen, sources = [], set(), []
    for path in sorted(capture_dir.glob(f"*-rank{rank}.trace.json.gz")):
        with gzip.open(path, "rt") as stream:
            batches = extract_batch_kernels(json.load(stream))
        sources.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for batch_id, kernels in batches.items():
            if batch_id not in ledger or batch_id in seen:
                raise ValueError(f"Unexpected or duplicate profiled batch {batch_id}")
            record = ledger[batch_id]
            seen.add(batch_id)
            rows.append({"batch_id": batch_id, "rank": rank, "shape": record.to_dict(),
                         **decompose_decoder(kernels, phase=record.phase,
                                             layer_kinds=model_layer_kinds(model_config))})
    if not rows or seen != set(ledger):
        raise ValueError("Trace coverage does not match all profiled rank ledger rows")
    return {"schema_version": 1, "contract": CONTRACT, "manifest": manifest,
            "trace_sources": sources, "batches": rows,
            "profile_perturbation": summarize_profile_perturbation(
                records, tensor_parallel_size=topology["tp"]),
            "limitations": ["Not CUDA-event timings; no native CSV export by relabeling.",
                            "Collectives include rank skew; not isolated bandwidth measurements.",
                            "No expert-routing load vectors; no EP or multi-node extrapolation."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from frontier.profiling.common.model_config import ModelConfig

    result = import_capture(args.capture_dir, model_config=ModelConfig.from_model_name(args.model),
                            rank=args.rank)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"batches": len(result["batches"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
