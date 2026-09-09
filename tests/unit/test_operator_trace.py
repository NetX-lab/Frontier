from copy import deepcopy
from dataclasses import replace

import pytest

from frontier.validation.batch_record import BatchRecord
from frontier.validation.operator_trace import decompose_decoder, model_layer_kinds
from frontier.validation.report import summarize_profile_perturbation


def trace(phase="decode", kinds=("gdn", "attention"), epilogues=True):
    gemm = ["Cijk_Alik_gemm"] + (["Cijk_SB_epilogue"] if epilogues else [])
    ar = "aiter::cross_device_reduce"
    names = ["prepare", "_vocab_parallel_embedding_kernel", ar]
    for kind in kinds:
        names += ["_gemma_fused_add_rmsnorm_kernel"]
        if kind == "gdn":
            names += gemm * 2 + ["at::native::transform", "_causal_conv1d_update_kernel",
                "chunk_gated_delta_rule_fwd_kernel_h_blockdim64" if phase == "prefill"
                else "fused_recurrent_gated_delta_rule_packed_decode_kernel",
                "_layer_norm_fwd_1pass_kernel"] + gemm
        else:
            names += gemm + ["_fused_qk_gemma_rmsnorm_gate_kernel", "_triton_mrope_forward_fused",
                             "sglang::store_kvcache<512>"]
            names += (["FmhaBatchPrefillWithPagedKVCacheKernel"] if phase == "prefill" else
                      ["paged_attention_ll4mi_QKV_mfma16_kernel", "paged_attention_ll4mi_reduce_kernel"])
            names += ["_fused_sigmoid_mul_kernel"] + gemm
        names += [ar, "_gemma_fused_add_rmsnorm_kernel"] + gemm + ["act_and_mul_kernel"]
        names += gemm * 2 + ["topkGatingSoftmax", "aiter::opus_moe_sorting_entry<P0>",
                            "aiter::opus_moe_sorting_entry<P23>",
                            "fused_mx_quant_moe_sort_kernel", "mfma_moe1_silu_mul",
                            "fused_mx_quant_moe_sort_kernel", "mfma_moe2_combine",
                            "_fused_gate_sigmoid_mul_add_kernel", ar]
    names += ["_gemma_fused_add_rmsnorm_kernel", "logits", "sampling"]
    return [dict(name=n, ts=i * 10, dur=5, cat="kernel", ph="X", args={"device": 0, "stream": 0})
            for i, n in enumerate(names)]


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("epilogues", [False, True])
@pytest.mark.parametrize("kinds", [("gdn",), ("attention",), ("gdn", "gdn", "gdn", "attention")])
def test_partition_conserves_every_decoder_kernel_once(phase, epilogues, kinds):
    events = trace(phase, kinds, epilogues)
    result = decompose_decoder(events, phase=phase, layer_kinds=list(kinds))
    rows = result["components"]
    indices = [i for r in rows for i in range(r["first_kernel"], r["end_kernel_exclusive"])]
    assert indices == list(range(3, len(events) - 3))
    assert result["outside_decoder_kernel_count"] == 6
    assert result["decoder_kernel_sum_ms"] == pytest.approx(len(indices) * .005)
    assert sum(result["component_totals_ms"].values()) == pytest.approx(result["decoder_kernel_sum_ms"])
    assert sum(r["component"] == "mlp_tp_allreduce" for r in rows) == len(kinds)
    assert sum(r["component"] == "shared_gate_sigmoid_mul_add" for r in rows) == len(kinds)


@pytest.mark.parametrize("kernel", ["_gemma_fused_add_rmsnorm_kernel", "Cijk_Alik_gemm",
    "fused_recurrent_gated_delta_rule_packed_decode_kernel", "_layer_norm_fwd_1pass_kernel",
    "_fused_qk_gemma_rmsnorm_gate_kernel", "_triton_mrope_forward_fused", "sglang::store_kvcache<512>",
    "paged_attention_ll4mi_QKV_mfma16_kernel", "paged_attention_ll4mi_reduce_kernel",
    "_fused_sigmoid_mul_kernel", "act_and_mul_kernel", "topkGatingSoftmax",
    "fused_mx_quant_moe_sort_kernel", "mfma_moe1_silu_mul", "mfma_moe2_combine",
    "_fused_gate_sigmoid_mul_add_kernel", "aiter::cross_device_reduce"])
def test_unknown_operator_cannot_be_silently_dropped(kernel):
    events = trace()
    next(e for e in events if e["name"] == kernel)["name"] = "unknown_fused_replacement"
    with pytest.raises(ValueError):
        decompose_decoder(events, phase="decode", layer_kinds=["gdn", "attention"])


def test_order_phase_schedule_and_stream_are_strict():
    original = trace()
    for kinds in (["attention", "gdn"], ["gdn"], ["gdn", "attention", "gdn"]):
        with pytest.raises(ValueError):
            decompose_decoder(original, phase="decode", layer_kinds=kinds)
    with pytest.raises(ValueError, match="core"):
        decompose_decoder(original, phase="prefill", layer_kinds=["gdn", "attention"])
    for mutation in (lambda e: e[6]["args"].update(stream=1),
                     lambda e: e[6]["args"].update(device=1),
                     lambda e: e[6].update(ts=-1),
                     lambda e: e[6].update(dur=float("nan"))):
        events = deepcopy(original)
        mutation(events)
        with pytest.raises(ValueError):
            decompose_decoder(events, phase="decode", layer_kinds=["gdn", "attention"])


def test_overlap_is_reported_without_correcting_kernel_sums():
    events = trace()
    events[10]["dur"] = 12
    result = decompose_decoder(events, phase="decode", layer_kinds=["gdn", "attention"])
    assert result["decoder_kernel_overlap_ms"] == pytest.approx(.002)


def test_layer_schedule_is_model_owned():
    from frontier.profiling.common.model_config import ModelConfig
    config = ModelConfig.from_model_name("Qwen3.8-2.4T-A95B-Quark-MXFP4")
    schedule = model_layer_kinds(config)
    assert schedule == ["gdn", "gdn", "gdn", "attention"] * 23


def timing(**updates):
    fields = dict(batch_id="b", rank=0, phase="decode", request_ids=("a",),
                  query_lens=(1,), context_lens=(1024,), prefill_mask=(False,),
                  graph_mode="FULL", capture_size=1, forward_gpu_ms=10., step=1)
    fields.update(updates)
    return BatchRecord(**fields)


def test_quality_matches_exact_early_step_and_uses_max_rank_not_sum():
    baseline = [timing(batch_id=f"b{i}", repetition=i, forward_gpu_ms=t)
                for i, t in enumerate((9., 10., 11.))]
    records = baseline + [timing(batch_id="late", step=8, context_lens=(1031,), forward_gpu_ms=3.),
                          timing(batch_id="trace", profiled=True, forward_gpu_ms=11.)]
    records += [replace(r, rank=1, forward_gpu_ms=r.forward_gpu_ms * 2) for r in records]
    report = summarize_profile_perturbation(records, tensor_parallel_size=2)
    row = report["batches"][0]
    assert row["baseline_forward_gpu_median_ms"] == 20.
    assert row["profiled_forward_gpu_ms"] == 22.
    assert row["signed_change_pct"] == pytest.approx(10.)
    assert not report["all_pass_latency_check"]


@pytest.mark.parametrize("change", [{"context_lens": (2048,)}, {"capture_size": 2}, {"step": 3}])
def test_quality_rejects_unmatched_trace_shapes(change):
    with pytest.raises(ValueError, match="No matching"):
        summarize_profile_perturbation([timing(), timing(batch_id="trace", profiled=True, **change)],
                                      tensor_parallel_size=1)


def test_quality_empty_missing_timing_and_negative_perturbation():
    assert not summarize_profile_perturbation([timing()], tensor_parallel_size=1)["all_pass_latency_check"]
    with pytest.raises(ValueError, match="timing"):
        summarize_profile_perturbation([timing(forward_gpu_ms=None)], tensor_parallel_size=1)
    report = summarize_profile_perturbation([
        timing(), timing(batch_id="trace", profiled=True, forward_gpu_ms=8.)], tensor_parallel_size=1)
    assert not report["all_pass_latency_check"]  # faster profiles are suspect too
    with pytest.raises(ValueError, match="threshold"):
        summarize_profile_perturbation([timing()], tensor_parallel_size=1, max_absolute_change_pct=float("nan"))


def test_quality_requires_identical_fixed_input_trajectories():
    baseline = timing(decode_input_sha256="a" * 64)
    for digest in (None, "b" * 64):
        with pytest.raises(ValueError, match="No matching"):
            summarize_profile_perturbation([baseline, timing(batch_id="trace", profiled=True,
                decode_input_sha256=digest)], tensor_parallel_size=1)
    result = summarize_profile_perturbation([baseline, timing(batch_id="trace", profiled=True,
        decode_input_sha256="a" * 64)], tensor_parallel_size=1)
    assert result["batches"][0]["fixed_decode_inputs"]
