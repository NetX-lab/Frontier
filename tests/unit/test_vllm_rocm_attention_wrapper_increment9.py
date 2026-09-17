"""CPU contracts for the explicit vLLM ROCm attention backend."""

from contextlib import contextmanager
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from frontier.profiling.attention.backends import AttentionBackend, get_attention_wrapper
from frontier.profiling.attention.backends.vllm_rocm_attention_wrapper import (
    VllmRocmAttentionWrapper,
    build_rocm_sequence_plan,
)
from frontier.profiling.attention.sequence_proxy import SequenceMetadataProxy
from frontier.profiling.common.constants import OperationMetrics
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_config import ParallelConfig


def test_rocm_sequence_plan_orders_prefill_then_decode_and_maps_slots() -> None:
    prefill = SequenceMetadataProxy(
        is_prompt=True,
        total_len=20,
        processed_len=16,
        block_table=[3, 4],
    )
    decode = SequenceMetadataProxy(
        is_prompt=False,
        total_len=18,
        processed_len=17,
        block_table=[8, 9],
    )
    plan = build_rocm_sequence_plan([prefill, decode], block_size=16)
    assert plan.query_lengths == [4, 1]
    assert plan.sequence_lengths == [20, 18]
    assert plan.block_tables == [[3, 4], [8, 9]]
    assert plan.slot_mapping == [64, 65, 66, 67, 145]


def test_rocm_sequence_plan_rejects_short_block_table() -> None:
    sequence = SequenceMetadataProxy(
        is_prompt=True,
        total_len=33,
        processed_len=32,
        block_table=[0, 1],
    )
    try:
        build_rocm_sequence_plan([sequence], block_size=16)
    except ValueError as exc:
        assert "Block table is too short" in str(exc)
    else:
        raise AssertionError("Expected a short block table to be rejected")


def test_rocm_backend_is_explicit_and_lazy() -> None:
    assert AttentionBackend.VLLM_ROCM.value == "VLLM_ROCM"
    assert get_attention_wrapper.__module__.endswith("backends")


def _cpu_rocm_wrapper(monkeypatch, model=None):
    """Initialize the production wrapper with CPU native/timer stand-ins."""
    impl = SimpleNamespace(do_kv_cache_update=Mock(), forward=Mock(
        side_effect=lambda layer, query, key, value, cache, metadata, output:
        output.copy_(query + 1),
    ))
    monkeypatch.setattr(torch.version, "hip", "cpu-contract")
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.rocm_attn", SimpleNamespace(
        RocmAttentionImpl=lambda **kwargs: impl,
        RocmAttentionMetadata=lambda **kwargs: SimpleNamespace(**kwargs),
    ))
    wrapper = VllmRocmAttentionWrapper()
    wrapper.init(model or ModelConfig.from_model_name("meta-llama/Llama-2-7b-hf"),
                 ParallelConfig(), 16, torch.device("cpu"))
    events = []

    @contextmanager
    def timer(operation, layer_id):
        events.append(("enter", operation.value, layer_id))
        yield
        events.append(("exit", operation.value, layer_id))

    monkeypatch.setattr(wrapper, "get_timer", timer)
    return wrapper, events


def _exercise_rocm_lifecycle(monkeypatch):
    wrapper, events = _cpu_rocm_wrapper(monkeypatch)
    phases = [
        (True, [(20, 16, [3, 4]), (3, 0, [6])], [0, 4, 7],
         [64, 65, 66, 67, 96, 97, 98], [[3, 4], [6, 0]]),
        (False, [(18, 17, [8, 9]), (1, 0, [2])], [0, 1, 2],
         [145, 32], [[8, 9], [2, 0]]),
    ]
    snapshots = []
    for is_prompt, shapes, query_start, slots, tables in phases:
        query = torch.zeros(len(slots), wrapper.num_q_heads * wrapper.head_dim)
        cache = torch.empty(0)
        with pytest.raises(RuntimeError, match="metadata is not initialized"):
            wrapper.forward(query, query, query, cache, softmax_scale=wrapper._scale)
        sequences = [SequenceMetadataProxy(is_prompt, *shape) for shape in shapes]
        wrapper.begin_forward(sequences)
        active = wrapper._prefill_metadata if is_prompt else wrapper._decode_metadata
        inactive = wrapper._decode_metadata if is_prompt else wrapper._prefill_metadata
        assert inactive is None
        assert active.query_start_loc.tolist() == query_start
        assert active.seq_lens.tolist() == [shape[0] for shape in shapes]
        assert active.block_table.tolist() == tables
        assert active.slot_mapping.tolist() == wrapper._slot_mapping.tolist() == slots
        assert active.num_actual_tokens == len(slots)
        assert active.max_query_len == (4 if is_prompt else 1)
        assert active.max_seq_len == shapes[0][0]
        output = wrapper.forward(query, query, query, cache,
                                 softmax_scale=wrapper._scale, layer_id=7)
        assert torch.equal(output, query + 1)
        assert wrapper._impl.forward.call_args.args[-2] is active
        assert wrapper._impl.do_kv_cache_update.call_args.args[-1] is wrapper._slot_mapping
        assert events == [
            (edge, operation.value, 7)
            for operation in (
                OperationMetrics.ATTN_INPUT_RESHAPE, OperationMetrics.ATTN_KV_CACHE_SAVE,
                OperationMetrics.ATTN_PREFILL, OperationMetrics.ATTN_DECODE,
                OperationMetrics.ATTN_OUTPUT_RESHAPE,
            )
            for edge in ("enter", "exit")
        ]
        snapshots.append({
            "metadata": {name: value.tolist() if isinstance(value, torch.Tensor) else value
                         for name, value in vars(active).items()},
            "slots": wrapper._slot_mapping.tolist(), "timers": list(events),
            "output_shape": list(output.shape), "output_sum": output.sum().item(),
            "contains_prefill": wrapper.contains_prefill,
            "contains_decode": wrapper.contains_decode,
        })
        wrapper.end_forward()
        assert wrapper._prefill_metadata is wrapper._decode_metadata is wrapper._slot_mapping is None
        with pytest.raises(RuntimeError, match="metadata is not initialized"):
            wrapper.forward(query, query, query, cache, softmax_scale=wrapper._scale)
        events.clear()
    assert wrapper._impl.forward.call_count == 2
    wrapper.begin_forward([])
    assert not wrapper.contains_prefill and not wrapper.contains_decode
    assert wrapper._slot_mapping.tolist() == []
    assert wrapper._prefill_metadata is wrapper._decode_metadata is None
    wrapper.end_forward()
    return snapshots


def test_rocm_wrapper_preserves_metadata_timers_and_phase_lifecycle(monkeypatch):
    _exercise_rocm_lifecycle(monkeypatch)


def test_rocm_wrapper_rejects_mixed_before_materializing_metadata(monkeypatch):
    wrapper, events = _cpu_rocm_wrapper(monkeypatch)
    with pytest.raises(NotImplementedError, match="mixed execution"):
        wrapper.begin_forward([
            SequenceMetadataProxy(True, 20, 16, [3, 4]),
            SequenceMetadataProxy(False, 18, 17, [8, 9]),
        ])
    assert wrapper._prefill_metadata is wrapper._decode_metadata is wrapper._slot_mapping is None
    assert events == []


def test_rocm_wrapper_accepts_hybrid_full_attention_without_weakening_binder(monkeypatch):
    from frontier.attention.model_binding import bind_attention_family

    model = ModelConfig.from_model_name("Qwen3.8-2.4T-A95B-Quark-MXFP4")
    assert model.get_num_gdn_layers() > 0
    with pytest.raises(ValueError, match="explicit global layer id"):
        bind_attention_family(model)
    wrapper, _ = _cpu_rocm_wrapper(monkeypatch, model)
    assert wrapper.num_q_heads == model.get_num_q_heads(ParallelConfig())
    assert wrapper.num_kv_heads == model.get_num_kv_heads(ParallelConfig())


@pytest.mark.parametrize("kv_heads", [32, 8, 1])
def test_rocm_wrapper_retains_dense_head_topologies(monkeypatch, kv_heads):
    model = ModelConfig.from_model_name("meta-llama/Llama-2-7b-hf")
    model.num_kv_heads = kv_heads
    wrapper, _ = _cpu_rocm_wrapper(monkeypatch, model)
    assert wrapper.num_kv_heads == kv_heads


def test_rocm_wrapper_rejects_latent_mla(monkeypatch):
    model = ModelConfig.from_model_name("meta-llama/Llama-2-7b-hf")
    model.use_mla = True
    model.kv_lora_rank = 32
    model.qk_nope_head_dim = 32
    model.qk_rope_head_dim = 32
    model.v_head_dim = 32
    with pytest.raises(NotImplementedError, match="not latent MLA"):
        _cpu_rocm_wrapper(monkeypatch, model)
