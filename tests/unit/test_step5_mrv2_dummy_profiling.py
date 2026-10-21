"""Execute locked MRv2 dummy methods on CPU, not the MRv1 profiling path."""

import ast
from contextlib import nullcontext
import copy
from dataclasses import dataclass, field
import os
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from frontier.profiling.step5.inputs import MRv2DummyDescriptor, Step5ProfileInput, read_profile_inputs


@pytest.fixture(scope="module")
def native_source():
    root = os.environ.get("STEP5_NATIVE_SOURCE_ROOT")
    if root is None:
        pytest.skip("Set STEP5_NATIVE_SOURCE_ROOT to the locked exported vllm package")
    return Path(root)


def native_method(root, relative, name, namespace):
    source = ast.parse((root / relative).read_text())
    method = copy.deepcopy(next(node for node in ast.walk(source)
                                if isinstance(node, ast.FunctionDef) and node.name == name))
    method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(root / relative), "exec"), namespace)
    return namespace[name]


def native_class(root, relative, name, namespace, methods=None):
    source = ast.parse((root / relative).read_text())
    definition = copy.deepcopy(next(node for node in source.body
                                    if isinstance(node, ast.ClassDef) and node.name == name))
    if methods is not None:
        definition.body = [node for node in definition.body
                           if isinstance(node, ast.FunctionDef) and node.name in methods]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             definition], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(root / relative), "exec"), namespace)
    return namespace[name]


def test_serving_dummy_binding_matches_actual_mrv2_worker_and_execute_path(native_source):
    """An eight-token descriptor is a CPU fixture, not an observed P0 width."""
    calls = {}

    class MetadataBoundary(Exception):
        pass

    descriptor = SimpleNamespace(num_reqs=8, num_tokens=8, max_query_len=1,
                                 cg_mode="FULL", num_ubatches=1)

    def synchronize(*args, **kwargs):
        calls["local_dispatch"] = args[1:4]
        return descriptor, object()

    def make_dummy(num_reqs, num_tokens, buffers, **kwargs):
        calls["physical_dummy"] = num_reqs, num_tokens, kwargs
        return object()

    def metadata(*args, **kwargs):
        calls["metadata"] = args, kwargs
        raise MetadataBoundary

    namespace = dict(np=np, SchedulerOutput=SimpleNamespace(make_empty=SimpleNamespace),
        dispatch_cg_and_sync_dp=synchronize, InputBatch=SimpleNamespace(make_dummy=make_dummy),
        CUDAGraphMode=SimpleNamespace(FULL="FULL"), build_slot_mappings_by_layer=lambda *args: {"attn": "slots"},
        make_cudagraph_stats=lambda *a: pytest.fail("MRv2 excludes dummy from ordinary cudagraph statistics"))
    run = native_method(native_source, "v1/worker/gpu/model_runner.py", "_dummy_run", namespace)
    execute = native_method(native_source, "v1/worker/gpu/model_runner.py", "execute_model", namespace)
    runner = SimpleNamespace(max_num_reqs=128, max_num_tokens=8192, decode_query_len=1,
        is_first_pp_rank=True, lora_config=None, kv_connector=SimpleNamespace(set_disabled=lambda _: None),
        maybe_dummy_run_with_lora=lambda *a, **k: nullcontext(), gather_batch_req_state=lambda *a: (None, 1),
        pcp_manager=None, is_encoder_decoder=False, cudagraph_manager=object(), dp_size=2, dp_rank=1,
        parallel_config=object(), ubatch_runner=None, input_buffers=object(), dcp_size=1,
        observability_config=SimpleNamespace(cudagraph_metrics=True),
        prepare_dummy_attn=lambda *a: ("dummy_tables", "slots"),
        model_state=SimpleNamespace(prepare_attn=metadata), attn_groups=object(), kv_cache_config=object())
    runner._dummy_run = MethodType(run, runner)
    runner.execute_model = MethodType(execute, runner)
    worker = SimpleNamespace(model_runner=runner, randomize_dummy_inputs=False)
    worker_run = native_method(native_source, "v1/worker/gpu_worker.py", "execute_dummy_batch", namespace)
    with pytest.raises(MetadataBoundary):
        worker_run(worker)
    assert calls["local_dispatch"] == (1, 1, 1)
    assert calls["physical_dummy"] == (8, 8, dict(max_query_len=1, is_padding=True))
    assert calls["metadata"][1] == {"for_capture": True}
    root = Path(__file__).resolve().parents[2]
    for role in ("prefill", "decode"):
        cells = read_profile_inputs(root / f"frontier/profiling/step5/p0_{role}_inputs.json",
                                    cudagraph_capture_sizes=(1, 2, 4, 8, 16))
        assert all(cell.has_attention_metadata for cell in cells), (
            "MRv2 serving dummy prepares attention metadata; isolated no-metadata cells "
            "must not be included as serving idle-DP coverage")


@pytest.fixture
def mrv2_wrapper(native_source, monkeypatch):
    """Only CUDA allocation/UVA/FlashAttention kernels are CPU boundaries."""
    from abc import ABC
    from frontier.profiling.step5.vllm_wrapper import VllmStep5AttentionWrapper
    import torch
    import sys

    def module(name, **values):
        result = ModuleType(name)
        result.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    enums = SimpleNamespace(NONE="none", PIECEWISE="piecewise", FULL="full")
    module("vllm.config", CUDAGraphMode={key: value for key, value in vars(enums).items()})
    inputs = module("vllm.v1.worker.gpu.input_batch", np=np, torch=torch, dataclass=dataclass,
                    random_uuid=lambda: "cpu-fixture")
    for name in ("InputBuffers", "InputBatch"):
        native_class(native_source, "v1/worker/gpu/input_batch.py", name, inputs.__dict__)
    tables = module("vllm.v1.worker.gpu.block_table", torch=torch, PAD_SLOT_ID=-1,
        StagedWriteTensor=lambda shape, **kw: SimpleNamespace(gpu=torch.zeros(shape, **kw)),
        UvaBackedTensor=lambda shape, **kw: SimpleNamespace(gpu=torch.zeros(shape, **kw)))
    native_class(native_source, "v1/worker/gpu/block_table.py", "BlockTables", tables.__dict__)
    backend = module("vllm.v1.attention.backend", CommonAttentionMetadata=SimpleNamespace)
    utils = module("vllm.v1.worker.utils", dataclass=dataclass, field=field)
    native_class(native_source, "v1/worker/utils.py", "AttentionGroup", utils.__dict__)
    attn = module("vllm.v1.worker.gpu.attn_utils", torch=torch,
                  CommonAttentionMetadata=backend.CommonAttentionMetadata)
    native_method(native_source, "v1/worker/gpu/attn_utils.py", "build_attn_metadata", attn.__dict__)
    state = module("step5_mrv2_cpu_state", torch=torch, ABC=ABC, CUDAGraphMode=enums,
                   build_attn_metadata=attn.build_attn_metadata)
    native_method(native_source, "v1/worker/gpu/mm/rope.py", "get_rope_state", state.__dict__)
    native_method(native_source, "v1/worker/gpu/model_states/mm_pruning.py", "maybe_create_mm_pruner", state.__dict__)
    native_class(native_source, "v1/worker/gpu/model_states/interface.py", "ModelState",
                 state.__dict__, {"__init__", "max_model_len"})
    native_class(native_source, "v1/worker/gpu/model_states/default.py", "DefaultModelState",
                 state.__dict__, {"__init__", "prepare_attn"})
    step_state = module("vllm.models.step5.nvidia.model_state", torch=torch,
        DefaultModelState=state.DefaultModelState, __package__="vllm.models.step5.nvidia")
    native_class(native_source, "models/step5/nvidia/model_state.py", "Step5ModelState", step_state.__dict__)

    class FlashBuilder:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            return common_attn_metadata

    sparse = module("vllm.models.step5.nvidia.sparse_attention", torch=torch, cast=cast, Any=Any,
        FlashAttentionMetadataBuilder=FlashBuilder,
        AttentionCGSupport=SimpleNamespace(UNIFORM_SINGLE_TOKEN_DECODE=1, UNIFORM_BATCH=2),
        AttentionSpec=object, _PrefillLayout=lambda *args: args, is_quantized_kv_cache=lambda _: False)
    builder_class = native_class(native_source, "models/step5/nvidia/sparse_attention.py",
                                  "Step5DSAMetadataBuilder", sparse.__dict__)
    native_method(native_source, "v1/attention/backend.py", "build_for_cudagraph_capture", sparse.__dict__)
    FlashBuilder.build_for_cudagraph_capture = sparse.build_for_cudagraph_capture
    module("vllm.v1.kv_cache_interface",
        KVCacheConfig=lambda blocks, tensors, groups: SimpleNamespace(kv_cache_groups=groups),
        KVCacheGroupSpec=lambda names, spec: SimpleNamespace(layer_names=names, kv_cache_spec=spec))
    impl = SimpleNamespace(adapter=object(), summary_cache=object(), num_speculative_tokens=0,
                           num_heads=8, kv_cache_dtype="auto", plan_prefill_step=lambda *args: "CPU_KERNEL_PLAN")
    native = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=128, max_num_batched_tokens=8192),
        model_config=SimpleNamespace(get_inputs_embeds_size=lambda: 4, dtype=torch.float32,
                                    max_model_len=7168, enable_prompt_embeds=False, uses_mrope=False),
        compilation_config=SimpleNamespace(static_forward_context={"attn": SimpleNamespace(impl=impl)}))
    wrapper = VllmStep5AttentionWrapper.__new__(VllmStep5AttentionWrapper)
    wrapper.torch = SimpleNamespace(device=lambda _: torch.device("cpu"))
    wrapper.native, wrapper.max_batch_size, wrapper.block_size = native, 128, 16
    wrapper.layer = SimpleNamespace(attn=SimpleNamespace(layer_name="attn", attn_backend=object()))
    wrapper.cache_spec = SimpleNamespace(page_size_bytes=16, num_kv_heads=1, head_size=192, block_size=16)
    wrapper.raw_cache = torch.zeros(32)
    wrapper.dummy_batches, wrapper._set_current = {}, lambda _: nullcontext()
    wrapper.builder = builder_class(vllm_config=native, kv_cache_spec=wrapper.cache_spec, layer_names=["attn"])
    return wrapper


@pytest.mark.parametrize("tokens,requests,mode,max_query,queries", [
    (8, 8, "full", 1, (1,) * 8),
    (16, None, "piecewise", None, (16,)),
    (8, 3, "none", 3, (2, 3, 3)),
])
def test_real_mrv2_input_block_table_step_state_and_builder_path(
        mrv2_wrapper, tokens, requests, mode, max_query, queries):
    descriptor = MRv2DummyDescriptor(tokens, requests, mode, max_query, "CPU_SOURCE_FIXTURE_NOT_P0_CENSUS")
    cell = Step5ProfileInput.from_mrv2_dummy(3, descriptor, block_table_columns=448)
    positions, slots, metadata, graph, padding = mrv2_wrapper._metadata(cell)
    assert cell.query_lens == queries and not any(cell.context_lens)
    assert positions.tolist() == [0] * tokens
    assert slots.tolist() == [-1] * tokens and padding.tolist() == [True] * tokens
    assert metadata.block_table_tensor.shape == (len(queries), 448)
    assert metadata.block_table_tensor.count_nonzero().item() == 0
    assert metadata.seq_lens.tolist() == list(queries)
    assert metadata.seq_lens_cpu_upper_bound.tolist() == list(queries)
    assert metadata.max_seq_len == (7168 if mode == "full" else max(queries))
    assert metadata.step5_request_state.tolist() == list(range(len(queries)))
    buffers, tables, state, _, _ = mrv2_wrapper.dummy_batches[448]
    assert state.request_state[len(queries):].tolist() == [-1] * (128 - len(queries))
    assert tables.slot_mappings.tolist() == [[-1] * 8192]
    pointer = metadata.block_table_tensor.data_ptr()
    metadata.block_table_tensor.fill_(27)
    tables.slot_mappings.fill_(33)
    _, again_slots, again, _, _ = mrv2_wrapper._metadata(cell)
    assert again.block_table_tensor.data_ptr() == pointer
    assert not again.block_table_tensor.count_nonzero()
    assert again_slots.tolist() == [-1] * tokens


def test_native_csa_negative_slots_update_scratch_not_live_pages(native_source):
    import torch
    namespace = dict(torch=torch, np=np, Any=Any)
    cache_class = native_class(native_source, "models/step5/nvidia/sparse_summary_cache.py",
                               "Step5SparseSummaryCache", namespace)
    cache = cache_class(2, 16, torch.device("cpu"), max_num_seqs=8)
    before = cache.mean.clone()
    cache.update_decode(torch.ones(8, 1, 256), torch.zeros(8, 1, 256), torch.full((8,), -1))
    assert torch.equal(cache.mean, before)
    assert cache._mean[2:].count_nonzero() > 0
    assert cache.numerator[:2].count_nonzero() == 0
    assert cache.numerator[2:].count_nonzero() > 0


def test_real_producer_forward_context_keeps_metadata_slots_and_padding(mrv2_wrapper, monkeypatch):
    import torch
    import sys
    from contextlib import contextmanager
    calls = []

    @contextmanager
    def context(metadata, config, **kwargs):
        calls.append((metadata, kwargs))
        yield

    module = ModuleType("vllm.forward_context")
    module.set_forward_context, module.BatchDescriptor = context, lambda **kw: kw
    monkeypatch.setitem(sys.modules, module.__name__, module)
    descriptor = MRv2DummyDescriptor(8, 8, "full", 1, "CPU_SOURCE_FIXTURE_NOT_P0_CENSUS")
    cell = Step5ProfileInput.from_mrv2_dummy(3, descriptor, block_table_columns=448)
    prepared = mrv2_wrapper._metadata(cell)
    hidden = torch.ones(8, 4)
    assert torch.equal(mrv2_wrapper._forward(hidden, prepared, layer=lambda p, h: h), hidden)
    metadata, kwargs = calls[0]
    assert metadata["attn"] is prepared[2]
    assert kwargs["slot_mapping"]["attn"].tolist() == [-1] * 8
    assert kwargs["is_padding"].tolist() == [True] * 8
    assert kwargs["cudagraph_runtime_mode"] == "full"


def test_explicit_descriptor_input_never_inherits_local_one_token_dispatch(tmp_path):
    from dataclasses import asdict
    import json
    descriptor = MRv2DummyDescriptor(8, 8, "full", 1, "CPU_SOURCE_FIXTURE_NOT_P0_CENSUS")
    cell = Step5ProfileInput.from_mrv2_dummy(0, descriptor, block_table_columns=448)
    path = tmp_path / "input.json"
    path.write_text(json.dumps([asdict(cell)]))
    assert read_profile_inputs(path, cudagraph_capture_sizes=(1, 2, 4, 8)) == [cell]
    with pytest.raises(ValueError, match="configured captured width"):
        read_profile_inputs(path, cudagraph_capture_sizes=(1, 2))


@pytest.mark.parametrize("change", [
    {"has_attention_metadata": False}, {"context_lens": (8,) * 8},
    {"query_lens": (1,), "context_lens": (0,)}, {"graph_mode": "none"},
    {"block_table_columns": None},
])
def test_descriptor_cannot_be_relabelled_or_drop_its_metadata(change):
    from dataclasses import replace
    descriptor = MRv2DummyDescriptor(8, 8, "full", 1, "CPU_SOURCE_FIXTURE_NOT_P0_CENSUS")
    cell = Step5ProfileInput.from_mrv2_dummy(0, descriptor, block_table_columns=448)
    with pytest.raises(ValueError):
        replace(cell, **change)


@pytest.mark.parametrize("mutation", [None, "kv", "summary"])
def test_real_profile_allows_native_scratch_but_rejects_live_cache_writes(mrv2_wrapper, monkeypatch, mutation):
    import torch
    from frontier.attention.step5 import describe_step5_layer
    from frontier.profiling.common.model_config import ModelConfig

    wrapper = mrv2_wrapper
    descriptor = MRv2DummyDescriptor(8, 8, "full", 1, "CPU_SOURCE_FIXTURE_NOT_P0_CENSUS")
    cell = Step5ProfileInput.from_mrv2_dummy(3, descriptor, block_table_columns=448)
    wrapper._metadata(cell)
    wrapper.torch = torch
    wrapper.config = ModelConfig.from_model_name("stepfun-ai/Step-5-Preview")
    wrapper.layer_id, wrapper.graph_mode, wrapper.max_model_len = 3, "full", 7168
    wrapper.profile_index, wrapper.seed = 0, 0
    wrapper.layer.use_dsa_backend = True
    wrapper.layer.sparse_summary_cache = SimpleNamespace(_cache=SimpleNamespace(pages=2))
    wrapper.state = [torch.zeros(1), *(torch.zeros(4, 2) for _ in range(5))]
    wrapper.empty_state = wrapper._snapshot()
    wrapper.description = describe_step5_layer(wrapper.config, 3, attention_tp_size=8,
                                               moe_tp_size=1, expert_parallel_size=1)
    active = dict(wrapper.description.attention_scopes(cell.workload(wrapper.config,
        block_size=16, max_model_len=7168)))
    wrapper.scopes = SimpleNamespace(observe=lambda _: nullcontext(), times=lambda: active)
    wrapper._bindings = lambda: []
    randn = torch.randn
    monkeypatch.setattr(torch, "randn", lambda *a, **kw: randn(*a, **{**kw, "device": "cpu"}))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def forward(hidden, prepared):
        for tensor in wrapper.state[1:]:
            tensor[2:] = 1
        if mutation == "kv":
            wrapper.state[0].add_(1)
        if mutation == "summary":
            wrapper.state[1][0].add_(1)
        return hidden

    def timing_boundary(*args, **kwargs):
        raise RuntimeError("CPU source-bound profile reached unchanged timing seam")

    wrapper._forward = forward
    monkeypatch.setattr("frontier.profiling.step5.vllm_wrapper.capture_native_replay", timing_boundary)
    with pytest.raises(AssertionError if mutation else RuntimeError,
                       match=None if mutation else "reached unchanged timing seam"):
        wrapper.profile(cell, warmup_iterations=1, profile_iterations=2, max_time_cv=.05)


@pytest.mark.parametrize("layer", [0, 3])
def test_dummy_cost_identity_is_not_silently_admitted_as_live_or_no_metadata(layer):
    from frontier.profiling.common.model_config import ModelConfig
    from frontier.profiling.step5.rows import build_row
    config = ModelConfig.from_model_name("stepfun-ai/Step-5-Preview")
    descriptor = MRv2DummyDescriptor(8, 8, "full", 1, "CPU_SOURCE_FIXTURE_NOT_P0_CENSUS")
    cell = Step5ProfileInput.from_mrv2_dummy(layer, descriptor, block_table_columns=448)
    row = build_row(config, cell, tp=8, block_size=16, max_model_len=7168,
        runtime_signature="CPU_FIXTURE", device="h800", time_stats={}, evidence={"attention_backend": "CPU_FIXTURE"})
    assert row["has_attention_metadata"] == 1
    assert row["physical_scope"] == "mrv2_dp_dummy_attention_layer_without_tp_allreduce"
    assert row["native_workload"] == "mrv2_dp_dummy"
    assert row["dummy_owned_requests"] == 0 and row["batch_num_tokens"] == 8
    assert row["dummy_is_padding"] and row["slot_mapping_kind"] == "PAD_SLOT_ID"
