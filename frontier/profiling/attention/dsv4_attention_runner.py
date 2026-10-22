"""vLLM engine driver for DeepSeek-V4 attention profiling (GPU worker only).

The step classification, case window, timing, and raw-record writing are the
DSA driver's (``dsa_vllm_runner``); this module installs the DeepSeek-V4
module hooks and builds the engine.

Each ``DeepseekV4Attention`` layer of the reduced engine is wrapped as follows
(the enclosing ``dsv4_attention_layer`` scope has CUDA-event wall time only,
because kernel-only annotations must not nest):

* ``_run_parallel_input_projections``: the fused q/kv GEMM on the default
  stream and the compressor and indexer input GEMMs on aux streams, when the
  step has at most ``VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD`` tokens;
* the module-level ``fused_q_kv_rmsnorm``;
* ``wq_b`` and ``_fused_qnorm_rope_kv_insert``: the query projection, then
  query norm/RoPE and the SWA KV insert, on the default stream;
* ``compressor``: the layer's KV compressor, on an aux stream;
* ``indexer``: the index compressor plus either the short-context top-k fill
  or the index query projection and its RoPE/quantization, on an aux stream;
* ``indexer.indexer_op``: index scores and top-k (long contexts only);
* ``forward_mqa``: the FlashMLA sparse attention;
* ``_o_proj``: inverse RoPE, the grouped ``wo_a`` and ``wo_b``.

A CUDA-event scope on an aux stream starts after the stream joined the
default stream, so it excludes the join wait; kernels that overlap on other
streams can stretch it. Hooks are installed after engine start-up, so the
memory profiling and dummy runs are never recorded.

The weights are vLLM dummy weights. The accepted C26 dummy-loader hook is
preserved unchanged. For the pristine vLLM 0.28 source without that hook,
``patch_dummy_weight_finalizers`` supplies it only during engine construction.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from typing import Any

from frontier.profiling.attention import dsa_vllm_runner as case_runner
from frontier.profiling.attention.dsv4_attention_plan import (
    CSA_VARIANT,
    LAYER_SCOPE,
    METADATA_SOURCES_BY_VARIANT,
    UNCOMPRESSED_VARIANT,
    Dsv4ProfilePlan,
    metadata_key,
)

_ATTENTION_MODULE = "vllm.models.deepseek_v4.attention"
_METHOD_SCOPES = (
    ("_run_parallel_input_projections", "dsv4_parallel_input_projections"),
    ("_fused_qnorm_rope_kv_insert", "dsv4_fused_q_kv_prepare"),
    ("forward_mqa", "dsv4_sparse_attention"),
    ("_o_proj", "dsv4_o_proj"),
)
_RMSNORM_SCOPE = "dsv4_fused_qkv_rmsnorm"
_COMPRESSOR_METADATA_FIELDS = ("block_size", "num_decode_tokens", "c128_boundary")
# Scalar fields of the native attention metadata, by source.
METADATA_FIELDS = {
    "swa": (
        "block_size",
        "num_decodes",
        "num_prefills",
        "num_decode_tokens",
        "num_prefill_tokens",
        "max_decode_query_len",
        "decode_swa_width",
        "prefill_window_size",
        "prefill_max_model_len",
        "prefill_max_num_batched_tokens",
    ),
    "attention": (
        "num_reqs",
        "max_query_len",
        "max_seq_len",
        "num_actual_tokens",
        "block_size",
        "topk_tokens",
    ),
    "compressor": _COMPRESSOR_METADATA_FIELDS,
    "indexer": (
        "max_seq_len",
        "num_decodes",
        "num_decode_tokens",
        "num_prefills",
        "num_prefill_tokens",
    ),
    "index_compressor": _COMPRESSOR_METADATA_FIELDS,
}

# Layer whose forward is running; the module-level norm reads it.
_CURRENT_LAYER: int | None = None


def _metadata_fields(metadata: Any, names: tuple[str, ...]) -> dict[str, int | bool | None]:
    values = {}
    for name in names:
        value = getattr(metadata, name)
        if value is not None and type(value) not in (int, bool):
            raise TypeError(
                f"{type(metadata).__name__}.{name} is {type(value).__name__}, "
                "expected an int, bool, or None"
            )
        values[name] = value
    return values


def metadata_prefixes(module: Any) -> dict[str, str]:
    """Attention-metadata prefix of each native metadata source of a layer."""

    prefixes = {"swa": module.swa_cache_layer.prefix}
    if module.compressor is not None:
        prefixes["attention"] = module.prefix
        prefixes["compressor"] = module.compressor.state_cache.prefix
    if module.indexer is not None:
        prefixes["indexer"] = module.indexer.k_cache.prefix
        prefixes["index_compressor"] = module.indexer.compressor.state_cache.prefix
    return prefixes


def _wrap(owner: Any, attribute: str, layer_id: int, scope: str) -> None:
    original = getattr(owner, attribute)

    def call(*args, **kwargs):
        return case_runner._require_state().timed_call(layer_id, scope, original, args, kwargs)

    setattr(owner, attribute, call)


def _wrap_layer(module: Any, layer_id: int, variant: str, prefixes: dict[str, str]) -> None:
    original = module.forward

    def read(source: str):
        return lambda: _metadata_fields(
            case_runner._layer_metadata(prefixes[source]), METADATA_FIELDS[source]
        )

    def forward(*args, **kwargs):
        global _CURRENT_LAYER
        state = case_runner._require_state()
        for source in prefixes:
            state.capture_metadata(metadata_key(variant, source), read(source))
        _CURRENT_LAYER = layer_id
        try:
            return state.timed_call(
                layer_id, LAYER_SCOPE, original, args, kwargs, annotate=False
            )
        finally:
            _CURRENT_LAYER = None

    module.forward = forward


def _wrap_rmsnorm(attention_module: Any) -> None:
    original = attention_module.fused_q_kv_rmsnorm

    def fused_q_kv_rmsnorm(*args, **kwargs):
        if _CURRENT_LAYER is None:
            raise RuntimeError(
                "fused_q_kv_rmsnorm ran outside a hooked DeepSeek-V4 attention layer"
            )
        return case_runner._require_state().timed_call(
            _CURRENT_LAYER, _RMSNORM_SCOPE, original, args, kwargs
        )

    attention_module.fused_q_kv_rmsnorm = fused_q_kv_rmsnorm


def _single(values: set[Any], label: str) -> Any:
    if len(values) != 1:
        raise ValueError(f"Layers disagree on {label}: {sorted(map(str, values))}")
    return values.pop()


def install_dsv4_hooks(
    worker: Any, engine_index: int, layers: list[tuple[int, str, int]]
) -> dict[str, Any]:
    """Wrap every attention layer; return the engine's attention inventory.

    ``layers`` lists ``(global_layer_id, attention_variant, compress_ratio)``
    of the plan.
    """

    import importlib

    from vllm import envs

    attention_module = importlib.import_module(_ATTENTION_MODULE)
    planned = {layer_id: (variant, ratio) for layer_id, variant, ratio in layers}
    step_metadata = tuple(
        metadata_key(variant, source)
        for variant in dict.fromkeys(variant for _, variant, _ in layers)
        for source in METADATA_SOURCES_BY_VARIANT[variant]
    )
    case_runner.install_case_recorder(worker, engine_index, step_metadata)

    modules: dict[int, Any] = {}
    for name, module in worker.get_model().named_modules():
        if isinstance(module, attention_module.DeepseekV4Attention):
            layer_id = case_runner._layer_id(name)
            if layer_id in modules:
                raise ValueError(f"Layer {layer_id} has two DeepseekV4Attention modules")
            modules[layer_id] = module
    if set(modules) != set(planned):
        raise ValueError(
            f"Engine attention layers {sorted(modules)} differ from the plan {sorted(planned)}"
        )

    layer_inventory = []
    for layer_id in sorted(modules):
        module = modules[layer_id]
        variant, ratio = planned[layer_id]
        if module.compress_ratio != max(1, ratio):
            raise ValueError(
                f"Layer {layer_id} compresses by {module.compress_ratio}; the plan says {ratio}"
            )
        if (module.indexer is not None) != (variant == CSA_VARIANT):
            raise ValueError(f"Layer {layer_id} ({variant}) indexer presence is wrong")
        if (module.compressor is not None) != (variant != UNCOMPRESSED_VARIANT):
            raise ValueError(f"Layer {layer_id} ({variant}) compressor presence is wrong")
        prefixes = metadata_prefixes(module)
        if tuple(prefixes) != METADATA_SOURCES_BY_VARIANT[variant]:
            raise ValueError(
                f"Layer {layer_id} ({variant}) reads metadata {sorted(prefixes)}, expected "
                f"{list(METADATA_SOURCES_BY_VARIANT[variant])}"
            )
        _wrap_layer(module, layer_id, variant, prefixes)
        for attribute, scope in _METHOD_SCOPES:
            _wrap(module, attribute, layer_id, scope)
        _wrap(module.wq_b, "forward", layer_id, "dsv4_q_projection")
        if module.compressor is not None:
            _wrap(module.compressor, "forward", layer_id, "dsv4_compressor")
        if module.indexer is not None:
            _wrap(module.indexer, "forward", layer_id, "dsv4_index_prepare")
            _wrap(module.indexer.indexer_op, "forward", layer_id, "dsv4_index_scores_topk")
        layer_inventory.append(
            {
                "global_layer_id": layer_id,
                "attention_variant": variant,
                "attention_prefix": module.prefix,
                "compress_ratio": module.compress_ratio,
                "metadata_prefixes": prefixes,
            }
        )
    _wrap_rmsnorm(attention_module)

    attention = list(modules.values())
    indexers = [module.indexer for module in attention if module.indexer is not None]
    compressors = [module.compressor for module in attention if module.compressor is not None]
    aux_streams = {
        0 if module.aux_stream_list is None else len(module.aux_stream_list)
        for module in attention
    }
    return {
        "attention_backend": _single(
            {module.get_attn_backend().get_name() for module in attention}, "attention backend"
        ),
        "attention_impl": _single({type(module).__name__ for module in attention}, "class"),
        "swa_backend": _single(
            {module.swa_cache_layer.get_attn_backend().get_name() for module in attention},
            "SWA backend",
        ),
        "indexer_backend": _single(
            {indexer.k_cache.get_attn_backend().get_name() for indexer in indexers},
            "indexer backend",
        ),
        "compressor_backend": _single(
            {
                compressor.state_cache.get_attn_backend().get_name()
                for compressor in [*compressors, *(i.compressor for i in indexers)]
            },
            "compressor backend",
        ),
        "attention_kv_cache_dtype": _single(
            {module.kv_cache_dtype for module in attention}, "KV-cache dtype"
        ),
        "indexer_uses_fp4_kv": _single({i.use_fp4_kv for i in indexers}, "indexer FP4 KV"),
        "num_local_heads": _single({module.n_local_heads for module in attention}, "heads"),
        "padded_heads": _single({module.padded_heads for module in attention}, "padded heads"),
        "num_local_groups": _single(
            {module.n_local_groups for module in attention}, "output groups"
        ),
        "aux_stream_count": _single(aux_streams, "aux streams"),
        "multi_stream_gemm_token_threshold": envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
        "prepare_and_attn_fn": _single(
            {module._prepare_and_attn_fn.__name__ for module in attention},
            "attention preparation path",
        ),
        "uses_eager_scratch_pool": _single(
            {module.eager_scratch_pool is not None for module in attention}, "scratch pool"
        ),
        "use_v2_model_runner": bool(worker.vllm_config.use_v2_model_runner),
        "num_gpu_blocks": worker.vllm_config.cache_config.num_gpu_blocks,
        "layers": layer_inventory,
    }


@contextmanager
def patch_dummy_weight_finalizers() -> Iterator[None]:
    """Preserve C26, or temporarily supply the pristine dummy-loader hook.

    The loader alone invokes the hook. C26 initializes missing mHC broadcast
    weights without replacing existing weights; the pristine source needs its
    original MegaMoE and mHC finalizers. Unknown native hooks fail closed.
    """

    from vllm.models.deepseek_v4 import DeepseekV4ForCausalLM

    if hasattr(DeepseekV4ForCausalLM, "process_weights_after_loading"):
        native_hook = DeepseekV4ForCausalLM.process_weights_after_loading
        approved_c26 = (
            "def process_weights_after_loading(self) -> None:\n"
            "    self.model.finalize_mhc_broadcast_weights(only_if_missing=True)"
        )
        try:
            source = dedent(inspect.getsource(native_hook.__code__)).strip()
        except (AttributeError, OSError, TypeError) as error:
            raise RuntimeError("Unapproved DeepSeek-V4 dummy finalizer") from error
        if source != approved_c26:
            raise RuntimeError("Unapproved DeepSeek-V4 dummy finalizer; expected C26")
        yield
        return

    def process_weights_after_loading(self) -> None:
        self.model.finalize_mega_moe_weights()
        self.model.finalize_mhc_broadcast_weights()

    DeepseekV4ForCausalLM.process_weights_after_loading = process_weights_after_loading
    try:
        yield
    finally:
        del DeepseekV4ForCausalLM.process_weights_after_loading


def run_engine(
    *,
    plan: Dsv4ProfilePlan,
    engine_index: int,
    model_dir: str,
    vocab_size: int,
    seed: int,
    gpu_memory_utilization: float,
    expected_attention_backend: str,
    raw_path: Path,
    trace_dir: Path,
    keep_traces: bool,
    case_ids: set[str] | None = None,
) -> None:
    """Measure every case of one engine; append raw records to ``raw_path``."""

    # Import the model after selecting the in-process engine core.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    with patch_dummy_weight_finalizers():
        llm = case_runner.build_engine(
            plan,
            engine_index,
            model_dir,
            seed,
            gpu_memory_utilization,
            kv_cache_dtype=plan.kv_cache_dtype,
        )
    layers = [
        (layer.global_layer_id, layer.attention_variant, layer.compress_ratio)
        for layer in plan.shape.layers
    ]
    inventory = llm.collective_rpc(install_dsv4_hooks, args=(engine_index, layers))[0]
    case_runner.measure_engine(
        llm=llm,
        plan=plan,
        engine_index=engine_index,
        inventory=inventory,
        vocab_size=vocab_size,
        seed=seed,
        expected_attention_backend=expected_attention_backend,
        raw_path=raw_path,
        trace_dir=trace_dir,
        keep_traces=keep_traces,
        case_ids=case_ids,
        log_label="dsv4",
    )


__all__ = [
    "METADATA_FIELDS",
    "install_dsv4_hooks",
    "metadata_prefixes",
    "patch_dummy_weight_finalizers",
    "run_engine",
]
