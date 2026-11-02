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

import hashlib
import importlib
import inspect
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any

from frontier.attention.deepseek_v4 import (
    DEEPSEEK_V4_CACHES_BY_VARIANT, DEEPSEEK_V4_CACHE_METADATA_SOURCES,
)
from frontier.profiling.attention import dsa_vllm_runner as case_runner
from frontier.profiling.attention.dsv4_attention_plan import (
    CSA_VARIANT,
    LAYER_SCOPE,
    KV_CACHE_GROUPS,
    METADATA_SOURCES_BY_VARIANT,
    UNCOMPRESSED_VARIANT,
    Dsv4ProfilePlan,
    metadata_key,
)

_ATTENTION_MODULE = "vllm.models.deepseek_v4.attention"
ROUTING_NATIVE_MODULES = (
    "vllm.envs", "vllm.forward_context", "vllm.distributed.parallel_state",
    "vllm.model_executor.layers.fused_moe.router.router_factory",
    "vllm.model_executor.layers.fused_moe.router.routing_simulator_router",
    "vllm.model_executor.layers.fused_moe.router.base_router",
    "vllm.model_executor.layers.fused_moe.router.fused_moe_router",
    "vllm.model_executor.layers.fused_moe.runner.moe_runner",
    "vllm.model_executor.layers.fused_moe.layer",
)
routing_config = None
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


def _routing_identity(model, config):
    """Observe constructed routers and bind their imported native sources."""
    import importlib
    from vllm import envs
    from vllm.model_executor.layers.fused_moe.router import routing_simulator_router as simulator
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    factory = importlib.import_module("vllm.model_executor.layers.fused_moe.layer").create_fused_moe_router
    if factory.__globals__["envs"] is not envs or simulator.envs is not envs:
        raise ValueError("Routing factory and simulator use different environment modules")

    selected = config["moe_gating_routing_runtime_path"]
    if (selected != "balanced_topk" or os.environ.get("VLLM_MOE_ROUTING_SIMULATION_STRATEGY") != selected
            or envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY != selected):
        raise ValueError("Constructed model routing environment mismatch")
    strategy = simulator.RoutingSimulator._routing_strategies[selected]
    origin = Path(sys.modules[type(strategy).__module__].__file__)
    digest = hashlib.sha256(origin.read_bytes()).hexdigest()
    if (origin != Path(config["script"]).with_name("balanced_routing.py")
            or str(origin) != strategy.route_tokens.__func__.__code__.co_filename
            or digest != config["tool_hashes"][str(origin)]):
        raise ValueError("Constructed model routing strategy origin or hash mismatch")
    native_sources = {}
    for name in ROUTING_NATIVE_MODULES:
        module = importlib.import_module(name)
        path = Path(module.__file__)
        relative = name.replace(".", "/") + ".py"
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if (path != Path(config["native_package"]) / relative.removeprefix("vllm/")
                or checksum != config["expected_source_hashes"][relative]):
            raise ValueError(f"Constructed model routing native source mismatch: {name}")
        native_sources[relative] = dict(origin=str(path), sha256=checksum)
    routers = {name: module.router for name, module in model.named_modules() if isinstance(module, MoERunner)}
    if not routers or any(type(router) is not simulator.RoutingSimulatorRouter for router in routers.values()):
        raise ValueError("Constructed model does not use balanced simulation routers")
    codes = (envs.__getattr__, envs.environment_variables["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"], factory,
        simulator.RoutingSimulator.simulate_routing, simulator.RoutingSimulatorRouter._compute_routing,
        inspect.getclosurevars(strategy.route_tokens).nonlocals["get_forward_context"],
        inspect.getclosurevars(strategy.route_tokens).nonlocals["get_ep_group"],
        MoERunner.__init__, MoERunner._forward_impl, MoERunner._apply_quant_method,
        *(router.select_experts.__func__ for router in routers.values()),
        *(router._select_experts.__func__ for router in routers.values()))
    codes = [inspect.unwrap(function) for function in codes]
    for function in codes:
        source = function.__module__.replace(".", "/") + ".py"
        if function.__code__.co_filename != native_sources[source]["origin"]:
            raise ValueError("Constructed model routing callable origin mismatch")
    return dict(
        environment=selected, runtime_path=envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY,
        strategy=type(strategy).__module__ + "." + type(strategy).__name__,
        origin=str(origin), sha256=digest,
        routers={name: type(router).__module__ + "." + type(router).__name__ for name, router in routers.items()},
        runners={name: type(module).__module__ + "." + type(module).__name__ for name, module in model.named_modules()
                 if isinstance(module, MoERunner)}, native_sources=native_sources)


@dataclass(frozen=True)
class AttentionLayers:
    """Validated native modules and their planned variants and metadata prefixes."""

    modules: dict[int, Any]
    variants: dict[int, str]
    prefixes: dict[int, dict[str, str]]


def discover_attention_layers(worker: Any, layers: list[tuple[int, str, int]]) -> AttentionLayers:
    """Validate the native attention layers against the reduced-model plan."""

    attention_module = importlib.import_module(_ATTENTION_MODULE)
    planned = {layer_id: (variant, ratio) for layer_id, variant, ratio in layers}
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

    variants = {}
    layer_prefixes = {}
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
        variants[layer_id] = variant
        layer_prefixes[layer_id] = prefixes
    return AttentionLayers(modules, variants, layer_prefixes)


def attention_inventory(
    modules: dict[int, Any], layer_inventory: list[dict[str, Any]], worker: Any, config=None
) -> dict[str, Any]:
    """Observe attention backends, cache layouts, streams, and routing identity."""

    from vllm import envs

    cache_names = {}
    for layer in layer_inventory:
        layer_id, variant = layer["global_layer_id"], layer["attention_variant"]
        prefixes = metadata_prefixes(modules[layer_id])
        for cache in DEEPSEEK_V4_CACHES_BY_VARIANT[variant]:
            cache_names[prefixes[DEEPSEEK_V4_CACHE_METADATA_SOURCES[cache]]] = (layer_id, cache)
    kv_cache_groups = []
    for group in worker.model_runner.kv_cache_config.kv_cache_groups:
        members = []
        for name in group.layer_names:
            if name not in cache_names:
                raise ValueError(f"KV-cache group name {name!r} is not a discovered DeepSeek-V4 cache")
            members.append(cache_names[name])
        kv_cache_groups.append([list(member) for member in sorted(members)])

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
        KV_CACHE_GROUPS: kv_cache_groups,
        **({"moe_routing": _routing_identity(worker.get_model(), config)} if config is not None else {}),
    }


def install_dsv4_hooks(
    worker: Any, engine_index: int, layers: list[tuple[int, str, int]], config=None
) -> dict[str, Any]:
    """Wrap every attention layer; return the engine's attention inventory.

    ``layers`` lists ``(global_layer_id, attention_variant, compress_ratio)``
    of the plan.
    """

    attention_module = importlib.import_module(_ATTENTION_MODULE)
    step_metadata = tuple(
        metadata_key(variant, source)
        for variant in dict.fromkeys(variant for _, variant, _ in layers)
        for source in METADATA_SOURCES_BY_VARIANT[variant]
    )
    case_runner.install_case_recorder(worker, engine_index, step_metadata)
    discovered = discover_attention_layers(worker, layers)
    layer_inventory = []
    for layer_id in sorted(discovered.modules):
        module = discovered.modules[layer_id]
        variant = discovered.variants[layer_id]
        prefixes = discovered.prefixes[layer_id]
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
    return attention_inventory(discovered.modules, layer_inventory, worker, config)


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


def build_dsv4_engine(
    plan: Dsv4ProfilePlan,
    engine_index: int,
    model_dir: str,
    seed: int,
    gpu_memory_utilization: float,
    *,
    routed_expert_count: int | None = None,
) -> Any:
    """Build the in-process V4 engine with the accepted dummy-weight finalizers."""

    if routed_expert_count is not None:
        from dataclasses import replace
        plan = replace(plan, hf_overrides=dict(plan.hf_overrides, n_routed_experts=routed_expert_count))
    # Import the model after selecting the in-process engine core.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    with patch_dummy_weight_finalizers():
        return case_runner.build_engine(
            plan,
            engine_index,
            model_dir,
            seed,
            gpu_memory_utilization,
            kv_cache_dtype=plan.kv_cache_dtype,
        )


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

    llm = build_dsv4_engine(plan, engine_index, model_dir, seed, gpu_memory_utilization)
    layers = [
        (layer.global_layer_id, layer.attention_variant, layer.compress_ratio)
        for layer in plan.shape.layers
    ]
    inventory = llm.collective_rpc(install_dsv4_hooks, args=(engine_index, layers, routing_config))[0]
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
    "AttentionLayers",
    "METADATA_FIELDS",
    "attention_inventory",
    "build_dsv4_engine",
    "discover_attention_layers",
    "install_dsv4_hooks",
    "metadata_prefixes",
    "patch_dummy_weight_finalizers",
    "run_engine",
]
