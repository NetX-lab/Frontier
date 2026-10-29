"""Turn raw DeepSeek-V4 attention runner records into profiling rows.

The raw records use the DSA runner format (``engine``, ``step``, ``call``).
A step record carries one metadata entry per ``(variant, source)`` of the
reduced model, see ``dsv4_attention_plan.METADATA_SOURCES_BY_VARIANT``.

A row aggregates every call of one scope over the reduced model's layers of
one attention variant and one tile-schedule role (the first variant in each
SWA cache group plans the FlashMLA decode tile schedule, later layers reuse it), over
the steps of one case that share the same scheduled requests and native
metadata of that variant. ``operator_scopes`` maps each timed scope to the
canonical family operators whose kernels run in it.

The CUDA-event table has the enclosing ``dsv4_attention_layer`` scope; the
kernel-only table does not, because kernel-only annotations must not nest.
Scopes on aux streams overlap the default-stream scopes, so the layer time is
not the sum of its scopes. All times are milliseconds.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from frontier.config.precision_type import PrecisionType
from frontier.profiling.attention.dsv4_step_scopes import is_step_scope
from frontier.profiling.attention.dsa_plan import require_scope_timing
from frontier.profiling.attention.dsa_rows import (
    KERNEL_ONLY,
    MEASUREMENT_TYPES,
    StepRequest,
    _request_signature,
    _requests,
    _RowAccumulator,
    batch_load_columns,
    index_records,
    scope_columns,
)
from frontier.profiling.attention.dsv4_attention_plan import (
    CSA_VARIANT,
    DSV4_SCOPE_NAMES,
    HCA_VARIANT,
    METADATA_SOURCES_BY_VARIANT,
    UNCOMPRESSED_VARIANT,
    Dsv4Layer,
    Dsv4ProfilePlan,
    expected_scopes,
    metadata_key,
    scope_operators,
    require_realized_kv_cache_groups,
)
from frontier.profiling.common.runtime_identity import canonical_json

DSV4_ROW_SCHEMA_VERSION = "dsv4_attention.v1"
BALANCED_ROUTING_SHA256 = "6562ec1c7e1ee7a8021d71a1a698c36f126307f00681320b07397b53c4c3f4ea"


def routing_columns(identity):
    from frontier.profiling.attention.dsv4_attention_runner import ROUTING_NATIVE_MODULES

    if (identity.get("environment") != "balanced_topk" or identity.get("runtime_path") != "balanced_topk"
            or identity.get("strategy") != "balanced_routing.BalancedTopKRouting"
            or identity.get("sha256") != BALANCED_ROUTING_SHA256
            or not identity.get("origin", "").endswith("/balanced_routing.py")
            or not identity.get("routers") or set(identity["routers"].values()) != {
                "vllm.model_executor.layers.fused_moe.router.routing_simulator_router.RoutingSimulatorRouter"}
            or identity.get("runners") != {
                name: "vllm.model_executor.layers.fused_moe.runner.moe_runner.MoERunner" for name in identity["routers"]}
            or set(identity.get("native_sources", {})) != {name.replace(".", "/") + ".py" for name in ROUTING_NATIVE_MODULES}):
        raise ValueError("Missing or mismatched constructed balanced routing identity")
    for name, source in identity["native_sources"].items():
        origin, digest = source.get("origin", ""), source.get("sha256", "")
        if (not Path(origin).is_absolute() or not origin.endswith("/" + name)
                or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError(f"Missing or mismatched balanced native source identity: {name}")
    return dict(routing_runtime_path=identity["runtime_path"], routing_strategy_sha256=identity["sha256"],
        moe_routing=canonical_json(identity))


_ENGINE_COLUMNS = (
    "attention_backend",
    "attention_impl",
    "swa_backend",
    "indexer_backend",
    "compressor_backend",
    "attention_kv_cache_dtype",
    "indexer_uses_fp4_kv",
    "padded_heads",
    "aux_stream_count",
    "multi_stream_gemm_token_threshold",
)


def _variant_metadata(step: Mapping[str, Any], variant: str) -> dict[str, dict[str, Any]]:
    metadata = {}
    for source in METADATA_SOURCES_BY_VARIANT[variant]:
        value = step.get(metadata_key(variant, source))
        if value is None:
            raise ValueError(
                f"Step {step['step_index']} of {step['case_id']} lacks the native "
                f"{source} metadata of the {variant} layers"
            )
        metadata[source] = value
    return metadata


def _index_scores_executed(
    plan: Dsv4ProfilePlan, step: Mapping[str, Any], requests: Sequence[StepRequest]
) -> bool:
    """Whether the step's CSA layers score and select compressed entries.

    vLLM decides from the indexer metadata's ``max_seq_len``; with the V2
    model runner and synchronous scheduling it equals the scheduled
    requests' largest key length, which the row's length columns use.
    """

    max_seq_len = _variant_metadata(step, CSA_VARIANT)["indexer"]["max_seq_len"]
    max_key_len = max(r.num_computed_tokens + r.num_scheduled_tokens for r in requests)
    if max_seq_len != max_key_len:
        raise ValueError(
            f"Step {step['step_index']} of {step['case_id']}: indexer max_seq_len "
            f"{max_seq_len} differs from the scheduled max key length {max_key_len}"
        )
    return plan.shape.index_scores_executed(max_seq_len)


def _validate_step_calls(
    plan: Dsv4ProfilePlan,
    step_key: tuple,
    calls: Sequence[Mapping[str, Any]],
    index_scores_executed: bool,
) -> None:
    seen: dict[int, list[str]] = defaultdict(list)
    for call in calls:
        seen[call["global_layer_id"]].append(call["scope"])
    for layer in plan.shape.layers:
        observed = sorted(seen.pop(layer.global_layer_id, []))
        expected = sorted(
            expected_scopes(
                layer.attention_variant,
                index_scores_executed,
                annotated_only=step_key[0] == KERNEL_ONLY,
            )
        )
        if observed != expected:
            raise ValueError(
                f"Step {step_key}: layer {layer.global_layer_id} "
                f"({layer.attention_variant}) ran scopes {observed}, expected {expected}"
            )
    if seen:
        raise ValueError(f"Step {step_key}: calls for unknown layers {sorted(seen)}")


def _model_columns(plan: Dsv4ProfilePlan, engine: Mapping[str, Any]) -> dict[str, Any]:
    shape = plan.shape
    return {
        "model_name": shape.model_name,
        "model_arch": shape.model_arch,
        "model_architecture_profile": shape.model_architecture_profile,
        "attention_family": "deepseek_v4_attention",
        **{name: engine[name] for name in _ENGINE_COLUMNS},
        "profiling_precision": PrecisionType.from_string(shape.dtype).name,
        "n_embd": shape.hidden_size,
        "n_q_head": shape.num_attention_heads,
        "n_q_head_per_rank": shape.heads_per_rank,
        "num_tensor_parallel_workers": shape.target_tensor_parallel_size,
        "q_lora_rank": shape.q_lora_rank,
        "o_lora_rank": shape.o_lora_rank,
        "o_groups": shape.o_groups,
        "o_groups_per_rank": shape.o_groups_per_rank,
        "head_dim": shape.head_dim,
        "qk_rope_head_dim": shape.qk_rope_head_dim,
        "sliding_window": shape.sliding_window,
        "index_topk": shape.index_topk,
        "index_n_heads": shape.index_n_heads,
        "index_head_dim": shape.index_head_dim,
        "block_size": plan.limits.block_size,
        "kv_cache_dtype": engine["kv_cache_dtype"],
        "max_model_len": plan.limits.max_context_len,
        "max_num_batched_tokens": engine["chunk_budget"],
        "max_num_seqs": plan.limits.max_num_seqs,
        **(routing_columns(engine["moe_routing"]) if engine.get("moe_routing") is not None else {}),
    }


def _key_length_columns(
    plan: Dsv4ProfilePlan, layer: Dsv4Layer, max_key_len: int, index_scores_executed: bool
) -> dict[str, Any]:
    """Key lengths the variant's sparse attention reads for the longest request."""

    shape = plan.shape
    columns: dict[str, Any] = {
        "swa_key_len": min(max_key_len, shape.sliding_window),
        "compressed_key_len": 0,
        "selected_compressed_key_len": 0,
        "index_scores_executed": False,
    }
    if layer.attention_variant == CSA_VARIANT:
        compressed = max_key_len // shape.csa_compress_ratio
        columns["compressed_key_len"] = compressed
        columns["selected_compressed_key_len"] = min(compressed, shape.index_topk)
        columns["index_scores_executed"] = index_scores_executed
    elif layer.attention_variant == HCA_VARIANT:
        compressed = max_key_len // shape.hca_compress_ratio
        columns["compressed_key_len"] = compressed
        columns["selected_compressed_key_len"] = compressed
    elif layer.attention_variant != UNCOMPRESSED_VARIANT:
        raise ValueError(f"Unknown DeepSeek-V4 attention variant {layer.attention_variant!r}")
    return columns


def build_dsv4_rows(
    plan: Dsv4ProfilePlan, records: Iterable[Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Group raw records into rows, keyed by measurement type."""

    require_scope_timing(plan.limits.engine_policy)
    engines, steps, calls = index_records(record for record in records
                                         if record["kind"] != "call" or not is_step_scope(record))
    for engine in engines.values():
        require_realized_kv_cache_groups(plan, engine)
    identities ={canonical_json(engine.get("moe_routing")) for engine in engines.values()}
    if len(identities) > 1:
        raise ValueError("Raw engines disagree on constructed routing identity")
    groups: dict[tuple, _RowAccumulator] = {}
    for key, step in sorted(steps.items()):
        measurement, engine_index, step_index = key
        engine = engines[engine_index]
        step_calls = calls.get(key, [])
        requests = _requests(step)
        index_scores_executed = _index_scores_executed(plan, step, requests)
        _validate_step_calls(plan, key, step_calls, index_scores_executed)
        signature = _request_signature(requests)
        load, context = batch_load_columns(requests)
        for call in step_calls:
            layer = plan.layer(call["global_layer_id"])
            variant = layer.attention_variant
            metadata = _variant_metadata(step, variant)
            group_key = (
                measurement,
                engine_index,
                step["case_id"],
                variant,
                layer.plans_tile_schedule,
                signature,
                canonical_json(metadata),
            )
            accumulator = groups.get(group_key)
            if accumulator is None:
                template = {
                    "measurement_type": measurement,
                    "case_id": step["case_id"],
                    "phase": step["phase"],
                    "attention_variant": variant,
                    "compress_ratio": max(1, layer.compress_ratio),
                    "plans_tile_schedule": layer.plans_tile_schedule,
                    **_model_columns(plan, engine),
                    **load,
                    **_key_length_columns(
                        plan, layer, load["max_key_len"], index_scores_executed
                    ),
                    **context,
                    **{
                        f"{source}_metadata.{name}": value
                        for source, fields in metadata.items()
                        for name, value in sorted(fields.items())
                    },
                }
                accumulator = groups[group_key] = _RowAccumulator(template)
            accumulator.times[call["scope"]].append(float(call["time_ms"]))
            if measurement == KERNEL_ONLY:
                accumulator.kernel_counts[call["scope"]].add(int(call["kernel_count"]))
            accumulator.layers.add(layer.global_layer_id)
            accumulator.steps.add(step_index)

    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in MEASUREMENT_TYPES}
    for group_key, accumulator in groups.items():
        row = dict(accumulator.template)
        variant = row["attention_variant"]
        layers = sorted(accumulator.layers)
        row["global_layer_ids"] = json.dumps(layers)
        row["num_layers_sampled"] = len(layers)
        row["num_steps_sampled"] = len(accumulator.steps)
        scopes = [scope for scope in DSV4_SCOPE_NAMES if accumulator.times.get(scope)]
        row["operator_scopes"] = canonical_json(
            {scope: list(scope_operators(scope, variant)) for scope in scopes}
        )
        for scope in scopes:
            row.update(
                scope_columns(
                    scope,
                    accumulator.times[scope],
                    accumulator.kernel_counts.get(scope),
                    group_key[:5],
                )
            )
        rows[group_key[0]].append(row)
    for measurement in MEASUREMENT_TYPES:
        rows[measurement].sort(
            key=lambda row: (
                row["case_id"],
                row["attention_variant"],
                not row["plans_tile_schedule"],
                row["prefill_kv_cache_sizes"],
                row["decode_kv_cache_sizes"],
            )
        )
    return rows


__all__ = ["DSV4_ROW_SCHEMA_VERSION", "build_dsv4_rows"]
