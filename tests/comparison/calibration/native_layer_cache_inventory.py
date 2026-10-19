"""Compare Frontier's layer and KV-cache representation with native inventories.

The native side is the retained vLLM v0.28.0 start-up inventory written by
``native_full_structure.py`` (one record per worker rank) and, for GLM-5.3,
the C39 prefill/decode observer and log snapshots. No GPU work is started.

Example:

    PYTHONPATH=$PWD python tests/comparison/calibration/native_layer_cache_inventory.py \\
        --glm-inventory <run05>/full_results_selected_logs181.json.gz \\
        --dsv4-inventory <run01>/nodes/<host0>/result.json \\
        --dsv4-inventory <run01>/nodes/<host1>/result.json \\
        --glm-pd-observer <c39>/observer_snapshot.json \\
        --glm-pd-native-log <c39>/native_log_snapshot.json \\
        --output-dir <dir>

Outputs per model: ``<model>_layers.csv``, ``<model>_cache_owners.csv`` and
``<model>_cache_groups.csv``; plus ``checks.csv`` and ``checks.json``. Status
``PASS``/``FAIL`` compares a Frontier value with a native value; ``NOTE``
records a native fact that has no Frontier counterpart. The exit code is 1
when any check fails.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import gzip
import json
import os
from pathlib import Path
import re
import sys
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
LAYER_NAME = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
MULTIMODAL_NAME = re.compile(r"vision|visual|image|multi_?modal|audio|mm_projector", re.I)
KV_CACHE_SIZE_LOG = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
BLOCK_OVERRIDE_LOG = re.compile(r"Overriding num_gpu_blocks=(\d+) with num_gpu_blocks_override=(\d+)")
DTYPE_BYTES = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4, "torch.uint8": 1}
NATIVE_FFN_KIND = {"DeepseekV2MLP": "dense", "DeepseekV2MoE": "moe", "DeepseekV4MoE": "moe"}
FRONTIER_DEVICE = {"NVIDIA H800": ("h800", "h800_dgx")}
# vLLM v0.28.0 BlockPool always takes block 0 as its null block (block_pool.py:190).
NATIVE_NULL_BLOCKS = 1
NULL_BLOCK_SOURCE = "vllm/v1/core/block_pool.py:190"


@dataclass(frozen=True)
class ModelCase:
    key: str
    model_name: str
    attention_module: str
    ffn_module: str
    indexer_module: str
    # Frontier operator-name prefix of the indexer work in a layer manifest.
    indexer_operator_prefix: str
    # Shared-index layers consume the last producer's top-k buffer.
    index_reuse: bool
    indexer_executed_when_built: str
    short_prefill_skip_when_built: str
    native_attention_variant: Callable[[dict], str]
    configuration_variant: Callable[[dict, int], str]
    cache_components: dict
    build_frontier: Callable[[dict, str], tuple]
    compare_routing: bool


def _glm_configuration_variant(hf: dict, layer: int) -> str:
    """vLLM deepseek_v2.py:1087-1098 index-reuse rule."""
    frequency = hf.get("index_topk_freq") or 1
    offset = hf.get("index_skip_topk_offset")
    offset = 2 if offset is None else offset
    pattern = hf.get("index_topk_pattern")
    if pattern is None:
        shared = max(layer - offset + 1, 0) % frequency != 0
    else:
        shared = layer < len(pattern) and pattern[layer] == "S"
    return "shared_index" if shared else "full_index"


DSV4_VARIANT_BY_RATIO = {0: "uncompressed", 4: "csa", 128: "hca"}
# (has compressor module, has indexer module) -> attention variant.
DSV4_VARIANT_BY_MODULES = {
    (False, False): "uncompressed", (True, False): "hca", (True, True): "csa",
}


def _build_glm(reference: dict, device: str) -> tuple:
    from frontier.attention.memory import get_attention_kv_cache_capacity
    from frontier.config.model_config import BaseModelConfig

    model = BaseModelConfig.create_from_name(GLM.model_name)
    block_size = reference["initialization"]["resolved"]["cache_config"]["block_size"]
    capacity = get_attention_kv_cache_capacity(
        model.get_attention_family(), model, block_size=block_size,
    )
    return model, capacity


def _build_dsv4(reference: dict, device: str) -> tuple:
    from frontier.attention.families import DEEPSEEK_V4_ATTENTION_FAMILY
    from frontier.attention.memory import (
        AttentionKVCacheSelection,
        get_attention_kv_cache_capacity,
    )
    from frontier.config import ReplicaConfig

    groups = reference["groups"]
    block_size = reference["initialization"]["resolved"]["cache_config"]["block_size"]
    device_name, network_device = FRONTIER_DEVICE[device]
    # Pinned target selection names; the native run resolves them to
    # FLASHMLA_SPARSE_DSV4 with fp8_ds_mla main and fp8 indexer caches.
    replica = ReplicaConfig(
        model_name=DSV4.model_name, device=device_name, network_device=network_device,
        attn_tensor_parallel_size=groups["tp"]["world_size"],
        attn_dp=groups["dp"]["world_size"],
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=groups["ep"]["world_size"],
        attention_backend="FLASHMLA_SPARSE_DSV4",
        kv_cache_dtype="fp8", indexer_kv_cache_dtype="fp8",
    )
    capacity = get_attention_kv_cache_capacity(
        DEEPSEEK_V4_ATTENTION_FAMILY, replica.model_config, block_size=block_size,
        selection=AttentionKVCacheSelection.from_replica(replica, block_size=block_size),
    )
    return replica.model_config, capacity


GLM = ModelCase(
    key="glm53", model_name="zai-org/GLM-5.3",
    attention_module="self_attn", ffn_module="mlp", indexer_module="self_attn.indexer",
    indexer_operator_prefix="attn_dsa_indexer", index_reuse=True,
    # Built indexers can skip scoring for short prefills (mla.py:128-145,
    # deepseek_v2.py:1178); no retained per-step evidence shows which ran.
    indexer_executed_when_built="unknown_built_conditional",
    short_prefill_skip_when_built="eligible",
    native_attention_variant=lambda modules: (
        "full_index" if "self_attn.indexer" in modules else "shared_index"
    ),
    configuration_variant=_glm_configuration_variant,
    cache_components={
        ("self_attn.attn", None): "main_mla",
        ("self_attn.indexer.k_cache", None): "indexer_k_and_scale",
    },
    build_frontier=_build_glm, compare_routing=False,
)
DSV4 = ModelCase(
    key="dsv4", model_name="deepseek-ai/DeepSeek-V4-Flash",
    attention_module="attn", ffn_module="ffn", indexer_module="attn.indexer",
    indexer_operator_prefix="dsv4_index_", index_reuse=False,
    indexer_executed_when_built="unknown_built",
    short_prefill_skip_when_built="not_applicable",
    native_attention_variant=lambda modules: DSV4_VARIANT_BY_MODULES[
        ("attn.compressor" in modules, "attn.indexer" in modules)
    ],
    configuration_variant=lambda hf, layer: DSV4_VARIANT_BY_RATIO[hf["compress_ratios"][layer]],
    cache_components={
        ("attn", "csa"): "main_csa", ("attn", "hca"): "main_hca",
        ("attn.indexer.k_cache", "csa"): "indexer_csa",
        ("attn.swa_cache", None): "swa",
        ("attn.indexer.compressor.state_cache", "csa"): "indexer_csa_state",
        ("attn.compressor.state_cache", "csa"): "main_csa_state",
        ("attn.compressor.state_cache", "hca"): "main_hca_state",
    },
    build_frontier=_build_dsv4, compare_routing=True,
)


@dataclass
class Check:
    model: str
    area: str
    name: str
    status: str
    frontier: str
    native: str
    evidence: str


class Checks:
    def __init__(self) -> None:
        self.rows: list[Check] = []

    def compare(self, model, area, name, frontier, native, evidence) -> bool:
        status = "PASS" if frontier == native else "FAIL"
        self.rows.append(Check(model, area, name, status, str(frontier), str(native), evidence))
        return status == "PASS"

    def require(self, model, area, name, passed, detail, evidence) -> None:
        self.rows.append(Check(model, area, name, "PASS" if passed else "FAIL",
                               "", str(detail), evidence))

    def note(self, model, area, name, native, evidence) -> None:
        self.rows.append(Check(model, area, name, "NOTE", "", str(native), evidence))


def load_ranks(paths: list[Path]) -> tuple[list[dict], float, str]:
    """Return all worker ranks, the engine memory utilization and the GPU name."""
    results = []
    for path in paths:
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as handle:
                results.extend(node["result"] for node in json.load(handle)["nodes"])
        else:
            results.append(json.loads(path.read_text()))
    utilizations = {result["configuration"]["llm"]["gpu_memory_utilization"] for result in results}
    gpus = {gpu["name"] for result in results for gpu in result["gpus"]}
    if len(utilizations) != 1 or len(gpus) != 1:
        raise ValueError(f"Inventories disagree: utilization={utilizations} gpus={gpus}")
    ranks = [rank for result in results for rank in result["inventory"]]
    return ranks, utilizations.pop(), gpus.pop()


def rank_label(rank: dict) -> str:
    return f"{rank['host'].split('.')[0]}:{rank['worker_rank']}"


def short_class(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def layer_view(rank: dict) -> tuple[dict, dict]:
    modules: dict[int, dict] = {}
    for name, cls in rank["initialization"]["modules"].items():
        match = LAYER_NAME.match(name)
        if match:
            modules.setdefault(int(match[1]), {})[match[2]] = short_class(cls)
    tensors: dict[int, set] = {}
    for tensor in rank["initialization"]["tensors"]:
        match = LAYER_NAME.match(tensor["name"])
        if match:
            tensors.setdefault(int(match[1]), set()).add(match[2])
    return modules, tensors


def native_cache_owners(rank: dict) -> dict[str, dict]:
    owners = {}
    for group_index, group in enumerate(rank["cache"]["kv_cache_groups"]):
        spec = group["kv_cache_spec"]
        specs = spec.get("kv_cache_specs") or {name: spec for name in group["layer_names"]}
        for name in group["layer_names"]:
            owners[name] = dict(specs[name], group=group_index)
    return owners


def native_page(spec: dict) -> tuple[int, int, int]:
    """Return stored rows, content bytes and padded page bytes per block."""
    rows = spec["block_size"] // spec["compress_ratio"]
    if spec["state_content_bytes"]:
        content = rows * spec["state_content_bytes"]
    else:
        content = rows * spec["num_kv_heads"] * spec["head_size"] * DTYPE_BYTES[spec["dtype"]]
    return rows, content, spec["page_size_padded"] or content


def check_rank_identity(case: ModelCase, checks: Checks, ranks: list[dict]) -> None:
    def tensor_shapes(rank):
        return sorted((t["name"], tuple(t["shape"]), t["dtype"])
                      for t in rank["initialization"]["tensors"])

    def tensor_layout(rank):
        num_blocks = rank["cache"]["num_blocks"]
        return [(tuple(t["shared_by"]), t["offset"], t["block_stride"], t["size"] / num_blocks)
                for t in rank["cache"]["kv_cache_tensors"]]

    keys = {
        "modules": lambda rank: rank["initialization"]["modules"],
        "tensor names, shapes and dtypes": tensor_shapes,
        "attention backend groups": lambda rank: rank["attention_groups"],
        "KV cache groups and specs": lambda rank: rank["cache"]["kv_cache_groups"],
        "KV cache tensor layout per block": tensor_layout,
    }
    for name, key in keys.items():
        reference = key(ranks[0])
        differing = [rank_label(rank) for rank in ranks if key(rank) != reference]
        checks.require(case.key, "identity", f"{name} identical on all ranks", not differing,
                       f"{len(ranks)} ranks; differing={differing}",
                       "inventory[*].initialization / cache / attention_groups")


def frontier_layer_rows(case: ModelCase, model_config) -> dict[int, dict]:
    from frontier.operators.binding import build_operator_manifest

    rows = {}
    for spec in model_config.get_layer_attention_specs():
        layer = spec.global_layer_id
        manifest = build_operator_manifest(model_config, global_layer_id=layer)
        families = {binding.family_id: binding.variant_id for binding in manifest.family_bindings}
        indexer_listed = any(operator.name.startswith(case.indexer_operator_prefix)
                             for operator in manifest.operators())
        if "ffn" in families:
            ffn_kind = "dense"
        else:
            ffn_kind = "moe_with_shared" if "share_expert" in families else "moe"
        producer, source = spec.index_producer_layer_id, "LayerAttentionSpec"
        if producer is None and indexer_listed and not case.index_reuse:
            # Without index reuse, a layer that runs indexer work selects its own rows.
            producer, source = layer, "operator_manifest"
        rows[layer] = {
            "global_layer_id": layer,
            "layer_kind": "backbone",
            "attention_variant": spec.variant_id,
            "index_producer_layer": "" if producer is None else producer,
            "index_producer_source": source if producer is not None else "",
            "indexer_operators_listed": indexer_listed,
            "ffn_kind": ffn_kind,
            "routing": families.get("moe", ""),
            "mhc": "present" if "deepseek_v4_mhc" in families else "absent",
            "op_owners": ";".join(
                f"{binding.family_id}:{binding.variant_id}"
                f"[{','.join(operator.name for operator in binding.family.operators)}]"
                for binding in manifest.family_bindings
            ),
        }
    return rows


def native_layer_rows(case: ModelCase, rank: dict) -> dict[int, dict]:
    modules, tensors = layer_view(rank)
    hf = rank["initialization"]["hf_config"]
    rows, producer = {}, None
    for layer in sorted(modules):
        layer_modules, layer_tensors = modules[layer], tensors.get(layer, set())
        built = case.indexer_module in layer_modules
        if built:
            producer = layer
        elif not case.index_reuse:
            producer = None
        ffn_class = layer_modules[case.ffn_module]
        ffn_kind = NATIVE_FFN_KIND[ffn_class]
        if ffn_kind == "moe" and f"{case.ffn_module}.shared_experts" in layer_modules:
            ffn_kind = "moe_with_shared"
        if f"{case.ffn_module}.gate.tid2eid" in layer_tensors:
            routing = "hash"
        elif f"{case.ffn_module}.gate.e_score_correction_bias" in layer_tensors:
            routing = "score"
        else:
            routing = ""
        rows[layer] = {
            "native_attention_variant": case.native_attention_variant(layer_modules),
            "configuration_variant": case.configuration_variant(hf, layer),
            "native_index_producer_layer": "" if producer is None else producer,
            "native_indexer_built": built,
            "indexer_executed": case.indexer_executed_when_built if built else "no_not_built",
            "short_prefill_scoring_skip":
                case.short_prefill_skip_when_built if built else "not_applicable",
            "native_ffn_kind": ffn_kind,
            "native_routing": routing,
            "native_mhc": "present" if {"hc_attn_fn", "hc_ffn_fn"} <= layer_tensors else "absent",
            "native_attention_class": layer_modules[case.attention_module],
            "native_ffn_class": ffn_class,
        }
    return rows


def compare_layers(case: ModelCase, checks: Checks, rank: dict, model_config) -> list[dict]:
    hf = rank["initialization"]["hf_config"]
    num_layers = hf["num_hidden_layers"]
    native = native_layer_rows(case, rank)
    frontier = frontier_layer_rows(case, model_config)
    checks.compare(case.key, "layer", "backbone layer count", len(frontier), len(native),
                   "Frontier get_layer_attention_specs(); native modules model.layers.N")
    checks.compare(case.key, "layer", "native layer ids equal configuration range",
                   list(range(num_layers)), sorted(native), "hf_config.num_hidden_layers")
    names = rank["initialization"]["modules"]
    checks.note(case.key, "layer", "configuration next-token (MTP) layers excluded from backbone",
                f"num_nextn_predict_layers={hf.get('num_nextn_predict_layers')}; "
                f"native modules with layer id >= {num_layers}: "
                f"{[n for n in names if (m := LAYER_NAME.match(n)) and int(m[1]) >= num_layers]}",
                "hf_config; initialization.modules")
    multimodal = sorted(n for n, cls in names.items() if MULTIMODAL_NAME.search(f"{n} {cls}"))
    checks.compare(case.key, "layer", "vision or multimodal modules", [], multimodal,
                   "initialization.modules name/class scan")
    checks.note(case.key, "layer", "non-layer modules",
                sorted(f"{n or '<root>'}={short_class(cls)}" for n, cls in names.items()
                       if not n.startswith("model.layers.") and n != "model.layers"),
                "initialization.modules")
    compared = ["attention_variant", "index_producer_layer", "ffn_kind", "mhc"]
    if case.compare_routing:
        compared.append("routing")
    rows = []
    mismatches = {field: [] for field in compared + ["configuration_variant"]}
    for layer in sorted(set(native) | set(frontier)):
        row = {**frontier.get(layer, {"global_layer_id": layer}), **native.get(layer, {})}
        for field in compared:
            if row.get(field) != row.get(f"native_{field}"):
                mismatches[field].append(layer)
        if row.get("configuration_variant") != row.get("native_attention_variant"):
            mismatches["configuration_variant"].append(layer)
        row["match"] = all(layer not in layers for layers in mismatches.values())
        rows.append(row)
    for field, layers in mismatches.items():
        label = ("configuration rule equals native module structure"
                 if field == "configuration_variant" else f"per-layer {field}")
        checks.require(case.key, "layer", label, not layers,
                       f"{len(rows) - len(layers)}/{len(rows)} layers match; mismatched={layers}",
                       "Frontier manifest/spec vs native modules and tensors")
    listed_without_module = [row["global_layer_id"] for row in rows
                             if row.get("indexer_operators_listed")
                             and not row.get("native_indexer_built")]
    built_without_listing = [row["global_layer_id"] for row in rows
                             if row.get("native_indexer_built")
                             and not row.get("indexer_operators_listed")]
    detail = (f"listed without native module={listed_without_module}; "
              f"native module without listing={built_without_listing}")
    if case.index_reuse:
        # Shared-index layers keep the indexer operator in their family listing;
        # dummy timing charges it only on the producer layer.
        checks.note(case.key, "layer", "indexer operators listed in the layer manifest", detail,
                    "sklearn_execution_time_predictor.py:7502-7518")
    else:
        checks.require(case.key, "layer", "indexer operators listed only where native builds one",
                       not (listed_without_module or built_without_listing), detail,
                       "Frontier manifest operator names vs native indexer modules")
    from_manifest = [row["global_layer_id"] for row in rows
                     if row.get("index_producer_source") == "operator_manifest"]
    if from_manifest:
        checks.note(case.key, "layer",
                    "index producer read from the operator manifest "
                    "(LayerAttentionSpec.index_producer_layer_id is None)",
                    from_manifest, "frontier/attention/model_binding.py:308 builds the spec without a producer")
    counts = Counter(row.get("native_attention_variant") for row in rows)
    checks.note(case.key, "layer", "native attention variant counts", dict(counts),
                "initialization.modules")
    counts = Counter((row.get("native_ffn_kind"), row.get("native_routing")) for row in rows)
    checks.note(case.key, "layer", "native FFN kind and routing counts",
                {f"{kind}/{routing or '-'}": n for (kind, routing), n in counts.items()},
                "initialization.modules/tensors")
    return rows


def compare_cache(case: ModelCase, checks: Checks, ranks: list[dict], capacity,
                  variants: dict[int, str], utilization: float,
                  gpu_name: str) -> tuple[list, list]:
    from frontier.config import device_sku_config

    reference = ranks[0]
    owners = native_cache_owners(reference)
    frontier_components = {
        (component.component_id, layer): component
        for component in capacity.components for layer in component.owner_layer_ids
    }
    owner_rows, native_identity, unmapped = [], {}, []
    def owner_order(name):
        match = LAYER_NAME.match(name)
        return int(match[1]), match[2]

    for name in sorted(owners, key=owner_order):
        spec = owners[name]
        match = LAYER_NAME.match(name)
        layer, suffix = int(match[1]), match[2]
        component_id = (case.cache_components.get((suffix, variants[layer]))
                        or case.cache_components.get((suffix, None)))
        if component_id is None:
            unmapped.append(name)
            continue
        native_identity[name] = (component_id, layer)
        rows, content, page = native_page(spec)
        component = frontier_components.get((component_id, layer))
        owner_rows.append({
            "native_owner": name, "global_layer_id": layer, "frontier_component": component_id,
            "native_group": spec["group"], "native_block_size": spec["block_size"],
            "native_dtype": spec["dtype"], "native_cache_dtype": spec["cache_dtype_str"],
            "native_head_size": spec["head_size"], "native_compress_ratio": spec["compress_ratio"],
            "native_sliding_window": spec["sliding_window"],
            "native_rows": rows, "frontier_rows": getattr(component, "stored_rows", ""),
            "native_content_bytes": content,
            "frontier_content_bytes": getattr(component, "content_size_bytes", ""),
            "native_page_bytes": page,
            "frontier_page_bytes": getattr(component, "page_size_bytes", ""),
            "match": component is not None and component.page_size_bytes == page
            and component.stored_rows in (None, rows)
            and component.content_size_bytes in (None, content),
        })
    checks.compare(case.key, "cache", "native owners without a Frontier component", [], unmapped,
                   "cache.kv_cache_groups[*].layer_names")
    checks.compare(case.key, "cache", "cache owner set (component, layer)",
                   sorted(frontier_components), sorted(native_identity.values()),
                   "Frontier capacity.components vs native kv_cache_specs")
    bad = [row["native_owner"] for row in owner_rows if not row["match"]]
    checks.require(case.key, "cache", "per-owner rows, content and padded page bytes", not bad,
                   f"{len(owner_rows) - len(bad)}/{len(owner_rows)} owners match; mismatched={bad}",
                   "native kv_cache_specs page_size_padded/head_size/dtype")
    page_by_name = {row["native_owner"]: row["native_page_bytes"] for row in owner_rows}

    # Allocation groups and their per-block bytes.
    group_rows = []
    native_groups = reference["cache"]["kv_cache_groups"]
    frontier_groups = {frozenset(group.component_owners): group for group in capacity.groups}
    for index, group in enumerate(native_groups):
        identities = frozenset(native_identity[name] for name in group["layer_names"]
                               if name in native_identity)
        windows = {owners[name]["sliding_window"] for name in group["layer_names"]}
        frontier_group = frontier_groups.get(identities)
        group_rows.append({
            "native_group": index, "num_owners": len(group["layer_names"]),
            "native_block_size": group["kv_cache_spec"]["block_size"],
            "native_sliding_window": "/".join(str(w) for w in sorted(windows, key=str)),
            "native_group_bytes": sum(page_by_name[name] for name in group["layer_names"]),
            "frontier_group": frontier_group.group_id if frontier_group else "",
            "frontier_logical_block_size": frontier_group.logical_block_size if frontier_group else "",
            "frontier_window_tokens": frontier_group.window_tokens if frontier_group else "",
            "frontier_group_bytes": capacity.get_group_size_bytes(frontier_group)
            if frontier_group else "",
        })
    if capacity.is_grouped:
        checks.compare(case.key, "cache", "allocation group count", len(capacity.groups),
                       len(native_groups), "Frontier capacity.groups vs native kv_cache_groups")
        bad = [row["native_group"] for row in group_rows if not (
            row["frontier_group"]
            and row["frontier_logical_block_size"] == row["native_block_size"]
            and str(row["frontier_window_tokens"]) == row["native_sliding_window"]
            and row["frontier_group_bytes"] == row["native_group_bytes"])]
        checks.require(case.key, "cache", "group owners, block size, window and bytes", not bad,
                       f"mismatched native groups={bad}", "Frontier groups vs native groups")
        strides = {t["block_stride"] for t in reference["cache"]["kv_cache_tensors"]}
        native_block_bytes = strides.pop() if len(strides) == 1 else sorted(strides)
    else:
        checks.compare(case.key, "cache", "allocation group count (one uniform-type group)",
                       1, len(native_groups), "Frontier ungrouped additive capacity")
        unwindowed = all(row["native_sliding_window"] == "None" for row in group_rows)
        checks.require(case.key, "cache", "every owner keeps full history (no sliding window)",
                       unwindowed, [row["native_sliding_window"] for row in group_rows],
                       "native kv_cache_specs sliding_window")
        native_block_bytes = sum(page_by_name.values())
    checks.compare(case.key, "cache", "bytes per common block (block-count divisor)",
                   capacity.physical_block_size_bytes, native_block_bytes,
                   "Frontier physical_block_size_bytes; native page sum or block_stride")

    # Tensor placement and aliases on the reference rank.
    num_blocks = reference["cache"]["num_blocks"]
    tensors = reference["cache"]["kv_cache_tensors"]
    owner_tensor = Counter(name for t in tensors for name in t["shared_by"])
    checks.require(case.key, "cache", "each owner registered in exactly one tensor",
                   set(owner_tensor) == set(owners) and set(owner_tensor.values()) == {1},
                   f"{len(tensors)} tensors for {len(owners)} owners",
                   "cache.kv_cache_tensors[*].shared_by")
    if capacity.is_grouped:
        group_of = {name: spec["group"] for name, spec in owners.items()}
        bad_size = [t["offset"] for t in tensors if t["size"] != num_blocks * native_block_bytes]
        overflow = [name for t in tensors for name in t["shared_by"]
                    if t["offset"] + page_by_name[name] > native_block_bytes]
        overlap = []
        for index in range(len(native_groups)):
            spans = sorted((t["offset"], t["offset"] + page_by_name[name])
                           for t in tensors for name in t["shared_by"] if group_of[name] == index)
            overlap += [index for a, b in zip(spans, spans[1:]) if b[0] < a[1]]
        same_group_alias = [t["shared_by"] for t in tensors
                            if len({group_of[name] for name in t["shared_by"]}) < len(t["shared_by"])]
        checks.require(case.key, "cache",
                       "packed views: size=num_blocks*stride, pages fit the stride, "
                       "no overlap inside a group, aliases only across groups",
                       not (bad_size or overflow or overlap or same_group_alias),
                       f"bad_size={bad_size} overflow={overflow} overlap={overlap} "
                       f"same_group_alias={same_group_alias}", "cache.kv_cache_tensors")
        aliases = [t for t in tensors if len(t["shared_by"]) > 1]
        checks.note(case.key, "cache", "cross-group aliased byte ranges (not represented in Frontier)",
                    f"{len(aliases)} of {len(tensors)} tensors alias "
                    f"{sum(len(t['shared_by']) for t in aliases)} owners; "
                    f"distinct offsets={len({t['offset'] for t in tensors})}",
                    "cache.kv_cache_tensors[*].offset/shared_by")
    else:
        bad = [t["shared_by"] for t in tensors if len(t["shared_by"]) != 1 or t["offset"]
               or t["block_stride"] or t["size"] != num_blocks * page_by_name[t["shared_by"][0]]]
        checks.require(case.key, "cache",
                       "one unaliased tensor per owner, size=num_blocks*page", not bad,
                       f"bad={bad}", "cache.kv_cache_tensors")

    # Block count: vLLM uses the minimum over one engine's workers.
    lanes: dict[tuple, list] = {}
    for rank in ranks:
        lanes.setdefault(tuple(rank["groups"]["tp"]["ranks"]), []).append(rank)
    lane_rows, bad = [], []
    for lane, members in lanes.items():
        floors = [rank["memory"]["native_scalars"]["available_kv_cache_memory_bytes"]
                  // native_block_bytes for rank in members]
        native_blocks = {rank["cache"]["num_blocks"] for rank in members}
        if native_blocks != {min(floors)}:
            bad.append(lane)
        lane_rows.append(f"host={members[0]['host'].split('.')[0]} floors={floors} "
                         f"min={min(floors)} native={sorted(native_blocks)}")
    checks.require(case.key, "cache",
                   "num_blocks = min over the engine's workers of floor(available/bytes per block)",
                   not bad, " | ".join(lane_rows), "memory.native_scalars; cache.num_blocks")
    scalars = [rank["memory"]["native_scalars"] for rank in ranks]
    formula = all(s["available_kv_cache_memory_bytes"]
                  == s["requested_memory"] - s["total_consumed"] - s["peak_activation_memory"]
                  for s in scalars)
    checks.require(case.key, "cache",
                   "available = requested - total_consumed - peak_activation (CUDA graph estimate "
                   "not subtracted)", formula,
                   f"cudagraph_memory_estimate={sorted({s['cudagraph_memory_estimate'] for s in scalars})}",
                   "memory.native_scalars")
    checks.note(case.key, "cache", "Frontier MemoryPlanner.get_num_blocks uses one device value",
                "no per-worker minimum; vLLM takes min over workers (kv_cache_utils.py:2082)",
                "memory_planner.py:234-328")

    # Requested memory: Frontier SKU total versus the native per-node device total.
    sku = getattr(device_sku_config, f"{FRONTIER_DEVICE[gpu_name][0].upper()}DeviceSKUConfig")()
    frontier_requested = int(sku.total_memory_gb * 1024**3 * utilization)
    by_host = sorted({(rank["host"].split(".")[0], rank["memory"]["native_scalars"]["requested_memory"])
                      for rank in ranks})
    for host, requested in by_host:
        gap = frontier_requested - requested
        checks.compare(case.key, "cache", f"requested memory bytes at utilization {utilization} ({host})",
                       frontier_requested, requested,
                       "memory_planner.py:57 total_memory_gb*1024**3*utilization; "
                       f"device_sku_config.py:59 total_memory_gb={sku.total_memory_gb}; native device "
                       f"total = requested/utilization = {requested / utilization / 1024**3:.2f} GiB; "
                       f"gap {gap} B = {gap / native_block_bytes:.1f} blocks")

    checks.compare(case.key, "cache",
                   "blocks withheld from allocation (vLLM null block) out of num_blocks",
                   capacity.reserved_blocks, NATIVE_NULL_BLOCKS,
                   f"Frontier capacity.reserved_blocks used by memory_planner.py:328 and "
                   f"vllm_v1_kv_allocation.py:71-77; native {NULL_BLOCK_SOURCE}")
    return owner_rows, group_rows


def check_pd_snapshots(checks: Checks, observer_path: Path, log_path: Path,
                       capacity, block_size: int) -> None:
    """Check C39 GLM PD descriptor and block-pool evidence against Frontier."""
    observer = json.loads(observer_path.read_text())
    events = {role: [event for file in value["files"] for event in file["selected_events"]]
              for role, value in observer["roles"].items()}
    owners = sum(len(component.owner_layer_ids) for component in capacity.components)
    reads = [event for event in events["decode"]
             if event.get("kind") == "nixl_read" and event.get("phase") == "prepare_begin"]
    bad = [event["read_sequence"] for event in reads
           if any(shape != [block * owners for block in event["local_block_counts"]]
                  for shape in event["descriptor_shapes"])]
    checks.require("glm53", "pd", "NIXL descriptors per read = blocks x Frontier cache owners",
                   bool(reads) and not bad,
                   f"{len(reads)} reads; shapes={sorted({json.dumps(e['descriptor_shapes']) for e in reads})}; "
                   f"blocks={sorted({json.dumps(e['local_block_counts']) for e in reads})}; "
                   f"owners={owners}; bad={bad}", "observer decode nixl_read prepare_begin")

    decode = [event for event in events["decode"] if event.get("kind") == "scheduler_request"]
    totals = {event["num_gpu_blocks"] for event in decode}
    held = [event for event in decode if event.get("block_ids") and "free_blocks" in event]
    ids = {block for event in held for group in event["block_ids"] for block in group}
    # Blocks neither free nor held by the observed request; the minimum is the
    # pool's permanent reservation.
    withheld = min(event["num_gpu_blocks"] - event["free_blocks"]
                   - sum(len(group) for group in event["block_ids"]) for event in held)
    checks.compare("glm53", "pd", "decode blocks never free and never allocated (actual)",
                   capacity.reserved_blocks, withheld,
                   f"Frontier capacity.reserved_blocks; native num_gpu_blocks={sorted(totals)} "
                   f"allocated_ids={min(ids)}..{max(ids)} block_0_allocated={0 in ids}; "
                   "observer decode scheduler_request")
    reservations = [event for event in decode if event.get("phase") == "remote_reservation"
                    and event.get("allocated")]
    applied = [event for event in decode if event.get("phase") == "remote_receive_applied"]
    finished = [event for event in decode if event.get("phase") == "request_free_begin"]
    checks.note("glm53", "pd", "decode logical token range per request",
                f"external_tokens={sorted({e['num_external_computed_tokens'] for e in reservations})} "
                f"reserved_blocks={sorted({len(e['block_ids'][0]) for e in reservations})} "
                f"computed_after_receive={sorted({e['num_computed_tokens'] for e in applied})} "
                f"computed_at_free={sorted({e['num_computed_tokens'] for e in finished})} "
                f"blocks_at_free={sorted({len(e['block_ids'][0]) for e in finished})}",
                "observer decode scheduler_request phases")

    prefill_totals = sorted({event["num_gpu_blocks"] for event in events["prefill"]
                             if event.get("kind") == "scheduler_request"})
    logs = json.loads(log_path.read_text())["nodes"]
    log_text = {host.split(".")[0]: json.dumps(node["selected"]) for host, node in logs.items()}
    logged = {host: sorted({int(v.replace(",", "")) for v in KV_CACHE_SIZE_LOG.findall(text)})
              for host, text in log_text.items()}
    overrides = {host: sorted(set(BLOCK_OVERRIDE_LOG.findall(text)))
                 for host, text in log_text.items()}
    expected = sorted({total * block_size for total in prefill_totals} | {
        total * block_size for total in totals})
    checks.compare("glm53", "pd", "logged KV cache tokens = num_gpu_blocks x block_size",
                   expected, sorted({v for values in logged.values() for v in values}),
                   f"native logs {logged}; prefill num_gpu_blocks={prefill_totals}")
    checks.note("glm53", "pd", "decode per-worker block counts before override",
                {host: values for host, values in overrides.items() if values},
                "native log 'Overriding num_gpu_blocks'")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_case(case: ModelCase, checks: Checks, paths: list[Path], output: Path) -> tuple:
    ranks, utilization, gpu_name = load_ranks(paths)
    check_rank_identity(case, checks, ranks)
    reference = ranks[0]
    checks.note(case.key, "identity", "native ranks", [rank_label(rank) for rank in ranks],
                "inventory[*].host/worker_rank")
    model_config, capacity = case.build_frontier(reference, gpu_name)
    layer_rows = compare_layers(case, checks, reference, model_config)
    variants = {row["global_layer_id"]: row["native_attention_variant"] for row in layer_rows}
    owner_rows, group_rows = compare_cache(case, checks, ranks, capacity, variants,
                                           utilization, gpu_name)
    write_csv(output / f"{case.key}_layers.csv", layer_rows)
    write_csv(output / f"{case.key}_cache_owners.csv", owner_rows)
    write_csv(output / f"{case.key}_cache_groups.csv", group_rows)
    return capacity, reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--glm-inventory", type=Path, required=True)
    parser.add_argument("--dsv4-inventory", type=Path, action="append", required=True)
    parser.add_argument("--glm-pd-observer", type=Path, required=True)
    parser.add_argument("--glm-pd-native-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = [args.glm_inventory, *args.dsv4_inventory, args.glm_pd_observer,
              args.glm_pd_native_log]
    inputs = [path.resolve() for path in inputs]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Frontier resolves model configuration files relative to the repository root.
    os.chdir(REPO_ROOT)

    checks = Checks()
    glm_capacity, glm_reference = run_case(GLM, checks, [inputs[0]], output)
    run_case(DSV4, checks, inputs[1:-2], output)
    block_size = glm_reference["initialization"]["resolved"]["cache_config"]["block_size"]
    check_pd_snapshots(checks, inputs[-2], inputs[-1], glm_capacity, block_size)

    rows = [asdict(row) for row in checks.rows]
    write_csv(output / "checks.csv", rows)
    summary = Counter(row["status"] for row in rows)
    (output / "checks.json").write_text(json.dumps({
        "inputs": [str(path) for path in inputs], "summary": dict(summary), "checks": rows,
    }, indent=2) + "\n")
    for row in rows:
        if row["status"] == "FAIL":
            print(f"FAIL {row['model']} {row['area']}: {row['name']} "
                  f"frontier={row['frontier']} native={row['native']}")
    print(json.dumps(dict(summary)))
    return 1 if summary["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
