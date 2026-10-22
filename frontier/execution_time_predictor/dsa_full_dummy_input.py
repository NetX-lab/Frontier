"""Read qualified FULL-dummy scopes separately from ordinary DSA serving rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from frontier.attention.dsa_features import DSACoverageError
from frontier.attention.model_binding import bind_layer_attention
from frontier.config import global_vars


SCHEMA = "dsa_full_dummy.v1"
# This reader interprets the invalid-slot replay contract of this native revision.
NATIVE_REVISION = "2cf0a6915ce544dc493a0990f2ea38d81601128a"
SCOPES = {"full_index": ("attn_dsa_indexer", "attn_dsa_mla"),
          "shared_index": ("attn_dsa_mla",)}


def _require(condition, message):
    if not condition:
        raise DSACoverageError(f"DSA FULL dummy coverage missing: {message}")


def _read_reference(directory, reference):
    raw = (directory / reference["path"]).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == reference["sha256"], "evidence digest")
    return raw


def load_full_dummy_rows(path: Path, model, tp: int, block_size: int,
                         replica, scheduler, serving: pd.DataFrame) -> list[dict]:
    """Read externally admitted scopes; this reader never qualifies measurements.

    A reviewed admission receipt binds the whole sidecar, effective native config
    and qualification evidence by content. Its PASS is an external trust boundary,
    not a claim that a CPU reader can establish GPU timing correctness.
    """
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
        _require(payload["schema_version"] == SCHEMA, "schema")
        for field in ("gpu_execution_observed", "kernel_only_qualified", "consumer_admission"):
            _require(payload[field] is True, field)
        _require(payload["measurement_type"] == "KERNEL_ONLY", "timing method")
        admission_raw = path.with_suffix(".admission.json").read_bytes()
        admission = json.loads(admission_raw)
        digest = hashlib.sha256(raw).hexdigest()
        _require(admission["schema_version"] == SCHEMA and admission["verdict"] == "PASS"
                 and admission["sidecar_sha256"] == digest, "external qualification receipt")
        native = json.loads(_read_reference(path.parent, admission["native_config"]))
        _require(bool(admission["evidence"]), "qualification evidence")
        for reference in admission["evidence"]:
            _require(bool(_read_reference(path.parent, reference)), "empty qualification evidence")
        identity, plan = payload["identity"], payload["plan"]
        _require(payload["native_revision"] == identity["source_revision"] == NATIVE_REVISION,
                 "native revision")
        _require(identity["schema_version"] == SCHEMA and identity["graph_mode"] == "FULL",
                 "native graph identity")
        _require(replica.device == "h800" and identity["device_name"] == "NVIDIA H800"
                 and identity["device_capability"] == "sm90", "H800 device")
        _require(identity["parallel_groups"] == {
            "engine_tensor_parallel": tp, "target_tensor_parallel": tp}, "physical TP")
        _require(identity["quant_signature"] == model.get_quant_signature(), "quantization")
        _require(identity["weight_source"] == "dummy", "weight source")
        model_path = Path(__file__).resolve().parents[2] / "data/config/models"
        model_path = model_path / f"{model.get_name().replace('/', '__')}.json"
        _require(identity["model_config_sha256"] == payload["model_config_sha256"]
                 == hashlib.sha256(model_path.read_bytes()).hexdigest(), "model source hash")
        versions = json.loads(identity["runtime_versions"])
        _require(versions["vllm"] == "0.28.0", "runtime version")
        for column, value in (("runtime_versions", versions),
                              ("device_name", identity["device_name"]),
                              ("device_capability", identity["device_capability"])):
            _require(column in serving, f"serving identity {column}")
            values = [json.loads(item) if column == "runtime_versions" else item
                      for item in serving[column].unique()]
            _require(values == [value], f"serving identity {column}")
        inventory = payload["inventory"]
        for name in ("attention_backend", "attention_impl"):
            _require(set(serving[name]) == {inventory[name]}, name)
        _require(replica.attention_backend in (None, inventory["attention_backend"]), "selected backend")
        shape, limits = plan["shape"], plan["limits"]
        expected = {"hidden_size": model.embedding_dim, "num_attention_heads": model.num_q_heads,
                    "heads_per_rank": model.num_q_heads // tp, "target_tensor_parallel_size": tp,
                    "model_architecture_profile": model.get_model_architecture_profile().profile_id}
        expected.update({name: getattr(model, name) for name in (
            "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
            "v_head_dim", "index_topk", "index_n_heads", "index_head_dim")})
        _require(all(shape[name] == value for name, value in expected.items()), "model/TP shape")
        _require(not plan["hf_overrides"] and shape["dtype"] == model.torch_dtype,
                 "unmodified native backbone/dtype")
        _require(limits["block_size"] == block_size, "cache block size")
        _require(payload["kv_cache_dtype"] == replica.kv_cache_dtype
                 and replica.kv_cache_dtype in ("auto", "bfloat16")
                 and replica.indexer_kv_cache_dtype == "fp8", "effective cache dtype")
        parallel = native["parallel_config"]
        expected_parallel = {"tensor_parallel_size": tp, "data_parallel_size": replica.attn_dp,
                             "pipeline_parallel_size": replica.num_pipeline_stages,
                             "enable_expert_parallel": model.is_moe,
                             "decode_context_parallel_size": 1, "prefill_context_parallel_size": 1}
        _require(all(parallel[name] == value for name, value in expected_parallel.items())
                 and replica.moe_expert_parallel_size == tp * replica.attn_dp
                 and replica.num_pipeline_stages == 1, "effective physical parallel layout")
        expected_cache = {"block_size": block_size, "cache_dtype": replica.kv_cache_dtype,
                          "num_gpu_blocks": scheduler.num_blocks, "enable_prefix_caching": False}
        _require(all(native["cache_config"][name] == value for name, value in expected_cache.items())
                 and type(scheduler.num_blocks) is int and scheduler.num_blocks > 0
                 and not scheduler.enable_prefix_caching,
                 "effective cache allocation")
        _require(native["model_config"]["dtype"] == model.torch_dtype
                 and native["model_config"]["quantization"] == model.quantization_config.quant_method
                 and native["model_config"]["max_model_len"] == scheduler.max_model_len
                 == limits["max_context_len"], "effective model limit")
        expected_scheduler = {"max_num_seqs": scheduler.batch_size_cap,
                              "max_num_batched_tokens": scheduler.max_tokens_in_batch,
                              "enable_chunked_prefill": False}
        _require(all(native["scheduler_config"][name] == value for name, value in expected_scheduler.items())
                 and limits["max_num_seqs"] == scheduler.batch_size_cap
                 and not scheduler.enable_chunked_prefill
                 and native["speculative_config"] is None
                 and not replica.speculative_decoding_config.enabled, "effective scheduler/speculation")
        compilation = native["compilation_config"]
        # CompilationMode.VLLM_COMPILE is 3 in the pinned native revision.
        _require(compilation["mode"] == 3 and compilation["cudagraph_mode"] == "FULL_DECODE_ONLY"
                 and global_vars.get_decode_cuda_graph_mode() == "full_decode_only",
                 "native compiled FULL graph")
        widths = payload["capture_sizes"]
        _require(bool(widths) and all(type(w) is int and 0 < w <= limits["max_num_seqs"]
                                    for w in widths) and len(set(widths)) == len(widths), "capture widths")
        _require(sorted(widths) == payload["resolved_capture_sizes"], "resolved captures")
        _require(compilation["cudagraph_capture_sizes"] == global_vars.get_cudagraph_capture_sizes()
                 and set(widths) <= set(compilation["cudagraph_capture_sizes"]), "effective captures")
        repeats = limits["decode_measured_steps"]
        _require(type(repeats) is int and repeats > 0, "measured repeats")
        layers = {}
        for layer in shape["layers"]:
            index, variant = layer["global_layer_id"], layer["attention_variant"]
            _require(type(index) is int and 0 <= index < model.num_layers and index not in layers,
                     "measured layer identity")
            spec = bind_layer_attention(model, index)
            expected_variant = "full_index" if spec.index_producer_layer_id == index else "shared_index"
            _require(variant == expected_variant and layer["index_producer_layer"]
                     == spec.index_producer_layer_id, "index ownership")
            layers[index] = variant
        _require(set(layers) == set(range(model.num_layers)), "complete native backbone")
        inventory_layers = {layer["global_layer_id"]: layer for layer in inventory["layers"]}
        _require(len(inventory_layers) == len(inventory["layers"]) == len(layers)
                 and set(inventory_layers) == set(layers), "native layer inventory")
        for index, layer in inventory_layers.items():
            _require(layer["has_indexer"] is (layers[index] == "full_index")
                     and layer["skip_topk"] is (layers[index] == "shared_index"), "scope ownership")
        seen, rows = set(), []
        for row in payload["records"]:
            witness = row["witness"]
            width, index, scope = witness["capture_width"], row["global_layer_id"], row["scope"]
            key = (width, index, scope, row["repeat"])
            _require(type(width) is int and type(index) is int
                     and key not in seen and width in widths and index in layers
                     and scope in SCOPES[layers[index]] and type(row["repeat"]) is int
                     and row["repeat"] in range(repeats), "duplicate/unplanned replay scope")
            seen.add(key)
            _require(row["schema_version"] == SCHEMA and row["measurement_type"] == "KERNEL_ONLY"
                     and row["execution_kind"] == "full_dummy" and row["graph_mode"] == "FULL"
                     and row["attention_variant"] == layers[index], "replay scope identity")
            slots = witness["slot_mappings"]
            _require(witness["actual_query_tokens"] == witness["actual_requests"] == 1
                      and witness["cached_context_tokens"] == 0
                      and witness["seq_lens"] == [1] + [0] * (width - 1)
                      and witness["query_start_loc"] == [0] + [1] * width
                      and bool(slots) and all(value == [-1] * width for value in slots.values()),
                     "invalid-slot/padded metadata")
            layer = inventory_layers[index]
            metadata = witness["metadata"]
            mla = metadata[layer["mla_layer_name"]]
            expected_metadata = {"num_reqs": width, "num_actual_tokens": width,
                "max_query_len": 1, "max_seq_len": scheduler.max_model_len,
                "block_size": block_size, "topk_tokens": model.index_topk,
                "num_decodes": width, "num_prefills": 0, "num_decode_tokens": width,
                "prefill_max_seq_len": 0}
            _require(all(type(mla[name]) is int and mla[name] == value
                         for name, value in expected_metadata.items()), "native MLA metadata")
            if layers[index] == "full_index":
                prefix, suffix = layer["mla_layer_name"].rsplit(".", 1)
                _require(suffix == "attn", "native MLA cache name")
                indexer = metadata[f"{prefix}.indexer.k_cache"]
                expected_indexer = {"max_seq_len": scheduler.max_model_len, "num_decodes": width,
                    "num_decode_tokens": width, "num_prefills": 0, "num_prefill_tokens": 0}
                _require(all(type(indexer[name]) is int and indexer[name] == value
                             for name, value in expected_indexer.items()), "native indexer metadata")
            duration, count = row["time_ms"], row["kernel_count"]
            _require(type(duration) in (int, float) and np.isfinite(duration) and duration > 0
                      and type(count) is int and count > 0,
                      "kernel duration/count")
            rows.append({"attention_variant": layers[index], "capture_width": width,
                          "scope": scope, "time_ms": duration, "kernel_count": count,
                          "profiling_precision": serving["profiling_precision"].iat[0],
                          "measurement_type": payload["measurement_type"],
                          "sidecar_sha256": digest,
                          "admission_sha256": hashlib.sha256(admission_raw).hexdigest()})
        expected_keys = {(width, index, scope, repeat) for width in widths
                         for index, variant in layers.items() for scope in SCOPES[variant]
                         for repeat in range(repeats)}
        _require(seen == expected_keys, "incomplete replay scope coverage")
        return rows
    except DSACoverageError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        raise DSACoverageError(f"DSA FULL dummy coverage missing in {path}: {error}") from error
