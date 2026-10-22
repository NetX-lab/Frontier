"""Verify isolated V4 consumer receipts against locally qualified CSVs on CPU."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pandas as pd

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError
from frontier.config.model_config import BaseModelConfig
from frontier.entities import Batch, Request
from frontier.execution_time_predictor.deepseek_v4 import build_isolated_prefill_layouts, read_isolated_scope
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import SklearnMoEExecutionTimePredictor
from frontier.moe_ep_workload import split_global_expert_tokens_into_lanes
from frontier.types import ClusterType, MeasurementType


class ScopePredictor(SklearnMoEExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise NotImplementedError


def source_batch(query=6144, context=0, decoding=False):
    return Batch(0, [Request(0, query + context, 1024)], [query], is_moe=True,
                 num_context_tokens=[context], request_is_decoding=[decoding])


def check_accepted_prefill(p, acceptance_path, package):
    from frontier.profiling.attention.dsa_rows import read_records
    from frontier.profiling.attention.dsv4_attention_main import build_parser, build_plan
    from frontier.profiling.attention.dsv4_attention_plan import expected_scopes
    from frontier.profiling.common.runtime_identity import file_sha256

    acceptance = json.loads(acceptance_path.read_text())
    assert acceptance["status"] == "ACCEPTED_ISOLATED_DATA" and not acceptance["serving_admission"]
    assert file_sha256(package / "manifest.json") == acceptance["manifest_sha256"]
    manifest = json.loads((package / "manifest.json").read_text())
    for name in ("plan.json", "model/config.json", "jobs/PREFILL_WARM_6144_C0_BALANCED.json"):
        assert file_sha256(package / name) == manifest["files"][name]
    job = json.loads((package / "jobs/PREFILL_WARM_6144_C0_BALANCED.json").read_text())
    argv = job["phases"][0]["steps"][1]["commands"][0]["argv"]
    plan_args = build_parser().parse_args(argv[3:])
    plan_args.model_config_path = package / "model/config.json"
    plan = build_plan(plan_args)
    assert json.loads(json.dumps(plan.to_json())) == json.loads((package / "plan.json").read_text())
    inputs = {}
    for name, pin in acceptance["retained_files"].items():
        if name.endswith((".csv", "dsv4_raw_e0.jsonl", "dsv4_aggregate_summary.json")):
            path = acceptance_path.parents[3] / name
            assert file_sha256(path) == pin["sha256"]
            inputs[path.name] = path, pin["sha256"]
    identity = json.loads(inputs["dsv4_aggregate_summary.json"][0].read_text())["identity"]
    assert identity["source_revision"] == acceptance["source_revision"] == manifest["source_revision"]
    assert identity["model_config_sha256"] == manifest["files"]["model/config.json"]
    p._replica_config.device = job["gpu_type"].lower()
    p._replica_config.attn_dp = job["target_per_role"]["dp"]
    p._replica_config.moe_expert_parallel_size = job["target_per_role"]["ep"]
    p._replica_config.moe_gating_routing_runtime_path = "balanced_topk"
    p._replica_config.attn_tensor_parallel_size = plan.shape.target_tensor_parallel_size
    p._block_size = plan.limits.block_size
    records = read_records([inputs["dsv4_raw_e0.jsonl"][0]])
    receipts = []
    for method, name in ((MeasurementType.CUDA_EVENT, "attention.csv"),
                         (MeasurementType.KERNEL_ONLY, "attention_kernel_only.csv")):
        path, digest = inputs[name]
        p._attention_input_file = str(path)
        p._active_measurement_type = method
        layouts = build_isolated_prefill_layouts(plan, records, expected_identity={
            key: identity[key] for key in ("source_revision", "runtime_fingerprint", "model_config_sha256")},
            input_file_sha256=digest)
        rows = pd.read_csv(p._attention_input_file, float_precision="round_trip")
        for layer in plan.shape.layers:
            row = rows[rows.global_layer_ids == json.dumps([layer.global_layer_id])].iloc[0]
            for scope in expected_scopes(layer.attention_variant,
                    plan.shape.index_scores_executed(plan.cases[0].prompt_len),
                    annotated_only=method is MeasurementType.KERNEL_ONLY):
                result = read_isolated_scope(p, source_batch(plan.cases[0].prompt_len),
                    layer.global_layer_id, scope, attention_layout=layouts[layer.global_layer_id])
                assert result["time_ms"] == row[f"time_stats.{scope}.median"]
                assert result["serving_admission"] is False
                receipts.append(result)
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-root", type=Path)
    parser.add_argument("--compute-root", type=Path)
    parser.add_argument("--accepted-prefill", type=Path)
    parser.add_argument("--prefill-package", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.accepted_prefill is not None) != (args.prefill_package is not None):
        parser.error("--accepted-prefill and --prefill-package must be supplied together")
    if args.accepted_prefill is None and (args.attention_root is None or args.compute_root is None):
        parser.error("legacy checking requires --attention-root and --compute-root")
    p = ScopePredictor.__new__(ScopePredictor)
    p._model_config = BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V4-Flash")
    p._replica_config = SimpleNamespace(
        model_config=p._model_config, device="h200", num_pipeline_stages=1,
        attn_tensor_parallel_size=8, attn_dp=4, moe_tensor_parallel_size=1,
        moe_expert_parallel_size=32, attention_backend="FLASHMLA_SPARSE_DSV4",
        kv_cache_dtype="fp8_ds_mla", speculative_decoding_config=SimpleNamespace(enabled=False),
    )
    p._cluster_type = ClusterType.MONOLITHIC
    p._block_size = 256
    if args.accepted_prefill is not None:
        receipts = check_accepted_prefill(p, args.accepted_prefill, args.prefill_package)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"status": "PASS", "positive_count": len(receipts),
            "serving_admission": False, "receipts": receipts}, indent=2) + "\n")
        print(f"PASS {len(receipts)} accepted isolated prefill scopes; no serving admission")
        return
    layouts = json.loads((Path(__file__).parents[1] /
        "fixtures/deepseek_v4_attention_layouts.json").read_text())["layouts"]

    def read_scope(batch, layer, scope, **kwargs):
        return read_isolated_scope(p, batch, layer, scope,
            attention_layout=layouts[p._model_config.get_layer_attention_spec(layer).variant_id],
            **kwargs)

    receipts = []
    negative = []
    mixed_layout_controls = []
    for measurement, suffix in ((MeasurementType.CUDA_EVENT, ""), (MeasurementType.KERNEL_ONLY, "_kernel_only")):
        p._active_measurement_type = measurement
        p._attention_input_file = str(next(args.attention_root.rglob(f"attention{suffix}.csv")))
        p._compute_input_file = str(next(args.compute_root.rglob(f"linear_op{suffix}.csv")))
        p._moe_input_file = str(next(args.compute_root.rglob(f"moe{suffix}.csv")))
        attention_rows = pd.read_csv(p._attention_input_file)
        for _, row in attention_rows.iterrows():
            layer = json.loads(row.global_layer_ids)[0]
            for scope in json.loads(row.operator_scopes):
                result = read_scope(source_batch(), layer, scope)
                assert result["time_ms"] == row[f"time_stats.{scope}.median"]
                assert result["serving_admission"] is False
                receipts.append(result)
        for scope in ("share_expert_up_proj", "share_expert_act", "share_expert_down_proj"):
            result = read_scope(source_batch(), 0, scope)
            rows = pd.read_csv(p._compute_input_file)
            assert result["time_ms"] == rows[rows.num_tokens == 768][f"time_stats.{scope}.median"].iloc[0]
            receipts.append(result)

        # An explicit reachable synthetic lane matches a small measured expert
        # histogram. This checks consumer mathematics, not real native routing.
        counts = {i: int(i < 2 or 8 <= i < 198) for i in range(256)}
        lane = split_global_expert_tokens_into_lanes(counts, total_expert_num=256,
            moe_expert_parallel_size=32, router_topk=6)[0]
        result = read_scope(source_batch(32), 0, "moe_grouped_gemm", lane_workload=lane)
        rows = pd.read_csv(p._moe_input_file, float_precision="round_trip")
        assert result["time_ms"] == rows[(rows.num_tokens == 32) & (rows.seed == 1)][
            "time_stats.moe_grouped_gemm.median"].iloc[0]
        receipts.append(result)

        for query, context, decoding, scope in (
            (6145, 0, False, "dsv4_sparse_attention"),
            (6144, 1, False, "dsv4_sparse_attention"),
            (1, 6143, False, "dsv4_sparse_attention"),
            (1, 6144, True, "dsv4_sparse_attention"),
            (6144, 0, False, "moe_gating_routing_topk"),
            (6144, 0, False, "dsv4_compressor_projection"),
        ):
            try:
                read_scope(source_batch(query, context, decoding), 2, scope)
            except DeepSeekV4CoverageError as exc:
                negative.append({"measurement_type": measurement.value,
                    "query": query, "context": context, "scope": scope, "error": str(exc)})
            else:
                raise AssertionError("Unqualified scope did not raise a typed coverage error")
        p._replica_config.device = "h800"
        p._replica_config.attn_dp = 2
        p._replica_config.moe_expert_parallel_size = 16
        try:
            read_scope(source_batch(), 0, "dsv4_sparse_attention")
        except DeepSeekV4CoverageError as exc:
            assert "device does not match" in str(exc)
            negative.append({"measurement_type": measurement.value,
                "device": "h800", "topology": "TP8/DP2/EP16", "scope": "dsv4_sparse_attention",
                "error": str(exc)})
        else:
            raise AssertionError("H200 rows must not qualify missing H800 coverage")
        p._replica_config.device = "h200"
        p._replica_config.attn_dp = 4
        p._replica_config.moe_expert_parallel_size = 32
        original_attention = p._attention_input_file
        original_compute = p._compute_input_file
        original_moe = p._moe_input_file
        with TemporaryDirectory(prefix="v4_scope_review_") as temporary:
            table = Path(temporary) / "attention.csv"
            row = attention_rows[(attention_rows.attention_variant == "csa")
                                 & attention_rows.plans_tile_schedule].iloc[0].copy()
            p._attention_input_file = str(table)
            for column, value in (
                ("attention_kv_cache_dtype", "bfloat16"), ("kv_cache_dtype", "bfloat16"),
                ("padded_heads", 128), ("aux_stream_count", 4),
                ("multi_stream_gemm_token_threshold", 2048),
                ("indexer_uses_fp4_kv", True), ("attention_impl", "unqualified"),
                ("max_model_len", 8192), ("max_num_batched_tokens", 8192),
                ("max_num_seqs", 256), ("swa_metadata.block_size", 128),
                ("attention_metadata.block_size", 64), ("compressor_metadata.block_size", 8),
                ("index_compressor_metadata.block_size", 8),
                ("indexer_metadata.max_seq_len", 8192),
                ("attention_metadata.num_actual_tokens", 8192),
                ("compressor_metadata.c128_boundary", 1),
            ):
                changed = row.copy()
                changed[column] = value
                pd.DataFrame([changed]).to_csv(table, index=False)
                try:
                    read_scope(source_batch(), 2, "dsv4_sparse_attention")
                except DeepSeekV4CoverageError as exc:
                    negative.append({"measurement_type": measurement.value,
                        "input_file": original_attention, "column": column, "value": value,
                        "scope": "dsv4_sparse_attention", "error": str(exc)})
                else:
                    raise AssertionError(f"Unqualified attention layout accepted: {column}={value}")
            other = row.copy()
            other["padded_heads"] = 128
            other["time_stats.dsv4_sparse_attention.median"] *= 10
            pd.DataFrame([row, other]).to_csv(table, index=False)
            result = read_scope(source_batch(), 2, "dsv4_sparse_attention")
            assert result["time_ms"] == row["time_stats.dsv4_sparse_attention.median"]
            mixed_layout_controls.append({"measurement_type": measurement.value,
                "expected_ms": float(row["time_stats.dsv4_sparse_attention.median"]),
                "actual_ms": result["time_ms"], "excluded_padded_heads": 128})
            p._attention_input_file = original_attention

            for scope, original in (("share_expert_up_proj", original_compute),
                                    ("moe_grouped_gemm", original_moe)):
                rows = pd.read_csv(original, float_precision="round_trip")
                changes = ["missing_file", "missing_tp", "invalid_contract", "missing_contract", "unknown_operator"]
                if scope == "moe_grouped_gemm":
                    changes.extend(("missing_ep", "unmatched_ep"))
                for change in changes:
                    changed = rows.copy()
                    table = Path(temporary) / f"{scope}_{change}.csv"
                    if change == "missing_tp":
                        changed = changed.drop(columns=["num_tensor_parallel_workers"])
                    elif change == "missing_ep":
                        changed = changed.drop(columns=["expert_parallel_size"])
                    elif change == "unmatched_ep":
                        changed["expert_parallel_size"] = 16
                    elif change == "missing_contract":
                        changed = changed.drop(columns=["typed_operator_contracts"])
                    elif change == "invalid_contract":
                        changed["typed_operator_contracts"] = "{}"
                    elif change == "unknown_operator":
                        def add_unknown_operator(raw):
                            contracts = json.loads(raw)
                            contracts["unqualified_operator"] = dict(next(iter(contracts.values())))
                            return json.dumps(contracts)
                        changed["typed_operator_contracts"] = changed.typed_operator_contracts.map(add_unknown_operator)
                    if change != "missing_file":
                        changed.to_csv(table, index=False)
                    if scope == "moe_grouped_gemm":
                        p._moe_input_file = str(table)
                    else:
                        p._compute_input_file = str(table)
                    try:
                        read_scope(source_batch(32 if scope == "moe_grouped_gemm" else 6144),
                                   0, scope, lane_workload=lane)
                    except DeepSeekV4CoverageError as exc:
                        if change == "unknown_operator":
                            assert type(exc.__cause__) is ValueError
                            assert str(exc.__cause__).startswith("Unknown typed operator ")
                        negative.append({"measurement_type": measurement.value, "scope": scope,
                            "input_file": original, "change": change, "error": str(exc),
                            "cause": type(exc.__cause__).__name__ if exc.__cause__ is not None else None})
                    else:
                        raise AssertionError(f"Missing V4 component coverage accepted: {scope}/{change}")
                p._compute_input_file = original_compute
                p._moe_input_file = original_moe
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "PASS", "positive_count": len(receipts),
        "typed_negative_count": len(negative), "serving_admission": False,
        "receipts": receipts, "negative_controls": negative,
        "mixed_layout_controls": mixed_layout_controls}, indent=2) + "\n")
    print(f"PASS {len(receipts)} isolated scopes, {len(negative)} typed negative controls; no serving admission")


if __name__ == "__main__":
    main()
