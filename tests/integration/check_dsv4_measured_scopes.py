"""Verify isolated V4 consumer receipts against locally qualified CSVs on CPU."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError
from frontier.config.model_config import BaseModelConfig
from frontier.entities import Batch, Request
from frontier.execution_time_predictor.deepseek_v4 import read_isolated_scope
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-root", type=Path, required=True)
    parser.add_argument("--compute-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
    receipts = []
    negative = []
    for measurement, suffix in ((MeasurementType.CUDA_EVENT, ""), (MeasurementType.KERNEL_ONLY, "_kernel_only")):
        p._active_measurement_type = measurement
        p._attention_input_file = str(next(args.attention_root.rglob(f"attention{suffix}.csv")))
        p._compute_input_file = str(next(args.compute_root.rglob(f"linear_op{suffix}.csv")))
        p._moe_input_file = str(next(args.compute_root.rglob(f"moe{suffix}.csv")))
        rows = pd.read_csv(p._attention_input_file)
        for _, row in rows.iterrows():
            layer = json.loads(row.global_layer_ids)[0]
            for scope in json.loads(row.operator_scopes):
                result = read_isolated_scope(p, source_batch(), layer, scope)
                assert result["time_ms"] == row[f"time_stats.{scope}.median"]
                assert result["serving_admission"] is False
                receipts.append(result)
        for scope in ("share_expert_up_proj", "share_expert_act", "share_expert_down_proj"):
            result = read_isolated_scope(p, source_batch(), 0, scope)
            rows = pd.read_csv(p._compute_input_file)
            assert result["time_ms"] == rows[rows.num_tokens == 768][f"time_stats.{scope}.median"].iloc[0]
            receipts.append(result)

        # An explicit reachable synthetic lane matches a small measured expert
        # histogram. This checks consumer mathematics, not real native routing.
        counts = {i: int(i < 2 or 8 <= i < 198) for i in range(256)}
        lane = split_global_expert_tokens_into_lanes(counts, total_expert_num=256,
            moe_expert_parallel_size=32, router_topk=6)[0]
        result = read_isolated_scope(p, source_batch(32), 0, "moe_grouped_gemm", lane_workload=lane)
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
                read_isolated_scope(p, source_batch(query, context, decoding), 2, scope)
            except DeepSeekV4CoverageError as exc:
                negative.append({"measurement_type": measurement.value,
                    "query": query, "context": context, "scope": scope, "error": str(exc)})
            else:
                raise AssertionError("Unqualified scope did not raise a typed coverage error")
        p._replica_config.device = "h800"
        p._replica_config.attn_dp = 2
        p._replica_config.moe_expert_parallel_size = 16
        try:
            read_isolated_scope(p, source_batch(), 0, "dsv4_sparse_attention")
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "PASS", "positive_count": len(receipts),
        "typed_negative_count": len(negative), "serving_admission": False,
        "receipts": receipts, "negative_controls": negative}, indent=2) + "\n")
    print(f"PASS {len(receipts)} isolated scopes, {len(negative)} typed negative controls; no serving admission")


if __name__ == "__main__":
    main()
