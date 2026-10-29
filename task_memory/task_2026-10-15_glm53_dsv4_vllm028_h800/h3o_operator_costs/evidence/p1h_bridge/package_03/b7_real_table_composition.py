"""B7 lead check: train the replay predictor on the B6 H200 aggregate and compose every row.

The check is in-sample: each row's own measured window is the target. Error
comes from scopes that pool both planner roles and from rows that share one
feature tuple, whose targets the exact-lookup fit averages.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.dsv4_predictor import (
    load_dsv4_attention_dataframe, train_dsv4_attention_models, DeepSeekV4AttentionPredictor,
)
from frontier.execution_time_predictor.dsv4_composition import compose_attention_phase
from frontier.types import MeasurementType
from test_dsa_predictor import _fit
from test_dsv4_predictor import batch_features, options

OUTPUT = Path(__file__).with_suffix(".json")
PATH = Path("/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_review/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv")
model = BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V4-Flash")
replica = SimpleNamespace(num_pipeline_stages=1, attn_tensor_parallel_size=8, attn_dp=8,
                         moe_tensor_parallel_size=1, moe_expert_parallel_size=64,
                         attention_backend="FLASHMLA_SPARSE_DSV4",
                         speculative_decoding_config=SimpleNamespace(enabled=False),
                         device="h200", kv_cache_dtype="fp8_ds_mla",
                         moe_gating_routing_runtime_path="balanced_topk")
df = load_dsv4_attention_dataframe(str(PATH), model, replica, 256, MeasurementType.KERNEL_ONLY)
calls = []
def fit(**kwargs):
    calls.append((kwargs["df"].phase.iloc[0], kwargs["target_col"].split(".")[0]))
    return _fit(**kwargs)
models = train_dsv4_attention_models(df, fit)
predictor = DeepSeekV4AttentionPredictor(models, multi_stream_gemm_token_threshold=int(df.multi_stream_gemm_token_threshold.iloc[0]))
errors, above = [], []
for index, row in df.iterrows():
    scopes = json.loads(row.operator_scopes)
    prediction = predictor.scope_times(batch_features(row, model), variant=row.attention_variant,
                                       graph_runtime_mode=row.graph_runtime_mode,
                                       plans_tile_schedule=bool(row.plans_tile_schedule), scopes=scopes)
    schedule = compose_attention_phase(prediction.cost_ms, prediction.gap_ms,
                                       **options(row, predictor.multi_stream_gemm_token_threshold))
    target = float(row["layer_window_stats.median"])
    error = abs(schedule.elapsed_ms - target) / target
    errors.append(error)
    if error > 0.05:
        contributions = []
        for scope in scopes:
            cost_delta = prediction.cost_ms[scope] - row[f"time_stats.{scope}.median"]
            gap_delta = prediction.gap_ms[scope] - row[f"gap_stats.{scope}.median"]
            if cost_delta or gap_delta:
                contributions.append(dict(scope=scope, cost_delta_ms=cost_delta, gap_delta_ms=gap_delta,
                                          weight_delta_ms=cost_delta + gap_delta))
        contributions.sort(key=lambda item: abs(item["weight_delta_ms"]), reverse=True)
        coordinates = {column: row[column] for column in (
            "case_id", "attention_variant", "graph_runtime_mode", "phase", "plans_tile_schedule", "padded_rows",
            "prefill_seq_lens", "prefill_kv_cache_sizes", "decode_kv_cache_sizes")}
        coordinates = {k: (v.item() if hasattr(v, "item") else v) for k, v in coordinates.items()}
        above.append(dict(row_index=int(index), key=coordinates, predicted_ms=schedule.elapsed_ms,
                          target_ms=target, absolute_relative_error=error, pooled_scope_deltas=contributions))
summary = dict(path=str(PATH), sha256=hashlib.sha256(PATH.read_bytes()).hexdigest(), rows=len(df), models=len(models),
               models_by_target=dict(Counter(target for _, target in calls)),
               models_by_phase=dict(Counter(phase for phase, _ in calls)),
               median_absolute_relative_error=float(np.median(errors)),
               p90_absolute_relative_error=float(np.percentile(errors, 90)),
               maximum_absolute_relative_error=float(max(errors)),
               rows_above_five_percent=len(above),
               limits_passed=bool(np.median(errors) <= .05 and np.percentile(errors, 90) <= .10))
OUTPUT.write_text(json.dumps(dict(summary=summary, rows_above_five_percent=above), indent=2, allow_nan=False) + "\n")
print(json.dumps(summary, indent=2))
for item in above:
    print(json.dumps({key: value for key, value in item.items() if key != "pooled_scope_deltas"}))
