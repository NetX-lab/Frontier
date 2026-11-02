"""B8 lead check A3: price every H200 replay row through the sklearn runtime entry.

Independent of the executor's script: a full-depth decision tree is the exact
in-sample lookup (rows sharing a feature tuple average), and batches are built
here from the row columns. The composed layer window must reproduce the B7
in-sample result.
"""
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from frontier.config import (
    MetricsConfig, RandomForrestExecutionTimePredictorConfig, ReplicaConfig, VllmV1SchedulerConfig,
)
from frontier.entities.batch import Batch, DecodeCudaGraphMetadata, DummyForwardBatch
from frontier.entities.request import Request
from frontier.execution_time_predictor.dsv4_composition import compose_attention_phase
from frontier.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor
from frontier.types import ClusterType, MeasurementType

TABLE = Path("/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_review/profiling/compute/h200/"
             "deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv")
B7 = Path("/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/"
          "task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/evidence/p1h_bridge/"
          "package_03/b7_real_table_composition.json")
OUT = Path(sys.argv[1])
SCRATCH = OUT.parent / "a3_owner"


class LookupOwner(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        return DecisionTreeRegressor(random_state=0)

    def _get_grid_search_params(self):
        return {}


def batch_of(row):
    mode, rows = row["graph_runtime_mode"], int(row["padded_rows"])
    if row["phase"] == "dummy":
        batch = DummyForwardBatch(0, 0, 0)
    else:
        q = json.loads(row["prefill_seq_lens"])
        c = json.loads(row["prefill_kv_cache_sizes"])
        d = json.loads(row["decode_kv_cache_sizes"])
        requests = [Request(0.0, a + b, 4) for a, b in zip(q, c)] + [Request(0.0, x, 4) for x in d]
        batch = Batch(0, requests, num_tokens=q + [1] * len(d), num_context_tokens=c + d,
                      request_is_decoding=[False] * len(q) + [True] * len(d), is_moe=True)
    if mode != "NONE":
        batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
            "full_and_piecewise", mode, True, False, batch.total_num_tokens, rows,
            max(1, len(batch.requests)), rows)
    return batch


logging.disable(logging.WARNING)
owner = LookupOwner(
    RandomForrestExecutionTimePredictorConfig(enable_dummy_mode=True, k_fold_cv_splits=2,
                                              num_training_job_threads=1),
    ReplicaConfig(model_name="deepseek-ai/DeepSeek-V4-Flash", device="h200", network_device="h200_dgx",
                  attn_tensor_parallel_size=8, attn_dp=8, moe_tensor_parallel_size=1,
                  moe_expert_parallel_size=64, total_expert_num=256, router_topk=6,
                  attention_backend="FLASHMLA_SPARSE_DSV4", kv_cache_dtype="fp8",
                  indexer_kv_cache_dtype="fp8", moe_routing_distribution_type="balanced"),
    VllmV1SchedulerConfig(block_size=256),
    MetricsConfig(cache_dir=str(SCRATCH / "cache"), output_dir=str(SCRATCH / "metrics")),
    cluster_type=ClusterType.MONOLITHIC,
)
owner._enable_dummy_mode = False
owner._active_measurement_type = MeasurementType.KERNEL_ONLY
owner._attention_input_file = str(TABLE.with_name("attention_kernel_only.csv"))
df = owner._load_attention_df(owner._attention_input_file)
owner._models = owner._train_attention_layer_models()
owner._predictions = owner._predict_for_attention_layer_models()
errors, above, scope_mismatch = [], [], []
for index, row in df.iterrows():
    layer = json.loads(row.global_layer_ids)[0]
    timing = owner._predict_attention_layer_time(batch_of(row), layer, ClusterType.MONOLITHIC).operator_times
    if set(timing.op_times) != set(json.loads(row.operator_scopes)):
        scope_mismatch.append(int(index))
        continue
    schedule = compose_attention_phase(timing.op_times, timing._dsv4_scope_gap_ms, **timing._dsv4_composition_inputs)
    target = float(row["layer_window_stats.median"])
    errors.append(abs(schedule.elapsed_ms - target) / target)
    if errors[-1] > 0.05:
        above.append(dict(row_index=int(index), predicted_ms=schedule.elapsed_ms, target_ms=target,
                          absolute_relative_error=errors[-1]))
b7 = json.loads(B7.read_text())
b7_above = {item["row_index"]: item for item in b7["rows_above_five_percent"]}
summary = dict(
    sha256=hashlib.sha256(TABLE.read_bytes()).hexdigest(), rows=len(df), models=len(owner._models),
    scope_mismatches=scope_mismatch,
    median_absolute_relative_error=float(np.median(errors)),
    p90_absolute_relative_error=float(np.percentile(errors, 90)),
    maximum_absolute_relative_error=float(max(errors)),
    rows_above_five_percent=[item["row_index"] for item in above],
)
summary["b7_equal"] = bool(
    summary["sha256"] == b7["summary"]["sha256"] and not scope_mismatch
    and all(abs(summary[k] - b7["summary"][k]) <= 1e-12 for k in (
        "median_absolute_relative_error", "p90_absolute_relative_error", "maximum_absolute_relative_error"))
    and set(summary["rows_above_five_percent"]) == set(b7_above)
    and all(abs(item["predicted_ms"] - b7_above[item["row_index"]]["predicted_ms"]) <= 1e-12 for item in above))
OUT.write_text(json.dumps(dict(summary=summary, above=above), indent=2) + "\n")
print(json.dumps(summary, indent=2))
