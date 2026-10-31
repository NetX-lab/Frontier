"""Reproduce the F1b coverage failures from frozen receipts and measured rows.

This only constructs measured lookup values and admission bounds. It never
fits timings, changes tables, runs a simulator, or emits replacement inputs.
"""
from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import json
import shutil
import sys

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError
from frontier.attention.dsv4_features import build_dsv4_features
from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.dsv4_operator_predictor import (
    DeepSeekV4OperatorPredictor, operator_model_name, DEEPSEEK_V4_OPERATOR_AXES,
)
from frontier.execution_time_predictor.dsv4_predictor import (
    DeepSeekV4AttentionPredictor, load_dsv4_attention_dataframe,
    train_dsv4_attention_models,
)
from frontier.types import MeasurementType

ROOT = Path(__file__).resolve().parent
RUNS = Path("/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/f_variants/runs")
result = {"classification": "COVERAGE_MISS", "cases": {}}
receipts = {}
for case in ("p2_32", "p2_64"):
    run = RUNS / f"f1b_{case}_links03_codex_20261101"
    receipt = json.loads((run / "receipt.json").read_text())
    assert receipt["status"] == "FAILED"
    assert receipt["error"]["type"] == "DeepSeekV4CoverageError"
    receipts[case] = receipt
    for source, name in ((run / "receipt.json", f"{case}_terminal_receipt.json"),
                         (Path(str(run) + ".log"), f"{case}_terminal.log")):
        destination = ROOT / name
        if destination.exists():
            assert destination.read_bytes() == source.read_bytes()
        else:
            shutil.copyfile(source, destination)
    result["cases"][case] = {"error": receipt["error"], "init_wall_s": receipt["init_wall_s"],
                            "run_wall_s": receipt["run_wall_s"], "tables": []}
    for table in receipt["profiling_tables"]:
        digest = hashlib.sha256(Path(table["path"]).read_bytes()).hexdigest()
        assert digest == table["sha256"]
        result["cases"][case]["tables"].append({"path": table["path"], "sha256": digest})

linear_path = next(Path(t["path"]) for t in receipts["p2_32"]["profiling_tables"]
                   if "PREFILL/compute" in t["consumers"])
with linear_path.open() as stream:
    rows = list(csv.DictReader(stream))
name = "dsv4_embedding"
column = f"time_stats.{name}.median"
selected = [r for r in rows if r.get(column) not in (None, "", "nan")]
lookup = {(float(r["num_tokens"]),): float(r[column]) for r in selected}
key = operator_model_name(name, DEEPSEEK_V4_OPERATOR_AXES[name], 8)
predictor = DeepSeekV4OperatorPredictor({key: SimpleNamespace(_frontier_exact_lookup=lookup)})
try:
    predictor.operator(name, 32, 8)
except DeepSeekV4CoverageError as exc:
    assert str(exc) == receipts["p2_32"]["error"]["message"]
    result["cases"]["p2_32"]["reproduced_error"] = str(exc)
    result["cases"]["p2_32"]["measured_embedding_rows"] = sorted(k[0] for k in lookup)
else:
    raise AssertionError("The recorded embedding coverage failure did not reproduce")

model = BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V4-Flash")
replica = SimpleNamespace(attn_tensor_parallel_size=8, device="h200", kv_cache_dtype="fp8_ds_mla",
                          moe_gating_routing_runtime_path="balanced_topk")
attention_path = next(Path(t["path"]) for t in receipts["p2_64"]["profiling_tables"]
                      if "PREFILL/attention" in t["consumers"])
frame = load_dsv4_attention_dataframe(str(attention_path), model, replica, 256, MeasurementType.KERNEL_ONLY)
models = train_dsv4_attention_models(frame, lambda **kw: SimpleNamespace(
    _frontier_feature_names=tuple(kw["feature_cols"])))
predictor = DeepSeekV4AttentionPredictor(models)
batch = receipts["p2_64"]["error"]["batch"]
features = build_dsv4_features(batch["query_lengths"], batch["context_lengths"],
    batch["request_is_decoding"], variant="uncompressed", compress_ratio=1,
    sliding_window=model.sliding_window, index_topk=model.index_topk,
    padded_rows=batch["graph_metadata"]["padded_total_tokens"])
try:
    predictor.scope_times(features, variant="uncompressed", graph_runtime_mode="PIECEWISE",
                          plans_tile_schedule=False, scopes=["dsv4_fused_q_kv_prepare"])
except DeepSeekV4CoverageError as exc:
    # Runtime uses an integer padded width; serialized bounds may print it as float.
    assert str(exc) == receipts["p2_64"]["error"]["message"]
    result["cases"]["p2_64"]["reproduced_error"] = str(exc)
    result["cases"]["p2_64"]["features"] = dict(features.values)
else:
    raise AssertionError("The recorded attention coverage failure did not reproduce")
result["python"] = sys.version
result["status"] = "PASS_REPRODUCED_BOTH_COVERAGE_FAILURES"
(ROOT / "coverage_reproduction.json").write_text(json.dumps(result, indent=2) + "\n")
print(result["status"])
