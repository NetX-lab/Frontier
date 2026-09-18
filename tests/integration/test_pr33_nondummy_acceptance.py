"""Synthetic CPU acceptance through the normal Simulator and profile loader.

These profiles are deterministic fixtures adapted from the established hybrid
production-constructor smoke. They provide no native timing/parity evidence.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from frontier.config import (
    BaseModelConfig, ClusterConfig, FixedRequestLengthGeneratorConfig,
    MetricsConfig, PoissonRequestIntervalGeneratorConfig,
    RandomForrestExecutionTimePredictorConfig, ReplicaConfig, SimulationConfig,
    SyntheticRequestGeneratorConfig, VllmV1SchedulerConfig,
)
from frontier.simulator import Simulator
from frontier.types import ActivationType, ClusterType, NormType


CASES = (
    ("dense", "co-location", 1, 1),
    ("dense", "pd-disaggregation", 1, 1),
    ("dense", "pd-af-disaggregation", 1, 1),
    ("mla", "co-location", 2, 1),
    ("moe", "co-location", 1, 2),
    ("moe", "pd-disaggregation", 1, 1),
    ("moe", "pd-af-disaggregation", 1, 2),
    ("hybrid", "co-location", 1, 1),
)


def _model(family: str) -> BaseModelConfig:
    if family == "hybrid":
        from tests.unit.test_gdn_hybrid_e2e_increment14ab import _qwen35_fixture_config
        model = _qwen35_fixture_config()
    else:
        mla = dict(
            use_mla=True, kv_lora_rank=32, q_lora_rank=32,
            qk_nope_head_dim=32, qk_rope_head_dim=32, v_head_dim=32,
        ) if family == "mla" else {}
        model = BaseModelConfig(
            num_layers=4, num_q_heads=4, num_kv_heads=2,
            embedding_dim=256, mlp_hidden_dim=64,
            max_position_embeddings=4096, use_gated_mlp=True,
            use_bias=False, use_qkv_bias=False, activation=ActivationType.SILU,
            norm=NormType.RMS_NORM, post_attn_norm=True, vocab_size=1024,
            is_moe=family == "moe", num_experts=8 if family == "moe" else 0,
            num_experts_per_tok=2 if family == "moe" else 0,
            torch_dtype="bfloat16", **mla,
        )
    model._model_name = f"pr33_synthetic_{family}"
    return model


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*rows)))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _profiles(root: Path, model: BaseModelConfig, family: str) -> dict:
    """Use constant targets so numerical checks are independent of RF fit."""
    profile = "qwen3_5_moe" if family == "hybrid" else "generic"
    metadata = dict(
        profiling_precision="BF16", model_arch="generic",
        model_architecture_profile=profile, quant_signature="none",
        measurement_type="CUDA_EVENT",
    )
    linear, attention, moe = [], [], []
    for tp in (1, 2):
        for tokens in (1, 4, 16, 32):
            row = dict(
                **metadata, num_tensor_parallel_workers=tp, num_tokens=tokens,
                n_embd=256, n_head=4, n_kv_head=2, n_expanded_embd=64,
                use_gated_mlp=True, use_qk_norm=family == "hybrid", vocab_size=1024,
            )
            for op, value in {
                "input_layernorm": .01, "post_attention_layernorm": .02,
                "attn_pre_proj": .03, "attn_post_proj": .04, "attn_rope": .005,
                "mlp_up_proj": .06, "mlp_down_proj": .07, "mlp_act": .01,
                "attn_wq_proj": .03, "attn_inter_norm": .01,
            }.items():
                row[f"time_stats.{op}.median"] = value
            linear.append(row)
        for prefill, context in ((16, 0), (32, 0), (0, 16), (0, 32)):
            row = dict(
                **metadata, n_embd=256, n_q_head=4, n_kv_head=2,
                block_size=16, num_tensor_parallel_workers=tp, batch_size=1,
                prefill_chunk_size=prefill, kv_cache_size=context,
                total_tokens=max(prefill, context, 1), is_prefill=bool(prefill),
                is_mixed_batch=False, is_true_mixed_batch=False,
                attention_backend="SYNTHETIC_CPU",
            )
            if family == "mla":
                row.update(
                    n_kv_head=1, head_size=64, qk_nope_head_dim=32,
                    qk_rope_head_dim=32, qk_head_dim=64, kv_lora_rank=32,
                    v_head_dim=32, max_model_len=4096,
                    batch_num_tokens=prefill or 1,
                    batch_num_prefill_tokens=prefill,
                    batch_num_decode_tokens=0 if prefill else 1,
                    max_seqlen_q=prefill or 1, max_seqlen_k=prefill or context,
                    num_actual_tokens=prefill or 1, max_seq_len=prefill or context,
                    is_mla_profile_import=True,
                )
                names = (
                    "attn_mla_kv_cache_save", "attn_mla_prefill_kv_up_proj",
                    "attn_mla_prefill", "attn_mla_decode_q_latent_proj",
                    "attn_mla_decode", "attn_mla_v_up_proj",
                )
                for index, op in enumerate(names, 1):
                    for statistic in ("min", "max", "mean", "median"):
                        row[f"time_stats.{op}.{statistic}"] = index * .01
                    row[f"time_stats.{op}.std"] = 0.0
                    row[f"time_stats.{op}.count"] = 1
            else:
                row.update({
                    "time_stats.attn_prefill.median": .08 if prefill else 0,
                    "time_stats.attn_decode.median": .05 if not prefill else 0,
                    "time_stats.attn_kv_cache_save.median": .01,
                })
            attention.append(row)
    for ep in (1, 2):
        for tokens in (1, 4, 16, 32):
            row = dict(
                **metadata, num_experts=8, router_topk=2, hidden_dim=256,
                expert_hidden_dim=64, num_tensor_parallel_workers=1,
                expert_parallel_size=ep, routing_runtime_path="standard_fused_topk",
                gating_runtime_context="standalone_legacy", num_tokens=tokens,
                total_routed_tokens=tokens * 2, num_experts_per_device=8 // ep,
                model_expansion_ratio=.25, tokens_per_expert_avg=tokens / 4,
                tokens_to_experts_ratio=tokens / 4, expert_utilization=1.,
                min_load_ratio=1., load_imbalance_cv=0., max_load_ratio=1.,
                load_entropy=1., load_gini_coefficient=0.,
            )
            for op, value in {
                "moe_gating_linear": .03, "moe_gating_routing_topk": .04,
                "moe_shuffling": .05, "moe_grouped_gemm": .12,
            }.items():
                row[f"time_stats.{op}.median"] = value
            moe.append(row)
    paths = {}
    for key, filename, rows in (
        ("linear_op", "linear_op", linear),
        ("atten", "attention", attention),
        ("moe", "moe", moe),
    ):
        paths[f"{key}_input_file"] = str(_write_csv(root / f"{filename}.csv", rows))
        kernel_rows = [{**row, "measurement_type": "KERNEL_ONLY"} for row in rows]
        paths[f"{key}_kernel_only_input_file"] = str(
            _write_csv(root / f"{filename}_kernel_only.csv", kernel_rows)
        )
    return paths


def _config(root, family, architecture, pp, ep, monkeypatch):
    model = _model(family)
    original = BaseModelConfig.create_from_name
    monkeypatch.setattr(BaseModelConfig, "create_from_name", classmethod(
        lambda cls, name: model if name == model._model_name else original(name)
    ))
    profiles = _profiles(root / "profiles", model, family)
    cache = root / "cache"
    if family == "hybrid":
        from tests.unit.test_gdn_hybrid_e2e_increment14ab import _train_gdn
        cache = _train_gdn(root / "gdn", device="a100", measurement_type="CUDA_EVENT")
        profiles["gdn_input_file"] = str(root / "gdn" / "gdn.csv")
    predictor = RandomForrestExecutionTimePredictorConfig(
        enable_dummy_mode=False, **profiles, num_estimators=[2], max_depth=[2],
        min_samples_split=[2], k_fold_cv_splits=2, num_training_job_threads=1,
        prediction_max_tokens_per_request=64, prediction_max_prefill_chunk_size=32,
        prediction_max_batch_size=4, kv_cache_prediction_granularity=16,
        skip_cpu_overhead_modeling=True,
    )
    replica = ReplicaConfig(
        model_name=model._model_name, device="a100",
        network_device="a100_pairwise_nvlink", num_pipeline_stages=pp,
        attn_tensor_parallel_size=ep, moe_tensor_parallel_size=1,
        moe_expert_parallel_size=ep, total_expert_num=8 if model.is_moe else 1,
        router_topk=2 if model.is_moe else 1,
    )
    scheduler = VllmV1SchedulerConfig(
        num_blocks=128, block_size=16, batch_size_cap=1,
        max_tokens_in_batch=32, enable_chunked_prefill=True,
    )
    role_args = {}
    if architecture != "co-location":
        role_args["prefill_cluster_num_replicas"] = 1
        if architecture == "pd-disaggregation":
            role_args["decode_cluster_num_replicas"] = 1
        else:
            role_args.update(
                decode_attn_cluster_num_replicas=1, decode_ffn_cluster_num_replicas=1,
                decode_attn_af_pipeline_num_micro_batch=1, decode_attn_micro_batch_size=1,
                decode_ffn_af_pipeline_num_micro_batch=1,
                decode_ffn_replica_scheduler_config_type="orca",
            )
    cluster = ClusterConfig(
        replica_config=replica, replica_scheduler_config=scheduler,
        execution_time_predictor_config=predictor, **role_args,
    )
    metrics = MetricsConfig(
        output_dir=str(root / "metrics"), cache_dir=str(cache), run_id="acceptance",
        write_metrics=True, store_request_metrics=True, store_batch_metrics=True,
        store_operation_metrics=True, keep_individual_batch_metrics=True,
        store_utilization_metrics=True, store_frontier_stage_batch_ledger=True,
        store_frontier_stage_batch_ledger_summary=True, store_plots=False,
        enable_chrome_trace=False, write_json_trace=False, enable_op_level_tracing=True,
        trace_output_file="op_traces.jsonl", enable_per_layer_expansion=True,
        num_requests_to_trace_per_layer=100,
    )
    return model, SimulationConfig(
        simulation_mode="online" if family == "hybrid" else "offline",
        sys_arch=architecture, enable_parallel_clusters=False,
        decode_cuda_graph_mode="none", cluster_config=cluster, metrics_config=metrics,
        request_generator_config=SyntheticRequestGeneratorConfig(
            num_requests=2,
            length_generator_config=FixedRequestLengthGeneratorConfig(prefill_tokens=16, decode_tokens=2),
            interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=1000000.),
        ),
    )


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _verify(root, model, family, architecture, ep):
    paths = list(root.rglob("request_metrics.csv"))
    assert len(paths) == 1
    output = paths[0].parent
    requests = _read_csv(paths[0])
    assert len(requests) == 2
    assert all(int(row["request_num_tokens"]) == 18 for row in requests)
    assert all(float(row["request_e2e_time"]) > 0 for row in requests)
    summary = json.loads((output / "system_metrics.json").read_text())
    assert summary["simulation_metadata"]["completed_requests"] == 2
    assert summary["simulation_metadata"]["total_requests"] == 2
    ledger = [json.loads(line) for line in (output / "frontier_stage_batch_ledger.jsonl").read_text().splitlines()]
    traces = [json.loads(line) for line in (output / "op_traces.jsonl").read_text().splitlines()]
    events = [row for row in traces if "name" in row]
    assert ledger and events
    lane_rows = []
    af_lane_rows = []
    if model.is_moe:
        lane_path = output / "frontier_ep_wave_lane_ledger.jsonl"
        lane_rows = [json.loads(line) for line in lane_path.read_text().splitlines()]
        # PDD retains two decode passes for the two requested decode tokens.
        # Co-location and PD-AF retain one post-prefill decode pass. PD-AF
        # reports its decode FFN lanes in the existing stage ledger.
        passes = {"co-location": 2, "pd-disaggregation": 3,
                  "pd-af-disaggregation": 1}[architecture]
        assert len(lane_rows) == 2 * passes * model.num_layers * ep
        if architecture == "pd-af-disaggregation":
            af_lane_rows = [row for row in ledger
                            if row.get("execution_scope") == "EP_WAVE_LANE"]
            assert len(af_lane_rows) == 2 * model.num_layers * ep
            assert {row["cluster_type"] for row in af_lane_rows} == {"DECODE_FFN"}
            assert {(row["layer_id"], row["ep_id"]) for row in af_lane_rows} == {
                (layer, lane) for layer in range(model.num_layers) for lane in range(ep)
            }
            positive_gemm = [row["execution_time"]["component_ledger_ms"]["moe_grouped_gemm_time"]
                             for row in af_lane_rows
                             if sum(row["per_expert_tokens"].values()) > 0]
            assert positive_gemm and all(math.isclose(value, .12, rel_tol=1e-12, abs_tol=1e-9)
                                         for value in positive_gemm)
        assert {row["ep_id"] for row in lane_rows} == set(range(ep))
        grouped_gemm = []
        for row in lane_rows:
            assert row["wave_end_time_s"] >= row["wave_start_time_s"]
            assert set(row["phases"]) == {
                "pre_dispatch", "dispatch", "routed_compute", "combine", "post_combine",
            }
            for phase in row["phases"].values():
                values = list(phase["operators_ms"].values())
                assert all(math.isfinite(value) and value >= 0 for value in values)
                assert math.isclose(sum(values), phase["duration_ms"], rel_tol=1e-12, abs_tol=1e-9)
                if phase["operators_ms"].get("moe_grouped_gemm", 0) > 0:
                    grouped_gemm.append(phase["operators_ms"]["moe_grouped_gemm"])
        assert grouped_gemm and all(math.isclose(value, .12, rel_tol=1e-12, abs_tol=1e-9)
                                   for value in grouped_gemm)
    assert all(row["stage_end_ts"] >= row["stage_start_ts"] for row in ledger)
    operation_paths = list(output.glob("*_operation_metrics.csv"))
    operation_paths = [path for path in operation_paths if "cpu_operation" not in path.name]
    expected_roles = 1 if architecture == "co-location" else (2 if architecture == "pd-disaggregation" else 3)
    assert len(operation_paths) == expected_roles
    for path in operation_paths:
        rows = _read_csv(path)
        assert rows, path
        numerical = [float(value) for row in rows for key, value in row.items()
                     if key != "batch_id" and value not in ("", None)]
        assert numerical and all(math.isfinite(value) and value >= 0 for value in numerical)
        assert any(value > 0 for value in numerical)
        if family in {"dense", "mla"} and architecture == "co-location":
            key = "attn_mla_kv_cache_save" if family == "mla" else "input_layernorm"
            assert all(math.isclose(float(row[key]), model.num_layers * .01,
                                    rel_tol=1e-12, abs_tol=1e-9) for row in rows)
    attention_events = [row for row in events if row.get("meta", {}).get("attention_family_id")]
    assert attention_events
    expected = {"dense_attention"}
    if family == "mla":
        expected = {"latent_mla_attention"}
    elif family == "hybrid":
        expected.add("gated_delta_net")
    assert {row["meta"]["attention_family_id"] for row in attention_events} == expected
    ids = {row["layer_id"] for row in attention_events if row.get("layer_id", -1) >= 0}
    for row in attention_events:
        if row.get("layer_id") == -1:
            ids.update(row["meta"]["global_layer_ids"])
    assert ids == set(range(model.num_layers))
    if family == "hybrid":
        membership = {"gated_delta_net": set(), "dense_attention": set()}
        for row in attention_events:
            members = ([row["layer_id"]] if row["layer_id"] >= 0
                       else row["meta"]["global_layer_ids"])
            membership[row["meta"]["attention_family_id"]].update(members)
        assert membership == {
            "gated_delta_net": {0, 1, 2, 4, 5, 6}, "dense_attention": {3, 7},
        }
        for name, per_layer_ms in (("gdn_core_prefill", .22), ("gdn_core_decode", .05)):
            phase_rows = [row for row in events if row["name"] == name and row["duration_ms"] > 0]
            assert phase_rows
            for row in phase_rows:
                count = 1 if row["layer_id"] >= 0 else len(row["meta"]["global_layer_ids"])
                assert math.isclose(row["duration_ms"], count * per_layer_ms,
                                    rel_tol=1e-12, abs_tol=1e-9)
    # Select a family-native scope with a constant 0.01 ms target. This
    # numerical oracle is independent of the predictor's own aggregation.
    oracle_op = "attn_mla_kv_cache_save" if family == "mla" else "input_layernorm"
    oracle_events = [row for row in events if row["name"] == oracle_op
                     and row.get("layer_id", -1) >= 0]
    assert oracle_events
    assert all(math.isclose(row["duration_ms"], .01, rel_tol=1e-12, abs_tol=1e-9)
               for row in oracle_events)
    assert all(math.isfinite(row["duration_ms"]) and row["duration_ms"] >= 0
               for row in events)
    return dict(completed_requests=2, ledger_rows=len(ledger), trace_events=len(events),
                operation_files=len(operation_paths), attention_families=sorted(expected),
                ep_lane_records=len(lane_rows), af_stage_lane_records=len(af_lane_rows))


@pytest.mark.parametrize("family,architecture,pp,ep", CASES)
def test_nondummy_simulator_acceptance(tmp_path, family, architecture, pp, ep):
    # Simulator process globals intentionally describe one deployment. Run each
    # deployment in its own interpreter, matching the existing fidelity runner.
    case_index = CASES.index((family, architecture, pp, ep))
    command = [sys.executable, str(Path(__file__).resolve()), "--case", str(case_index),
               "--output", str(tmp_path / "run"), "--verify"]
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
           "TMPDIR": str(tmp_path), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    with (tmp_path / "run.log").open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=180)
    assert result.returncode == 0, (tmp_path / "run.log").read_text()[-12000:]
    assert (tmp_path / "run" / "acceptance_evidence.json").is_file()


def compare_baseline(baseline_root: Path, candidate_root: Path) -> dict:
    """Retain every stable discrepancy under the established comparator."""
    from tests.integration.run_scheduler_refactor_fidelity import compare, load_artifact

    baseline = next(baseline_root.rglob("request_metrics.csv")).parent
    candidate = next(candidate_root.rglob("request_metrics.csv")).parent
    results = {}
    paths = sorted(path for path in candidate.iterdir()
                   if path.suffix in {".csv", ".jsonl"} or path.name == "system_metrics.json")
    for path in paths:
        if not (baseline / path.name).is_file():
            results[path.name] = {"status": "ADDED_ARTIFACT"}
            continue
        left, right = load_artifact(baseline / path.name), load_artifact(path)
        if path.name == "op_traces.jsonl":
            # The JSONL header contains wall-clock time; compare event rows.
            left = [row for row in left if "name" in row]
            right = [row for row in right if "name" in row]
        try:
            compare(left, right)
            results[path.name] = {"status": "PASS"}
        except AssertionError as error:
            results[path.name] = {"status": "DIFFERENCE", "first_difference": str(error)}
            if path.name.endswith("operation_metrics.csv") and left and right:
                shared = sorted(left[0].keys() & right[0].keys())
                results[path.name]["removed_columns"] = sorted(left[0].keys() - right[0].keys())
                try:
                    compare([{key: row[key] for key in shared} for row in left],
                            [{key: row[key] for key in shared} for row in right])
                    results[path.name]["common_columns"] = "PASS"
                except AssertionError as error:
                    results[path.name]["common_columns"] = str(error)
    return results


def main():
    """Run this same campaign case against an explicitly selected PYTHONPATH."""
    import argparse
    import frontier

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, required=True, choices=range(len(CASES)))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    family, architecture, pp, ep = CASES[args.case]
    (args.output / "invocation.json").write_text(json.dumps({
        "frontier_module": str(Path(frontier.simulator.__file__).resolve()),
        "case": CASES[args.case], "synthetic_profiles": True,
    }, indent=2) + "\n")
    with pytest.MonkeyPatch.context() as monkeypatch:
        model, config = _config(args.output, family, architecture, pp, ep, monkeypatch)
        simulator = Simulator(config)
        assert all(not predictor._enable_dummy_mode for predictor in simulator._predictors.values())
        simulator.run()
        if family == "hybrid":
            scheduler = simulator._global_scheduler.get_cluster_scheduler(ClusterType.MONOLITHIC)
            replica_id = next(iter(simulator._clusters[ClusterType.MONOLITHIC].replicas))
            replica_scheduler = scheduler.get_replica_scheduler(replica_id, 0)
            slots = replica_scheduler._gdn_state_slot_manager
            assert slots.capacity == 1
            assert slots.active_request_ids == ()
            assert replica_scheduler._allocation_map == {}
            assert replica_scheduler.num_allocated_blocks == 0
        if args.verify:
            evidence = _verify(args.output / "metrics", model, family, architecture, ep)
            (args.output / "acceptance_evidence.json").write_text(
                json.dumps(evidence, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
