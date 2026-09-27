"""Step 9 P5 / C2: PP=1 `vllm_load_balancing` scenarios, before vs after P2.

    python c2_pp1_policy_matrix.py run <tree> <label> <out_root>   # every case, one child each
    python c2_pp1_policy_matrix.py child <tree> <case> <case_root>
    python c2_pp1_policy_matrix.py compare <out_root> <before_tree> <after_tree>

A child imports `frontier` from <tree> only (the editable-install finder is
stripped), runs one case with metrics written, and records every load report
(time, engine, key, load) and every selection (time, snapshot, engine).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PY = "/data/ycfeng/envs/frontier-py310/bin/python"

SHAPES = {
    "moe_dp2": dict(is_moe=True, attn_dp=2, moe_ep=2),
    "moe_dp4": dict(is_moe=True, attn_dp=4, moe_ep=4),
    "dense_dp1": dict(is_moe=False, attn_dp=1, moe_ep=1),
}

ASYMMETRIC_TRACE = """arrived_at,num_prefill_tokens,num_decode_tokens
0.0,16,40
0.0,16,1
0.4,16,1
0.6,16,1
0.8,16,1
1.0,16,1
"""

# (mode, num_requests, lengths, qps, num_blocks); lengths is "fixed" (16, 3)
# or "uniform" (8..96 tokens, prefill:decode = 4).
WORKLOADS = {
    "burst_fixed_n4": ("offline", 4, "fixed", 1e6, 128),
    "burst_fixed_n16": ("offline", 16, "fixed", 1e6, 128),
    "burst_uniform_n24": ("offline", 24, "uniform", 1e6, 128),
    "poisson_fixed_n16_q20": ("online", 16, "fixed", 20.0, 128),
    "poisson_uniform_n24_q50": ("online", 24, "uniform", 50.0, 128),
    "poisson_uniform_n24_q200": ("online", 24, "uniform", 200.0, 128),
    "poisson_uniform_n24_q200_tight_kv": ("online", 24, "uniform", 200.0, 12),
    "asymmetric_trace": ("trace", None, None, None, 128),
}

CASES = [f"{shape}__{workload}" for shape in SHAPES for workload in WORKLOADS]


def _isolate(tree: str) -> None:
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if "editable" not in getattr(type(finder), "__module__", "").lower()
    ]
    sys.path.insert(0, tree)
    import frontier.scheduler.cluster_scheduler.vllm_load_balancing_cluster_scheduler as probe

    assert probe.__file__.startswith(tree + "/"), f"wrong tree: {probe.__file__}"


def _config(case_root: Path, case: str):
    from frontier.config import (
        BaseModelConfig,
        ClusterConfig,
        FixedRequestLengthGeneratorConfig,
        MetricsConfig,
        PoissonRequestIntervalGeneratorConfig,
        RandomForrestExecutionTimePredictorConfig,
        ReplicaConfig,
        SimulationConfig,
        SyntheticRequestGeneratorConfig,
        TraceRequestGeneratorConfig,
        UniformRequestLengthGeneratorConfig,
        VllmLoadBalancingClusterSchedulerConfig,
        VllmV1SchedulerConfig,
    )
    from frontier.types import ActivationType, NormType

    shape_name, workload_name = case.split("__")
    shape = SHAPES[shape_name]
    mode, num_requests, lengths, qps, num_blocks = WORKLOADS[workload_name]
    is_moe = shape["is_moe"]
    model = BaseModelConfig(
        num_layers=4,
        num_q_heads=4,
        num_kv_heads=2,
        embedding_dim=256,
        mlp_hidden_dim=64,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=1024,
        is_moe=is_moe,
        num_experts=8 if is_moe else 0,
        num_experts_per_tok=2 if is_moe else 0,
        torch_dtype="bfloat16",
    )
    model._model_name = f"c2_{'moe' if is_moe else 'dense'}_4l"
    original = BaseModelConfig.create_from_name.__func__
    BaseModelConfig.create_from_name = classmethod(
        lambda cls, name: model if name == model._model_name else original(cls, name)
    )
    moe_fields = (
        dict(
            moe_tensor_parallel_size=1,
            moe_expert_parallel_size=shape["moe_ep"],
            total_expert_num=8,
            router_topk=2,
        )
        if is_moe
        else {}
    )
    replica = ReplicaConfig(
        model_name=model._model_name,
        device="a100",
        network_device="a100_pairwise_nvlink",
        num_pipeline_stages=1,
        attn_tensor_parallel_size=1,
        attn_dp=shape["attn_dp"],
        memory_margin_fraction=0.1,
        **moe_fields,
    )
    cluster = ClusterConfig(
        replica_config=replica,
        replica_scheduler_config=VllmV1SchedulerConfig(
            num_blocks=num_blocks,
            block_size=16,
            batch_size_cap=4,
            max_tokens_in_batch=16,
            enable_chunked_prefill=True,
        ),
        cluster_scheduler_config=VllmLoadBalancingClusterSchedulerConfig(),
        execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
            enable_dummy_mode=True
        ),
    )
    if mode == "trace":
        trace = case_root / "asymmetric_arrivals.csv"
        trace.write_text(ASYMMETRIC_TRACE)
        generator = TraceRequestGeneratorConfig(trace_file=str(trace))
    else:
        length_config = (
            FixedRequestLengthGeneratorConfig(prefill_tokens=16, decode_tokens=3)
            if lengths == "fixed"
            else UniformRequestLengthGeneratorConfig(
                min_tokens=8, max_tokens=96, prefill_to_decode_ratio=4.0
            )
        )
        generator = SyntheticRequestGeneratorConfig(
            num_requests=num_requests,
            length_generator_config=length_config,
            interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=qps),
        )
    return SimulationConfig(
        simulation_mode="offline" if mode == "offline" else "online",
        sys_arch="co-location",
        enable_parallel_clusters=False,
        decode_cuda_graph_mode="none",
        cluster_config=cluster,
        metrics_config=MetricsConfig(
            output_dir=str(case_root / "metrics"),
            cache_dir=str(case_root / "cache"),
            run_id="c2",
            write_metrics=True,
            store_request_metrics=True,
            store_batch_metrics=False,
            store_operation_metrics=False,
            store_utilization_metrics=False,
            store_plots=False,
            enable_chrome_trace=False,
            write_json_trace=False,
        ),
        request_generator_config=generator,
    )


def child(tree: str, case: str, case_root: Path) -> None:
    _isolate(tree)
    from frontier.scheduler.utils.vllm_dp_load_balancer import VllmDPLoadBalancer
    from frontier.simulator import Simulator

    reports: list[list] = []
    selections: list[list] = []
    original_report = VllmDPLoadBalancer.report
    original_select = VllmDPLoadBalancer.select

    def report(self, time, engine, step, load):
        reports.append([float(time), engine, step, list(load)])
        return original_report(self, time, engine, step, load)

    def select(self, time):
        self._advance(time)
        snapshot = [list(load) for load in self.frontend_counts]
        engine = original_select(self, time)
        selections.append([float(time), snapshot, engine])
        return engine

    VllmDPLoadBalancer.report = report
    VllmDPLoadBalancer.select = select
    case_root.mkdir(parents=True, exist_ok=True)
    simulator = Simulator(_config(case_root, case))
    simulator.run()
    requests = list(simulator._all_requests)
    (case_root / "evidence.json").write_text(
        json.dumps(
            {
                "num_requests": len(requests),
                "completed": sum(1 for request in requests if request.completed),
                "reports": reports,
                "selections": selections,
            }
        )
    )


def run(tree: str, label: str, out_root: Path) -> int:
    failures = 0
    for case in CASES:
        case_root = out_root / label / case
        result = subprocess.run(
            [PY, __file__, "child", tree, case, str(case_root)],
            cwd=tree,
            env={
                **os.environ,
                "PYTHONPATH": tree,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=900,
        )
        case_root.mkdir(parents=True, exist_ok=True)
        (case_root / "child.log").write_text(result.stdout)
        status = "ok" if result.returncode == 0 else f"FAILED rc={result.returncode}"
        failures += result.returncode != 0
        print(f"{label} {case}: {status}", flush=True)
    return failures


def _order_isomorphic(before: list, after: list) -> bool:
    """Every pair of keys compares the same way on both sides."""

    return len(before) == len(after) and all(
        (before[i] > before[j]) - (before[i] < before[j])
        == (after[i] > after[j]) - (after[i] < after[j])
        for i in range(len(before))
        for j in range(i + 1, len(before))
    )


def compare(out_root: Path, before_tree: str, after_tree: str) -> int:
    sys.path.insert(0, after_tree)
    from tests.e2e.refactor_fidelity.compare import (
        PathSubstitution,
        compare_artifact_directories,
    )

    rows = []
    for case in CASES:
        sides = {label: out_root / label / case for label in ("before", "after")}
        if not any((root / "evidence.json").exists() for root in sides.values()):
            # Both runs stopped without draining: compare the stop diagnostics.
            stops = {
                label: [
                    line.split("simulator.py:708] ", 1)[1]
                    for line in (root / "child.log").read_text().splitlines()
                    if "simulator.py:708] " in line
                ]
                for label, root in sides.items()
            }
            row = {"case": case, "outcome": "stuck_on_both",
                   "identical": bool(stops["before"]) and stops["before"] == stops["after"]}
            rows.append(row)
            print(f"{'SAME' if row['identical'] else 'DIFF'} {case}: stopped without draining on both sides")
            continue
        evidence = {
            label: json.loads((root / "evidence.json").read_text())
            for label, root in sides.items()
        }
        metrics = {
            label: next((root / "metrics").rglob("request_metrics.csv")).parent
            for label, root in sides.items()
        }
        subs = {
            label: [
                PathSubstitution(str(sides[label].resolve()), "<CASE_ROOT>"),
                PathSubstitution(tree, "<TREE>"),
            ]
            for label, tree in (("before", before_tree), ("after", after_tree))
        }
        differences = compare_artifact_directories(
            metrics["before"], metrics["after"], subs["before"], subs["after"]
        )
        b, a = evidence["before"], evidence["after"]
        strip_key = lambda reports: [[t, e, load] for t, e, _, load in reports]  # noqa: E731
        row = {
            "case": case,
            "artifacts": sorted(p.name for p in metrics["after"].iterdir()),
            "metric_differences": [d.as_record() for d in differences],
            "completed": [b["completed"], a["completed"], a["num_requests"]],
            "reports": [len(b["reports"]), len(a["reports"])],
            "reports_equal_without_key": strip_key(b["reports"]) == strip_key(a["reports"]),
            "keys_equal": [r[2] for r in b["reports"]] == [r[2] for r in a["reports"]],
            "keys_order_isomorphic": _order_isomorphic(
                [r[2] for r in b["reports"]], [r[2] for r in a["reports"]]
            ),
            "selections_equal": b["selections"] == a["selections"],
            "num_selections": len(a["selections"]),
            "distinct_snapshots": len({json.dumps(s[1]) for s in a["selections"]}),
            "lanes_used": sorted({s[2] for s in a["selections"]}),
        }
        row["outcome"] = "drained" if a["completed"] == a["num_requests"] else "incomplete"
        row["identical"] = (
            not row["metric_differences"]
            and row["reports_equal_without_key"]
            and row["keys_order_isomorphic"]
            and row["selections_equal"]
            and b["completed"] == a["completed"]
        )
        rows.append(row)
        print(
            f"{'SAME' if row['identical'] else 'DIFF'} {case} [{row['outcome']} {a['completed']}/{a['num_requests']}]: reports {row['reports']}, "
            f"keys_equal={row['keys_equal']}, selections={row['num_selections']}, "
            f"snapshots={row['distinct_snapshots']}, lanes={row['lanes_used']}, "
            f"metric_diffs={len(row['metric_differences'])}"
        )
    (out_root / "c2_comparison.json").write_text(json.dumps(rows, indent=1))
    same = sum(row["identical"] for row in rows)
    print(f"=== {same} of {len(rows)} identical ===")
    return len(rows) - same


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "child":
        child(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
    elif command == "run":
        sys.exit(run(sys.argv[2], sys.argv[3], Path(sys.argv[4])))
    else:
        sys.exit(compare(Path(sys.argv[2]), sys.argv[3], sys.argv[4]))
