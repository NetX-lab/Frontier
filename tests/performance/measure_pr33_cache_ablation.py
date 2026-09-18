"""Compare cache bypasses without changing workloads or requested output flags.

Each worker is a fresh process. W00 retains its original dummy PDD scenario;
the hybrid arm invokes the existing trained-CPU production-constructor test.
Microbenchmarks identify local operation costs separately from Simulator timing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests/unit"), str(ROOT / "tests/performance")]


def install_controls(patch, mode):
    import frontier.execution_time_predictor.sklearn_moe_execution_time_predictor as module

    cls = module.SklearnMoEExecutionTimePredictor
    counters = Counter()
    attention = cls._predict_attention_layer_time_with_query_cache
    workload = cls._materialize_layer_ep_workload
    materialize = module.materialize_layer_ep_workload

    def attention_call(self, *, batch, layer_id, cluster_type, cache=None):
        counters["attention_queries"] += 1
        before = self._attention_query_cache_hits
        result = attention(self, batch=batch, layer_id=layer_id,
                           cluster_type=cluster_type,
                           cache=None if mode == "attention_bypass" else cache)
        counters["attention_hits"] += self._attention_query_cache_hits - before
        return result

    def workload_call(self, *args, **kwargs):
        counters["workload_queries"] += 1
        capacity = self._layer_workload_cache_capacity
        if mode == "ep_bypass":
            self._layer_workload_cache_capacity = 0
        try:
            return workload(self, *args, **kwargs)
        finally:
            self._layer_workload_cache_capacity = capacity

    def materialize_call(*args, **kwargs):
        counters["workload_materializations"] += 1
        return materialize(*args, **kwargs)

    patch.setattr(cls, "_predict_attention_layer_time_with_query_cache", attention_call)
    patch.setattr(cls, "_materialize_layer_ep_workload", workload_call)
    patch.setattr(module, "materialize_layer_ep_workload", materialize_call)
    return counters


def request_values(simulator):
    return [dict(id=r.id, completed=r.completed, tokens=r.num_processed_tokens,
                 arrived_at=r.arrived_at, completed_at=r.completed_at)
            for r in simulator._all_requests]


def w00(output):
    from measure_pr33_paired import CASES
    from sim_walltime_scaling.run_case import CaseSpec, run_case
    from frontier.simulator import Simulator

    payload = {k: v for k, v in CASES[2].items() if k != "name"}
    payload["attempt_index"] = 0
    payload.update(seed=42, simulation_mode="online", dummy_execution_time_ms=1.0,
                   device="h800", network_device="h800_dgx")
    case = CaseSpec.from_dict(payload)
    instances = []

    def create(config):
        simulator = Simulator(config)
        instances.append(simulator)
        return simulator

    result = run_case(case, output / "case_result.json", simulator_factory=create)
    if result["status"] != "success":
        raise RuntimeError(result)
    result["request_values"] = request_values(instances[0])
    return result


def hybrid(output, patch):
    from frontier.simulator import Simulator
    from sim_walltime_scaling.run_case import _SequentialEventCounter
    from test_gdn_hybrid_e2e_increment14ab import test_hybrid_gdn_production_constructor_cpu_e2e

    result = {}
    original_run = Simulator.run

    def measured_run(self):
        counter = _SequentialEventCounter()
        self._profiler = counter
        started = time.perf_counter()
        original_run(self)
        result.update(sim_wallclock_s=time.perf_counter() - started,
                      event_count=counter.event_count,
                      completed_requests=self.metric_store.get_completed_requests(),
                      request_values=request_values(self))

    patch.setattr(Simulator, "run", measured_run)
    started = time.perf_counter()
    test_hybrid_gdn_production_constructor_cpu_e2e(output / "fixture", patch, 3)
    result["fixture_total_s"] = time.perf_counter() - started
    result["status"] = "success"
    # Only the per-attempt output root differs in otherwise identical configs.
    result["row_artifacts"] = {
        path.name: path.read_text().replace(str(output), "{OUTPUT_ROOT}")
        for path in (output / "fixture").rglob("*")
        if path.is_file() and path.suffix in (".csv", ".json", ".jsonl")
        and "hybrid_production_sim" in path.parts
    }
    return result


def micro():
    from frontier.entities import StageExecutionTime
    from frontier.types import ClusterType
    from predictor_cache_fixtures import CacheFixturePredictor
    from test_attention_query_cache import _Batch, _Predictor, _stage

    predictor = _Predictor()
    batch = _Batch()
    stage = _stage(predictor, batch)
    mutable_layers = [layer.as_single_layer(
        global_layer_id=layer.global_layer_id,
        attention_family_id=layer.attention_family_id,
        attention_variant_id=layer.attention_variant_id,
    ) for layer in stage.layer_execution_times]
    attention = predictor.predict_attention_layer_time(
        batch=batch, layer_id=0, cluster_type=ClusterType.MONOLITHIC)
    workload_predictor = CacheFixturePredictor()
    workload_batch = SimpleNamespace(replica_id=0, total_num_tokens=4)
    workload_predictor._materialize_layer_ep_workload(workload_batch, ClusterType.MONOLITHIC, 0)

    def key():
        spec = predictor._model_config.get_layer_attention_spec(0)
        return (spec.family_id, spec.variant_id)

    def workload_miss():
        workload_predictor._layer_workload_cache.clear()
        return workload_predictor._materialize_layer_ep_workload(workload_batch, ClusterType.MONOLITHIC, 0)

    operations = {
        "attention_key": key,
        "attention_payload_copy": lambda: predictor._clone_attention_time(attention),
        "stage_8_finalized_layers": lambda: StageExecutionTime(stage.layer_execution_times),
        "stage_8_mutable_layers": lambda: StageExecutionTime(mutable_layers),
        "one_layer_snapshot": lambda: mutable_layers[0].finalized_copy(),
        "ep_workload_hit": lambda: workload_predictor._materialize_layer_ep_workload(workload_batch, ClusterType.MONOLITHIC, 0),
        "ep_workload_miss_with_clear": workload_miss,
    }
    rows = {}
    for name, operation in operations.items():
        samples = []
        for _ in range(5):
            started = time.perf_counter_ns()
            for _ in range(1000):
                operation()
            samples.append((time.perf_counter_ns() - started) / 1000)
        rows[name] = {"ns_per_call": samples, "iterations_per_sample": 1000}
    return {"status": "success", "components": rows}


def worker(args):
    import pytest

    args.output.mkdir(parents=True, exist_ok=False)
    with pytest.MonkeyPatch.context() as patch:
        counters = install_controls(patch, args.mode)
        if args.scenario == "w00":
            result = w00(args.output)
        elif args.scenario == "hybrid":
            result = hybrid(args.output, patch)
        else:
            result = micro()
        result.update(scenario=args.scenario, mode=args.mode, counters=dict(counters))
        (args.output / "measurement.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def comparable_artifacts(artifacts):
    """Ignore only the trace header's wall-clock file creation timestamp."""
    normalized = dict(artifacts)
    if "op_traces.jsonl" in normalized:
        lines = normalized["op_traces.jsonl"].splitlines(keepends=True)
        header = json.loads(lines[0])
        if set(header) == {"meta"} and "simulation_info" in header["meta"]:
            header["meta"].pop("timestamp", None)
            lines[0] = json.dumps(header, sort_keys=True) + "\n"
        normalized["op_traces.jsonl"] = "".join(lines)
    return normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scenario", choices=("w00", "hybrid", "micro"))
    parser.add_argument("--mode", choices=("normal", "attention_bypass", "ep_bypass"), default="normal")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=args.resume)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", WANDB_DISABLED="true",
               VIDUR_DISABLE_WANDB="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", FRONTIER_LOG_LEVEL="WARNING",
               TMPDIR=str(args.output), FRONTIER_TMP_ROOT=str(args.output))
    manifest = {"python": sys.version, "executable": sys.executable,
                "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "dirty": subprocess.check_output(["git", "status", "--porcelain"], text=True),
                "command": sys.argv, "thread_limits": 1}
    manifest_name = "resume_manifest.json" if args.resume else "manifest.json"
    (args.output / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    reference = {}
    for repetition in range(args.repetitions):
        for scenario in ("w00", "hybrid", "micro"):
            modes = ["normal"] if scenario == "micro" else ["normal", "attention_bypass", "ep_bypass"]
            if repetition % 2:
                modes.reverse()
            for mode in modes:
                output = args.output / f"{scenario}-{repetition}-{mode}"
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--scenario", scenario,
                           "--mode", mode, "--output", str(output)]
                if not (args.resume and (output / "measurement.json").exists()):
                    with (args.output / f"{output.name}.log").open("w") as log:
                        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                result = json.loads((output / "measurement.json").read_text())
                stable = {key: result[key] for key in ("event_count", "completed_requests", "request_values", "row_artifacts") if key in result}
                if "row_artifacts" in stable:
                    stable["row_artifacts"] = comparable_artifacts(stable["row_artifacts"])
                if scenario not in reference:
                    reference[scenario] = stable
                if stable != reference[scenario]:
                    raise AssertionError(f"Control changed simulation result: {output}")
                result.pop("row_artifacts", None)
                result["repetition"] = repetition
                result["artifacts_equal_to_reference"] = True
                rows.append(result)
                (args.output / "results.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
                print(json.dumps({key: result[key] for key in ("scenario", "mode", "repetition", "counters", "sim_wallclock_s") if key in result}), flush=True)


if __name__ == "__main__":
    main()
