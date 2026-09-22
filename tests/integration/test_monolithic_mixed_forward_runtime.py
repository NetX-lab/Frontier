"""Real-runtime acceptance for one shared monolithic forward across phases.

The 71-case fidelity matrix cannot reach this shape: the public MoE wrappers
enforce `ATTN_TP == MOE_TP * MOE_EP` while the runtime enforces
`attn_tp * attn_dp == moe_tp * moe_ep`, and those have no common solution above
one attention-DP lane. This test therefore builds a valid runtime configuration
directly -- `attn_tp=1, attn_dp=2, moe_tp=1, moe_ep=2` on a monolithic MoE
Replica -- and runs the real `Simulator` event loop over it, with real
admission, ownership, synchronization and completion code. Deterministic
durations enter only through the predictor: constant profiling targets, plus an
observer that wraps `predict_stage_execution_time` without replacing it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_shared_monolithic_forward_completes_every_request(tmp_path):
    # The child must import the same checkout this test file came from, not
    # whichever tree an editable install happens to point at.
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(repo_root), os.environ.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
    )
    (tmp_path / "run.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout[-15000:]
    evidence = json.loads((tmp_path / "shared_forward_evidence.json").read_text())
    # The run has to reach the shape under test, or it proves nothing.
    assert evidence["mixed_phase_cohorts"] > 0, evidence


# Requests chosen so that chunked prefill leaves one lane prefilling while the
# other has already started decoding: unequal prefill lengths, unequal decode
# budgets, all arriving at once.
REQUEST_SHAPES = ((32, 4), (16, 4), (24, 3), (16, 3))


def _build_config(root, patch):
    import pandas as pd

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
        VllmV1SchedulerConfig,
    )
    from tests.integration.test_pr33_nondummy_acceptance import _model, _profiles

    model = _model("moe")
    original = BaseModelConfig.create_from_name
    patch.setattr(
        BaseModelConfig,
        "create_from_name",
        classmethod(
            lambda cls, name: model if name == model._model_name else original(name)
        ),
    )
    profiles = _profiles(root / "profiles", model, "moe")
    # Chunked prefill produces true mixed batches, whose decode attention is a
    # separate profiled operator. Add those rows at the same constant targets.
    for filename in ("attention.csv", "attention_kernel_only.csv"):
        path = root / "profiles" / filename
        frame = pd.read_csv(path, keep_default_na=False)
        rows = []
        for tp in (1, 2):
            for chunk in (1, 4, 8, 16):
                for decodes in (1, 2, 3):
                    row = frame.iloc[0].to_dict()
                    row.update(
                        num_tensor_parallel_workers=tp,
                        is_true_mixed_batch=True,
                        is_mixed_batch=True,
                        is_prefill=True,
                        batch_size=1 + decodes,
                        total_batch_size=1 + decodes,
                        num_prefill_seqs=1,
                        prefill_chunk_size=chunk,
                        total_prefill_tokens=chunk,
                        total_tokens=chunk + decodes,
                        decode_batch_size=decodes,
                        decode_avg_kv_cache_size=16,
                        kv_cache_size=16,
                        batch_composition_ratio=chunk / (chunk + decodes),
                        prefill_seq_lens=json.dumps([chunk]),
                        prefill_kv_cache_sizes=json.dumps([0]),
                    )
                    row["time_stats.attn_prefill.median"] = 0.08
                    row["time_stats.attn_decode.median"] = 0.05
                    row["time_stats.attn_kv_cache_save.median"] = 0.01
                    rows.append(row)
        pd.concat([frame, pd.DataFrame(rows)], ignore_index=True).to_csv(
            path, index=False
        )

    predictor = RandomForrestExecutionTimePredictorConfig(
        enable_dummy_mode=False,
        **profiles,
        num_estimators=[2],
        max_depth=[2],
        min_samples_split=[2],
        k_fold_cv_splits=2,
        num_training_job_threads=1,
        prediction_max_tokens_per_request=64,
        prediction_max_prefill_chunk_size=32,
        prediction_max_batch_size=4,
        kv_cache_prediction_granularity=16,
        skip_cpu_overhead_modeling=True,
    )
    replica = ReplicaConfig(
        model_name=model._model_name,
        device="a100",
        network_device="a100_pairwise_nvlink",
        num_pipeline_stages=1,
        attn_tensor_parallel_size=1,
        attn_dp=2,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=2,
        total_expert_num=8,
        router_topk=2,
    )
    return model, predictor, replica, (
        VllmV1SchedulerConfig,
        ClusterConfig,
        MetricsConfig,
        SimulationConfig,
        SyntheticRequestGeneratorConfig,
        FixedRequestLengthGeneratorConfig,
        PoissonRequestIntervalGeneratorConfig,
    )


def _phase_of(batch) -> str:
    """The rule a monolithic Replica uses to pick a batch's sync path.

    Stated here rather than imported, so the test's expectation does not move
    when the implementation helper does -- and so it can be run against a
    checkout that has no such helper.
    """

    return "prefill" if int(batch.num_prefill_tokens) > 0 else "decode"


def _drained(room) -> bool:
    """Return whether every allocated waiting-room leaf is empty."""

    for stages in room.values():
        for steps in stages.values():
            for layers in steps.values():
                for sync_stages in layers.values():
                    for entry in sync_stages.values():
                        if entry.get("batches"):
                            return False
    return True


def run_case(root: Path, *, reporting: bool):
    from frontier.entities import Request
    from frontier.request_generator.synthetic_request_generator import (
        SyntheticRequestGenerator,
    )
    from frontier.scheduler.utils import ep_wave_schedule
    from frontier.simulator import Simulator
    from frontier.types import ClusterType

    with pytest.MonkeyPatch.context() as patch:
        model, predictor_config, replica, classes = _build_config(root, patch)
        (
            VllmV1SchedulerConfig,
            ClusterConfig,
            MetricsConfig,
            SimulationConfig,
            SyntheticRequestGeneratorConfig,
            FixedRequestLengthGeneratorConfig,
            PoissonRequestIntervalGeneratorConfig,
        ) = classes
        cluster = ClusterConfig(
            replica_config=replica,
            replica_scheduler_config=VllmV1SchedulerConfig(
                num_blocks=128,
                block_size=16,
                batch_size_cap=4,
                max_tokens_in_batch=16,
                enable_chunked_prefill=True,
            ),
            execution_time_predictor_config=predictor_config,
        )
        config = SimulationConfig(
            simulation_mode="offline",
            sys_arch="co-location",
            enable_parallel_clusters=False,
            decode_cuda_graph_mode="none",
            cluster_config=cluster,
            metrics_config=MetricsConfig(
                output_dir=str(root / "metrics"),
                cache_dir=str(root / "cache"),
                run_id="shared_forward",
                write_metrics=True,
                store_request_metrics=True,
                store_batch_metrics=reporting,
                store_operation_metrics=reporting,
                store_utilization_metrics=reporting,
                store_plots=False,
                enable_chrome_trace=False,
                write_json_trace=False,
            ),
            request_generator_config=SyntheticRequestGeneratorConfig(
                num_requests=len(REQUEST_SHAPES),
                length_generator_config=FixedRequestLengthGeneratorConfig(
                    prefill_tokens=16, decode_tokens=3
                ),
                interval_generator_config=PoissonRequestIntervalGeneratorConfig(
                    qps=1e6
                ),
            ),
        )
        requests = [
            Request(0.0, prefill, decode) for prefill, decode in REQUEST_SHAPES
        ]
        patch.setattr(
            SyntheticRequestGenerator, "generate", lambda self: list(requests)
        )

        # Cohort membership and the prediction log, both observed without
        # changing what the runtime does.
        cohorts: list[dict] = []
        predictions: list[int] = []
        real_wave = ep_wave_schedule.schedule_layer_wave

        def observe_wave(scheduler, *, mode, batch, layer_id, cohort_batches=None, **kw):
            sources = cohort_batches if cohort_batches else {0: batch}
            cohorts.append(
                {
                    "layer_id": layer_id,
                    # Where this cohort starts in the prediction log, so the
                    # calls its own completion makes can be isolated.
                    "first_call": len(predictions),
                    "members": {
                        source.id: _phase_of(source)
                        for source in sources.values()
                        if not source.is_idle
                    },
                }
            )
            return real_wave(
                scheduler,
                mode=mode,
                batch=batch,
                layer_id=layer_id,
                cohort_batches=cohort_batches,
                **kw,
            )

        patch.setattr(ep_wave_schedule, "schedule_layer_wave", observe_wave)
        import frontier.scheduler.cluster_scheduler.base_cluster_scheduler as bcs

        patch.setattr(bcs, "schedule_layer_wave", observe_wave)

        simulator = Simulator(config)

        # The one injection point: wrap the predictor to record which batch each
        # attention-scope prediction was made for. Wrapping keeps the real
        # prediction; it only makes source attribution observable.
        predictor = simulator._global_scheduler.get_cluster_scheduler(
            ClusterType.MONOLITHIC
        )._predictor
        real_predict = predictor.predict_stage_execution_time

        def observe_predict(batch, stage_id, cluster_type=None, **kwargs):
            if kwargs.get("include_ffn") is False:
                predictions.append(int(batch.id))
            if cluster_type is None:
                return real_predict(batch, stage_id, **kwargs)
            return real_predict(batch, stage_id, cluster_type, **kwargs)

        patch.setattr(
            predictor, "predict_stage_execution_time", observe_predict, raising=False
        )

        simulator.run()

        cluster_scheduler = simulator._global_scheduler.get_cluster_scheduler(
            ClusterType.MONOLITHIC
        )
        mixed = [c for c in cohorts if len(set(c["members"].values())) > 1]
        evidence = {
            "reporting": reporting,
            "total_cohorts": len(cohorts),
            "mixed_phase_cohorts": len(mixed),
            "completed_requests": sum(request.completed for request in requests),
            "makespan": simulator._time,
        }

        # Every request finishes, exactly once, with every token accounted for.
        assert all(request.completed for request in requests), evidence
        for request, (prefill, decode) in zip(requests, REQUEST_SHAPES):
            assert request.num_prefill_tokens == prefill
            assert request.num_decode_tokens == decode
            assert request.num_processed_tokens == prefill + decode
        rows = _read_request_metrics(root)
        assert len(rows) == len(requests)
        assert len({row["Request Id"] for row in rows}) == len(requests)
        assert sum(int(float(row["request_num_tokens"])) for row in rows) == sum(
            prefill + decode for prefill, decode in REQUEST_SHAPES
        )

        # Each live source in a mixed cohort continued on its own prediction.
        # A cohort's completion runs after its own wave and before the next
        # wave is scheduled, so that slice of the prediction log belongs to it.
        boundaries = [cohort["first_call"] for cohort in cohorts] + [len(predictions)]
        for index, cohort in enumerate(cohorts):
            if len(set(cohort["members"].values())) < 2:
                continue
            window = set(predictions[cohort["first_call"] : boundaries[index + 1]])
            assert set(cohort["members"]) <= window, (cohort, sorted(window))

        # Nothing is stranded: no waiting room holds a batch and no stage
        # execution context still owns or queues a ticket.
        assert _drained(cluster_scheduler._forward_sync_waiting_room)
        for key, context in cluster_scheduler._stage_execution_contexts.items():
            assert context.is_idle, (key, context)
            assert context.queued_tickets == (), (key, context.queued_tickets)
        return evidence


def _read_request_metrics(root: Path):
    import csv

    paths = list(root.rglob("request_metrics.csv"))
    assert len(paths) == 1, paths
    with paths[0].open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main(root: Path) -> None:
    evidence = {}
    for reporting in (False, True):
        case_root = root / ("reporting_on" if reporting else "reporting_off")
        case_root.mkdir(parents=True, exist_ok=True)
        evidence["on" if reporting else "off"] = run_case(
            case_root, reporting=reporting
        )
    # Reporting is demand-driven: enabling it records more, but the simulated
    # execution -- the same cohorts and the same makespan -- does not move.
    assert evidence["on"]["makespan"] == evidence["off"]["makespan"], evidence
    assert evidence["on"]["total_cohorts"] == evidence["off"]["total_cohorts"]
    assert (
        evidence["on"]["mixed_phase_cohorts"] == evidence["off"]["mixed_phase_cohorts"]
    )
    merged = dict(evidence["off"])
    merged["reporting_variants"] = evidence
    (root / "shared_forward_evidence.json").write_text(
        json.dumps(merged, indent=2) + "\n"
    )
    print("mixed_phase_cohorts:", merged["mixed_phase_cohorts"])
    print("completed_requests:", merged["completed_requests"])


if __name__ == "__main__":
    main(Path(sys.argv[1]))
