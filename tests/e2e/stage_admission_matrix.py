#!/usr/bin/env python3
"""Case matrix for stage admission of attention-DP lanes under pipeline parallelism.

Runs the case list of the stage-admission-ordering plan
(``task_memory/task_2026-09-22_stage_admission_ordering/plan.md`` §4) on the
current source tree and writes, for each case, its inputs (``case.json``), its
run provenance (``run.json``) and one outcome artifact:

* ``success``: ``sha256sums.txt`` over the copied metrics tree;
* ``admission_deadlock``: ``state_report.json`` read from the live scheduler
  objects after the sequential run ends with work left;
* ``configuration_rejection`` / ``other_failure``: ``error.txt``.

Each case runs in its own child process because ``IS_MOE`` is process-global.
Every child writes its simulator output under ``<root>/work/<case_id>``, a path
shared by all sets, so that files embedding the output path compare byte for
byte between a set run before a change and one run after it.

Usage::

    python -m tests.e2e.stage_admission_matrix run --set base [--group G3a]
    python -m tests.e2e.stage_admission_matrix compare --before base --after after
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Iterable, Sequence

from tests.scratch_root import resolve_scratch_root


REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_DIR_NAME = "stage_admission_ordering"
DRAIN_MESSAGE = "Sequential simulation ended with non-empty scheduler state"
ATTN_DP_LANE = "ATTN_DP_LANE"

SUCCESS = "success"
ADMISSION_DEADLOCK = "admission_deadlock"
CONFIGURATION_REJECTION = "configuration_rejection"
OTHER_FAILURE = "other_failure"

# Plan §4.1 synthetic fixture and §4.7 vLLM-aligned fixture.
SYNTHETIC = "synthetic"
VLLM_ALIGNED = "vllm_aligned"
VLLM_ALIGNED_MODELS = {True: "Qwen3-30B-A3B-tiny", False: "Llama-3.2-1B-Instruct"}

PREFILL_ONLY = (16, 1)
PREFILL_DECODE = (16, 3)
VLLM_ALIGNED_PREFILL_ONLY = (256, 1)


@dataclass(frozen=True)
class Case:
    case_id: str
    group: str
    fixture: str = SYNTHETIC
    is_moe: bool = False
    attn_dp: int = 1
    stages: int = 1
    num_requests: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    arrival: str = "static"
    cc_backend: str = "analytical"
    recipe: str | None = None
    contention_witness: bool = False

    @property
    def path(self) -> str:
        """Plan §4.2 path before P0 classification refines L/T."""
        if self.recipe is not None or self.stages == 1 or self.attn_dp == 1:
            return "U"
        return "LT"


def _shape_id(group: str, is_moe: bool, attn_dp: int, stages: int, num_requests: int) -> str:
    kind = "moe" if is_moe else "dense"
    return f"{group}-{kind}-dp{attn_dp}-pp{stages}-n{num_requests}"


def _synthetic(group: str, is_moe: bool, attn_dp: int, stages: int, num_requests: int,
               lengths: tuple[int, int], **fields) -> Case:
    return Case(
        case_id=_shape_id(group, is_moe, attn_dp, stages, num_requests),
        group=group,
        is_moe=is_moe,
        attn_dp=attn_dp,
        stages=stages,
        num_requests=num_requests,
        prefill_tokens=lengths[0],
        decode_tokens=lengths[1],
        **fields,
    )


def _release_recipes() -> list[Case]:
    architecture_root = REPO_ROOT / "examples" / "architecture"
    recipes = []
    for architecture in ("co-location", "pdd", "pd-af-disagg"):
        for mode in ("offline", "online"):
            for script in sorted((architecture_root / architecture / mode).glob("*.sh")):
                recipes.append(
                    Case(
                        case_id=f"G1-{architecture}-{mode}-{script.stem}",
                        group="G1",
                        recipe=str(script.relative_to(REPO_ROOT)),
                    )
                )
    return recipes


def build_cases() -> list[Case]:
    """Return the plan §4.2 case list in a fixed order."""
    cases: list[Case] = []
    # R0 reproduces the author-reported shapes of design.md with their inputs:
    # Poisson arrivals, the default CC backend and prefill 16 / decode 3.
    r0_shapes = [
        (True, 2, 2, 3), (True, 2, 2, 4), (True, 2, 2, 6), (True, 4, 2, 8),
        (True, 2, 1, 6), (True, 2, 1, 12), (True, 4, 1, 8), (True, 4, 1, 12),
        (True, 1, 2, 6), (True, 1, 3, 6),
        (False, 2, 2, 6), (False, 4, 2, 8), (False, 2, 1, 6), (False, 4, 1, 8),
        (False, 1, 2, 6), (True, 2, 3, 6),
    ]
    for is_moe, attn_dp, stages, num_requests in r0_shapes:
        cases.append(
            _synthetic("R0", is_moe, attn_dp, stages, num_requests, PREFILL_DECODE,
                       arrival="poisson", cc_backend="default")
        )
    cases.extend(_release_recipes())
    for attn_dp in (2, 4):
        for stages in (1, 2, 3):
            for num_requests in (4, 8, 12):
                cases.append(_synthetic("G3a", True, attn_dp, stages, num_requests, PREFILL_ONLY))
    for attn_dp in (2, 4):
        for stages in (1, 2, 3):
            for num_requests in (4, 8):
                cases.append(_synthetic("G3b", True, attn_dp, stages, num_requests, PREFILL_DECODE))
    for attn_dp in (2, 4):
        for stages in (1, 2, 3):
            for num_requests in (4, 8):
                cases.append(
                    _synthetic("G4", False, attn_dp, stages, num_requests, PREFILL_DECODE,
                               contention_witness=stages > 1 and num_requests == 8)
                )
    for is_moe in (True, False):
        for stages in (1, 2, 3):
            cases.append(_synthetic("G5", is_moe, 1, stages, 6, PREFILL_DECODE))
    for is_moe in (True, False):
        for num_requests in (8, 16):
            cases.append(
                _synthetic("G7", is_moe, 2, 2, num_requests, VLLM_ALIGNED_PREFILL_ONLY,
                           fixture=VLLM_ALIGNED)
            )
    return cases


# ---------------------------------------------------------------------------
# Fixture (runs inside the child process)
# ---------------------------------------------------------------------------


def _synthetic_model(is_moe: bool):
    from frontier.config import BaseModelConfig
    from frontier.types import ActivationType, NormType

    model = BaseModelConfig(
        num_layers=6, num_q_heads=4, num_kv_heads=2, embedding_dim=256,
        mlp_hidden_dim=64, max_position_embeddings=4096, use_gated_mlp=True,
        use_bias=False, use_qkv_bias=False, activation=ActivationType.SILU,
        norm=NormType.RMS_NORM, post_attn_norm=True, vocab_size=1024,
        is_moe=is_moe, num_experts=8 if is_moe else 0,
        num_experts_per_tok=2 if is_moe else 0, torch_dtype="bfloat16",
    )
    model._model_name = f"stage_admission_{'moe' if is_moe else 'dense'}"
    registered = BaseModelConfig.create_from_name.__func__
    BaseModelConfig.create_from_name = classmethod(
        lambda cls, name: model if name == model._model_name else registered(cls, name)
    )
    return model._model_name


def build_config(case: Case, output_dir: Path, cache_dir: Path):
    """Build the SimulationConfig of one synthetic or vLLM-aligned case."""
    from frontier.cc_backend.cc_backend_config import AnalyticalCCBackendConfig
    from frontier.config import (
        ClusterConfig, FixedRequestLengthGeneratorConfig, MetricsConfig,
        PoissonRequestIntervalGeneratorConfig, RandomForrestExecutionTimePredictorConfig,
        ReplicaConfig, RoundRobinClusterSchedulerConfig, SimulationConfig,
        StaticRequestIntervalGeneratorConfig, SyntheticRequestGeneratorConfig,
        VllmV1SchedulerConfig,
    )

    if case.fixture == SYNTHETIC:
        model_name = _synthetic_model(case.is_moe)
        device, network_device = "a100", "a100_pairwise_nvlink"
        scheduler = VllmV1SchedulerConfig(
            num_blocks=128, block_size=16, batch_size_cap=4,
            max_tokens_in_batch=16, enable_chunked_prefill=True,
        )
    elif case.fixture == VLLM_ALIGNED:
        model_name = VLLM_ALIGNED_MODELS[case.is_moe]
        device, network_device = "h800", "h800_dgx"
        scheduler = VllmV1SchedulerConfig(
            num_blocks=1024, block_size=16, batch_size_cap=4,
            max_tokens_in_batch=case.prefill_tokens, enable_chunked_prefill=True,
        )
    else:
        raise ValueError(f"unknown fixture {case.fixture!r}")

    moe_fields = (
        dict(moe_tensor_parallel_size=1, moe_expert_parallel_size=case.attn_dp)
        if case.is_moe else {}
    )
    replica = ReplicaConfig(
        model_name=model_name, device=device, network_device=network_device,
        num_pipeline_stages=case.stages, attn_tensor_parallel_size=1,
        attn_dp=case.attn_dp, memory_margin_fraction=0.1, **moe_fields,
    )
    cluster_fields = {}
    if case.cc_backend == "analytical":
        cluster_fields["cc_backend_config"] = AnalyticalCCBackendConfig()
    elif case.cc_backend != "default":
        raise ValueError(f"unknown CC backend selector {case.cc_backend!r}")
    cluster = ClusterConfig(
        replica_config=replica,
        replica_scheduler_config=scheduler,
        cluster_scheduler_config=RoundRobinClusterSchedulerConfig(),
        execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
            enable_dummy_mode=True
        ),
        **cluster_fields,
    )
    if case.arrival == "static":
        interval = StaticRequestIntervalGeneratorConfig()
    elif case.arrival == "poisson":
        interval = PoissonRequestIntervalGeneratorConfig(qps=1e6)
    else:
        raise ValueError(f"unknown arrival process {case.arrival!r}")
    return SimulationConfig(
        simulation_mode="offline", sys_arch="co-location",
        enable_parallel_clusters=False, decode_cuda_graph_mode="none",
        cluster_config=cluster,
        metrics_config=MetricsConfig(
            output_dir=str(output_dir), cache_dir=str(cache_dir),
            run_id=case.case_id, write_metrics=True, store_request_metrics=True,
            store_plots=False, enable_chrome_trace=False, write_json_trace=False,
        ),
        request_generator_config=SyntheticRequestGeneratorConfig(
            num_requests=case.num_requests,
            length_generator_config=FixedRequestLengthGeneratorConfig(
                prefill_tokens=case.prefill_tokens, decode_tokens=case.decode_tokens,
            ),
            interval_generator_config=interval,
        ),
    )


# ---------------------------------------------------------------------------
# Drain state and outcome classification (child process)
# ---------------------------------------------------------------------------


def _ticket_view(ticket) -> dict:
    return {
        "admission_seq": ticket.admission_seq,
        "operation_id": str(ticket.operation_id),
        "scope": ticket.scope,
    }


def build_state_report(simulator) -> dict:
    """Read stage contexts, lane queues and sync rooms after a drain."""
    from frontier.types import ClusterType

    cluster_scheduler = simulator.scheduler.get_cluster_scheduler(ClusterType.MONOLITHIC)
    lanes = {}
    queued_owner = {}
    for (replica_id, lane_id), replica_scheduler in sorted(
        cluster_scheduler._replica_schedulers.items(), key=lambda item: str(item[0])
    ):
        stage_views = []
        for stage_id in range(replica_scheduler._num_stages):
            stage = replica_scheduler.get_replica_stage_scheduler(stage_id)
            heap = []
            for batch in stage.get_queue_batches():
                ticket = batch._stage_admission_ticket
                queued_owner[(replica_id, stage_id, ticket.admission_seq)] = lane_id
                heap.append({"batch_id": batch.id, "global_id": batch.global_id,
                             **_ticket_view(ticket)})
            stage_views.append({"busy": stage.is_busy, "heap": heap})
        lanes[f"{replica_id}/{lane_id}"] = {
            "replica_id": replica_id, "lane": lane_id, "stages": stage_views,
        }

    contexts = []
    for (replica_id, stage_id), context in sorted(cluster_scheduler._stage_execution_contexts.items()):
        contexts.append({
            "replica_id": replica_id,
            "stage_id": stage_id,
            "capacity": context.full_stage_capacity,
            "sealed": context.forward_group_sealed,
            "bound_group": context._forward_group_id,
            "ep_wave_active": context._active_ep_ticket is not None,
            "active_full_stage": sorted(
                (_ticket_view(ticket) for ticket in context._active_full_stage_tickets),
                key=lambda view: view["admission_seq"],
            ),
            "fifo": [
                {**_ticket_view(ticket),
                 "lane": queued_owner.get((replica_id, stage_id, ticket.admission_seq))}
                for ticket in context.queued_tickets
            ],
        })

    rooms = []
    for room_name in ("_prefill_sync_waiting_room", "_decode_sync_waiting_room"):
        by_replica = getattr(cluster_scheduler, room_name) or {}
        for replica_id, by_stage in by_replica.items():
            for stage_id, by_step in by_stage.items():
                for step, by_layer in by_step.items():
                    for layer, by_sync in by_layer.items():
                        for sync_stage, room in by_sync.items():
                            if not room["batches"]:
                                continue
                            rooms.append({
                                "room": room_name.strip("_"),
                                "replica_id": replica_id, "stage_id": stage_id,
                                "step": step, "layer": layer, "sync_stage": str(sync_stage),
                                "lanes_present": sorted(room["batches"]),
                            })
    return {
        "simulation_time": simulator._time,
        "contexts": contexts,
        "lanes": lanes,
        "sync_rooms": rooms,
    }


def has_admission_deadlock_signature(report: dict) -> bool:
    """Plan §4.3: a busy lane's queued ticket heads the FIFO while an idle lane
    with queued work, needed by that lane's sync room, is refused behind it."""
    lanes = report["lanes"]
    for context in report["contexts"]:
        if (context["ep_wave_active"] or context["sealed"] or not context["fifo"]
                or len(context["active_full_stage"]) >= context["capacity"]):
            continue
        head = context["fifo"][0]
        if head["scope"] != "FULL_STAGE_WORLD" or head["lane"] is None:
            continue
        replica_id, stage_id = context["replica_id"], context["stage_id"]
        head_stage = lanes[f"{replica_id}/{head['lane']}"]["stages"][stage_id]
        if not head_stage["busy"]:
            continue
        for lane in lanes.values():
            if lane["replica_id"] != replica_id or lane["lane"] == head["lane"]:
                continue
            stage = lane["stages"][stage_id]
            if stage["busy"] or not stage["heap"]:
                continue
            if stage["heap"][0]["admission_seq"] <= head["admission_seq"]:
                continue
            for room in report["sync_rooms"]:
                if (room["replica_id"] == replica_id and room["stage_id"] == stage_id
                        and head["lane"] in room["lanes_present"]
                        and lane["lane"] not in room["lanes_present"]):
                    return True
    return False


def _run_simulator_case(case: Case, work_dir: Path, case_dir: Path) -> dict:
    output_root = work_dir / "metrics"
    try:
        config = build_config(case, output_root, work_dir / "cache")
        from frontier.simulator import Simulator

        simulator = Simulator(config)
    except ValueError as exc:
        (case_dir / "error.txt").write_text(traceback.format_exc())
        return {"outcome": CONFIGURATION_REJECTION, "exception": repr(exc)}
    resolved_config = json.loads(
        (Path(config.metrics_config.output_dir) / "config.json").read_text()
    )
    (case_dir / "resolved_config.json").write_text(json.dumps(resolved_config, indent=1, sort_keys=True))
    try:
        simulator.run()
    except RuntimeError as exc:
        if not str(exc).startswith(DRAIN_MESSAGE):
            (case_dir / "error.txt").write_text(traceback.format_exc())
            return {"outcome": OTHER_FAILURE, "exception": repr(exc)[:2000]}
        report = build_state_report(simulator)
        (case_dir / "state_report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
        outcome = ADMISSION_DEADLOCK if has_admission_deadlock_signature(report) else OTHER_FAILURE
        return {"outcome": outcome, "exception": DRAIN_MESSAGE,
                "simulation_time": report["simulation_time"]}
    except Exception as exc:  # classified, never converted into success
        (case_dir / "error.txt").write_text(traceback.format_exc())
        return {"outcome": OTHER_FAILURE, "exception": repr(exc)[:2000]}
    requests = list(simulator._all_requests)
    completed = sum(1 for request in requests if request.completed)
    if completed != len(requests):
        (case_dir / "error.txt").write_text(
            f"run returned with {completed} of {len(requests)} requests completed\n"
        )
        return {"outcome": OTHER_FAILURE, "exception": "incomplete requests"}
    return {"outcome": SUCCESS, "completed_requests": completed,
            "metrics_run_dir": str(Path(config.metrics_config.output_dir).relative_to(work_dir))}


def _run_recipe_case(case: Case, work_dir: Path, case_dir: Path) -> dict:
    env = dict(os.environ)
    env.update({
        "PYTHON_BIN": sys.executable,
        "METRICS_OUTPUT_DIR": str(work_dir / "metrics"),
        "RUN_ID": case.case_id,
    })
    result = subprocess.run(
        ["bash", str(REPO_ROOT / case.recipe)], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True,
    )
    (case_dir / "stdout.log").write_text(result.stdout[-200_000:] + result.stderr[-200_000:])
    if result.returncode != 0:
        (case_dir / "error.txt").write_text(result.stderr[-50_000:])
        return {"outcome": OTHER_FAILURE, "exception": f"exit code {result.returncode}"}
    return {"outcome": SUCCESS}


def run_case_in_child(case: Case, root: Path, set_name: str) -> None:
    """Child entry point: run one case, then publish its artifacts."""
    case_dir = root / set_name / case.case_id
    work_dir = root / "work" / case.case_id
    for directory in (case_dir, work_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    started = time.time()
    if case.recipe is None:
        result = _run_simulator_case(case, work_dir, case_dir)
    else:
        result = _run_recipe_case(case, work_dir, case_dir)
    result["wall_start"] = started
    result["wall_end"] = time.time()
    if result["outcome"] == SUCCESS:
        shutil.copytree(work_dir / "metrics", case_dir / "metrics")
        (case_dir / "sha256sums.txt").write_text(sha256_lines(case_dir / "metrics"))
    shutil.rmtree(work_dir)
    (case_dir / "outcome.json").write_text(json.dumps(result, indent=1, sort_keys=True))


def sha256_lines(directory: Path) -> str:
    lines = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(directory)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Lane-overlap metric (plan §4.5)
# ---------------------------------------------------------------------------


def interval_overlap(intervals: Iterable[tuple[float, float, object]]) -> dict:
    """Half-open ``[start, end)`` intervals keyed by lane.

    Returns the time with at least one open lane, the time with at least two
    distinct open lanes, the peak number of distinct open lanes, the latest
    end, and whether any lane overlaps itself.
    """
    intervals = list(intervals)
    events = []
    for start, end, lane in intervals:
        if end > start:
            events.append((start, 1, lane))
            events.append((end, -1, lane))
    # Ends sort before starts at the same instant: touching intervals do not overlap.
    events.sort(key=lambda event: (event[0], event[1]))
    open_by_lane: dict[object, int] = defaultdict(int)
    busy_time = multi_lane_time = 0.0
    peak_lanes = 0
    self_overlap = False
    previous_time = None
    for event_time, delta, lane in events:
        if previous_time is not None:
            open_lanes = sum(1 for count in open_by_lane.values() if count > 0)
            if open_lanes >= 1:
                busy_time += event_time - previous_time
            if open_lanes >= 2:
                multi_lane_time += event_time - previous_time
        open_by_lane[lane] += delta
        if open_by_lane[lane] > 1:
            self_overlap = True
        peak_lanes = max(peak_lanes, sum(1 for count in open_by_lane.values() if count > 0))
        previous_time = event_time
    return {
        "busy_time": busy_time,
        "multi_lane_busy_time": multi_lane_time,
        "peak_lanes": peak_lanes,
        "makespan": max((end for _, end, _ in intervals), default=0.0),
        "self_overlap": self_overlap,
    }


def read_ledger(metrics_dir: Path) -> list[dict]:
    paths = sorted(metrics_dir.rglob("frontier_stage_batch_ledger.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected one stage ledger under {metrics_dir}, found {len(paths)}")
    return [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]


def lane_intervals(rows: Sequence[dict]) -> dict[tuple, list[tuple[float, float, int]]]:
    """``ATTN_DP_LANE`` ledger intervals per physical stage."""
    by_stage: dict[tuple, list[tuple[float, float, int]]] = defaultdict(list)
    for row in rows:
        if row["execution_scope"] != ATTN_DP_LANE:
            continue
        key = (row["cluster_type"], row["replica_id"], row["stage_id"])
        by_stage[key].append((row["stage_start_ts"], row["stage_end_ts"], row["replica_local_id"]))
    return dict(by_stage)


def ledger_lane_metric(metrics_dir: Path) -> dict:
    """Plan §4.5 metric per physical stage, keyed ``cluster/replica/stage``."""
    stages = {}
    for (cluster_type, replica_id, stage_id), intervals in sorted(lane_intervals(read_ledger(metrics_dir)).items()):
        metric = interval_overlap(intervals)
        metric["lanes"] = sorted({lane for _, _, lane in intervals})
        stages[f"{cluster_type}/{replica_id}/{stage_id}"] = metric
    return stages


# ---------------------------------------------------------------------------
# Parent: set runner and provenance
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def set_provenance() -> dict:
    distributions = sorted(
        f"{dist.metadata['Name']}=={dist.version}" for dist in importlib_metadata.distributions()
    )
    status = _git("status", "--porcelain", "--", ".", ":!task_memory")
    return {
        "interpreter": sys.executable,
        "python_vv": subprocess.run([sys.executable, "-VV"], check=True,
                                    capture_output=True, text=True).stdout.strip(),
        "distributions_sha256": hashlib.sha256("\n".join(distributions).encode()).hexdigest(),
        "git_head": _git("rev-parse", "HEAD"),
        "clean_outside_task_memory": status == "",
        "status_outside_task_memory": status,
    }


def matrix_root() -> Path:
    return resolve_scratch_root() / MATRIX_DIR_NAME


def _run_one(case: Case, root: Path, set_name: str, provenance: dict) -> dict:
    command = [sys.executable, "-m", "tests.e2e.stage_admission_matrix", "child",
               "--set", set_name, "--case", case.case_id]
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), WANDB_DISABLED="true",
               VIDUR_DISABLE_WANDB="1")
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    case_dir = root / set_name / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    outcome_path = case_dir / "outcome.json"
    if result.returncode != 0 or not outcome_path.exists():
        (case_dir / "error.txt").write_text(result.stdout[-50_000:] + result.stderr[-50_000:])
        outcome = {"outcome": OTHER_FAILURE, "exception": f"child exit code {result.returncode}"}
    else:
        outcome = json.loads(outcome_path.read_text())
    (case_dir / "case.json").write_text(json.dumps(asdict(case), indent=1, sort_keys=True))
    run = {"command": command, **provenance, **outcome}
    (case_dir / "run.json").write_text(json.dumps(run, indent=1, sort_keys=True))
    return {"case_id": case.case_id, "group": case.group, "outcome": outcome["outcome"],
            "exception": outcome.get("exception")}


def run_set(set_name: str, cases: Sequence[Case], jobs: int) -> list[dict]:
    root = matrix_root()
    (root / set_name).mkdir(parents=True, exist_ok=True)
    provenance = set_provenance()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(lambda case: _run_one(case, root, set_name, provenance), cases))
    if set_provenance()["git_head"] != provenance["git_head"]:
        raise RuntimeError("git HEAD changed while the set was running")
    index = root / set_name / "cases.jsonl"
    existing = {}
    if index.exists():
        existing = {row["case_id"]: row for row in map(json.loads, index.read_text().splitlines())}
    existing.update({row["case_id"]: row for row in rows})
    order = [case.case_id for case in build_cases()]
    index.write_text("".join(json.dumps(existing[case_id]) + "\n"
                             for case_id in order if case_id in existing))
    return rows


# ---------------------------------------------------------------------------
# Parent: before/after comparison (plan §4.4)
# ---------------------------------------------------------------------------


def _case_state(root: Path, set_name: str, case_id: str) -> dict:
    case_dir = root / set_name / case_id
    run = json.loads((case_dir / "run.json").read_text())
    state = {"outcome": run["outcome"]}
    if run["outcome"] == SUCCESS:
        state["sha256sums"] = (case_dir / "sha256sums.txt").read_text()
        if (case_dir / "metrics").exists():
            state["metrics_dir"] = case_dir / "metrics"
    return state


def _conservation(case: Case, metrics_dir: Path) -> dict:
    paths = sorted(metrics_dir.rglob("request_metrics.csv"))
    if len(paths) != 1:
        return {"ok": False, "reason": f"{len(paths)} request_metrics.csv files"}
    with paths[0].open() as handle:
        rows = list(csv.DictReader(handle))
    prefill = sum(int(float(row["request_num_prefill_tokens"])) for row in rows)
    decode = sum(int(float(row["request_num_decode_tokens"])) for row in rows)
    expected = (case.num_requests, case.num_requests * case.prefill_tokens,
                case.num_requests * case.decode_tokens)
    observed = (len(rows), prefill, decode)
    return {"ok": observed == expected, "observed": observed, "expected": expected}


def _differing_files(before: str, after: str) -> list[str]:
    def parse(text):
        return dict(reversed(line.split("  ", 1)) for line in text.splitlines() if line)
    left, right = parse(before), parse(after)
    return sorted(name for name in set(left) | set(right) if left.get(name) != right.get(name))


def compare_sets(before: str, after: str) -> list[dict]:
    root = matrix_root()
    rows = []
    for case in build_cases():
        if not (root / before / case.case_id / "run.json").exists():
            continue
        base = _case_state(root, before, case.case_id)
        new = _case_state(root, after, case.case_id)
        row = {"case_id": case.case_id, "group": case.group,
               "before": base["outcome"], "after": new["outcome"]}
        if case.group == "R0":
            row.update(path="R0", verdict="informational")
        elif case.path == "U":
            row["path"] = "U"
            identical = base["outcome"] == new["outcome"] == SUCCESS and base["sha256sums"] == new["sha256sums"]
            row["verdict"] = "PASS" if identical else "STOP"
            if base["outcome"] == new["outcome"] == SUCCESS and not identical:
                row["differing_files"] = _differing_files(base["sha256sums"], new["sha256sums"])
        elif base["outcome"] == ADMISSION_DEADLOCK:
            row["path"] = "L"
            if new["outcome"] == SUCCESS:
                row["conservation"] = _conservation(case, new["metrics_dir"])
                row["verdict"] = "PASS" if row["conservation"]["ok"] else "STOP"
            else:
                row["verdict"] = "STOP"
        elif base["outcome"] == SUCCESS:
            row["path"] = "T"
            if new["outcome"] != SUCCESS:
                row["verdict"] = "STOP"
            else:
                before_metric = ledger_lane_metric(base["metrics_dir"])
                after_metric = ledger_lane_metric(new["metrics_dir"])
                row["lane_metric_before"] = before_metric
                row["lane_metric_after"] = after_metric
                checks_ok = all(
                    not stage["self_overlap"] and stage["peak_lanes"] <= case.attn_dp
                    for stage in after_metric.values()
                )
                identical = base["sha256sums"] == new["sha256sums"]
                if not identical:
                    row["differing_files"] = _differing_files(base["sha256sums"], new["sha256sums"])
                if case.contention_witness:
                    # The fraction, not the absolute overlap: admitting more lanes
                    # together also shortens the busy period.
                    fraction = lambda metric: (sum(stage["multi_lane_busy_time"] for stage in metric.values())
                                               / sum(stage["busy_time"] for stage in metric.values()))
                    row["witness_increase"] = fraction(after_metric) > fraction(before_metric)
                    checks_ok = checks_ok and row["witness_increase"]
                row["verdict"] = ("PASS" if identical and checks_ok
                                  else "EXPLAIN" if checks_ok else "STOP")
        else:
            row.update(path="class", verdict="STOP")
        rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run cases into one named set")
    run_parser.add_argument("--set", required=True)
    run_parser.add_argument("--group", action="append", default=[])
    run_parser.add_argument("--case", action="append", default=[])
    run_parser.add_argument("--jobs", type=int, default=8)
    child_parser = commands.add_parser("child", help=argparse.SUPPRESS)
    child_parser.add_argument("--set", required=True)
    child_parser.add_argument("--case", required=True)
    compare_parser = commands.add_parser("compare", help="apply the plan §4.4 paths")
    compare_parser.add_argument("--before", required=True)
    compare_parser.add_argument("--after", required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    cases_by_id = {case.case_id: case for case in build_cases()}
    if args.command == "child":
        run_case_in_child(cases_by_id[args.case], matrix_root(), args.set)
        return 0
    if args.command == "compare":
        rows = compare_sets(args.before, args.after)
        args.output.write_text(json.dumps(rows, indent=1, sort_keys=True, default=str))
        for row in rows:
            print(f"{row['case_id']:<48} {row['path']:<5} {row['before']:<24} {row['after']:<24} {row['verdict']}")
        return 0
    selected = [case for case in cases_by_id.values()
                if (not args.group or case.group in args.group)
                and (not args.case or case.case_id in args.case)]
    for row in run_set(args.set, selected, args.jobs):
        print(f"{row['case_id']:<48} {row['outcome']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
