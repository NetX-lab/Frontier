"""CPU-overhead CSVs from vLLM CPU-probe logs.

The CPU probe of Frontier's instrumented vLLM (``VLLM_FRONTIER_CPU_PROBE_LOG_PATH``)
writes ``time.perf_counter`` stamps, which every process on one host shares:

- The engine log ``cpu_probe[_dp<d>].jsonl`` holds one record per engine step:
  step_start, schedule_end, execute_end, update_end and the tokens scheduled per
  request.
- Each pipeline stage writes ``<engine log stem>_pp<p><suffix>``, ``_pp0`` for a
  single-stage engine, with one record per forward: the runner's stamps
  (execute_start, preprocess_end, forward_start, forward_context_entered,
  forward_end, and on the last stage sample_end), the start and end of the
  forward's DP token-count all-reduce, the CUDA graph mode, and the forward's
  device time forward_device_ms, timed by CUDA events from
  forward_context_entered. The n-th record of every stage log is the forward of
  the engine's n-th step that scheduled tokens.

Frontier prices each stage's forward from its operator tables, and prices itself
the waits between stages, between DP engines and for a PD decode instance's KV
load. The CPU-overhead terms price every other host interval of a step, each
once, on the stage that runs it:

    schedule               stage 0: step_start -> schedule_end
    prepare_inputs_e2e     each stage: dispatch start -> forward_context_entered,
                           less the DP wait and the KV-load wait
    forward_launch         each stage, eager forwards: launch start -> forward_end
    forward_drain          each stage, eager forwards: forward_end -> device end,
                           at least 0
    sampler_e2e            last stage: max(forward_end, device end) -> sample_end
    process_model_outputs  last stage: sample_end -> outputs end
    ray_comm_time          0 (no Ray hop)

and every other term is 0, where

- the dispatch start of stage 0 is the later of schedule_end and the end of
  stage 0's previous forward (send_end, or execute_return without a send), which
  leaves out queueing behind the previous batch; a later stage starts at
  execute_start, after its receive wait;
- the DP wait is the all-reduce's duration beyond the all-reduce's own latency,
  the median duration on the DP engine that reaches it last. The engines of one
  all-reduce overlap in it, so it is found as a set of overlapping intervals
  with one record from every DP engine of the serving instance. When no
  all-reduce has one (every partner ran a DP dummy forward, which writes no
  record), the latency is the median of all recorded all-reduce durations. The
  all-reduce runs at forward-context entry for eager forwards and in the input
  preparation for CUDA graphs;
- the KV-load wait is the forward-context entry, less an all-reduce inside it,
  beyond its median over the stage log. A PD decode instance's KV connector
  loads there; elsewhere the excess is noise around zero;
- the launch start is the later of forward_context_entered and the upstream
  stage's device end; the device end is the launch start plus forward_device_ms;
- the outputs end is the next engine record's step_start when that step shares a
  request and starts after update_end. Otherwise it is update_end, which leaves
  out the engine's wait for an arrival and, in vLLM's batch-queue loop, the
  overlap with the next batch.

Forwards are grouped per stage by Frontier's CPU-overhead identity (batch_size,
num_prefill_tokens, num_decode_tokens), counting a request scheduled more than
one token as prefill, which holds for logs without speculative decoding. A
forward that replayed a FULL CUDA graph belongs to the kernel-only family and
publishes no forward_launch or forward_drain; every other forward is eager.
Rows are keyed by pipeline_stage_id.

With ``--engine_idle_edges_ms``, rows are also keyed by engine_idle_ms: the
largest edge at or below the engine's idle time before the step, from the latest
update_end of its earlier steps or the end of its latest DP dummy forward to the
step's step_start. An engine's first step that schedules tokens carries the
engine's one-time start costs, which a ground-truth run's warmups absorb, and is
left out. A DP>1
instance reads its dummy forwards from the ``dummy_pass`` records of
``dp_placement/*.jsonl`` beside its engine logs, whose ``engine`` field is the
position of the engine log in ``--cpu_probe_logs``; they share the
perf_counter clock (CLOCK_MONOTONIC).

    python -m frontier.profiling.cpu_overhead.vllm_cpu_probe \\
        --cpu_probe_logs run/prefill/cpu_probe.jsonl \\
        --cpu_probe_logs run/decode/cpu_probe.jsonl \\
        --model_name llama2_7b_dense_example --tensor_parallel_degree 1 \\
        --profiling_precision BF16 --scheduling_mode sync \\
        --eager_output_file cpu_overheads.csv \\
        --kernel_only_output_file cpu_overheads_kernel_only.csv

Each ``--cpu_probe_logs`` lists the engine logs of one serving instance, one per
DP engine. A PP>1 engine adds ``--num_pipeline_stages``.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence

import numpy as np
import pandas as pd

from frontier.config.precision_type import PrecisionType
from frontier.logger import init_logger
from frontier.profiling.cpu_overhead.schema import (
    CPU_OVERHEAD_ENGINE_IDLE_COLUMN,
    CPU_OVERHEAD_PIPELINE_STAGE_COLUMN,
    VALID_SCHEDULING_MODES,
)
from frontier.profiling.cpu_overhead.validation import validate_cpu_overhead_dataframe
from frontier.types import MeasurementType

logger = init_logger(__name__)

StepIdentity = tuple[int, int, int]


class StageStep(NamedTuple):
    """One forward's row key and CPU-overhead terms in milliseconds."""

    stage_id: int
    family: MeasurementType
    identity: StepIdentity
    terms: dict[str, float]
    engine_idle_ms: float | None = None


def load_cpu_probe_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def step_identity(num_scheduled_tokens: Mapping[str, int]) -> StepIdentity:
    tokens = list(num_scheduled_tokens.values())
    return len(tokens), sum(t for t in tokens if t > 1), sum(t for t in tokens if t == 1)


def stage_log_path(engine_log: Path, pipeline_stage_id: int) -> Path:
    return engine_log.with_name(f"{engine_log.stem}_pp{pipeline_stage_id}{engine_log.suffix}")


def dp_allreduce_latency(stage_logs_per_engine: Sequence[Sequence[Sequence[Mapping]]]) -> float:
    """Return the DP token-count all-reduce's own latency in milliseconds.

    ``stage_logs_per_engine[e][p]`` holds DP engine e's stage-p records. The latency is the
    median duration on the engine that reaches an all-reduce last, over the all-reduces with a
    record from every engine; one with a DP dummy forward has none. Without such an all-reduce,
    it is the median of all recorded all-reduce durations.
    """

    num_engines = len(stage_logs_per_engine)
    if num_engines < 2:
        raise ValueError(
            "the CPU-probe records carry DP all-reduce stamps; list the logs of every DP engine "
            "of the serving instance in one --cpu_probe_logs"
        )
    latest_starter_ms, recorded_ms = [], []
    for stage_logs in zip(*stage_logs_per_engine):
        intervals = sorted(
            (record["dp_allreduce_start"], record["dp_allreduce_end"], engine)
            for engine, records in enumerate(stage_logs)
            for record in records
            if record["dp_allreduce_start"] is not None
        )
        recorded_ms += [(end - start) * 1e3 for start, end, _ in intervals]
        groups, group_end = [], -math.inf
        for start, end, engine in intervals:
            if start > group_end:
                groups.append([])
            groups[-1].append((start, end, engine))
            group_end = max(group_end, end)
        for group in groups:
            if sorted(engine for *_, engine in group) == list(range(num_engines)):
                start, end, _ = max(group)
                latest_starter_ms.append((end - start) * 1e3)
    return float(np.median(latest_starter_ms or recorded_ms))


def _context_entry_ms(stage: Mapping) -> float:
    entry = stage["forward_context_entered"] - stage["forward_start"]
    if stage["dp_allreduce_start"] is not None and stage["dp_allreduce_start"] >= stage["forward_start"]:
        entry -= stage["dp_allreduce_end"] - stage["dp_allreduce_start"]
    return entry * 1e3


def dummy_pass_ends(placement_dir: Path) -> dict[int, list[float]]:
    """Return each DP engine's dummy-forward end stamps, in order, from its dp_placement records."""

    ends = defaultdict(list)
    for path in sorted(placement_dir.glob("*.jsonl")):
        for record in load_cpu_probe_log(path):
            if record["kind"] == "dummy_pass":
                ends[record["engine"]].append(record["monotonic"])
    return {engine: sorted(engine_ends) for engine, engine_ends in ends.items()}


def stage_overhead_terms(
    engine_records: Sequence[Mapping],
    stage_records: Sequence[Sequence[Mapping]],
    *,
    dp_allreduce_ms: float | None,
    engine_idle_edges_ms: Sequence[float] | None = None,
    dummy_ends: Sequence[float] = (),
) -> list[StageStep]:
    """Return each forward's row key and CPU-overhead terms."""

    scheduled = [record["num_scheduled_tokens"] for record in engine_records if record["num_scheduled_tokens"]]
    for stage_id, records in enumerate(stage_records):
        if [record["num_scheduled_tokens"] for record in records] != scheduled:
            raise ValueError(
                f"pipeline stage {stage_id} logged {len(records)} forwards whose scheduled tokens do not "
                f"match the engine's {len(scheduled)} steps that scheduled tokens"
            )
    entry_medians = [float(np.median([_context_entry_ms(stage) for stage in records]) if records else 0.0)
                     for records in stage_records]
    last_stage = len(stage_records) - 1
    stage_steps = []
    previous_forward_end = -math.inf
    # A batch-queue step's update_end stamps the update of its own batch's output,
    # which can follow later steps, so the engine is busy until the latest one.
    previous_iteration_end = -math.inf
    n = 0
    for record, following in zip(engine_records, [*engine_records[1:], None]):
        engine_busy_until = previous_iteration_end
        previous_iteration_end = max(previous_iteration_end, record["update_end"])
        if not record["num_scheduled_tokens"]:
            continue
        identity = step_identity(record["num_scheduled_tokens"])
        idle_edge = None
        if engine_idle_edges_ms is not None:
            dummy_index = bisect.bisect_right(dummy_ends, record["step_start"])
            if dummy_index:
                engine_busy_until = max(engine_busy_until, dummy_ends[dummy_index - 1])
            idle_ms = max(0.0, (record["step_start"] - engine_busy_until) * 1e3)
            idle_edge = engine_idle_edges_ms[bisect.bisect_right(engine_idle_edges_ms, idle_ms) - 1]
        device_end = -math.inf
        for stage_id, records in enumerate(stage_records):
            stage = records[n]
            if stage_id == 0:
                dispatch_start = max(record["schedule_end"], previous_forward_end)
                previous_forward_end = stage["send_end"] if stage["send_end"] is not None else stage["execute_return"]
            else:
                dispatch_start = stage["execute_start"]
            dp_wait = 0.0
            if stage["dp_allreduce_start"] is not None:
                duration_ms = (stage["dp_allreduce_end"] - stage["dp_allreduce_start"]) * 1e3
                dp_wait = max(0.0, duration_ms - dp_allreduce_ms)
            kv_load_wait = max(0.0, _context_entry_ms(stage) - entry_medians[stage_id])
            launch_start = max(stage["forward_context_entered"], device_end)
            device_end = launch_start + stage["forward_device_ms"] * 1e-3
            terms = {
                "schedule": (record["schedule_end"] - record["step_start"]) * 1e3 if stage_id == 0 else 0.0,
                "prepare_inputs_e2e": (stage["forward_context_entered"] - dispatch_start) * 1e3 - dp_wait - kv_load_wait,
                "sampler_e2e": 0.0,
                "process_model_outputs": 0.0,
            }
            if stage_id == last_stage:
                outputs_end = record["update_end"]
                if (following is not None and following["step_start"] >= record["update_end"]
                        and set(following["num_scheduled_tokens"]) & set(record["num_scheduled_tokens"])):
                    outputs_end = following["step_start"]
                terms["sampler_e2e"] = (stage["sample_end"] - max(stage["forward_end"], device_end)) * 1e3
                terms["process_model_outputs"] = (outputs_end - stage["sample_end"]) * 1e3
            if stage["cudagraph_runtime_mode"] == "FULL":
                family = MeasurementType.KERNEL_ONLY
            else:
                family = MeasurementType.CUDA_EVENT
                terms["forward_launch"] = (stage["forward_end"] - launch_start) * 1e3
                terms["forward_drain"] = max(0.0, device_end - stage["forward_end"]) * 1e3
            if n or engine_idle_edges_ms is None:
                stage_steps.append(StageStep(stage_id, family, identity, terms, idle_edge))
        n += 1
    return stage_steps


def instance_stage_steps(
    engine_logs: Sequence[Path], num_pipeline_stages: int, engine_idle_edges_ms: Sequence[float] | None = None
) -> list[StageStep]:
    """Read one serving instance's DP engine logs with their stage logs and return their forwards' terms."""

    engines = [
        (load_cpu_probe_log(log), [load_cpu_probe_log(stage_log_path(log, p)) for p in range(num_pipeline_stages)])
        for log in engine_logs
    ]
    dp_allreduce_ms = None
    if any(stage["dp_allreduce_start"] is not None
           for _, stage_logs in engines for records in stage_logs for stage in records):
        dp_allreduce_ms = dp_allreduce_latency([stage_logs for _, stage_logs in engines])
        logger.info("DP all-reduce latency of %s: %.3f ms.", [str(log) for log in engine_logs], dp_allreduce_ms)
    dummy_ends = {}
    if engine_idle_edges_ms is not None and len(engine_logs) > 1:
        placement_dir = Path(engine_logs[0]).parent / "dp_placement"
        dummy_ends = dummy_pass_ends(placement_dir)
        if not dummy_ends:
            raise ValueError(
                f"engine-idle buckets of a DP>1 instance need its dummy forwards; no dummy_pass record in "
                f"{placement_dir}"
            )
    return [
        step
        for engine, (engine_records, stage_logs) in enumerate(engines)
        for step in stage_overhead_terms(
            engine_records, stage_logs, dp_allreduce_ms=dp_allreduce_ms,
            engine_idle_edges_ms=engine_idle_edges_ms, dummy_ends=dummy_ends.get(engine, ()),
        )
    ]


def cpu_overhead_tables(
    stage_steps: Sequence[StageStep],
    *,
    model_name: str,
    tensor_parallel_degree: int,
    profiling_precision: str,
    scheduling_mode: str,
) -> dict[MeasurementType, pd.DataFrame]:
    """Aggregate forward terms into one validated CPU-overhead table per family, keyed by stage."""

    groups = defaultdict(list)
    for step in stage_steps:
        groups[step.family, step.stage_id, step.identity, step.engine_idle_ms].append(step.terms)
    rows = defaultdict(list)
    for (family, stage_id, identity, idle_edge), group in sorted(groups.items(), key=lambda item: item[0][1:]):
        batch_size, num_prefill_tokens, num_decode_tokens = identity
        row = {
            "model_name": model_name,
            "batch_size": batch_size,
            "tensor_parallel_degree": tensor_parallel_degree,
            "num_prefill_tokens": num_prefill_tokens,
            "num_decode_tokens": num_decode_tokens,
            "scheduling_mode": scheduling_mode,
        }
        for term in group[0]:
            term_values = [terms[term] for terms in group]
            row[f"{term}_mean"] = float(np.mean(term_values))
            row[f"{term}_median"] = float(np.median(term_values))
        row.update(
            ray_comm_time_mean=0.0,
            profiling_precision=profiling_precision,
            measurement_type=family.value,
            cpu_overhead_source="vllm_cpu_probe",
            num_steps=len(group),
            **{CPU_OVERHEAD_PIPELINE_STAGE_COLUMN: stage_id},
        )
        if idle_edge is not None:
            row[CPU_OVERHEAD_ENGINE_IDLE_COLUMN] = idle_edge
        rows[family].append(row)
    return {
        family: validate_cpu_overhead_dataframe(pd.DataFrame(family_rows), expected_precision=profiling_precision)
        for family, family_rows in rows.items()
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpu_probe_logs", type=Path, nargs="+", action="append", required=True,
                        help="the engine logs of one serving instance, one per DP engine in DP rank order; repeat "
                             "the flag for each instance of the engine configuration, e.g. a PD prefill and a PD "
                             "decode instance")
    parser.add_argument("--engine_idle_edges_ms", type=float, nargs="+",
                        help="increasing engine-idle bucket edges in ms, the first 0; rows are keyed by bucket")
    parser.add_argument("--num_pipeline_stages", type=int, default=1,
                        help="the engine's pipeline-parallel size; each engine log is read with its stage logs "
                             "<stem>_pp<p><suffix>")
    parser.add_argument("--model_name", required=True, help="model name as Frontier's model config reports it")
    parser.add_argument("--tensor_parallel_degree", type=int, required=True)
    parser.add_argument("--profiling_precision", required=True, choices=[p.name for p in PrecisionType])
    parser.add_argument("--scheduling_mode", required=True, choices=VALID_SCHEDULING_MODES)
    parser.add_argument("--eager_output_file", type=Path, required=True)
    parser.add_argument("--kernel_only_output_file", type=Path, required=True)
    args = parser.parse_args(argv)
    edges = args.engine_idle_edges_ms
    if edges is not None and (edges[0] != 0 or any(b <= a for a, b in zip(edges, edges[1:]))):
        parser.error(f"--engine_idle_edges_ms must increase from 0, got {edges}")

    stage_steps = [step for engine_logs in args.cpu_probe_logs
                   for step in instance_stage_steps(engine_logs, args.num_pipeline_stages, edges)]
    tables = cpu_overhead_tables(
        stage_steps,
        model_name=args.model_name,
        tensor_parallel_degree=args.tensor_parallel_degree,
        profiling_precision=args.profiling_precision,
        scheduling_mode=args.scheduling_mode,
    )
    outputs = {MeasurementType.CUDA_EVENT: args.eager_output_file,
               MeasurementType.KERNEL_ONLY: args.kernel_only_output_file}
    for family, path in outputs.items():
        if family not in tables:
            logger.info("No %s tuples; %s is not written.", family.value, path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tables[family].to_csv(path, index=False)
        logger.info("Wrote %d %s rows to %s.", len(tables[family]), family.value, path)


if __name__ == "__main__":
    main()
