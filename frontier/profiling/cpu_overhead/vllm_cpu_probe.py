"""CPU-overhead CSVs from vLLM CPU-probe logs.

The CPU probe of Frontier's instrumented vLLM (``VLLM_FRONTIER_CPU_PROBE_LOG_PATH``)
writes one JSON record per engine step: ``time.perf_counter`` stamps around the
step's host phases, the tokens scheduled per request, and the CUDA-event device
times of the model forward (``forward_device_ms``) and of compute_logits plus
sampling (``sample_device_ms``). With one worker all stamps share one clock.

Frontier prices the model forward from its operator tables. The CPU-overhead
terms price the rest of the step period:

    schedule               step_start -> schedule_end
    prepare_inputs_e2e     schedule_end -> preprocess_end
    sampler_e2e            preprocess_end -> sample_end, minus the forward's time
    process_model_outputs  sample_end -> the next step's step_start
    ray_comm_time          0 (one worker, no Ray hop)

so the four measured terms add up to the step period minus the forward's time.
A CUDA-graph replay's forward takes forward_device_ms. An eager step's host
launches the forward's kernels one by one while the device runs them, so its
forward takes the slower of the two streams, max(forward_launch,
forward_device_ms), and its tuple also publishes

    forward_launch         preprocess_end -> forward_end

for Frontier to price the launch time the device stream cannot hide.
process_model_outputs runs to the next step's start only when the next
step shares a request with it; otherwise the engine may have waited for an
arrival before that step, so the term ends at update_end and leaves out the
engine loop gap. A PD prefill instance, whose requests run one step each, always
takes the second form.

Steps are grouped by Frontier's CPU-overhead identity (batch_size,
num_prefill_tokens, num_decode_tokens), counting a request scheduled more than
one token as prefill, which holds for logs without speculative decoding. A pure
decode tuple whose batch fits a decode CUDA-graph capture size ran as a FULL
graph replay and belongs to the kernel-only family; every other tuple is eager.

Pipeline stages. A PP>1 engine runs vLLM's batch-queue loop. Its engine log
(``cpu_probe[_dp<d>].jsonl``) stamps schedule when a batch is submitted and the
output stage's runner and sampling when the batch is popped. Each stage's
workers write ``<engine log stem>_pp<p><suffix>`` with one record per forward;
the n-th record of every stage file is the forward of the engine's n-th step
that scheduled tokens. Stamps are compared across the stage processes, so all
stages of one engine run on one host. Each interval is charged once, to the
stage that runs it, in rows keyed by ``pipeline_stage_id``:

    schedule               stage 0: step_start -> schedule_end
    prepare_inputs_e2e     each stage: its execute_start -> preprocess_end
    forward_launch         each stage: from the later of its dp_sync_end (after
                           the DP token-count all-reduce) and the upstream
                           stage's device end, to its forward_end
    sampler_e2e            last stage: from the later of its forward_end and its
                           device end, to sample_end
    process_model_outputs  last stage: sample_end -> update_end

and every other term is 0. The device end of stage p is
D_p = max(forward_start_p, D_(p-1)) + forward_device_ms_p: stage p's device work
starts after its own forward_start and after its upstream stage's device work.
The batch-queue loop pops batches in submission order, so the next record's
step_start may precede this batch's update_end, and process_model_outputs
always ends at update_end. Frontier prices each stage's forward and the waits
between stages itself, so no stage row carries them.

    python -m frontier.profiling.cpu_overhead.vllm_cpu_probe \\
        --cpu_probe_logs run/prefill/cpu_probe.jsonl run/decode/cpu_probe.jsonl \\
        --decode_cudagraph_capture_sizes 1 2 4 8 16 24 32 40 48 56 64 \\
        --model_name llama2_7b_dense_example --tensor_parallel_degree 1 \\
        --profiling_precision BF16 --scheduling_mode sync \\
        --eager_output_file cpu_overheads.csv \\
        --kernel_only_output_file cpu_overheads_kernel_only.csv

A PP>1 engine adds ``--num_pipeline_stages`` and lists each DP engine's log;
each is read with its stage logs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from frontier.config.precision_type import PrecisionType
from frontier.logger import init_logger
from frontier.profiling.cpu_overhead.schema import CPU_OVERHEAD_PIPELINE_STAGE_COLUMN, VALID_SCHEDULING_MODES
from frontier.profiling.cpu_overhead.validation import validate_cpu_overhead_dataframe
from frontier.types import MeasurementType

logger = init_logger(__name__)

StepIdentity = tuple[int, int, int]


def load_cpu_probe_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def step_identity(num_scheduled_tokens: Mapping[str, int]) -> StepIdentity:
    tokens = list(num_scheduled_tokens.values())
    return len(tokens), sum(t for t in tokens if t > 1), sum(t for t in tokens if t == 1)


def stage_log_path(engine_log: Path, pipeline_stage_id: int) -> Path:
    return engine_log.with_name(f"{engine_log.stem}_pp{pipeline_stage_id}{engine_log.suffix}")


def replays_decode_graph(identity: StepIdentity, largest_graph_batch: int) -> bool:
    batch_size, num_prefill_tokens, _ = identity
    return num_prefill_tokens == 0 and batch_size <= largest_graph_batch


def step_overhead_terms(
    records: Sequence[Mapping], *, largest_graph_batch: int,
) -> list[tuple[StepIdentity, dict[str, float]]]:
    """Return each scheduled step's identity and CPU-overhead terms in milliseconds."""

    steps = []
    for record, following in zip(records, [*records[1:], None]):
        if record.get("engine_loop") == "batch_queue":
            raise ValueError(
                f"CPU-probe step {record['step']} comes from vLLM's batch-queue loop, whose stamps "
                "the single-stage terms cannot use; read a PP>1 engine's logs with --num_pipeline_stages."
            )
        scheduled = record["num_scheduled_tokens"]
        if not scheduled:
            continue
        if record.get("forward_device_ms") is None:
            raise ValueError(
                f"CPU-probe step {record['step']} has no forward_device_ms; "
                "the log predates the probe's CUDA events"
            )
        identity = step_identity(scheduled)
        next_shares_request = following is not None and bool(
            set(following["num_scheduled_tokens"]) & set(scheduled)
        )
        outputs_end = following["step_start"] if next_shares_request else record["update_end"]
        forward_launch = (record["forward_end"] - record["preprocess_end"]) * 1e3
        graph = replays_decode_graph(identity, largest_graph_batch)
        forward_time = (record["forward_device_ms"] if graph
                        else max(forward_launch, record["forward_device_ms"]))
        terms = {
            "schedule": (record["schedule_end"] - record["step_start"]) * 1e3,
            "prepare_inputs_e2e": (record["preprocess_end"] - record["schedule_end"]) * 1e3,
            "sampler_e2e": (record["sample_end"] - record["preprocess_end"]) * 1e3 - forward_time,
            "process_model_outputs": (outputs_end - record["sample_end"]) * 1e3,
        }
        if not graph:
            terms["forward_launch"] = forward_launch
        steps.append((identity, terms))
    return steps


def stage_overhead_terms(
    engine_records: Sequence[Mapping], stage_records: Sequence[Sequence[Mapping]],
) -> list[list[tuple[StepIdentity, dict[str, float]]]]:
    """Return, per pipeline stage, each scheduled step's identity and CPU-overhead terms in milliseconds."""

    steps = [record for record in engine_records if record["num_scheduled_tokens"]]
    scheduled = [record["num_scheduled_tokens"] for record in steps]
    for stage_id, records in enumerate(stage_records):
        if [record["num_scheduled_tokens"] for record in records] != scheduled:
            raise ValueError(
                f"pipeline stage {stage_id} logged {len(records)} forwards whose scheduled tokens do not "
                f"match the engine's {len(steps)} steps that scheduled tokens"
            )
    last_stage = len(stage_records) - 1
    stage_steps = [[] for _ in stage_records]
    for n, record in enumerate(steps):
        identity = step_identity(record["num_scheduled_tokens"])
        device_end = -math.inf
        for stage_id, records in enumerate(stage_records):
            stage = records[n]
            launch_start = max(stage["dp_sync_end"], device_end)
            device_end = max(stage["forward_start"], device_end) + stage["forward_device_ms"] * 1e-3
            last = stage_id == last_stage
            stage_steps[stage_id].append((identity, {
                "schedule": (record["schedule_end"] - record["step_start"]) * 1e3 if stage_id == 0 else 0.0,
                "prepare_inputs_e2e": (stage["preprocess_end"] - stage["execute_start"]) * 1e3,
                "sampler_e2e": (record["sample_end"] - max(stage["forward_end"], device_end)) * 1e3 if last else 0.0,
                "process_model_outputs": (record["update_end"] - record["sample_end"]) * 1e3 if last else 0.0,
                "forward_launch": (stage["forward_end"] - launch_start) * 1e3,
            }))
    return stage_steps


def cpu_overhead_tables(
    steps: Sequence[tuple[StepIdentity, Mapping[str, float]]],
    *,
    decode_capture_sizes: Sequence[int],
    model_name: str,
    tensor_parallel_degree: int,
    profiling_precision: str,
    scheduling_mode: str,
) -> dict[MeasurementType, pd.DataFrame]:
    """Aggregate step terms into one validated CPU-overhead table per measurement family."""

    groups = defaultdict(list)
    for identity, terms in steps:
        groups[identity].append(terms)
    largest_graph_batch = max(decode_capture_sizes, default=0)
    rows = defaultdict(list)
    for identity, group in sorted(groups.items()):
        batch_size, num_prefill_tokens, num_decode_tokens = identity
        family = (MeasurementType.KERNEL_ONLY if replays_decode_graph(identity, largest_graph_batch)
                  else MeasurementType.CUDA_EVENT)
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
        )
        rows[family].append(row)
    return {
        family: validate_cpu_overhead_dataframe(pd.DataFrame(family_rows), expected_precision=profiling_precision)
        for family, family_rows in rows.items()
    }


def stage_cpu_overhead_tables(
    stage_steps: Sequence[Sequence[tuple[StepIdentity, Mapping[str, float]]]],
    *,
    profiling_precision: str,
    **table_identity,
) -> dict[MeasurementType, pd.DataFrame]:
    """Aggregate each pipeline stage's step terms into one validated table per family, keyed by stage."""

    stage_tables = defaultdict(list)
    for stage_id, steps in enumerate(stage_steps):
        tables = cpu_overhead_tables(steps, profiling_precision=profiling_precision, **table_identity)
        for family, table in tables.items():
            stage_tables[family].append(table.assign(**{CPU_OVERHEAD_PIPELINE_STAGE_COLUMN: stage_id}))
    return {
        family: validate_cpu_overhead_dataframe(pd.concat(tables, ignore_index=True), expected_precision=profiling_precision)
        for family, tables in stage_tables.items()
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpu_probe_logs", type=Path, nargs="+", required=True,
                        help="CPU-probe logs of one engine configuration, e.g. both PD roles or each DP engine")
    parser.add_argument("--num_pipeline_stages", type=int, default=1,
                        help="the engine's pipeline-parallel size; above 1 each engine log is read with its "
                             "stage logs <stem>_pp<p><suffix> and rows are keyed by pipeline_stage_id")
    parser.add_argument("--decode_cudagraph_capture_sizes", type=int, nargs="*", required=True,
                        help="the engine's decode CUDA-graph capture sizes; none for an eager engine")
    parser.add_argument("--model_name", required=True, help="model name as Frontier's model config reports it")
    parser.add_argument("--tensor_parallel_degree", type=int, required=True)
    parser.add_argument("--profiling_precision", required=True, choices=[p.name for p in PrecisionType])
    parser.add_argument("--scheduling_mode", required=True, choices=VALID_SCHEDULING_MODES)
    parser.add_argument("--eager_output_file", type=Path, required=True)
    parser.add_argument("--kernel_only_output_file", type=Path, required=True)
    args = parser.parse_args(argv)

    table_identity = dict(
        decode_capture_sizes=args.decode_cudagraph_capture_sizes,
        model_name=args.model_name,
        tensor_parallel_degree=args.tensor_parallel_degree,
        profiling_precision=args.profiling_precision,
        scheduling_mode=args.scheduling_mode,
    )
    if args.num_pipeline_stages == 1:
        largest_graph_batch = max(args.decode_cudagraph_capture_sizes, default=0)
        steps = [step for log in args.cpu_probe_logs
                 for step in step_overhead_terms(load_cpu_probe_log(log), largest_graph_batch=largest_graph_batch)]
        tables = cpu_overhead_tables(steps, **table_identity)
    else:
        stage_steps = [[] for _ in range(args.num_pipeline_stages)]
        for log in args.cpu_probe_logs:
            stage_logs = [load_cpu_probe_log(stage_log_path(log, p)) for p in range(args.num_pipeline_stages)]
            for steps, terms in zip(stage_steps, stage_overhead_terms(load_cpu_probe_log(log), stage_logs)):
                steps.extend(terms)
        tables = stage_cpu_overhead_tables(stage_steps, **table_identity)
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
