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
    sampler_e2e            preprocess_end -> sample_end, minus forward_device_ms
    process_model_outputs  sample_end -> the next step's step_start
    ray_comm_time          0 (one worker, no Ray hop)

so the four measured terms add up to the step period minus the forward's device
time. process_model_outputs runs to the next step's start only when the next
step shares a request with it; otherwise the engine may have waited for an
arrival before that step, so the term ends at update_end and leaves out the
engine loop gap. A PD prefill instance, whose requests run one step each, always
takes the second form.

Steps are grouped by Frontier's CPU-overhead identity (batch_size,
num_prefill_tokens, num_decode_tokens), counting a request scheduled more than
one token as prefill, which holds for logs without speculative decoding. A pure
decode tuple whose batch fits a decode CUDA-graph capture size ran as a FULL
graph replay and belongs to the kernel-only family; every other tuple is eager.

    python -m frontier.profiling.cpu_overhead.vllm_cpu_probe \\
        --cpu_probe_logs run/prefill/cpu_probe.jsonl run/decode/cpu_probe.jsonl \\
        --decode_cudagraph_capture_sizes 1 2 4 8 16 24 32 40 48 56 64 \\
        --model_name llama2_7b_dense_example --tensor_parallel_degree 1 \\
        --profiling_precision BF16 --scheduling_mode sync \\
        --eager_output_file cpu_overheads.csv \\
        --kernel_only_output_file cpu_overheads_kernel_only.csv
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from frontier.config.precision_type import PrecisionType
from frontier.logger import init_logger
from frontier.profiling.cpu_overhead.schema import VALID_SCHEDULING_MODES
from frontier.profiling.cpu_overhead.validation import validate_cpu_overhead_dataframe
from frontier.types import MeasurementType

logger = init_logger(__name__)

TERMS = ("schedule", "prepare_inputs_e2e", "sampler_e2e", "process_model_outputs")

StepIdentity = tuple[int, int, int]


def load_cpu_probe_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def step_overhead_terms(records: Sequence[Mapping]) -> list[tuple[StepIdentity, dict[str, float]]]:
    """Return each scheduled step's identity and CPU-overhead terms in milliseconds."""

    steps = []
    for record, following in zip(records, [*records[1:], None]):
        scheduled = record["num_scheduled_tokens"]
        if not scheduled:
            continue
        if record.get("forward_device_ms") is None:
            raise ValueError(
                f"CPU-probe step {record['step']} has no forward_device_ms; "
                "the log predates the probe's CUDA events"
            )
        tokens = list(scheduled.values())
        identity = (len(tokens), sum(t for t in tokens if t > 1), sum(t for t in tokens if t == 1))
        next_shares_request = following is not None and bool(
            set(following["num_scheduled_tokens"]) & set(scheduled)
        )
        outputs_end = following["step_start"] if next_shares_request else record["update_end"]
        steps.append((identity, {
            "schedule": (record["schedule_end"] - record["step_start"]) * 1e3,
            "prepare_inputs_e2e": (record["preprocess_end"] - record["schedule_end"]) * 1e3,
            "sampler_e2e": (record["sample_end"] - record["preprocess_end"]) * 1e3
                           - record["forward_device_ms"],
            "process_model_outputs": (outputs_end - record["sample_end"]) * 1e3,
        }))
    return steps


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
        family = (MeasurementType.KERNEL_ONLY
                  if num_prefill_tokens == 0 and batch_size <= largest_graph_batch
                  else MeasurementType.CUDA_EVENT)
        row = {
            "model_name": model_name,
            "batch_size": batch_size,
            "tensor_parallel_degree": tensor_parallel_degree,
            "num_prefill_tokens": num_prefill_tokens,
            "num_decode_tokens": num_decode_tokens,
            "scheduling_mode": scheduling_mode,
        }
        for term in TERMS:
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpu_probe_logs", type=Path, nargs="+", required=True,
                        help="CPU-probe logs of one engine configuration, e.g. both PD roles")
    parser.add_argument("--decode_cudagraph_capture_sizes", type=int, nargs="*", required=True,
                        help="the engine's decode CUDA-graph capture sizes; none for an eager engine")
    parser.add_argument("--model_name", required=True, help="model name as Frontier's model config reports it")
    parser.add_argument("--tensor_parallel_degree", type=int, required=True)
    parser.add_argument("--profiling_precision", required=True, choices=[p.name for p in PrecisionType])
    parser.add_argument("--scheduling_mode", required=True, choices=VALID_SCHEDULING_MODES)
    parser.add_argument("--eager_output_file", type=Path, required=True)
    parser.add_argument("--kernel_only_output_file", type=Path, required=True)
    args = parser.parse_args(argv)

    steps = [step for log in args.cpu_probe_logs for step in step_overhead_terms(load_cpu_probe_log(log))]
    tables = cpu_overhead_tables(
        steps,
        decode_capture_sizes=args.decode_cudagraph_capture_sizes,
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
