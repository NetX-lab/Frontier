"""Command line entry for DeepSeek-V4 attention profiling.

The subcommands match ``dsa_main`` and share the plan arguments, so the plan a
GPU run measures is the plan its dry run printed:

* ``plan``: CPU-only dry run; prints the plan summary and optionally writes
  the full plan as JSON.
* ``run``: measures every case of one engine on the visible GPU and writes raw
  JSON-lines records (GPU worker with vLLM).
* ``aggregate``: CPU-only; turns the raw records of all engines into the
  canonical ``attention.csv`` (CUDA events) and ``attention_kernel_only.csv``
  (kernel-only) tables with runtime identity columns.

Example dry run for DeepSeek-V4-Flash on H800. ``--max_kv_tokens_per_lane``
is the native engine's KV block count times the compressed-group block size
(58212 x 256), the same convention as the GLM-5.3 producer:

    python -m frontier.profiling.attention.dsv4_attention_main plan \\
        --model_name deepseek-ai/DeepSeek-V4-Flash \\
        --model_config_path data/config/models/deepseek-ai__DeepSeek-V4-Flash.json \\
        --target_tensor_parallel_size 8 --block_size 256 --kv_cache_dtype fp8 \\
        --max_context_len 33024 --max_kv_tokens_per_lane 14902272 \\
        --max_num_seqs 1024 --chunk_budgets 256 512 1024 2048 4096 8192
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from frontier.profiling.attention.dsa_main import (
    _PROFILE_METHOD_BY_MEASUREMENT,
    GRAPH_MODE_EAGER,
    _add_plan_arguments,
    _engine_identity_fields,
)
from frontier.profiling.attention.dsa_plan import DsaMeasurementLimits, load_hf_config
from frontier.profiling.attention.dsa_rows import read_records, write_rows_csv
from frontier.profiling.attention.dsv4_attention_plan import (
    Dsv4ProfilePlan,
    build_dsv4_profile_plan,
)
from frontier.profiling.attention.dsv4_attention_rows import (
    DSV4_ROW_SCHEMA_VERSION,
    build_dsv4_rows,
)
from frontier.profiling.common.runtime_identity import RuntimeIdentity, file_sha256

_RAW_FILE_PATTERN = "dsv4_raw_e*.jsonl"


def _add_dsv4_plan_arguments(parser: argparse.ArgumentParser) -> None:
    _add_plan_arguments(parser)
    parser.add_argument(
        "--query_one_context_min", type=int, default=0,
        help="Inclusive cached-context lower bound for --query_one_context_max. "
             "A positive bound uses an unmeasured local seed, not a NIXL first-tail.",
    )
    parser.add_argument(
        "--prefill_prompt_len", type=int,
        help="Exact ordinary prefill prompt length instead of the context grid.",
    )
    parser.add_argument(
        "--kv_cache_dtype",
        required=True,
        help="KV-cache dtype of the target deployment (vLLM --kv-cache-dtype).",
    )
    # By default the reduced engine keeps the fewest leading layers that hold
    # two layers of every attention variant.
    parser.set_defaults(num_layers=None)


def build_plan(args: argparse.Namespace) -> Dsv4ProfilePlan:
    limits = DsaMeasurementLimits(
        max_context_len=args.max_context_len,
        max_kv_tokens_per_lane=args.max_kv_tokens_per_lane,
        max_num_seqs=args.max_num_seqs,
        chunk_budgets=tuple(args.chunk_budgets),
        block_size=args.block_size,
        decode_warmup_steps=args.decode_warmup_steps,
        decode_measured_steps=args.decode_measured_steps,
        prefill_warmup_repeats=args.prefill_warmup_repeats,
        prefill_measured_repeats=args.prefill_measured_repeats,
    )
    return build_dsv4_profile_plan(
        model_name=args.model_name,
        hf_config=load_hf_config(args.model_config_path),
        target_tensor_parallel_size=args.target_tensor_parallel_size,
        num_layers=args.num_layers,
        limits=limits,
        kv_cache_dtype=args.kv_cache_dtype,
        query_one_context_min=getattr(args, "query_one_context_min", 0),
        query_one_context_max=getattr(args, "query_one_context_max", None),
        prefill_prompt_len=getattr(args, "prefill_prompt_len", None),
    )


def _plan_command(args: argparse.Namespace) -> None:
    plan = build_plan(args)
    print(json.dumps(plan.summary(), indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(plan.to_json(), indent=2) + "\n", encoding="utf-8")
        print(f"[dsv4] wrote plan {output}")


def _run_command(args: argparse.Namespace) -> None:
    from frontier.profiling.attention.dsv4_attention_runner import run_engine

    engine_config = Path(args.model_dir) / "config.json"
    if file_sha256(engine_config) != file_sha256(args.model_config_path):
        raise ValueError(
            f"{engine_config} differs from --model_config_path {args.model_config_path}; "
            "the engine must load the configuration the plan was built from"
        )
    plan = build_plan(args)
    run_engine(
        plan=plan,
        engine_index=args.engine_index,
        model_dir=args.model_dir,
        vocab_size=load_hf_config(args.model_config_path)["vocab_size"],
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        expected_attention_backend=args.expected_attention_backend,
        raw_path=Path(args.raw_dir) / f"dsv4_raw_e{args.engine_index}.jsonl",
        trace_dir=Path(args.trace_dir),
        keep_traces=args.keep_traces,
        case_ids=set(args.case_ids) if args.case_ids else None,
    )


def _aggregate_command(args: argparse.Namespace) -> None:
    from frontier.profiling.common.model_config import ModelConfig
    from frontier.profiling.utils import build_profile_method_output_path

    plan = build_plan(args)
    raw_paths = sorted(Path(args.raw_dir).glob(_RAW_FILE_PATTERN))
    if not raw_paths:
        raise ValueError(f"No {_RAW_FILE_PATTERN} files under {args.raw_dir}")
    records = read_records(raw_paths)
    engine_indices = {r["engine_index"] for r in records if r["kind"] == "engine"}
    missing = {engine.engine_index for engine in plan.engines} - engine_indices
    if missing and not args.allow_partial_engines:
        raise ValueError(f"Raw records lack engines {sorted(missing)}")
    identity = RuntimeIdentity(
        **_engine_identity_fields(records),
        source_revision=args.source_revision,
        model_config_revision=args.model_config_revision,
        model_config_sha256=file_sha256(args.model_config_path),
        schema_version=DSV4_ROW_SCHEMA_VERSION,
        quant_signature=ModelConfig.from_model_name(args.model_name).get_quant_signature(),
        parallel_groups={
            "target_tensor_parallel": plan.shape.target_tensor_parallel_size,
            "engine_tensor_parallel": 1,
        },
        graph_mode=GRAPH_MODE_EAGER,
    )
    rows = build_dsv4_rows(plan, records)
    written = {}
    for measurement, measurement_rows in rows.items():
        path = build_profile_method_output_path(
            output_root=args.output_root,
            profiling_type="compute",
            hardware=args.device,
            model_name=args.model_name,
            op_name="attention",
            profile_method=_PROFILE_METHOD_BY_MEASUREMENT[measurement],
        )
        write_rows_csv(measurement_rows, identity, path)
        written[measurement] = {"path": str(path), "rows": len(measurement_rows)}
        print(f"[dsv4] wrote {len(measurement_rows)} {measurement} rows to {path}")
    summary_path = Path(args.raw_dir) / "dsv4_aggregate_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "plan": plan.summary(),
                "identity": identity.to_json(),
                "missing_engines": sorted(missing),
                "outputs": written,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="CPU-only dry run of the measurement plan")
    _add_dsv4_plan_arguments(plan)
    plan.add_argument("--output", help="Write the full plan JSON here.")
    plan.set_defaults(handler=_plan_command)

    run = commands.add_parser("run", help="Measure one engine's cases on the visible GPU")
    _add_dsv4_plan_arguments(run)
    run.add_argument("--engine_index", type=int, required=True)
    run.add_argument("--model_dir", required=True, help="Model directory with config.json.")
    run.add_argument("--raw_dir", required=True)
    run.add_argument("--trace_dir", required=True, help="Worker-local profiler trace directory.")
    run.add_argument("--expected_attention_backend", required=True)
    run.add_argument("--gpu_memory_utilization", type=float, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--keep_traces", action="store_true")
    run.add_argument("--case_ids", nargs="+", help="Measure only these planned cases.")
    run.set_defaults(handler=_run_command)

    aggregate = commands.add_parser("aggregate", help="Write the profiling CSVs")
    _add_dsv4_plan_arguments(aggregate)
    aggregate.add_argument("--raw_dir", required=True)
    aggregate.add_argument("--output_root", required=True, help="Usually data/profiling.")
    aggregate.add_argument("--device", required=True, help="Device directory, e.g. h800.")
    aggregate.add_argument("--source_revision", required=True)
    aggregate.add_argument("--model_config_revision", required=True)
    aggregate.add_argument(
        "--allow_partial_engines",
        action="store_true",
        help="Aggregate a smoke run that measured only some engines.",
    )
    aggregate.set_defaults(handler=_aggregate_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main(sys.argv[1:])
