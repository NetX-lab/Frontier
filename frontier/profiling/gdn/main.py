"""CLI for profiling Qwen3.5 gated-delta-network layers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from frontier.gdn.profiling_schema import validate_gdn_profiling_dataframe
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.gdn.inputs import GDNProfileInput
from frontier.profiling.gdn.vllm_wrapper import VllmQwen35GDNWrapper
from frontier.profiling.utils import build_profile_method_output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Frontier model config name")
    parser.add_argument("--model-path", required=True, help="Local HF model directory")
    parser.add_argument("--device", default="mi355x")
    parser.add_argument("--output-dir", default="data/profiling")
    parser.add_argument("--profile-method", default="cuda_event")
    parser.add_argument("--prefill-seq-lens", nargs="+", type=int, default=[16, 128, 512])
    parser.add_argument("--prefill-batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--decode-batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--decode-context-len", type=int, default=512)
    parser.add_argument("--continuation-context-len", type=int, default=512)
    parser.add_argument("--include-continuation-prefill", action="store_true")
    parser.add_argument("--include-mixed", action="store_true")
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--profile-iterations", type=int, default=10)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    return parser.parse_args()


def build_profile_inputs(args: argparse.Namespace) -> list[GDNProfileInput]:
    prefill_batch_sizes = getattr(args, "prefill_batch_sizes", [1])
    inputs = [
        GDNProfileInput.prefill(seq_len=seq_len, batch_size=batch_size)
        for batch_size in prefill_batch_sizes
        for seq_len in args.prefill_seq_lens
    ]
    if args.include_continuation_prefill:
        inputs.extend(
            GDNProfileInput.prefill(
                seq_len=seq_len,
                batch_size=batch_size,
                context_len=args.continuation_context_len,
            )
            for batch_size in prefill_batch_sizes
            for seq_len in args.prefill_seq_lens
        )
    inputs.extend(
        GDNProfileInput.decode(
            batch_size=batch_size,
            context_len=args.decode_context_len,
        )
        for batch_size in args.decode_batch_sizes
    )
    if args.include_mixed:
        inputs.append(
            GDNProfileInput.mixed(
                decode_batch_size=max(args.decode_batch_sizes),
                decode_context_len=args.decode_context_len,
                prefill_seq_len=max(args.prefill_seq_lens),
            )
        )
    return inputs


def main() -> None:
    args = _parse_args()
    model_config = ModelConfig.from_model_name(args.model)
    rows = []
    with VllmQwen35GDNWrapper(
        frontier_model_config=model_config,
        model_path=args.model_path,
        device_name=args.device,
        profile_method=args.profile_method,
        max_model_len=args.max_model_len,
        max_batch_size=args.max_batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
    ) as wrapper:
        for profile_input in build_profile_inputs(args):
            rows.append(
                wrapper.profile(
                    profile_input,
                    warmup_iterations=args.warmup_iterations,
                    profile_iterations=args.profile_iterations,
                )
            )
        writer_rank = wrapper.rank == 0

    if not writer_rank:
        return

    dataframe = pd.DataFrame(rows)
    dataframe = (
        pd.json_normalize(dataframe["time_stats"])
        .add_prefix("time_stats.")
        .join(dataframe.drop(columns=["time_stats"]))
    )
    validate_gdn_profiling_dataframe(dataframe)
    output_path = build_profile_method_output_path(
        output_root=args.output_dir,
        profiling_type="compute",
        hardware=args.device,
        model_name=args.model,
        op_name="gdn",
        profile_method=args.profile_method,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    print(f"Wrote {len(dataframe)} GDN profiling rows to {output_path}")


if __name__ == "__main__":
    main()
