"""Repeat the existing attention profiler at one explicit prefill input."""

import argparse
import json
from pathlib import Path

import torch

from frontier.profiling.attention.attention_input import AttentionInput
from frontier.profiling.attention.attention_wrapper import AttentionWrapper
from frontier.profiling.attention.backends import AttentionBackend
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_config import ParallelConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = ModelConfig.from_model_name(args.model)
    wrapper = AttentionWrapper(
        model_config=model,
        parallel_config=ParallelConfig(tensor_parallel_size=args.tp, pipeline_parallel_size=1),
        max_num_blocks=args.blocks,
        max_model_len=args.max_model_len,
        block_size=16,
        attention_backend=AttentionBackend.FLASHINFER,
        dtype=torch.bfloat16,
        profile_method="cuda_event",
        output_dir=str(args.output),
    )
    inputs = AttentionInput(batch_size=1, prefill_chunk_size=args.tokens,
                            kv_cache_size=0, is_prefill=True)
    rows = [wrapper.profile(inputs) for _ in range(args.repeats)]
    (args.output / "samples.json").write_text(json.dumps({
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "measurement_type": "cuda_event",
        "rows": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
