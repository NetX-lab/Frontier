"""CPU-only workloads for the native Step attention producer."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from math import ceil
from pathlib import Path

from frontier.attention.step5 import Step5AttentionWorkload
from frontier.profiling.step5.replay import choose_graph_mode


@dataclass(frozen=True)
class Step5ProfileInput:
    global_layer_id: int
    query_lens: tuple[int, ...]
    context_lens: tuple[int, ...]
    phase: str
    graph_mode: str = "none"

    def __post_init__(self):
        if type(self.global_layer_id) is not int or self.global_layer_id < 0:
            raise ValueError("global_layer_id must be a nonnegative integer")
        if not self.query_lens or len(self.query_lens) != len(self.context_lens):
            raise ValueError("One context length is required per query")
        if any(type(q) is not int or q <= 0 for q in self.query_lens):
            raise ValueError("Query lengths must be positive integers")
        if any(type(c) is not int or c < 0 for c in self.context_lens):
            raise ValueError("Context lengths must be nonnegative integers")
        if self.phase not in ("prefill", "decode"):
            raise ValueError("This producer supports prefill and decode")
        if self.phase == "decode" and any(q != 1 for q in self.query_lens):
            raise ValueError("Decode requires one query token per request")
        if self.phase == "prefill" and max(self.query_lens) == 1:
            raise ValueError("The native one-token path is decode, not prefill")
        if self.graph_mode not in ("none", "piecewise", "full"):
            raise ValueError("graph_mode must be none, piecewise or full")
        if self.graph_mode == "full" and self.phase != "decode":
            raise ValueError("Full graph capture is decode-only")

    @property
    def num_tokens(self):
        return sum(self.query_lens)

    @property
    def batch_size(self):
        return len(self.query_lens)

    @property
    def sequence_lens(self):
        return tuple(q + c for q, c in zip(self.query_lens, self.context_lens))

    def workload(self, config, *, block_size, max_model_len):
        # Use the physical table width actually allocated by this producer.
        return Step5AttentionWorkload.from_lengths(
            config, phase=self.phase, query_lengths=self.query_lens,
            sequence_lengths=self.sequence_lens, physical_query_rows=self.num_tokens,
            block_size_tokens=block_size,
            block_table_columns=ceil(max(self.sequence_lens) / block_size),
            max_model_len_tokens=max_model_len, graph_mode=self.graph_mode,
        )


def build_profile_inputs(config, *, prefill_tokens, prefixes, decode_batches,
                          decode_contexts, cudagraph_capture_sizes=(1, 2), prefill_batches=(1,)):
    """Choose real global layer IDs from the configured schedule, not labels."""
    layers = tuple(config.layer_types.index(kind)
                   for kind in ("sliding_attention", "full_attention"))
    result = []
    for layer, tokens, prefix, batch in product(layers, prefill_tokens, prefixes, prefill_batches):
        mode = choose_graph_mode("prefill", tokens * batch, cudagraph_capture_sizes)
        result.append(Step5ProfileInput(layer, (tokens,) * batch, (prefix,) * batch, "prefill", mode))
    for layer, batch, context in product(layers, decode_batches, decode_contexts):
        graph = choose_graph_mode("decode", batch, cudagraph_capture_sizes)
        result.append(Step5ProfileInput(layer, (1,) * batch, (context,) * batch, "decode", graph))
    if len(set(result)) != len(result):
        raise ValueError("The profile grid contains duplicate cells")
    if not result:
        raise ValueError("The profile grid must contain workloads")
    return result


def read_profile_inputs(path, *, cudagraph_capture_sizes):
    """Read explicit reached shapes without forming an unrelated Cartesian grid."""
    records = json.loads(Path(path).read_text())
    result = []
    for record in records:
        values = {**record, "query_lens": tuple(record["query_lens"]),
                  "context_lens": tuple(record["context_lens"])}
        values["graph_mode"] = choose_graph_mode(
            values["phase"], sum(values["query_lens"]), cudagraph_capture_sizes)
        result.append(Step5ProfileInput(**values))
    if not result or len(set(result)) != len(result):
        raise ValueError("Explicit profile inputs must contain distinct cells")
    return result
