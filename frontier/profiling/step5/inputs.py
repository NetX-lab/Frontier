"""CPU-only workloads for the native Step attention producer."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from math import ceil
from pathlib import Path

from frontier.attention.step5 import Step5AttentionWorkload
from frontier.attention.step5_inputs import MRv2DummyDescriptor
from frontier.profiling.step5.replay import choose_graph_mode


@dataclass(frozen=True)
class Step5ProfileInput:
    global_layer_id: int
    query_lens: tuple[int, ...]
    context_lens: tuple[int, ...]
    phase: str
    graph_mode: str = "none"
    has_attention_metadata: bool = True
    block_table_columns: int | None = None
    dummy_descriptor: MRv2DummyDescriptor | None = None

    @classmethod
    def from_mrv2_dummy(cls, global_layer_id, descriptor, *, block_table_columns):
        queries = descriptor.query_lens
        return cls(global_layer_id, queries, (0,) * len(queries),
                   "decode" if max(queries) == 1 else "prefill", descriptor.graph_mode,
                   block_table_columns=block_table_columns, dummy_descriptor=descriptor)

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
        if type(self.has_attention_metadata) is not bool:
            raise ValueError("has_attention_metadata must be boolean")
        if not self.has_attention_metadata and not (
                self.phase == "decode" and self.graph_mode == "none"
                and self.query_lens == (1,) and self.context_lens == (0,)
                and self.block_table_columns in (None, 0)):
            raise ValueError("Isolated no-metadata profiling requires one eager slot without history or block table")
        if self.block_table_columns is not None and (
                type(self.block_table_columns) is not int
                or self.block_table_columns < int(self.has_attention_metadata)):
            raise ValueError("block_table_columns must be an integral physical extent")
        if self.dummy_descriptor is not None:
            descriptor = self.dummy_descriptor
            if (not self.has_attention_metadata or self.query_lens != descriptor.query_lens
                    or any(self.context_lens) or self.graph_mode != descriptor.graph_mode
                    or self.block_table_columns is None):
                raise ValueError("MRv2 dummy input must preserve descriptor split, metadata, zero history and explicit table extent")

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
        columns = (self.block_table_columns if self.block_table_columns is not None else
                   ceil(max(self.sequence_lens) / block_size) if self.has_attention_metadata else 0)
        if columns > ceil(max_model_len / block_size):
            raise ValueError("Physical block table exceeds the producer's allocated serving capacity")
        return Step5AttentionWorkload.from_lengths(
            config, phase=self.phase, query_lengths=self.query_lens,
            sequence_lengths=self.sequence_lens, physical_query_rows=self.num_tokens,
            block_size_tokens=block_size,
            block_table_columns=columns,
            max_model_len_tokens=max_model_len, graph_mode=self.graph_mode,
            has_attention_metadata=self.has_attention_metadata,
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
        # Explicit eager inputs collect the independent GPU-active family even
        # when serving could dispatch a graph. Never label them graph timings.
        if values.get("dummy_descriptor") is not None:
            values["dummy_descriptor"] = MRv2DummyDescriptor(**values["dummy_descriptor"])
            mode = values["dummy_descriptor"].graph_mode
            if mode != "none" and sum(values["query_lens"]) not in cudagraph_capture_sizes:
                raise ValueError("Explicit MRv2 descriptor width must be a configured captured width")
        else:
            mode = (choose_graph_mode(values["phase"], sum(values["query_lens"]), cudagraph_capture_sizes)
                    if values.get("has_attention_metadata", True) and values.get("graph_mode") != "none"
                    else "none")
        if "graph_mode" in values and values["graph_mode"] != mode:
            raise ValueError("Explicit graph mode must match native dispatch, or request independent eager work")
        values["graph_mode"] = mode
        result.append(Step5ProfileInput(**values))
    if not result or len(set(result)) != len(result):
        raise ValueError("Explicit profile inputs must contain distinct cells")
    return result
