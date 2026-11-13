"""Check that every reachable Frontier attention query of the fixed workload lies in the planned ranges.

Usage: PYTHONPATH=<repo root> python h200_profile_coverage.py <repo root> [max prompt tails per step, default 3].
Each lane state of the 6144/1024 workload is mapped to its attention features and compared with the
range of the profiled cells of the same category; DSA context columns may exceed it by one region.
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

from frontier.attention.step5 import Step5AttentionWorkload
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.step5.inputs import read_profile_inputs

ROOT = Path(sys.argv[1])
CAPTURE = [1, 2, 4] + list(range(8, 257, 8))
config = ModelConfig.from_model_name("stepfun-ai/Step-5-Preview")
DSA = {3}
SLACK_COLUMNS = ("min_ctx", "max_ctx", "mean_ctx", "max_seq")
LO, HI, TAIL = 6144, 7166, 6128
MAX_TAILS = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def features(phase, mode, kind, queries, contexts, rows, prefill, decode):
    w = Step5AttentionWorkload.from_lengths(
        config, phase=phase, query_lengths=tuple(queries),
        sequence_lengths=tuple(q + c for q, c in zip(queries, contexts)), physical_query_rows=rows,
        block_size_tokens=16, block_table_columns=448, max_model_len_tokens=7168, graph_mode=mode,
        has_attention_metadata=True)
    mean_q = sum(queries) / len(queries)
    return (phase, mode, kind), {
        "batch": len(queries), "tokens": w.valid_tokens, "prefill": prefill, "decode": decode,
        "padded": w.padded_tokens, "score_regions": w.score_regions,
        "min_q": min(queries), "max_q": max(queries),
        "cv": math.sqrt(sum((q - mean_q) ** 2 for q in queries) / len(queries)) / mean_q,
        "min_ctx": min(contexts), "max_ctx": max(contexts), "mean_ctx": sum(contexts) / len(contexts),
        "max_seq": max(q + c for q, c in zip(queries, contexts))}


def capture(tokens):
    return next(size for size in CAPTURE if size >= tokens)


def planned(path):
    ranges = defaultdict(lambda: defaultdict(lambda: [math.inf, -math.inf]))
    for cell in read_profile_inputs(path, cudagraph_capture_sizes=CAPTURE):
        w = cell.workload(config, block_size=16, max_model_len=7168)
        kind = "dummy" if cell.dummy_descriptor else "real"
        category, values = features(cell.phase, cell.graph_mode, kind, cell.query_lens, cell.context_lens,
                                    cell.physical_query_rows, cell.num_prefill_tokens, cell.num_decode_tokens)
        for name, value in values.items():
            low_high = ranges[(cell.global_layer_id, category)][name]
            low_high[0] = min(low_high[0], value)
            low_high[1] = max(low_high[1], value)
    return ranges


def reachable_decode():
    widths = lambda tokens, top: [s for s in CAPTURE if capture(tokens) <= s <= top]
    for n in range(1, 129):
        for c in (LO, HI):
            for W in widths(n, 128):
                yield features("decode", "full", "real", [1] * n, [c] * n, W, 0, n)
            for W in widths(max(n, 16), 256):
                yield features("decode", "piecewise", "real", [1] * n, [c] * n, W, 0, n)
    for k in range(1, MAX_TAILS + 1):
        for W in widths(16 * k, 256):
            yield features("prefill", "piecewise", "real", [16] * k, [TAIL] * k, W, 16 * k, 0)
        for n in range(1, 129 - k):
            if n + 16 * k > 256:
                continue
            for c in (LO, HI):
                for W in widths(n + 16 * k, 256):
                    yield features("mixed", "piecewise", "real", [1] * n + [16] * k,
                                   [c] * n + [TAIL] * k, W, 16 * k, n)
    for W in (s for s in CAPTURE if s <= 128):
        yield features("decode", "full", "dummy", [1] * W, [0] * W, W, 0, W)
    for W in (s for s in CAPTURE if s >= 16):
        yield features("prefill", "piecewise", "dummy", [W], [0], W, W, 0)


def reachable_prefill():
    for tokens in (6128, 6144):
        yield features("prefill", "none", "real", [tokens], [0], tokens, tokens, 0)
    yield features("decode", "none", "dummy", [1], [0], 1, 0, 1)


failures = defaultdict(int)
examples = {}
checked = 0
for role, generator in (("prefill", reachable_prefill), ("decode", reachable_decode)):
    ranges = planned(ROOT / f"frontier/profiling/step5/h200_{role}_inputs.json")
    for category, values in generator():
        for layer in (0, 3):
            checked += 1
            measured = ranges.get((layer, category))
            if measured is None:
                failures[(role, layer, category, "category")] += 1
                continue
            for name, value in values.items():
                low, high = measured[name]
                if layer in DSA and name in SLACK_COLUMNS:
                    high += 7
                if not low - 1e-9 <= value <= high + 1e-9:
                    key = (role, layer, category, name)
                    failures[key] += 1
                    examples.setdefault(key, (value, low, high))
print("checked", checked)
for key, count in sorted(failures.items(), key=str):
    print("OUTSIDE", key, count, examples.get(key))
print("PASS" if not failures else "FAIL")
