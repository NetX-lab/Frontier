"""Write the H200 Step-5 role attention inputs of h200_profile_plan.json.

Usage: python h200_profile_inputs.py <repo root>. Check the result with h200_profile_coverage.py.
"""
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
CAPTURE = [1, 2, 4] + list(range(8, 257, 8))
EVIDENCE = "MRv2 gpu/dp_utils.py:155-210; gpu/model_runner.py execute_dummy_batch InputBatch"
COLUMNS = 448
TAIL_CONTEXT = 6128
LAYERS = {0: 7166, 3: 7160}  # highest decode context; DSA timing needs region-aligned contexts
LO, MID = 6144, 6656


def cell(layer, queries, contexts, phase, mode=None, rows=None, dummy=None):
    record = {"global_layer_id": layer, "query_lens": queries, "context_lens": contexts,
              "phase": phase, "block_table_columns": COLUMNS}
    if mode is not None:
        record["graph_mode"] = mode
    if rows is not None:
        record["physical_query_rows"] = rows
    if dummy is not None:
        record["dummy_descriptor"] = dummy
    return record


def prefill_cells():
    result = []
    for layer in LAYERS:
        result.append(cell(layer, [6128], [0], "prefill"))
        result.append(cell(layer, [6144], [0], "prefill"))
        result.append(cell(layer, [1], [0], "decode", dummy={
            "num_tokens": 1, "num_reqs": 1, "graph_mode": "none", "max_query_len": None,
            "evidence": EVIDENCE, "uniform_token_count": None}))
    return result


def decode_cells():
    result = []
    for layer, hi in LAYERS.items():
        ctx = {"lo": LO, "mid": MID, "hi": hi}
        full = [(n, None, c) for n in (1, 2, 4, 8, 16, 32, 64, 128) for c in ("lo", "hi")]
        full += [(n, None, "mid") for n in (8, 32, 128)]
        full += [(3, None, "lo"), (50, None, "hi"), (100, None, "mid")]
        full += [(1, 128, "lo"), (1, 128, "hi"), (1, 32, "mid"), (16, 128, "hi"), (64, 128, "mid")]
        for n, width, c in full:
            result.append(cell(layer, [1] * n, [ctx[c]] * n, "decode", rows=width))
        piecewise = [(1, 16, "lo"), (1, 24, "lo"), (1, 256, "hi"), (8, 24, "hi"), (16, 32, "mid"),
                     (32, 144, "lo"), (64, 80, "mid"), (128, 128, "lo"), (128, 256, "hi")]
        for n, width, c in piecewise:
            result.append(cell(layer, [1] * n, [ctx[c]] * n, "decode", "piecewise", width))
        mixed = [(1, 1, "lo", 24), (1, 1, "hi", 144), (1, 2, "lo", 40), (1, 3, "lo", 56),
                 (1, 3, "hi", 56), (16, 1, "mid", 32), (16, 1, "hi", 256), (32, 1, "lo", 48),
                 (64, 2, "hi", 96), (127, 1, "lo", 144), (127, 1, "hi", 144),
                 (126, 2, "hi", 160), (125, 3, "hi", 176), (1, 1, "mid", 256),
                 (8, 1, "hi", 24), (100, 1, "mid", 120)]
        for n, k, c, width in mixed:
            result.append(cell(layer, [1] * n + [16] * k, [ctx[c]] * n + [TAIL_CONTEXT] * k,
                               "mixed", "piecewise", width))
        for k, width in ((1, 16), (1, 128), (1, 256), (2, 32), (2, 144), (3, 48)):
            result.append(cell(layer, [16] * k, [TAIL_CONTEXT] * k, "prefill", "piecewise", width))
        for width in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128):
            result.append(cell(layer, [1] * width, [0] * width, "decode", dummy={
                "num_tokens": width, "num_reqs": width, "graph_mode": "full", "max_query_len": None,
                "evidence": EVIDENCE, "uniform_token_count": 1}))
        for width in (16, 24, 32, 48, 64, 96, 128, 144, 160, 192, 256):
            result.append(cell(layer, [width], [0], "prefill", dummy={
                "num_tokens": width, "num_reqs": None, "graph_mode": "piecewise", "max_query_len": None,
                "evidence": EVIDENCE, "uniform_token_count": None}))
    return result


def dump(records, path):
    lines = ",\n".join("  " + json.dumps(r) for r in records)
    path.write_text("[\n" + lines + "\n]\n")


dump(prefill_cells(), ROOT / "frontier/profiling/step5/h200_prefill_inputs.json")
dump(decode_cells(), ROOT / "frontier/profiling/step5/h200_decode_inputs.json")
print(len(prefill_cells()), len(decode_cells()))
