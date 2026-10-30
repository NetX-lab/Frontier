"""Print per-repeat relative errors of an A10 gate reading. Usage: per_repeat.py <reading.json>"""
import json, sys
r = json.load(open(sys.argv[1]))
for side, rep in r["repetitions"].items():
    cells = " ".join(f"{m['metric']}: native {m['vllm']:.3f} rel {m['relative_error']*100:.2f}%" for m in rep["metrics"])
    print(side.split("-")[-1], cells)
