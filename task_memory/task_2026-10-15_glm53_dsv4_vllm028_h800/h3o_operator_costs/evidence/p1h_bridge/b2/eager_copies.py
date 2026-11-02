"""Census of eager memcpy/memset activities relative to graph spans per step (scratch)."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
from frontier.profiling.attention.dsa_kernel_trace import load_trace_events, parse_annotation_name
from frontier.profiling.attention.dsv4_graph_replay_runner import CONTAINER_SCOPES
from frontier.profiling.attention.graph_capture import CaptureRecord
from frontier.profiling.attention.graph_replay_trace import attribute_replay_trace, parse_step_annotation

root = Path(sys.argv[1])
for mode in ("NONE", "PIECEWISE", "FULL_AND_PIECEWISE"):
    record = CaptureRecord.from_json(json.loads((root / mode / "raw" / "dsv4_capture_e0.json").read_text())["record"])
    for path in sorted((root / mode / "traces").glob("*.json")):
        events = load_trace_events(path)
        att = attribute_replay_trace(events, record, CONTAINER_SCOPES)
        span = {}
        for iv in att.intervals:
            if iv.kind == "graph":
                a, b = span.get(iv.step_index, (iv.start_us, iv.end_us))
                span[iv.step_index] = (min(a, iv.start_us), max(b, iv.end_us))
        steps = defaultdict(list); scopes = defaultdict(list)
        launches = defaultdict(list)
        copies = []
        for e in events:
            c = e.get("cat")
            if c == "user_annotation":
                s = parse_step_annotation(e.get("name"))
                if s is not None:
                    steps[(e.get("pid"), e.get("tid"))].append((float(e["ts"]), float(e["ts"]) + float(e["dur"]), s))
                else:
                    k = parse_annotation_name(e.get("name"))
                    if k is not None:
                        scopes[(e.get("pid"), e.get("tid"))].append((float(e["ts"]), float(e["ts"]) + float(e["dur"]), k))
            elif c in ("cuda_runtime", "cuda_driver"):
                launches[e.get("args", {}).get("correlation")].append(e)
            elif c in ("gpu_memcpy", "gpu_memset"):
                nid = e.get("args", {}).get("graph node id")
                if nid is None or int(nid) <= 0:
                    copies.append(e)
        census = Counter()
        for e in copies:
            cands = [l for l in launches.get(e["args"].get("correlation"), ()) if not str(l.get("name", "")).startswith("cudaGraphLaunch")]
            if len(cands) != 1:
                census[("launch_candidates", len(cands))] += 1; continue
            l = cands[0]; t = float(l["ts"]); th = (l.get("pid"), l.get("tid"))
            step = next((s for a, b, s in steps.get(th, ()) if a <= t < b), None)
            if step is None:
                census[("outside_step", e["cat"])] += 1; continue
            inner = [k for a, b, k in scopes.get(th, ()) if a <= t < b]
            owner = min(inner, key=lambda k: 0) if inner else None
            where = "nograph"
            if step in span:
                a, b = span[step]
                where = "before" if e["ts"] + e["dur"] <= a else "after" if e["ts"] >= b else "inside"
            census[(e["cat"], "owned" if inner else "none", where, l["name"])] += 1
        print(mode, path.stem, dict(census))
