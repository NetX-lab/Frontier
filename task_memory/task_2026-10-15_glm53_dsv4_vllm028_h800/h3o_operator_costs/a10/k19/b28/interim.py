"""Interim comparison of completed requests between two A10 runs (not a gate result).

    python interim.py <case> <base run name> <candidate run name>

For the request IDs completed in both runs, prints the mean prefill-side time
(arrival to prefill end) and the mean time from arrival to the first decode token.
"""
import json
import statistics
import sys

S = "/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10"
case, base, cand = sys.argv[1:4]


def completions(name):
    path = f"{S}/{name}/metrics/deepseek_ai_deepseek_v4_flash/online_serving/a10_{case}_operator_path/metrics_ground_truth.jsonl"
    out = {}
    for line in open(path):
        record = json.loads(line)
        if record["event_type"] == "request_completion":
            out[record["request_id"]] = record
    return out


b, c = completions(base), completions(cand)
ids = sorted(set(b) & set(c))
for label, sel in (("all", ids), ("without request 0", [i for i in ids if i != 0])):
    if not sel:
        continue
    print(case, label, "n", len(sel))
    for field in ("ttft_ms", "decode_first_token_completed_at_ms", "tpot_ms", "request_e2e_time_ms"):
        vb = [b[i][field] - (b[i]["arrived_at_ms"] if field == "decode_first_token_completed_at_ms" else 0) for i in sel]
        vc = [c[i][field] - (c[i]["arrived_at_ms"] if field == "decode_first_token_completed_at_ms" else 0) for i in sel]
        print(f"  {field:36s} base {statistics.mean(vb):9.3f} candidate {statistics.mean(vc):9.3f} "
              f"difference {statistics.mean(vc) - statistics.mean(vb):+8.3f}")
