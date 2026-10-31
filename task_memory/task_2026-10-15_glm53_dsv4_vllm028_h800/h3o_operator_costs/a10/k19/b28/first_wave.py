"""Lanes of the first real prefill wave in an A10 run: composed time and EP scopes per lane.

    python first_wave.py <run name>
"""
import gzip
import json
import sys

S = "/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10"
wave_start = None
with gzip.open(f"{S}/{sys.argv[1]}/forwards.jsonl.gz", "rt") as stream:
    for line in stream:
        record = json.loads(line)
        if record["role"] != "prefill":
            continue
        if wave_start is None and record["real"] and record["local_tokens"] == 6144:
            wave_start = record["start_s"]
        if wave_start is not None and record["start_s"] == wave_start:
            print(record["dp"], record["real"], record["local_tokens"], round(record["composed_ms"], 3),
                  round(record["end_s"] * 1e3, 3),
                  {k: round(v, 3) for k, v in record["scope_ms"].items() if "_ep_" in k})
        elif wave_start is not None and record["start_s"] > wave_start:
            break
