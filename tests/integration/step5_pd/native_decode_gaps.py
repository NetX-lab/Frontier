"""Summarize the decode token gaps of a native Step P/D workload replay. Diagnostic only.

The native step times read here explain where Frontier and vLLM differ; they are never a
Frontier prediction input (D31.10).

For each repeat of the formal requests the script reports:
- stalls: token gaps above `--stall-ms` between two tokens of one request, their share of all
  decode time, split into the growth phase (before the first formal request finishes) and the
  rest;
- first-token delay: the decode-side first token minus the proxy's decode submit, split the
  same way;
- the median of the other gaps for each bucket of decoding requests per DP engine, as a proxy
  of the steady decode step time at that batch size.

Usage: native_decode_gaps.py <replay_dir> --dp 2 [--stall-ms 150] [--output summary.json]
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
import statistics


def load(replay_dir: Path):
    def rows(name):
        with (replay_dir / name).open() as handle:
            return [json.loads(line) for line in handle]
    proxy = {row["request_id"]: row for row in rows("proxy_requests.jsonl")}
    return [row for row in rows("client_requests.jsonl") if row["role"] == "formal"], proxy


def median(values):
    return statistics.median(values) if values else None


def summarize_repeat(requests, proxy, *, dp: int, stall_s: float):
    tokens = {row["request_id"]: [delta["monotonic"] for delta in row["token_id_deltas"]]
              for row in requests}
    growth_end = min(times[-1] for times in tokens.values())
    starts = sorted(times[0] for times in tokens.values())
    ends = sorted(times[-1] for times in tokens.values())

    def decoding(at):
        return bisect.bisect_right(starts, at) - bisect.bisect_right(ends, at)

    phases = {"growth": {"gap_s": 0.0, "stall_s": 0.0, "stalls": 0, "first_token_ms": []},
              "after": {"gap_s": 0.0, "stall_s": 0.0, "stalls": 0, "first_token_ms": []}}
    steady_gaps: dict[int, list[float]] = {}
    for row in requests:
        times = tokens[row["request_id"]]
        submit = proxy[row["request_id"]]["http_capture"]["attempts"]["decode"]["submit_monotonic"]
        phase = phases["growth" if submit < growth_end else "after"]
        phase["first_token_ms"].append(1e3 * (times[0] - submit))
        for before, after in zip(times, times[1:]):
            gap = after - before
            phase = phases["growth" if after <= growth_end else "after"]
            phase["gap_s"] += gap
            if gap > stall_s:
                phase["stall_s"] += gap
                phase["stalls"] += 1
                continue
            bucket = 5 * round(decoding(after) / dp / 5)
            steady_gaps.setdefault(bucket, []).append(1e3 * gap)
    gap_total = sum(phase["gap_s"] for phase in phases.values())
    stall_total = sum(phase["stall_s"] for phase in phases.values())
    return {
        "requests": len(requests),
        "growth_phase_s": growth_end - min(starts),
        "stall_share_of_decode_time": stall_total / gap_total,
        "phases": {
            name: {
                "stalls": phase["stalls"],
                "stall_share": phase["stall_s"] / phase["gap_s"] if phase["gap_s"] else None,
                "stall_s": phase["stall_s"],
                "requests_admitted": len(phase["first_token_ms"]),
                "first_token_after_decode_submit_ms_p50": median(phase["first_token_ms"]),
            }
            for name, phase in phases.items()
        },
        "non_stall_gap_ms_p50_by_requests_per_engine": {
            str(bucket): {"p50": statistics.median(gaps), "gaps": len(gaps)}
            for bucket, gaps in sorted(steady_gaps.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("--dp", type=int, required=True, help="decode DP engines of the run")
    parser.add_argument("--stall-ms", type=float, default=150.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    requests, proxy = load(args.replay_dir)
    summary = {
        "replay_dir": str(args.replay_dir), "dp": args.dp, "stall_ms": args.stall_ms,
        "repeats": {
            str(repeat): summarize_repeat([row for row in requests if row["repeat"] == repeat],
                                          proxy, dp=args.dp, stall_s=args.stall_ms * 1e-3)
            for repeat in sorted({row["repeat"] for row in requests})
        },
    }
    text = json.dumps(summary, indent=1)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
