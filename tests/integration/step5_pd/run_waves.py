"""Replay the waves of one native P/D run through pd_replay and read both roles' /metrics.

The trace directory holds `waves.json` and one pd_replay trace directory per wave. The
driver reads `/metrics` of the prefill and decode API heads before the first wave, after
each wave, and once more `release_wait_s` after the last wave, when P's lease on the
transferred blocks has ended. Each wave is one `pd_replay.main` call with that wave's trace
and output directories and the remaining pd_replay arguments, so each wave starts and stops
its own proxy against the externally managed P/D endpoints.

Outputs in `--output-dir`: one pd_replay output directory per wave, `metrics/<point>_<role>.prom`
with every raw page read, and `native_transfer_metrics.json` with the per-wave changes of the
NIXL, prompt-source and request series of both roles. The exit status is the gating wave's
pd_replay status; the other waves are recorded only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

HARNESS_DIR = Path(__file__).resolve().parents[2] / "comparison" / "calibration"
sys.path.insert(0, str(HARNESS_DIR))
import pd_replay  # noqa: E402
from serving_metrics import scrape, transfer_delta  # noqa: E402

ROLES = ("prefill", "decode")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    args, replay_args = parser.parse_known_args(argv)
    plan = json.loads((args.trace_dir / "waves.json").read_text())
    urls = {"prefill": args.prefill_url, "decode": args.decode_url}
    pages = args.output_dir / "metrics"
    pages.mkdir(parents=True, exist_ok=True)
    report: dict = {"waves": plan["waves"], "gating_wave": plan["gating_wave"],
                    "release_wait_s": plan["release_wait_s"], "points": {}, "statuses": {},
                    "errors": {}, "changes": {}}
    samples: dict = {}

    def read(point: str) -> None:
        samples[point] = {}
        report["points"][point] = {}
        for role in ROLES:
            wall, monotonic = time.time(), time.monotonic()
            page = scrape(urls[role], plan["scrape_timeout_s"])
            if page["text"] is not None:
                (pages / f"{point}_{role}.prom").write_text(page["text"])
            samples[point][role] = page["samples"]
            report["points"][point][role] = {"url": page["url"], "error": page["error"],
                                             "wall": wall, "monotonic": monotonic}

    def change(start: str, end: str) -> dict:
        return {role: (transfer_delta(samples[start][role], samples[end][role])
                       if samples[start][role] is not None and samples[end][role] is not None
                       else None) for role in ROLES}

    read("baseline")
    previous = "baseline"
    for wave in plan["waves"]:
        try:
            report["statuses"][wave] = pd_replay.main([
                "--trace-dir", str(args.trace_dir / wave), "--output-dir", str(args.output_dir / wave),
                "--prefill-url", args.prefill_url, "--decode-url", args.decode_url, *replay_args])
        except (Exception, SystemExit):
            report["statuses"][wave] = None
            report["errors"][wave] = traceback.format_exc()
            print("WAVE_ERROR", wave, report["errors"][wave], flush=True)
        read(wave)
        report["changes"][wave] = change(previous, wave)
        previous = wave
    time.sleep(plan["release_wait_s"])
    read("released")
    report["changes"]["released"] = change(previous, "released")
    (args.output_dir / "native_transfer_metrics.json").write_text(json.dumps(report, indent=1))
    status = report["statuses"][plan["gating_wave"]]
    print("WAVES_DONE", json.dumps({"statuses": report["statuses"]}), flush=True)
    return 1 if status is None else status


if __name__ == "__main__":
    raise SystemExit(main())
