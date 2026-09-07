"""Measure long-interval CUDA/host clock divergence on all allocated GPUs."""

import json
import sys
import time
from pathlib import Path

import torch


def bracket(event):
    before = time.monotonic()
    event.record()
    event.synchronize()
    return before, time.monotonic()


def main():
    path = Path(sys.argv[1])
    anchors = []
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            anchor = torch.cuda.Event(enable_timing=True)
            reference = torch.cuda.Event(enable_timing=True)
            bracket(anchor)
            bracket(reference)
            anchors.append((anchor, reference, *bracket(anchor)))
    started = time.monotonic()
    with path.open("x") as stream:
        for delay in (0, 30, 60, 90, 120, 150):
            remaining = started + delay - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            for device, (anchor, reference, anchor_before, anchor_after) in enumerate(anchors):
                with torch.cuda.device(device):
                    bracket(reference)  # Materialize the wake-up outside the measured bracket.
                    before, after = bracket(reference)
                    elapsed_s = anchor.elapsed_time(reference) * 1e-3
                    mapped_before = anchor_before + elapsed_s
                    mapped_after = anchor_after + elapsed_s
                    row = {"gpu": device, "host_elapsed_s": before - anchor_before,
                           "cuda_elapsed_s": elapsed_s,
                           "anchor_width_us": (anchor_after - anchor_before) * 1e6,
                           "reference_width_us": (after - before) * 1e6,
                           "mapped_minus_observed_midpoint_us":
                               (mapped_before + mapped_after - before - after) * 5e5,
                           "interval_gap_us": max(before - mapped_after, mapped_before - after, 0) * 1e6}
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    print(json.dumps(row), flush=True)
    print("CLOCK_DRIFT_PROBE_COMPLETE")


if __name__ == "__main__":
    main()
