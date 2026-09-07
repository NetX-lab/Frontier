"""Validate CUDA endpoint clocks and recorder cost on the H200 runtime."""

import json
import statistics
import sys
import time
from pathlib import Path

import torch

from vllm.v1.metrics.frontier_prefill import FrontierPrefillEndpointRecorder


def main() -> None:
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=False)
    results = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            recorder = FrontierPrefillEndpointRecorder(str(root / "endpoints"), index)
            inputs = torch.randn((4096, 2048), device="cuda", dtype=torch.bfloat16)
            weights = torch.randn((2048, 2048), device="cuda", dtype=torch.bfloat16)
            for _ in range(3):
                (inputs @ weights)[0, 0].item()
            baseline_us, recorded_us = [], []
            for iteration in range(64):
                # Alternate order to avoid assigning all warmest runs to one mode.
                for enabled in ([False, True] if iteration % 2 == 0 else [True, False]):
                    started = time.monotonic()
                    output = inputs @ weights
                    if enabled:
                        recorder.mark_forward_end()
                    output[0, 0].item()  # Natural output synchronization analogue.
                    if enabled:
                        recorder.finish_after_sync([f"clock-check:{index}:{iteration}"])
                    elapsed_us = (time.monotonic() - started) * 1e6
                    (recorded_us if enabled else baseline_us).append(elapsed_us)

            # Compare the retained GPU anchor to an independent current idle bracket.
            reference = torch.cuda.Event(enable_timing=True)
            reference.record()
            reference.synchronize()
            lower = time.monotonic()
            reference.record()
            reference.synchronize()
            upper = time.monotonic()
            delta_s = recorder._anchor.elapsed_time(reference) * 1e-3
            predicted_lower = recorder._anchor_before + delta_s
            predicted_upper = recorder._anchor_after + delta_s
            interval_gap_us = max(lower - predicted_upper,
                                  predicted_lower - upper, 0.0) * 1e6
            rows = [json.loads(line) for line in
                    (root / f"endpoints.rank{index}.jsonl").read_text().splitlines()]
            completed = [row for row in rows if row["event"] == "prefill_completed"]
            assert len(completed) == 64
            assert len({row["request_ids"][0] for row in completed}) == 64
            anchor_width_us = (recorder._anchor_after - recorder._anchor_before) * 1e6
            results.append({
                "gpu": index,
                "anchor_width_us": anchor_width_us,
                "clock_interval_gap_us": interval_gap_us,
                "baseline_mean_us": statistics.mean(baseline_us),
                "recorded_mean_us": statistics.mean(recorded_us),
                "mean_added_us": statistics.mean(recorded_us) - statistics.mean(baseline_us),
                "baseline_median_us": statistics.median(baseline_us),
                "recorded_median_us": statistics.median(recorded_us),
                "record_count": len(completed),
            })
            assert anchor_width_us <= 100, (index, "anchor precision", anchor_width_us)
            assert interval_gap_us <= 100, (index, "clock drift", interval_gap_us)
    (root / "clock_overhead.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))
    print("PREFILL_ENDPOINT_CLOCK_PASS")


if __name__ == "__main__":
    main()
