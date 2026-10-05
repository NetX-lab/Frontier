"""kernel_gap.csv schema shared by the kernel-gap profiler and the predictor that reads it."""

from __future__ import annotations

from typing import Final

# One row per execution mode: eager kernel launches, or the replay of a CUDA graph.
EXECUTION_MODES: Final[tuple[str, ...]] = ("eager", "cuda_graph")
EXECUTION_MODE_COLUMN: Final[str] = "execution_mode"
# Mean device idle time before a kernel, in microseconds.
KERNEL_GAP_US_COLUMN: Final[str] = "kernel_gap_us"
