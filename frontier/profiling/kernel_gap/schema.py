"""kernel_gap.csv schema shared by the kernel-gap profiler and the predictor that reads it."""

from __future__ import annotations

from typing import Final

# One row per execution mode and TP size: eager kernel launches, or the replay of a CUDA graph.
EXECUTION_MODES: Final[tuple[str, ...]] = ("eager", "cuda_graph")
EXECUTION_MODE_COLUMN: Final[str] = "execution_mode"
# Attention TP size of the profiled decoder layer and of the vLLM deployment whose graphs were replayed.
TENSOR_PARALLEL_COLUMN: Final[str] = "num_tensor_parallel_workers"
# Mean device idle time before a kernel, in microseconds.
KERNEL_GAP_US_COLUMN: Final[str] = "kernel_gap_us"

# Segment classes of the DeepSeek-V4 replay stream program.
CUDA_GRAPH: Final[str] = "cuda_graph"
CUDA_GRAPH_PIECEWISE: Final[str] = "cuda_graph_piecewise"
EAGER: Final[str] = "eager"
EAGER_PREPARATION: Final[str] = "eager_preparation"
DSV4_EXECUTION_MODES: Final[tuple[str, ...]] = (
    CUDA_GRAPH, CUDA_GRAPH_PIECEWISE, EAGER, EAGER_PREPARATION,
)
