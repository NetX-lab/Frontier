"""Phase dispatch and native captures for address-stable attention forwards."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import gc


def _capture_sizes(capture_sizes: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(capture_sizes, Sequence):
        raise ValueError("capture_sizes must be a non-empty sequence of positive, sorted, unique integers")
    sizes = tuple(capture_sizes)
    if (
        not sizes
        or any(type(size) is not int or size <= 0 for size in sizes)
        or any(left >= right for left, right in zip(sizes, sizes[1:]))
    ):
        raise ValueError("capture_sizes must be a non-empty sequence of positive, sorted, unique integers")
    return sizes


def _positive_tokens(num_tokens: int) -> None:
    if type(num_tokens) is not int or num_tokens <= 0:
        raise ValueError("num_tokens must be a positive integer")


def choose_graph_mode(
    phase: str, num_tokens: int, capture_sizes: Sequence[int] = (1, 2),
) -> str:
    """Resolve FULL decode, breakable PIECEWISE prefill/mixed, or an eager miss.

    Decode denotes uniform single-token decode requests. The caller owns batch
    admission and supplies the native capture sizes; this is not a scheduler.
    A graph runs the padded rows given by :func:`padded_rows`.
    """
    if phase not in ("decode", "prefill", "mixed"):
        raise ValueError("phase must be decode, prefill, or mixed")
    _positive_tokens(num_tokens)
    if num_tokens > _capture_sizes(capture_sizes)[-1]:
        return "none"
    return "full" if phase == "decode" else "piecewise"


def padded_rows(num_tokens: int, capture_sizes: Sequence[int]) -> int:
    """Physical rows of native dispatch: the smallest capture size that holds the tokens.

    Tokens above the largest capture size run eager without padding.
    """
    # Imported here because the stdlib-only smoke judge imports this module.
    from frontier.config.graph_policy import resolve_capture_size

    _positive_tokens(num_tokens)
    return resolve_capture_size(num_tokens, _capture_sizes(capture_sizes))[1]


def capture_native_replay(
    torch, native, layer, forward: Callable[..., object], graph_mode: str,
) -> Callable[[], object]:
    """Capture once outside timing and return the native replay and its output.

    The caller owns warmup, current vLLM/forward contexts (including a
    BatchDescriptor for PIECEWISE), instrumentation and all input/state buffers.
    Keep those buffers alive and update them in place across replays. Each call
    creates an independent graph or breakable wrapper, even for the same shape.
    Native break decorators must already be enabled when the layer is imported.
    """
    if graph_mode not in ("none", "piecewise", "full"):
        raise ValueError("graph_mode must be none, piecewise, or full")
    if graph_mode == "none":
        return lambda: forward(runtime_mode="none")
    if graph_mode == "piecewise":
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
        from vllm.config import CUDAGraphMode
        from vllm.distributed.parallel_state import graph_capture

        # vLLM keeps its graphs for the process lifetime; the profiler drops them per capture.
        # A dropped graph leaves the global pool id freeable in the pinned-host allocator, and
        # a new capture into that id fails its use-count assert until the host cache is
        # emptied. torch.cuda.graph does this before capture; the breakable wrapper does not.
        gc.collect()
        torch.cuda.synchronize()
        torch.accelerator.empty_host_cache()
        wrapped = BreakableCUDAGraphWrapper(
            layer, native, runtime_mode=CUDAGraphMode.PIECEWISE,
        )
        with graph_capture(device=torch.device("cuda")):
            forward(runtime_mode="piecewise", layer=wrapped)
        return lambda: forward(runtime_mode="piecewise", layer=wrapped)

    from vllm.compilation.cudagraph_pool import capture_pool
    from vllm.distributed.parallel_state import graph_capture
    from vllm.model_executor.offloader.base import get_offloader
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import current_stream

    with graph_capture(device=torch.device("cuda")):
        graph = torch.cuda.CUDAGraph()
        get_offloader().sync_prev_onload()
        with (
            capture_pool(current_platform.get_global_graph_pool(), native) as pool,
            torch.cuda.graph(graph, pool, stream=current_stream()),
        ):
            output = forward(runtime_mode="none")
            get_offloader().join_after_forward()

    def replay():
        get_offloader().sync_prev_onload()
        graph.replay()
        return output

    return replay
