"""Trace parsing of the kernel-gap profiler (frontier/profiling/kernel_gap)."""

from __future__ import annotations

import pytest

from frontier.profiling.kernel_gap.main import chain_gaps_us, chain_kernels, require_host_ahead, split_chains

ADD_KERNEL = "vectorized_elementwise_kernel<add>"
NORM_KERNEL = "fused_add_rms_norm_kernel"


def _kernel(ts: float, dur: float, correlation: int, name: str = ADD_KERNEL) -> dict:
    return {"cat": "kernel", "name": name, "ts": ts, "dur": dur, "args": {"correlation": correlation}}


def _launch(ts: float, dur: float, correlation: int) -> dict:
    return {"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": ts, "dur": dur,
            "args": {"correlation": correlation}}


def test_chain_gaps_skip_spin_kernels_and_the_boundary_between_repeats() -> None:
    events = [
        _kernel(0.0, 50.0, 1, name="spin_kernel(long)"),
        _kernel(53.0, 2.0, 3, name=NORM_KERNEL),
        _kernel(51.0, 0.5, 2),
        _kernel(1000.0, 0.5, 4),
        _kernel(1004.5, 2.0, 5, name=NORM_KERNEL),
        _launch(0.0, 3.0, 1),
    ]

    kernels = chain_kernels(events)
    chains = split_chains(kernels, repeats=2)

    assert [[kernel["args"]["correlation"] for kernel in chain] for chain in chains] == [[2, 3], [4, 5]]
    assert chain_gaps_us(chains) == [1.5, 4.0]


def test_split_chains_rejects_a_kernel_count_that_does_not_divide_into_repeats() -> None:
    kernels = [_kernel(0.0, 1.0, 1), _kernel(2.0, 1.0, 2), _kernel(4.0, 1.0, 3)]

    with pytest.raises(ValueError, match="do not split into 2 repeats"):
        split_chains(kernels, repeats=2)


def test_split_chains_rejects_repeats_with_different_kernel_sequences() -> None:
    kernels = [_kernel(0.0, 1.0, 1), _kernel(2.0, 1.0, 2, name=NORM_KERNEL),
               _kernel(10.0, 1.0, 3, name=NORM_KERNEL), _kernel(12.0, 1.0, 4)]

    with pytest.raises(ValueError, match="2 different kernel sequences"):
        split_chains(kernels, repeats=2)


def test_host_ahead_accepts_chains_queued_before_their_first_kernel() -> None:
    kernels = [_kernel(100.0, 1.0, 1), _kernel(102.0, 1.0, 2), _kernel(300.0, 1.0, 3), _kernel(302.0, 1.0, 4)]
    events = kernels + [_launch(10.0, 3.0, 1), _launch(14.0, 3.0, 2), _launch(200.0, 3.0, 3), _launch(204.0, 3.0, 4)]

    require_host_ahead(events, split_chains(kernels, repeats=2))


def test_host_ahead_rejects_a_launch_that_returns_after_its_chain_started() -> None:
    kernels = [_kernel(100.0, 1.0, 1), _kernel(102.0, 1.0, 2), _kernel(300.0, 1.0, 3), _kernel(302.0, 1.0, 4)]
    events = kernels + [_launch(10.0, 3.0, 1), _launch(14.0, 3.0, 2), _launch(200.0, 3.0, 3), _launch(299.0, 3.0, 4)]

    with pytest.raises(RuntimeError, match="host was not ahead"):
        require_host_ahead(events, split_chains(kernels, repeats=2))
