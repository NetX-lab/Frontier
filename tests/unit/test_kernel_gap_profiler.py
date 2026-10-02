"""Trace parsing of the kernel-gap profiler (frontier/profiling/kernel_gap)."""

from __future__ import annotations

import pytest

from frontier.profiling.kernel_gap.main import chain_gaps_us, chain_kernels, require_host_ahead

CHAIN_KERNEL = "vectorized_elementwise_kernel<add>"


def _kernel(ts: float, dur: float, correlation: int, name: str = CHAIN_KERNEL) -> dict:
    return {"cat": "kernel", "name": name, "ts": ts, "dur": dur, "args": {"correlation": correlation}}


def _launch(ts: float, dur: float, correlation: int) -> dict:
    return {"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": ts, "dur": dur,
            "args": {"correlation": correlation}}


def test_chain_gaps_skip_spin_kernels_and_the_boundary_between_repeats() -> None:
    events = [
        _kernel(0.0, 50.0, 1, name="spin_kernel(long)"),
        _kernel(101.0, 2.0, 3),
        _kernel(51.0, 2.0, 2),
        _kernel(1000.0, 2.0, 4),
        _kernel(1004.5, 2.0, 5),
        _launch(0.0, 3.0, 1),
    ]

    kernels = chain_kernels(events)

    assert [kernel["args"]["correlation"] for kernel in kernels] == [2, 3, 4, 5]
    assert chain_gaps_us(kernels, num_kernels=2, repeats=2) == [48.0, 2.5]


def test_chain_gaps_reject_a_missing_kernel_record() -> None:
    kernels = [_kernel(0.0, 1.0, 1), _kernel(2.0, 1.0, 2), _kernel(4.0, 1.0, 3)]

    with pytest.raises(ValueError, match="expected 2 x 2 chain kernels"):
        chain_gaps_us(kernels, num_kernels=2, repeats=2)


def test_chain_gaps_reject_a_chain_of_two_kernel_names() -> None:
    kernels = [_kernel(0.0, 1.0, 1), _kernel(2.0, 1.0, 2, name="unrolled_elementwise_kernel<add>")]

    with pytest.raises(ValueError, match="one kernel name"):
        chain_gaps_us(kernels, num_kernels=2, repeats=1)


def test_host_ahead_accepts_a_chain_queued_before_its_first_kernel() -> None:
    kernels = [_kernel(100.0, 1.0, 1), _kernel(102.0, 1.0, 2)]
    events = kernels + [_launch(10.0, 3.0, 1), _launch(14.0, 3.0, 2)]

    require_host_ahead(events, kernels, num_kernels=2)


def test_host_ahead_rejects_a_launch_that_returns_after_the_chain_started() -> None:
    kernels = [_kernel(100.0, 1.0, 1), _kernel(102.0, 1.0, 2)]
    events = kernels + [_launch(10.0, 3.0, 1), _launch(99.0, 3.0, 2)]

    with pytest.raises(RuntimeError, match="host was not ahead"):
        require_host_ahead(events, kernels, num_kernels=2)
