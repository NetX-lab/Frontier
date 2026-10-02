"""Trace parsing of the kernel-gap profiler (frontier/profiling/kernel_gap)."""

from __future__ import annotations

import pytest

from frontier.profiling.kernel_gap.main import (chain_gaps_us, chain_kernels, graph_launch_chains, require_host_ahead,
                                                split_chains)

ADD_KERNEL = "vectorized_elementwise_kernel<add>"
NORM_KERNEL = "fused_add_rms_norm_kernel"


def _kernel(ts: float, dur: float, correlation: int, name: str = ADD_KERNEL) -> dict:
    return {"cat": "kernel", "name": name, "ts": ts, "dur": dur, "args": {"correlation": correlation}}


def _launch(ts: float, dur: float, correlation: int, name: str = "cudaLaunchKernel") -> dict:
    return {"cat": "cuda_runtime", "name": name, "ts": ts, "dur": dur, "args": {"correlation": correlation}}


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


def test_chain_gaps_count_an_overlap_as_no_idle_time() -> None:
    chain = [_kernel(0.0, 10.0, 1), _kernel(9.5, 1.0, 2), _kernel(10.2, 0.2, 3), _kernel(11.0, 1.0, 4)]

    assert chain_gaps_us([chain]) == [0.0, 0.0, 0.5]


def test_graph_launch_chains_keep_only_the_kernels_of_each_graph_launch() -> None:
    events = [
        _launch(0.0, 300.0, 10, name="cudaGraphLaunch"),
        _kernel(5.0, 1.0, 8, name="FillFunctor<long>"),
        _kernel(320.0, 2.0, 10, name=NORM_KERNEL),
        _kernel(310.0, 4.0, 10),
        _launch(400.0, 300.0, 20, name="cudaGraphLaunch"),
        _kernel(710.0, 4.0, 20),
        _kernel(714.5, 2.0, 20, name=NORM_KERNEL),
    ]

    chains = graph_launch_chains(events, repeats=2)

    assert [[kernel["ts"] for kernel in chain] for chain in chains] == [[310.0, 320.0], [710.0, 714.5]]
    assert chain_gaps_us(chains) == [6.0, 0.5]


def test_graph_launch_chains_reject_a_launch_with_a_different_kernel_count() -> None:
    events = [_launch(0.0, 3.0, 10, name="cudaGraphLaunch"), _kernel(10.0, 1.0, 10), _kernel(12.0, 1.0, 10),
              _launch(20.0, 3.0, 20, name="cudaGraphLaunch"), _kernel(30.0, 1.0, 20)]

    with pytest.raises(ValueError, match="kernel counts \\[2, 1\\]"):
        graph_launch_chains(events, repeats=2)


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
