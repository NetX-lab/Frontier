"""Require actual active GDN measurements before constructing profiling rows."""

from __future__ import annotations

from copy import deepcopy

import pytest

from frontier.profiling.gdn.vllm_wrapper import VllmQwen35GDNWrapper


ACTIVE_SAMPLE_NAMES = (
    "gdn_input_projections",
    "gdn_core_prefill",
    "gdn_output_projection",
    "gdn_layer_e2e",
)


def _samples(phase: str) -> dict[str, list[float]]:
    return {
        "gdn_input_projections": [1.0, 3.0],
        f"gdn_core_{phase}": [2.0, 4.0],
        "gdn_output_projection": [3.0, 5.0],
        "gdn_layer_e2e": [9.0, 11.0],
    }


def _validate(samples, phase="prefill"):
    return VllmQwen35GDNWrapper._validated_time_stats(
        samples, phase=phase, profile_iterations=2
    )


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_valid_phase_preserves_measured_stats_and_zeros_only_inactive_core(phase):
    samples = _samples(phase)
    original = deepcopy(samples)

    stats = _validate(samples, phase)

    inactive_name = "gdn_core_decode" if phase == "prefill" else "gdn_core_prefill"
    assert set(stats) == set(samples) | {inactive_name}
    for name, minimum, maximum, mean in (
        ("gdn_input_projections", 1.0, 3.0, 2.0),
        (f"gdn_core_{phase}", 2.0, 4.0, 3.0),
        ("gdn_output_projection", 3.0, 5.0, 4.0),
        ("gdn_layer_e2e", 9.0, 11.0, 10.0),
    ):
        assert stats[name] == pytest.approx(
            {"min": minimum, "max": maximum, "mean": mean,
             "median": mean, "std": 1.0, "count": 2}
        )
    assert stats[inactive_name] == {
        "min": 0.0, "max": 0.0, "mean": 0.0,
        "median": 0.0, "std": 0.0, "count": 2,
    }
    assert samples == original


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("missing_name", ACTIVE_SAMPLE_NAMES)
def test_missing_active_or_e2e_samples_are_rejected(phase, missing_name):
    samples = _samples(phase)
    missing_name = missing_name.replace("core_prefill", f"core_{phase}")
    del samples[missing_name]

    with pytest.raises((ValueError, RuntimeError)):
        _validate(samples, phase)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_empty_store_cannot_become_zero_measurements(phase):
    with pytest.raises((ValueError, RuntimeError)):
        _validate({}, phase)


@pytest.mark.parametrize("name", ACTIVE_SAMPLE_NAMES)
@pytest.mark.parametrize("count", [0, 1, 3], ids=["empty", "too-few", "too-many"])
def test_active_and_e2e_counts_must_match_requested_iterations(name, count):
    samples = _samples("prefill")
    samples[name] = [1.0] * count

    with pytest.raises((ValueError, RuntimeError)):
        _validate(samples)


@pytest.mark.parametrize("name", ACTIVE_SAMPLE_NAMES)
@pytest.mark.parametrize(
    "invalid_sample", [float("nan"), float("inf"), float("-inf"), -0.01],
    ids=["nan", "positive-infinity", "negative-infinity", "negative"],
)
def test_active_and_e2e_samples_must_be_finite_and_nonnegative(name, invalid_sample):
    samples = _samples("prefill")
    samples[name][1] = invalid_sample

    with pytest.raises((ValueError, RuntimeError)):
        _validate(samples)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_observed_zero_samples_are_valid_measurements(phase):
    samples = {name: [0.0, 0.0] for name in _samples(phase)}

    stats = _validate(samples, phase)

    inactive_name = "gdn_core_decode" if phase == "prefill" else "gdn_core_prefill"
    assert set(stats) == set(samples) | {inactive_name}
    expected = {
        "min": 0.0, "max": 0.0, "mean": 0.0,
        "median": 0.0, "std": 0.0, "count": 2,
    }
    assert all(record == expected for record in stats.values())


@pytest.mark.parametrize("remote_problem", ["missing-operator", "bad-count", "invalid-value"])
def test_tp_contract_rejects_rank_mismatch_before_tensor_collective(remote_problem):
    from types import SimpleNamespace

    def gather(records, local):
        signature, error = local
        assert error is None
        remote_signature, remote_error = signature, None
        if remote_problem == "missing-operator":
            remote_signature = signature[:-1]
        elif remote_problem == "bad-count":
            remote_signature = ((signature[0][0], 1),) + signature[1:]
        else:
            remote_error = "GDN measurement must be finite and nonnegative"
        records[:] = [local, (remote_signature, remote_error)]

    owner = SimpleNamespace(
        tensor_parallel_size=2,
        torch=SimpleNamespace(distributed=SimpleNamespace(all_gather_object=gather)),
        _validated_time_stats=VllmQwen35GDNWrapper._validated_time_stats,
    )
    with pytest.raises(ValueError, match="GDN rank samples"):
        VllmQwen35GDNWrapper._aggregate_time_samples(
            owner, _samples("prefill"), phase="prefill", profile_iterations=2,
        )
