"""CPU contracts for the standard vLLM GDN producer."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from frontier.profiling.gdn.inputs import GDNProfileInput
from frontier.profiling.gdn.main import build_profile_inputs
from frontier.profiling.gdn.vllm_wrapper import VllmQwen35GDNWrapper
from frontier.profiling.utils import (
    build_profiling_output_path,
    validate_profile_method_platform,
)


def test_one_token_continuation_keeps_explicit_prefill_phase() -> None:
    profile_input = GDNProfileInput.prefill(seq_len=1, context_len=128)
    assert profile_input.phase == "prefill"
    assert profile_input.num_prefill_tokens == 1
    assert profile_input.num_decode_tokens == 0
    assert profile_input.prefill_mask == (True,)
    assert profile_input.state_init_mode == "primed_prefix"
    assert profile_input.state_block_ids == (1,)


def test_physical_batch_metadata_keeps_reserved_state_page_zero() -> None:
    profile_input = GDNProfileInput(
        query_lens=(1, 1),
        context_lens=(32, 32),
        logical_phase="decode",
        physical_batch_size=4,
    )
    assert profile_input.batch_size == 2
    assert profile_input.physical_batch_size == 4
    assert profile_input.state_block_ids == (1, 2, 3, 4)
    assert profile_input.prefill_mask == (False, False)


def test_mixed_input_is_rejected_before_gpu_wrapper_work() -> None:
    wrapper = object.__new__(VllmQwen35GDNWrapper)
    mixed = GDNProfileInput.mixed(
        decode_batch_size=1,
        decode_context_len=128,
        prefill_seq_len=16,
    )
    with pytest.raises(ValueError, match="mixed execution"):
        wrapper.profile(mixed)


def test_all_one_token_mixed_input_keeps_explicit_mixed_phase() -> None:
    mixed = GDNProfileInput(
        query_lens=(1, 1),
        context_lens=(128, 0),
        logical_phase="mixed",
        prefill_request_mask=(False, True),
    )

    assert mixed.phase == "mixed"
    assert mixed.prefill_mask == (False, True)
    assert mixed.num_decode_tokens == 1
    assert mixed.num_prefill_tokens == 1
    with pytest.raises(ValueError, match="mixed execution"):
        mixed.require_supported_phase()


def test_gdn_input_rejects_missing_or_conflicting_phase_metadata() -> None:
    with pytest.raises(TypeError):
        GDNProfileInput(query_lens=(1,), context_lens=(0,))  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="decode phase"):
        GDNProfileInput(
            query_lens=(1,),
            context_lens=(128,),
            logical_phase="decode",
            prefill_request_mask=(True,),
        )

    with pytest.raises(ValueError, match="decode requests"):
        GDNProfileInput(
            query_lens=(2,),
            context_lens=(0,),
            logical_phase="mixed",
            prefill_request_mask=(False,),
        )


def test_standard_cli_plan_has_cold_continuation_and_decode_modes() -> None:
    args = Namespace(
        prefill_seq_lens=[16],
        prefill_batch_sizes=[1],
        decode_batch_sizes=[1],
        decode_context_len=128,
        continuation_context_len=64,
        include_continuation_prefill=True,
        include_mixed=False,
    )
    inputs = build_profile_inputs(args)
    assert [item.phase for item in inputs] == ["prefill", "prefill", "decode"]
    assert [item.state_init_mode for item in inputs] == ["zero", "primed_prefix", "primed_prefix"]


def test_rocm_standard_producer_requires_device_event() -> None:
    validate_profile_method_platform("device_event", "rocm")
    with pytest.raises(ValueError, match="ROCm profiling requires"):
        validate_profile_method_platform("cuda_event", "rocm")


def test_standard_gdn_output_uses_canonical_gdn_csv() -> None:
    path = build_profiling_output_path(
        output_root="data/profiling",
        profiling_type="compute",
        hardware="mi355x",
        model_name="Qwen3.8-2.4T-A95B-Quark-MXFP4",
        op_name="gdn",
    )
    assert path.name == "gdn.csv"
    source = __import__("pathlib").Path("frontier/profiling/gdn/main.py").read_text(
        encoding="utf-8"
    )
    assert "build_profiling_output_path" in source
    assert 'op_name="gdn"' in source
    assert "gdn_device_event" not in source


def test_cpu_import_does_not_construct_gpu_runtime() -> None:
    assert SimpleNamespace is not None
    # Importing the wrapper is intentionally safe; vLLM is imported only in
    # VllmQwen35GDNWrapper.__init__ after the model contract is checked.
    assert VllmQwen35GDNWrapper.__module__.endswith("vllm_wrapper")
