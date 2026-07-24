from tests.e2e.pd_af_parity.harness import (
    ComparisonResult,
    ParityLayer,
    compare_requests,
)


def _complete_row() -> dict[str, int | float]:
    return {
        "request_num_prefill_tokens": 512,
        "request_num_decode_tokens": 128,
        "request_num_tokens": 640,
        "request_num_restarts": 0,
        "request_thinking_round_count": 0,
        "request_e2e_time": 10.0,
        "request_execution_time": 9.0,
        "prefill_e2e_time": 4.0,
        "decode_e2e_time": 6.0,
        "ttft": 4.0,
        "tpot": 0.05,
        "transfer_kv_cache": 0.1,
        "transfer_m2n_total": 0.2,
        "transfer_m2n_attn_to_ffn": 0.1,
        "transfer_m2n_ffn_to_attn": 0.1,
        "cluster_prefill_computation": 3.0,
        "cluster_decode_attn_computation": 2.0,
        "cluster_decode_ffn_computation": 4.0,
        "cross_branch_first_token_ttft_ms": 4.5,
    }


def test_missing_metric_on_both_branches_is_a_hard_failure() -> None:
    main_row = _complete_row()
    ref_row = _complete_row()
    del main_row["request_e2e_time"]
    del ref_row["request_e2e_time"]

    [comparison] = compare_requests(
        {0: main_row},
        {0: ref_row},
        ParityLayer.L2_TRAINED,
    )

    field = next(
        item for item in comparison.fields if item.field_name == "request_e2e_time"
    )
    assert field.result is ComparisonResult.MISSING_FIELD
    assert comparison.passed is False
    assert comparison.first_divergence_field == "request_e2e_time"


def test_missing_discrete_field_on_one_branch_is_a_hard_failure() -> None:
    main_row = _complete_row()
    ref_row = _complete_row()
    del ref_row["request_num_decode_tokens"]

    [comparison] = compare_requests(
        {0: main_row},
        {0: ref_row},
        ParityLayer.L2_TRAINED,
    )

    field = next(
        item
        for item in comparison.fields
        if item.field_name == "request_num_decode_tokens"
    )
    assert field.result is ComparisonResult.MISSING_FIELD
    assert comparison.passed is False
    assert comparison.first_divergence_field == "request_num_decode_tokens"


def test_missing_discrete_field_on_both_branches_is_a_hard_failure() -> None:
    main_row = _complete_row()
    ref_row = _complete_row()
    del main_row["request_num_decode_tokens"]
    del ref_row["request_num_decode_tokens"]

    [comparison] = compare_requests(
        {0: main_row},
        {0: ref_row},
        ParityLayer.L2_TRAINED,
    )

    field = next(
        item
        for item in comparison.fields
        if item.field_name == "request_num_decode_tokens"
    )
    assert field.result is ComparisonResult.MISSING_FIELD
    assert comparison.passed is False
    assert comparison.first_divergence_field == "request_num_decode_tokens"


def test_real_numeric_mismatch_is_a_hard_failure() -> None:
    main_row = _complete_row()
    ref_row = _complete_row()
    main_row["request_e2e_time"] = 11.0

    [comparison] = compare_requests(
        {0: main_row},
        {0: ref_row},
        ParityLayer.L2_TRAINED,
    )

    field = next(
        item for item in comparison.fields if item.field_name == "request_e2e_time"
    )
    assert field.result is ComparisonResult.MISMATCH
    assert comparison.passed is False
    assert comparison.first_divergence_field == "request_e2e_time"


def test_complete_identical_rows_still_pass() -> None:
    row = _complete_row()

    [comparison] = compare_requests(
        {0: row},
        {0: dict(row)},
        ParityLayer.L2_TRAINED,
    )

    assert comparison.passed is True
    assert comparison.first_divergence_field is None
    assert all(
        field.result
        in {ComparisonResult.EXACT_MATCH, ComparisonResult.WITHIN_TOLERANCE}
        for field in comparison.fields
    )
