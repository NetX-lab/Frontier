"""Registry feature and physical profiling boundaries for measured Step compute."""

from frontier.attention.families import STEP5_ATTENTION_FAMILY
from frontier.attention.profiling_mapping import (
    STEP5_FEATURE_COLUMNS,
    get_attention_profiling_feature_schema,
    get_enabled_predictor_feature_columns,
    get_enabled_shared_predictor_feature_columns,
)
from frontier.execution_time_predictor.step5 import STEP5_FEATURE_COLUMNS as CONSUMER_FEATURES
from frontier.profiling.step5.rows import OPERATOR_NAMES


def test_predictor_and_shared_registry_use_one_step_feature_schema():
    expected = {"step5_layer_e2e": STEP5_FEATURE_COLUMNS}
    assert CONSUMER_FEATURES is STEP5_FEATURE_COLUMNS
    assert get_enabled_predictor_feature_columns(STEP5_ATTENTION_FAMILY) == expected
    assert get_enabled_shared_predictor_feature_columns(STEP5_ATTENTION_FAMILY) == expected
    assert get_attention_profiling_feature_schema(STEP5_ATTENTION_FAMILY).predictor_feature_columns == STEP5_FEATURE_COLUMNS


def test_predictor_root_is_not_an_instrumented_native_scope():
    assert len(OPERATOR_NAMES) == 9
    assert OPERATOR_NAMES == tuple(op.name for op in STEP5_ATTENTION_FAMILY.profiling_ops())
    assert "step5_layer_e2e" not in OPERATOR_NAMES
    assert [op.name for op in STEP5_ATTENTION_FAMILY.predictor_ops()] == ["step5_layer_e2e"]
