from __future__ import annotations

import pytest

from frontier.attention.families import DENSE_ATTENTION_FAMILY
from frontier.attention.ops import AttentionFamilySpec, AttentionOperatorSpec
from frontier.operators.spec import (
    OperatorFamilySpec,
    OperatorPhase,
    OperatorRole,
    OperatorSpec,
    ProjectionOwnership,
    ResourceClass,
    TraceKind,
)


def test_attention_specs_are_operator_specs_with_byte_stable_public_names() -> None:
    operator = DENSE_ATTENTION_FAMILY.operators[0]

    assert isinstance(operator, OperatorSpec)
    assert isinstance(operator, AttentionOperatorSpec)
    assert isinstance(DENSE_ATTENTION_FAMILY, OperatorFamilySpec)
    assert isinstance(DENSE_ATTENTION_FAMILY, AttentionFamilySpec)
    assert operator.role is OperatorRole.CACHE_WRITE
    assert operator.phases == (
        OperatorPhase.PREFILL,
        OperatorPhase.DECODE,
        OperatorPhase.MIXED,
    )
    assert operator.trace_kind is TraceKind.COMPUTE
    assert operator.projection_ownership is ProjectionOwnership.NOT_PROJECTION
    assert DENSE_ATTENTION_FAMILY.resource_class is ResourceClass.COMP


def test_operator_spec_rejects_invalid_projection_without_attention_coupling() -> None:
    with pytest.raises(ValueError, match="Projection operator bad_projection"):
        OperatorSpec(
            name="bad_projection",
            role=OperatorRole.PROJECTION,
            phases=(OperatorPhase.PREFILL,),
            execution_time_attr="bad_projection_time",
            projection_ownership=ProjectionOwnership.NOT_PROJECTION,
        )


def test_operator_family_total_ops_preserves_ordered_operator_views() -> None:
    family = OperatorFamilySpec(
        family_id="unit_family",
        display_name="Unit Family",
        supported_variants=("unit",),
        operators=(
            OperatorSpec(
                name="first",
                role=OperatorRole.PREFILL_KERNEL,
                phases=(OperatorPhase.PREFILL,),
                execution_time_attr="first_time",
                resource_class=ResourceClass.COMP,
            ),
            OperatorSpec(
                name="second",
                role=OperatorRole.DECODE_KERNEL,
                phases=(OperatorPhase.DECODE,),
                execution_time_attr="second_time",
                resource_class=ResourceClass.COMP,
                profiling_target=False,
            ),
        ),
    )

    assert [op.name for op in family.predictor_ops()] == ["first", "second"]
    assert [op.name for op in family.profiling_ops()] == ["first"]
    assert [op.name for op in family.e2e_trace_ops()] == ["first", "second"]


def test_operator_spec_can_expose_legacy_profiling_key_without_renaming_operator() -> None:
    operator = OperatorSpec(
        name="add_attn_residual",
        role=OperatorRole.RESIDUAL,
        phases=(OperatorPhase.PREFILL,),
        execution_time_attr="add_attn_residual_time",
        profiling_key="add",
    )

    assert operator.name == "add_attn_residual"
    assert operator.profiling_name() == "add"


def test_operator_spec_can_expose_precision_key_without_renaming_operator() -> None:
    operator = OperatorSpec(
        name="add_ffn_residual",
        role=OperatorRole.RESIDUAL,
        phases=(OperatorPhase.PREFILL,),
        execution_time_attr="add_ffn_residual_time",
        precision_op="add",
    )

    assert operator.name == "add_ffn_residual"
    assert operator.precision_name() == "add"


def test_operator_spec_can_expose_calibration_key_without_renaming_operator() -> None:
    operator = OperatorSpec(
        name="mlp_up_proj",
        role=OperatorRole.PROJECTION,
        phases=(OperatorPhase.PREFILL,),
        execution_time_attr="mlp_layer_up_proj_execution_time",
        projection_ownership=ProjectionOwnership.OUTSIDE_ATTENTION,
        calibration_key="mlp_up_proj",
    )

    assert operator.name == "mlp_up_proj"
    assert operator.calibration_field_name() == "mlp_up_proj_calibration_scale"
    assert operator.calibration_attr_name() == "_mlp_up_proj_calibration_scale"


def test_operator_family_profiling_ops_can_preserve_legacy_profile_order() -> None:
    family = OperatorFamilySpec(
        family_id="unit_family",
        display_name="Unit Family",
        supported_variants=("unit",),
        profiling_order=("third", "first"),
        operators=(
            OperatorSpec(
                name="first",
                role=OperatorRole.PREFILL_KERNEL,
                phases=(OperatorPhase.PREFILL,),
                execution_time_attr="first_time",
                resource_class=ResourceClass.COMP,
            ),
            OperatorSpec(
                name="second",
                role=OperatorRole.DECODE_KERNEL,
                phases=(OperatorPhase.DECODE,),
                execution_time_attr="second_time",
                resource_class=ResourceClass.COMP,
                profiling_target=False,
            ),
            OperatorSpec(
                name="third",
                role=OperatorRole.DECODE_KERNEL,
                phases=(OperatorPhase.DECODE,),
                execution_time_attr="third_time",
                resource_class=ResourceClass.COMP,
            ),
        ),
    )

    assert [op.name for op in family.profiling_ops()] == ["third", "first"]


def test_operator_family_rejects_operator_without_resource_class() -> None:
    with pytest.raises(ValueError, match="missing resource_class"):
        OperatorFamilySpec(
            family_id="missing_resource_class_family",
            display_name="Missing Resource Class Family",
            supported_variants=("unit",),
            operators=(
                OperatorSpec(
                    name="missing_resource_class_op",
                    role=OperatorRole.PREFILL_KERNEL,
                    phases=(OperatorPhase.PREFILL,),
                    execution_time_attr="missing_resource_class_time",
                ),
            ),
        )


def test_operator_spec_rejects_invalid_resource_class_value() -> None:
    with pytest.raises(ValueError, match="resource_class must be ResourceClass"):
        OperatorSpec(
            name="invalid_resource_class_op",
            role=OperatorRole.PREFILL_KERNEL,
            phases=(OperatorPhase.PREFILL,),
            execution_time_attr="invalid_resource_class_time",
            resource_class="GPU",  # type: ignore[arg-type]
        )


def test_operator_family_rejects_invalid_family_resource_class_value() -> None:
    with pytest.raises(ValueError, match="resource_class must be ResourceClass"):
        OperatorFamilySpec(
            family_id="invalid_resource_class_family",
            display_name="Invalid Resource Class Family",
            supported_variants=("unit",),
            resource_class="GPU",  # type: ignore[arg-type]
            operators=(
                OperatorSpec(
                    name="valid_resource_class_op",
                    role=OperatorRole.PREFILL_KERNEL,
                    phases=(OperatorPhase.PREFILL,),
                    execution_time_attr="valid_resource_class_time",
                    resource_class=ResourceClass.COMP,
                ),
            ),
        )


def test_operator_family_rejects_operator_with_invalid_resource_class_value() -> None:
    operator = OperatorSpec(
        name="corrupted_resource_class_op",
        role=OperatorRole.PREFILL_KERNEL,
        phases=(OperatorPhase.PREFILL,),
        execution_time_attr="corrupted_resource_class_time",
        resource_class=ResourceClass.COMP,
    )
    object.__setattr__(operator, "resource_class", "GPU")

    with pytest.raises(ValueError, match="invalid resource_class"):
        OperatorFamilySpec(
            family_id="corrupted_resource_class_family",
            display_name="Corrupted Resource Class Family",
            supported_variants=("unit",),
            operators=(operator,),
        )
