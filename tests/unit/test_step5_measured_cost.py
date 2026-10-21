"""Physical-cost ownership checks; fixture values do not admit native profiles."""

from types import SimpleNamespace

import pytest

from frontier.entities.time_components import AttentionOperatorTimes, AttentionTime, OperatorCost
from frontier.execution_time_predictor.kernel_gap import KernelGapPricing
from frontier.types import MeasurementType


def test_inclusive_attention_root_does_not_charge_diagnostic_children():
    cost = OperatorCost(.04, 9, MeasurementType.CUDA_EVENT, "duration-fixture",
                        "count-fixture", "cpu_fixture", True)
    attention = AttentionTime(
        attn_norm_time=.005,
        operator_times=AttentionOperatorTimes({
            "step5_layer_e2e": .04,
        }, costs={"step5_layer_e2e": cost}, diagnostics={
            "step5_qkv_projection": .02, "step5_swa_attention": .03,
        }),
    )
    assert attention.total_time() == pytest.approx(.045)
    assert attention.operator_times.total_time() == pytest.approx(.04)
    assert sum(attention.operator_times.diagnostics.values()) == pytest.approx(.05)
    assert attention.kernel_count == 9
    assert cost.includes_kernel_gap and cost.evidence_kind == "cpu_fixture"


@pytest.mark.parametrize("active", [MeasurementType.CUDA_EVENT, MeasurementType.KERNEL_ONLY])
@pytest.mark.parametrize("attention_type", [MeasurementType.CUDA_EVENT, MeasurementType.KERNEL_ONLY])
def test_mixed_modalities_charge_only_their_unmeasured_kernel_gaps(active, attention_type):
    predictor = KernelGapPricing()
    predictor._kernel_gap_ms = {"eager": .001, "cuda_graph": .002}
    predictor._active_measurement_type = active
    predictor._step_measurement_type = active
    predictor._is_event_measurement_type = lambda method: method is MeasurementType.CUDA_EVENT
    charges = []

    def predict():
        predictor._record_attention_kernels(9, attention_type)
        predictor._kernel_count_scopes[-1].operators["mlp_up_proj"] = 2
        return SimpleNamespace(add_kernel_gap=lambda gap, attention, operators:
                               charges.append((gap, attention, dict(operators))))

    predictor._predict_layer_with_kernel_gap(predict)
    if active is MeasurementType.KERNEL_ONLY:
        assert charges == [(.002, 9 if attention_type is MeasurementType.KERNEL_ONLY else 0,
                            {"mlp_up_proj": 2})]
    elif attention_type is MeasurementType.KERNEL_ONLY:
        assert charges == [(.001, 9, {})]
    else:
        assert charges == []


def test_legacy_attention_kernel_gap_policy_is_unchanged():
    predictor = KernelGapPricing()
    predictor._kernel_gap_ms = {"eager": .001, "cuda_graph": .002}
    predictor._active_measurement_type = MeasurementType.KERNEL_ONLY
    predictor._step_measurement_type = MeasurementType.KERNEL_ONLY
    predictor._is_event_measurement_type = lambda _: False
    charges = []

    def predict():
        predictor._record_attention_kernels(4)
        return SimpleNamespace(add_kernel_gap=lambda *args: charges.append(args))

    predictor._predict_layer_with_kernel_gap(predict)
    assert charges == [(.002, 4, {})]
