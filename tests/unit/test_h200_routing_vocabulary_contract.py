from tests.performance.profiling.validate_h200_six_model_non_dummy_e2e import (
    build_model_contract,
)


def test_h200_validator_uses_canonical_routing_distribution_type() -> None:
    default_contract = build_model_contract("step3-moe-noquant")
    assert default_contract.moe_routing_distribution_type == "balanced"

    random_contract = build_model_contract(
        "mixtral_8x7b_moe",
        moe_routing_distribution_type="random",
    )
    assert random_contract.moe_routing_distribution_type == "random"
    assert random_contract.routing_runtime_path == "uniform_topk"
