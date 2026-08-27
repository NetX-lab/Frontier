import pytest

from frontier.profiling.moe.load_distribution import generate_expert_routing


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_tokens": 0}, "num_tokens"),
        ({"num_tokens": -1}, "num_tokens"),
        ({"num_experts": 0, "top_k": 0}, "num_experts"),
        ({"num_experts": -1, "top_k": 0}, "num_experts"),
        ({"top_k": 0}, "top_k"),
        ({"top_k": -1}, "top_k"),
    ],
)
def test_generate_expert_routing_rejects_non_positive_dimensions(
    kwargs,
    message,
):
    base_kwargs = {
        "num_tokens": 4,
        "num_experts": 8,
        "top_k": 2,
        "load_distribution": "uniform",
        "seed": 42,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        generate_expert_routing(**base_kwargs)


def test_generate_expert_routing_accepts_full_width_topk():
    weights, expert_ids = generate_expert_routing(
        num_tokens=4,
        num_experts=8,
        top_k=8,
        load_distribution="uniform",
        seed=42,
    )

    assert tuple(weights.shape) == (4, 8)
    assert tuple(expert_ids.shape) == (4, 8)


def test_generate_expert_routing_rejects_topk_above_expert_count():
    with pytest.raises(ValueError, match="top_k.*cannot exceed num_experts"):
        generate_expert_routing(
            num_tokens=4,
            num_experts=8,
            top_k=9,
            load_distribution="uniform",
            seed=42,
        )
