from dataclasses import replace
from types import SimpleNamespace
import gzip
import json

import numpy as np
import pytest

from frontier.validation.routing import read_routing, snapshot_routing


def sample(ids=None, weights=None, **changes):
    options = dict(batch_id="b", rank=0, layer_id=0, capture_id=1,
                   logical_size=2, num_experts=4, top_k=2)
    options.update(changes)
    return snapshot_routing([[0, 1], [1, 2], [0, 0]] if ids is None else ids,
        [[.5, .5], [.4, .6], [0., 0.]] if weights is None else weights, **options)


def test_padding_and_zero_weight_assignments_are_not_labeled_live_tokens():
    row = sample()
    assert row.logical_expert_counts == (1, 2, 1, 0)
    assert row.padding_expert_counts == (2, 0, 0, 0)
    assert row.logical_positive_weight_slots == 4 and row.padding_positive_weight_slots == 0
    assert row.to_lane_workload(include_padding=False).local_token_counts == (1, 2, 1, 0)
    assert row.to_lane_workload(include_padding=True).local_token_counts == (3, 2, 1, 0)
    model = SimpleNamespace(num_experts=4, num_experts_per_tok=2, embedding_dim=8, mlp_hidden_dim=16)
    logical = row.native_load_features(model_config=model, include_padding=False)
    physical = row.native_load_features(model_config=model, include_padding=True)
    assert logical["total_routed_tokens"] == 4 and physical["total_routed_tokens"] == 6
    assert logical["max_load_ratio"] == 2
    assert physical["max_load_ratio"] == 2
    model.num_experts = 8
    with pytest.raises(ValueError, match="match model"):
        row.native_load_features(model_config=model, include_padding=False)


def test_sentinels_only_in_padding_and_slot_order_is_distinct_from_expert_set():
    row = sample(ids=[[0, 1], [1, 2], [-1, -1]])
    assert row.padding_expert_counts == (0, 0, 0, 0) and row.invalid_padding_slots == 2
    reordered = sample(ids=[[1, 0], [2, 1], [-1, -1]])
    assert row.logical_assignment_sha256 != reordered.logical_assignment_sha256
    assert row.logical_expert_set_sha256 == reordered.logical_expert_set_sha256
    assert row.logical_weights_sha256 == reordered.logical_weights_sha256


@pytest.mark.parametrize("changes", [
    {"ids": [[0, 0], [1, 2], [0, 0]]}, {"ids": [[-1, 1], [1, 2], [0, 0]]},
    {"ids": [[0, 4], [1, 2], [0, 0]]}, {"ids": [[0., 1.], [1., 2.], [0., 0.]]},
    {"weights": [[.5, .5], [.4, float("nan")], [0., 0.]]},
    {"weights": [[.5, -.5], [.4, .6], [0., 0.]]},
    {"ids": [[0, 1], [1, 2], [-1, -1]], "weights": [[.5, .5], [.4, .6], [.1, 0.]]},
    {"logical_size": 4}, {"top_k": 3}, {"rank": True},
])
def test_invalid_topk_fails(changes):
    with pytest.raises(ValueError):
        sample(**changes)


def test_roundtrip_schema_and_duplicate_coverage(tmp_path):
    row = sample()
    path = tmp_path / "routing.jsonl.gz"
    with gzip.open(path, "wt") as stream:
        stream.write(json.dumps(row.to_dict()) + "\n")
    assert list(read_routing(path)) == [row]
    with gzip.open(path, "at") as stream:
        stream.write(json.dumps(row.to_dict()) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        list(read_routing(path))
    with pytest.raises(ValueError, match="Logical counts"):
        replace(row, logical_expert_counts=(2, 2, 2, 0))
    with pytest.raises(ValueError, match="Padding counts"):
        replace(row, padding_expert_counts=(0, 0, 0, 0))
    with pytest.raises(ValueError, match="digest"):
        replace(row, logical_assignment_sha256="bad")
