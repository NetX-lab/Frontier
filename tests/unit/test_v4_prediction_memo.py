"""Caching preserves balanced EP validation, V4 coverage and replica-local facts."""
from dataclasses import replace
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError
from frontier.attention.dsv4_features import DeepSeekV4BatchFeatures, dsv4_feature_columns
from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.deepseek_v4 import _layer_facts
from frontier.execution_time_predictor.dsv4_operator_predictor import operator_tensor_parallel_size
from frontier.execution_time_predictor.dsv4_predictor import (
    DeepSeekV4AttentionPredictor, TARGET_TIME, TARGET_GAP, dsv4_model_key, dsv4_model_name,
)
from frontier.moe_ep_workload import materialize_balanced_ep_workload, materialize_balanced_rank_classes
from frontier.operators.binding import build_operator_manifest
from frontier.operators.deepseek_v4_axes import DEEPSEEK_V4_OPERATOR_AXES


MATERIALIZERS = (materialize_balanced_ep_workload, materialize_balanced_rank_classes)


@pytest.mark.parametrize("materialize", MATERIALIZERS)
@pytest.mark.parametrize("population,topk,experts,ep", [
    (population, topk, experts, ep)
    for population, topk, experts in product((0, 1, 3, 17, 128, 6144), (1, 2, 4), (4, 16, 256))
    for ep in (1, 2, 4) if ep <= experts
])
def test_balanced_materializer_equals_uncached(materialize, population, topk, experts, ep):
    args = dict(routing_token_count=population, router_topk=topk,
                total_expert_num=experts, moe_expert_parallel_size=ep)
    assert materialize(**args) == materialize.__wrapped__(**args)
    assert materialize(**args) is materialize(**args)


@pytest.mark.parametrize("materialize", MATERIALIZERS)
@pytest.mark.parametrize("field", ("routing_token_count", "router_topk", "total_expert_num", "moe_expert_parallel_size"))
@pytest.mark.parametrize("bad_type", (bool, np.int64))
def test_numeric_alias_still_rejected_after_int_hit(materialize, field, bad_type):
    args = dict(routing_token_count=1, router_topk=1, total_expert_num=1, moe_expert_parallel_size=1)
    materialize(**args)
    args[field] = bad_type(1)
    for _ in range(2):
        with pytest.raises(ValueError):
            materialize(**args)


def attention_models():
    scope = "dsv4_q_projection"
    columns = dsv4_feature_columns(scope, "decode")
    key = dsv4_model_key(scope, "csa", "FULL", "decode", False)
    models = {}
    for target, value in ((TARGET_TIME, 0.25), (TARGET_GAP, -0.1)):
        model = RandomForestRegressor(n_estimators=2, random_state=0, n_jobs=1).fit(
            pd.DataFrame([(1.0,), (8.0,)], columns=list(columns)), [value, value])
        model._frontier_feature_names = columns
        model._frontier_dsv4_bounds = {"padded_rows": (1, 8), "max_context_len": (1, 100)}
        model._frontier_exact_lookup = {(1.0,): value + 0.01}
        model._frontier_dsv4_multi_stream_gemm_token_threshold = 256
        model.predict = Mock(wraps=model.predict)
        models[dsv4_model_name(key, target)] = model
    return scope, models


def attention_call(predictor, scope, features):
    return predictor.scope_times(features, variant="csa", graph_runtime_mode="FULL",
                                 plans_tile_schedule=False, scopes=[scope])


def test_attention_interpolation_memo_is_per_instance_model_and_features():
    scope, models = attention_models()
    first, second = DeepSeekV4AttentionPredictor(models), DeepSeekV4AttentionPredictor(models)
    feature = DeepSeekV4BatchFeatures("decode", {"padded_rows": 2, "max_context_len": 10})
    a = attention_call(first, scope, feature)
    b = attention_call(first, scope, feature)
    assert a == b
    assert a.cost_ms[scope] == 0.25 and a.gap_ms[scope] == -0.1
    for model in models.values():
        assert model.predict.call_count == 1
    attention_call(first, scope, replace(feature, values={**feature.values, "padded_rows": 3}))
    attention_call(second, scope, feature)
    for model in models.values():
        assert model.predict.call_count == 3


@pytest.mark.parametrize("values", ({"padded_rows": 9, "max_context_len": 10},
                                     {"padded_rows": 2, "max_context_len": 101}))
def test_attention_coverage_runs_before_memo_on_every_call(values):
    scope, models = attention_models()
    predictor = DeepSeekV4AttentionPredictor(models)
    attention_call(predictor, scope, DeepSeekV4BatchFeatures("decode", {"padded_rows": 2, "max_context_len": 10}))
    for _ in range(2):
        with pytest.raises(DeepSeekV4CoverageError, match="outside"):
            attention_call(predictor, scope, DeepSeekV4BatchFeatures("decode", values))
    for model in models.values():
        assert model.predict.call_count == 1


def test_attention_exact_lookup_and_invalid_values_still_checked_on_memo_hits():
    scope, models = attention_models()
    predictor = DeepSeekV4AttentionPredictor(models)
    feature = DeepSeekV4BatchFeatures("decode", {"padded_rows": 1, "max_context_len": 10})
    assert attention_call(predictor, scope, feature).cost_ms[scope] == 0.26
    assert all(model.predict.call_count == 0 for model in models.values())
    for model in models.values():
        model._frontier_exact_lookup[(1.0,)] = float("nan")
    with pytest.raises(ValueError, match="Invalid V4"):
        attention_call(predictor, scope, feature)
    feature = replace(feature, values={**feature.values, "padded_rows": 2})
    time_model = models[dsv4_model_name(dsv4_model_key(scope, "csa", "FULL", "decode", False), TARGET_TIME)]
    time_model.predict.return_value = np.array([-0.1])
    for _ in range(2):
        with pytest.raises(ValueError, match="Invalid V4"):
            attention_call(predictor, scope, feature)
    assert time_model.predict.call_count == 1


def test_layer_facts_equal_uncached_for_all_layers_and_replica_configs(monkeypatch):
    model = BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V4-Flash")
    predictors = [SimpleNamespace(_model_config=model, _replica_config=SimpleNamespace(
        attn_tensor_parallel_size=tp, attn_dp=dp, num_pipeline_stages=1,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=tp * dp,
    )) for tp, dp in ((8, 2), (4, 4))]
    facts = [_layer_facts(predictor) for predictor in predictors]
    assert facts[0] is not facts[1]
    for predictor, cached in zip(predictors, facts):
        assert _layer_facts(predictor) is cached
        replica = predictor._replica_config
        profile = model.get_model_architecture_profile()
        for layer in range(model.num_layers):
            expected = frozenset(op.name for op in build_operator_manifest(model, global_layer_id=layer).operators())
            assert cached.operator_names(layer) == expected
            assert cached.shared_expert_contract(layer) == profile.resolve_layer_contract(
                model, layer_id=layer, operator_name="share_expert_up_proj",
                attention_tp_size=replica.attn_tensor_parallel_size, attention_dp_size=replica.attn_dp,
                num_pipeline_stages=replica.num_pipeline_stages,
                moe_tp_size=replica.moe_tensor_parallel_size, expert_parallel_size=replica.moe_expert_parallel_size,
            )
            for name in DEEPSEEK_V4_OPERATOR_AXES:
                assert cached.operator_tensor_parallel_size(name) == operator_tensor_parallel_size(model, replica, name)
    assert facts[0].operator_tensor_parallel_size("dsv4_lm_head") != facts[1].operator_tensor_parallel_size("dsv4_lm_head")
    # Once filled, calls do not rebuild any configuration-only facts.
    from frontier.execution_time_predictor import deepseek_v4, dsv4_operator_predictor
    monkeypatch.setattr(deepseek_v4, "build_operator_manifest", Mock(side_effect=AssertionError("rebuilt manifest")))
    monkeypatch.setattr(type(model.get_model_architecture_profile()), "resolve_layer_contract", Mock(side_effect=AssertionError("rebuilt contract")))
    monkeypatch.setattr(dsv4_operator_predictor, "operator_tensor_parallel_size", Mock(side_effect=AssertionError("rebuilt TP")))
    for cached in facts:
        for layer in range(model.num_layers):
            cached.operator_names(layer)
            cached.shared_expert_contract(layer)
            for name in DEEPSEEK_V4_OPERATOR_AXES:
                cached.operator_tensor_parallel_size(name)


@pytest.mark.parametrize("layer", (True, np.int64(1), 1.0, -1, 43))
def test_layer_fact_hits_preserve_layer_id_validation(layer):
    model = BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V4-Flash")
    predictor = SimpleNamespace(_model_config=model, _replica_config=SimpleNamespace(
        attn_tensor_parallel_size=8, attn_dp=2, num_pipeline_stages=1,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=16,
    ))
    cached = _layer_facts(predictor)
    cached.operator_names(1)
    cached.shared_expert_contract(1)
    for _ in range(2):
        with pytest.raises(ValueError):
            cached.operator_names(layer)
        with pytest.raises(ValueError):
            cached.shared_expert_contract(layer)
