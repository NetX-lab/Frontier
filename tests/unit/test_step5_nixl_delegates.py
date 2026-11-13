"""Lazy integration with the sole externally owned NIXL measured-cost loader."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.execution_time_predictor.cache_io import dataset_fingerprint
from frontier.execution_time_predictor.prediction_model_registry import PredictionModelRegistry
from frontier.types import ClusterType


class Predictor:
    _get_nixl_read_cost = SklearnExecutionTimePredictor._get_nixl_read_cost
    get_nixl_transfer_coverage = SklearnExecutionTimePredictor.get_nixl_transfer_coverage
    get_nixl_transfer_time_ms = SklearnExecutionTimePredictor.get_nixl_transfer_time_ms


@pytest.mark.parametrize("config", [SimpleNamespace(), SimpleNamespace(nixl_read_profile=None)])
def test_missing_explicit_profile_does_not_import_loader(config, monkeypatch):
    monkeypatch.delitem(sys.modules, "frontier.kv_cache_transfer.nixl_read_cost", raising=False)
    predictor = Predictor()
    predictor._config = config
    with pytest.raises(ValueError, match="^Missing measured NIXL profile$"):
        predictor.get_nixl_transfer_coverage()
    assert "frontier.kv_cache_transfer.nixl_read_cost" not in sys.modules


def test_both_delegates_share_one_lazy_loader_and_preserve_arguments(monkeypatch):
    calls = []
    cost = SimpleNamespace(
        get_coverage=lambda **kwargs: calls.append(("coverage", kwargs)) or "covered",
        get_time_ms=lambda **kwargs: calls.append(("time", kwargs)) or 1.25,
    )
    module = ModuleType("frontier.kv_cache_transfer.nixl_read_cost")
    module.NixlReadCost = SimpleNamespace(
        from_file=lambda path: calls.append(("load", path)) or cost,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    predictor = Predictor()
    predictor._config = SimpleNamespace(nixl_read_profile="explicit_fixture.json")
    assert predictor.get_nixl_transfer_coverage(source="p", target="d") == "covered"
    assert predictor.get_nixl_transfer_time_ms(coverage="covered", payload=123) == 1.25
    assert calls == [
        ("load", "explicit_fixture.json"),
        ("coverage", {"source": "p", "target": "d"}),
        ("time", {"coverage": "covered", "payload": 123}),
    ]


@pytest.mark.parametrize("change", ["bytes", "path", "unbound"])
def test_first_lazy_load_reuses_published_input_binding_validation(tmp_path, monkeypatch, change):
    profile = tmp_path / "binding_fixture.json"
    profile.write_text('{"evidence_kind":"cpu_fixture","version":"A"}')
    binding = {"path": str(profile), "sha256": dataset_fingerprint(profile)}
    identity = {"input_bindings": {"nixl_read_profile": binding}}
    predictor = Predictor()
    predictor._config = SimpleNamespace(nixl_read_profile=str(profile))
    predictor._cluster_type = ClusterType.PREFILL
    predictor._model_manager = SimpleNamespace(
        _step5_model_bundles={ClusterType.PREFILL: {"kernel_only": {"identity": identity}}},
        _validate_bundle_input_bindings=PredictionModelRegistry._validate_bundle_input_bindings,
    )
    module = ModuleType("frontier.kv_cache_transfer.nixl_read_cost")
    module.NixlReadCost = SimpleNamespace(from_file=lambda *_: pytest.fail("read unbound input"))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    if change == "bytes":
        profile.write_text('{"evidence_kind":"cpu_fixture","version":"B"}')
        message = "input binding checksum mismatch"
    else:
        if change == "path":
            other = tmp_path / "other.json"
            other.write_text(profile.read_text())
            predictor._config.nixl_read_profile = str(other)
        else:
            identity["input_bindings"].clear()
        message = "NIXL profile conflicts with bound model input"
    with pytest.raises(ValueError, match=message):
        predictor.get_nixl_transfer_coverage()
