"""CPU acceptance of G2h grids, contracts, producer iteration and CSV metadata."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from frontier.operators.deepseek_v4_axes import DEEPSEEK_V4_OPERATOR_AXES
from frontier.operators.families import DEEPSEEK_V4_MHC_FAMILY
from frontier.operators.typed_contracts import (
    parse_typed_operator_contracts, validate_typed_operator_contracts,
)
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.linear_op import main
from frontier.profiling.linear_op.deepseek_v4_plan import (
    build_operator_plans, load_operator_grid, operator_shapes, row_contracts,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g2h"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"


@pytest.fixture
def model():
    return ModelConfig.from_model_name(MODEL)


def args(monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["linear_op", "--models", MODEL,
                        "--deepseek_v4_operator_grid", str(PACKAGE / "operator_grid.json"),
                        "--disable_ray", "--num_gpus", "1", "--dry_run", *extra])
    return main.parse_args()


def test_axis_table_matches_registry_and_is_immutable():
    names = {op.name for op in DEEPSEEK_V4_MHC_FAMILY.operators}
    assert names == {name for name in DEEPSEEK_V4_OPERATOR_AXES if name.startswith("dsv4_")}
    assert len(names) == 8
    assert all(op.profiling_target for op in DEEPSEEK_V4_MHC_FAMILY.operators)
    assert DEEPSEEK_V4_MHC_FAMILY.execution_enabled is False
    assert {DEEPSEEK_V4_OPERATOR_AXES[n] for n in names} == {
        "sp_local_rows", "descriptor_rows", "logits_rows"}
    with pytest.raises(TypeError):
        DEEPSEEK_V4_OPERATOR_AXES["dsv4_mhc_head"] = "sp_local_rows"


@pytest.mark.parametrize("field,value,message", [
    ("attention_tp_size", True, "positive integer"),
    ("attention_tp_size", 0, "positive integer"),
    ("sp_local_rows", [], "non-empty"),
    ("sp_local_rows", [True], "positive integer"),
    ("descriptor_rows", [1, 1], "duplicate"),
    ("logits_rows", [1.5], "positive integer"),
    ("logits_rows", [0], "positive integer"),
    ("logits_rows", "1,2", "non-empty"),
])
def test_grid_rejects_invalid_axes(tmp_path, field, value, message):
    grid = load_operator_grid(PACKAGE / "operator_grid.json")
    grid[field] = value
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(grid))
    with pytest.raises(ValueError, match=message):
        load_operator_grid(path)


@pytest.mark.parametrize("value", [{}, [], {"attention_tp_size": 8, "unknown": []}])
def test_grid_requires_exact_schema(tmp_path, value):
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="requires exactly"):
        load_operator_grid(path)


def test_grid_uses_architecture_profile_and_validates_vocab(model):
    grid = load_operator_grid(PACKAGE / "operator_grid.json")
    model.name = "a-name-that-does-not-mention-v4"
    assert len(build_operator_plans(model, grid)) == 3
    grid["attention_tp_size"] = 3
    with pytest.raises(ValueError, match="vocab_size"):
        build_operator_plans(model, grid)


@pytest.mark.parametrize("extra,message", [
    (["--num_tokens_list", "1"], "cannot be combined with --num_tokens_list"),
    (["--sequence_parallel_moe_topology", "8", "2", "16"], "cannot be combined with --sequence_parallel_moe_topology"),
    (["--models", "zai-org/GLM-5.3"], "requires a DeepSeek-V4 architecture profile"),
])
def test_grid_rejected_combinations(monkeypatch, extra, message):
    parsed = args(monkeypatch, *extra)
    with pytest.raises(ValueError, match=message):
        main._dry_run(parsed, [1])


def test_grid_dry_run_exact_rows_and_typed_contracts(monkeypatch, model):
    parsed = args(monkeypatch)
    summary = main._dry_run(parsed, [999])
    points = summary["models"][MODEL]
    assert points["profiled_steps"] == 36
    assert [p["num_token_points"] for p in points["plans"]] == [5, 7, 24]
    actual = [{"plan": p["plan_name"], "operator": name, "axis": p["row_axis"],
               "rows": p["num_tokens"], "tp_size": contract["selected_tensor_parallel_size"]}
              for p in points["plans"] for name, contract in p["operator_contracts"].items()]
    expected = json.loads((PACKAGE / "expected_rows.json").read_text())
    assert actual == expected["phases"]["G2H_OPERATORS_CUDA_EVENT"]["plans"]
    for plan in build_operator_plans(model, load_operator_grid(PACKAGE / "operator_grid.json")):
        contracts = row_contracts(model, plan["typed_operator_contracts"], plan["num_tokens"][0])
        assert parse_typed_operator_contracts(contracts) == contracts
        assert {c["row_axis"] for c in contracts.values()} == {plan["row_axis"]}
        assert {c["selected_tensor_parallel_size"] for c in contracts.values()} <= {1, 8}


def test_existing_plan_owns_cli_rows_without_new_summary_fields(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["linear_op", "--models", "zai-org/GLM-5.3",
                        "--num_tokens_list", "1", "2", "768", "--dry_run"])
    parsed = main.parse_args()
    plans = main._build_profiling_plans(parsed, ModelConfig.from_model_name(parsed.models[0]))
    assert all(p["num_tokens"] == [768, 2, 1] for p in plans)
    assert all("num_tokens" not in main._plan_summary(p, [999]) for p in plans)
    assert all(main._plan_summary(p, [999])["num_token_points"] == 3 for p in plans)


@pytest.mark.parametrize("descriptor_rows,logits_rows", [(24, 3), (6144, 1)])
def test_shapes_equal_consumer_from_same_batch(model, descriptor_rows, logits_rows):
    from test_dsv4_source_execution import predictor, source_batch, component

    consumer = predictor.__wrapped__()
    batch = source_batch(descriptor_rows)
    results = [component(consumer, batch, layer=layer,
                         selected_logits_rows=logits_rows if layer == 42 else None)
               for layer in (0, 1, 42)]
    workloads = {name: shape for result in results for name, shape in result["workloads"].items()}
    local = (descriptor_rows + 7) // 8
    for name, axis in DEEPSEEK_V4_OPERATOR_AXES.items():
        rows = {"descriptor_rows": descriptor_rows, "sp_local_rows": local,
                "logits_rows": logits_rows}[axis]
        shape = operator_shapes(model, name, rows, 8 if name == "dsv4_lm_head" else 1)
        assert shape["input_shape"] == workloads[name]["input_shape"], name
        assert shape["output_shape"] == workloads[name]["output_shape"], name
        for field in ("residual_shape", "post_mix_shape", "res_mix_shape", "weight_tp"):
            if field in workloads[name]:
                assert shape[field] == workloads[name][field], (name, field)
    assert operator_shapes(model, "dsv4_lm_head", logits_rows, 8)["local_output_shape"] == (logits_rows, model.vocab_size // 8)


def test_profile_model_iterates_each_plans_rows_and_preserves_them(monkeypatch, tmp_path, model):
    seen = []

    class Wrapper:
        def __init__(self, **kwargs):
            self.plan = kwargs["profiling_plan"]

        def profile(self, rows):
            seen.append((self.plan["row_axis"], rows))
            return {"time_stats": {name: {"median": 1.0} for name in self.plan["enabled_ops"]},
                    "num_tokens": rows, "num_tensor_parallel_workers": self.plan["tp_size"],
                    "row_axis": self.plan["row_axis"],
                    "typed_operator_contracts": row_contracts(model, self.plan["typed_operator_contracts"], rows)}

    module = SimpleNamespace(linear_op_wrapper_class=lambda plan: Wrapper)
    monkeypatch.setitem(sys.modules, "frontier.profiling.linear_op.linear_op_wrapper", module)
    monkeypatch.setattr(main, "_get_available_gpus", lambda n: [0])
    monkeypatch.setattr(main, "_resolve_precision_for_model", lambda *a: (None, "BF16"))
    monkeypatch.setattr(main, "_ensure_torch_available", lambda: SimpleNamespace(cuda=SimpleNamespace(set_device=lambda n: None)))
    monkeypatch.setattr(main, "set_process_visible_device", lambda *a, **k: None)
    parsed = args(monkeypatch, "--output_dir", str(tmp_path), "--profile_method", "cuda_event")
    progress = SimpleNamespace(count=0)
    def update(n):
        progress.count += n
    progress.update = update
    frame = main.profile_model(parsed, MODEL, [999], progress)
    grid = load_operator_grid(PACKAGE / "operator_grid.json")
    assert seen == [(axis, rows) for axis in ("sp_local_rows", "descriptor_rows", "logits_rows")
                    for rows in grid[axis]]
    assert progress.count == len(frame) == 36
    attached = main._attach_linear_op_output_metadata(
        frame, precision_str="BF16", model_arch=model.model_arch,
        model_architecture_profile="deepseek_v4", quant_signature=model.get_quant_signature(),
        measurement_type="CUDA_EVENT")
    serialized = main._serialize_linear_op_output(attached)
    assert all(json.loads(value) for value in serialized["typed_operator_contracts"])


def test_builder_check_plan_matches_all_expected_rows():
    import runpy

    builder = runpy.run_path(str(PACKAGE / "prepare_package.py"))
    spec = json.loads((PACKAGE / "package_plan.json").read_text())
    # Check every actual invocation, including unchanged shared CLI row ordering.
    assert len(builder["check_plan"](spec, ROOT)) == 6


def test_new_modules_import_without_worker_packages():
    # Imports themselves cannot load torch/vLLM; runtime-boundary tests also
    # exercise the main entrypoint in a fresh process.
    for name in ("deepseek_v4_plan", "deepseek_v4_operators", "fp8_quantization"):
        module = importlib.import_module("frontier.profiling.linear_op." + name)
        assert module is not None


def test_canonical_admission_is_blocked_by_disabled_family(model):
    plan = build_operator_plans(model, load_operator_grid(PACKAGE / "operator_grid.json"))[0]
    contracts = row_contracts(model, plan["typed_operator_contracts"], 1)
    with pytest.raises(NotImplementedError, match="not enabled for execution"):
        validate_typed_operator_contracts(contracts)
