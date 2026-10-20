"""CPU contracts for native Step attention profiling; no timing claims."""

import json
import sys
from contextlib import nullcontext
from itertools import product
from types import ModuleType, SimpleNamespace

import pytest

from frontier.attention.families import STEP5_ATTENTION_FAMILY
from frontier.attention.step5 import resolve_step5_attention_geometry
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.step5.inputs import Step5ProfileInput, build_profile_inputs
from frontier.profiling.step5.rows import OPERATOR_NAMES, build_row, rows_to_dataframe
from frontier.profiling.step5.scopes import NativeScopes, trace_signature
from frontier.profiling.step5.job import describe_smoke, IMAGE, CLOUD_MOUNT


@pytest.fixture(scope="module")
def config():
    return ModelConfig.from_model_name("stepfun-ai/Step-5-Preview")


@pytest.mark.parametrize("layer,tokens,prefix", list(product((0, 3), (7, 8, 9, 15, 16, 17, 511, 512, 513), (0, 4096, 4104))))
def test_native_shape_boundaries(config, layer, tokens, prefix):
    cell = Step5ProfileInput(layer, (tokens,), (prefix,), "prefill")
    workload = cell.workload(config, block_size=16, max_model_len=8192)
    geometry = resolve_step5_attention_geometry(config, layer, tensor_parallel_size=8)
    assert workload.valid_tokens == tokens
    assert workload.sequence_lengths == (prefix + tokens,)
    assert geometry.local_q_heads == 8
    assert geometry.local_kv_heads == 1
    assert geometry.kv_replication == 2
    if layer == 3:
        # The native provider-sharded indexer replicates four Q heads per
        # provider, not the naive global index_heads / TP = two heads.
        assert geometry.local_index_q_heads == 4
        assert geometry.local_fused_indexer_width == 1544


def test_smoke_grid_and_runtime_stays_disabled(config):
    cells = build_profile_inputs(config, prefill_tokens=[16, 64], prefixes=[0, 4104],
                                decode_batches=[1], decode_contexts=[512, 4104])
    assert len(cells) == 12
    assert {c.global_layer_id for c in cells} == {0, 3}
    assert STEP5_ATTENTION_FAMILY.execution_enabled is False
    assert config.get_model_architecture_profile().attention_linear_ops.profiling_enabled is False


@pytest.mark.parametrize("kwargs", [
    {"query_lens": (0,)}, {"context_lens": (-1,)}, {"context_lens": ()},
    {"phase": "mixed"}, {"phase": "decode"}, {"graph_mode": "piecewise"},
    {"graph_mode": "full"}, {"global_layer_id": True},
])
def test_invalid_cells(kwargs):
    values = dict(global_layer_id=0, query_lens=(16,), context_lens=(0,), phase="prefill")
    values.update(kwargs)
    with pytest.raises(ValueError):
        Step5ProfileInput(**values)


def fake_stats():
    return {name: dict(min=1., max=1., mean=1., median=1., std=0., count=10)
            for name in (*OPERATOR_NAMES, "step5_layer_e2e")}


def row(config):
    return build_row(config, Step5ProfileInput(3, (16,), (4104,), "prefill"),
                     tp=8, block_size=16, max_model_len=8192,
                     runtime_signature="cpu_schema_fixture_not_measurement", device="h800",
                     time_stats=fake_stats(), evidence={"attention_backend": "native_fixture"})


def test_typed_row_has_complete_family_schema(config):
    frame = rows_to_dataframe([row(config)])
    assert set(STEP5_ATTENTION_FAMILY.required_profiling_feature_columns) <= set(frame)
    contracts = json.loads(frame.iloc[0].typed_operator_contracts)
    assert set(contracts) == set(OPERATOR_NAMES)
    assert {c["selected_tensor_parallel_size"] for c in contracts.values()} == {8}
    assert {c["operator_family_id"] for c in contracts.values()} == {"step5_attention"}
    assert frame.iloc[0].local_index_q_heads == 4
    assert STEP5_ATTENTION_FAMILY.execution_enabled is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.])
def test_nonfinite_or_negative_timing_rejected(config, bad):
    result = row(config)
    result["time_stats"]["step5_qkv_projection"]["mean"] = bad
    with pytest.raises(ValueError):
        rows_to_dataframe([result])


def test_missing_active_timing_is_not_zero_filled(config):
    result = row(config)
    del result["time_stats"]["step5_sparse_attention"]
    with pytest.raises(ValueError, match="missing required"):
        rows_to_dataframe([result])


def test_trace_includes_device_copies(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": [
        {"cat": "kernel", "name": "_sparse_decode_split", "dur": 4},
        {"cat": "gpu_memcpy", "name": "Memcpy DtoD", "dur": 2},
    ]}))
    result = trace_signature(trace)
    assert result["kernel_count"] == 1
    assert result["memory_copy_count"] == 1
    assert result["memory_copy_time_ms"] == .002


def test_smoke_description_uses_locked_image_and_personal_cloud_root(tmp_path):
    job = describe_smoke(tmp_path / "stage", "attention_smoke_cpu_fixture")
    assert job["image"] == IMAGE
    assert IMAGE.endswith("sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20")
    assert CLOUD_MOUNT in job["mounts"]
    assert job["charged_group"] == "codesign" and job["gpu_tag"] == "H800"
    assert job["cloud_output"].startswith("/mnt/codesign-exp/ycfeng/")
    assert job["expected_rows"] == 16
    assert job["gpus"] == 1
    assert "--tensor-parallel-size 8" in job["worker_command"]
    assert "submit_profiling_job.py" in job["submission_command"]


@pytest.mark.parametrize("run_id", ["", "../escape", "/absolute", ".", ".."])
def test_smoke_rejects_cloud_path_escape(tmp_path, run_id):
    with pytest.raises(ValueError):
        describe_smoke(tmp_path, run_id)


def test_native_metadata_preserves_request_page_ids_between_prefix_and_continuation(monkeypatch):
    torch = pytest.importorskip("torch")
    from frontier.profiling.step5.vllm_wrapper import VllmStep5AttentionWrapper

    arange = torch.arange
    monkeypatch.setattr(torch, "arange", lambda *a, **k: arange(*a, **{**k, "device": "cpu"}))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    backend = ModuleType("vllm.v1.attention.backend")
    backend.CommonAttentionMetadata = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backend", backend)
    wrapper = VllmStep5AttentionWrapper.__new__(VllmStep5AttentionWrapper)
    wrapper.torch, wrapper.block_size, wrapper.max_model_len = torch, 16, 8192
    wrapper.native = object()
    wrapper._set_current = lambda _: nullcontext()
    wrapper.builder = SimpleNamespace(build=lambda _, common, **kw: common)
    prefix = Step5ProfileInput(3, (512, 512), (0, 0), "prefill")
    current = Step5ProfileInput(3, (16, 16), (512, 512), "prefill")
    _, _, p, _ = wrapper._metadata(prefix)
    positions, slots, c, _ = wrapper._metadata(current)
    assert p.block_table_tensor[1, 0] == c.block_table_tensor[1, 0] == 513
    assert c.query_start_loc_cpu.tolist() == [0, 16, 32]
    assert c.seq_lens_cpu_upper_bound.tolist() == [528, 528]
    assert positions.tolist() == list(range(512, 528)) * 2
    assert slots.tolist() == list(range(528, 544)) + list(range(8720, 8736))


def test_nested_native_scopes_charge_only_the_outer_fused_owner():
    events = []

    class Event:
        def __init__(self, **kwargs):
            events.append(kwargs)

        def record(self):
            pass

        def elapsed_time(self, other):
            return 2.

    fake = SimpleNamespace(cuda=SimpleNamespace(Event=Event),
                           profiler=SimpleNamespace(record_function=lambda _: nullcontext()))
    scopes = NativeScopes(fake)
    module = SimpleNamespace(inner=lambda: 3)
    module.outer = lambda: module.inner()
    with scopes.observe([(module, "outer", "fused"), (module, "inner", "indexer")]):
        assert module.outer() == 3
    assert scopes.times() == {"fused": 2.}
    assert len(events) == 2
    assert not scopes.enabled


def test_one_token_prefill_is_rejected():
    with pytest.raises(ValueError, match="native one-token"):
        Step5ProfileInput(3, (1,), (512,), "prefill")


def test_dry_run_grid_does_not_import_native_runtime(tmp_path, monkeypatch, capsys):
    from frontier.profiling.step5.main import main

    monkeypatch.setitem(sys.modules, "vllm", None)
    main(["--model-path", str(tmp_path / "no_model_needed"), "--max-model-len", "8192",
          "--output-dir", str(tmp_path / "no_output_created"), "--dry-run",
          "--prefixes", "0", "4104", "--graph-modes", "none", "full"])
    result = json.loads(capsys.readouterr().out)
    assert result["rows"] == 16
    assert not (tmp_path / "no_output_created").exists()


def test_empty_profile_grid_is_rejected(config):
    with pytest.raises(ValueError, match="must contain"):
        # Keep workload admission explicit before the GPU wrapper exists.
        build_profile_inputs(config, prefill_tokens=[], prefixes=[],
                             decode_batches=[], decode_contexts=[])


@pytest.mark.parametrize("failure", ["whole_missing", "zero_mean", "short_count", "mismatched_count"])
def test_whole_layer_and_sample_count_contract(config, failure):
    result = row(config)
    stats = result["time_stats"]
    if failure == "whole_missing":
        del stats["step5_layer_e2e"]
    elif failure == "zero_mean":
        stats["step5_layer_e2e"]["mean"] = 0
    elif failure == "short_count":
        for timing in stats.values():
            timing["count"] = 1
    else:
        stats["step5_sparse_attention"]["count"] = 3
    with pytest.raises(ValueError):
        rows_to_dataframe([result])


@pytest.fixture
def fake_native_wrapper(config, monkeypatch):
    constructed = []

    class Wrapper:
        def __init__(self, **kwargs):
            self.args = kwargs
            constructed.append((kwargs["layer_id"], kwargs["graph_mode"]))

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def profile(self, cell, **kwargs):
            assert cell.graph_mode == self.args["graph_mode"]
            return build_row(config, cell, tp=8, block_size=16, max_model_len=8192,
                runtime_signature="cpu_schema_fixture_not_measurement", device="h800",
                time_stats=fake_stats(), evidence={"attention_backend": "native_fixture",
                "stability_pass": True, "time_cv": 0., "trace_path": "cpu_fixture_no_trace"})

    monkeypatch.setattr("frontier.profiling.step5.vllm_wrapper.VllmStep5AttentionWrapper", Wrapper)
    return constructed


def small_main_args(tmp_path):
    return ["--model-path", str(tmp_path / "model"), "--max-model-len", "8192",
            "--output-dir", str(tmp_path / "output"), "--prefill-tokens", "16",
            "--prefixes", "0", "--decode-contexts", "512", "--graph-modes", "none", "full"]


def test_main_separates_native_graph_policies_and_writes_complete_summary(tmp_path, fake_native_wrapper):
    from frontier.profiling.step5.main import main

    main(small_main_args(tmp_path))
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    assert summary["expected_rows"] == summary["actual_rows"] == summary["stable_rows"] == 6
    assert summary["complete"] and not summary["admitted"]
    assert set(fake_native_wrapper) == {(0, "none"), (0, "full"), (3, "none"), (3, "full")}


def test_failed_csv_write_is_not_a_completed_row(tmp_path, fake_native_wrapper, monkeypatch):
    import pandas as pd
    from frontier.profiling.step5.main import main

    def unavailable(*args, **kwargs):
        raise OSError("CPU test injected output failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", unavailable)
    with pytest.raises(OSError, match="injected output failure"):
        main(small_main_args(tmp_path))
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    assert summary["actual_rows"] == 0
    assert not summary["complete"]
