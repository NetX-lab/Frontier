"""CPU tests for the DeepSeek-V4 attention profiling plan, hooks, and rows."""

from __future__ import annotations

import csv
import functools
import json
import sys
import types
from types import SimpleNamespace

import pytest

from frontier.attention.families import DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT
from frontier.profiling.attention import dsa_vllm_runner as case_runner
from frontier.profiling.attention import dsv4_attention_runner as dsv4_runner
from frontier.profiling.attention.dsa_plan import (
    PHASE_DECODE,
    PHASE_PREFILL,
    DsaMeasurementLimits,
    load_hf_config,
)
from frontier.profiling.attention.dsa_rows import CUDA_EVENT, KERNEL_ONLY
from frontier.profiling.attention.dsv4_attention_main import main as dsv4_main
from frontier.profiling.attention.dsv4_attention_plan import (
    DSV4_SCOPES,
    LAYER_SCOPE,
    METADATA_SOURCES_BY_VARIANT,
    WARMUP_ONLY_OPERATORS,
    build_dsv4_hf_overrides,
    build_dsv4_profile_plan,
    expected_scopes,
    metadata_key,
    scope_operators,
)
from frontier.profiling.attention.dsv4_attention_rows import build_dsv4_rows
from frontier.profiling.common.runtime_identity import IDENTITY_COLUMNS

MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
CONFIG_PATH = "data/config/models/deepseek-ai__DeepSeek-V4-Flash.json"
PLAN_ARGS = [
    "--model_name", MODEL_NAME,
    "--model_config_path", CONFIG_PATH,
    "--target_tensor_parallel_size", "8",
    "--block_size", "256",
    "--kv_cache_dtype", "fp8",
    "--max_context_len", "4160",
    "--max_kv_tokens_per_lane", "20000",
    "--max_num_seqs", "8",
    "--chunk_budgets", "1024", "256",
]
VARIANTS = ("uncompressed", "csa", "hca")


@pytest.fixture
def native_model_module(monkeypatch):
    for name in ("vllm", "vllm.models", "vllm.models.deepseek_v4"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    return sys.modules["vllm.models.deepseek_v4"]


def test_pristine_dummy_finalizer_is_temporary_and_loader_owned(native_model_module):
    calls = []

    class PristineModel:
        def __init__(self):
            self.model = SimpleNamespace(
                finalize_mega_moe_weights=lambda: calls.append("mega"),
                finalize_mhc_broadcast_weights=lambda: calls.append("mhc"),
            )

    native_model_module.DeepseekV4ForCausalLM = PristineModel
    for _ in range(2):
        before = len(calls)
        with dsv4_runner.patch_dummy_weight_finalizers():
            assert len(calls) == before
            PristineModel().process_weights_after_loading()
            assert calls[before:] == ["mega", "mhc"]
        assert "process_weights_after_loading" not in PristineModel.__dict__


@pytest.mark.parametrize("already_initialized", [False, True])
def test_accepted_c26_finalizer_is_unchanged_and_called_once(
    native_model_module, already_initialized
):
    calls = []
    existing_weights = object()

    class Core:
        weights = existing_weights if already_initialized else None

        def finalize_mhc_broadcast_weights(self, *, only_if_missing):
            calls.append(only_if_missing)
            if not only_if_missing or self.weights is None:
                self.weights = object()

    class AcceptedC26Model:
        def process_weights_after_loading(self) -> None:
            self.model.finalize_mhc_broadcast_weights(only_if_missing=True)

    native_model_module.DeepseekV4ForCausalLM = AcceptedC26Model
    original = AcceptedC26Model.process_weights_after_loading
    model = AcceptedC26Model()
    model.model = Core()
    with dsv4_runner.patch_dummy_weight_finalizers():
        assert AcceptedC26Model.process_weights_after_loading is original
        assert calls == []
        model.process_weights_after_loading()
    assert AcceptedC26Model.process_weights_after_loading is original
    assert calls == [True]
    assert model.model.weights is not None
    if already_initialized:
        assert model.model.weights is existing_weights


@pytest.mark.parametrize("accepted_c26", [False, True])
def test_dummy_finalizer_restoration_after_load_failure(native_model_module, accepted_c26):
    class Model:
        pass

    if accepted_c26:
        def process_weights_after_loading(self) -> None:
            self.model.finalize_mhc_broadcast_weights(only_if_missing=True)

        Model.process_weights_after_loading = process_weights_after_loading
    native_model_module.DeepseekV4ForCausalLM = Model
    original = Model.__dict__.get("process_weights_after_loading")
    with pytest.raises(RuntimeError, match="dummy load failed"):
        with dsv4_runner.patch_dummy_weight_finalizers():
            raise RuntimeError("dummy load failed")
    assert Model.__dict__.get("process_weights_after_loading") is original


def test_unknown_dummy_finalizer_is_rejected_without_mutation(native_model_module):
    class UnknownModel:
        def process_weights_after_loading(self) -> None:
            self.model.finalize_mhc_broadcast_weights(only_if_missing=False)

    native_model_module.DeepseekV4ForCausalLM = UnknownModel
    original = UnknownModel.process_weights_after_loading
    with pytest.raises(RuntimeError, match="Unapproved"):
        with dsv4_runner.patch_dummy_weight_finalizers():
            pytest.fail("An unapproved hook entered the loader scope")
    assert UnknownModel.process_weights_after_loading is original


def test_wrapped_c26_hook_is_not_mistaken_for_approved_hook(native_model_module):
    def process_weights_after_loading(self) -> None:
        self.model.finalize_mhc_broadcast_weights(only_if_missing=True)

    @functools.wraps(process_weights_after_loading)
    def wrapped(self):
        process_weights_after_loading(self)
        process_weights_after_loading(self)

    model = type("WrappedModel", (), {"process_weights_after_loading": wrapped})
    native_model_module.DeepseekV4ForCausalLM = model
    with pytest.raises(RuntimeError, match="Unapproved"):
        with dsv4_runner.patch_dummy_weight_finalizers():
            pytest.fail("A wrapper that calls C26 twice was admitted")
    assert model.process_weights_after_loading is wrapped


def test_run_engine_restores_pristine_hook_when_constructor_fails(
    native_model_module, monkeypatch, tmp_path
):
    calls = []

    class Model:
        model = SimpleNamespace(
            finalize_mega_moe_weights=lambda: calls.append("mega"),
            finalize_mhc_broadcast_weights=lambda: calls.append("mhc"),
        )

    native_model_module.DeepseekV4ForCausalLM = Model

    def fail_constructor(*args, **kwargs):
        assert "process_weights_after_loading" in Model.__dict__
        Model().process_weights_after_loading()
        raise RuntimeError("constructor failed")

    monkeypatch.setattr(case_runner, "build_engine", fail_constructor)
    with pytest.raises(RuntimeError, match="constructor failed"):
        dsv4_runner.run_engine(
            plan=SimpleNamespace(kv_cache_dtype="fp8"), engine_index=0,
            model_dir="unused", vocab_size=128, seed=0,
            gpu_memory_utilization=0.85,
            expected_attention_backend="FLASHMLA_SPARSE_DSV4",
            raw_path=tmp_path / "raw.jsonl", trace_dir=tmp_path / "traces",
            keep_traces=True,
        )
    assert calls == ["mega", "mhc"]
    assert "process_weights_after_loading" not in Model.__dict__


def _limits(**overrides):
    values = dict(
        max_context_len=4160,
        max_kv_tokens_per_lane=20000,
        max_num_seqs=8,
        chunk_budgets=(1024, 256),
        block_size=256,
        decode_warmup_steps=2,
        decode_measured_steps=3,
        prefill_warmup_repeats=1,
        prefill_measured_repeats=2,
    )
    values.update(overrides)
    return DsaMeasurementLimits(**values)


def _build_plan(hf_config, num_layers=None, **limit_overrides):
    return build_dsv4_profile_plan(
        model_name=MODEL_NAME,
        hf_config=hf_config,
        target_tensor_parallel_size=8,
        num_layers=num_layers,
        limits=_limits(**limit_overrides),
        kv_cache_dtype="fp8",
    )


@pytest.fixture(scope="module")
def hf_config():
    return load_hf_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def plan(hf_config):
    return _build_plan(hf_config)


def test_reduced_model_holds_two_layers_per_variant(plan, hf_config):
    layers = [
        (l.global_layer_id, l.attention_variant, l.compress_ratio, l.plans_tile_schedule)
        for l in plan.shape.layers
    ]
    assert layers == [
        (0, "uncompressed", 0, True), (1, "uncompressed", 0, False),
        (2, "csa", 4, True), (3, "hca", 128, True),
        (4, "csa", 4, False), (5, "hca", 128, False),
    ]
    assert plan.hf_overrides == {
        "num_hidden_layers": 6,
        "num_attention_heads": hf_config["num_attention_heads"] // 8,
        "o_groups": hf_config["o_groups"] // 8,
        "compress_ratios": hf_config["compress_ratios"][:6],
    }
    shape = plan.shape
    assert (shape.heads_per_rank, shape.o_groups_per_rank) == (8, 1)
    assert (shape.csa_compress_ratio, shape.hca_compress_ratio) == (4, 128)
    assert shape.dtype == hf_config["torch_dtype"]
    assert shape.short_context_max_seq_len == (hf_config["index_topk"] + 1) * 4 - 1
    assert plan.kv_cache_dtype == "fp8"


def test_explicit_num_layers_must_hold_every_variant_twice(hf_config):
    # Layers 0-3 hold one csa and one hca layer.
    with pytest.raises(ValueError, match="needs 2 'csa' layers"):
        _build_plan(hf_config, num_layers=4)
    with pytest.raises(ValueError, match="needs 2 'hca' layers"):
        _build_plan(hf_config, num_layers=5)
    assert len(_build_plan(hf_config, num_layers=8).shape.layers) == 8


def test_plan_rejects_ratios_that_disagree_with_frontier(hf_config):
    config = dict(hf_config)
    config["compress_ratios"] = list(hf_config["compress_ratios"])
    config["compress_ratios"][2] = 128
    with pytest.raises(ValueError, match="Layer 2: native compress ratio 128"):
        _build_plan(config)
    config["compress_ratios"] = hf_config["compress_ratios"][:5]
    with pytest.raises(ValueError, match="one ratio per decoder layer"):
        _build_plan(config)


def test_overrides_reject_unknown_per_layer_lists(hf_config, plan):
    config = dict(hf_config)
    config["unknown_layer_types"] = ["x"] * hf_config["num_hidden_layers"]
    with pytest.raises(ValueError, match="unknown_layer_types"):
        build_dsv4_hf_overrides(config, plan.shape.layers, 8, 1)


def test_tensor_parallel_size_must_divide_heads_and_output_groups(hf_config):
    with pytest.raises(ValueError, match="o_groups=8 is not divisible"):
        build_dsv4_profile_plan(
            model_name=MODEL_NAME, hf_config=hf_config, target_tensor_parallel_size=16,
            num_layers=None, limits=_limits(), kv_cache_dtype="fp8",
        )


def test_grid_straddles_the_short_context_boundary(plan, hf_config):
    boundary = plan.shape.short_context_max_seq_len
    wide = _build_plan(hf_config, chunk_budgets=(256, 4096))
    prompts = {c.prompt_len for c in wide.cases if c.phase == PHASE_PREFILL}
    assert {boundary - 1, boundary, boundary + 1} <= prompts
    decode_contexts = {c.prompt_len for c in plan.cases if c.phase == PHASE_DECODE}
    assert boundary + 1 in decode_contexts
    assert not plan.shape.index_scores_executed(boundary)
    assert plan.shape.index_scores_executed(boundary + 1)


def test_scopes_cover_every_reachable_family_operator():
    for variant in VARIANTS:
        covered = [
            operator
            for scope in DSV4_SCOPES
            if variant in scope.variants
            for operator in scope_operators(scope.name, variant)
        ]
        assert len(covered) == len(set(covered)), variant
        family = {op.name for op in DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT[variant]}
        assert set(covered) == family - set(WARMUP_ONLY_OPERATORS), variant
    every_operator = {op.name for op in DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT["csa"]}
    for scope in DSV4_SCOPES:
        assert set(scope.operators) <= every_operator, scope.name


def test_expected_scopes_follow_variant_and_context():
    assert expected_scopes("uncompressed", True, annotated_only=True) == (
        "dsv4_parallel_input_projections", "dsv4_fused_qkv_rmsnorm", "dsv4_q_projection",
        "dsv4_fused_q_kv_prepare", "dsv4_sparse_attention", "dsv4_o_proj",
    )
    hca = expected_scopes("hca", True, annotated_only=False)
    assert hca[0] == LAYER_SCOPE and "dsv4_compressor" in hca
    assert "dsv4_index_prepare" not in hca
    short = expected_scopes("csa", False, annotated_only=True)
    long = expected_scopes("csa", True, annotated_only=True)
    assert set(long) - set(short) == {"dsv4_index_scores_topk"}


def test_plan_dry_run_cli(tmp_path, capsys):
    output = tmp_path / "plan.json"
    dsv4_main(["plan", *PLAN_ARGS, "--output", str(output)])
    summary = json.loads(capsys.readouterr().out.split("[dsv4]")[0])
    full = json.loads(output.read_text())
    assert summary["num_cases"] == len(full["cases"])
    assert summary["num_engines"] == 2
    assert summary["kv_cache_dtype"] == "fp8"
    assert full["shape"]["short_context_max_seq_len"] == 2051


# ---------------------------------------------------------------------------
# Raw records and rows


def _engine_record(engine_index=1, chunk_budget=1024):
    return {
        "kind": "engine",
        "engine_index": engine_index,
        "chunk_budget": chunk_budget,
        "kv_cache_dtype": "fp8",
        "runtime_versions": {"vllm": "0.28.0", "torch": "2.9.0"},
        "device_name": "NVIDIA H800",
        "device_capability": "sm90",
        "attention_backend": "FLASHMLA_SPARSE_DSV4",
        "attention_impl": "DeepseekV4FlashMLAAttention",
        "swa_backend": "DEEPSEEK_SPARSE_SWA",
        "indexer_backend": "DEEPSEEK_V4_INDEXER",
        "compressor_backend": "CompressorBackend",
        "attention_kv_cache_dtype": "fp8_ds_mla",
        "indexer_uses_fp4_kv": False,
        "padded_heads": 64,
        "aux_stream_count": 3,
        "multi_stream_gemm_token_threshold": 1024,
        "layers": [],
    }


def _metadata(requests):
    tokens = sum(r["num_scheduled_tokens"] for r in requests)
    max_seq_len = max(r["num_computed_tokens"] + r["num_scheduled_tokens"] for r in requests)
    prefills = sum(r["num_computed_tokens"] < r["prompt_len"] for r in requests)
    swa = {"block_size": 64, "num_decodes": len(requests) - prefills, "num_prefills": prefills}
    attention = {"max_seq_len": max_seq_len, "num_actual_tokens": tokens}
    compressor = {"block_size": 64, "num_decode_tokens": None, "c128_boundary": None}
    indexer = {"max_seq_len": max_seq_len, "num_prefill_tokens": tokens}
    by_source = {"swa": swa, "attention": attention, "compressor": compressor,
                 "indexer": indexer, "index_compressor": compressor}
    return {
        metadata_key(variant, source): by_source[source]
        for variant in VARIANTS
        for source in METADATA_SOURCES_BY_VARIANT[variant]
    }


def _step_records(plan, measurement, step_index, requests, case_id, phase, time_ms,
                  kernels=7, drop=None):
    step = {
        "kind": "step", "measurement": measurement, "engine_index": 1,
        "case_id": case_id, "phase": phase, "step_index": step_index,
        "requests": requests, **_metadata(requests),
    }
    long_context = plan.shape.index_scores_executed(
        step[metadata_key("csa", "indexer")]["max_seq_len"])
    records = [step]
    for layer in plan.shape.layers:
        scopes = expected_scopes(layer.attention_variant, long_context,
                                 annotated_only=measurement == KERNEL_ONLY)
        for scope in scopes:
            if (layer.global_layer_id, scope) == drop:
                continue
            records.append({
                "kind": "call", "measurement": measurement, "engine_index": 1,
                "case_id": case_id, "step_index": step_index,
                "global_layer_id": layer.global_layer_id, "scope": scope,
                "time_ms": time_ms + layer.global_layer_id,
                "kernel_count": kernels if measurement == KERNEL_ONLY else None,
            })
    return records


def _prefill_request(q, ctx, prompt=4159):
    return [{"num_scheduled_tokens": q, "num_computed_tokens": ctx, "prompt_len": prompt}]


def test_rows_split_variant_and_tile_schedule_role(plan):
    records = [_engine_record()]
    for step_index in (0, 1):
        records += _step_records(plan, CUDA_EVENT, step_index, _prefill_request(1024, 2048),
                                 "e1-prefill-p4159", PHASE_PREFILL, 1.0)
    records += _step_records(plan, KERNEL_ONLY, 0, _prefill_request(1024, 2048),
                             "e1-prefill-p4159", PHASE_PREFILL, 0.5)
    rows = build_dsv4_rows(plan, records)
    assert len(rows[CUDA_EVENT]) == 6 and len(rows[KERNEL_ONLY]) == 6
    keys = [(r["attention_variant"], r["plans_tile_schedule"], r["global_layer_ids"])
            for r in rows[CUDA_EVENT]]
    assert keys == [
        ("csa", True, "[2]"), ("csa", False, "[4]"), ("hca", True, "[3]"),
        ("hca", False, "[5]"), ("uncompressed", True, "[0]"), ("uncompressed", False, "[1]"),
    ]
    csa = rows[CUDA_EVENT][0]
    assert csa["max_key_len"] == 3072 and csa["compressed_key_len"] == 768
    assert csa["selected_compressed_key_len"] == plan.shape.index_topk
    assert csa["index_scores_executed"] is True and csa["swa_key_len"] == 128
    assert csa[f"time_stats.{LAYER_SCOPE}.count"] == 2
    assert csa["time_stats.dsv4_index_scores_topk.count"] == 2
    assert csa["indexer_metadata.max_seq_len"] == 3072
    assert json.loads(csa["operator_scopes"])["dsv4_index_scores_topk"] == [
        "dsv4_index_scores", "dsv4_index_topk"]
    hca = rows[CUDA_EVENT][2]
    assert hca["compressed_key_len"] == hca["selected_compressed_key_len"] == 24
    assert "indexer_metadata.max_seq_len" not in hca
    uncompressed = rows[CUDA_EVENT][4]
    assert uncompressed["compress_ratio"] == 1 and uncompressed["compressed_key_len"] == 0
    assert "time_stats.dsv4_compressor.count" not in uncompressed
    kernel_csa = rows[KERNEL_ONLY][0]
    assert f"time_stats.{LAYER_SCOPE}.count" not in kernel_csa
    assert kernel_csa["time_stats.dsv4_sparse_attention.kernel_count"] == 7


def test_short_context_rows_skip_index_scoring(plan):
    boundary = plan.shape.short_context_max_seq_len
    records = [_engine_record()] + _step_records(
        plan, CUDA_EVENT, 0, _prefill_request(boundary, 0, prompt=boundary),
        f"e1-prefill-p{boundary}", PHASE_PREFILL, 1.0)
    csa = build_dsv4_rows(plan, records)[CUDA_EVENT][0]
    assert csa["index_scores_executed"] is False
    assert "time_stats.dsv4_index_scores_topk.count" not in csa
    assert csa["selected_compressed_key_len"] == csa["compressed_key_len"] == 512


def test_rows_reject_scope_and_metadata_mismatches(plan):
    dropped = [_engine_record()] + _step_records(
        plan, CUDA_EVENT, 0, _prefill_request(1024, 2048), "c", PHASE_PREFILL, 1.0,
        drop=(4, "dsv4_index_scores_topk"))
    with pytest.raises(ValueError, match="layer 4"):
        build_dsv4_rows(plan, dropped)
    skewed = [_engine_record()] + _step_records(
        plan, CUDA_EVENT, 0, _prefill_request(1024, 2048), "c", PHASE_PREFILL, 1.0)
    skewed[1][metadata_key("csa", "indexer")] = {"max_seq_len": 3071}
    with pytest.raises(ValueError, match="indexer max_seq_len 3071 differs"):
        build_dsv4_rows(plan, skewed)
    missing = [_engine_record()] + _step_records(
        plan, CUDA_EVENT, 0, _prefill_request(1024, 2048), "c", PHASE_PREFILL, 1.0)
    missing[1][metadata_key("hca", "compressor")] = None
    with pytest.raises(ValueError, match="compressor metadata of the hca layers"):
        build_dsv4_rows(plan, missing)


def test_decode_rows_keep_one_row_per_step(plan):
    def decode(ctx):
        return [{"num_scheduled_tokens": 1, "num_computed_tokens": c, "prompt_len": 512}
                for c in ctx]

    records = [_engine_record()]
    records += _step_records(plan, CUDA_EVENT, 0, decode((514, 520)), "d", PHASE_DECODE, 0.2)
    records += _step_records(plan, CUDA_EVENT, 1, decode((515, 521)), "d", PHASE_DECODE, 0.2)
    rows = build_dsv4_rows(plan, records)[CUDA_EVENT]
    assert len(rows) == 12
    assert {row["num_steps_sampled"] for row in rows} == {1}
    assert rows[0]["decode_kv_cache_size_max"] in (520, 521)


def test_aggregate_cli_writes_canonical_tables(plan, tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    records = [_engine_record()]
    records += _step_records(plan, CUDA_EVENT, 0, _prefill_request(1024, 0), "e1-prefill-p4159", PHASE_PREFILL, 1.0)
    records += _step_records(plan, KERNEL_ONLY, 0, _prefill_request(1024, 0), "e1-prefill-p4159", PHASE_PREFILL, 0.5)
    (raw_dir / "dsv4_raw_e1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    common = ["--raw_dir", str(raw_dir), "--output_root", str(tmp_path / "out"), "--device", "h800",
              "--source_revision", "abc", "--model_config_revision", "rev"]
    with pytest.raises(ValueError, match="lack engines"):
        dsv4_main(["aggregate", *PLAN_ARGS, *common])
    dsv4_main(["aggregate", *PLAN_ARGS, *common, "--allow_partial_engines"])
    base = tmp_path / "out" / "compute" / "h800" / "deepseek-ai" / "DeepSeek-V4-Flash"
    for name, measurement in (("attention.csv", CUDA_EVENT), ("attention_kernel_only.csv", KERNEL_ONLY)):
        with open(base / name, newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 6
        assert {row["measurement_type"] for row in rows} == {measurement}
        assert set(IDENTITY_COLUMNS) <= set(rows[0])
        assert rows[0]["schema_version"] == "dsv4_attention.v1"
        assert rows[0]["quant_signature"].endswith("block=128x128|scale_fmt=ue8m0")
        assert rows[0]["attention_family"] == "deepseek_v4_attention"


# ---------------------------------------------------------------------------
# Hooks on a fake engine

_ATTENTION_MODULE = "vllm.models.deepseek_v4.attention"


class _FakeEvent:
    clock = 0.0

    def __init__(self, enable_timing):
        assert enable_timing
        self.time = None

    def record(self):
        _FakeEvent.clock += 1.0
        self.time = _FakeEvent.clock

    def elapsed_time(self, other):
        return other.time - self.time


def _backend(name):
    return SimpleNamespace(get_name=lambda: name)


class _Cache:
    def __init__(self, prefix, backend):
        self.prefix = prefix
        self.backend = backend

    def get_attn_backend(self):
        return _backend(self.backend)


def _callable(log, name):
    return SimpleNamespace(forward=lambda *args: log.append(name))


class _FakeAttention:
    """Calls the hooked members in the order of ``DeepseekV4Attention.forward``."""

    def __init__(self, layer_id, ratio, log):
        self.prefix = f"model.layers.{layer_id}.attn"
        self.compress_ratio = max(1, ratio)
        self.log = log
        self.swa_cache_layer = _Cache(f"{self.prefix}.swa_cache", "DEEPSEEK_SPARSE_SWA")
        self.compressor = self.indexer = None
        if ratio > 1:
            self.compressor = _callable(log, "compressor")
            self.compressor.state_cache = _Cache(f"{self.prefix}.compressor.state_cache",
                                                 "CompressorBackend")
        if ratio == 4:
            self.indexer = _callable(log, "indexer")
            self.indexer.k_cache = _Cache(f"{self.prefix}.indexer.k_cache", "DEEPSEEK_V4_INDEXER")
            self.indexer.compressor = SimpleNamespace(state_cache=_Cache(
                f"{self.prefix}.indexer.compressor.state_cache", "CompressorBackend"))
            self.indexer.indexer_op = _callable(log, "indexer_op")
            self.indexer.use_fp4_kv = False
        self.wq_b = _callable(log, "wq_b")
        self.kv_cache_dtype = "fp8_ds_mla"
        self.n_local_heads, self.padded_heads, self.n_local_groups = 8, 64, 1
        self.aux_stream_list = [object(), object(), object()]
        self.eager_scratch_pool = object()
        self._prepare_and_attn_fn = self._prepare_and_attn

    def get_attn_backend(self):
        return _backend("FLASHMLA_SPARSE_DSV4")

    def _prepare_and_attn(self):
        raise AssertionError("not called by the fake")

    def _run_parallel_input_projections(self, x):
        self.log.append("input_projections")

    def _fused_qnorm_rope_kv_insert(self, x):
        self.log.append("q_kv_prepare")

    def forward_mqa(self, x):
        self.log.append("forward_mqa")

    def _o_proj(self, x):
        self.log.append("o_proj")
        return x

    def forward(self, x):
        from vllm.forward_context import get_forward_context

        self._run_parallel_input_projections(x)
        sys.modules[_ATTENTION_MODULE].fused_q_kv_rmsnorm(x)
        self.wq_b.forward(x)
        self._fused_qnorm_rope_kv_insert(x)
        if self.indexer is not None:
            self.indexer.forward(x)
        if self.compressor is not None:
            self.compressor.forward(x)
        if self.indexer is not None:
            metadata = get_forward_context().attn_metadata[self.indexer.k_cache.prefix]
            if metadata.max_seq_len // 4 > 512:
                self.indexer.indexer_op.forward(x)
        self.forward_mqa(x)
        return self._o_proj(x)


def _fake_metadata(requests):
    max_seq_len = max(r["num_computed_tokens"] + r["num_scheduled_tokens"] for r in requests)
    tokens = sum(r["num_scheduled_tokens"] for r in requests)
    swa = SimpleNamespace(**{name: 1 for name in dsv4_runner.METADATA_FIELDS["swa"]})
    attention = SimpleNamespace(**{name: 2 for name in dsv4_runner.METADATA_FIELDS["attention"]})
    attention.max_seq_len = max_seq_len
    compressor = SimpleNamespace(block_size=64, num_decode_tokens=0, c128_boundary=None)
    indexer = SimpleNamespace(max_seq_len=max_seq_len, num_decodes=0, num_decode_tokens=0,
                              num_prefills=1, num_prefill_tokens=tokens)
    return {"swa_cache": swa, "attn": attention, "compressor.state_cache": compressor,
            "indexer.k_cache": indexer, "indexer.compressor.state_cache": compressor}


@pytest.fixture
def fake_engine(plan, monkeypatch):
    log = []
    modules = [(f"model.layers.{l.global_layer_id}.attn",
                _FakeAttention(l.global_layer_id, l.compress_ratio, log))
               for l in plan.shape.layers]
    attention_module = types.ModuleType(_ATTENTION_MODULE)
    attention_module.DeepseekV4Attention = _FakeAttention
    attention_module.fused_q_kv_rmsnorm = lambda *args: log.append("rmsnorm")
    context = SimpleNamespace(attn_metadata=None)
    vllm = types.ModuleType("vllm")
    vllm.envs = SimpleNamespace(VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1024)
    forward_context = types.ModuleType("vllm.forward_context")
    forward_context.get_forward_context = lambda: context
    for name, module in (("vllm", vllm), ("vllm.forward_context", forward_context),
                         (_ATTENTION_MODULE, attention_module)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(Event=_FakeEvent, synchronize=lambda: None)))
    monkeypatch.setattr(case_runner, "_STATE", None)
    monkeypatch.setattr(dsv4_runner, "_CURRENT_LAYER", None)

    def execute_model(scheduler_output):
        requests = case_runner._require_state().active.record["requests"]
        context.attn_metadata = {
            f"{prefix}.{suffix}": value
            for prefix, _ in modules
            for suffix, value in _fake_metadata(requests).items()
        }
        for prefix, module in modules:
            context.attn_metadata[prefix] = context.attn_metadata[f"{prefix}.attn"]
            module.forward("x")
        return "output"

    worker = SimpleNamespace(
        model_runner=SimpleNamespace(execute_model=execute_model),
        get_model=lambda: SimpleNamespace(named_modules=lambda: iter(modules)),
        vllm_config=SimpleNamespace(
            use_v2_model_runner=True, cache_config=SimpleNamespace(num_gpu_blocks=4096)
        ),
    )
    return worker, log


def _prefill_output(prompt):
    return SimpleNamespace(
        finished_req_ids=set(),
        num_scheduled_tokens={"a": prompt},
        scheduled_new_reqs=[SimpleNamespace(req_id="a", prompt_token_ids=[0] * prompt,
                                            num_computed_tokens=0)],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], num_computed_tokens=[]),
    )


@pytest.mark.parametrize("prompt", [2051, 2052])
def test_hooks_record_every_scope_of_every_layer(plan, fake_engine, prompt):
    worker, log = fake_engine
    layers = [(l.global_layer_id, l.attention_variant, l.compress_ratio) for l in plan.shape.layers]
    inventory = dsv4_runner.install_dsv4_hooks(worker, 1, layers)
    assert inventory["attention_backend"] == "FLASHMLA_SPARSE_DSV4"
    assert inventory["compressor_backend"] == "CompressorBackend"
    assert inventory["prepare_and_attn_fn"] == "_prepare_and_attn"
    assert inventory["aux_stream_count"] == 3 and inventory["use_v2_model_runner"] is True
    assert inventory["num_gpu_blocks"] == 4096
    assert [layer["metadata_prefixes"] for layer in inventory["layers"]][2] == {
        "swa": "model.layers.2.attn.swa_cache", "attention": "model.layers.2.attn",
        "compressor": "model.layers.2.attn.compressor.state_cache",
        "indexer": "model.layers.2.attn.indexer.k_cache",
        "index_compressor": "model.layers.2.attn.indexer.compressor.state_cache",
    }
    with pytest.raises(RuntimeError, match="already installed"):
        dsv4_runner.install_dsv4_hooks(worker, 1, layers)

    case_runner.begin_case(worker, "c", PHASE_PREFILL, 1, 0, 1, CUDA_EVENT)
    assert worker.model_runner.execute_model(_prefill_output(prompt)) == "output"
    result = case_runner.finish_case(worker, None)
    assert result["recorded"] == 1
    assert log.count("rmsnorm") == 6 and log.count("indexer_op") == (2 if prompt > 2051 else 0)
    records = [_engine_record(), *result["steps"], *result["calls"]]
    for record in records[1:]:
        record["engine_index"] = 1
    step = result["steps"][0]
    assert step[metadata_key("hca", "compressor")]["c128_boundary"] is None
    rows = build_dsv4_rows(plan, records)[CUDA_EVENT]
    assert len(rows) == 6
    csa = rows[0]
    assert csa["index_scores_executed"] is (prompt > 2051)
    assert csa[f"time_stats.{LAYER_SCOPE}.count"] == 1


def test_hooks_reject_engine_layers_that_disagree_with_the_plan(plan, fake_engine):
    worker, _ = fake_engine
    layers = [(l.global_layer_id, l.attention_variant, l.compress_ratio) for l in plan.shape.layers]
    layers[3] = (3, "hca", 64)
    with pytest.raises(ValueError, match="Layer 3 compresses by 128; the plan says 64"):
        dsv4_runner.install_dsv4_hooks(worker, 1, layers)


def test_norm_outside_a_hooked_layer_is_an_error(plan, fake_engine):
    worker, _ = fake_engine
    layers = [(l.global_layer_id, l.attention_variant, l.compress_ratio) for l in plan.shape.layers]
    dsv4_runner.install_dsv4_hooks(worker, 1, layers)
    with pytest.raises(RuntimeError, match="outside a hooked DeepSeek-V4 attention layer"):
        sys.modules[_ATTENTION_MODULE].fused_q_kv_rmsnorm("x")
