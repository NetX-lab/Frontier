"""CPU contracts for V4 graph policies, native scope hooks, and re-capture."""

from __future__ import annotations

import json
import random
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.profiling.attention import dsa_vllm_runner as case_runner
from frontier.profiling.attention import dsv4_attention_main as cli
from frontier.profiling.attention import dsv4_attention_runner as attention_runner
from frontier.profiling.attention import dsv4_graph_replay_runner as replay
from frontier.profiling.attention.dsa_plan import (
    ENGINE_POLICY_EAGER_CHUNKED,
    ENGINE_POLICY_GRAPH_REPLAY,
    ENGINE_POLICY_NATIVE_DEFAULTS,
    DsaProfilePlan,
)
from frontier.profiling.attention.dsa_rows import KERNEL_ONLY, MEASUREMENT_TYPES
from frontier.profiling.attention.dsv4_attention_plan import LAYER_SCOPE
from frontier.profiling.attention.dsv4_attention_rows import build_dsv4_rows
from frontier.profiling.attention.graph_capture import (
    CaptureRecord, EagerBreak, GraphCapture, GraphCaptureRecorder, GraphNode, NodeClone, ScopeEntry,
)
from frontier.profiling.attention.graph_replay_trace import step_annotation_name
from test_dsv4_attention_profiling import (
    PLAN_ARGS, _FakeAttention, _build_plan, _engine_record, _limits, hf_config,
)
from test_graph_replay_trace import _activity, _launch, _node, _step

GRAPH_FLAGS = ["--engine_policy", ENGINE_POLICY_GRAPH_REPLAY, "--cudagraph_mode", "FULL_AND_PIECEWISE",
               "--cudagraph_capture_sizes", "1", "2", "4", "8"]
_LIMIT_KEYS = frozenset({
    "max_context_len", "max_kv_tokens_per_lane", "max_num_seqs", "chunk_budgets", "block_size",
    "decode_warmup_steps", "decode_measured_steps", "prefill_warmup_repeats", "prefill_measured_repeats",
})
_SUMMARY_KEYS = frozenset({
    "schema_version", "model_name", "heads_per_rank", "target_tensor_parallel_size", "reduced_layers",
    "num_engines", "num_cases", "num_prefill_cases", "num_decode_cases", "estimated_steps", "engines", "hf_overrides",
})


@pytest.fixture
def plan():
    return cli.build_plan(cli.build_parser().parse_args(["plan", *PLAN_ARGS, *GRAPH_FLAGS]))


@pytest.mark.parametrize("mode,sizes", [("FULL_AND_PIECEWISE", (1, 2, 4)), ("PIECEWISE", (1,)), ("NONE", ())])
def test_graph_limits_accept_modes(mode, sizes):
    limits = _limits(engine_policy=ENGINE_POLICY_GRAPH_REPLAY, cudagraph_mode=mode, cudagraph_capture_sizes=sizes)
    assert (limits.cudagraph_mode, limits.cudagraph_capture_sizes) == (mode, sizes)


@pytest.mark.parametrize("mode,sizes,error", [
    (None, (), "requires cudagraph_mode"), ("FULL", (1,), "requires cudagraph_mode"),
    ("invalid", (), "requires cudagraph_mode"),
    ("FULL_AND_PIECEWISE", (), "non-empty"), ("PIECEWISE", (), "non-empty"),
    ("PIECEWISE", (0,), "positive int"), ("PIECEWISE", (-1,), "positive int"),
    ("PIECEWISE", (True,), "positive int"), ("PIECEWISE", (1.5,), "positive int"),
    ("PIECEWISE", (1, 1), "strictly increasing"), ("PIECEWISE", (2, 1), "strictly increasing"),
    ("PIECEWISE", (257,), "smallest chunk budget 256"), ("NONE", (1,), "requires empty"),
])
def test_graph_limits_reject_each_invalid_configuration(mode, sizes, error):
    with pytest.raises(ValueError, match=error):
        _limits(engine_policy=ENGINE_POLICY_GRAPH_REPLAY, cudagraph_mode=mode, cudagraph_capture_sizes=sizes)


@pytest.mark.parametrize("policy", [ENGINE_POLICY_EAGER_CHUNKED, ENGINE_POLICY_NATIVE_DEFAULTS])
@pytest.mark.parametrize("mode,sizes", [("NONE", ()), (None, (1,)), ("PIECEWISE", (1,))])
def test_other_policies_refuse_cudagraph_fields(policy, mode, sizes):
    with pytest.raises(ValueError, match="requires cudagraph_mode=None and empty"):
        _limits(engine_policy=policy, chunk_budgets=(8192,), cudagraph_mode=mode, cudagraph_capture_sizes=sizes)


def test_unknown_policy_still_fails():
    with pytest.raises(ValueError, match="Unknown DSA engine_policy"):
        _limits(engine_policy="unknown")


@pytest.mark.parametrize("command", ["plan", "run", "aggregate"])
def test_cli_rebuilds_graph_plan_for_every_command(command, plan):
    extra = {
        "plan": [],
        "run": ["--engine_index", "0", "--model_dir", "unused", "--raw_dir", "unused",
                "--trace_dir", "unused", "--expected_attention_backend", "unused",
                "--gpu_memory_utilization", "0.85", "--seed", "1234"],
        "aggregate": ["--raw_dir", "unused", "--output_root", "unused", "--device", "unused",
                      "--source_revision", "unused", "--model_config_revision", "unused"],
    }
    args = cli.build_parser().parse_args([command, *PLAN_ARGS, *GRAPH_FLAGS, *extra[command]])
    assert cli.build_plan(args).to_json() == plan.to_json()


def test_cli_none_has_empty_sizes():
    args = cli.build_parser().parse_args(["plan", *PLAN_ARGS, "--engine_policy", ENGINE_POLICY_GRAPH_REPLAY,
                                          "--cudagraph_mode", "NONE"])
    assert cli.build_plan(args).limits.cudagraph_capture_sizes == ()


def test_graph_serialization_and_summary(plan):
    payload = plan.to_json()
    assert set(payload["limits"]) == _LIMIT_KEYS | {"engine_policy", "cudagraph_mode", "cudagraph_capture_sizes"}
    assert payload["limits"]["engine_policy"] == ENGINE_POLICY_GRAPH_REPLAY
    assert payload["limits"]["cudagraph_mode"] == "FULL_AND_PIECEWISE"
    assert payload["limits"]["cudagraph_capture_sizes"] == (1, 2, 4, 8)
    summary = plan.summary()
    assert summary["engine_policy"] == ENGINE_POLICY_GRAPH_REPLAY
    assert summary["cudagraph_mode"] == "FULL_AND_PIECEWISE"
    assert summary["cudagraph_capture_sizes"] == (1, 2, 4, 8)
    assert "timing_collection" not in payload and "timing_collection" not in summary


@pytest.mark.parametrize("policy", [ENGINE_POLICY_EAGER_CHUNKED, ENGINE_POLICY_NATIVE_DEFAULTS])
def test_legacy_serialized_key_sets_remain_frozen(plan, policy):
    limits = replace(plan.limits, engine_policy=policy, cudagraph_mode=None, cudagraph_capture_sizes=(),
                     chunk_budgets=(8192,))
    legacy = replace(plan, limits=limits)
    payload = legacy.to_json()
    keys = _LIMIT_KEYS if policy == ENGINE_POLICY_EAGER_CHUNKED else _LIMIT_KEYS | {"engine_policy"}
    assert set(payload["limits"]) == keys
    assert set(payload) == set(asdict(legacy)) | ({"timing_collection"} if policy == ENGINE_POLICY_NATIVE_DEFAULTS else set())
    summary = DsaProfilePlan.summary(legacy)
    assert set(summary) == _SUMMARY_KEYS | ({"engine_policy", "timing_collection"} if policy == ENGINE_POLICY_NATIVE_DEFAULTS else set())
    if policy == ENGINE_POLICY_NATIVE_DEFAULTS:
        assert payload["timing_collection"] == summary["timing_collection"] == "pending_native_scope_instrumentation"


def test_graph_grid_matches_eager_and_checks_chunk_budget(plan, hf_config):
    eager = _build_plan(hf_config)
    graph = _build_plan(hf_config, engine_policy=ENGINE_POLICY_GRAPH_REPLAY, cudagraph_mode="PIECEWISE",
                        cudagraph_capture_sizes=(1, 2, 4, 8))
    assert graph.engines == eager.engines and graph.cases == eager.cases
    with pytest.raises(ValueError, match="chunk budgets must stay below max_context_len"):
        _build_plan(hf_config, engine_policy=ENGINE_POLICY_GRAPH_REPLAY, cudagraph_mode="PIECEWISE",
                    cudagraph_capture_sizes=(1,), chunk_budgets=(8192,))


def test_timing_measurements_and_legacy_refusals(plan):
    assert case_runner.timing_measurements(ENGINE_POLICY_EAGER_CHUNKED) == MEASUREMENT_TYPES
    assert case_runner.timing_measurements(ENGINE_POLICY_GRAPH_REPLAY) == (KERNEL_ONLY,)
    for policy in (ENGINE_POLICY_NATIVE_DEFAULTS, "invalid"):
        with pytest.raises(ValueError, match="cannot collect or aggregate.*native_defaults supports planning"):
            case_runner.timing_measurements(policy)
    with pytest.raises(ValueError, match="cannot collect or aggregate"):
        case_runner.run_engine(plan=plan, engine_index=0, model_dir="unused", vocab_size=10, seed=0,
                               gpu_memory_utilization=0.85, expected_attention_backend="unused",
                               raw_path=Path("unused"), trace_dir=Path("unused"), keep_traces=True)
    with pytest.raises(ValueError, match="cannot collect or aggregate.*native_defaults supports planning"):
        build_dsv4_rows(plan, [dict(_engine_record(), engine_policy=ENGINE_POLICY_GRAPH_REPLAY)])


@pytest.mark.parametrize("mode,sizes", [("FULL_AND_PIECEWISE", (1, 2, 4, 8)), ("NONE", ())])
def test_build_engine_preserves_native_stream_threshold_and_graph_options(plan, monkeypatch, mode, sizes):
    plan = replace(plan, limits=replace(plan.limits, cudagraph_mode=mode, cudagraph_capture_sizes=sizes))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=lambda **options: options))
    result = case_runner.build_engine(plan, 0, "fake_model", 1234, 0.85, kv_cache_dtype="fp8")
    assert "enforce_eager" not in result
    assert result["enable_chunked_prefill"] is True and result["async_scheduling"] is False
    assert result["long_prefill_token_threshold"] == plan.engine(0).prefill_query_len
    assert result["kv_cache_dtype"] == "fp8"
    assert result["compilation_config"] == dict(cudagraph_mode=mode, **({"cudagraph_capture_sizes": list(sizes)} if sizes else {}))
    assert not any("multi_stream" in key for key in result)


def test_cli_runner_policy_table(plan, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "file_sha256", lambda path: "same")
    monkeypatch.setattr(replay, "run_engine", lambda **kwargs: calls.append(kwargs))
    args = cli.build_parser().parse_args([
        "run", *PLAN_ARGS, *GRAPH_FLAGS, "--engine_index", "0", "--model_dir", "unused",
        "--raw_dir", str(tmp_path), "--trace_dir", str(tmp_path), "--expected_attention_backend", "fake",
        "--gpu_memory_utilization", "0.85", "--seed", "1234", "--case_ids", plan.cases[0].case_id,
    ])
    cli._run_command(args)
    assert calls[0]["plan"].to_json() == plan.to_json()
    assert calls[0]["raw_path"] == tmp_path / "dsv4_raw_e0.jsonl"
    assert calls[0]["case_ids"] == {plan.cases[0].case_id}


@pytest.fixture
def native_tree(plan, monkeypatch):
    entries, annotations, launches, parallel_calls = [], [], [], []
    native = {name: SimpleNamespace() for name in {
        replay._ATTENTION_MODULE, replay._COMPRESSOR_MODULE, replay._COMPRESS_STORE_MODULE,
        replay._INDEXER_MODULE, replay._CUSTOM_OPS_MODULE, replay._OUTPUT_MODULE, replay._MODEL_RUNNER_MODULE,
    }}
    for name, module in native.items():
        monkeypatch.setitem(sys.modules, name, module)
    for module_name, attribute, _ in (*replay._MODULE_SCOPES, *replay._COMPRESSOR_FUNCTIONS):
        setattr(native[module_name], attribute, lambda *args, **kwargs: launches.append((args, kwargs)))

    def parallel(default_fn, aux_fns, start_event, done_events, aux_streams=None, enable=False):
        parallel_calls.append(aux_fns)
        return default_fn(), [None if fn is None else fn() for fn in aux_fns]

    class FillKernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append((grid, args, kwargs))

    attn = native[replay._ATTENTION_MODULE]
    attn.execute_in_parallel = parallel
    attn._fill_short_context_topk_indices = FillKernel()

    def compressor_call():
        native[replay._COMPRESSOR_MODULE].save_partial_states()
        native[replay._COMPRESS_STORE_MODULE].compress_norm_rope_store_cutedsl()

    class Attention(_FakeAttention):
        def __init__(self, layer, ratio):
            super().__init__(layer, ratio, [])
            self.wo_b = SimpleNamespace(forward=lambda *args: None)
            if self.compressor is not None:
                self.compressor.forward = compressor_call
            if self.indexer is not None:
                self.indexer.compressor.forward = compressor_call
                self.indexer.wq_b = SimpleNamespace(forward=lambda: None)
                self.indexer.forward = self.index_prepare
                self.indexer.indexer_op.forward = self.index_topk
            self.short_context = False

        def _run_parallel_input_projections(self, x):
            aux = [None, None, None]
            if self.compressor is not None:
                aux[0] = lambda: None
            if self.indexer is not None:
                aux[1:] = [lambda: None, lambda: None]
            attn.execute_in_parallel(lambda: self._fused_wqa_wkv_gemm(x), aux, None, [], enable=True)

        def _fused_wqa_wkv_gemm(self, x):
            return x

        def index_prepare(self):
            self.indexer.compressor.forward()
            if self.short_context:
                attn._fill_short_context_topk_indices[(2,)]("buffer", TOP_K=512)
            else:
                self.indexer.wq_b.forward()
                attn.fused_indexer_q_rope_quant()

        def index_topk(self):
            for module_name, attribute, scope in replay._MODULE_SCOPES:
                if scope == "dsv4_index_scores":
                    getattr(native[module_name], attribute)()

        def _o_proj(self, x):
            native[replay._OUTPUT_MODULE].fused_inv_rope_fp8_quant()
            native[replay._OUTPUT_MODULE].fp8_einsum()
            self.wo_b.forward(x)
            return x

        def forward(self, x):
            self._run_parallel_input_projections(x)
            attn.fused_q_kv_rmsnorm(x)
            self.wq_b.forward(x)
            self._fused_qnorm_rope_kv_insert(x)
            if self.indexer is not None:
                self.indexer.forward()
            if self.compressor is not None:
                self.compressor.forward()
            if self.indexer is not None:
                self.indexer.indexer_op.forward()
            self.forward_mqa(x)
            return self._o_proj(x)

    attn.DeepseekV4Attention = Attention
    modules = {layer.global_layer_id: Attention(layer.global_layer_id, layer.compress_ratio) for layer in plan.shape.layers}
    metadata = {}
    for module in modules.values():
        for source, prefix in attention_runner.metadata_prefixes(module).items():
            metadata[prefix] = SimpleNamespace(**dict.fromkeys(attention_runner.METADATA_FIELDS[source], 1))
    descriptor = SimpleNamespace(cg_mode=SimpleNamespace(name="FULL"), num_tokens=8, num_reqs=None)
    runner_module = native[replay._MODEL_RUNNER_MODULE]
    runner_module.dispatch_cg_and_sync_dp = lambda *args, **kwargs: (descriptor, "tensor")
    runner = SimpleNamespace(execute_model_state=None)

    def execute_model(*args):
        runner_module.dispatch_cg_and_sync_dp()
        runner.execute_model_state = SimpleNamespace(attn_metadata=metadata)
        return "replayed without Python layer hooks"

    runner.execute_model = execute_model
    worker = SimpleNamespace(
        model_runner=runner, get_model=lambda: SimpleNamespace(named_modules=lambda: [
            (f"model.layers.{layer}.attn", module) for layer, module in modules.items()
        ]), vllm_config=SimpleNamespace(use_v2_model_runner=True, cache_config=SimpleNamespace(num_gpu_blocks=12)),
    )

    @contextmanager
    def annotation(name):
        annotations.append(name)
        yield

    class Recorder(GraphCaptureRecorder):
        @contextmanager
        def scope(self, owner):
            entries.append((owner, self.current_owner))
            with super().scope(owner):
                yield

    monkeypatch.setattr(replay, "GraphCaptureRecorder", Recorder)
    monkeypatch.setattr(replay, "CudaGraphQueries", lambda: None)
    monkeypatch.setattr(replay, "_RECORDER", None)
    monkeypatch.setattr(case_runner, "_STATE", None)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(envs=SimpleNamespace(VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1024)))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(profiler=SimpleNamespace(record_function=annotation)))
    layers = [(layer.global_layer_id, layer.attention_variant, layer.compress_ratio) for layer in plan.shape.layers]
    inventory = replay.install_graph_replay_hooks(worker, 0, layers, plan)
    state = case_runner._require_state()
    monkeypatch.setattr(state, "begin_step", lambda scheduler: None)
    state.case = case_runner._CaseRecorder("fake", "decode", 1, 0, 1, KERNEL_ONLY)
    state.active = case_runner._ActiveStep(7, dict.fromkeys(state.step_metadata))
    return SimpleNamespace(entries=entries, annotations=annotations, launches=launches,
                           parallel_calls=parallel_calls, native=native, modules=modules, worker=worker,
                           state=state, inventory=inventory, metadata=metadata, descriptor=descriptor)


@pytest.mark.parametrize("variant", ["csa", "hca", "uncompressed"])
def test_scope_tree_parents_and_push_order(plan, native_tree, variant):
    layer = next(layer.global_layer_id for layer in plan.shape.layers if layer.attention_variant == variant)
    assert native_tree.modules[layer].forward("input") == "input"
    scopes = [owner[1] for owner, _ in native_tree.entries]
    expected = [LAYER_SCOPE, "dsv4_parallel_input_projections", "dsv4_fused_qkv_projection"]
    if variant != "uncompressed":
        expected += ["dsv4_compressor_projection"]
    if variant == "csa":
        expected += ["dsv4_index_weights_projection", "dsv4_index_compressor_projection"]
    expected += ["dsv4_fused_qkv_rmsnorm", "dsv4_q_projection", "dsv4_fused_q_kv_prepare"]
    if variant == "csa":
        expected += ["dsv4_index_prepare", "dsv4_index_compressor", "dsv4_index_compressor_save_partial_states",
                     "dsv4_index_compressor_fused_compress_store", "dsv4_index_query_projection", "dsv4_index_fused_query_prepare"]
    if variant != "uncompressed":
        expected += ["dsv4_compressor", "dsv4_compressor_save_partial_states", "dsv4_compressor_fused_compress_store"]
    if variant == "csa":
        expected += ["dsv4_index_topk", "dsv4_index_scores", "dsv4_index_scores", "dsv4_index_scores"]
    expected += ["dsv4_sparse_attention", "dsv4_o_proj", "dsv4_output_inv_rope_quant",
                 "dsv4_grouped_output_projection", "dsv4_output_projection"]
    assert scopes == expected
    for owner, parent in native_tree.entries:
        parent_scope = dict(replay.SCOPE_TREE)[owner[1]]
        assert parent == (None if parent_scope is None else (layer, parent_scope))
    assert len(native_tree.annotations) == len(expected)
    assert all(name.startswith("frontier_dsa|7|") for name in native_tree.annotations)
    assert replay._require_recorder().current_owner is None
    assert native_tree.state.active.record == dict.fromkeys(native_tree.state.step_metadata)


def test_aux_slots_pass_through_under_layer_and_outside(native_tree):
    aux = [lambda: "one", None]
    execute = native_tree.native[replay._ATTENTION_MODULE].execute_in_parallel
    for owner in (None, (0, LAYER_SCOPE)):
        with replay._require_recorder().scope(owner):
            assert execute(lambda: "default", aux, None, []) == ("default", ["one", None])
    assert all(item is aux for item in native_tree.parallel_calls)
    assert not native_tree.annotations
    with replay._require_recorder().scope((0, "dsv4_parallel_input_projections")):
        with pytest.raises(ValueError, match="zip"):
            execute(lambda: None, aux, None, [])


@pytest.mark.parametrize("owner", [None, (100, LAYER_SCOPE), (0, "dsv4_index_prepare")])
def test_parent_mismatch_reports_scope_expected_and_actual(native_tree, owner):
    layer = 0
    with replay._require_recorder().scope(owner):
        with pytest.raises(ValueError, match="Scope 'dsv4_q_projection' expected parent.*actual owner"):
            native_tree.modules[layer].wq_b.forward("input")


def test_module_name_hook_outside_layer_fails(native_tree):
    with pytest.raises(RuntimeError, match="outside a hooked DeepSeek-V4 attention layer"):
        native_tree.native[replay._ATTENTION_MODULE].fused_q_kv_rmsnorm()


@pytest.mark.parametrize("parent", ["dsv4_compressor", "dsv4_index_compressor"])
@pytest.mark.parametrize("attribute", ["compress_norm_rope_store_triton", "compress_norm_rope_store_two_stage_triton"])
def test_both_compressor_names_and_triton_imports(native_tree, parent, attribute):
    with replay._require_recorder().scope((0, parent)):
        getattr(native_tree.native[replay._COMPRESSOR_MODULE], attribute)("input", block_size=8)
    assert native_tree.entries[-1] == ((0, parent + "_fused_compress_store"), (0, parent))
    assert native_tree.launches[-1] == (("input",), {"block_size": 8})


def test_compressor_module_hook_requires_compressor_parent(native_tree):
    with replay._require_recorder().scope((0, LAYER_SCOPE)):
        with pytest.raises(ValueError, match="Compressor function.*expected parent.*actual owner"):
            native_tree.native[replay._COMPRESSOR_MODULE].save_partial_states()


def test_short_context_fill_keeps_grid_and_arguments(plan, native_tree):
    layer = next(layer.global_layer_id for layer in plan.shape.layers if layer.attention_variant == "csa")
    native_tree.modules[layer].short_context = True
    native_tree.modules[layer].forward("input")
    assert ((2,), ("buffer",), {"TOP_K": 512}) in native_tree.launches
    assert ((layer, "dsv4_index_short_context_fill"), (layer, "dsv4_index_prepare")) in native_tree.entries
    assert not any(owner[1] == "dsv4_index_query_projection" for owner, _ in native_tree.entries)


def test_eager_scope_runner_annotates_registration_owner(native_tree):
    owner = (4, LAYER_SCOPE)
    assert replay._eager_scope_runner(owner, lambda: "eager") == "eager"
    assert native_tree.entries == [(owner, None)]
    assert native_tree.annotations == ["frontier_dsa|7|4|dsv4_attention_layer"]
    assert replay._require_recorder().current_owner is None


def test_step_range_reads_full_replay_metadata_and_dispatch(native_tree):
    assert native_tree.worker.model_runner.execute_model("ignored") == "replayed without Python layer hooks"
    assert native_tree.annotations == [step_annotation_name(7)]
    record = native_tree.state.active.record
    assert all(record[key] is not None for key in native_tree.state.step_metadata)
    assert record[replay.CUDAGRAPH_DISPATCH_KEY] == {"cg_mode": "FULL", "num_tokens": 8, "num_reqs": None}
    assert record["csa_compressor_metadata"] == {"block_size": 1, "num_decode_tokens": 1, "c128_boundary": 1}
    native_tree.state.active = None
    native_tree.worker.model_runner.execute_model("unmeasured")
    assert native_tree.annotations == [step_annotation_name(7)]


def test_case_wrapper_is_outermost_and_metadata_precedes_end_step(native_tree, monkeypatch):
    state = native_tree.state
    active = state.active
    monkeypatch.setattr(state, "begin_step", lambda scheduler: active)
    # Installation wraps the inner step before creating the outer case classifier.
    ended = []
    monkeypatch.setattr(state, "end_step", lambda finished: ended.append(dict(finished.record)))
    state.active = None
    native_tree.worker.model_runner.execute_model("schedule")
    assert state.active is None and ended[0][replay.CUDAGRAPH_DISPATCH_KEY]["cg_mode"] == "FULL"


def test_step_metadata_non_dict_error_is_unchanged(native_tree):
    runner = native_tree.worker.model_runner
    # Replace the fake native execution, then exercise the installed inner wrapper directly.
    discovered = attention_runner.discover_attention_layers(native_tree.worker, [
        (item["global_layer_id"], item["attention_variant"], item["compress_ratio"])
        for item in native_tree.inventory["layers"]
    ])
    runner.execute_model = lambda *args: setattr(runner, "execute_model_state", SimpleNamespace(attn_metadata=[]))
    replay._wrap_step(runner, discovered)
    with pytest.raises(ValueError, match="Micro-batched attention metadata is not supported by the DSA profiler"):
        runner.execute_model()


def test_graph_hooks_refuse_v1_runner(plan):
    with pytest.raises(ValueError, match="requires use_v2_model_runner=True"):
        replay.install_graph_replay_hooks(SimpleNamespace(vllm_config=SimpleNamespace(use_v2_model_runner=False)), 0, [], plan)


@pytest.fixture
def recapture(monkeypatch):
    calls = []
    record = CaptureRecord(
        captures=(GraphCapture(0, "graph", None, 1), GraphCapture(1, "breakable_segment", 0, 1)),
        nodes=(_node(10, "kernel", owner=(0, "dsv4_q_projection")),),
        eager_breaks=(EagerBreak(0, 1, (0, LAYER_SCOPE)),), clones=(NodeClone(1010, 10),),
    )
    cm = SimpleNamespace(cudagraph_mode=SimpleNamespace(name="FULL_AND_PIECEWISE"), graphs={"old": "graph"},
                         use_breakable_cg=True, _graphs_captured=True,
                         breakable_cg_runner=SimpleNamespace(clear_graphs=lambda: calls.append("clear_breakable")))
    runner = SimpleNamespace(cudagraph_manager=cm, compilation_config=SimpleNamespace(cudagraph_capture_sizes=[8, 4, 2, 1]))

    def capture():
        assert not cm.graphs and cm._graphs_captured is False
        calls.append("capture")

    runner.capture_model = capture

    @contextmanager
    def installed(*args):
        calls.append("install")
        yield
        calls.append("restore")

    class Clones:
        clones = [(1010, 10)]

        def __enter__(self):
            calls.append("subscribe")
            return self

        def __exit__(self, *args):
            calls.append("unsubscribe")

    saved = SimpleNamespace(record=lambda clones: record, install=installed)
    monkeypatch.setattr(replay, "_RECORDER", saved)
    monkeypatch.setattr(replay, "CuptiCloneRecorder", Clones)
    monkeypatch.setenv(replay.BREAKABLE_CUDAGRAPH_ENV, "1")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(CUDAGraph=object)))
    monkeypatch.setitem(sys.modules, "vllm.compilation.breakable_cudagraph", SimpleNamespace(BreakableCUDAGraphCapture=object))
    return SimpleNamespace(worker=SimpleNamespace(model_runner=runner), cm=cm, calls=calls, record=record, saved=saved)


def test_recapture_clears_graphs_and_stops_subscriber_before_record(recapture):
    result = replay.recapture_graphs(recapture.worker, "FULL_AND_PIECEWISE", (1, 2, 4, 8))
    assert recapture.calls == ["clear_breakable", "subscribe", "install", "capture", "restore", "unsubscribe"]
    assert CaptureRecord.from_json(result["record"]) == recapture.record
    assert result["summary"] == {
        "cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1, 2, 4, 8],
        "captures_by_kind": {"graph": 1, "breakable_segment": 1}, "node_count": 1,
        "kernel_node_count": 1, "clone_count": 1, "eager_break_count": 1, "eager_break_owner_scopes": [LAYER_SCOPE],
    }


@pytest.mark.parametrize("mutation,error", [
    ("mode", "Resolved cudagraph mode.*differs from plan"),
    ("sizes", "Resolved cudagraph capture sizes.*differ from plan"),
    ("breakable", "requires Breakable capture"), ("env", "requires Breakable capture"),
])
def test_recapture_checks_resolved_configuration_before_mutation(recapture, monkeypatch, mutation, error):
    if mutation == "mode":
        recapture.cm.cudagraph_mode.name = "PIECEWISE"
    elif mutation == "sizes":
        recapture.worker.model_runner.compilation_config.cudagraph_capture_sizes = [1, 2]
    elif mutation == "breakable":
        recapture.cm.use_breakable_cg = False
    else:
        monkeypatch.delenv(replay.BREAKABLE_CUDAGRAPH_ENV)
    with pytest.raises(ValueError, match=error):
        replay.recapture_graphs(recapture.worker, "FULL_AND_PIECEWISE", (1, 2, 4, 8))
    assert recapture.calls == [] and recapture.cm.graphs == {"old": "graph"}


@pytest.mark.parametrize("kind,error", [
    ("zero", "recorded zero eager breaks.*before vLLM imports"),
    ("outside", "Eager break expected owner scope"),
    ("child", "Eager break expected owner scope"),
    ("no_full", "recorded no FULL graph capture"),
])
def test_recapture_validates_recorded_breaks_and_full_capture(recapture, kind, error):
    record = recapture.record
    if kind == "zero":
        record = replace(record, eager_breaks=())
    elif kind in ("outside", "child"):
        record = replace(record, eager_breaks=(EagerBreak(0, 1, None if kind == "outside" else (0, "dsv4_q_projection")),))
    else:
        record = replace(record, captures=record.captures[1:])
    recapture.saved.record = lambda clones: record
    with pytest.raises(ValueError, match=error):
        replay.recapture_graphs(recapture.worker, "FULL_AND_PIECEWISE", (1, 2, 4, 8))


def test_none_returns_empty_capture_without_gpu_or_mutation(recapture):
    recapture.cm.cudagraph_mode.name = "NONE"
    recapture.worker.model_runner.compilation_config.cudagraph_capture_sizes = []
    recapture.cm.use_breakable_cg = False
    result = replay.recapture_graphs(recapture.worker, "NONE", ())
    assert CaptureRecord.from_json(result["record"]) == CaptureRecord()
    assert result["summary"]["captures_by_kind"] == {"graph": 0, "breakable_segment": 0}
    assert result["summary"]["eager_break_count"] == 0
    assert recapture.calls == [] and recapture.cm.graphs == {"old": "graph"}


def test_piecewise_capture_needs_no_full_graph(recapture):
    recapture.cm.cudagraph_mode.name = "PIECEWISE"
    recapture.cm.breakable_cg_runner = None
    record = replace(recapture.record, captures=recapture.record.captures[1:])
    recapture.saved.record = lambda clones: record
    assert replay.recapture_graphs(recapture.worker, "PIECEWISE", (1, 2, 4, 8))["summary"]["captures_by_kind"]["graph"] == 0


def test_graph_calls_exclude_containers_keep_zero_scopes_and_validate_steps(tmp_path):
    owner, zero = (4, "dsv4_q_projection"), (4, "dsv4_output_projection")
    record = CaptureRecord(
        captures=(GraphCapture(0, "graph", None, 7),), nodes=(_node(10, "kernel", owner=owner),),
        scope_entries=(ScopeEntry(0, (4, LAYER_SCOPE), None), ScopeEntry(0, owner, (4, LAYER_SCOPE)),
                       ScopeEntry(0, zero, (4, "dsv4_o_proj"))), clones=(NodeClone(1010, 10),),
    )
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": [_step(), _launch(1, 10), _activity(1010, "kernel", 100, 20)]}))
    calls = case_runner.graph_replay_calls(path, [{"step_index": 0, "case_id": "case"}], 2,
                                          record=record, container_scopes=replay.CONTAINER_SCOPES)
    assert [(call["scope"], call["time_ms"], call["kernel_count"]) for call in calls] == [
        (zero[1], 0.0, 0), (owner[1], 0.020, 1),
    ]
    assert all(call["engine_index"] == 2 and call["case_id"] == "case" for call in calls)
    with pytest.raises(ValueError, match="scopes of unrecorded step"):
        case_runner.graph_replay_calls(path, [], 2, record=record, container_scopes=replay.CONTAINER_SCOPES)


def test_run_engine_keeps_capture_passes_routing_and_uses_trace_callback(plan, monkeypatch, tmp_path):
    calls = []
    capture = replay._capture_result(CaptureRecord(), "FULL_AND_PIECEWISE", (1, 2, 4, 8))
    routing = {"identity": "read at call time"}
    monkeypatch.setattr(attention_runner, "routing_config", routing)
    monkeypatch.setenv(replay.BREAKABLE_CUDAGRAPH_ENV, "1")

    def rpc(fn, args):
        calls.append((fn, args))
        return [{"attention_backend": "fake"} if fn is replay.install_graph_replay_hooks else capture]

    monkeypatch.setattr(attention_runner, "build_dsv4_engine", lambda *args: SimpleNamespace(collective_rpc=rpc))
    measured = []
    monkeypatch.setattr(case_runner, "measure_engine", lambda **kwargs: measured.append(kwargs))
    replay.run_engine(plan=plan, engine_index=0, model_dir="fake", vocab_size=100, seed=1234,
                      gpu_memory_utilization=0.85, expected_attention_backend="fake", raw_path=tmp_path / "raw.jsonl",
                      trace_dir=tmp_path / "traces", keep_traces=True)
    assert calls[0][1][-1] is routing
    assert calls[1] == (replay.recapture_graphs, ("FULL_AND_PIECEWISE", (1, 2, 4, 8)))
    assert json.loads((tmp_path / "dsv4_capture_e0.json").read_text()) == capture
    assert measured[0]["inventory"]["cudagraph_capture"] == capture["summary"]
    assert measured[0]["log_label"] == "dsv4_graph"
    assert measured[0]["trace_calls"].keywords == {"record": CaptureRecord(), "container_scopes": replay.CONTAINER_SCOPES}


def test_run_engine_requires_decoration_environment_before_build(plan, monkeypatch, tmp_path):
    monkeypatch.delenv(replay.BREAKABLE_CUDAGRAPH_ENV, raising=False)
    monkeypatch.setattr(attention_runner, "build_dsv4_engine", lambda *args: pytest.fail("built before environment check"))
    with pytest.raises(ValueError, match="VLLM_USE_BREAKABLE_CUDAGRAPH=1.*before.*import"):
        replay.run_engine(plan=plan, engine_index=0, model_dir="fake", vocab_size=100, seed=1234,
                          gpu_memory_utilization=0.85, expected_attention_backend="fake", raw_path=tmp_path / "raw.jsonl",
                          trace_dir=tmp_path / "traces", keep_traces=True)


def test_measure_engine_runs_only_kernel_timing_and_forwards_trace_callback(plan, monkeypatch, tmp_path):
    measured = []
    callback = lambda *args: []
    monkeypatch.setattr(case_runner, "collect_device_identity", lambda: ("fake", "sm90"))
    monkeypatch.setattr(case_runner, "collect_runtime_versions", lambda: {})
    monkeypatch.setattr(case_runner, "run_case", lambda *args: measured.append(args) or [])
    llm = SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="fp8"))))
    case_id = plan.cases_for_engine(0)[0].case_id
    raw_path = tmp_path / "raw.jsonl"
    case_runner.measure_engine(llm=llm, plan=plan, engine_index=0, inventory={"attention_backend": "fake"},
                               vocab_size=100, seed=1234, expected_attention_backend="fake", raw_path=raw_path,
                               trace_dir=tmp_path / "traces", keep_traces=True, case_ids={case_id},
                               log_label="fake", trace_calls=callback)
    assert len(measured) == 1 and measured[0][3] == KERNEL_ONLY and measured[0][-1] is callback
    assert json.loads(raw_path.read_text())["engine_policy"] == ENGINE_POLICY_GRAPH_REPLAY


def test_legacy_namespace_without_graph_fields_builds_same_plan():
    args = cli.build_parser().parse_args(["plan", *PLAN_ARGS])
    expected = cli.build_plan(args).to_json()
    del args.engine_policy
    del args.cudagraph_mode
    del args.cudagraph_capture_sizes
    assert cli.build_plan(args).to_json() == expected


def test_run_case_uses_supplied_trace_calls(plan, monkeypatch, tmp_path):
    case = next(case for case in plan.cases if case.phase == "prefill")
    _, limit = case_runner._case_window(case, plan.engine(case.engine_index))
    step = {"kind": "step", "step_index": 42, "case_id": case.case_id}
    call = {"kind": "call", "scope": "supplied replay attribution"}
    callback_args = []
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: kwargs))
    engine = SimpleNamespace(add_request=lambda *args: None, has_unfinished_requests=lambda: False)

    def rpc(fn, args):
        return [{"recorded": limit, "steps": [step], "calls": []}] if fn is case_runner.finish_case else [None]

    def trace_calls(*args):
        callback_args.append(args)
        return [call]

    result = case_runner.run_case(SimpleNamespace(collective_rpc=rpc, llm_engine=engine), plan, case,
                                  KERNEL_ONLY, random.Random(1234), 100, tmp_path, True, trace_calls)
    assert result == [step, call]
    assert callback_args == [(tmp_path / f"{case.case_id}.json", [step], case.engine_index)]
