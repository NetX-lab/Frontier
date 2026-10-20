"""CPU contracts only; no test in this module measures a GPU."""

import json
import os
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from frontier.profiling.attention import dsa_full_dummy as full
from frontier.profiling.attention.dsa_plan import INDEXER_SCOPE, MLA_SCOPE

PLAN_ARGS = ["--model_name", "zai-org/GLM-5.3", "--model_config_path",
             "data/config/models/zai-org__GLM-5.3.json", "--target_tensor_parallel_size", "8",
             "--block_size", "64", "--max_context_len", "7168",
             "--max_kv_tokens_per_lane", "14336", "--max_num_seqs", "8",
             "--chunk_budgets", "1024", "--capture_sizes", "2", "4", "8",
             "--source_revision", full.NATIVE_REVISION, "--model_config_revision", "fixture",
             "--expected_attention_backend", "FLASHMLA_SPARSE"]


@pytest.mark.parametrize("sizes", [[], [0], [1, 1], [9], [True]])
def test_capture_sizes_reject_unreachable_decode_keys(sizes):
    with pytest.raises(ValueError, match="unique positive"):
        full.capture_sizes(sizes, 8)


def test_offline_plan_is_isolated_from_serving_and_kernel_only(tmp_path):
    output = tmp_path / "plan.json"
    full.main(["plan", *PLAN_ARGS, "--output", str(output)])
    payload = json.loads(output.read_text())
    assert payload["capture_sizes"] == [2, 4, 8]
    assert payload["measurement_type"] == "CUDA_EVENT"
    assert payload["consumer_admission"] is False
    assert payload["kernel_only_qualified"] is False
    layers = payload["plan"]["shape"]["layers"]
    assert {layer["attention_variant"] for layer in layers} == {"full_index", "shared_index"}
    with pytest.raises(FileExistsError):
        full.main(["plan", *PLAN_ARGS, "--output", str(output)])
    with pytest.raises(SystemExit):
        full.main(["plan", *PLAN_ARGS, "--query_one_context_max", "0", "--output", str(output)])


class Tensor:
    def __init__(self, values):
        self.values = np.array(values)

    def cpu(self):
        return self.values


def test_invalid_slot_or_padding_cannot_qualify_as_dummy():
    native = NS(optimistic_seq_lens_cpu=np.array([1, 0]), query_start_loc=NS(np=np.array([0, 1, 1])))
    kwargs = {"num_tokens": 1, "num_tokens_padded": 2, "num_reqs": 2,
              "slot_mappings": {0: Tensor([-1, -1])}}
    assert full.dummy_witness(native, kwargs, {})["seq_lens"] == [1, 0]
    kwargs["slot_mappings"] = {0: Tensor([0, -1])}
    with pytest.raises(ValueError, match="padding witness"):
        full.dummy_witness(native, kwargs, {})
    kwargs["slot_mappings"] = {0: Tensor([-1, -1])}
    native.query_start_loc.np[-1] = 2
    with pytest.raises(ValueError, match="padding witness"):
        full.dummy_witness(native, kwargs, {})


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int


@pytest.mark.parametrize("fault", [None, "python_replay", "slots", "missing_metadata", "partial_capture"])
def test_whole_model_capture_then_replay_uses_persistent_events_not_hooks(monkeypatch, fault):
    cuda = NS(capturing=False, width=0, synchronize=lambda: None)

    class Event:
        def __init__(self, *, enable_timing, external):
            assert enable_timing and external

        def record(self):
            assert cuda.capturing

        def elapsed_time(self, end):
            return 0.25  # Synthetic fixture value, never a GPU observation.

    cuda.Event = Event
    cuda.is_current_stream_capturing = lambda: cuda.capturing
    torch = NS(cuda=cuda, inference_mode=nullcontext)
    mode = NS(FULL="FULL")

    class GraphWrapper:
        def __init__(self):
            self.concrete_cudagraph_entries = {}

        def clear_graphs(self):
            self.concrete_cudagraph_entries.clear()

    for name, attributes in {
        "torch": torch.__dict__,
        "vllm.compilation.cuda_graph": {"CUDAGraphWrapper": GraphWrapper},
        "vllm.config": {"CUDAGraphMode": mode},
        "vllm.forward_context": {"get_forward_context": lambda: NS(batch_descriptor=Descriptor(cuda.width))},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(full.runner, "install_dsa_hooks", lambda *a: {"attention_backend": "fixture"})
    monkeypatch.setattr(full.runner, "_STATE", None)
    monkeypatch.setattr(full, "collect_runtime_versions", lambda: {"torch": "mock"})
    monkeypatch.setattr(full, "collect_device_identity", lambda: ("CPU fixture", "none"))
    layers = (NS(global_layer_id=0, attention_variant="full_index", indexer_executed=True),
              NS(global_layer_id=1, attention_variant="shared_index", indexer_executed=False))
    plan = NS(shape=NS(layers=layers))
    native = NS(model=GraphWrapper(), uniform_decode_query_len=1,
                vllm_config=NS(cache_config=NS(cache_dtype="auto")))
    native.cudagraph_dispatcher = NS(dispatch=lambda width, **k: ("FULL", Descriptor(width)))

    def capture():
        cuda.capturing = True
        for width in (2, 4):
            cuda.width = width
            native.model.concrete_cudagraph_entries[Descriptor(width)] = NS(cudagraph=object())
            scopes = ((0, INDEXER_SCOPE), (0, MLA_SCOPE), (1, MLA_SCOPE))
            for layer, scope in (scopes[:-1] if fault == "partial_capture" else scopes):
                full.runner._STATE.timed_call(layer, scope, lambda: None, (), {})
        cuda.capturing = False

    native.capture_model = capture
    native._determine_batch_execution_and_padding = lambda **k: ("FULL", Descriptor(1), False, None, None)
    native._build_attention_metadata = lambda **k: ({"mla": NS(num_actual_tokens=k["num_tokens_padded"])}, None)
    calls = []

    def dummy(tokens, **kwargs):
        assert tokens == 1 and kwargs["cudagraph_runtime_mode"] == "FULL"
        assert kwargs["uniform_decode"] and not kwargs["allow_microbatching"]
        _, desc, *_ = native._determine_batch_execution_and_padding(num_tokens=tokens)
        width = desc.num_tokens
        calls.append(width)
        native.optimistic_seq_lens_cpu = np.array([1] + [0] * (width - 1))
        native.query_start_loc = NS(np=np.array([0] + [1] * width))
        if fault == "python_replay":
            full.runner._STATE.hook_calls += 1
        if fault != "missing_metadata":
            native._build_attention_metadata(num_tokens=1, num_tokens_padded=width, num_reqs=width,
                slot_mappings={0: Tensor([0 if fault == "slots" else -1] * width)})

    native._dummy_run = dummy
    if fault:
        with pytest.raises(ValueError):
            full.measure_worker(NS(model_runner=native), plan, (2, 4), 1, 2, "fixture")
        return
    result = full.measure_worker(NS(model_runner=native), plan, (2, 4), 1, 2, "fixture")
    assert calls == [2, 2, 2, 4, 4, 4]
    assert len(result["records"]) == 12
    assert {r["witness"]["capture_width"] for r in result["records"]} == {2, 4}
    assert {r["time_ms"] for r in result["records"]} == {0.25}
    assert full.runner._STATE.hook_calls == 6  # Capture only, none of six replays.


def test_locked_native_source_contract():
    root = os.environ.get("FRONTIER_VLLM_BS_ROOT")
    if not root:
        pytest.skip("Set FRONTIER_VLLM_BS_ROOT to check the pinned native git objects")

    def source(path):
        return subprocess.check_output(
            ["git", "-C", root, "show", f"{full.NATIVE_REVISION}:{path}"], text=True)

    native = source("vllm/v1/worker/gpu_model_runner.py")
    dummy = native.split("    def _dummy_run(")[1].split("    def _dummy_sampler_run(")[0]
    for fragment in ("sm.fill_(-1)", "num_tokens_padded = batch_desc.num_tokens",
                     "num_tokens=num_tokens_unpadded", "num_reqs=num_reqs_padded",
                     "self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)",
                     "or cudagraph_runtime_mode == CUDAGraphMode.FULL"):
        assert fragment in dummy
    wrapper = source("vllm/compilation/cuda_graph.py")
    assert "entry.cudagraph.replay()" in wrapper
    assert "output = self.runnable(*args, **kwargs)" in wrapper
    dispatcher = source("vllm/v1/cudagraph_dispatcher.py")
    assert "num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)" in dispatcher
