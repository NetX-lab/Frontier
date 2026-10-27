"""Host-step probe contracts against CPU stand-ins of the native vLLM modules and torch."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

BOOTSTRAP = Path(__file__).resolve().parents[2] / "tests/integration/step5_pd/host_probe_bootstrap"
sys.path.insert(0, str(BOOTSTRAP))
import host_step_probe  # noqa: E402

# A recorded event completes once a host `.tolist()` or a synchronize waited for the device.
FAKE_TORCH = """
DEVICE = {"recorded": 0, "done": 0}

def synchronize():
    DEVICE["done"] = DEVICE["recorded"]

class Tensor:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def __len__(self):
        return len(self.values)

    def tolist(self):
        DEVICE["done"] = DEVICE["recorded"]
        return list(self.values)

class cuda:
    class Event:
        def __init__(self, enable_timing=False):
            assert enable_timing
            self.time = None

        def record(self):
            DEVICE["recorded"] += 1
            self.time = DEVICE["recorded"]

        def query(self):
            return self.time is None or self.time <= DEVICE["done"]

        def elapsed_time(self, other):
            assert self.query() and other.query()
            return float(other.time - self.time)
"""

FAKE_NATIVE = {
    "torch/distributed.py": """
CALLS = []

def all_reduce(tensor, group=None):
    CALLS.append(group)
""",
    "vllm/__init__.py": "",
    "vllm/distributed/__init__.py": "",
    "vllm/distributed/parallel_state.py": """
def get_tensor_model_parallel_rank():
    return 3

def get_tensor_model_parallel_world_size():
    return 8
""",
    "vllm/models/__init__.py": "",
    "vllm/models/step5/__init__.py": "",
    "vllm/models/step5/nvidia/__init__.py": "",
    "vllm/models/step5/nvidia/sparse_summary_cache.py": """
import torch

RESETS = []

class Step5SparseSummaryCacheLayer:
    def zero_kv_cache_side_storage(self, block_ids):
        RESETS.append(self._ids(block_ids))

    def release_kv_cache_side_storage(self, block_ids):
        self._ids(block_ids)

    @staticmethod
    def _ids(block_ids):
        return block_ids.tolist() if isinstance(block_ids, torch.Tensor) else block_ids
""",
    "vllm/v1/__init__.py": "",
    "vllm/v1/worker/__init__.py": "",
    "vllm/v1/worker/utils.py": """
import torch

class KVBlockZeroer:
    def __init__(self, hooks):
        self._side_storage_zero_hooks = hooks

    def zero_block_ids(self, block_ids, block_ids_by_group=None):
        flat = torch.Tensor([block for ids in block_ids_by_group for block in ids])
        offset = 0
        for group_id, ids in enumerate(block_ids_by_group):
            for hook in self._side_storage_zero_hooks.get(group_id, ()):
                if ids:
                    hook(flat[offset:offset + len(ids)])
            offset += len(ids)
""",
    "vllm/v1/worker/gpu/__init__.py": "",
    "vllm/v1/worker/gpu/dp_utils.py": """
import time

import torch.distributed as dist

def sync_cudagraph_and_dp_padding(num_tokens):
    time.sleep(0.005)  # local tensor preparation
    dist.all_reduce([num_tokens], group="dp_cpu")
    time.sleep(0.005)  # local graph dispatch
    return ("synced", num_tokens)

def dispatch_cg_and_sync_dp(num_tokens):
    return sync_cudagraph_and_dp_padding(num_tokens)
""",
    "vllm/v1/worker/gpu/async_utils.py": """
class StepTimingCollector:
    def forward_start(self):
        pass
""",
    "vllm/v1/worker/gpu/model_runner.py": """
from types import SimpleNamespace

import torch
from vllm.v1.worker.gpu.async_utils import StepTimingCollector
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp

class GPUModelRunner:
    def __init__(self, zeroer):
        self.kv_block_zeroer = zeroer
        self.step_timing = StepTimingCollector()

    @property
    def dp_rank(self):
        return 1

    @property
    def dp_size(self):
        return 2

    def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False,
                      skip_attn_for_dummy_run=False, is_profile=False):
        if not dummy_run:
            if scheduler_output.new_block_ids_to_zero_by_group:
                self.kv_block_zeroer.zero_block_ids([], scheduler_output.new_block_ids_to_zero_by_group)
            if scheduler_output.total_num_scheduled_tokens == 0:
                return "no_forward"
        if dispatch_cg_and_sync_dp(scheduler_output.total_num_scheduled_tokens)[0] != "synced":
            raise AssertionError("dispatch did not return the native agreement")
        self.step_timing.forward_start()
        return None

    def sample_tokens(self, grammar_output):
        return ("sampled", grammar_output)

    def _dummy_run(self, num_tokens, is_profile=False):
        self.execute_model(SimpleNamespace(num_scheduled_tokens={"dummy": num_tokens},
                                           total_num_scheduled_tokens=num_tokens),
                           dummy_run=True, is_profile=is_profile)
        torch.cuda.Event(enable_timing=True).record()  # hidden_states[logits_indices]
        return ("dummy_ran", num_tokens)

    def shutdown(self):
        torch.synchronize()
""",
}

DRIVER = """
import json
import os
import sys
from types import SimpleNamespace

import torch.distributed
from vllm.models.step5.nvidia.sparse_summary_cache import RESETS, Step5SparseSummaryCacheLayer
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.utils import KVBlockZeroer

layers = [Step5SparseSummaryCacheLayer() for _ in range(3)]
zeroer = KVBlockZeroer({1: [layer.zero_kv_cache_side_storage for layer in layers]})
runner = GPUModelRunner(zeroer)

def output(counts, groups=None):
    return SimpleNamespace(num_scheduled_tokens=counts, total_num_scheduled_tokens=sum(counts.values()),
                           new_block_ids_to_zero_by_group=groups)

zeroer.zero_block_ids([], [[], [0]])  # startup warmup outside any step
results = [
    runner._dummy_run(1, is_profile=True),
    runner.execute_model(output({"a": 1, "b": 1}, [[7], [8, 9]])),
    runner.sample_tokens("grammar"),
    runner.execute_model(output({"a": 1, "b": 1})),
    runner.sample_tokens(None),
    runner.execute_model(output({}, [[], [10]])),
    runner._dummy_run(1),
]
layers[0].release_kv_cache_side_storage([11])
torch.distributed.all_reduce([0], group="other")
results.append(runner.shutdown())
print(json.dumps({"results": results, "resets": RESETS, "all_reduces": torch.distributed.CALLS,
                  "global_all_reduce_probed": hasattr(torch.distributed.all_reduce, "_step5_host_probe")}))
sys.stdout.flush()
os._exit(0)  # a forked worker never runs atexit
"""


def _fake_tree(root):
    (root / "torch").mkdir()
    (root / "torch/__init__.py").write_text(FAKE_TORCH)
    for relative, text in FAKE_NATIVE.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(textwrap.dedent(text))


def _run(tmp_path, setting):
    _fake_tree(tmp_path)
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment.update(PYTHONPATH=os.pathsep.join([str(BOOTSTRAP), str(tmp_path)]),
                       STEP5_HOST_STEP_PROBE=setting)
    return subprocess.run([sys.executable, "-B", "-c", DRIVER], cwd=tmp_path, env=environment,
                          capture_output=True, text=True, timeout=60)


def test_probe_records_steps_and_leaves_native_results_unchanged(tmp_path):
    output_dir = tmp_path / "probe"
    result = _run(tmp_path, json.dumps({"role": "decode", "output_dir": str(output_dir)}))
    assert result.returncode == 0, result.stderr
    assert "UNVERIFIED" not in result.stderr
    native = json.loads(result.stdout)
    assert native["results"] == [["dummy_ran", 1], None, ["sampled", "grammar"], None, ["sampled", None],
                                 "no_forward", ["dummy_ran", 1], None]
    assert native["resets"] == [[0]] * 3 + [[8, 9]] * 3 + [[10]] * 3
    # One native all_reduce per sync; only the DP sync module's own name is probed.
    assert native["all_reduces"] == ["dp_cpu"] * 4 + ["other"]
    assert native["global_all_reduce_probed"] is False
    [path] = output_dir.glob("decode_*.jsonl")
    header, *steps, end = [json.loads(line) for line in path.read_text().splitlines()]
    assert {key: header[key] for key in ("kind", "role", "tp_rank", "tp_size", "dp_rank", "dp_size")} == {
        "kind": "header", "role": "decode", "tp_rank": 3, "tp_size": 8, "dp_rank": 1, "dp_size": 2}
    # Shutdown drained every step although the process left through os._exit.
    assert end == {"kind": "end", "steps": 5}
    assert [step["seq"] for step in steps] == [1, 2, 3, 4, 5]
    profile, reset, plain, empty, final = steps
    assert (profile["dummy"], profile["profile"], profile["reqs"], profile["sync"]["ordinal"]) == (True, True, 1, 1)
    # Events: launch 1, the dummy tail 2, end 3; the end follows the tail.
    assert profile["span_ms"] == 2.0 and "zero" not in profile and "sample_exit" not in profile
    # The all_reduce stamps exclude the local work before and after it.
    for step in profile, reset, plain, final:
        sync = step["sync"]
        assert sync["reduce_enter"] - sync["enter"] >= 5e6 and sync["exit"] - sync["reduce_exit"] >= 5e6
    zero = reset["zero"]
    assert (zero["blocks"], zero["groups"], zero["hooks"], zero["prior_pending"]) == (0, [1, 2], 3, True)
    assert zero["enter"] <= zero["first_enter"] <= zero["first_sync"] <= zero["first_exit"] <= zero["exit"]
    assert zero["exit"] <= reset["sync"]["enter"] <= reset["sync"]["exit"] <= reset["launch"] <= reset["exit"]
    assert reset["exit"] <= reset["sample_exit"]
    assert (reset["sync"]["ordinal"], reset["reqs"], reset["tokens"], reset["dummy"]) == (2, 2, 2, False)
    # Events 4-7 complete at the empty step's reset; the next step writes their device times.
    assert reset["gap_ms"] == 1.0 and reset["span_ms"] == 1.0
    assert "zero" not in plain and plain["sync"]["ordinal"] == 3
    assert plain["gap_ms"] == 1.0 and plain["span_ms"] == 1.0
    assert empty["zero"]["hooks"] == 3 and empty["zero"]["prior_pending"] is True
    assert not {"sync", "launch", "sample_exit", "gap_ms", "span_ms"} & set(empty)
    # The last dummy's events (launch 8, tail 9, end 10) complete only at the shutdown synchronize.
    assert final["dummy"] and final["sync"]["ordinal"] == 4
    assert final["gap_ms"] == 1.0 and final["span_ms"] == 2.0


def test_bootstrap_failure_never_blocks_the_native_process(tmp_path):
    result = _run(tmp_path, "{not json")
    assert result.returncode == 0
    assert "STEP5_HOST_STEP_PROBE UNVERIFIED bootstrap" in result.stderr
    assert json.loads(result.stdout)["resets"] == [[0]] * 3 + [[8, 9]] * 3 + [[10]] * 3


class _Native:
    calls = 0

    def step(self, value, fail=False):
        _Native.calls += 1
        if fail:
            raise KeyError("native")
        return value * 2


def test_a_probe_error_disables_the_probe_and_keeps_native_behavior(tmp_path, capsys):
    probe = host_step_probe.Probe({"role": "prefill", "output_dir": str(tmp_path)})
    seen = []

    def before(arguments):
        seen.append(dict(arguments))
        if arguments["value"] == 2:
            raise RuntimeError("probe bug")

    host_step_probe.wrap(_Native, "step", probe, before=before, bind=True)
    wrapped = _Native.step
    host_step_probe.wrap(_Native, "step", probe, before=before, bind=True)
    assert _Native.step is wrapped
    native = _Native()
    assert native.step(1) == 2
    with pytest.raises(KeyError, match="native"):
        native.step(3, fail=True)
    assert native.step(2) == 4
    assert native.step(5) == 10
    assert _Native.calls == 4
    assert [entry["value"] for entry in seen] == [1, 3, 2]
    assert probe.disabled
    assert "STEP5_HOST_STEP_PROBE UNVERIFIED step: RuntimeError('probe bug')" in capsys.readouterr().err


def test_a_forked_worker_starts_its_own_record_file(tmp_path, monkeypatch):
    fake = tmp_path / "fake"
    fake.mkdir()
    _fake_tree(fake)
    monkeypatch.syspath_prepend(str(fake))
    for name in [name for name in sys.modules if name == "torch" or name.startswith("vllm")]:
        monkeypatch.delitem(sys.modules, name)
    from types import SimpleNamespace

    probe = host_step_probe.Probe({"role": "prefill", "output_dir": str(tmp_path / "probe")})
    runner = SimpleNamespace(dp_rank=0, dp_size=2)
    step = {"self": runner, "scheduler_output": SimpleNamespace(
        num_scheduled_tokens={"a": 4}, total_num_scheduled_tokens=4)}
    try:
        for pid in (100, 100, 200):
            monkeypatch.setattr(host_step_probe.os, "getpid", lambda: pid)
            probe.step_after(probe.step_before(step))
    finally:
        for name in [name for name in sys.modules if name == "torch" or name.startswith("vllm")]:
            del sys.modules[name]
    assert probe.pid == 200 and probe.sequence == 1
    probe.close(None)
    probe.close(None)
    records = {path.name.split("_")[-1]: path.read_text().splitlines()
               for path in (tmp_path / "probe").glob("prefill_*.jsonl")}
    assert sorted(records) == ["100.jsonl", "200.jsonl"]
    assert len(records["100.jsonl"]) == 2
    assert [json.loads(line)["kind"] for line in records["200.jsonl"]] == ["header", "step", "end"]
