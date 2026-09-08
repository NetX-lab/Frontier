"""Separate linear-op event-construction and surrounding-queue effects on H200."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from types import SimpleNamespace


@contextmanager
def timer_mode(mode, trace_label=None):
    """Change only event object construction order in this diagnostic process."""
    import torch
    from frontier.profiling.common.cuda_timer import CudaTimer

    original_enter, original_exit = CudaTimer.__enter__, CudaTimer.__exit__

    def enter(timer):
        if timer.disabled:
            return original_enter(timer)
        if trace_label:
            timer._diagnostic_scope = torch.profiler.record_function(
                f"frontier_{trace_label}/{timer.name.removeprefix('vidur_')}"
            )
            timer._diagnostic_scope.__enter__()
        if mode == "original":
            return original_enter(timer)
        timer.start_event = torch.cuda.Event(enable_timing=True)
        timer.end_event = torch.cuda.Event(enable_timing=True)
        timer.start_event.record()
        return timer

    def leave(timer, *args):
        if timer.disabled:
            return original_exit(timer, *args)
        try:
            if mode == "original":
                return original_exit(timer, *args)
            timer.end_event.record()
            timer.timer_stats_store.record_time(
                timer.name, [timer.start_event, timer.end_event]
            )
        finally:
            if trace_label:
                timer._diagnostic_scope.__exit__(*args)

    CudaTimer.__enter__, CudaTimer.__exit__ = enter, leave
    try:
        yield
    finally:
        CudaTimer.__enter__, CudaTimer.__exit__ = original_enter, original_exit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if args.samples < 2 or args.rounds < 1:
        parser.error("samples must be >= 2 and rounds must be >= 1")
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    import vllm
    from frontier.profiling.common.model_config import ModelConfig
    from frontier.profiling.linear_op.linear_op_wrapper import LinearOpWrapper
    from frontier.profiling.linear_op.profiling_plan import build_profiling_plan
    from frontier.profiling.moe.moe_wrapper import MoEWrapper
    from frontier.moe_gating_runtime import PREFILL_HOT_MOE_GATING_PREFIX_REPEATS
    from frontier.profiling.utils import build_profile_position_indices

    torch.cuda.set_device(0)
    torch.manual_seed(13)
    model = ModelConfig.from_model_name("qwen3-a3b-30b-moe")
    assert model.dtype == torch.bfloat16, model.dtype
    plan = build_profiling_plan(model, 4, [4], [1], is_moe=True, moe_tp=[1])
    wrapper = LinearOpWrapper(model, 4, "cuda_event", 0, str(args.output), plan)
    store = wrapper.timer_stats_store
    tokens = 4096
    input_ids = torch.randint(
        wrapper.padded_vocab_size // 4, (tokens,), device="cuda", dtype=torch.long
    )
    positions = torch.tensor(build_profile_position_indices(
        num_tokens=tokens, max_position_embeddings=model.max_position_embeddings,
    ), device="cuda", dtype=torch.long)
    # Reuse the existing prefix methods without constructing unrelated MoE kernels.
    prefix = SimpleNamespace(hidden_dim=model.embedding_dim,
                             expert_hidden_dim=model.routed_mlp_hidden_dim,
                             use_gated=model.use_gated_mlp, _dtype=model.dtype)
    MoEWrapper._init_gating_runtime_context_state(prefix)
    prefix_input = torch.randn(tokens, model.embedding_dim, device="cuda", dtype=model.dtype)
    contexts = ("synthetic", "cold_drain", "prefill_hot")
    modes = ("original", "precreated")

    def iteration(context):
        if context == "cold_drain":
            torch.cuda.synchronize()
        elif context == "prefill_hot":
            MoEWrapper._run_prefill_hot_gating_prefix(prefix, prefix_input)
        wrapper.model(input_ids, positions)

    def write(name, data):
        (args.output / name).write_text(json.dumps(data, indent=2) + "\n")

    receipt = {
        "status": "RUNNING", "python": platform.python_version(),
        "python_executable": sys.executable, "torch": torch.__version__,
        "cuda": torch.version.cuda, "vllm": vllm.__version__, "vllm_path": vllm.__file__,
        "gpu": torch.cuda.get_device_name(), "tokens": tokens, "tp": 4,
        "hidden": model.embedding_dim, "q_heads": model.num_q_heads,
        "kv_heads": model.num_kv_heads, "dtype": str(model.dtype),
        "use_qk_norm": model.use_qk_norm, "inference_mode": True,
        "prefix_repeats": PREFILL_HOT_MOE_GATING_PREFIX_REPEATS,
        "prefix_width": prefix.expert_hidden_dim, "seed": 13,
        "frontier_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True,
        ).strip(), "profiling_plan": plan,
        "limits": "Context controls are synthetic, not a vLLM forward. Precreated changes Event object construction, not lazy CUDA event initialization. Kernel-profile pass is diagnostic only.",
    }
    # The plan can contain enum values; retain its structure and labels explicitly.
    receipt["profiling_plan"] = json.loads(json.dumps(plan, default=str))
    write("receipt.json", receipt)
    rows = []
    with torch.inference_mode():
        for round_index in range(args.rounds):
            for context in contexts if round_index % 2 == 0 else reversed(contexts):
                for mode in modes if round_index % 2 == 0 else reversed(modes):
                    torch.manual_seed(17)
                    with timer_mode(mode):
                        store.clear_stats()
                        for warmup_index in range(3):
                            iteration(context)
                            observed = {op: len(pairs) for op, pairs in store.TIMING_STATS.items()}
                            if warmup_index == 0:
                                calls_per_iteration = observed
                            assert observed == {op: count * (warmup_index + 1)
                                                for op, count in calls_per_iteration.items()}, observed
                        torch.cuda.synchronize()
                        for op in ("attn_pre_proj", "attn_rope", "attn_post_proj"):
                            assert calls_per_iteration.get(op) == 1, (op, calls_per_iteration)
                        store.clear_stats()
                        for _ in range(args.samples):
                            iteration(context)
                        torch.cuda.synchronize()
                    assert store.TIMING_STATS.keys() == calls_per_iteration.keys()
                    for op, pairs in store.TIMING_STATS.items():
                        values = [start.elapsed_time(end) for start, end in pairs]
                        count = calls_per_iteration[op]
                        assert len(values) == args.samples * count and all(value > 0 for value in values), (op, values)
                        rows.append(dict(round=round_index, context=context, timer=mode,
                                         op=op, calls_per_iteration=count, samples_ms=values,
                                         samples_by_call_ms=[values[index::count] for index in range(count)],
                                         median_ms=statistics.median(values),
                                         mean_ms=statistics.mean(values), min_ms=min(values), max_ms=max(values)))
                    store.clear_stats()
                    write("event_samples.json", rows)
                    print(json.dumps({"round": round_index, "context": context, "timer": mode}), flush=True)

        # One profiler lifecycle avoids repeatedly restarting CUPTI for each condition.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True) as profiler:
            for context in contexts:
                for mode in modes:
                    torch.manual_seed(17)
                    label = f"{context}/{mode}"
                    with timer_mode(mode, label):
                        for _ in range(2):
                            with torch.profiler.record_function(f"frontier_{label}/surrounding"):
                                iteration(context)
                    torch.cuda.synchronize()
                    store.clear_stats()
        trace_path = args.output / "kernel_trace.json"
        profiler.export_chrome_trace(str(trace_path))

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e2e"))
    from issue26_first_batch_op_rca_vllm import kernel_summary
    summary, activities = kernel_summary(trace_path)
    families = defaultdict(lambda: {"count": 0, "duration_ms": 0.0})
    for activity in activities:
        family = families[(activity["scope"], activity["category"], activity["name"])]
        family["count"] += 1
        family["duration_ms"] += activity["duration_ms"]
    write("kernel_families.json", [dict(scope=scope, category=category, name=name, **values)
                                   for (scope, category, name), values in families.items()])
    write("kernel_activities.json", activities)
    write("kernel_summary.json", summary)
    events = json.loads(trace_path.read_text())["traceEvents"]
    correlations = {event.get("args", {}).get("correlation") for event in events if event.get("cat") == "kernel"}
    launches = [event for event in events if event.get("cat") in {"cuda_runtime", "cuda_driver"}
                and "LaunchKernel" in event.get("name", "")]
    missing = [event for event in launches if event.get("args", {}).get("correlation") not in correlations]
    write("kernel_coverage.json", dict(kernel_launches=len(launches), missing_device_correlations=missing))
    receipt["status"] = "COMPLETE_DIAGNOSTIC" if not missing else "PARTIAL_KERNEL_COVERAGE"
    receipt["event_row_count"] = len(rows)
    write("receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
