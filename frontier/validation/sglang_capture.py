"""Capture reproducible static batches using SGLang's real checkpoint runner.

No SGLang source changes are required. This opt-in benchmark uses its one_batch
allocation/forward helpers; it does not simulate a serving scheduler.
"""

from __future__ import annotations

import argparse
import dataclasses
from contextlib import ExitStack
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from frontier.validation.batch_record import BatchRecord, read_batches, validate_rank_cohorts


def _host_list(value, name: str) -> list[int]:
    if value is None:
        raise ValueError(f"SGLang did not provide {name} on the CPU")
    if hasattr(value, "device") and value.device.type != "cpu":
        raise ValueError(f"{name} must be CPU-resident; capture must not copy GPU tensors")
    return [int(v) for v in (value.tolist() if hasattr(value, "tolist") else value)]


def snapshot_forward_batch(batch) -> dict:
    """Read the pinned runtime's CPU metadata without synchronizing a device."""
    size = int(batch.batch_size)
    if batch.forward_mode.is_decode():
        query = [1] * size
        context = [v - 1 for v in _host_list(batch.seq_lens_cpu, "seq_lens_cpu")]
        phase = "decode"
    elif batch.forward_mode.is_extend() and not batch.forward_mode.is_mixed():
        query = _host_list(batch.extend_seq_lens_cpu, "extend_seq_lens_cpu")
        context = _host_list(batch.extend_prefix_lens_cpu, "extend_prefix_lens_cpu")
        phase = "prefill"
    else:
        raise ValueError(f"Static capture does not support forward mode {batch.forward_mode}")
    if len(query) != size or len(context) != size:
        raise ValueError("SGLang CPU lengths disagree with batch_size")
    return dict(phase=phase, query_lens=tuple(query), context_lens=tuple(context),
                prefill_mask=(phase == "prefill",) * size)


class ForwardObserver:
    """Measure only model forward on the GPU; sample/prepare remain step work."""

    def __init__(self, runner):
        import torch

        self.runner = runner
        self.original = runner.forward
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.snapshot = None
        self.enabled = False
        runner.forward = self.forward

    def forward(self, forward_batch, *args, **kwargs):
        if not self.enabled:
            return self.original(forward_batch, *args, **kwargs)
        self.snapshot = snapshot_forward_batch(forward_batch)
        self.start.record()
        output = self.original(forward_batch, *args, **kwargs)
        self.end.record()
        graph = bool(output.can_run_graph)
        graph_runner = self.runner.decode_cuda_graph_runner
        self.snapshot.update(
            graph_mode="FULL" if graph else "NONE",
            capture_size=int(graph_runner.bs) if graph else 0,
        )
        return output


def _git_revision(path: str | Path) -> str | None:
    directory = Path(path).resolve()
    directory = next((p for p in (directory, *directory.parents) if (p / ".git").exists()), directory)
    result = subprocess.run(["git", "-c", f"safe.directory={directory}",
                             "-C", str(directory), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def decode_trajectory(token_ids, batch_size: int, output_len: int):
    """Return decode input rows and a content digest, never write token IDs.

    A normal first repetition provides one fixed trajectory per workload.
    The final generated row is excluded because no forward consumes it.
    Tensor allocation/copies happen outside all measured steps.
    """
    if (batch_size < 1 or output_len < 2 or len(token_ids) != batch_size * output_len
            or any(type(v) is not int or v < 0 for v in token_ids)):
        raise ValueError("Decode trajectory requires complete nonnegative integer token rows")
    rows = tuple(tuple(token_ids[i:i + batch_size])
                 for i in range(0, batch_size * (output_len - 1), batch_size))
    digest = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    return rows, digest


def _worker(rank, server_args, port_args, options):
    import numpy as np
    import torch
    import torch.distributed as dist
    from sglang.benchmark import one_batch as bench

    bench.initialize_moe_config(server_args)
    bench.initialize_fp8_gemm_config(server_args)
    bench.initialize_fp4_gemm_config(server_args)
    if bench.get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        bench.set_gpu_proc_affinity(1, server_args.tp_size, 1, rank)
    bench.configure_logger(server_args, prefix=f" TP{rank}")
    runner, _tokenizer = bench.load_model(server_args, port_args, rank, rank)
    actual = runner.torch_runner
    observer = ForwardObserver(actual)
    output_dir = Path(options["output_dir"])
    output_checks = []
    graph_runner = actual.decode_cuda_graph_runner
    properties = torch.cuda.get_device_properties(rank)
    runtime = {
        "rank": rank, "device_name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "gcn_arch_name": getattr(properties, "gcnArchName", None),
        "allocated_bytes_after_init": torch.cuda.memory_allocated(rank),
        "reserved_bytes_after_init": torch.cuda.memory_reserved(rank),
        "max_total_num_tokens": actual.max_total_num_tokens,
        "request_pool_size": getattr(actual.req_to_token_pool, "size", None),
        "capture_sizes": list(getattr(graph_runner, "capture_bs", [])),
    }
    with (output_dir / f"runtime-rank{rank}.json").open("x") as stream:
        json.dump(runtime, stream, indent=2)

    graph_observer = None
    graph_samples = []
    decoder_graph_observer = None
    decoder_graph_samples = []
    frozen_inputs, frozen_digests = {}, {}
    routing_observer, routing_stream = None, None

    def run_case(bs, length, repetition, *, measured, profiled=False, operator_events=False,
                 graph_events=False, routing=False):
        # Each shape/repetition sees identical prompts on all TP ranks. Cache
        # pools are reset outside timing and no prefix-cache lookup is used.
        np.random.seed(options["seed"])
        torch.manual_seed(options["seed"])
        reqs = bench.prepare_synthetic_inputs_for_latency_test(bs, length)
        runner.clear()
        observer.enabled = measured
        batch = None
        next_ids = None
        records = []
        routing_snapshots = []
        digest = hashlib.sha256()
        token_ids = []
        steps = (options["output_len"] if graph_events or routing else 1 if operator_events else
                 options["trace_steps"] + 1 if profiled else options["output_len"])
        for step in range(steps):
            suffix = ("-graph-events" if graph_events else "-routing" if routing else "-operator-events" if operator_events
                      else "-trace" if profiled else "")
            identifier = f"b{bs}-l{length}-r{repetition}-s{step}" + suffix
            decode_ids = (frozen_inputs[(bs, length)][step - 1]
                          if step > 0 and (bs, length) in frozen_inputs else next_ids)
            runner.synchronize()
            # Synchronization is inside the trace marker so its GPU interval is
            # unambiguous even when HIP graph launches omit correlation IDs.
            with torch.profiler.record_function(f"frontier.batch:{identifier}"):
                tic = time.perf_counter()
                if step == 0:
                    next_ids, logits, batch = runner.extend(reqs)
                else:
                    next_ids, logits = runner.decode(decode_ids, batch)
                runner.synchronize()
                wall_ms = (time.perf_counter() - tic) * 1000
            if measured:
                records.append(BatchRecord(
                    batch_id=identifier, rank=rank,
                    request_ids=tuple(str(req.rid) for req in reqs),
                    prompt_lens=tuple(len(req.origin_input_ids) for req in reqs),
                    repetition=repetition, step=step, profiled=profiled,
                    decode_input_sha256=frozen_digests.get((bs, length)) if step > 0 else None,
                    forward_gpu_ms=observer.start.elapsed_time(observer.end),
                    step_wall_ms=wall_ms, **observer.snapshot,
                ))
                if graph_events and step > 0:
                    if records[-1].graph_mode != "FULL":
                        raise ValueError("Graph-event validation must execute a full decode graph")
                    if graph_observer is not None:
                        graph_samples.extend({"batch_id": identifier, "rank": rank, **sample}
                            for sample in graph_observer.collect_graph(records[-1].capture_size))
                    if decoder_graph_observer is not None:
                        decoder_graph_samples.append({"batch_id": identifier, "rank": rank,
                            **decoder_graph_observer.collect_graph(records[-1].capture_size)})
                if routing and step > 0:
                    record = records[-1]
                    if record.graph_mode != "FULL":
                        raise ValueError("Routing observation requires a full decode graph")
                    routing_snapshots.append(routing_observer.snapshot_graph(
                        physical_size=record.capture_size, logical_size=bs, batch_id=identifier, rank=rank))
            # Only digests are retained; the ledger contains no prompt/output text.
            tokens = next_ids.detach().cpu().numpy()
            digest.update(tokens.tobytes())
            token_ids.extend(int(v) for v in tokens)
            if step == 0 and not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Checkpoint forward returned nonfinite logits")
        observer.enabled = False
        runner.cleanup(batch)
        # Only the necessary device-to-host snapshots happen between steps.
        # CPU validation, hashing, histograms and compression wait until the
        # entire repetition ends, avoiding long serialization gaps in decode.
        for snapshot in routing_snapshots:
            for sample in routing_observer.summarize_snapshot(snapshot):
                routing_stream.write(json.dumps(sample.to_dict(), separators=(",", ":")) + "\n")
        return records, digest.hexdigest(), token_ids

    operator_records = []
    operator_samples = []
    with (output_dir / f"batches-rank{rank}.jsonl").open("x") as ledger:
        for bs, length in itertools.product(options["batch_size"], options["input_len"]):
            capacity = min(runner.max_batch_size(length, options["output_len"]),
                           getattr(actual.req_to_token_pool, "size", bs))
            if bs > capacity:
                raise ValueError(f"Requested batch {bs} exceeds runtime capacity {capacity}")
            for warmup in range(options["warmups"]):
                run_case(bs, length, warmup, measured=False)
            reference_tokens = None
            for repetition in range(options["repetitions"]):
                records, digest, tokens = run_case(bs, length, repetition, measured=True)
                if options["freeze_decode_inputs"] and (bs, length) not in frozen_inputs:
                    trajectory, input_digest = decode_trajectory(tokens, bs, options["output_len"])
                    frozen_inputs[(bs, length)] = torch.tensor(
                        trajectory, dtype=torch.int64, device=f"cuda:{rank}")
                    frozen_digests[(bs, length)] = input_digest
                    # The first repetition generated the very trajectory that
                    # subsequent repetitions consume; include it in the cohort.
                    records = [dataclasses.replace(r, decode_input_sha256=input_digest)
                               if r.phase == "decode" else r for r in records]
                if reference_tokens is None:
                    reference_tokens = tokens
                differing = sum(a != b for a, b in zip(tokens, reference_tokens))
                if server_args.tp_size > 1:
                    digests = [None] * server_args.tp_size
                    dist.all_gather_object(digests, digest)
                    if len(set(digests)) != 1:
                        raise RuntimeError("TP ranks generated different output digests")
                output_checks.append({"batch_size": bs, "input_len": length,
                    "repetition": repetition, "output_sha256": digest,
                    "decode_input_sha256": frozen_digests.get((bs, length)),
                    "output_tokens": len(tokens), "different_from_first_repeat": differing,
                    "tp_output_agreement": True})
                for record in records:
                    ledger.write(json.dumps(record.to_dict()) + "\n")
                ledger.flush()
                if rank == 0:
                    decode = [r.forward_gpu_ms for r in records if r.phase == "decode"]
                    print(f"capture b={bs} l={length} repeat={repetition}: "
                          f"prefill={records[0].forward_gpu_ms:.3f}ms "
                          f"decode={np.median(decode):.3f}ms "
                          f"changed_tokens={differing}/{len(tokens)}", flush=True)
            if options["trace"]:
                # Profiling is a separate pass, excluded from all timing summaries.
                profiler = None
                if rank in options["trace_ranks"]:
                    profiler = torch.profiler.profile(activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ])
                    profiler.start()
                records, _, _ = run_case(bs, length, 0, measured=True, profiled=True)
                if profiler is not None:
                    profiler.stop()
                    profiler.export_chrome_trace(str(
                        output_dir / f"b{bs}-l{length}-rank{rank}.trace.json.gz"
                    ))
                for record in records:
                    ledger.write(json.dumps(record.to_dict()) + "\n")
                ledger.flush()
            if options["operator_event_layers"]:
                from frontier.validation.sglang_operator_events import OperatorEventObserver

                # Install after the baseline and trace passes. Graphs remain
                # untouched; only eager prefill executes these Python scopes.
                with OperatorEventObserver(actual.model, options["operator_event_layers"]) as scopes:
                    for repetition in range(options["operator_event_repetitions"]):
                        records, _, _ = run_case(bs, length, repetition, measured=True,
                                                 profiled=True, operator_events=True)
                        if records[0].graph_mode != "NONE":
                            raise ValueError("Operator-event capture requires eager prefill")
                        samples = scopes.collect()
                        if not samples:
                            raise ValueError("Operator-event scopes did not execute")
                        operator_records.extend(records)
                        operator_samples.extend({"batch_id": records[0].batch_id, "rank": rank, **s}
                                                for s in samples)
    if options["operator_event_layers"]:
        with (output_dir / f"operator-batches-rank{rank}.jsonl").open("x") as stream:
            for record in operator_records:
                stream.write(json.dumps(record.to_dict()) + "\n")
        with (output_dir / f"operator-events-rank{rank}.jsonl").open("x") as stream:
            for sample in operator_samples:
                stream.write(json.dumps(sample) + "\n")
    capture_routing = bool(options["routing_layers"] or options["routing_all_layers"])
    # Routing-only and timing+routing are separate passes. The untouched
    # original graphs remain the unprofiled timing baseline.
    graph_events_enabled = bool(options["graph_event_layers"] or options["graph_decoder_event"])
    for pass_name in (["routing"] if capture_routing else []) + (["graph"] if graph_events_enabled else []):
        from frontier.validation.sglang_graph_events import (
            GraphDecoderEventObserver, GraphOperatorEventObserver,
        )

        # Only replace the decode runner after every ordinary baseline/trace
        # pass has finished. No production server or persistent runtime source
        # is changed. Keep event handles alive until this worker exits.
        with ExitStack() as stack:
            if pass_name == "graph":
                if options["graph_event_layers"]:
                    graph_observer = stack.enter_context(GraphOperatorEventObserver(actual.model,
                        options["graph_event_layers"], components=options["graph_event_components"]))
                if options["graph_decoder_event"]:
                    decoder_graph_observer = stack.enter_context(
                        GraphDecoderEventObserver(actual.model))
            if capture_routing:
                from frontier.validation.sglang_routing import RoutingGraphObserver
                routing_observer = stack.enter_context(RoutingGraphObserver(actual.model,
                    None if options["routing_all_layers"] else options["routing_layers"]))
            actual.init_cuda_graphs()
        if pass_name == "graph":
            if graph_observer is not None:
                graph_observer.validate_captures()
            if decoder_graph_observer is not None:
                decoder_graph_observer.validate_captures()
        if capture_routing:
            routing_observer.validate_captures()
            routing_stream = gzip.open(output_dir / f"{pass_name}-routing-rank{rank}.jsonl.gz", "xt")
            if rank == 0:
                print(f"routing capture: {len(routing_observer.layers)} layers, pass={pass_name}", flush=True)
        graph_records, graph_checks = [], []
        for bs, length in itertools.product(options["batch_size"], options["input_len"]):
            for warmup in range(options["warmups"]):
                run_case(bs, length, warmup, measured=False, graph_events=pass_name == "graph", routing=capture_routing)
            reference_tokens = None
            for repetition in range(options["graph_event_repetitions"] if pass_name == "graph"
                                    else options["routing_repetitions"]):
                records, digest, tokens = run_case(bs, length, repetition, measured=True,
                    profiled=True, graph_events=pass_name == "graph", routing=capture_routing)
                if reference_tokens is None:
                    reference_tokens = tokens
                differing = sum(a != b for a, b in zip(tokens, reference_tokens))
                if server_args.tp_size > 1:
                    digests = [None] * server_args.tp_size
                    dist.all_gather_object(digests, digest)
                    if len(set(digests)) != 1:
                        raise RuntimeError("Instrumented TP ranks generated different outputs")
                graph_records.extend(records)
                graph_checks.append({"batch_size": bs, "input_len": length, "repetition": repetition,
                    "output_sha256": digest, "output_tokens": len(tokens),
                    "decode_input_sha256": frozen_digests.get((bs, length)),
                    "different_from_first_repeat": differing, "tp_output_agreement": True})
                if rank == 0:
                    print(f"{pass_name} b={bs} l={length} repeat={repetition}: "
                          f"decode={np.median([r.forward_gpu_ms for r in records[1:]]):.3f}ms",
                          flush=True)
        if routing_stream is not None:
            routing_stream.close()
            routing_stream = None
        for name, rows in ((f"{pass_name}-batches", [r.to_dict() for r in graph_records]),
                           *([("graph-events", graph_samples)]
                             if pass_name == "graph" and graph_observer is not None else []),
                           *([("decoder-events", decoder_graph_samples)]
                             if pass_name == "graph" and decoder_graph_observer is not None else [])):
            with (output_dir / f"{name}-rank{rank}.jsonl").open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
        with (output_dir / f"{pass_name}-output-checks-rank{rank}.json").open("x") as stream:
            json.dump(graph_checks, stream, indent=2)
    with (output_dir / f"output-checks-rank{rank}.json").open("x") as stream:
        json.dump(output_checks, stream, indent=2)
    if server_args.tp_size > 1:
        bench.destroy_model_parallel()
        bench.destroy_distributed_environment()


def main():
    from sglang.benchmark import one_batch as bench
    import torch
    import torch.multiprocessing as mp
    import sglang
    import aiter

    parser = argparse.ArgumentParser(description=__doc__)
    bench.ServerArgs.add_cli_args(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[16, 24, 32])
    parser.add_argument("--input-len", type=int, nargs="+", default=[1024])
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--capture-warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-decode-inputs", action="store_true",
                        help="Replay the first baseline's tokens as fixed decode inputs for each shape")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-ranks", type=int, nargs="+", default=[0])
    parser.add_argument("--trace-steps", type=int, default=3)
    parser.add_argument("--operator-event-layers", type=int, nargs="+", default=[])
    parser.add_argument("--operator-event-repetitions", type=int, default=3)
    parser.add_argument("--graph-event-layers", type=int, nargs="+", default=[])
    parser.add_argument("--graph-decoder-event", action="store_true",
                        help="Add one HIP event pair around the full decoder layer sequence")
    parser.add_argument("--graph-event-repetitions", type=int, default=3)
    routing_group = parser.add_mutually_exclusive_group()
    routing_group.add_argument("--routing-layers", type=int, nargs="+", default=[])
    routing_group.add_argument("--routing-all-layers", action="store_true")
    parser.add_argument("--routing-repetitions", type=int, default=3)
    from frontier.validation.sglang_graph_events import DEFAULT_COMPONENTS
    parser.add_argument("--graph-event-components", nargs="+", default=list(DEFAULT_COMPONENTS))
    args = parser.parse_args()
    if (min(args.batch_size + args.input_len) <= 0 or args.output_len < 2
            or args.capture_warmups < 1 or args.repetitions < 1 or args.trace_steps < 1):
        parser.error("Positive shapes, warmups/repetitions/trace-steps and output-len >=2 required")
    if len(set(args.batch_size)) != len(args.batch_size) or len(set(args.input_len)) != len(args.input_len):
        parser.error("Duplicate shapes would produce duplicate batch IDs")
    if args.freeze_decode_inputs and args.trace and args.trace_steps >= args.output_len:
        parser.error("Fixed-input trace steps must fit the baseline decode trajectory")
    if (args.routing_repetitions < 1 or any(v < 0 for v in args.routing_layers)
            or len(set(args.routing_layers)) != len(args.routing_layers)):
        parser.error("Routing requires unique nonnegative layers and positive repetitions")
    if (args.routing_layers or args.routing_all_layers) and not args.freeze_decode_inputs:
        parser.error("Routing comparison requires --freeze-decode-inputs")
    for prefix in ("operator", "graph"):
        layers = getattr(args, f"{prefix}_event_layers")
        if (getattr(args, f"{prefix}_event_repetitions") < 1 or any(v < 0 for v in layers)
                or len(set(layers)) != len(layers)):
            parser.error("Events require unique nonnegative layer IDs and positive repetitions")
    server_args = bench.ServerArgs.from_cli_args(args)
    if (args.routing_layers or args.routing_all_layers) and (
            getattr(server_args, "enable_eplb", False)
            or getattr(server_args, "ep_num_redundant_experts", 0)
            or getattr(server_args, "enable_return_routed_experts", False)):
        parser.error("Routing observation requires no expert remapping or other routed-expert capturer")
    if (server_args.nnodes != 1 or server_args.pp_size != 1 or
            server_args.ep_size != 1 or server_args.dp_size != 1 or
            server_args.speculative_algorithm is not None):
        parser.error("Initial capture supports one node, TP only, and no speculation")
    if any(r < 0 or r >= server_args.tp_size for r in args.trace_ranks):
        parser.error("trace-ranks must belong to the TP group")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    options = {key: getattr(args, key) for key in (
        "batch_size", "input_len", "output_len", "repetitions", "seed", "freeze_decode_inputs",
        "trace", "trace_ranks", "trace_steps",
        "operator_event_layers", "operator_event_repetitions",
        "graph_event_layers", "graph_decoder_event", "graph_event_repetitions",
        "graph_event_components",
        "routing_layers", "routing_all_layers", "routing_repetitions",
    )}
    options["warmups"] = args.capture_warmups
    options["output_dir"] = str(output_dir)
    manifest = {
        "schema_version": 1, "status": "running", "engine": "sglang",
        "workload_mode": "static_batch", "hostname": platform.node(),
        "model_path": server_args.model_path, "options": options,
        "topology": {"tp": server_args.tp_size, "ep": 1, "pp": 1, "nodes": 1},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "rocm": torch.version.hip, "sglang": _version("sglang"),
                     "sglang_commit": _git_revision(Path(sglang.__file__).parent),
                     "frontier_commit": _git_revision(Path(__file__).parent),
                     "frontier_source_sha256": _source_digest(Path(__file__).parents[1]),
                     "aiter": _version("aiter") or getattr(aiter, "__version__", None),
                     "aiter_commit": _git_revision(Path(aiter.__file__).parent)},
        "server_args": dataclasses.asdict(server_args),
        "environment": {k: v for k, v in os.environ.items() if
                        k.startswith(("SGLANG_", "HIP_VISIBLE", "ROCR_VISIBLE", "CUDA_VISIBLE"))
                        and not any(s in k for s in ("TOKEN", "SECRET", "KEY", "PASSWORD"))},
        "timing_contract": {
            "forward_gpu_ms": "current-stream GPU events around ModelRunner.forward; excludes sampling",
            "step_wall_ms": "synchronized prepare+forward+sample; excludes ledger serialization",
            "rank_aggregation": "max of per-rank elapsed durations, not sum; ranks have no common wall origin",
            "profiled_rows": "diagnostic only; excluded from latency correlation",
            "operator_events": "separate eager prefill passes; nested inclusive current-stream GPU "
                               "events on selected layers, not additive or kernel-only; no Kineto",
            "graph_events": "separate recaptured full-decode graphs with HIP event-record nodes; "
                            "disjoint inclusive event spans, not KERNEL_ONLY; no Kineto or eager fallback",
            "fixed_decode_inputs": "opt-in teacher forcing from first baseline repetition per shape; "
                                   "trajectory hashes, not tokens, are recorded; fixed token inputs do not "
                                   "guarantee identical intermediate activations or expert routing",
            "routing": "separate recaptured full graphs retaining top-k output references; no GPU nodes "
                       "added; copies after measured synchronization, CPU hashes/histograms after the "
                       "whole repetition. Retaining buffers "
                       "can change graph memory allocation. Logical/padding selections are separate; "
                       "valid zero-weight IDs still count as selections, not proof of executed GEMMs.",
        },
    }
    # Avoid serializing authentication fields from ServerArgs into artifacts.
    for key in list(manifest["server_args"]):
        if any(s in key.lower() for s in ("api_key", "password", "auth_token")):
            manifest["server_args"].pop(key)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    try:
        bench._set_envs_and_config(server_args)
        port_args = bench.PortArgs.init_new(server_args)
        # torch.spawn propagates child failures and terminates the peer workers.
        mp.spawn(_worker, args=(server_args, port_args, options),
                 nprocs=server_args.tp_size, join=True)
        all_records = []
        for rank in range(server_args.tp_size):
            all_records.extend(read_batches(output_dir / f"batches-rank{rank}.jsonl"))
        validate_rank_cohorts(all_records, tensor_parallel_size=server_args.tp_size)
        with (output_dir / "batches.jsonl").open("x") as stream:
            for record in sorted(all_records, key=lambda r: (r.batch_id, r.rank)):
                stream.write(json.dumps(record.to_dict()) + "\n")
        for prefix in ("operator", "graph", "routing"):
            if prefix == "routing":
                enabled = options["routing_layers"] or options["routing_all_layers"]
            elif prefix == "graph":
                enabled = options["graph_event_layers"] or options["graph_decoder_event"]
            else:
                enabled = options["operator_event_layers"]
            if enabled:
                event_records = []
                for rank in range(server_args.tp_size):
                    event_records.extend(read_batches(output_dir / f"{prefix}-batches-rank{rank}.jsonl"))
                validate_rank_cohorts(event_records, tensor_parallel_size=server_args.tp_size)
                with (output_dir / f"{prefix}-batches.jsonl").open("x") as stream:
                    for record in sorted(event_records, key=lambda r: (r.batch_id, r.rank)):
                        stream.write(json.dumps(record.to_dict()) + "\n")
        manifest["status"] = "complete"
        manifest["devices"] = [json.loads((output_dir / f"runtime-rank{rank}.json").read_text())
                               for rank in range(server_args.tp_size)]
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
