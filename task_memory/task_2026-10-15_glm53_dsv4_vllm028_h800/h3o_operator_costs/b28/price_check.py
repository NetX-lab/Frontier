"""Price B28's four H200 calls through V4 communication specs and real contexts."""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from statistics import median
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("mode", choices=("base", "candidate"))
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    from frontier.cc_backend import MeasuredCCBackend, MeasuredCCBackendConfig
    from frontier.cc_backend.backends.measured_cc_backend import logger
    from frontier.config.model_config import BaseModelConfig
    from frontier.config.parallel_semantics import sequence_parallel_moe_token_widths
    from frontier.entities import Batch, Request
    from frontier.operators.families import get_comm_operator
    from frontier.operators.sequence_parallel_moe import measured_collective_time
    from frontier.operators.spec import CommPayloadContext
    from frontier.types import ClusterType

    logging.disable(logging.CRITICAL)
    backend = MeasuredCCBackend(
        MeasuredCCBackendConfig(summary_csv_path=str(args.root / "data/profiling/network/h200_dgx/summary.csv")),
        ClusterType.MONOLITHIC, "h200", "h200_dgx", 32,
    )
    model = BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V4-Flash")
    assert model.num_layers == 43
    expected_base = {"P2-32": (801.7, 763.9), "P2-64": (1076.0, 2011.1)}
    expected_candidate = {"P2-32": (560.2, 734.6), "P2-64": (1043.0, 2172.0)}
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    results = {}
    print(f"{args.mode}: {args.root}")
    for case, lanes in (("P2-32", (6144, 1)), ("P2-64", (6144, 1, 1, 1))):
        ep = 8 * len(lanes)
        widths = sequence_parallel_moe_token_widths(lanes, 8)
        replica = SimpleNamespace(num_pipeline_stages=1, attn_tensor_parallel_size=8,
                                  attn_dp=len(lanes), moe_tensor_parallel_size=1,
                                  moe_expert_parallel_size=ep, router_topk=model.num_experts_per_tok)
        batch = Batch(0, [Request(0.0, max(lanes) + 120, 4)], [max(lanes)],
                      num_context_tokens=[120], request_is_decoding=[False], is_moe=True)
        ctx = CommPayloadContext(batch, model, replica, ClusterType.MONOLITHIC,
                                 None, sp_token_widths=widths)
        scenario = "eager_t" + "_".join(map(str, lanes))
        isolated_path = Path(f"/data/ycfeng/tmp/glm53_dsv4/h200/g5h{ep}_iso_01/download/summary.csv")
        with isolated_path.open(newline="") as stream:
            isolated = [r for r in csv.DictReader(stream) if r["scenario"] == scenario]
        times, iso_times = [], []
        for phase, name, call in (
            ("dispatch", "dsv4_ep_dispatch_allgather", "moe_dispatch_all_gatherv"),
            ("combine", "dsv4_ep_output_reduce_scatter", "moe_combine_reduce_scatterv"),
        ):
            operator = get_comm_operator(name)
            size = operator.build_payload_bytes(ctx)
            ms = measured_collective_time(operator, ctx, backend, size)
            samples = [float(r["median_us"]) for r in isolated if r["call"] == call]
            assert len(samples) == 3
            iso_us = median(samples)
            times.append(ms * 1000)
            iso_times.append(iso_us)
            if args.mode == "candidate":
                assert ms == iso_us / 1000
            print(f"{case} EP{ep} {phase}: {ms * 1000:.1f} us, input {size} bytes, "
                  f"SP rows {widths.local_tokens}; isolated median {iso_us:.1f} us")
        expected = expected_base if args.mode == "base" else expected_candidate
        assert tuple(round(t, 1) for t in times) == expected[case]
        results[case] = times
        if baseline:
            changes = [(a - b) * model.num_layers / 1000 for a, b in zip(times, baseline[case])]
        else:
            changes = [(a - b) * model.num_layers / 1000 for a, b in zip(iso_times, times)]
        assert round(sum(changes), 2) == {"P2-32": -11.64, "P2-64": 5.50}[case]
        print(f"{case} {model.num_layers} layers, {'actual' if baseline else 'planned'} change from base: "
              f"dispatch {changes[0]:+.2f} ms, combine {changes[1]:+.2f} ms, total {sum(changes):+.2f} ms")
    args.json_output.write_text(json.dumps(results, indent=2) + "\n")
    print("PASS: operator prices and forward changes match B28 decisions")


if __name__ == "__main__":
    main()
