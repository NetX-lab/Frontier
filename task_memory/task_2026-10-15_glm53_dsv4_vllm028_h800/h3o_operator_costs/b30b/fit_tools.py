"""CPU fits for B30b; G5h/K5 target curves never enter a fitted parameter.

ASTRA features are evaluations of Frontier's own backend at unit link latency
and inverse bandwidth. Native grouped dispatch remains one summed-byte call.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ALIASES = {"all_reduce": "allreduce", "all_gather": "allgather",
           "sp_all_gather": "allgather", "sp_reduce_scatter": "reduce_scatter",
           "all_gatherv": "allgather", "reduce_scatterv": "reduce_scatter"}


def csv_rows(paths):
    """Read final CSVs or completed raw JSONL cases after a partial phase."""
    from frontier.profiling.collectives.vllm_collective_benchmark import summarize
    from frontier.profiling.collectives.nixl_kv_read_benchmark import summarize_case

    rows = []
    for path in paths:
        with path.open(newline="") as handle:
            if path.suffix == ".jsonl":
                for line in handle:
                    record = json.loads(line)
                    row = (summarize_case(record) if "readers" in record else
                           summarize(dict(record, kernels=record.get("kernels", []))))
                    rows.append(dict(row, source=str(path)))
            else:
                rows.extend(dict(row, source=str(path)) for row in csv.DictReader(handle))
    return rows


def astra_price(row, link, latency_us, bandwidth_gbps):
    from frontier.cc_backend.backends.astra_sim_analytical_cc_backend import AstraSimAnalyticalCCBackend
    from frontier.cc_backend.cc_backend_config import AstraSimAnalyticalCCBackendConfig
    from frontier.types import ClusterType

    n = int(row["group_size"])
    if n < 2 or (link == "inter" and n != 2):
        raise ValueError("Fit rows need >=2 intra ranks or exactly two inter ranks")
    # The two-rank probe materializes one participating GPU per physical server.
    # Both GPU placements measure the same inter-link class; local GPU numbering
    # does not add synthetic unused ranks to the probe's communication world.
    config = AstraSimAnalyticalCCBackendConfig(
        cluster_servers=2 if link == "inter" else 1,
        cluster_gpus_per_server=1 if link == "inter" else n,
        intra_server_latency_us=latency_us, intra_server_bandwidth_gbps=bandwidth_gbps,
        inter_server_latency_us=latency_us, inter_server_bandwidth_gbps=bandwidth_gbps)
    backend = AstraSimAnalyticalCCBackend(config, ClusterType.DECODE, "h200", "h200_dgx", n)
    op = ALIASES[row["op"]]
    method = {"allreduce": "predict_allreduce", "allgather": "predict_allgather",
              "reduce_scatter": "predict_reduce_scatter"}[op]
    return getattr(backend, method)(int(row["max_input_bytes"]), n,
                                   comm_domain="MOE_EP" if row["group"] == "ep" else "ATTN_TP") * 1000


def pool_collectives(rows):
    pools = defaultdict(list)
    metadata = {}
    for row in rows:
        if row["op"] not in ALIASES or int(row["group_size"]) < 2:
            continue
        key = tuple(str(row[k]) for k in ("scenario", "call", "op", "group", "group_size",
                                         "cuda_graph", "max_input_bytes"))
        pools[key].append(float(row["median_us"]))
        metadata[key] = row
    return [{**metadata[k], "median_us": statistics.median(values), "samples": len(values)}
            for k, values in sorted(pools.items())]


def least_squares(features, observations):
    matrix = np.asarray(features, dtype=float)
    values = np.asarray(observations, dtype=float)
    if (matrix.ndim != 2 or len(values) < 2 or matrix.shape[1] != 2 or
            not np.isfinite(matrix).all() or not np.isfinite(values).all() or
            (values <= 0).any()):
        raise ValueError("Fit needs >=2 finite positive observations and two features")
    # Scale columns before solving: bytes coefficients can be millions of times
    # larger than latency coefficients. Report an unconstrained fit honestly.
    scales = np.linalg.norm(matrix, axis=0)
    if (scales <= 0).any():
        raise ValueError("Unidentifiable fit")
    coefficients, _, rank, _ = np.linalg.lstsq(matrix / scales, values, rcond=None)
    if rank != 2:
        raise ValueError("Unidentifiable fit: payloads do not span latency and bandwidth")
    coefficients = coefficients / scales
    if coefficients[0] < -1e-9 or coefficients[1] <= 0:
        raise ValueError(f"Unphysical unconstrained fit {coefficients.tolist()}; inspect residuals")
    coefficients[0] = max(0, coefficients[0])
    return coefficients


def residual_summary(rows):
    curves = defaultdict(list)
    for row in rows:
        curves[f"{row.get('op', 'READ')}/{row.get('group', '')}{row.get('group_size', '')}/"
               f"{row.get('cuda_graph', '')}"].append(row)
    return {k: dict(n=len(v), rmse_us=math.sqrt(statistics.fmean(r["residual_us"]**2 for r in v)),
                    mean_residual_us=statistics.fmean(r["residual_us"] for r in v),
                    mean_absolute_relative_error=statistics.fmean(abs(r["relative_error"]) for r in v),
                    min_relative_error=min(r["relative_error"] for r in v),
                    max_relative_error=max(r["relative_error"] for r in v))
            for k, v in sorted(curves.items())}


def fit_astra(rows, link):
    rows = pool_collectives(rows)
    features = [[astra_price(row, link, 1, 1) - astra_price(row, link, 0, 1),
                 astra_price(row, link, 0, 1)] for row in rows]
    latency, inverse_bandwidth = least_squares(features, [r["median_us"] for r in rows])
    bandwidth = 1 / inverse_bandwidth
    residuals = []
    for row in rows:
        predicted = astra_price(row, link, latency, bandwidth)
        measured = float(row["median_us"])
        residuals.append({**row, "predicted_us": predicted,
                          "residual_us": predicted - measured,
                          "relative_error": predicted / measured - 1})
    return dict(link=link, latency_us=float(latency), bandwidth_gbps=float(bandwidth),
                residuals=residuals, curves=residual_summary(residuals),
                payload_convention="largest rank input bytes; dispatch sums the three native fields",
                operation_correction=False)


def fit_nixl(rows):
    selected = [r for r in rows if int(r["active_pairs"]) == 8]
    for row in selected:
        if str(row.get("data_verified", "")).lower() != "true":
            raise ValueError("NIXL fit refuses unverified data")
        if int(row["notifications_received"]) != int(row["notifications_expected"]):
            raise ValueError("NIXL fit refuses missing notifications")
    observations, features, metadata = [], [], []
    for row in selected:
        requests = int(row["requests_in_flight"])
        if requests < 1:
            raise ValueError("NIXL requests in flight must be positive")
        size = int(row["bytes_per_reader"]) / requests
        measured = float(row["complete_median_us"]) / requests
        features.append([1, size * 8 / 1000])  # microseconds at 1 Gbit/s
        observations.append(measured)
        metadata.append(dict(row, bytes_per_request=size, measured_us=measured))
    latency, inverse_bw = least_squares(features, observations)
    residuals = []
    for row in metadata:
        predicted = latency + row["bytes_per_request"] * 8 / 1000 * inverse_bw
        baseline = 64.7 + row["bytes_per_request"] * 8 / 1000 / 126.2
        residuals.append(dict(row, predicted_us=float(predicted),
                              residual_us=float(predicted - row["measured_us"]),
                              relative_error=float(predicted / row["measured_us"] - 1),
                              h800_predicted_us=baseline,
                              h800_relative_error=baseline / row["measured_us"] - 1))
    return dict(latency_us=float(latency), bandwidth_gbps=float(1 / inverse_bw),
                h800_baseline=dict(latency_us=64.7, bandwidth_gbps=126.2),
                method="W7 section 2.8: per-reader completion and bytes divided by requests in flight; eight pairs only",
                residuals=residuals, curves=residual_summary(residuals))


def astra_fields(intra, inter):
    fields = {"cc_backend_config_type": "astra_sim_analytical",
              "cluster_config_prefill_cc_backend_config_type": "astra_sim_analytical",
              "cluster_config_decode_cc_backend_config_type": "astra_sim_analytical"}
    for prefix in ("astra_sim_analytical_cc_backend_config",
                   "cluster_config_prefill_cc_backend_config", "cluster_config_decode_cc_backend_config"):
        for name, fit in (("intra", intra), ("inter", inter)):
            fields[f"{prefix}_{name}_server_latency_us"] = fit["latency_us"]
            fields[f"{prefix}_{name}_server_bandwidth_gbps"] = fit["bandwidth_gbps"]
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    astra = commands.add_parser("astra")
    astra.add_argument("--intra", action="append", required=True, type=Path)
    astra.add_argument("--inter", action="append", required=True, type=Path)
    nixl = commands.add_parser("nixl")
    nixl.add_argument("--input", action="append", required=True, type=Path)
    for command in (astra, nixl):
        command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "astra":
        intra = fit_astra(csv_rows(args.intra), "intra")
        inter = fit_astra(csv_rows(args.inter), "inter")
        result = dict(intra=intra, inter=inter,
                      inter_by_placement={str(p): fit_astra(csv_rows([p]), "inter") for p in args.inter})
        assignments = astra_fields(intra, inter)
    else:
        result = fit_nixl(csv_rows(args.input))
        assignments = {"analytical_kv_cache_transfer_config_network_latency_ms": result["latency_us"] / 1000,
                       "analytical_kv_cache_transfer_config_network_bandwidth_gbps": result["bandwidth_gbps"]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "fit.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "fields.txt").write_text("".join(f"{k}={v}\n" for k, v in assignments.items()))
    print(json.dumps({k:v for k,v in result.items() if k not in ("residuals",)}, indent=2))


if __name__ == "__main__":
    main()
