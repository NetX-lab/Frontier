"""Price each G5h device collective row with a Frontier CC backend (F1 check).

Usage: f1_astra_per_call.py <out_dir> [field=value ...]

The fields are flat SimulationConfig overrides, for example
cc_backend_config_type=astra_sim_analytical plus its link parameters. Each
G5h row is priced through the DECODE cluster of the A10 case whose serving
pod has that row's group size, with the payload the measured backend uses
(the largest rank's input bytes). Rows of one call are pooled over passes
by their median, as the measured backend pools them.
"""

import csv
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "a10"))
import a10_driver

# G5h op name -> (Frontier collective alias, communication domain).
OP_COLLECTIVES = {
    ("all_reduce", "tp"): ("allreduce", "ATTN_TP"),
    ("all_gather", "tp"): ("allgather", "ATTN_TP"),
    ("sp_all_gather", "tp"): ("allgather", "ATTN_TP"),
    ("sp_reduce_scatter", "tp"): ("reduce_scatter", "ATTN_TP"),
    ("all_gatherv", "ep"): ("allgather", "MOE_EP"),
    ("reduce_scatterv", "ep"): ("reduce_scatter", "MOE_EP"),
}
PREDICTORS = {
    "allreduce": "predict_allreduce",
    "allgather": "predict_allgather",
    "reduce_scatter": "predict_reduce_scatter",
}


def decode_backends(fields):
    """Build each A10 case's DECODE-cluster backend, keyed by its EP group size."""
    from frontier.entities.cluster import Cluster
    from frontier.types import ClusterType

    backends = {}
    with tempfile.TemporaryDirectory(dir="/data/ycfeng/tmp") as root:
        for name in a10_driver.CASES:
            config, _ = a10_driver.build_config(name, Path(root) / name, fields)
            cluster_config = config.get_clusters()[ClusterType.DECODE]
            replica = cluster_config.replica_config
            cluster = Cluster(cluster_config, config.metrics_config, config.request_generator_config)
            group = int(replica.attn_tensor_parallel_size) * int(replica.attn_dp)
            backends[group] = (name, cluster.cc_backend)
    return backends


def measured_source():
    """The G5h table the formal A10 inputs consume, resolved as A10 resolves it."""
    from frontier.types import ClusterType

    with tempfile.TemporaryDirectory(dir="/data/ycfeng/tmp") as root:
        config, _ = a10_driver.build_config("p2_32", Path(root), {})
    cluster = config.get_clusters()[ClusterType.DECODE]
    cc = cluster.cc_backend_config
    return Path(cc.summary_csv_path.format(
        profiling_data_dir=cc.profiling_data_dir,
        NETWORK_DEVICE=cluster.replica_config.network_device,
    ))


def main(out_dir, fields):
    backends = decode_backends(fields)
    source = measured_source()
    samples = defaultdict(list)
    meta = {}
    with open(source, newline="") as stream:
        for row in csv.DictReader(stream):
            if row["group"] == "dp":
                continue
            key = (row["scenario"], row["call"], int(row["group_size"]))
            samples[key].append(float(row["median_us"]))
            meta[key] = row
    out = []
    for key, times in sorted(samples.items()):
        row = meta[key]
        alias, domain = OP_COLLECTIVES[(row["op"], row["group"])]
        group = key[2]
        # TP rows stay inside one server, so the smallest pod prices them.
        case, backend = backends[group if row["group"] == "ep" else min(backends)]
        size = int(row["max_input_bytes"])
        predicted_us = getattr(backend, PREDICTORS[alias])(size, group, comm_domain=domain) * 1000.0
        measured_us = median(times)
        tokens = [int(value) for value in row["tokens_per_dp_rank"].split("/") if value]
        out.append(dict(
            scenario=key[0], call=key[1], op=row["op"], group=row["group"], group_size=group,
            case=case, cuda_graph=row["cuda_graph"], calls_per_forward=int(row["calls_per_forward"]),
            uniform=len(set(tokens)) == 1 and "single_active" not in key[0],
            max_input_bytes=size, measured_us=round(measured_us, 3),
            predicted_us=round(predicted_us, 3), ratio=round(predicted_us / measured_us, 4),
        ))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_call.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    forwards = defaultdict(lambda: [0.0, 0.0])
    for item in out:
        if item["scenario"].startswith("graph_t"):
            total = forwards[(item["scenario"], item["group"], item["group_size"])]
            total[0] += item["calls_per_forward"] * item["measured_us"]
            total[1] += item["calls_per_forward"] * item["predicted_us"]
    by_curve = defaultdict(list)
    for item in out:
        by_curve[f"{item['op']}/{item['group']}{item['group_size']}"].append(item["ratio"])
    summary = dict(
        fields=fields,
        source=str(source.resolve()),
        rows=len(out),
        ratio_by_curve={k: dict(n=len(v), min=min(v), median=median(v), max=max(v))
                        for k, v in sorted(by_curve.items())},
        graph_forward_us={f"{s}/{g}{n}": dict(measured=round(m, 1), predicted=round(p, 1),
                                               ratio=round(p / m, 4))
                          for (s, g, n), (m, p) in sorted(forwards.items())},
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2 or not all("=" in a for a in sys.argv[2:]):
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), dict(a10_driver.h4a4.parse_field(a) for a in sys.argv[2:]))
