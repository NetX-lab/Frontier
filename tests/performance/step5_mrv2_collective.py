"""Measure real actual05 collectives using an existing built submodule."""

from __future__ import annotations

import argparse
import cProfile
from dataclasses import fields
import io
import json
from pathlib import Path
import pstats
import sys
import time
from types import SimpleNamespace

from frontier.cc_backend.backends.collective_sim_cc_backend import CollectiveSimCCBackend
from frontier.cc_backend.cc_backend_config import CollectiveSimCCBackendConfig
from frontier.entities.cluster import Cluster
from frontier.types import ClusterType


class ExistingCollectiveBackend(CollectiveSimCCBackend):
    """Only reuse the requested built submodule location; costs are unchanged."""

    def _resolve_collective_sim_repo_root(self):
        root = Path(args.collective_repo)
        assert (root / "sim/datacenter/htsim_ndp").is_file()
        return root


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads(Path(args.config).read_text())["cluster_config"]
    sys.path.insert(0, str(Path(args.collective_repo) / "python"))
    values = original["cc_backend_config"]
    names = {field.name for field in fields(CollectiveSimCCBackendConfig)}
    config = CollectiveSimCCBackendConfig(**{name: value for name, value in values.items() if name in names})
    config.runner_out_dir = str(output / "runner")
    config.cache_dir = str(output / "cache")
    replica = SimpleNamespace(**original["decode_replica_config"])
    replica.node_config = SimpleNamespace(**replica.node_config)
    cluster = Cluster.__new__(Cluster)
    cluster._config = SimpleNamespace(num_replicas=original["decode_cluster_num_replicas"])
    cluster._cluster_type = ClusterType.DECODE
    config = cluster._materialize_collective_sim_cc_config(config, replica, original["decode_world_size"])
    backend = ExistingCollectiveBackend(config, ClusterType.DECODE, "h800", "h800_dgx", 16)
    profile = cProfile.Profile()
    rows = []
    for kind, domain, devices, byte_count in (("allreduce", "ATTN_TP", 8, 4096 * 2),
                                             ("alltoall", "EP", 16, 2 * 4096 * 2)):
        for iteration in range(2):
            start = time.perf_counter()
            profile.enable()
            method = backend.predict_allreduce if kind == "allreduce" else backend.predict_all_to_all
            value = method(data_size_bytes=byte_count, num_devices=devices,
                           cluster_type=ClusterType.DECODE, comm_domain=domain)
            profile.disable()
            seconds = time.perf_counter() - start
            rows.append(dict(kind=kind, domain=domain, devices=devices, tensor_bytes=byte_count,
                             iteration=iteration, seconds=seconds, predicted_ms=value))
        assert rows[-1]["predicted_ms"] == rows[-2]["predicted_ms"]
    profile.dump_stats(str(output / "collective.prof"))
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(18)
    (output / "profile.txt").write_text(stream.getvalue())
    receipt = dict(task_id="D23-ACTUAL-MAIN/PERF", native_admitted=False,
                   rows=rows, requests=backend._prediction_cache_requests,
                   misses=backend._prediction_cache_misses, hits=backend._prediction_cache_hits,
                   config={field.name: getattr(config, field.name) for field in fields(config)})
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({name: receipt[name] for name in ("rows", "requests", "misses", "hits")}, indent=2))
    print(stream.getvalue())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="/data/ycfeng/tmp/d23_main_actual05_offline/offline/metrics/stepfun_ai_step_5_preview/offline_batch/step5_measured_cpu_main_offline/config.json")
    parser.add_argument("--collective-repo", default="/data/ycfeng/Frontier/.worktrees/step5-mrv2-main/frontier/cc_backend/backends/collective-sim")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args)
