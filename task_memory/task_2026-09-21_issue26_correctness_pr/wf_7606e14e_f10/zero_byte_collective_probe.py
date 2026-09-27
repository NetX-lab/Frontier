"""Probe: zero-byte cross-server allreduce/allgather/reduce_scatter in the companion runner."""
import sys, json, traceback
from pathlib import Path
REPO = Path(sys.argv[1]); OUT = Path(sys.argv[2])
sys.path.insert(0, str(REPO / "python"))
from collective_sim_core.predictor import predict_collective_time
from collective_sim_core.schema import Scenario
for kind in sys.argv[3].split(","):
    for payload in [int(x) for x in sys.argv[4].split(",")]:
        try:
            sc = Scenario.from_dict({
                "cluster": {"servers": 2, "gpus_per_server": 8},
                "parallelism": {"tp": 1, "cp": 1, "dp": 16, "ep": 1},
                "collective": {"kind": kind, "tensor_bytes": payload, "domain_dims": ["DP"],
                               "placement_order": ["TP", "CP", "DP", "EP"],
                               "participant_ranks": list(range(16))},
                "runner": {"out_dir": str(OUT / f"{kind}_{payload}")},
            })
            r = predict_collective_time(sc, repo_root=REPO)
            raw = r.get("raw_runner_payload", {})
            print("PROBE", kind, payload, "ok", round(r["predicted_time_ms"], 6),
                  "net", r["breakdown"].get("network_ms"), "flows", raw.get("expected_flows"), raw.get("finished_flows"))
        except Exception as e:
            print("PROBE", kind, payload, "exc", type(e).__name__, str(e)[-300:].replace("\n", " | "))
