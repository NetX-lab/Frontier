"""Price the A10 collective calls with the default analytical backend."""
import json
import sys

from frontier.cc_backend.backends.analytical_cc_backend import AnalyticalCCBackend
from frontier.cc_backend.cc_backend_config import AnalyticalCCBackendConfig
from frontier.types import ClusterType

cfg = AnalyticalCCBackendConfig()
print("analytical defaults:", {k: getattr(cfg, k) for k in
      ("network_bandwidth_gbps", "network_latency_us", "intra_node_bandwidth_gbps")})
for root in sys.argv[1:]:
    rows = json.load(open(f"{root}/collectives.json"))["rows"]
    print("==", root)
    for r in rows:
        op, grp, n = r["curve_key"]
        role = ClusterType.PREFILL if r["role"] == "prefill" else ClusterType.DECODE
        b = AnalyticalCCBackend(cfg, role, "h200", "h200_dgx", 64)
        fn = {"all_gather": b.predict_allgather, "sp_all_gather": b.predict_allgather, "logits_all_gather": b.predict_allgather,
              "all_gatherv": b.predict_allgather, "all_reduce": b.predict_allreduce,
              "reduce_scatterv": b.predict_reduce_scatter,
              "sp_reduce_scatter": b.predict_reduce_scatter}[op]
        a = fn(int(r["requested_bytes"]), n)
        if r["call_count"] >= 1000:
            print(f"{r['role']:7s} {op:17s} {grp} {n:2d} bytes={r['requested_bytes']:>9} "
                  f"exact={r['exact']} measured_ms={r['predicted_ms']:.4f} "
                  f"analytical_default_ms={a:.4f} calls={r['call_count']}")
