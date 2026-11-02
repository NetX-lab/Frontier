"""Profiler-perturbation check for the H4a step-cost sources (plan H4a-1).

Computes the four client metrics of native06 and of the profiled runs profile02 and profile04
from their persisted client records with the canonical client readers, and each profiled
repeat's relative difference from the same native06 repeat. Writes perturbation_check.json.
"""
import json
from pathlib import Path

from tests.comparison.calibration.client_metrics import native_client_rows
from tests.comparison.calibration.e2e_metrics_gap import formal_sample, metric_values
from tests.comparison.calibration.pd_replay import canonical_workload

HERE = Path(__file__).resolve().parent
TRACE = Path("data/processed_traces/canonical_fixed_6144_1024_req128_qps1")
RUNS = ("native06", "profile02", "profile04")
METRICS = ("ttft_ms", "tpot_ms", "request_e2e_time_ms", "token_throughput_tps")


def main() -> None:
    workload = canonical_workload(TRACE)
    request_ids = json.loads((TRACE / "request_ids.json").read_text())
    formal_ids = request_ids["formal_request_ids"]
    classified = set(formal_ids) | set(request_ids["warmup_request_ids"]) | set(request_ids["sizing_request_ids"])
    metrics = {}
    for run in RUNS:
        repeats = native_client_rows(HERE / "inputs" / run, workload, request_ids)
        metrics[run] = {}
        for repeat, rows in repeats.items():
            sample = formal_sample(rows, formal_ids, classified)
            if sample["missing"] or sample["duplicates"]:
                raise SystemExit(f"{run} {repeat}: incomplete formal sample")
            values = metric_values(sample["sample"])
            metrics[run][repeat] = {m: values[m][0] / values[m][1] for m in METRICS}
    relative = {run: {repeat: {m: value[m] / metrics["native06"][repeat][m] - 1 for m in METRICS}
                      for repeat, value in metrics[run].items()}
                for run in RUNS[1:]}
    result = dict(trace=str(TRACE), metrics=metrics, relative_to_native06=relative,
                  note="repeat01 of each profiled run is the profiled repeat; its timings are not client-valid.")
    (HERE / "perturbation_check.json").write_text(json.dumps(result, indent=1) + "\n")
    for run, repeats in relative.items():
        for repeat, value in repeats.items():
            print(run, repeat, " ".join(f"{m} {100 * v:+.2f}%" for m, v in value.items()))


if __name__ == "__main__":
    main()
