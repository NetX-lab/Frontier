"""Diagnostic step-cost tables whose k=1-with-decodes DECODE entry is priced from a native lifecycle run.

Usage: build_tables.py <task directory>

D-H4B-K1-REPRICE. For each case, the case's table (h4a4_driver.CASES) is copied with one change:
the DECODE entry with 1 prefill request of 1 token and 1-128 decode requests takes the median
native step of that composition in the case's lifecycle run, measured as in native_k1_step.py.
The lifecycle runs are native and unprofiled and are not the comparison targets. Every other entry
is unchanged. The scheduler records read are checked against the input set's manifest.

Output: <case>/dsv4_step_cost.json beside this script. The H4a-4 driver reads it with
random_forrest_execution_time_predictor_config_atten_input_file=<case>/attention.csv, because the
loader takes the table beside the attention input; attention.csv itself is not read on this path.
"""
import hashlib
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIFECYCLE_RUNS = {
    # case: (input set directory, input name)
    "native06": ("h4b_lifecycle_measurement/inputs", "h800l"),
    "p2_32": ("h200_mainline/diagnostic/inputs", "p2_32l"),
    "p2_64": ("h200_mainline/diagnostic/inputs", "p2_64l"),
}
SCHEDULER_RECORDS = "nodes/*/scheduler_lifecycle/dp*_pid*.jsonl"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned_records(input_set, name):
    """Digests of the run manifest and the scheduler records, checked against the input set's manifest."""
    run = input_set / name
    pins = {entry["path"]: entry["sha256"] for entry in json.loads((input_set / "manifest.json").read_text())[name]["files"]}
    records = sorted(str(path.relative_to(run)) for path in run.glob(SCHEDULER_RECORDS))
    for path in ["run_manifest.json", *records]:
        if sha256(run / path) != pins[path]:
            raise SystemExit(f"{run / path} differs from its manifest digest")
    return {path: pins[path] for path in ["run_manifest.json", *records]}


def k1_with_decodes(entries):
    [entry] = [e for e in entries if (e["num_prefill_requests"], e["num_prefill_tokens"]) == (1, 1)
               and e["num_decode_requests"]["max"] > 0]
    return entry


def main(root):
    sys.path[:0] = [str(root / "h4a_p0_diagnostic"), str(root / "h4b_lifecycle_measurement"),
                    str(root / "h4b_d_side_model/results")]
    import analyze_lifecycle
    import native_k1_step
    from h4a4_driver import CASES

    for case, (input_set, name) in LIFECYCLE_RUNS.items():
        source = CASES[case]["table"]
        table = json.loads(source.read_text())
        pins = pinned_records(root / input_set, name)
        problems = []
        k1_mixed, _, _ = native_k1_step.classify_steps(analyze_lifecycle.load_cores(root / input_set / name, problems))
        if problems:
            raise SystemExit(f"{name}: {problems}")
        entries = table["roles"]["DECODE"]
        old = k1_with_decodes(entries)
        cached = sorted({count for _, count in k1_mixed})
        if not old["num_decode_requests"]["min"] <= cached[0] <= cached[-1] <= old["num_decode_requests"]["max"]:
            raise SystemExit(f"{name}: decode request counts {cached} outside the entry's range")
        new = {
            "num_prefill_requests": 1, "num_prefill_tokens": 1, "num_decode_requests": old["num_decode_requests"],
            "time_ms": statistics.median(duration for duration, _ in k1_mixed), "samples": len(k1_mixed),
            "measured_num_decode_requests": cached,
            "measurement": "native D engine step from the scheduler records of the lifecycle run (native_k1_step.py)",
            "replaced_entry": old,
        }
        entries[entries.index(old)] = new
        table["sources"]["k1_with_decodes_entry"] = {
            "decision": "D-H4B-K1-REPRICE", "source_table": str(source.relative_to(root.parents[1])),
            "source_table_sha256": sha256(source), "lifecycle_input": f"{input_set}/{name}",
            "run_tag": json.loads((root / input_set / name / "run_manifest.json").read_text())["run_tag"],
            "files_sha256": pins,
        }
        output = HERE / case / "dsv4_step_cost.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(table, indent=1, allow_nan=False) + "\n")
        print(f"{case}: k=1 with decodes {old['time_ms']:.2f} ms ({old['samples']} samples) -> "
              f"{new['time_ms']:.2f} ms ({new['samples']} samples, decode requests {cached[0]}-{cached[-1]}) "
              f"from {name}; {output.relative_to(root)} {sha256(output)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]).resolve())
