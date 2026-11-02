"""Run one formal P2 fresh-cache CPU replay with the adopted H200 NIXL pair, its gate and both warm readings.

The replay uses the B29 formal inputs from the committed source and overrides only the
two analytical KV transfer fields with the pair adopted under D-P2-Q2-NIXL-TIMING
(`input_qualification_20261101/nixl_timing_q2/proposed_input.json`). The command record
next to this script is rewritten after each stage, so an interrupted run still shows
how far it got.

Usage: run_formal_replay.py <case> <scratch root>
"""

import datetime
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
H3O = HERE.parent
REPO = HERE.parents[3]
PYTHON = "/data/ycfeng/envs/frontier-py310/bin/python"
AS_LIMIT_BYTES = 64 * 1024 ** 3
ENVIRONMENT = {
    "PYTHONPATH": f"{REPO}:{REPO}/tests/analysis",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TMPDIR": "/data/ycfeng/tmp",
    "FRONTIER_TMP_ROOT": "/data/ycfeng/tmp",
    "WANDB_DISABLED": "true",
    "VIDUR_DISABLE_WANDB": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
TIMING_INPUT = H3O / "input_qualification_20261101/nixl_timing_q2/proposed_input.json"
TIMING_FIELDS = (
    "analytical_kv_cache_transfer_config_network_bandwidth_gbps",
    "analytical_kv_cache_transfer_config_network_latency_ms",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def find_field(document, name):
    """Return the single value stored under `name` anywhere in the timing input."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == name:
                    found.append(value)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(document)
    if len(set(map(repr, found))) != 1:
        raise SystemExit(f"{TIMING_INPUT} must hold exactly one value for {name}, found {found}")
    return found[0]


def run(command, log, *, limit_address_space=False):
    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (AS_LIMIT_BYTES, AS_LIMIT_BYTES))

    with open(log, "x") as stream:
        return subprocess.run(
            command, cwd=REPO, env={**os.environ, **ENVIRONMENT}, stdout=stream,
            stderr=subprocess.STDOUT, preexec_fn=limit if limit_address_space else None,
        ).returncode


def main(case, scratch):
    scratch = Path(scratch)
    run_root, gate_root, logs = scratch / "runs" / case, scratch / "gates" / case, scratch / "logs"
    if run_root.exists() or gate_root.exists():
        raise SystemExit(f"A fresh-cache replay needs new roots: {run_root}, {gate_root}")
    logs.mkdir(parents=True, exist_ok=True)
    timing = json.loads(TIMING_INPUT.read_text())
    fields = {name: find_field(timing, name) for name in TIMING_FIELDS}
    driver = H3O / "a10/a10_driver.py"
    command = ["timeout", "86400s", PYTHON, "-u", str(driver), case, str(run_root)]
    for name, value in fields.items():
        command += ["--field", f"{name}={json.dumps(value)}"]
    record_path = HERE / f"{case}_command.json"
    record = dict(
        classification="D-P2-FORMAL_CPU_REPLAY", status="RUNNING", command=command,
        timing_input=str(TIMING_INPUT), timing_input_sha256=sha256(TIMING_INPUT), timing_fields=fields,
        cwd=str(REPO), environment=ENVIRONMENT, source_commit=git("rev-parse", "HEAD"),
        source_dirty_paths=git("status", "--porcelain", "--", "frontier", "data").splitlines(),
        driver_sha256=sha256(driver), python=platform.python_version(),
        started_utc=utc_now(), run_root=str(run_root), gate_root=str(gate_root),
        log=str(logs / f"{case}.log"), as_limit_bytes=AS_LIMIT_BYTES, supervisor_pid=os.getpid(),
    )

    def save():
        record_path.write_text(json.dumps(record, indent=2) + "\n")

    if record["source_dirty_paths"]:
        raise SystemExit(f"Replay source must be committed: {record['source_dirty_paths']}")
    save()
    start = time.monotonic()
    code = run(command, record["log"], limit_address_space=True)
    record.update(exit_code=code, elapsed_seconds=time.monotonic() - start, finished_utc=utc_now(),
                  simulator_status="FINISHED" if code == 0 else "FAILED")
    if code:
        record["status"] = "FAILED"
        save()
        return code
    record["receipt_sha256"] = sha256(run_root / "receipt.json")
    save()

    gate_command = [PYTHON, str(H3O / "a10/a10_gate.py"), case, str(run_root), str(gate_root)]
    gate_log = logs / f"{case}_gate.log"
    gate = dict(command=gate_command, log=str(gate_log), started_utc=utc_now())
    gate["exit_code"] = run(gate_command, gate_log)
    gate["finished_utc"] = utc_now()
    if gate["exit_code"] == 0:
        gate["reading_sha256"] = sha256(gate_root / "reading.json")
        gate["verdict"] = json.loads((gate_root / "reading.json").read_text())["verdict"]
    record["gate"] = gate
    save()
    if gate["exit_code"]:
        record["status"] = "FAILED"
        save()
        return gate["exit_code"]

    readings = {}
    (HERE / "readings").mkdir(exist_ok=True)
    for kind, openers in (("warm", "repeat"), ("empty", "empty")):
        output = HERE / "readings" / f"{case}_{kind}_reading.json"
        reading_command = [
            PYTHON, str(REPO / "tests/analysis/e2e_warm_reading.py"),
            "--rows", str(gate_root / "rows"), "--reading", str(gate_root / "reading.json"),
            "--openers", str(H3O / f"warm_target_20261101/{case}_{openers}_openers.json"),
            "--output", str(output),
        ]
        reading_log = logs / f"{case}_{kind}_reading.log"
        reading = dict(command=reading_command, log=str(reading_log))
        reading["exit_code"] = run(reading_command, reading_log)
        if reading["exit_code"] == 0:
            reading["output_sha256"] = sha256(output)
        readings[kind] = reading
    record["readings"] = readings
    record["status"] = "FINISHED" if all(item["exit_code"] == 0 for item in readings.values()) else "FAILED"
    save()
    return 0 if record["status"] == "FINISHED" else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
