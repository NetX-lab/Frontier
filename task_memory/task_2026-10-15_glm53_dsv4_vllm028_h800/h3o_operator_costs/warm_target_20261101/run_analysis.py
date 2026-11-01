"""Execute the fixed six-reading CPU analysis and retain command receipts."""

import csv
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
READINGS = OUT / "readings"
READINGS.mkdir(exist_ok=True)
TASK = OUT.parent.parent
PYTHON = "/data/ycfeng/envs/frontier-py310/bin/python"
DOWNLOADS = Path("/data/ycfeng/tmp/glm53_dsv4/h200/downloads")
RUNS = {
    "canonical_p2_32": "dsv4-h200-p2-canonical-20261025-01",
    "canonical_p2_64": "dsv4-h200-p2-64-canonical-20261025-01",
    "lifecycle": "dsv4-h200-p2-64-lifecycle-20261026-01",
    "wave": "dsv4-h200-p2-64-wave-20261030-01",
}
ENV = dict(PYTHONDONTWRITEBYTECODE="1", TMPDIR="/data/ycfeng/tmp", PYTHONPATH=f"{ROOT}:{ROOT}/tests/analysis")
os.environ.update(ENV)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, document):
    with path.open("x") as stream:
        stream.write(json.dumps(document, indent=2, allow_nan=False) + "\n")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


receipts_path = OUT / "command_receipts.json"
receipts = read(receipts_path) if receipts_path.exists() else []


def execute(arguments, outputs):
    command = [PYTHON, *map(str, arguments)]
    record = dict(command=command, cwd=str(ROOT), env=ENV, started_utc=utc(),
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  dirty_paths=subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], text=True).splitlines(),
                  python_version=platform.python_version(), virtual_memory_limit_kib=31457280)
    result = subprocess.run(command, capture_output=True, text=True)
    record.update(finished_utc=utc(), exit_code=result.returncode)
    log = OUT / f"command_{len(receipts) + 1:02d}.log"
    with log.open("x") as stream:
        stream.write(result.stdout + result.stderr)
    record.update(log=str(log), output_sha256={str(p): digest(p) for p in [log, *outputs] if p.exists()})
    receipts.append(record)
    receipts_path.write_text(json.dumps(receipts, indent=2) + "\n")
    print(result.stdout, end="")
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)


openers_path = OUT / "openers.json"
if not openers_path.exists():
    arguments = [OUT / "identify_openers.py"]
    for label, run in RUNS.items():
        arguments.extend(("--run", f"{label}={DOWNLOADS / run / 'artifacts'}"))
    execute([*arguments, "--output", openers_path], [openers_path])
evidence = read(openers_path)
assert all(evidence["runs"][r]["validation"]["status"] == "PASS_EXACT" for r in ("lifecycle", "wave"))
assert evidence["runs"]["canonical_p2_32"]["counts"] == dict(handshake_log_lines=32, events=4, warmup_events=4, formal_events=0)
assert evidence["runs"]["canonical_p2_64"]["counts"] == dict(handshake_log_lines=128, events=16, warmup_events=11, formal_events=5)
assert {r.split("formal-")[1].split("-repeat")[0] for r in evidence["runs"]["canonical_p2_64"]["formal_openers"]} == {
    "r006", "r017", "r027", "r035", "r036"}
matrix, verifications = [], []
for generation in ("B29", "pass1_all", "pass2_all"):
    for case in ("p2_32", "p2_64"):
        if generation == "B29":
            gate = Path("/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/codex_resume_20261031") / f"{case}_warm_k19b29"
            reading = TASK / f"h3o_operator_costs/codex_resume_20261031/{case}_warm_k19b29/reading.json"
            assert digest(reading) == digest(gate / "reading.json")
        else:
            gate = Path("/data/ycfeng/tmp/glm53_dsv4/a6_pass_replays_20261101/gates") / f"{generation}_{case}"
            reading = gate / "reading.json"
        native = read(reading)["repetitions"]
        opened = evidence["runs"][f"canonical_{case}"]["formal_openers"]
        exclusions = {side: [rid.rsplit("-repeat", 1)[0] for rid in opened if rid.endswith(side.rsplit("-", 1)[1])]
                      for side in native}
        input_path = OUT / f"{case}_repeat_openers.json"
        empty_path = OUT / f"{case}_empty_openers.json"
        if not input_path.exists():
            write(input_path, exclusions)
            write(empty_path, {side: [] for side in native})
        output = READINGS / f"{generation}_{case}_warm_reading.json"
        empty_output = READINGS / f"{generation}_{case}_empty_reading.json"
        args = [ROOT / "tests/analysis/e2e_warm_reading.py", "--rows", gate / "rows", "--reading", reading]
        for opener, destination in ((input_path, output), (empty_path, empty_output)):
            if not destination.exists():
                execute([*args, "--openers", opener, "--output", destination], [destination])
        document, empty, original = read(output), read(empty_output), read(reading)
        assert document["all_request"] == original and document["all_request_verbatim"] == reading.read_text()
        assert document["reading_sha256"] == digest(reading)
        fields = ("metric", "vllm", "frontier", "absolute_error", "relative_error", "status")
        assert all(a[k] == b[k] for a, b in zip(empty["warm"]["metrics"], original["metrics"]) for k in fields)
        assert all(empty["warm"]["repetitions"][s]["metrics"] == original["repetitions"][s]["metrics"] for s in native)
        assert document["warm"]["retained_request_repeat_count"] == (379 if case == "p2_64" else 384)
        for side, group in document["warm"]["repetitions"].items():
            if not exclusions[side]:
                assert group["metrics"] == original["repetitions"][side]["metrics"]
        for source in document["sources"]:
            assert digest(Path(source["path"])) == source["sha256"]
        for source in evidence["runs"][f"canonical_{case}"]["sources"]:
            assert digest(Path(source["path"])) == source["sha256"]
        verifications.append(dict(generation=generation, case=case, original_reproduction="PASS_EXACT",
                                  empty_set_reproduction="PASS_EXACT", unchanged_sources="PASS_SHA256", output=str(output)))
        for kind in ("all_request", "warm"):
            groups = {"pooled": document[kind], **document[kind]["repetitions"]}
            for repeat, group in groups.items():
                for metric in group["metrics"]:
                    delta = metric["frontier"] - metric["vllm"]
                    matrix.append(dict(generation=generation, case=case, reading=kind, repeat=repeat,
                                       metric=metric["metric"], unit="tokens/s" if metric["metric"] == "token_throughput_tps" else "ms",
                                       native=metric["vllm"], frontier=metric["frontier"], signed_error=delta,
                                       absolute_error=abs(delta), signed_relative_error=delta / metric["vllm"],
                                       absolute_relative_error=abs(delta) / metric["vllm"], status=metric["status"],
                                       sample_count=metric["sample_count"], eligible_count=metric["eligible_count"],
                                       excluded_count=metric["excluded_count"]))
assert len(matrix) == 192 and len(verifications) == 6
csv_path = READINGS / "metrics.csv"
with csv_path.open("x") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(matrix[0]))
    writer.writeheader()
    writer.writerows(matrix)
write(READINGS / "verification.json", dict(status="PASS", checks=verifications, metric_rows=192,
                                      csv_sha256=digest(csv_path), python=sys.version, env=ENV))
print("PASS six original/empty-set reproductions; 192 metric rows; canonical source hashes unchanged")
