"""Export a fresh V4 isolated-input candidate; never issue consumer admission."""

import argparse
import json
import shutil
from pathlib import Path

from frontier.execution_time_predictor.deepseek_v4 import build_isolated_prefill_layouts
from frontier.execution_time_predictor.dsv4_attention_input import QUALIFICATION, SCHEMA
from frontier.profiling.attention.dsa_rows import read_records
from frontier.profiling.attention.dsv4_attention_main import build_parser, build_plan
from frontier.profiling.common.runtime_identity import file_sha256


JOB = "jobs/PREFILL_WARM_6144_C0_BALANCED.json"


def _require(condition, message):
    if not condition:
        raise ValueError(f"V4 input preparation rejected: {message}")


def accepted_prefill_inputs(acceptance_path, package):
    """Validate retained source bytes and reconstruct its plan using local code."""
    acceptance_path, package = Path(acceptance_path), Path(package)
    acceptance = json.loads(acceptance_path.read_text())
    _require(acceptance["status"] == "ACCEPTED_ISOLATED_DATA"
             and acceptance["serving_admission"] is False, "original isolated acceptance")
    _require(file_sha256(package / "manifest.json") == acceptance["manifest_sha256"], "manifest digest")
    manifest = json.loads((package / "manifest.json").read_text())
    for name in ("plan.json", "model/config.json", JOB):
        _require(file_sha256(package / name) == manifest["files"][name], f"package digest: {name}")
    job = json.loads((package / JOB).read_text())
    argv = job["phases"][0]["steps"][1]["commands"][0]["argv"]
    plan_args = build_parser().parse_args(argv[3:])
    plan_args.model_config_path = package / "model/config.json"
    plan = build_plan(plan_args)
    _require(json.loads(json.dumps(plan.to_json())) == json.loads((package / "plan.json").read_text()),
             "reconstructed plan")
    inputs = {}
    for name, pin in acceptance["retained_files"].items():
        path = acceptance_path.parents[3] / name
        _require(file_sha256(path) == pin["sha256"] and path.stat().st_size == pin["bytes"],
                 f"retained digest/size: {name}")
        inputs[path.name] = path
    identity = json.loads(inputs["dsv4_aggregate_summary.json"].read_text())["identity"]
    _require(identity["source_revision"] == acceptance["source_revision"] == manifest["source_revision"],
             "source identity")
    _require(identity["model_config_sha256"] == manifest["files"]["model/config.json"], "model identity")
    return plan, job, identity, inputs


def prefill_layouts(plan, identity, inputs):
    records = read_records([inputs["dsv4_raw_e0.jsonl"]])
    return {name: build_isolated_prefill_layouts(plan, records, expected_identity={
        key: identity[key] for key in ("source_revision", "runtime_fingerprint", "model_config_sha256")},
        input_file_sha256=file_sha256(inputs[name]))
        for name in ("attention.csv", "attention_kernel_only.csv")}


def prepare_attention_input(acceptance_path, package, output):
    plan, _, identity, inputs = accepted_prefill_inputs(acceptance_path, package)
    layouts = prefill_layouts(plan, identity, inputs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = {"acceptance": Path(acceptance_path), "manifest": Path(package) / "manifest.json",
        "plan": Path(package) / "plan.json", "model": Path(package) / "model/config.json",
        "job": Path(package) / JOB, "raw": inputs["dsv4_raw_e0.jsonl"],
        "summary": inputs["dsv4_aggregate_summary.json"]}
    evidence = {}
    (output / "evidence").mkdir()
    for name, source in sources.items():
        destination = output / "evidence" / f"{name}{source.suffix}"
        shutil.copyfile(source, destination)
        evidence[name] = {"path": str(destination.relative_to(output)), "sha256": file_sha256(source)}
        _require(file_sha256(destination) == evidence[name]["sha256"], f"copied evidence: {name}")
    evidence["job"]["manifest_path"] = JOB
    tables = {}
    for name, method in (("attention.csv", "CUDA_EVENT"), ("attention_kernel_only.csv", "KERNEL_ONLY")):
        shutil.copyfile(inputs[name], output / name)
        digest = file_sha256(inputs[name])
        _require(file_sha256(output / name) == digest, f"copied table: {name}")
        tables[name] = {"file": {"path": name, "sha256": digest}, "measurement_type": method,
                        "layouts": layouts[name]}
    sidecar = output / "dsv4_attention.json"
    sidecar.write_text(json.dumps({"schema_version": SCHEMA, "qualification": QUALIFICATION,
        "serving_admission": False, "identity": {name: identity[name] for name in (
            "source_revision", "runtime_fingerprint", "model_config_sha256")},
        "evidence": evidence, "tables": tables}, indent=2, allow_nan=False) + "\n")
    return sidecar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-prefill", type=Path, required=True)
    parser.add_argument("--prefill-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = prepare_attention_input(args.accepted_prefill, args.prefill_package, args.output)
    print(f"CANDIDATE_REQUIRES_EXTERNAL_REVIEW {sidecar}; no admission issued")


if __name__ == "__main__":
    main()
