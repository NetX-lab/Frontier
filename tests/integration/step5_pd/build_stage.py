"""Stage one native Step-5-Preview P/D run for native_pd_worker.py and submit_native_pd.py.

The run definition (`r1_run.json` for ticket 08 run R1) gives the resources, engine, ports,
timeouts and waves. The stage directory, which must lie inside the local workspace that the
job mounts, receives:

- the shared harness in its repository layout under `tests/`, and the worker and the pinned
  probe helpers at its root, because the worker imports the helpers from its own directory;
- `engine.json`, the base engine both roles serve, completed from the adapter record;
- `trace/waves.json` and one pd_replay trace directory per wave;
- `preflight.json`, the node plan, server command and environment of every node position,
  built by the same functions the worker calls, with placeholder hosts;
- `config.json`, which the worker and the submitter read.

The staged code must be committed: the stage records the source commit and the SHA-256 of
every staged file, the installed vLLM sources and the model metadata the worker checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
CALIBRATION = REPO / "tests/comparison/calibration"
sys.path[:0] = [str(CALIBRATION), str(REPO / "tests/comparison/dp_placement_pp")]
from native_pd_worker import node_plan  # noqa: E402
from pd_replay import instance_devices, instance_env, kv_transfer_config  # noqa: E402
from serving_metrics import expected_pull_prompt_sources  # noqa: E402
from vllm_replay import adapter_record, local_model_config, runtime_version, server_command  # noqa: E402

HARNESS_FILES = (
    "tests/comparison/calibration/pd_replay.py",
    "tests/comparison/calibration/pd_proxy.py",
    "tests/comparison/calibration/pd_metrics.py",
    "tests/comparison/calibration/serving_metrics.py",
    "tests/comparison/dp_placement_pp/vllm_replay.py",
    "tests/comparison/stage_admission_pp/vllm_burst_driver.py",
    "tests/integration/step5_pd/run_waves.py",
)
WORKER = "tests/comparison/calibration/native_pd_worker.py"
CLOUD_ROOT = Path("/mnt/codesign-exp/ycfeng")
ROLES = ("prefill", "decode")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def engine_config(definition: dict) -> dict:
    """Complete the run's engine with the adapter's version, image and tokenizer setting."""
    version = definition["vllm_version"]
    adapter = adapter_record(version)
    engine = dict(definition["engine"], vllm_version=version,
                  native_reference=dict(adapter["native_reference"]),
                  skip_tokenizer_init=adapter["skip_tokenizer_init"])
    runtime_version(engine)
    return engine


def installed_source_hashes(listing: Path, installed_package: str) -> dict[str, str]:
    """Map package-relative names to the SHA-256 recorded inside the image."""
    prefix = str(Path(installed_package).parent) + "/"
    hashes = {}
    for line in listing.read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        if path.startswith(prefix):
            hashes[path[len(prefix):]] = digest
    if not hashes:
        raise ValueError(f"{listing} lists no file under {prefix}")
    return hashes


def wave_traces(definition: dict, namespace: str) -> dict[str, list[dict]]:
    traces = {}
    for wave in definition["waves"]:
        traces[wave["name"]] = [
            {"request_id": f"{namespace}-{wave['name']}-{index}", "segment": wave["name"],
             "role": "gating" if wave["name"] == definition["gating_wave"] else "recorded",
             "arrived_at": round(index * wave["spacing_s"], 6), "num_prefill_tokens": length,
             "num_decode_tokens": wave["num_decode_tokens"]}
            for index, length in enumerate(wave["prompt_lengths"])]
    return traces


def preflight(config: dict, model_dir: Path) -> list[dict]:
    """Run the worker's node plan and the server preparation for every node position."""
    replica, gpu = config["replica"], config["gpu"]
    peers = [{"host": f"node{index}", "ip": f"10.255.0.{index + 1}"} for index in range(replica)]
    devices = [str(index) for index in range(gpu)]
    rows = []
    for peer in peers:
        plan = node_plan(config, peers, peer["host"], devices)
        role, port = plan["role"], config["http_ports"][plan["role"]]
        selected = instance_devices({}, engine=plan["engine"], role=role)
        env, changed = instance_env(config["mode"], plan["engine"], Path("/nonexistent"), {},
                                    selected[role], observer=None, role=role)
        command = server_command(plan["engine"], model_dir, port) + [
            "--kv-transfer-config", json.dumps(kv_transfer_config(plan["engine"], role, port))]
        rows.append({"host": peer["host"], "position": plan["position"], "role": role,
                     "lane": plan["lane"], "headless": plan["engine"]["headless"],
                     "urls": plan["urls"], "server_command": command[1:], "server_env_set": changed})
    return rows


def build(definition: dict, *, stage: Path, workspace: Path, output_dir: Path, model_dir: Path,
          image_source_hashes: Path, probe_dir: Path, job_tag: str, source: dict) -> dict:
    if not stage.resolve().is_relative_to(workspace.resolve()):
        raise ValueError("The stage must lie inside the workspace the job mounts")
    if not output_dir.is_absolute() or not output_dir.is_relative_to(CLOUD_ROOT):
        raise ValueError(f"Cloud output must stay inside {CLOUD_ROOT}")
    version = definition["vllm_version"]
    adapter = adapter_record(version)
    local_model_config(model_dir, version)
    stage.mkdir(parents=True, exist_ok=False)
    for relative in HARNESS_FILES:
        (stage / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, stage / relative)
    shutil.copy2(REPO / WORKER, stage / Path(WORKER).name)
    for name, expected in definition["probe_helpers"].items():
        if sha256(probe_dir / name) != expected:
            raise ValueError(f"Pinned probe helper changed: {probe_dir / name}")
        shutil.copy2(probe_dir / name, stage / name)
    engine = engine_config(definition)
    (stage / "engine.json").write_text(json.dumps(engine, indent=1, sort_keys=True))
    block_size = engine["block_size"]
    expected_sources = {}
    for wave, rows in wave_traces(definition, job_tag).items():
        (stage / "trace" / wave).mkdir(parents=True)
        (stage / "trace" / wave / "request_ids.json").write_text(json.dumps({"rows": rows}, indent=1))
        totals = {"local_compute": 0, "local_cache_hit": 0, "external_kv_transfer": 0}
        for row in rows:
            for name, value in expected_pull_prompt_sources(row["num_prefill_tokens"], block_size).items():
                totals[name] += value
        expected_sources[wave] = totals
    lease = adapter["kv_connector_extra_config"]["kv_lease_duration"]
    (stage / "trace" / "waves.json").write_text(json.dumps({
        "waves": [wave["name"] for wave in definition["waves"]],
        "gating_wave": definition["gating_wave"],
        "release_wait_s": lease + definition["release_margin_s"],
        "scrape_timeout_s": definition["scrape_timeout_s"],
        "expected_decode_prompt_sources": expected_sources}, indent=1))
    resources, paths = definition["resources"], definition["image_paths"]
    config = {
        "description": definition["description"], "job_tag": job_tag, "source": source,
        "workspace": str(workspace), "script": str(stage / Path(WORKER).name),
        "harness_dir": str(stage / "tests/comparison/calibration"),
        "replay_script": str(stage / "tests/integration/step5_pd/run_waves.py"),
        "evidence_dir": str(stage / "evidence"), "output_dir": str(output_dir), "mode": "clean",
        "model_dir": str(model_dir),
        "model_hashes": {path.name: sha256(path) for path in model_dir.iterdir()},
        "installed_package": paths["installed_package"],
        "expected_source_hashes": installed_source_hashes(image_source_hashes, paths["installed_package"]),
        "cuda_compat_path": paths["cuda_compat_path"], "image": engine["native_reference"]["image"],
        "engine": engine, "data_parallel_size_local": definition["data_parallel_size_local"],
        **{name: definition[name] for name in (
            "http_ports", "dp_rpc_ports", "proxy_port", "peer_wait_seconds", "startup_timeout_s",
            "replay_timeout_s", "drain_seconds", "stop_timeout_s", "cache_environment",
            "weight_suffixes")},
        **resources,
    }
    (stage / "preflight.json").write_text(json.dumps(preflight(config, model_dir), indent=1))
    config["tool_hashes"] = {str(path.relative_to(stage)): sha256(path)
                             for path in sorted(stage.rglob("*")) if path.is_file()}
    (stage / "config.json").write_text(json.dumps(config, indent=1))
    return config


def committed_source() -> dict:
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    if status.strip():
        raise ValueError("Commit the harness before staging; the stage records the source commit")
    return {"repository": str(REPO),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--run-definition", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image-source-hashes", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--job-tag", required=True)
    args = parser.parse_args()
    staged = build(json.loads(args.run_definition.read_text()), stage=args.stage_dir.resolve(),
                   workspace=args.workspace, output_dir=args.output_dir,
                   model_dir=args.model_dir.resolve(), image_source_hashes=args.image_source_hashes,
                   probe_dir=args.probe_dir, job_tag=args.job_tag, source=committed_source())
    print("STAGED", args.stage_dir, staged["source"]["commit"], flush=True)
