"""Verify a frozen R2 package locally, without importing vLLM or calling a platform API."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(stage: Path) -> dict:
    config = json.loads((stage / "config.json").read_text())
    definition = json.loads((stage / "r2_run.json").read_text())
    assert config["diagnostic_only"] and config["clean_timing_eligible"] is False
    assert (config["charged_group"], config["gpu_type"], config["replica"], config["gpu"]) == ("codesign", "H800", 4, 8)
    assert config["role_num_gpu_blocks_override"] == {"prefill": 1024, "decode": 512}
    assert config["image"].endswith("@sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20")
    engine = config["engine"]
    assert (engine["tensor_parallel_size"], engine["data_parallel_size"], engine["pipeline_parallel_size"]) == (8, 2, 1)
    assert engine["enable_expert_parallel"] and engine["enable_prefix_caching"] is False
    assert "speculative_config" not in engine and "thinking" not in engine
    assert engine["load_format"] == "dummy" and engine["max_num_seqs"] == 4
    assert engine["max_model_len"] == 2048 and engine["max_num_batched_tokens"] == 512
    for name, digest in config["tool_hashes"].items():
        assert sha256(stage / name) == digest, name
    for name, digest in config["model_hashes"].items():
        assert sha256(Path(config["model_dir"]) / name) == digest, name
    observation = config["pd_state_observation"]
    source_root = Path(definition["pd_state_observation"]["source_root"])
    for module, digest in (observation["source_hashes"] | observation["schema_source_hashes"]).items():
        assert sha256(source_root / (module.replace(".", "/") + ".py")) == digest, module
    for name, digest in observation["frozen_environment_hashes"].items():
        assert sha256(stage / name) == digest, name
        original = Path(config["source"]["repository"]) / "tests/integration/step5_environment" / name
        assert (stage / name).read_bytes() == original.read_bytes(), name
    rows = [row for path in sorted((stage / "trace").glob("*/request_ids.json"))
            for row in json.loads(path.read_text())["rows"]]
    assert len(rows) == len({row["request_id"] for row in rows}) == 20
    assert sorted(observation["selected_client_request_ids"]) == sorted(row["request_id"] for row in rows)
    content = set(observation["content_client_request_ids"])
    native_only = set(observation["readiness_only_client_request_ids"])
    assert len(content) == 12 and len(native_only) == 8 and not (content & native_only)
    assert content | native_only == {row["request_id"] for row in rows}
    assert all(row["stream"] and row["num_prefill_tokens"] + row["num_decode_tokens"] <= 2048 for row in rows)
    concurrent = [row for row in rows if row["segment"] == "concurrent6"]
    assert len(concurrent) == 6 and all(row["arrived_at"] == 0 and row["num_decode_tokens"] == 128 for row in concurrent)
    assert sum(row.get("cancel_after_first_content", False) for row in rows) == 1
    for boundary in (8, 16):
        assert {boundary - 1, 0, 1} <= {row["num_prefill_tokens"] % boundary for row in rows}
    plans = json.loads((stage / "preflight.json").read_text())
    assert [(plan["role"], plan["lane"]) for plan in plans] == [("prefill", 0), ("prefill", 1), ("decode", 0), ("decode", 1)]
    for plan in plans:
        command = plan["server_command"]
        expected = str(config["role_num_gpu_blocks_override"][plan["role"]])
        assert command[command.index("--num-gpu-blocks-override") + 1] == expected
        env = json.loads(plan["server_env_set"]["STEP5_PD_STATE_OBSERVATION"])
        assert env["role"] == plan["role"] and env["output_dir"] == "/worker-local-scratch/pd_state"
        assert plan["server_env_set"]["PYTHONPATH"].startswith(str(stage / "r2_bootstrap"))
    for name in ("r2_observation.py", "r2_bootstrap/sitecustomize.py", "r2_verdict.py"):
        compile((stage / name).read_text(), str(stage / name), "exec")
    return {
        "observed_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS_OFFLINE_STAGE",
        "native_qualification": "NOT_RUN", "submissions_or_controls": 0, "stage": str(stage),
        "source_commit": config["source"]["commit"], "config_sha256": sha256(stage / "config.json"),
        "tool_files": len(config["tool_hashes"]), "installed_native_source_pins": len(config["expected_source_hashes"]),
        "observation_source_pins": len(observation["source_hashes"]),
        "schema_source_pins": len(observation["schema_source_hashes"]),
        "frozen_environment_files": observation["frozen_environment_hashes"],
        "resources": {key: config[key] for key in ("charged_group", "gpu_type", "replica", "gpu", "timeout_seconds")},
        "requests": len(rows), "waves": len(definition["waves"]), "concurrent_requests": len(concurrent),
        "content_requests": len(content), "native_readiness_only_requests": len(native_only),
        "diagnostic_only": True, "clean_timing_eligible": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.stage_dir.resolve())
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))
