"""Freeze the opt-in R2 capture contract beside the existing native PD stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from tests.integration.step5_pd.r2_observation import NATIVE_SOURCE_PATHS

REPO = Path(__file__).resolve().parents[3]
RUNTIME_MODULES = tuple(path[:-3].replace("/", ".") for path in NATIVE_SOURCE_PATHS)
SCHEMA_MODULES = (
    "vllm.v1.kv_cache_interface", "vllm.v1.core.sched.output",
    "vllm.v1.engine.input_processor", "vllm.entrypoints.openai.completion.serving",
)
CAPTURE_FILES = {
    "tests/integration/step5_pd/r2_observation.py": "r2_observation.py",
    "tests/integration/step5_pd/r2_bootstrap/sitecustomize.py": "r2_bootstrap/sitecustomize.py",
    "tests/integration/step5_pd/r2_verdict.py": "r2_verdict.py",
    "tests/integration/step5_pd/verify_r2_stage.py": "verify_r2_stage.py",
    "tests/integration/step5_environment/role_state_observation.py": "role_state_observation.py",
    "tests/integration/step5_environment/role_capacity_observation.py": "role_capacity_observation.py",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_observation(config: dict, definition: dict, stage: Path, model_dir: Path) -> None:
    """Add diagnostic-only inputs before the canonical builder freezes tool hashes."""
    if definition["vllm_version"] != "step5":
        raise ValueError("R2 state observation requires the locked Step image")
    observation = dict(definition["pd_state_observation"])
    source_root = Path(observation.pop("source_root"))
    observation["source_hashes"] = {
        module: sha256(source_root / (module.replace(".", "/") + ".py"))
        for module in RUNTIME_MODULES}
    observation["schema_source_hashes"] = {
        module: sha256(source_root / (module.replace(".", "/") + ".py"))
        for module in SCHEMA_MODULES}
    for source, destination in CAPTURE_FILES.items():
        path = stage / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / source, path)
    observation["frozen_environment_hashes"] = {
        name: sha256(stage / name) for name in (
            "role_state_observation.py", "role_capacity_observation.py")}
    metadata = json.loads((model_dir / "config.json").read_text())["text_config"]
    observation["expected_layer_count"] = metadata["num_hidden_layers"]
    layer_types = metadata["layer_types"][:metadata["num_hidden_layers"]]
    if len(layer_types) != metadata["num_hidden_layers"]:
        raise ValueError("Model metadata has an incomplete native layer-type census")
    observation["expected_summary_layer_count"] = sum(
        kind in metadata["sparse_config"]["apply_to_layer_types"] for kind in layer_types)
    observation["block_size"] = definition["engine"]["block_size"]
    rows = [
        row["request_id"] for path in sorted((stage / "trace").glob("*/request_ids.json"))
        for row in json.loads(path.read_text())["rows"]]
    observation["selected_client_request_ids"] = rows
    content_waves = {wave["name"] for wave in definition["waves"] if wave.get("capture_content", True)}
    observation["content_client_request_ids"] = [
        row["request_id"] for path in sorted((stage / "trace").glob("*/request_ids.json"))
        if path.parent.name in content_waves for row in json.loads(path.read_text())["rows"]]
    content = set(observation["content_client_request_ids"])
    observation["readiness_only_client_request_ids"] = [row for row in rows if row not in content]
    observation["concurrent_waves"] = [wave["name"] for wave in definition["waves"]
                                         if len(wave["prompt_lengths"]) > 1]
    (stage / "r2_run.json").write_text(json.dumps(definition, indent=1, sort_keys=True))
    config.update(pd_state_observation=observation, diagnostic_only=True,
                  clean_timing_eligible=False,
                  role_num_gpu_blocks_override=definition["role_num_gpu_blocks_override"])
