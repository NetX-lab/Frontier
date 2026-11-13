"""Shared immutable source packages for Step-image profiling jobs."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path


IMAGE = ("a.z01.normandyfunc.com/docker-public/vllm/vllm-openai"
         "@sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20")
CLOUD_MOUNT = "juicefs+s3://oss.i.shaipower.com/codesign-exp:/mnt/codesign-exp"
CLOUD_ROOT = Path("/mnt/codesign-exp/ycfeng/frontier/step5_profiling")
STEPMIND_ENV = Path("/data/ycfeng/tmp/step5_native_pd_r1/stepmind_env.sh")


def describe_job(stage, run_id, *, worker_command, gpus=1,
                  cloud_root=CLOUD_ROOT, credential_env=STEPMIND_ENV,
                  gpu_tag="H800", charged_group="codesign", timeout_seconds=3600, preemptible=False, image=IMAGE):
    """Describe the existing personal RJob launcher without invoking it."""
    stage = Path(stage).resolve()
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a single nonempty path component")
    q = shlex.quote
    return {
        "image": image, "gpu_tag": gpu_tag, "charged_group": charged_group,
        "gpus": gpus, "cpus": 8 * gpus, "mem_gb": 32 * gpus,
        "mounts": [f"local_nfs:{stage}:{stage}", CLOUD_MOUNT],
        "source_mount": str(stage), "cloud_output": str(Path(cloud_root) / run_id),
        "timeout_seconds": timeout_seconds, "preemptible": preemptible,
        "credential_env": str(credential_env),
        "submit_wrapper_command": f"bash {q(str(stage / 'submit_with_env.sh'))}",
        "worker_command": worker_command,
        "submission_command": (
            "/data/ycfeng/tmp/stepmind-env/bin/python -u -B "
            f"{q(str(stage / 'tests/integration/step5_profiling/submit_profiling_job.py'))} "
            f"--job-tag {q(run_id)} --gpus {gpus} --cpus {8*gpus} --mem-gb {32*gpus} "
            f"--gpu-tag {q(gpu_tag)} --charged-group {q(charged_group)} "
             f"--timeout-seconds {timeout_seconds} "
             f"--image {q(image)} "
             f"{'--preemptible ' if preemptible else ''}{q(worker_command)}"),
    }


def stage_package(stage, manifest, *, repo_root, dependency_site, additional_paths=(), additional_files=()):
    """Copy sources and offline dependencies once, hashing every payload file."""
    stage = Path(stage).resolve()
    repo_root = Path(repo_root)
    dependency_site = Path(dependency_site).resolve(strict=True)
    stage.mkdir(parents=True, exist_ok=False)
    shutil.copytree(repo_root / "frontier", stage / "frontier",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    shutil.copytree(repo_root / "data/config", stage / "data/config")
    for relative_path in additional_paths:
        target = stage / relative_path
        target.resolve().relative_to(stage)
        if (repo_root / relative_path).is_dir():
            shutil.copytree(repo_root / relative_path, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo_root / relative_path, target)
    for source, relative in additional_files:
        target = stage / relative
        target.resolve().relative_to(stage)
        if target.exists():
            raise ValueError(f"external profile artifact duplicates staged source: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copytree(dependency_site, stage / "offline_site",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    scripts = stage / "tests/integration/step5_profiling"
    scripts.mkdir(parents=True)
    for name in ("submit_profiling_job.py", "run_on_worker.sh", *manifest.get("verification_scripts", ())):
        shutil.copy2(repo_root / "tests/integration/step5_profiling" / name, scripts / name)
    (stage / "submit.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" +
                                    manifest["submission_command"] + "\n")
    (stage / "submit_with_env.sh").write_text("\n".join([
        "#!/usr/bin/env bash", "set -euo pipefail", "set +x",
        f"source {shlex.quote(manifest['credential_env'])}",
        ': "${BRAINPP_ACCESS_KEY:?personal BRAINPP_ACCESS_KEY is not set}"',
        ': "${BRAINPP_SECRET_KEY:?personal BRAINPP_SECRET_KEY is not set}"',
        "export STEPMIND_BACKEND=rjob", "export TMPDIR=/data/ycfeng/tmp",
        'export PYTHONPATH="/data/ycfeng/steptron${PYTHONPATH:+:$PYTHONPATH}"',
        f"exec bash {shlex.quote(str(stage / 'submit.sh'))}", "",
    ]))
    manifest = dict(manifest)
    if "native_binding" in manifest:
        source = manifest["source"]
        for path in stage.rglob("*"):
            if path.is_file() and path.relative_to(stage).parts[0] in ("frontier", "data", "tests"):
                relative = str(path.relative_to(stage))
                committed = subprocess.check_output(["git", "show", f"{source['commit']}:{relative}"], cwd=source["repository"])
                if path.read_bytes() != committed:
                    raise ValueError(f"staged source differs from committed payload: {relative}")
    manifest["sha256"] = {
        str(path.relative_to(stage)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(stage.rglob("*")) if path.is_file()}
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def profile_native_binding(plan, run_id, repo_root):
    """Bind an explicit generation to the existing native adapter and receipts."""
    from tests.integration.step5_pd.build_stage import adapter_record, committed_source, installed_source_hashes
    from vllm_replay import check_native_reference
    adapter = adapter_record(plan["vllm_version"])
    check_native_reference(plan["native_reference"], adapter["native_reference"])
    reference = plan["native_reference"]
    if plan["generation"] != run_id or reference["routing_strategy"] != "position_cyclic":
        raise ValueError("explicit profile generation/runtime must match selected position adapter")
    source = committed_source()
    if source != plan["source"] or Path(source["repository"]).resolve() != Path(repo_root).resolve():
        raise ValueError("profile plan must use the current committed Frontier source")
    config_relative = "data/config/models/stepfun-ai__Step-5-Preview.json"
    config = (Path(repo_root) / config_relative).read_bytes()
    committed_config = subprocess.check_output(["git", "show", f"{plan['model_config_revision']}:{config_relative}"], cwd=repo_root)
    if config != committed_config or hashlib.sha256(config).hexdigest() != plan["model_config_sha256"]:
        raise ValueError("effective local model configuration differs from its real revision/hash")
    pins = installed_source_hashes(Path(plan["image_source_hashes"]), plan["image_paths"]["installed_package"])
    receipt = json.loads(Path(plan["native_source_receipt"]).read_text())
    derivation = json.loads(Path(plan["native_derivation_manifest"]).read_text())
    if len(pins) != 53 or any(receipt["sources"].get(key) != digest for key, digest in pins.items()):
        raise ValueError("profile serving53 pins differ from full installed-source receipt")
    if receipt["image"] != reference["image"] or receipt["base_image"] != reference["base_image"] or receipt["installed_package"] != plan["image_paths"]["installed_package"]:
        raise ValueError("installed receipt image/root differs from selected adapter")
    if (receipt["derivation_manifest_sha256"] != hashlib.sha256(Path(plan["native_derivation_manifest"]).read_bytes()).hexdigest()
            or derivation["base_image"] != reference["base_image"]
            or any(pins.get(key) != digest for key, digest in derivation["derived_source_sha256"].items())):
        raise ValueError("native derivation differs from installed image sources")
    return {"native_runtime_adapter": plan["vllm_version"], "native_reference": reference,
            "profile_generation": run_id, "installed_package": receipt["installed_package"],
            "installed_source_hashes": pins, "serving_pin_count": 53,
            "model_config_revision": plan["model_config_revision"], "model_config_sha256": plan["model_config_sha256"],
            "native_source_receipt": "profile_artifacts/native_installed_sources.json",
            "native_derivation_manifest": "profile_artifacts/derivation_manifest.json"}
