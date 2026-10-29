"""Prepare the frozen mirrored scope/lifetime package; never seal or submit it."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
TASK = Path("task_memory/task_2026-10-23_v4_remaining_measurements")
SCOPE = TASK / "profile01/scope_lifetime"
GRAPH = TASK / "profile01/graph_probe"
REVISION = "d55782d7094be412b034a72600da55f47b4a4392"
SOURCE_SHA256 = "bc638fbdf0111a8ce78383d182d41e19a5799bacc6cc5a657aa79b1a34672685"


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def prepare():
    acceptance = json.loads((ROOT / SCOPE / "primary_acceptance.json").read_text())
    assert acceptance["source_accepted"] and acceptance["source_commit"] == REVISION
    assert acceptance["source_test_sha256"][str(SCOPE / "probe_scope_lifetime.py")] == SOURCE_SHA256
    bindings = json.loads((ROOT / SCOPE / "source_binding.json").read_text())["bindings"]
    paths = [SCOPE / "probe_scope_lifetime.py", SCOPE / "source_binding.json"]
    paths += [Path(row["path"]) for row in bindings[:3]]
    payloads = {}
    for path in paths:
        content = subprocess.check_output(["git", "show", f"{REVISION}:{path}"], cwd=ROOT)
        assert content == (ROOT / path).read_bytes(), f"Frozen payload changed: {path}"
        payloads[str(path)] = content
    assert sha256(payloads[str(paths[0])]) == SOURCE_SHA256
    for row in bindings[:3]:
        assert sha256(payloads[row["path"]]) == row["sha256"]
    seal = json.loads((ROOT / GRAPH / "root_seal.json").read_text())
    graph_package = Path(seal["package"])
    graph_manifest = json.loads((graph_package / "manifest.json").read_text())
    assert sha256((graph_package / "manifest.json").read_bytes()) == seal["manifest_sha256"]
    for name, expected in graph_manifest["files"].items():
        assert sha256((graph_package / name).read_bytes()) == expected
    graph_lock = json.loads((graph_package / "source_lock.json").read_text())
    template_root = Path(graph_lock["template"]["path"])
    template = (template_root / "launch.py").read_text()
    old_config = json.loads((template_root / "config.json").read_text())
    template_manifest = json.loads((template_root / "manifest.json").read_text())
    assert sha256((template_root / "manifest.json").read_bytes()) == graph_lock["template"]["manifest_sha256"]
    for name, expected in template_manifest["files"].items():
        assert sha256((template_root / name).read_bytes()) == expected
    image_paths = [GRAPH / "package_checks/image/receipt.json", GRAPH / "package_checks/image/probe/probe_report.json"]
    image_receipt, image_report = [json.loads((ROOT / p).read_text()) for p in image_paths]
    sealed_checks = {row["path"]: row["sha256"] for row in seal["checks"]}
    assert sha256((ROOT / image_paths[0]).read_bytes()) == sealed_checks[str(image_paths[0])]
    assert image_receipt["manifest_sha256"] == seal["manifest_sha256"]
    assert old_config["image"] == seal["resources"]["image"]
    assert image_report["image_digest_from_launcher"] == old_config["image"].split("@", 1)[1]
    plan = json.loads((HERE / "profile_plan.json").read_text())
    package = HERE.parent / "h800_scope_lifetime_probe_01"
    config = old_config | plan
    config.update(workspace=str(package), script=str(package / paths[0]),
                  source_sha256=SOURCE_SHA256, source_revision=REVISION,
                  output_dir=plan["cloud_run_root"] + "/probe",
                  diagnostics_dir=plan["cloud_run_root"] + "/diagnostics")
    config["envs"] = plan["envs"] | {"FRONTIER_GRAPH_DIAGNOSTICS_DIR": config["diagnostics_dir"]}
    argv = ["python3", config["script"], "--output-dir", config["output_dir"],
            "--image-digest", config["image"].split("@", 1)[1],
            "--elements", str(config["elements"]), "--warmup", str(config["warmup"])]
    transformations = [
        ("Submit the native stream/capture probe", "Submit the standalone scope/lifetime probe"),
        ("Reviewed native profile probe source changed", "Reviewed scope/lifetime probe source changed"),
        ('    script = Path(config["script"])',
         '    manifest = json.loads((HERE / "manifest.json").read_text())\n'
         '    for name, expected in manifest["files"].items():\n'
         '        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:\n'
         '            raise ValueError(f"Prepared scope/lifetime package input changed: {name}")\n'
         '    script = Path(config["script"])'),
        ('argv = ["python3", str(script), "--output-dir", config["output_dir"]]',
         'argv = ["python3", str(script), "--output-dir", config["output_dir"],\n'
         '            "--image-digest", config["image"].split("@", 1)[1],\n'
         '            "--elements", str(config["elements"]), "--warmup", str(config["warmup"])]'),
        ('"import hashlib,os,pathlib,sys; "',
         '"import hashlib,json,os,pathlib,sys; "\n'
         '        f"manifest=json.loads(pathlib.Path({str(HERE / \'manifest.json\')!r}).read_text()); "\n'
         '        f"assert all(hashlib.sha256((pathlib.Path({str(HERE)!r})/n).read_bytes()).hexdigest()==h for n,h in manifest[\'files\'].items()); "'),
        ('        f"os.execv(sys.executable,[sys.executable]+{argv[1:]!r})"',
         '        f"pathlib.Path({config[\'diagnostics_dir\']!r}).mkdir(parents=True,exist_ok=False); "\n'
         '        f"os.execv(sys.executable,[sys.executable]+{argv[1:]!r})"'),
    ]
    launch = template
    for before, after in transformations:
        assert launch.count(before) == 1, f"Launcher template mismatch: {before}"
        launch = launch.replace(before, after)
    assert len(launch.splitlines()) <= 100
    compile(launch, str(package / "launch.py"), "exec")
    inputs = {name: HERE / name for name in ("build_package.py", "profile_plan.json", "image_check.py")}
    inputs.update(launcher_template=template_root / "launch.py", template_config=template_root / "config.json",
                  graph_image_receipt=ROOT / image_paths[0], graph_image_report=ROOT / image_paths[1],
                  graph_manifest=graph_package / "manifest.json", graph_seal=ROOT / GRAPH / "root_seal.json",
                  source_acceptance=ROOT / SCOPE / "primary_acceptance.json")
    files = payloads | {"launch.py": launch.encode(), "config.json": encoded(config)}
    files.update({"inputs/" + name: path.read_bytes() for name, path in inputs.items()})
    lock = dict(source_revision=REVISION, payloads={name: dict(sha256=sha256(content), bytes=len(content),
                lines=len(content.splitlines()), revision=REVISION) for name, content in payloads.items()},
                inputs={name: dict(path=str(path), sha256=sha256(path.read_bytes())) for name, path in inputs.items()},
                image=config["image"], callback_abi=image_report["api"]["callback_abi"],
                binaries=image_receipt["binaries"], torch_version=image_report["torch_version"],
                torch_cuda_version=image_report["torch_cuda_version"],
                cuda_bindings_version=image_report["api"]["cuda_bindings_version"],
                cupti_version=image_report["api"]["cupti_version"], argv=argv,
                launcher_transformations=transformations,
                qualification="CPU/source preparation only; root image/SDK/runtime qualification required.")
    files["source_lock.json"] = encoded(lock)
    files["manifest.json"] = encoded(dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
        status="PREPARED_NOT_SEALED", source_revision=REVISION,
        files={name: sha256(content) for name, content in sorted(files.items())},
        input_count=len(files), submitted=False))
    return package, files


def main():
    package, files = prepare()
    package.mkdir()
    for name, content in files.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    print(json.dumps(dict(package=str(package), inputs=len(files) - 1,
        bytes=sum(map(len, files.values())), launcher_lines=len(files["launch.py"].splitlines()),
        manifest_sha256=sha256(files["manifest.json"]), sealed=False)))


if __name__ == "__main__":
    main()
