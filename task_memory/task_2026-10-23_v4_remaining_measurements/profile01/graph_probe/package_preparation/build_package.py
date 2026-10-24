"""Prepare the frozen graph probe package; never seal or submit it."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PRIMARY = Path("/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028")
TASK = Path("task_memory/task_2026-10-23_v4_remaining_measurements")
TEMPLATE = PRIMARY / TASK / "profile01/stream_probe/h800_stream_probe_01"
PROBE = TASK / "profile01/graph_probe/probe_diagnostic.py"
REVISION = "1aa1e4a5ce24bba8aafc8dd9f2fb41236d016077"
SOURCE_SHA256 = "a9d6a24e0aff13ef61088d75b5f693b599f42e8c50c2702342fbb84d3754d1a0"


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    source = subprocess.check_output(["git", "show", f"{REVISION}:{PROBE}"], cwd=ROOT)
    if sha256(source) != SOURCE_SHA256 or source != (ROOT / PROBE).read_bytes():
        raise ValueError("Frozen graph probe source changed")
    template = (TEMPLATE / "launch.py").read_text()
    old_config = json.loads((TEMPLATE / "config.json").read_text())
    manifest = json.loads((TEMPLATE / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256((TEMPLATE / name).read_bytes()) != expected:
            raise ValueError(f"Original stream package input changed: {name}")
    plan = json.loads((HERE / "profile_plan.json").read_text())
    package = HERE.parent / "h800_graph_probe_01"
    config = old_config | plan
    config.update(workspace=str(package), script=str(package / "probe_diagnostic.py"),
                  source_sha256=SOURCE_SHA256, source_revision=REVISION,
                  output_dir=plan["cloud_run_root"] + "/probe",
                  diagnostics_dir=plan["cloud_run_root"] + "/diagnostics")
    config["envs"] = plan["envs"] | {"FRONTIER_GRAPH_DIAGNOSTICS_DIR": config["diagnostics_dir"]}
    transformations = [
        ("Submit the native stream/capture probe", "Submit the corrected graph payload probe"),
        ("Reviewed native profile probe source changed", "Reviewed graph payload probe source changed"),
        ('    script = Path(config["script"])',
         '    manifest = json.loads((HERE / "manifest.json").read_text())\n'
         '    for name, expected in manifest["files"].items():\n'
         '        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:\n'
         '            raise ValueError(f"Prepared graph package input changed: {name}")\n'
         '    script = Path(config["script"])'),
        ('argv = ["python3", str(script), "--output-dir", config["output_dir"]]',
         'argv = ["python3", str(script), "run", "--output-dir", config["output_dir"],\n'
         '            "--image-digest", config["image"].split("@", 1)[1]]'),
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
        if launch.count(before) != 1:
            raise ValueError(f"Launcher template mismatch: {before}")
        launch = launch.replace(before, after)
    if len(launch.splitlines()) > 100:
        raise ValueError("Generated launcher exceeds100 lines")
    compile(launch, str(package / "launch.py"), "exec")
    package.mkdir()
    (package / "inputs").mkdir()
    (package / "probe_diagnostic.py").write_bytes(source)
    (package / "launch.py").write_text(launch)
    write_json(package / "config.json", config)
    inputs = {"build_package.py": HERE / "build_package.py",
              "profile_plan.json": HERE / "profile_plan.json",
              "launcher_template.py": TEMPLATE / "launch.py",
              "template_config.json": TEMPLATE / "config.json"}
    for name, path in inputs.items():
        (package / "inputs" / name).write_bytes(path.read_bytes())
    source_lock = {
        "source_revision": REVISION,
        "probe": {"path": str(PROBE), "revision": REVISION, "sha256": SOURCE_SHA256,
                  "bytes": len(source), "lines": len(source.splitlines())},
        "template": {"path": str(TEMPLATE), "manifest_sha256": sha256((TEMPLATE / "manifest.json").read_bytes()),
                     "files": {name: {"path": str(path), "sha256": sha256(path.read_bytes()),
                         "bytes": path.stat().st_size, "lines": len(path.read_bytes().splitlines())}
                         for name, path in inputs.items()}},
        "launcher_transformations": transformations,
        "argv": ["python3", config["script"], "run", "--output-dir", config["output_dir"],
                 "--image-digest", config["image"].split("@", 1)[1]],
        "source_default_workload": {"elements": 1024, "warmup": 3},
        "library_hash_reporting": "Probe API reports paths/versions/exports, not binary hashes; root image preflight required.",
        "qualification": "Diagnostic only; no runtime decode, performance or consumer admission.",
    }
    write_json(package / "source_lock.json", source_lock)
    files = sorted(path for path in package.rglob("*") if path.is_file())
    write_json(package / "manifest.json", {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PREPARED_NOT_SEALED",
        "source_revision": REVISION,
        "files": {str(path.relative_to(package)): sha256(path.read_bytes()) for path in files},
        "input_count": len(files), "submitted": False,
    })
    print(json.dumps({"package": str(package), "inputs": len(files), "launcher_lines": len(launch.splitlines()),
                      "manifest_sha256": sha256((package / "manifest.json").read_bytes()), "sealed": False}))


if __name__ == "__main__":
    main()
