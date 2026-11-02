"""Check package imports and retained image pins on CPU; never execute the probe."""

import ast
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import shlex
import sys


def main():
    package = Path(sys.argv[1]).resolve()
    config = json.loads((package / "config.json").read_text())
    manifest = json.loads((package / "manifest.json").read_text())
    lock = json.loads((package / "source_lock.json").read_text())
    assert manifest["status"] == "PREPARED_NOT_SEALED" and not manifest["submitted"]
    assert len(manifest["files"]) == manifest["input_count"]
    assert {str(p.relative_to(package)) for p in package.rglob("*") if p.is_file()} == set(manifest["files"]) | {"manifest.json"}
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == expected, name
    for name, pin in lock["payloads"].items():
        assert manifest["files"][name] == pin["sha256"]
    assert config["image"] == lock["image"]
    saved_argv = sys.argv
    sys.argv = [str(package / "launch.py")]
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        runpy.run_path(str(package / "launch.py"), run_name="__main__")
    sys.argv = saved_argv
    dry = json.loads(captured.getvalue())
    assert not dry["submitted"] and dry["config"] == config
    assert config["script"] == str(package / next(iter(lock["payloads"])))
    assert lock["argv"] == ["python3", config["script"], "--output-dir", config["output_dir"],
                            "--image-digest", config["image"].split("@", 1)[1],
                            "--elements", str(config["elements"]), "--warmup", str(config["warmup"])]
    worker = ast.parse(shlex.split(dry["command"])[-1])
    exec_call = worker.body[-1].value
    assert isinstance(exec_call, ast.Call) and ast.unparse(exec_call.func) == "os.execv"
    assert ast.literal_eval(exec_call.args[1].right) == lock["argv"][1:]
    os.environ.update(config["envs"])
    spec = importlib.util.spec_from_file_location("scope_package_probe", config["script"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.HERE.parents[3] == package
    assert module.probe.callback_abi() == lock["callback_abi"]
    profile = module.serving_profile()
    assert not profile.errors and profile.limits == json.loads(config["envs"]["FRONTIER_NATIVE_PROFILE_LIMITS"])
    import torch
    import cuda.bindings.runtime as runtime
    assert not torch.cuda.is_initialized()
    assert torch.__version__ == lock["torch_version"] and torch.version.cuda == lock["torch_cuda_version"]
    assert importlib.metadata.version("cuda-bindings") == lock["cuda_bindings_version"]
    assert importlib.metadata.version("nvidia-cuda-cupti") == lock["cupti_version"]
    assert all(hasattr(runtime, name) for name in ("cudaGraphGetNodes", "cudaGraphGetEdges", "cudaGraphNodeGetType"))
    assert all(hasattr(torch.cuda.CUDAGraph, name) for name in ("raw_cuda_graph", "raw_cuda_graph_exec"))
    assert str(module.probe.cupti_library()) in lock["binaries"]
    for name, pin in lock["binaries"].items():
        path, digest = Path(name), hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(2**20), b""):
                digest.update(chunk)
        assert str(path.resolve()) == pin["resolved"] and path.stat().st_size == pin["bytes"]
        assert digest.hexdigest() == pin["sha256"], name
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status="PASS_CPU_PACKAGE_IMAGE_PINS", python=sys.version,
        manifest_sha256=hashlib.sha256((package / "manifest.json").read_bytes()).hexdigest(),
        source_sha256=config["source_sha256"], input_count=manifest["input_count"],
        cuda_initialized=False, binaries=lock["binaries"], launcher_dry_run=dry,
        limits="CPU imports/ABI/binary assertions only; no SDK, TLS, lifetime or GPU qualification.")))


if __name__ == "__main__":
    main()
