"""Run one StepMind replica of the prepared native communication experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

PLATFORM_LOG_FD = None


def validate_topology(job: dict) -> None:
    """Reject a prepared topology that cannot fill the allocated rank space."""
    engine, resources = job["engine"], job["resources"]
    tp, dp, pp = (engine[name] for name in (
        "tensor_parallel_size", "data_parallel_size", "pipeline_parallel_size"))
    gpu, nodes, local = resources["gpu"], resources["replica"], job["gpus_per_node"]
    if any(type(n) is not int or n <= 0 for n in (tp, dp, pp, gpu, nodes, local)):
        raise ValueError("Parallel sizes and resource counts must be positive integers")
    if pp != 1 or not engine["enable_expert_parallel"]:
        raise ValueError("The native Step probe requires PP1 and expert parallelism")
    if local != gpu or tp * dp != gpu * nodes:
        raise ValueError("TP x DP must equal allocated world size; gpus_per_node must equal gpu")
    vectors = (job["native_probe"]["input_vectors"].values() if "profile_manifest" in job else
               [job["native_probe"]["tokens_per_dp_rank"]])
    if any(len(tokens) != dp or any(type(n) is not int or n <= 0 for n in tokens) for tokens in vectors):
        raise ValueError("The native probe needs one positive token count per DP rank")
    ports = [job["rendezvous"][key] for key in ("master_port", "probe_port")]
    if any(type(p) is not int or not 1 <= p <= 65535 for p in ports) or ports[0] == ports[1]:
        raise ValueError("Collective and probe rendezvous ports must be valid and distinct")


def node_identity(job: dict, env=None) -> dict:
    """Read the same platform rank variables as Steptron's native rtorchrun.sh."""
    validate_topology(job)
    env = os.environ if env is None else env
    rank, nodes, local = (int(env[key]) for key in ("NODE_RANK", "NODE_COUNT", "PROC_PER_NODE"))
    if nodes != job["resources"]["replica"] or local != job["gpus_per_node"]:
        raise ValueError("Platform NODE_COUNT/PROC_PER_NODE differs from prepared resources")
    if not 0 <= rank < nodes:
        raise ValueError("Platform NODE_RANK is outside the allocated nodes")
    address = env["MASTER_ADDR"]
    if not address:
        raise ValueError("MASTER_ADDR must not be empty")
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        if nodes > 1 and address.rstrip(".").lower() == "localhost":
            raise ValueError("Multi-node MASTER_ADDR must not be loopback")
    else:
        mapped = ip.ipv4_mapped if ip.version == 6 else None
        if nodes > 1 and (ip.is_loopback or ip.is_unspecified or
                          (mapped and (mapped.is_loopback or mapped.is_unspecified))):
            raise ValueError("Multi-node MASTER_ADDR must not be loopback or unspecified")
    return {"node_rank": rank, "node_count": nodes, "gpus_per_node": local,
            "master_addr": address, "ranks": list(range(rank * local, (rank + 1) * local))}


def torchrun_command(job: dict, phase: str, node: dict | None = None) -> list[str]:
    validate_topology(job)
    command = [sys.executable, "-m", "torch.distributed.run",
               f"--nproc-per-node={job['gpus_per_node']}"]
    if job["resources"]["replica"] == 1:
        command.append("--standalone")
    else:
        node = node_identity(job) if node is None else node
        port = job["rendezvous"]["master_port" if phase == "collective" else "probe_port"]
        command.extend([f"--nnodes={node['node_count']}", f"--node-rank={node['node_rank']}",
                        f"--master-addr={node['master_addr']}", f"--master-port={port}"])
    return command


def collective_command(job: dict, node: dict | None = None) -> list[str]:
    return [*torchrun_command(job, "collective", node), "-m",
            "frontier.profiling.collectives.vllm_collective_benchmark",
            "--config", job["benchmark"]]


def probe_command(job: dict, node: dict | None = None) -> list[str]:
    probe = job["native_probe"]
    engine = job["engine"]
    command = [*torchrun_command(job, "probe", node), "-m",
               "frontier.profiling.collectives.step5_dispatch_probe",
               "--model-path", probe["model_path"], "--output-dir",
               str(Path(job["output_dir"]) / "native_dispatch"),
               "--tensor-parallel-size", str(engine["tensor_parallel_size"]),
               "--data-parallel-size", str(engine["data_parallel_size"]),
                "--gpus-per-node", str(job["gpus_per_node"])]
    if "profile_manifest" in job:
        command += ["--routing-input-file",probe["routing_input_file"],"--moe-input-ids",*probe["moe_input_ids"]]
    else:
        command += ["--tokens-per-dp-rank",*map(str,probe["tokens_per_dp_rank"])]
    if engine["disable_custom_all_reduce"]:
        command.append("--disable-custom-all-reduce")
    return command


def verify_sources(job: dict) -> None:
    for relative, expected in job["source_hashes"].items():
        path = Path(job["repository"]) / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed after preparation: {relative}")
    if hashlib.sha256(Path(job["benchmark"]).read_bytes()).hexdigest() != job["benchmark_sha256"]:
        raise ValueError("The prepared benchmark configuration changed")
    metadata = Path(job["native_probe"]["model_path"]) / "config.json"
    if hashlib.sha256(metadata.read_bytes()).hexdigest() != job["native_probe"]["config_sha256"]:
        raise ValueError("The staged native metadata changed")


def write_json(path: Path, value: dict) -> None:
    """Publish a complete record before another node can observe it."""
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def wait_records(paths: list[Path], deadline: float) -> list[dict]:
    while True:
        if all(path.is_file() for path in paths):
            return [json.loads(path.read_text()) for path in paths]
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Waiting for node lifecycle records: {paths}")
        time.sleep(1)


def prepare_output(job: dict, node: dict, coordination: Path, deadline: float) -> Path:
    """Only node zero creates the common cloud directory and readiness record."""
    root = Path(job["output_dir"])
    ready = coordination / "ready.json"
    if node["node_rank"] == 0:
        coordination.mkdir(exist_ok=False)
        root.mkdir(parents=True, exist_ok=False)
        write_json(ready, {"run_id": job["run_id"], "output_dir": str(root)})
    else:
        record, = wait_records([ready], deadline)
        if record != {"run_id": job["run_id"], "output_dir": str(root)}:
            raise ValueError("Coordinator readiness does not match this prepared job")
    output = root if node["node_count"] == 1 else root / f"node_{node['node_rank']}"
    if output != root:
        output.mkdir(exist_ok=False)
    return output


def local_module(name: str):
    """Load auxiliary modules only after the prepared source hashes were checked."""
    spec = importlib.util.spec_from_file_location(f"step5_comm_{name}", Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def advisory(operation: str, path: Path, call) -> dict:
    """Never turn an auxiliary evidence fault into a native workload failure."""
    try:
        value = call()
        if not isinstance(value, dict):
            raise TypeError("Auxiliary evidence must be a JSON object")
        json.dumps(value)
    except Exception as error:
        value = {"status": "UNVERIFIED", "operation": operation, "error": repr(error)}
    print("STEP5_COMM_ADVISORY", operation,
          json.dumps({key: value[key] for key in ("status", "error") if key in value}), flush=True)
    try:
        write_json(path, value)
    except Exception as error:
        value = {"status": "UNVERIFIED", "operation": operation,
                  "error": repr(error), "evidence": value}
        print("STEP5_COMM_ADVISORY_WRITE_ERROR", json.dumps(value), flush=True)
    if PLATFORM_LOG_FD is not None and str(value.get("status", "PASS")).upper() not in (
            "PASS", "MATCH", "NOT_APPLICABLE"):
        notice = {"operation": operation, "status": value.get("status"), "error": value.get("error")}
        try:
            os.write(PLATFORM_LOG_FD, ("STEP5_COMM_UNVERIFIED " + json.dumps(notice) + "\n").encode())
        except OSError as error:
            print("STEP5_COMM_NOTIFICATION_ERROR", repr(error), flush=True)
    return value


def runtime_manifest(job: dict, node: dict) -> dict:
    import torch
    import vllm

    if torch.cuda.device_count() != node["gpus_per_node"]:
        raise ValueError("Visible GPU count differs from the prepared GPUs per node")
    manifest = {"status": "PASS", "job": job, "node": node, "host": socket.gethostname(), "python": sys.version,
                "torch": torch.__version__, "vllm": vllm.__version__,
                "torch_nccl_version": torch.cuda.nccl.version(),
                "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "platform_environment": {name: os.environ.get(name) for name in
                                          ("RDMA_NETWORK_LINK_TYPE", "CUDA_VISIBLE_DEVICES",
                                           "NODE_RANK", "NODE_COUNT", "PROC_PER_NODE",
                                           "MASTER_ADDR", "MASTER_PORT")},
                "cuda_driver_environment": {name: os.environ.get(name) for name in
                                            ("VLLM_ENABLE_CUDA_COMPATIBILITY", "LD_LIBRARY_PATH")},
                "nccl_environment": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}}
    hardware = job["resources"]["positive_tags"][0]
    manifest["hardware_matches"] = all(hardware in name.split() for name in manifest["devices"])
    if not manifest["hardware_matches"]:
        manifest["status"] = "UNVERIFIED"
    try:
        manifest["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        manifest["host_driver_versions"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()
    except Exception as error:
        manifest["status"] = "UNVERIFIED"
        manifest["nvidia_smi"] = {"status": "UNVERIFIED", "error": repr(error)}
        print("STEP5_COMM_ADVISORY", "nvidia_smi", repr(error), flush=True)
    return manifest


def run_phase(job: dict, node: dict, output: Path, coordination: Path,
              phase: str, command: list[str], deadline: float) -> None:
    print(phase.upper() + "_COMMAND", command, flush=True)
    result = subprocess.run(command)
    record = {"node_rank": node["node_rank"], "returncode": result.returncode}
    advisory(phase + "_exit", output / f"{phase}_exit.json", lambda: record)
    write_json(coordination / f"node_{node['node_rank']}_{phase}_exit.json", record)
    if result.returncode:
        raise SystemExit(result.returncode)
    paths = [coordination / f"node_{rank}_{phase}_exit.json" for rank in range(node["node_count"])]
    records = wait_records(paths, deadline)
    for rank, row in enumerate(records):
        if row["node_rank"] != rank:
            raise ValueError("Node exit record has the wrong rank")
        if row["returncode"]:
            raise SystemExit(row["returncode"])


def run_job(job: dict, node: dict, output: Path, coordination: Path, deadline: float) -> dict | None:
    root = Path(job["output_dir"])
    if node["node_rank"] == 0:
        advisory("job",root/"job.json",lambda:job)
    if "profile_manifest" in job:
        manifest_path=Path(job["profile_manifest"])
        os.environ.update(job["native_environment"])
        os.environ["FRONTIER_STEP5_PROFILE_MANIFEST"]=str(manifest_path)
        advisory("native_binding",output/"native_binding.json",lambda:verify_native_binding(job,manifest_path))
    def source_integrity():
        verify_sources(job)
        return {"status": "PASS"}

    advisory("source_integrity", output / "source_integrity.json", source_integrity)
    manifest = advisory("manifest", output / "manifest.json",
                        lambda: runtime_manifest(job, node))
    advisory("transport_snapshot", output / "transport_snapshot.json",
             lambda: local_module("transport").snapshot())
    print("STEP5_COMM_START", job["run_id"], manifest.get("host", socket.gethostname()), flush=True)
    run_phase(job, node, output, coordination, "collective", collective_command(job, node), deadline)
    print("STEP5_COMM_COLLECTIVES_DONE", job["run_id"], flush=True)
    run_phase(job, node, output, coordination, "native_probe", probe_command(job, node), deadline)
    print("STEP5_COMM_NODE_DONE", job["run_id"], node["node_rank"], flush=True)
    if node["node_rank"] != 0:
        return None
    if node["node_count"] > 1:
        manifest = {"job": job, "node": node, "host": socket.gethostname(), **manifest}
        manifest["node_manifests"] = [str(root / f"node_{rank}/manifest.json")
                                      for rank in range(node["node_count"])]
        advisory("manifest", root / "manifest.json", lambda: manifest)
    logs = [root / f"node_{rank}/worker.log" for rank in range(node["node_count"])]
    if node["node_count"] == 1:
        logs = [root / "worker.log"]
    transport = advisory("transport_logs", root / "transport.json",
                         lambda: local_module("transport").check_logs(logs, node["node_count"]))
    validation = advisory("validate_result", root / "validation.json",
                          lambda: local_module("validate_result").validate(job, root))
    report = advisory("native_dispatch_report", root / "native_dispatch_status.json",
                      lambda: json.loads((root / "native_dispatch/dispatch_probe.json").read_text()))
    verdict = {"execution_completed": True, "native_dispatch_status": report.get("status", "UNVERIFIED"),
               "target_topology_qualified": validation.get("target_topology_qualified", False),
               "cost_data_qualified": validation.get("cost_data_qualified", False),
               "expected": job["expected"], "validation": validation, "transport": transport}
    advisory("result", root / "result.json", lambda: verdict)
    print("STEP5_COMM_DONE", json.dumps(verdict), flush=True)
    return verdict


def verify_native_binding(job,manifest_path):
    """Check the sealed prepared binding as advisory evidence, not workload control."""
    manifest=json.loads(manifest_path.read_text());binding=manifest["native_binding"]
    sys.path.insert(0,str(manifest_path.parent/"tests/comparison/stage_admission_pp"))
    try:
        spec=importlib.util.spec_from_file_location("pricing_worker_registry",manifest_path.parent/"tests/comparison/dp_placement_pp/vllm_replay.py")
        registry=importlib.util.module_from_spec(spec);spec.loader.exec_module(registry)
    finally:
        sys.path.pop(0)
    reference=binding["native_reference"]
    registry.check_native_reference(reference,registry.adapter_record(binding["native_runtime_adapter"])["native_reference"])
    if job["image"] != reference["image"] or reference["routing_strategy"] != "position_cyclic":
        raise ValueError("Worker image/reference differs from prepared position collection")
    benchmark=json.loads(Path(job["benchmark"]).read_text())
    if benchmark["profile_manifest"] != str(manifest_path) or benchmark["native_environment"] != job["native_environment"]:
        raise ValueError("Worker and collector must retain one frozen binding/environment")
    return {"status":"PASS"}


def main() -> None:
    global PLATFORM_LOG_FD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.config.read_text())
    root = Path(job["output_dir"]).resolve()
    if not root.is_relative_to("/mnt/codesign-exp/ycfeng/frontier/step5_comm") or root.name != job["run_id"]:
        raise ValueError("Outputs must remain inside the personal communication run directory")
    node = node_identity(job)
    deadline = time.monotonic() + job["resources"]["timeout_seconds"]
    # Coordination is on the mounted local source, not high-frequency cloud polling.
    coordination = args.config.resolve().parent / "runtime"
    output = prepare_output(job, node, coordination, deadline)
    print("STEP5_COMM_WORKER_LOG", str(output / "worker.log"), flush=True)
    log = (output / "worker.log").open("w", buffering=1)
    PLATFORM_LOG_FD = os.dup(1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    scratch = Path(tempfile.mkdtemp(prefix="step5_comm_"))
    for name in ("HF_HOME", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
                 "FLASHINFER_WORKSPACE_BASE"):
        os.environ[name] = str(scratch / name.lower())
    run_job(job, node, output, coordination, deadline)


if __name__ == "__main__":
    main()
