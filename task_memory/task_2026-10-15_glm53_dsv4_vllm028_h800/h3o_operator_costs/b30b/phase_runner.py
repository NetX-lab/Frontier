"""Run sequential B30b phases using the existing K5 rendezvous/runtime helpers.

Only invoked by the lead's approved GPU job. A failed child is recorded and
later phases continue after both nodes acknowledge the result. Hash checks
are soft; actual placement and rendezvous failures remain visible failures.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import signal
import subprocess
import sys
import time
from pathlib import Path

from frontier.profiling.collectives.rank_placement import select_visible_devices, select_physical_devices


def load_k5(config):
    path = Path(config["workspace"]) / config["k5_node_entry"]
    spec = importlib.util.spec_from_file_location("b30b_k5_node", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def phase_command(config, phase, peers, node_rank, phase_path):
    distributed = not phase.get("independent_nodes", False)
    count = len(peers) if distributed else 1
    rank = node_rank if distributed else 0
    master = peers[0] if distributed else peers[node_rank]
    return [sys.executable, "-m", "torch.distributed.run", "--nnodes", str(count),
            "--nproc-per-node", str(len(phase["gpu_indices"][node_rank])),
            "--node-rank", str(rank), "--master-addr", master["ip"],
            "--master-port", str(master["port"]), "-m", phase["benchmark_module"],
            "--config", str(phase_path)]


def await_phase(output, name, peers, seconds):
    deadline = time.monotonic() + seconds
    while True:
        paths = [output / "nodes" / p["host"] / f"{name}.json" for p in peers]
        if all(p.exists() for p in paths):
            return [json.loads(p.read_text()) for p in paths]
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Peer phase barrier expired: {name}")
        time.sleep(1)


def run_child(command, env, log, seconds):
    """Kill the entire torchrun process group before the next phase on timeout."""
    with subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                          start_new_session=True) as child:
        try:
            return child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            # A leader may exit before its descendants. Always kill remaining
            # members of the session's process group before releasing the phase.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)
            raise


def run_phases(config, output, node, peers, node_rank, helpers,
               allocated_uuids=None, inventory_lines=None):
    results = []
    for phase in config["phases"]:
        result = {"phase": phase["name"], "started": time.time(), "returncode": None}
        try:
            env = dict(os.environ)
            if phase.get("physical_gpu_indices", True):
                env["CUDA_VISIBLE_DEVICES"] = select_physical_devices(
                    allocated_uuids or [], phase["gpu_indices"][node_rank],
                    inventory_lines or [], config["gpu"])
            else:
                env["CUDA_VISIBLE_DEVICES"] = select_visible_devices(
                    env.get("CUDA_VISIBLE_DEVICES"), phase["gpu_indices"][node_rank], config["gpu"])
            ip = peers[node_rank]["ip"]
            env.setdefault("GLOO_SOCKET_IFNAME", helpers.interface_of(ip))
            if phase.get("set_ucx_net_devices"):
                rdma = {device.name: {port.name: {
                    "state": helpers.read_text(port / "state"),
                    "link_layer": helpers.read_text(port / "link_layer"),
                    "gids": helpers.gid_table(port)}
                    for port in (device / "ports").glob("*")}
                    for device in helpers.SYSFS_INFINIBAND.glob("*")}
                devices = helpers.roce_v2_ipv4_ports(rdma)
                if not devices:
                    raise RuntimeError("No ACTIVE RoCE v2 IPv4 port for UCX")
                env["UCX_NET_DEVICES"] = ",".join(devices)
            leaf = {**phase["config"], "benchmark_module": phase["benchmark_module"]}
            suffix = f"/node{node_rank}" if phase.get("independent_nodes") else ""
            leaf["output_dir"] = str(output / phase["name"]) + suffix
            path = node / f"{phase['name']}_config.json"
            helpers.write_json(path, leaf)
            command = phase_command(config, phase, peers, node_rank, path)
            result.update(command=command, visible_devices=env["CUDA_VISIBLE_DEVICES"],
                          ucx_net_devices=env.get("UCX_NET_DEVICES"), node_rank=node_rank)
            helpers.write_json(node / f"{phase['name']}_launch.json", result)
            # Make preparation errors visible to peers before either torchrun starts.
            prepared = {"ok": True}
        except Exception as error:
            prepared = {"ok": False, "error": repr(error)}
            result["exception"] = repr(error)
        helpers.write_json(node / f"{phase['name']}_prepared.json", prepared)
        prepared_rows = await_phase(output, phase["name"] + "_prepared", peers,
                                    config["peer_wait_seconds"])
        can_run = (prepared["ok"] if phase.get("independent_nodes")
                   else all(row["ok"] for row in prepared_rows))
        if can_run:
            try:
                with (node / f"{phase['name']}.log").open("w") as log:
                    result["returncode"] = run_child(command, env, log, phase["timeout_seconds"])
            except Exception as error:
                result["exception"] = repr(error)
        else:
            result["skipped"] = prepared_rows
        result["finished"] = time.time()
        helpers.write_json(node / f"{phase['name']}_result.json", result)
        await_phase(output, phase["name"] + "_result", peers,
                    phase["timeout_seconds"] + config["final_wait_seconds"])
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    helpers = load_k5(config)
    output = Path(config["output_dir"])
    if not output.resolve().is_relative_to(helpers.PERSONAL_ROOT):
        raise ValueError("Output must stay inside the personal cloud directory")
    host = socket.gethostname()
    node = output / "nodes" / host
    node.mkdir(parents=True, exist_ok=False)
    with (node / "node.log").open("w", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            helpers.write_json(node / "ready.json", dict(
                host=host, ip=helpers.primary_ipv4(), port=helpers.free_port()))
            peers = helpers.await_nodes(output, "ready.json", config["replica"],
                                        config["peer_wait_seconds"])
            node_rank = [p["host"] for p in peers].index(host)
            inventory = subprocess.run([
                "nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30)
            if inventory.returncode:
                raise RuntimeError("GPU inventory failed")
            # Enumerate UUIDs using the inherited CUDA visibility/order, so
            # reordered numeric entries and GPU UUID allocations are verified.
            census = subprocess.run([sys.executable, "-c",
                "import json,torch; print(json.dumps([str(torch.cuda.get_device_properties(i).uuid) "
                "for i in range(torch.cuda.device_count())]))"],
                capture_output=True, text=True, timeout=60)
            try:
                allocated_uuids = json.loads(census.stdout) if census.returncode == 0 else []
            except ValueError:
                allocated_uuids = []
            helpers.write_json(node / "placement.json", dict(
                allocated_uuids=allocated_uuids, census_returncode=census.returncode,
                census_error=census.stderr,
                node_rank=node_rank, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                inventory=inventory.stdout.splitlines(),
                topology=subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                                        text=True, timeout=30).stdout))
            results = run_phases(config, output, node, peers, node_rank, helpers,
                                 allocated_uuids, inventory.stdout.splitlines())
            failed = any(r["returncode"] != 0 for r in results)
            helpers.write_json(node / "result.json", {"phases": results, "failed": failed})
        except Exception as error:
            helpers.write_json(node / "failed.json", {"error": repr(error)})
            raise
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
