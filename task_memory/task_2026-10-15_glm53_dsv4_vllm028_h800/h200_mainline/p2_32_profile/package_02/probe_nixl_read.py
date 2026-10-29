"""Reproduce the inventory's cross-node NIXL READ without preceding NCCL work.

The shared cloud filesystem carries only the start barrier and the result files.
Cross-node visibility of a new file there was observed to lag by up to 40
seconds, so the target hands its NIXL metadata to the initiator over a TCP side
channel, as vLLM's NixlConnector does.
"""

import argparse
import ctypes
import hashlib
import ipaddress
import json
import os
import socket
import struct
import sys
import time
import traceback
from pathlib import Path

from probe_runtime_inventory import (mapped_libraries, primary_ipv4, read_text, sha256,
                                     wait_for, write_json)

SYSFS_INFINIBAND = Path("/sys/class/infiniband")
FRAME = struct.Struct("!II")  # header length, metadata length


def gid_table(port):
    """Return the populated GID entries of one RDMA port as sysfs reports them."""
    entries = {}
    for path in sorted((port / "gids").glob("*"), key=lambda p: int(p.name)):
        gid = read_text(path)
        if gid is None or ipaddress.IPv6Address(gid) == ipaddress.IPv6Address(0):
            continue
        entries[path.name] = {"gid": gid,
                              "type": read_text(port / "gid_attrs" / "types" / path.name),
                              "ndev": read_text(port / "gid_attrs" / "ndevs" / path.name)}
    return entries


def roce_v2_ipv4_ports(rdma):
    """Active RoCE ports holding a RoCE v2 GID with an IPv4-mapped address.

    UCX assumes any two RoCE v1 addresses are reachable (ib_iface.c,
    uct_ib_iface_roce_is_reachable), so a port that only offers a link-local
    RoCE v1 GID can be paired with another such port on a different node.
    """
    return [f"{device}:{port}" for device, ports in sorted(rdma.items())
            for port, info in sorted(ports.items())
            if "ACTIVE" in (info["state"] or "") and info["link_layer"] == "Ethernet"
            and any(entry["type"] == "RoCE v2"
                    and ipaddress.IPv6Address(entry["gid"]).ipv4_mapped is not None
                    for entry in info["gids"].values())]


def recv_exact(conn, size):
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise ConnectionError(f"Side channel closed after {len(data)} of {size} bytes")
        data += chunk
    return bytes(data)


def serve_metadata(listener, header, metadata, deadline, peer_failed):
    """Send the target's buffer description and NIXL metadata to the initiator."""
    body = json.dumps(header).encode()
    listener.settimeout(1.0)
    while time.monotonic() < deadline:
        if peer_failed():
            raise RuntimeError("Initiator failed; see its result.json and worker.log")
        try:
            conn, _ = listener.accept()
        except TimeoutError:
            continue
        with conn:
            conn.sendall(FRAME.pack(len(body), len(metadata)) + body + metadata)
        return
    raise TimeoutError("Initiator did not connect to the side channel")


def fetch_metadata(address, deadline):
    """Receive the target's buffer description and NIXL metadata.

    The target listens before it writes ready.json, so the connection is queued
    until the target finishes registration and accepts it.
    """
    with socket.create_connection(address,
                                  timeout=max(deadline - time.monotonic(), 1.0)) as conn:
        header_bytes, metadata_bytes = FRAME.unpack(recv_exact(conn, FRAME.size))
        header = json.loads(recv_exact(conn, header_bytes))
        return header, recv_exact(conn, metadata_bytes)


def run(config):
    output = Path(config["output_dir"])
    if not output.is_relative_to(Path("/mnt/codesign-exp/ycfeng")):
        raise ValueError("Output must stay inside the personal cloud directory")
    host = socket.gethostname()
    node = output / "nodes" / host
    node.mkdir(parents=True, exist_ok=False)
    log = (node / "worker.log").open("w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    log.close()
    result = {"host": host, "started": time.time(), "passed": False, "stages": []}

    def stage(name):
        result["stages"].append({"name": name, "time": time.time()})
        write_json(node / "partial.json", result)
        print("NIXL_REPRO", name, flush=True)

    agent = registration = request = local_handle = remote_handle = listener = None
    peer = None
    try:
        stage("inventory")
        result["rdma"] = {
            d.name: {p.name: {**{key: read_text(p / key) for key in
                                 ("state", "phys_state", "rate", "link_layer")},
                              "gids": gid_table(p)}
                     for p in (d / "ports").glob("*")}
            for d in SYSFS_INFINIBAND.glob("*")}
        result["roce_v2_ipv4_ports"] = roce_v2_ipv4_ports(result["rdma"])
        if config.get("ucx_roce_v2_ipv4_ports_only", False):
            if not result["roce_v2_ipv4_ports"]:
                raise RuntimeError("No active port holds a RoCE v2 IPv4 GID")
            os.environ["UCX_NET_DEVICES"] = ",".join(result["roce_v2_ipv4_ports"])
        result["environment"] = {k: v for k, v in os.environ.items()
                                 if k.startswith(("UCX_", "PYTORCH_CUDA_ALLOC_CONF",
                                                  "PYTORCH_ALLOC_CONF"))}
        # Workers use the host network and may share a node with other jobs, so
        # each node takes a kernel-assigned port and publishes it before the
        # barrier; only the target keeps its listener.
        listener = socket.create_server(("", 0))
        own_channel = {"ip": primary_ipv4(), "port": listener.getsockname()[1]}
        write_json(node / "ready.json", {"host": host, **own_channel})
        if not wait_for(lambda: len(list(output.glob("nodes/*/ready.json")))
                        == config["replica"], config["peer_wait_seconds"]):
            raise TimeoutError("Both physical nodes did not arrive")
        peers = sorted(p.parent.name for p in output.glob("nodes/*/ready.json"))
        side = "initiator" if host == peers[0] else "target"
        result["side"] = side
        peer = output / "nodes" / peers[1 if side == "initiator" else 0]
        if side == "initiator":
            listener.close()
            listener = None
            target_channel = json.loads((peer / "ready.json").read_text())
        else:
            target_channel = own_channel
        result["side_channel"] = {"ip": target_channel["ip"], "port": target_channel["port"]}
        stage("agent_initialization")
        os.environ.setdefault("UCX_RCACHE_MAX_UNRELEASED", "1024")
        import torch
        from nixl._api import nixl_agent, nixl_agent_config

        device = config["device"]
        torch.cuda.set_device(device)
        result["gpu"] = {"count": torch.cuda.device_count(),
                         "name": torch.cuda.get_device_name(device)}
        agent = nixl_agent(f"t01-{side}-{host}", nixl_agent_config(
            num_threads=4, capture_telemetry=True))
        result["backend_params"] = agent.get_backend_params("UCX")
        result["libraries"] = {key: mapped_libraries(key) for key in
                               ("libucp", "libuct", "libcuda.so")}
        result["binary_sha256"] = {path: sha256(path) for path in sorted(set(
            mapped_libraries("nixl") + result["libraries"]["libucp"]
            + result["libraries"]["libuct"]))}
        if result["libraries"]["libucp"]:
            ucp = ctypes.CDLL(result["libraries"]["libucp"][0])
            ucp.ucp_get_version_string.restype = ctypes.c_char_p
            result["ucx_version"] = ucp.ucp_get_version_string().decode()
        nbytes = config["nixl_read_mib"] << 20
        pattern = (torch.arange(nbytes, device="cuda", dtype=torch.int64) % 251).to(torch.uint8)
        buffer = pattern if side == "target" else torch.zeros_like(pattern)
        torch.cuda.synchronize()
        result["buffer"] = {"address": buffer.data_ptr(), "bytes": nbytes,
                            "device": buffer.get_device()}
        stage("register_memory")
        registration = agent.get_reg_descs([(buffer.data_ptr(), nbytes, device, "")], "VRAM")
        agent.register_memory(registration, backends=["UCX"])
        metadata = agent.get_agent_metadata()
        result["metadata"] = {"bytes": len(metadata),
                              "sha256": hashlib.sha256(metadata).hexdigest()}
        (node / "agent.meta").write_bytes(metadata)
        header = {**result["buffer"], "metadata_bytes": len(metadata),
                  "sha256": result["metadata"]["sha256"]}
        write_json(node / "agent.json", header)
        stage("metadata_published")
        handshake_deadline = time.monotonic() + config["peer_wait_seconds"]

        def peer_failed():
            path = peer / "result.json"
            return path.exists() and not json.loads(path.read_text())["passed"]

        message = b"t01-read-done"
        if side == "target":
            serve_metadata(listener, header, metadata, handshake_deadline, peer_failed)
            stage("metadata_sent")
            deadline = time.monotonic() + config["transfer_timeout_seconds"]
            notified = False
            while not notified and time.monotonic() < deadline:
                notifications = agent.get_new_notifs()
                if notifications:
                    result["notifications"] = repr(notifications)
                notified = any(message in messages for messages in notifications.values())
                if peer_failed():
                    raise RuntimeError("Initiator failed; see its result.json and worker.log")
                time.sleep(0.05)
            result["notified"] = notified
            result["passed"] = notified
        else:
            target, remote_metadata = fetch_metadata(
                (target_channel["ip"], target_channel["port"]), handshake_deadline)
            result["remote_metadata"] = {"bytes": len(remote_metadata),
                                         "sha256": hashlib.sha256(remote_metadata).hexdigest(),
                                         "published_sha256": target["sha256"]}
            if result["remote_metadata"]["sha256"] != target["sha256"]:
                raise RuntimeError("Received metadata does not match the target's hash")
            stage("add_remote_agent")
            deadline = time.monotonic() + config["transfer_timeout_seconds"]
            remote_agent = agent.add_remote_agent(remote_metadata)
            result["remote_agent"] = remote_agent
            local_handle = agent.prep_xfer_dlist(
                "NIXL_INIT_AGENT", [(buffer.data_ptr(), nbytes, device)], "VRAM")
            remote_handle = agent.prep_xfer_dlist(
                remote_agent, [(target["address"], target["bytes"], target["device"])], "VRAM")
            request = agent.make_prepped_xfer("READ", local_handle, [0], remote_handle, [0],
                                              notif_msg=message)
            stage("transfer")
            result["transfer_initial_state"] = agent.transfer(request)
            state = result["transfer_initial_state"]
            stage("poll_transfer")
            while state == "PROC" and time.monotonic() < deadline:
                state = agent.check_xfer_state(request)
                if peer_failed():
                    raise RuntimeError("Target failed; see its result.json and worker.log")
            torch.cuda.synchronize()
            result["state"] = state
            result["data_match"] = bool(torch.equal(buffer, pattern))
            result["passed"] = state == "DONE" and result["data_match"]
    except Exception:
        result["exception"] = traceback.format_exc()
        print(result["exception"], flush=True)
    finally:
        result["finished"] = time.time()
        write_json(node / "result.json", result)
        if listener is not None:
            listener.close()
        for handle, method in ((request, "release_xfer_handle"),
                               (local_handle, "release_dlist_handle"),
                               (remote_handle, "release_dlist_handle"),
                               (registration, "deregister_memory")):
            if handle is not None:
                try:
                    getattr(agent, method)(handle)
                except Exception as error:
                    result.setdefault("cleanup_errors", []).append(repr(error))
        if result.get("cleanup_errors"):
            write_json(node / "result.json", result)
    wait_for(lambda: len(list(output.glob("nodes/*/result.json"))) == config["replica"],
             config["final_wait_seconds"], interval=0.2)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    sys.exit(run(json.loads(parser.parse_args().config.read_text())))
