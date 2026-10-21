"""Read RDMA inventory and distinguish actual NCCL channels from bootstrap."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


def verbs_devices() -> list[str]:
    library = ctypes.CDLL("libibverbs.so.1", use_errno=True)
    library.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
    library.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
    library.ibv_get_device_name.argtypes = [ctypes.c_void_p]
    library.ibv_get_device_name.restype = ctypes.c_char_p
    library.ibv_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    library.ibv_free_device_list.restype = None
    count = ctypes.c_int()
    devices = library.ibv_get_device_list(ctypes.byref(count))
    if not devices:
        raise OSError(ctypes.get_errno(), "ibv_get_device_list returned no device list")
    try:
        return [library.ibv_get_device_name(devices[i]).decode() for i in range(count.value)]
    finally:
        library.ibv_free_device_list(devices)


def snapshot(sysfs: Path = Path("/sys/class/infiniband"),
             device_nodes: Path = Path("/dev/infiniband")) -> dict:
    """Inventory is evidence, never a reason to skip the native workload."""
    result = {"advisory_only": True, "platform_link_type": os.environ.get("RDMA_NETWORK_LINK_TYPE"),
              "device_nodes": [], "verbs_devices": [], "ports": [], "errors": []}
    try:
        result["device_nodes"] = sorted(p.name for p in device_nodes.iterdir())
    except OSError as error:
        result["errors"].append(repr(error))
    try:
        result["verbs_devices"] = verbs_devices()
    except OSError as error:
        result["errors"].append(repr(error))
    try:
        for port in sorted(sysfs.glob("*/ports/*")):
            result["ports"].append({"device": port.parent.parent.name, "port": port.name,
                                    **{field: (port / field).read_text().strip()
                                       for field in ("state", "link_layer", "rate")}})
    except OSError as error:
        result["errors"].append(repr(error))
    result["status"] = ("PASS" if result["device_nodes"] and result["verbs_devices"]
                        and any("ACTIVE" in p["state"] for p in result["ports"])
                        and not result["errors"] else "UNVERIFIED")
    return result


def check_logs(log_paths: list[Path], expected_nodes: int) -> dict:
    """Require actual inter-node IB/GDRDMA channels, not an available plugin."""
    observations = []
    for path in log_paths:
        record = {"path": str(path), "rdma_channels": 0, "gdrdma_channels": 0,
                  "socket_channels": 0, "rdma_example": None, "socket_example": None}
        with path.open() as stream:
            for line in stream:
                channel = "Channel" in line and "via " in line
                if channel and "via NET/IB/" in line:
                    record["rdma_channels"] += 1
                    record["gdrdma_channels"] += int("GDRDMA" in line)
                    if record["rdma_example"] is None:
                        record["rdma_example"] = line.strip()
                if ((channel and "via NET/Socket" in line) or "Using network Socket" in line):
                    record["socket_channels"] += 1
                    if record["socket_example"] is None:
                        record["socket_example"] = line.strip()
        observations.append(record)
    status = "PASS" if len(observations) == expected_nodes and all(
        r["rdma_channels"] and r["gdrdma_channels"] and not r["socket_channels"]
        for r in observations) else "UNVERIFIED"
    if expected_nodes == 1:
        status = "NOT_APPLICABLE"
    return {"status": status, "advisory_only": True, "expected_nodes": expected_nodes,
            "observed_logs": len(observations), "nodes": observations,
            "scope": "NCCL channel evidence; not a per-call wire-traffic measurement"}
