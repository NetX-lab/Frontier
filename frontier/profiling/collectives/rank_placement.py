"""Map requested node-local GPU indices to CUDA_VISIBLE_DEVICES entries (CPU safe)."""
from __future__ import annotations


def select_visible_devices(visible: str | None, indices: list[int], allocated: int) -> str:
    devices = visible.split(",") if visible else [str(i) for i in range(allocated)]
    if len(devices) != allocated or any(not d.strip() for d in devices):
        raise ValueError(f"Expected {allocated} allocated CUDA devices, got {visible!r}")
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("GPU placement must contain distinct indices")
    if any(type(i) is not int or i < 0 or i >= allocated for i in indices):
        raise ValueError(f"GPU placement {indices} outside allocation of {allocated}")
    return ",".join(devices[i].strip() for i in indices)


def select_physical_devices(allocated_uuids: list[str], indices: list[int],
                            inventory_lines: list[str], allocated: int) -> str:
    """Resolve NVML physical indices, verifying actual CUDA allocation UUIDs."""
    allocated_uuids = ["GPU-" + uuid.removeprefix("GPU-") for uuid in allocated_uuids]
    if len(allocated_uuids) != allocated or len(set(allocated_uuids)) != allocated:
        raise ValueError("CUDA allocation UUID census does not match requested allocation")
    inventory = {}
    for line in inventory_lines:
        fields = [v.strip() for v in line.split(",")]
        if len(fields) < 2 or not fields[1].startswith("GPU-"):
            raise ValueError("Cannot establish physical GPU index/UUID mapping")
        index = int(fields[0])
        if index in inventory:
            raise ValueError("Duplicate physical GPU index")
        inventory[index] = fields[1]
    if not indices or len(set(indices)) != len(indices) or any(type(i) is not int for i in indices):
        raise ValueError("Physical placement needs distinct integer indices")
    selected = [inventory.get(i) for i in indices]
    if any(uuid not in allocated_uuids for uuid in selected):
        raise ValueError(f"Physical placement {indices} is unavailable in CUDA allocation")
    return ",".join(selected)
