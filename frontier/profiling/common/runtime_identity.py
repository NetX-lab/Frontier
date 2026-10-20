"""Runtime identity columns carried by every measured profiling table.

Each measured row records which software stack, model configuration, device,
parallel layout, and execution mode produced it, so a consumer can reject a
table measured under a different runtime instead of silently mixing timings.

The identity has two parts:

* ``RuntimeIdentity`` holds the table-wide facts. ``collect_runtime_versions``
  reads the installed package versions on the measuring worker.
* Per-row facts that vary inside one table, such as the physical scope of a
  measured operator or the kernel class that executed it, stay ordinary row
  columns written by each producer. The compute dtype stays in the existing
  ``profiling_precision`` column.

Weights in these producers are always randomly initialized device tensors, so
``weight_source`` is fixed to ``dummy``; a producer that loads checkpoint
weights must not use this module unchanged.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

WEIGHT_SOURCE_DUMMY = "dummy"

# Distributions whose versions decide kernel selection on the worker.
RUNTIME_DISTRIBUTIONS: tuple[str, ...] = (
    "vllm",
    "torch",
    "flashinfer",
    "deep_gemm",
    "triton",
)

IDENTITY_COLUMNS: tuple[str, ...] = (
    "runtime_fingerprint",
    "runtime_versions",
    "source_revision",
    "model_config_revision",
    "model_config_sha256",
    "weight_source",
    "device_name",
    "device_capability",
    "schema_version",
    "quant_signature",
    "parallel_groups",
    "graph_mode",
)


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unversioned"))


def collect_runtime_versions(
    distributions: Iterable[str] = RUNTIME_DISTRIBUTIONS,
) -> dict[str, str | None]:
    """Return installed versions; an absent package is recorded as ``None``."""

    versions: dict[str, str | None] = {
        name: _module_version(name) for name in distributions
    }
    torch_module = importlib.import_module("torch")
    versions["torch_cuda"] = torch_module.version.cuda
    return versions


def collect_device_identity() -> tuple[str, str]:
    """Return the current CUDA device name and ``sm<major><minor>`` capability."""

    torch_module = importlib.import_module("torch")
    device = torch_module.cuda.current_device()
    major, minor = torch_module.cuda.get_device_capability(device)
    return torch_module.cuda.get_device_name(device), f"sm{major}{minor}"


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class RuntimeIdentity:
    """Table-wide runtime facts written as identical columns on every row."""

    runtime_versions: Mapping[str, str | None]
    source_revision: str
    model_config_revision: str
    model_config_sha256: str
    device_name: str
    device_capability: str
    schema_version: str
    quant_signature: str
    parallel_groups: Mapping[str, int]
    graph_mode: str

    def __post_init__(self) -> None:
        for name in (
            "source_revision",
            "model_config_revision",
            "model_config_sha256",
            "device_name",
            "device_capability",
            "schema_version",
            "quant_signature",
            "graph_mode",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"RuntimeIdentity.{name} must be a non-empty string")
        for group, size in self.parallel_groups.items():
            if type(size) is not int or size <= 0:
                raise ValueError(
                    f"parallel group {group!r} must have a positive int size, got {size!r}"
                )

    @property
    def fingerprint(self) -> str:
        """Hash of the facts that select kernels on the measuring worker."""

        payload = {
            "runtime_versions": dict(self.runtime_versions),
            "source_revision": self.source_revision,
            "device_name": self.device_name,
            "device_capability": self.device_capability,
            "graph_mode": self.graph_mode,
        }
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:16]

    def columns(self) -> dict[str, str]:
        return {
            "runtime_fingerprint": self.fingerprint,
            "runtime_versions": canonical_json(dict(self.runtime_versions)),
            "source_revision": self.source_revision,
            "model_config_revision": self.model_config_revision,
            "model_config_sha256": self.model_config_sha256,
            "weight_source": WEIGHT_SOURCE_DUMMY,
            "device_name": self.device_name,
            "device_capability": self.device_capability,
            "schema_version": self.schema_version,
            "quant_signature": self.quant_signature,
            "parallel_groups": canonical_json(dict(self.parallel_groups)),
            "graph_mode": self.graph_mode,
        }

    def to_json(self) -> dict[str, Any]:
        return {**self.columns(), "parallel_groups": dict(self.parallel_groups)}


def model_config_json_path(model_name: str) -> Path:
    """Return the checked-in JSON config ``ModelConfig.from_model_name`` loads."""

    path = Path("data", "config", "models", f"{model_name.replace('/', '__')}.json")
    if not path.is_file():
        raise ValueError(
            f"runtime identity needs the JSON model config of {model_name!r} at {path}"
        )
    return path


def measure_runtime_identity(
    *,
    model_name: str,
    source_revision: str,
    model_config_revision: str,
    schema_version: str,
    quant_signature: str,
    parallel_groups: Mapping[str, int],
    graph_mode: str,
) -> RuntimeIdentity:
    """Collect the identity of a table measured on the current CUDA device."""

    device_name, device_capability = collect_device_identity()
    return RuntimeIdentity(
        runtime_versions=collect_runtime_versions(),
        source_revision=source_revision,
        model_config_revision=model_config_revision,
        model_config_sha256=file_sha256(model_config_json_path(model_name)),
        device_name=device_name,
        device_capability=device_capability,
        schema_version=schema_version,
        quant_signature=quant_signature,
        parallel_groups=parallel_groups,
        graph_mode=graph_mode,
    )


def attach_identity_columns(
    rows: Iterable[dict[str, Any]], identity: RuntimeIdentity
) -> list[dict[str, Any]]:
    """Add identity columns to each row; a conflicting existing value is an error."""

    columns = identity.columns()
    attached: list[dict[str, Any]] = []
    for row in rows:
        for name, value in columns.items():
            existing = row.get(name)
            if existing not in (None, "") and str(existing) != value:
                raise ValueError(
                    f"Row already carries {name}={existing!r}; identity requires {value!r}"
                )
        attached.append({**row, **columns})
    return attached


__all__ = [
    "IDENTITY_COLUMNS",
    "RUNTIME_DISTRIBUTIONS",
    "RuntimeIdentity",
    "WEIGHT_SOURCE_DUMMY",
    "attach_identity_columns",
    "canonical_json",
    "collect_device_identity",
    "collect_runtime_versions",
    "file_sha256",
    "measure_runtime_identity",
    "model_config_json_path",
]
