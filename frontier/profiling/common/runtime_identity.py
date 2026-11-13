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
import importlib.metadata
import importlib.util
import inspect
import json
import os
import re
import sys
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


def step5_measurement_columns(native_facts: Mapping[str, Any] | None) -> dict[str, str]:
    """Return the measurement identity every Step compute row carries.

    The Step predictor publishes each role's models as one bundle with a single
    evidence kind and each producer's runtime stack. A bound run declares the
    native reference image its receipt verified; an unbound run declares the
    profiling job image.
    """
    torch_module = importlib.import_module("torch")
    from frontier.profiling.common.step5_job import IMAGE

    image = IMAGE if native_facts is None else native_facts["shared"]["native_reference"]["image"]
    version = importlib.metadata.version
    return {
        "evidence_kind": "native_measurement",
        "runtime_stack_signature": ";".join((
            f"declared_image={image}", f"vllm={version('vllm')}",
            f"torch={torch_module.__version__}", f"cuda={torch_module.version.cuda}",
            f"triton={version('triton')}", f"flashinfer={version('flashinfer-python')}",
        )),
    }


def verify_step5_profile_runtime(manifest_path, *, callables=()):
    """Verify installed Python origins against the sealed full-image receipt.

    The image digest is declared; only a separate platform receipt proves which
    image was launched. Compiled libraries retain their separate native receipt.
    """
    manifest_path = Path(manifest_path).resolve(strict=True)
    stage = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    binding = manifest["native_binding"]

    def sealed(relative):
        path = (stage / relative).resolve(strict=True)
        path.relative_to(stage)
        if file_sha256(path) != manifest["sha256"][relative]:
            raise ValueError(f"sealed profile artifact hash mismatch: {relative}")
        return path

    registry_path = stage / "tests/comparison/dp_placement_pp/vllm_replay.py"
    sealed("tests/comparison/dp_placement_pp/vllm_replay.py")
    sealed("tests/comparison/stage_admission_pp/vllm_burst_driver.py")
    sys.path.insert(0, str(stage / "tests/comparison/stage_admission_pp"))
    try:
        spec = importlib.util.spec_from_file_location("step5_profile_runtime_registry", registry_path)
        registry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(registry)
    finally:
        sys.path.pop(0)
    adapter = registry.adapter_record(binding["native_runtime_adapter"])
    reference = binding["native_reference"]
    registry.check_native_reference(reference, adapter["native_reference"])
    if reference["routing_strategy"] != "position_cyclic" or manifest["image"] != reference["image"]:
        raise ValueError("profile adapter/image/routing identity mismatch")
    for image in (reference["image"], reference["base_image"]):
        if re.fullmatch(r".+@sha256:[0-9a-f]{64}", image) is None:
            raise ValueError("profile native reference requires immutable image digests")
    if reference["image"] == reference["base_image"]:
        raise ValueError("position runtime requires a derived image, not the base image")
    if os.environ.get("VLLM_MOE_ROUTING_SIMULATION_STRATEGY") != reference["routing_strategy"] or os.environ.get("VLLM_MOE_SKIP_PADDING", "1") != "1":
        raise ValueError("effective native routing/mask environment differs from binding")

    receipt_path = sealed(binding["native_source_receipt"])
    derivation_path = sealed(binding["native_derivation_manifest"])
    receipt = json.loads(receipt_path.read_text())
    derivation = json.loads(derivation_path.read_text())
    root = Path(binding["installed_package"]).resolve(strict=True)
    if (receipt["schema_version"] != "native_installed_sources.v1"
            or receipt["image"] != reference["image"] or receipt["base_image"] != reference["base_image"]
            or receipt["installed_package"] != str(root)
            or receipt["python_version"] != sys.version.split()[0]
            or receipt["derivation_manifest_sha256"] != file_sha256(derivation_path)):
        raise ValueError("full installed-source receipt differs from native binding")
    pins = binding["installed_source_hashes"]
    if len(pins) != 53 or binding["serving_pin_count"] != 53:
        raise ValueError("serving pin inventory count differs from binding")
    for key, digest in pins.items():
        path = root.parent / key
        path.resolve(strict=True).relative_to(root)
        if receipt["sources"][key] != digest or file_sha256(path) != digest:
            raise ValueError(f"installed serving pin mismatch: {key}")
    if derivation["base_image"] != reference["base_image"]:
        raise ValueError("derivation base image differs from native reference")
    for key, digest in derivation["derived_source_sha256"].items():
        if pins[key] != digest or receipt["sources"][key] != digest:
            raise ValueError(f"derived native source mismatch: {key}")
    config_path = sealed("data/config/models/stepfun-ai__Step-5-Preview.json")
    if file_sha256(config_path) != binding["model_config_sha256"] or not binding["model_config_revision"]:
        raise ValueError("staged effective model configuration identity mismatch")

    imports = {}

    def verify_module(module):
        path = Path(module.__file__).resolve(strict=True)
        origin = Path(module.__spec__.origin).resolve(strict=True)
        relative = path.relative_to(root)
        key = "vllm/" + relative.as_posix()
        expected_relative = Path(*module.__name__.split(".")[1:])
        expected = root / (expected_relative / "__init__.py" if path.name == "__init__.py" else expected_relative.with_suffix(".py"))
        if path != expected or origin != path or file_sha256(path) != receipt["sources"][key]:
            raise ValueError(f"native module origin/spec/hash mismatch: {module.__name__}")
        fact = {"path": str(path), "spec_origin": str(origin), "sha256": receipt["sources"][key]}
        imports[module.__name__] = fact
        return fact

    for name in (
        "vllm", "vllm.models.step5.nvidia.model", "vllm.forward_context",
        "vllm.model_executor.layers.fused_moe.router.router_factory",
        "vllm.model_executor.layers.fused_moe.router.routing_simulator_router",
    ):
        verify_module(importlib.import_module(name))
    for name, module in tuple(sys.modules.items()):
        if (name == "vllm" or name.startswith("vllm.")) and str(getattr(module, "__file__", "")).endswith(".py"):
            verify_module(module)
    callable_facts = {}
    for supplied in callables:
        function = inspect.unwrap(getattr(supplied, "__func__", supplied))
        if not inspect.isclass(function) and not inspect.isfunction(function):
            function = inspect.unwrap(getattr(function, "fn", function))
        if not inspect.isclass(function) and not inspect.isfunction(function):
            raise ValueError("native callable has no inspectable Python source")
        module = sys.modules[function.__module__]
        fact = verify_module(module)
        source = Path(inspect.getsourcefile(function)).resolve(strict=True)
        code = getattr(function, "__code__", None)
        if source != Path(fact["path"]) or code is not None and Path(code.co_filename).resolve(strict=True) != source:
            raise ValueError("bound native callable code/source differs from its installed module")
        name = function.__module__ + "." + function.__qualname__
        callable_facts[name] = {**fact, "qualified_name": name, "code_filename": str(source)}
    shared = {key: binding[key] for key in (
        "native_runtime_adapter", "native_reference", "profile_generation", "installed_package",
        "model_config_revision", "model_config_sha256",
    )}
    shared.update(installed_source_hashes=pins, native_source_receipt_sha256=file_sha256(receipt_path),
                  native_derivation_manifest_sha256=file_sha256(derivation_path))
    return {"shared": shared, "imports": imports, "callables": callable_facts}


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
    native_facts: Mapping[str, Any] | None = None

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
        if self.native_facts is not None:
            payload["native_facts"] = self.native_facts["shared"]
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:16]

    def columns(self) -> dict[str, str]:
        columns = {
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
        if self.native_facts is not None:
            columns["native_facts"] = canonical_json(self.native_facts)
        return columns

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
    native_facts: Mapping[str, Any] | None = None,
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
        native_facts=native_facts,
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
    "step5_measurement_columns",
    "verify_step5_profile_runtime",
]
