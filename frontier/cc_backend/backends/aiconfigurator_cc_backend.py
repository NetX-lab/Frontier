"""Aiconfigurator CC backend implementation."""

from __future__ import annotations

import csv
import importlib
import importlib.util
import math
import re
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

from frontier.cc_backend.base_cc_backend import BaseCCBackend
from frontier.cc_backend.cc_backend_config import AiconfiguratorCCBackendConfig
from frontier.cc_backend.cc_backend_factory import CCBackendFactory
from frontier.config.config import AICONFIGURATOR_BACKEND_RELEASE_ERROR
from frontier.config.quantization_manager import get_quantization_manager
from frontier.logger import init_logger
from frontier.types import CCBackendType, ClusterType

logger = init_logger(__name__)


class AiconfiguratorCCBackend(BaseCCBackend):
    """CC backend backed by local aiconfigurator perf databases."""

    _ALLREDUCE_DOMAINS = frozenset(
        {"TP", "ATTN_TP", "MOE_TP", "DP", "ATTN_DP", "EP", "MOE_EP"}
    )
    _ALLGATHER_DOMAINS = frozenset(
        {"TP", "ATTN_TP", "MOE_TP", "DP", "ATTN_DP", "EP", "MOE_EP"}
    )
    _COLLECTIVE_DOMAINS = frozenset({"DP", "ATTN_DP", "EP", "MOE_EP"})
    _SEND_RECV_DOMAINS = frozenset({"PP"})
    _PRECISION_ALIASES = {
        "fp16": "half",
        "bf16": "half",
        "float16": "half",
        "half": "half",
        "int8": "int8",
        "fp8": "fp8",
    }
    _DATABASE_MODE_NAMES = {
        "silicon": "SILICON",
        "hybrid": "HYBRID",
        "empirical": "EMPIRICAL",
        "sol": "SOL",
        "sol_full": "SOL_FULL",
    }

    def __init__(
        self,
        config: AiconfiguratorCCBackendConfig,
        cluster_type: ClusterType,
        device_type: str,
        network_device: str,
        num_devices: int,
    ) -> None:
        raise ValueError(AICONFIGURATOR_BACKEND_RELEASE_ERROR)

        super().__init__(config, cluster_type, device_type, network_device, num_devices)

        self._config: AiconfiguratorCCBackendConfig = config
        self._repo_root = self._resolve_repo_root(config.repo_root)
        self._source_root = self._repo_root / "src"
        self._package_root = self._source_root / "aiconfigurator"
        self._systems_dir = self._package_root / "systems"
        self._pyproject_path = self._repo_root / "pyproject.toml"
        self._validate_repo_layout()

        self._bootstrap_source_tree_package()
        self._common_module = importlib.import_module("aiconfigurator.sdk.common")
        self._perf_database_module = importlib.import_module(
            "aiconfigurator.sdk.perf_database"
        )

        self._database_mode = self._resolve_database_mode(config.database_mode)
        self._validate_source_tuple(
            system=config.system,
            source_backend=config.source_backend,
            source_version=config.source_version,
        )
        self._database = self._perf_database_module.PerfDatabase(
            system=config.system,
            backend=config.source_backend,
            version=config.source_version,
            systems_dir=str(self._systems_dir),
        )
        self._custom_allreduce_table = self._load_custom_allreduce_table()

    def _resolve_repo_root(self, repo_root: str) -> Path:
        path = Path(str(repo_root)).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    def _validate_repo_layout(self) -> None:
        required_paths = (
            self._repo_root,
            self._pyproject_path,
            self._package_root,
            self._systems_dir,
        )
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Local aiconfigurator repository layout is incomplete. Missing paths: "
                f"{missing}"
            )

    def _bootstrap_source_tree_package(self) -> None:
        source_root_str = str(self._source_root)
        if source_root_str not in sys.path:
            sys.path.insert(0, source_root_str)

        package = types.ModuleType("aiconfigurator")
        package.__file__ = str(self._package_root / "__init__.py")
        package.__path__ = [str(self._package_root)]
        package.__package__ = "aiconfigurator"
        package.__spec__ = importlib.util.spec_from_file_location(
            "aiconfigurator",
            self._package_root / "__init__.py",
            submodule_search_locations=[str(self._package_root)],
        )
        package.__version__ = self._read_source_tree_version()
        sys.modules["aiconfigurator"] = package

    def _read_source_tree_version(self) -> str:
        content = self._pyproject_path.read_text(encoding="utf-8")
        match = re.search(r'^\s*version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        if match is None:
            raise ValueError(
                f"Unable to parse aiconfigurator version from {self._pyproject_path}"
            )
        return match.group(1)

    def _resolve_database_mode(self, database_mode: str):
        normalized_mode = str(database_mode).strip().lower()
        if normalized_mode not in self._DATABASE_MODE_NAMES:
            raise ValueError(
                f"Unsupported aiconfigurator database_mode={database_mode!r}. "
                f"Supported modes: {sorted(self._DATABASE_MODE_NAMES)}"
            )
        mode_name = self._DATABASE_MODE_NAMES[normalized_mode]
        return getattr(self._common_module.DatabaseMode, mode_name)

    def _validate_source_tuple(
        self,
        *,
        system: str,
        source_backend: str,
        source_version: str,
    ) -> None:
        if not str(source_version).strip():
            raise ValueError("aiconfigurator source_version must be non-empty")

        supported = self._perf_database_module.get_supported_databases(
            systems_dir=str(self._systems_dir)
        )
        available_versions = supported.get(system, {}).get(source_backend, [])
        if source_version not in available_versions:
            raise ValueError(
                "Unsupported aiconfigurator database tuple: "
                f"system={system!r}, source_backend={source_backend!r}, "
                f"source_version={source_version!r}. "
                f"Available versions: {available_versions}"
            )

    def _load_custom_allreduce_table(self):
        custom_allreduce_path = (
            self._systems_dir
            / self._database.system_spec["data_dir"]
            / self._config.source_backend
            / self._config.source_version
            / self._common_module.PerfDataFilename.custom_allreduce.value
        )
        if not custom_allreduce_path.exists():
            raise FileNotFoundError(
                f"Custom allreduce data file not found: {custom_allreduce_path}"
            )

        rows = []
        with custom_allreduce_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = (
                    row.get("backend")
                    or row.get("implementation")
                    or row.get("kernel_source")
                    or ""
                ).strip()
                rows.append((label, row))

        labels = sorted({label for label, _ in rows if label})
        if len(labels) > 1 and not self._config.custom_allreduce_variant:
            raise ValueError(
                "custom_allreduce_variant must be set explicitly because the raw "
                f"custom allreduce file contains multiple variants: {labels}"
            )
        if self._config.custom_allreduce_variant is not None:
            if self._config.custom_allreduce_variant not in labels:
                raise ValueError(
                    "Requested custom_allreduce_variant was not found in raw data: "
                    f"{self._config.custom_allreduce_variant!r}. Available variants: {labels}"
                )
            selected_label = self._config.custom_allreduce_variant
        elif len(labels) == 1:
            selected_label = labels[0]
        else:
            selected_label = ""

        filtered_rows = [
            row for label, row in rows if not selected_label or label == selected_label
        ]
        if not filtered_rows:
            raise ValueError(
                "No custom allreduce rows matched the selected runtime label "
                f"{selected_label!r} in {custom_allreduce_path}"
            )

        table = defaultdict(lambda: defaultdict(dict))
        for row in filtered_rows:
            quant_mode = self._parse_comm_quant_mode(row["allreduce_dtype"])
            tp_size = int(row["num_gpus"])
            message_size = int(row["message_size"])
            latency = float(row["latency"])
            table[quant_mode][tp_size][message_size] = latency

        return table

    def _parse_comm_quant_mode(self, raw_value: str):
        normalized = str(raw_value).strip().lower()
        if normalized not in self._PRECISION_ALIASES:
            raise ValueError(
                f"Unsupported aiconfigurator communication dtype {raw_value!r} in raw data"
            )
        alias = self._PRECISION_ALIASES[normalized]
        return getattr(self._common_module.CommQuantMode, alias)

    def _normalize_comm_domain(
        self,
        comm_domain: Optional[str],
        *,
        allowed_domains,
        operation: str,
    ) -> str:
        if not comm_domain:
            raise ValueError(
                f"aiconfigurator backend requires explicit comm_domain for {operation}"
            )
        normalized = str(comm_domain).strip().upper()
        if normalized not in allowed_domains:
            raise ValueError(
                f"Unsupported comm_domain={comm_domain!r} for {operation}. "
                f"Supported domains: {sorted(allowed_domains)}"
            )
        return normalized

    def _get_quant_mode(self, operation_name: str, cluster_type: Optional[ClusterType]):
        precision = get_quantization_manager().get_precision(operation_name, cluster_type)
        normalized = self._PRECISION_ALIASES.get(precision.value)
        if normalized is None:
            raise ValueError(
                f"Unsupported communication precision {precision.value!r} for "
                f"operation={operation_name!r} in aiconfigurator backend"
            )
        return getattr(self._common_module.CommQuantMode, normalized)

    def _get_bytes_per_element(
        self, operation_name: str, cluster_type: Optional[ClusterType]
    ) -> float:
        return get_quantization_manager().get_bytes_per_element(
            operation_name, cluster_type
        )

    def _bytes_to_elements(
        self,
        data_size_bytes: int,
        *,
        operation_name: str,
        cluster_type: Optional[ClusterType],
    ) -> int:
        bytes_per_element = self._get_bytes_per_element(operation_name, cluster_type)
        if bytes_per_element <= 0:
            raise ValueError(
                f"Invalid bytes_per_element={bytes_per_element} for operation {operation_name}"
            )
        return int(math.ceil(data_size_bytes / bytes_per_element))

    def _query_custom_allreduce(
        self,
        *,
        quant_mode,
        tp_size: int,
        message_elements: int,
    ) -> float:
        if tp_size <= 1:
            return 0.0

        tp_tables = self._custom_allreduce_table.get(quant_mode)
        if not tp_tables:
            raise ValueError(
                f"No custom allreduce data for quant_mode={quant_mode.name!r}"
            )

        max_tp = max(tp_tables.keys())
        base_tp = min(tp_size, max_tp)
        size_table = tp_tables[base_tp]
        size_keys = sorted(size_table.keys())
        if len(size_keys) < 2:
            raise ValueError(
                f"Custom allreduce table for tp_size={base_tp} is too small for interpolation"
            )
        size_left, size_right = self._database._nearest_1d_point_helper(
            message_elements,
            size_keys,
            inner_only=False,
        )
        latency = float(
            self._database._interp_1d(
                [size_left, size_right],
                [size_table[size_left], size_table[size_right]],
                message_elements,
            )
        )
        if tp_size > max_tp:
            if tp_size > self._database.system_spec["node"]["num_gpus_per_node"]:
                scale_factor = (
                    (tp_size - 1)
                    / tp_size
                    * max_tp
                    / (max_tp - 1)
                    * self._database.system_spec["node"]["intra_node_bw"]
                    / self._database.system_spec["node"]["inter_node_bw"]
                )
            else:
                scale_factor = (tp_size - 1) / tp_size * max_tp / (max_tp - 1)
            latency *= scale_factor
        return latency

    def predict_allreduce(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "allreduce")
        normalized_domain = self._normalize_comm_domain(
            comm_domain,
            allowed_domains=self._ALLREDUCE_DOMAINS,
            operation="allreduce",
        )
        if num_devices <= 1:
            return 0.0

        quant_mode = self._get_quant_mode("allreduce", cluster_type)
        message_elements = self._bytes_to_elements(
            data_size_bytes,
            operation_name="allreduce",
            cluster_type=cluster_type,
        )
        if normalized_domain in {"TP", "ATTN_TP", "MOE_TP"}:
            if self._config.tp_allreduce_impl == "custom_allreduce":
                return self._query_custom_allreduce(
                    quant_mode=quant_mode,
                    tp_size=num_devices,
                    message_elements=message_elements,
                )
            if self._config.tp_allreduce_impl != "nccl_all_reduce":
                raise ValueError(
                    f"Unsupported tp_allreduce_impl={self._config.tp_allreduce_impl!r}"
                )

        result = self._database.query_nccl(
            quant_mode,
            num_devices,
            "all_reduce",
            message_elements,
            database_mode=self._database_mode,
        )
        return float(result)

    def predict_allgather(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "allgather")
        self._normalize_comm_domain(
            comm_domain,
            allowed_domains=self._ALLGATHER_DOMAINS,
            operation="allgather",
        )
        if num_devices <= 1:
            return 0.0

        quant_mode = self._get_quant_mode("allgather", cluster_type)
        message_elements = self._bytes_to_elements(
            data_size_bytes,
            operation_name="allgather",
            cluster_type=cluster_type,
        )
        result = self._database.query_nccl(
            quant_mode,
            num_devices,
            "all_gather",
            message_elements,
            database_mode=self._database_mode,
        )
        return float(result)

    def predict_broadcast(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        raise NotImplementedError(
            "aiconfigurator backend does not support broadcast prediction"
        )

    def predict_send_recv(
        self,
        data_size_bytes: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        self._validate_data_size(data_size_bytes)
        self._normalize_comm_domain(
            comm_domain,
            allowed_domains=self._SEND_RECV_DOMAINS,
            operation="send_recv",
        )
        result = self._database.query_p2p(
            data_size_bytes,
            database_mode=self._database_mode,
        )
        return float(result)

    def predict_reduce_scatter(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "reduce_scatter")
        self._normalize_comm_domain(
            comm_domain,
            allowed_domains=self._COLLECTIVE_DOMAINS,
            operation="reduce_scatter",
        )
        if num_devices <= 1:
            return 0.0

        quant_mode = self._get_quant_mode("reduce_scatter", cluster_type)
        message_elements = self._bytes_to_elements(
            data_size_bytes,
            operation_name="reduce_scatter",
            cluster_type=cluster_type,
        )
        result = self._database.query_nccl(
            quant_mode,
            num_devices,
            "reduce_scatter",
            message_elements,
            database_mode=self._database_mode,
        )
        return float(result)

    def predict_all_to_all(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "all_to_all")
        self._normalize_comm_domain(
            comm_domain,
            allowed_domains=self._COLLECTIVE_DOMAINS,
            operation="all_to_all",
        )
        if num_devices <= 1:
            return 0.0

        quant_mode = self._get_quant_mode("all_to_all", cluster_type)
        message_elements = self._bytes_to_elements(
            data_size_bytes,
            operation_name="all_to_all",
            cluster_type=cluster_type,
        )
        result = self._database.query_nccl(
            quant_mode,
            num_devices,
            "alltoall",
            message_elements,
            database_mode=self._database_mode,
        )
        return float(result)


CCBackendFactory.register(CCBackendType.AICONFIGURATOR, AiconfiguratorCCBackend)
