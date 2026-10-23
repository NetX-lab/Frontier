"""vLLM backend for CPU overhead profiling (replay mode)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from frontier.profiling.cpu_overhead.backends.base_backend import (
    BaseCpuOverheadProfilerBackend,
)
from frontier.profiling.cpu_overhead.backends.vllm_mapping import (
    normalize_vllm_cpu_overhead_record,
)
from frontier.profiling.cpu_overhead.schema import (
    DEFAULT_NUM_DECODE_TOKENS_AMPLIFICATION_FACTOR,
    DEFAULT_NUM_PREFILL_TOKENS,
    DEFAULT_SCHEDULING_MODE,
)
from frontier.profiling.cpu_overhead.validation import validate_cpu_overhead_dataframe


@dataclass
class _VllmReplayRunner:
    normalized_row: dict

    def run(self) -> dict:
        return dict(self.normalized_row)


class VllmCpuOverheadProfilerBackend(BaseCpuOverheadProfilerBackend):
    """Replay backend using pre-collected vLLM CPU timing records."""

    def __init__(self, vllm_cpu_overhead_input_file: str | None = None) -> None:
        self._vllm_cpu_overhead_input_file = vllm_cpu_overhead_input_file
        self._normalized_records: list[dict] = []

    @property
    def name(self) -> str:
        return "vllm"

    def start(self) -> None:
        if not self._vllm_cpu_overhead_input_file:
            raise RuntimeError(
                "CPU overhead backend 'vllm' requires --vllm_cpu_overhead_input_file."
            )
        if not os.path.exists(self._vllm_cpu_overhead_input_file):
            raise RuntimeError(
                "vLLM CPU overhead input file does not exist: "
                f"{self._vllm_cpu_overhead_input_file}"
            )

        raw_records = self._load_raw_records(self._vllm_cpu_overhead_input_file)
        if not raw_records:
            raise RuntimeError(
                "vLLM CPU overhead input file is empty: "
                f"{self._vllm_cpu_overhead_input_file}"
            )

        self._normalized_records = [
            normalize_vllm_cpu_overhead_record(record) for record in raw_records
        ]
        validate_cpu_overhead_dataframe(
            df=self._to_dataframe(self._normalized_records)
        )

    def stop(self) -> None:
        self._normalized_records = []

    @staticmethod
    def _to_dataframe(records: list[dict]):
        import pandas as pd  # pylint: disable=import-outside-toplevel

        return pd.DataFrame(records)

    @staticmethod
    def _load_raw_records(input_file: str) -> list[dict]:
        if input_file.endswith(".jsonl"):
            records: list[dict] = []
            with open(input_file, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
            return records

        if input_file.endswith(".json"):
            with open(input_file, "r", encoding="utf-8") as file:
                payload = json.load(file)
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                if "records" not in payload or not isinstance(payload["records"], list):
                    raise ValueError(
                        "vLLM CPU overhead JSON payload must be a list or contain "
                        "'records' list."
                    )
                return payload["records"]
            raise ValueError(
                "Unsupported vLLM CPU overhead JSON payload type: "
                f"{type(payload).__name__}"
            )

        raise ValueError(
            "Unsupported vLLM CPU overhead input format. "
            "Use .json or .jsonl file."
        )

    def create_runner(
        self,
        model_name: str,
        batch_size: int,
        tensor_parallel_degree: int,
        output_dir: str,
        precision: str,
    ) -> Any:
        del output_dir
        if not self._normalized_records:
            raise RuntimeError(
                "vLLM backend is not started or has no loaded records. "
                "Call backend.start() before create_runner()."
            )

        normalized_precision = precision.upper()
        expected_num_prefill_tokens = DEFAULT_NUM_PREFILL_TOKENS
        expected_num_decode_tokens = (
            int(batch_size) * DEFAULT_NUM_DECODE_TOKENS_AMPLIFICATION_FACTOR
        )
        expected_scheduling_mode = DEFAULT_SCHEDULING_MODE
        matches = [
            row
            for row in self._normalized_records
            if row["model_name"] == model_name
            and int(row["batch_size"]) == int(batch_size)
            and int(row["tensor_parallel_degree"]) == int(tensor_parallel_degree)
            and row["profiling_precision"] == normalized_precision
            and int(row["num_prefill_tokens"]) == expected_num_prefill_tokens
            and int(row["num_decode_tokens"]) == expected_num_decode_tokens
            and str(row["scheduling_mode"]).lower() == expected_scheduling_mode
        ]
        if not matches:
            raise ValueError(
                "No vLLM replay row found for "
                f"(model_name={model_name}, batch_size={batch_size}, "
                f"tensor_parallel_degree={tensor_parallel_degree}, "
                f"profiling_precision={normalized_precision}, "
                f"num_prefill_tokens={expected_num_prefill_tokens}, "
                f"num_decode_tokens={expected_num_decode_tokens}, "
                f"scheduling_mode={expected_scheduling_mode})."
            )
        if len(matches) > 1:
            raise ValueError(
                "Found duplicate vLLM replay rows for "
                f"(model_name={model_name}, batch_size={batch_size}, "
                f"tensor_parallel_degree={tensor_parallel_degree}, "
                f"profiling_precision={normalized_precision}, "
                f"num_prefill_tokens={expected_num_prefill_tokens}, "
                f"num_decode_tokens={expected_num_decode_tokens}, "
                f"scheduling_mode={expected_scheduling_mode})."
            )
        return _VllmReplayRunner(matches[0])

    def run_runner(self, runner: Any) -> dict:
        if not hasattr(runner, "run"):
            raise ValueError(
                f"Invalid vLLM replay runner type: {type(runner).__name__}"
            )
        return runner.run()
