"""Speculative decoding and MTP configuration, including trace admission."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Dict, List, Optional

from frontier.spec_decode.proposer_profile import (
    load_decode_draft_proposer_latency_profile,
)


@dataclass
class SpeculativeDecodingConfig:
    enabled: bool = field(
        default=False,
        metadata={"help": "Enable speculative decoding simulation."},
    )
    method: str = field(
        default="eagle",
        metadata={
            "help": "Speculative decoding method. Must match vLLM method names."
        },
    )
    spec_model_name: str = field(
        default="",
        metadata={
            "help": "Optional draft/spec model name for methods whose proposer "
            "decoder comes from a separate draft model (for example draft-model MTP)."
        },
    )
    num_speculative_tokens: int = field(
        default=4,
        metadata={"help": "Number of draft tokens planned per speculative iteration."},
    )
    committed_tokens_per_iteration: int = field(
        default=2,
        metadata={
            "help": "Deterministic committed token count per speculative iteration "
            "(includes 1 target token + accepted drafts)."
        },
    )
    acceptance_trace_file: str = field(
        default="",
        metadata={
            "help": "Optional deterministic acceptance trace JSON file. Supported "
            "formats: list[int] or {'committed_tokens_per_iteration': list[int], "
            "'scheduled_draft_tokens_per_iteration': optional list[int], "
            "'per_request_committed_tokens_per_iteration': optional dict[str, list[int]], "
            "'per_request_scheduled_draft_tokens_per_iteration': optional dict[str, list[int]]}. "
            "When set, trace overrides committed_tokens_per_iteration and can "
            "optionally override planned draft widths per iteration."
        },
    )
    proposer_overhead_ms_by_method: Dict[str, float] = field(
        default_factory=dict,
        metadata={
            "help": "Method-aware proposer overhead in milliseconds per speculative "
            "verify request (method -> overhead_ms >= 0)."
        },
    )
    decode_draft_proposer_latency_profile_file: str = field(
        default="",
        metadata={
            "help": "Optional structured latency profile JSON for decode draft "
            "proposer overhead. Expected workload key: "
            "(method, model_name, attn_tp_size, num_speculative_tokens, "
            "spec_verify_request_count)."
        },
    )
    mtp_n_predict: int = field(
        default=0,
        metadata={
            "help": "Optional MTP capability metadata. Number of tokens predicted "
            "per MTP block. Only valid for MTP methods."
        },
    )
    mtp_num_layers: int = field(
        default=0,
        metadata={
            "help": "Optional MTP capability metadata. Number of MTP layers. "
            "Only valid for MTP methods."
        },
    )
    trace_calibration_file: str = field(
        default="",
        metadata={
            "help": "Optional calibration JSON file. Supported keys: "
            "proposer_overhead_ms_by_method and metadata."
        },
    )

    @staticmethod
    def _validate_method_float_map(
        *,
        map_name: str,
        raw_map: Optional[Dict[str, float]],
        supported_methods: set[str],
        min_value: float,
        inclusive_min: bool,
    ) -> Dict[str, float]:
        if raw_map is None:
            return {}
        if not isinstance(raw_map, dict):
            raise ValueError(
                f"SpeculativeDecodingConfig.{map_name} must be a dict, "
                f"got={type(raw_map).__name__}"
            )

        validated: Dict[str, float] = {}
        for method_name, value in raw_map.items():
            if method_name not in supported_methods:
                raise ValueError(
                    f"SpeculativeDecodingConfig.{map_name} contains unsupported method "
                    f"{method_name!r}; supported={sorted(supported_methods)}"
                )
            numeric_value = float(value)
            if inclusive_min:
                if numeric_value < min_value:
                    raise ValueError(
                        f"SpeculativeDecodingConfig.{map_name}[{method_name!r}] "
                        f"must be >= {min_value}, got={numeric_value!r}"
                    )
            elif numeric_value <= min_value:
                raise ValueError(
                    f"SpeculativeDecodingConfig.{map_name}[{method_name!r}] "
                    f"must be > {min_value}, got={numeric_value!r}"
                )
            validated[method_name] = numeric_value
        return validated

    @staticmethod
    def _load_trace_calibration_payload(
        trace_calibration_file: str,
    ) -> Dict[str, Dict[str, float]]:
        if not trace_calibration_file:
            return {}
        if not os.path.isfile(trace_calibration_file):
            raise ValueError(
                "SpeculativeDecodingConfig.trace_calibration_file does not exist: "
                f"{trace_calibration_file!r}"
            )
        try:
            with open(trace_calibration_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "SpeculativeDecodingConfig.trace_calibration_file must be valid JSON: "
                f"{trace_calibration_file!r}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                "SpeculativeDecodingConfig.trace_calibration_file must contain a JSON "
                f"object, got={type(payload).__name__}"
            )
        return payload

    @staticmethod
    def _load_acceptance_trace_payload(
        *,
        acceptance_trace_file: str,
    ):
        if not acceptance_trace_file:
            return None
        if not os.path.isfile(acceptance_trace_file):
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file does not exist: "
                f"{acceptance_trace_file!r}"
            )
        try:
            with open(acceptance_trace_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file must be valid JSON: "
                f"{acceptance_trace_file!r}"
            ) from exc

        if not isinstance(payload, (list, dict)):
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file must be list or dict, "
                f"got={type(payload).__name__}"
            )
        return payload

    @staticmethod
    def _load_committed_tokens_trace(
        *,
        acceptance_trace_payload,
        max_committed_tokens: int,
    ) -> Optional[List[int]]:
        if acceptance_trace_payload is None:
            return None

        if isinstance(acceptance_trace_payload, list):
            committed_tokens_trace_raw = acceptance_trace_payload
        else:
            if "committed_tokens_per_iteration" not in acceptance_trace_payload:
                return None
            committed_tokens_trace_raw = acceptance_trace_payload[
                "committed_tokens_per_iteration"
            ]

        if not isinstance(committed_tokens_trace_raw, list):
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file committed token trace "
                f"must be a list, got={type(committed_tokens_trace_raw).__name__}"
            )
        if len(committed_tokens_trace_raw) == 0:
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file committed token trace "
                "must be non-empty."
            )

        committed_tokens_trace: List[int] = []
        for idx, value in enumerate(committed_tokens_trace_raw):
            committed = int(value)
            if committed < 0:
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file values must be >= 0, "
                    f"got index={idx}, value={value!r}"
                )
            if committed > max_committed_tokens:
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file values must be <= "
                    f"1 + num_speculative_tokens ({max_committed_tokens}), "
                    f"got index={idx}, value={value!r}"
                )
            committed_tokens_trace.append(committed)
        return committed_tokens_trace

    @staticmethod
    def _load_per_request_committed_tokens_trace(
        *,
        acceptance_trace_payload,
        max_committed_tokens: int,
    ) -> Optional[Dict[str, List[int]]]:
        if acceptance_trace_payload is None or not isinstance(
            acceptance_trace_payload, dict
        ):
            return None
        if "per_request_committed_tokens_per_iteration" not in acceptance_trace_payload:
            return None

        raw_trace_map = acceptance_trace_payload[
            "per_request_committed_tokens_per_iteration"
        ]
        if not isinstance(raw_trace_map, dict):
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file "
                "per_request_committed_tokens_per_iteration must be a dict, "
                f"got={type(raw_trace_map).__name__}"
            )
        if len(raw_trace_map) == 0:
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file "
                "per_request_committed_tokens_per_iteration must be non-empty."
            )

        per_request_trace: Dict[str, List[int]] = {}
        for raw_request_id, raw_trace in raw_trace_map.items():
            request_id = str(raw_request_id)
            if not request_id:
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file per-request "
                    "trace keys must be non-empty strings."
                )
            if request_id in per_request_trace:
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file contains "
                    f"duplicate request_id={request_id!r} after normalization."
                )
            if not isinstance(raw_trace, list):
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file "
                    "per-request committed token trace must be a list, "
                    f"got request_id={request_id!r}, type={type(raw_trace).__name__}"
                )
            if len(raw_trace) == 0:
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file per-request "
                    f"committed token trace must be non-empty, request_id={request_id!r}"
                )

            validated_trace: List[int] = []
            for idx, value in enumerate(raw_trace):
                committed = int(value)
                if committed < 0:
                    raise ValueError(
                        "SpeculativeDecodingConfig.acceptance_trace_file per-request "
                        "committed token values must be >= 0, "
                        f"got request_id={request_id!r}, index={idx}, value={value!r}"
                    )
                if committed > max_committed_tokens:
                    raise ValueError(
                        "SpeculativeDecodingConfig.acceptance_trace_file per-request "
                        "committed token values must be <= 1 + num_speculative_tokens "
                        f"({max_committed_tokens}), got request_id={request_id!r}, "
                        f"index={idx}, value={value!r}"
                    )
                validated_trace.append(committed)
            per_request_trace[request_id] = validated_trace
        return per_request_trace

    @staticmethod
    def _load_scheduled_draft_tokens_trace(
        *,
        acceptance_trace_payload,
        max_scheduled_draft_tokens: int,
        committed_trace_length: int,
    ) -> Optional[List[int]]:
        if acceptance_trace_payload is None or not isinstance(acceptance_trace_payload, dict):
            return None
        if "scheduled_draft_tokens_per_iteration" not in acceptance_trace_payload:
            return None

        scheduled_draft_tokens_trace_raw = acceptance_trace_payload[
            "scheduled_draft_tokens_per_iteration"
        ]
        if not isinstance(scheduled_draft_tokens_trace_raw, list):
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file scheduled draft token "
                f"trace must be a list, got={type(scheduled_draft_tokens_trace_raw).__name__}"
            )
        if len(scheduled_draft_tokens_trace_raw) != committed_trace_length:
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file "
                "scheduled_draft_tokens_per_iteration length must match "
                "committed_tokens_per_iteration length."
            )

        scheduled_draft_tokens_trace: List[int] = []
        for idx, value in enumerate(scheduled_draft_tokens_trace_raw):
            scheduled_drafts = int(value)
            if scheduled_drafts < 0:
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file scheduled draft "
                    "trace values must be >= 0, "
                    f"got index={idx}, value={value!r}"
                )
            if scheduled_drafts > max_scheduled_draft_tokens:
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file scheduled draft "
                    "trace values must be <= num_speculative_tokens "
                    f"({max_scheduled_draft_tokens}), got index={idx}, value={value!r}"
                )
            scheduled_draft_tokens_trace.append(scheduled_drafts)
        return scheduled_draft_tokens_trace

    @staticmethod
    def _load_per_request_scheduled_draft_tokens_trace(
        *,
        acceptance_trace_payload,
        max_scheduled_draft_tokens: int,
        per_request_committed_trace: Optional[Dict[str, List[int]]],
    ) -> Optional[Dict[str, List[int]]]:
        if acceptance_trace_payload is None or not isinstance(
            acceptance_trace_payload, dict
        ):
            return None
        if (
            "per_request_scheduled_draft_tokens_per_iteration"
            not in acceptance_trace_payload
        ):
            return None
        if per_request_committed_trace is None:
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file "
                "per_request_scheduled_draft_tokens_per_iteration requires "
                "per_request_committed_tokens_per_iteration."
            )

        raw_trace_map = acceptance_trace_payload[
            "per_request_scheduled_draft_tokens_per_iteration"
        ]
        if not isinstance(raw_trace_map, dict):
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file "
                "per_request_scheduled_draft_tokens_per_iteration must be a dict, "
                f"got={type(raw_trace_map).__name__}"
            )

        normalized_keys = {str(request_id) for request_id in raw_trace_map.keys()}
        committed_keys = set(per_request_committed_trace.keys())
        if normalized_keys != committed_keys:
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file "
                "per_request_scheduled_draft_tokens_per_iteration keys must match "
                "per_request_committed_tokens_per_iteration keys."
            )

        per_request_trace: Dict[str, List[int]] = {}
        for request_id, committed_trace in per_request_committed_trace.items():
            raw_trace = raw_trace_map[request_id]
            if not isinstance(raw_trace, list):
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file per-request "
                    "scheduled draft token trace must be a list, "
                    f"got request_id={request_id!r}, type={type(raw_trace).__name__}"
                )
            if len(raw_trace) != len(committed_trace):
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file "
                    "per_request_scheduled_draft_tokens_per_iteration length must "
                    "match per_request_committed_tokens_per_iteration length, "
                    f"request_id={request_id!r}"
                )

            validated_trace: List[int] = []
            for idx, value in enumerate(raw_trace):
                scheduled_drafts = int(value)
                if scheduled_drafts < 0:
                    raise ValueError(
                        "SpeculativeDecodingConfig.acceptance_trace_file per-request "
                        "scheduled draft token values must be >= 0, "
                        f"got request_id={request_id!r}, index={idx}, value={value!r}"
                    )
                if scheduled_drafts > max_scheduled_draft_tokens:
                    raise ValueError(
                        "SpeculativeDecodingConfig.acceptance_trace_file per-request "
                        "scheduled draft token values must be <= num_speculative_tokens "
                        f"({max_scheduled_draft_tokens}), got request_id={request_id!r}, "
                        f"index={idx}, value={value!r}"
                    )
                validated_trace.append(scheduled_drafts)
            per_request_trace[request_id] = validated_trace
        return per_request_trace

    def __post_init__(self) -> None:
        supported_methods = {
            "ngram",
            "medusa",
            "eagle",
            "eagle3",
            "deepseek_mtp",
            "ernie_mtp",
            "qwen3_moe_mtp",
            "qwen3_next_mtp",
        }
        mtp_methods = {
            "deepseek_mtp",
            "ernie_mtp",
            "qwen3_moe_mtp",
            "qwen3_next_mtp",
        }
        if self.enabled and self.method not in supported_methods:
            raise ValueError(
                "SpeculativeDecodingConfig.method must match vLLM method names, "
                f"got={self.method!r}, supported={sorted(supported_methods)}"
            )
        if self.enabled and self.method in mtp_methods and self.mtp_n_predict <= 0:
            raise ValueError(
                "MTP methods require mtp_n_predict > 0 when enabled=True, "
                f"got method={self.method!r}, mtp_n_predict={self.mtp_n_predict!r}"
            )
        if self.enabled and self.method in mtp_methods and self.mtp_num_layers <= 0:
            raise ValueError(
                "MTP methods require mtp_num_layers > 0 when enabled=True, "
                f"got method={self.method!r}, mtp_num_layers={self.mtp_num_layers!r}"
            )
        if self.mtp_n_predict < 0:
            raise ValueError(
                "SpeculativeDecodingConfig.mtp_n_predict must be >= 0, "
                f"got={self.mtp_n_predict!r}"
            )
        if self.mtp_num_layers < 0:
            raise ValueError(
                "SpeculativeDecodingConfig.mtp_num_layers must be >= 0, "
                f"got={self.mtp_num_layers!r}"
            )
        if self.mtp_n_predict > 0 and self.method not in mtp_methods:
            raise ValueError(
                "SpeculativeDecodingConfig.mtp_n_predict is only valid for MTP "
                f"methods, got method={self.method!r}"
            )
        if self.mtp_num_layers > 0 and self.method not in mtp_methods:
            raise ValueError(
                "SpeculativeDecodingConfig.mtp_num_layers is only valid for MTP "
                f"methods, got method={self.method!r}"
            )
        if self.enabled and self.num_speculative_tokens <= 0:
            raise ValueError(
                "SpeculativeDecodingConfig.num_speculative_tokens must be > 0 when "
                f"enabled=True, got={self.num_speculative_tokens}"
            )
        if (
            self.method in mtp_methods
            and self.mtp_n_predict > 0
            and self.num_speculative_tokens % self.mtp_n_predict != 0
        ):
            raise ValueError(
                "SpeculativeDecodingConfig.num_speculative_tokens must be divisible "
                "by mtp_n_predict when mtp_n_predict > 0 for MTP methods, "
                f"got num_speculative_tokens={self.num_speculative_tokens}, "
                f"mtp_n_predict={self.mtp_n_predict}"
            )
        max_committed_tokens = int(self.num_speculative_tokens) + 1
        if self.committed_tokens_per_iteration < 1:
            raise ValueError(
                "SpeculativeDecodingConfig.committed_tokens_per_iteration must be >= 1, "
                f"got={self.committed_tokens_per_iteration!r}"
            )
        if self.committed_tokens_per_iteration > max_committed_tokens:
            raise ValueError(
                "SpeculativeDecodingConfig.committed_tokens_per_iteration must be <= "
                f"1 + num_speculative_tokens ({max_committed_tokens}), "
                f"got={self.committed_tokens_per_iteration!r}"
            )
        acceptance_trace_payload = self._load_acceptance_trace_payload(
            acceptance_trace_file=self.acceptance_trace_file,
        )
        self._committed_tokens_trace = self._load_committed_tokens_trace(
            acceptance_trace_payload=acceptance_trace_payload,
            max_committed_tokens=max_committed_tokens,
        )
        self._per_request_committed_tokens_trace = (
            self._load_per_request_committed_tokens_trace(
                acceptance_trace_payload=acceptance_trace_payload,
                max_committed_tokens=max_committed_tokens,
            )
        )
        if (
            acceptance_trace_payload is not None
            and self._committed_tokens_trace is None
            and self._per_request_committed_tokens_trace is None
        ):
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file JSON object must "
                "contain key 'committed_tokens_per_iteration' or "
                "'per_request_committed_tokens_per_iteration'."
            )
        self._scheduled_draft_tokens_trace = self._load_scheduled_draft_tokens_trace(
            acceptance_trace_payload=acceptance_trace_payload,
            max_scheduled_draft_tokens=int(self.num_speculative_tokens),
            committed_trace_length=(
                len(self._committed_tokens_trace)
                if self._committed_tokens_trace is not None
                else 0
            ),
        )
        self._per_request_scheduled_draft_tokens_trace = (
            self._load_per_request_scheduled_draft_tokens_trace(
                acceptance_trace_payload=acceptance_trace_payload,
                max_scheduled_draft_tokens=int(self.num_speculative_tokens),
                per_request_committed_trace=self._per_request_committed_tokens_trace,
            )
        )
        if self._scheduled_draft_tokens_trace is not None:
            for idx, (committed_tokens, scheduled_draft_tokens) in enumerate(
                zip(
                    self._committed_tokens_trace,
                    self._scheduled_draft_tokens_trace,
                )
            ):
                if committed_tokens > 1 + scheduled_draft_tokens:
                    raise ValueError(
                        "SpeculativeDecodingConfig.acceptance_trace_file committed "
                        "tokens must be <= 1 + scheduled_draft_tokens_per_iteration, "
                        f"got index={idx}, committed={committed_tokens}, "
                        f"scheduled_draft_tokens={scheduled_draft_tokens}"
                    )
        if self._per_request_scheduled_draft_tokens_trace is not None:
            for request_id, committed_trace in (
                self._per_request_committed_tokens_trace.items()
            ):
                scheduled_trace = self._per_request_scheduled_draft_tokens_trace[
                    request_id
                ]
                for idx, (committed_tokens, scheduled_draft_tokens) in enumerate(
                    zip(committed_trace, scheduled_trace)
                ):
                    if committed_tokens > 1 + scheduled_draft_tokens:
                        raise ValueError(
                            "SpeculativeDecodingConfig.acceptance_trace_file per-request "
                            "committed tokens must be <= 1 + "
                            "per_request_scheduled_draft_tokens_per_iteration, "
                            f"got request_id={request_id!r}, index={idx}, "
                            f"committed={committed_tokens}, "
                            f"scheduled_draft_tokens={scheduled_draft_tokens}"
                        )

        trace_payload = self._load_trace_calibration_payload(self.trace_calibration_file)
        supported_trace_keys = {
            "proposer_overhead_ms_by_method",
            "metadata",
        }
        unexpected_keys = sorted(set(trace_payload.keys()) - supported_trace_keys)
        if unexpected_keys:
            raise ValueError(
                "Unsupported keys in trace calibration file: "
                f"{unexpected_keys}, supported={sorted(supported_trace_keys)}"
            )

        trace_proposer_overheads = self._validate_method_float_map(
            map_name="proposer_overhead_ms_by_method",
            raw_map=trace_payload.get("proposer_overhead_ms_by_method", {}),
            supported_methods=supported_methods,
            min_value=0.0,
            inclusive_min=True,
        )
        config_proposer_overheads = self._validate_method_float_map(
            map_name="proposer_overhead_ms_by_method",
            raw_map=self.proposer_overhead_ms_by_method,
            supported_methods=supported_methods,
            min_value=0.0,
            inclusive_min=True,
        )

        # Config-driven values override trace-derived values for deterministic control.
        self.proposer_overhead_ms_by_method = {
            **trace_proposer_overheads,
            **config_proposer_overheads,
        }
        self._decode_draft_proposer_latency_profile = (
            load_decode_draft_proposer_latency_profile(
                profile_file=self.decode_draft_proposer_latency_profile_file,
                supported_methods=supported_methods,
            )
        )
