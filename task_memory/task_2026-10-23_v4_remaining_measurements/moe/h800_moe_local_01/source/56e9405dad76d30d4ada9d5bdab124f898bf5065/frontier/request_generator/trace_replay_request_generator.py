import logging
import math
import json
from typing import List

import pandas as pd

from frontier.config import TraceRequestGeneratorConfig
from frontier.entities import Request, RequestRoundPlan
from frontier.request_generator.base_request_generator import BaseRequestGenerator

logger = logging.getLogger(__name__)


def _parse_block_hash_ids(value) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1]
        delimiter = "|" if "|" in stripped else ","
        return [int(part.strip()) for part in stripped.split(delimiter) if part.strip()]
    if isinstance(value, int):
        return [int(value)]
    raise ValueError(f"Unsupported block_hash_ids value: {value!r}")


def _is_missing_value(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _parse_optional_int(value) -> int | None:
    if _is_missing_value(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return int(value)


def _parse_optional_float(value) -> float | None:
    if _is_missing_value(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def _parse_thinking_round_plans(value) -> list[RequestRoundPlan] | None:
    if _is_missing_value(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        raw_round_plans = json.loads(stripped)
    else:
        raw_round_plans = value

    if not isinstance(raw_round_plans, list):
        raise ValueError(
            "thinking_round_plans_json must decode to a list of round plans."
        )

    round_plans: list[RequestRoundPlan] = []
    for raw_round_plan in raw_round_plans:
        if not isinstance(raw_round_plan, dict):
            raise ValueError(
                "Each thinking round plan must be a dict with prefill/decode token counts."
            )
        round_plans.append(
            RequestRoundPlan(
                num_prefill_tokens=int(raw_round_plan["num_prefill_tokens"]),
                num_decode_tokens=int(raw_round_plan["num_decode_tokens"]),
            )
        )
    return round_plans


class TraceReplayRequestGenerator(BaseRequestGenerator):
    """
    Reads a trace csv file containing request arrival time, its prompt and completion token values to generate
    inter-request times, number of tokens.
    """

    def __init__(self, config: TraceRequestGeneratorConfig):
        super().__init__(config)

        # load into a pd dataframe
        self.trace_df = pd.read_csv(config.trace_file)

        # scale prefill and decode tokens
        self.trace_df["num_prefill_tokens"] = (
            self.trace_df["num_prefill_tokens"] * config.prefill_scale_factor
        )
        self.trace_df["num_decode_tokens"] = (
            self.trace_df["num_decode_tokens"] * config.decode_scale_factor
        )

        # make sure all the prefill and decode counts are integers
        self.trace_df["num_prefill_tokens"] = self.trace_df[
            "num_prefill_tokens"
        ].astype(int)
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].astype(
            int
        )

        # make sure that there is at least one prefill and decode token
        self.trace_df["num_prefill_tokens"] = self.trace_df["num_prefill_tokens"].clip(
            lower=1
        )
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].clip(
            lower=1
        )

        # make sure the total does not exceed the max tokens, adjust the prefill tokens if needed
        total_tokens = (
            self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
        )
        if "thinking_round_plans_json" in self.trace_df.columns:
            overflowing_multi_round_rows = []
            for row_index, row in self.trace_df.loc[
                total_tokens > config.max_tokens
            ].iterrows():
                row_thinking_depth = _parse_optional_int(row.get("thinking_depth"))
                row_thinking_round_plans = _parse_thinking_round_plans(
                    row.get("thinking_round_plans_json")
                )
                is_multi_round_row = (
                    row_thinking_round_plans is not None
                    or (
                        row_thinking_depth is not None
                        and row_thinking_depth > 1
                    )
                )
                if not is_multi_round_row:
                    continue
                session_id = _parse_optional_int(row.get("session_id"))
                row_label = (
                    f"session_id={session_id}"
                    if session_id is not None
                    else f"row_index={row_index}"
                )
                overflowing_multi_round_rows.append(row_label)
            if overflowing_multi_round_rows:
                raise ValueError(
                    "Trace replay multi-round rows exceed "
                    f"max_tokens={config.max_tokens}: "
                    f"{', '.join(overflowing_multi_round_rows)}. "
                    "Increase trace_request_generator_config_max_tokens "
                    "instead of relying on prefill clipping."
                )
        diff_tokens = total_tokens - config.max_tokens
        diff_tokens = diff_tokens.clip(lower=0)
        self.trace_df["num_prefill_tokens"] = (
            self.trace_df["num_prefill_tokens"] - diff_tokens
        )

        assert all(
            self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
            <= config.max_tokens
        )

        # rescale the time to change QPS
        self.trace_df["arrived_at"] = (
            self.trace_df["arrived_at"] * config.time_scale_factor
        )

        logger.info(
            f"Loaded trace file {config.trace_file} with {len(self.trace_df)} requests"
        )
        # compute pd ratio and log the 25, 50, 75, 90, 95, 99 percentiles
        pd_ratio = (
            self.trace_df["num_prefill_tokens"] / self.trace_df["num_decode_tokens"]
        )
        logger.debug(
            f"Prompt/decode token ratio stats\n:{pd_ratio.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])}"
        )

    def generate_requests(self) -> List[Request]:
        requests = []

        for _, row in self.trace_df.iterrows():
            # Read priority from trace file if available, otherwise default to 0
            priority = int(row.get("priority", 0))
            session_id = _parse_optional_int(row.get("session_id"))
            cohort = row.get("cohort")
            if _is_missing_value(cohort):
                cohort = None
            elif isinstance(cohort, str):
                cohort = cohort.strip() or None
            block_hash_ids = _parse_block_hash_ids(row.get("block_hash_ids"))
            row_thinking_depth = _parse_optional_int(row.get("thinking_depth"))
            row_tool_call_latency = _parse_optional_float(row.get("tool_call_latency"))
            row_thinking_round_plans = _parse_thinking_round_plans(
                row.get("thinking_round_plans_json")
            )

            if (
                row_thinking_depth is None
                and row_tool_call_latency is None
                and row_thinking_round_plans is None
            ):
                request = self._build_request(
                    arrived_at=row["arrived_at"],
                    num_prefill_tokens=row["num_prefill_tokens"],
                    num_decode_tokens=row["num_decode_tokens"],
                    priority=priority,
                    block_hash_ids=block_hash_ids,
                    session_id=session_id,
                    cohort=cohort,
                )
                requests.append(request)
                continue

            if row_thinking_round_plans is not None:
                inferred_thinking_depth = len(row_thinking_round_plans)
            else:
                inferred_thinking_depth = 1
            thinking_depth = (
                row_thinking_depth
                if row_thinking_depth is not None
                else inferred_thinking_depth
            )
            if thinking_depth < 1:
                raise ValueError(
                    f"thinking_depth must be >= 1 for trace row, got={thinking_depth}"
                )
            if row_thinking_round_plans is None and thinking_depth != 1:
                raise ValueError(
                    "Trace rows with thinking_depth > 1 must provide thinking_round_plans_json."
                )
            if row_tool_call_latency is None:
                row_tool_call_latency = 0.001

            request = Request(
                arrived_at=float(row["arrived_at"]),
                num_prefill_tokens=int(row["num_prefill_tokens"]),
                num_decode_tokens=int(row["num_decode_tokens"]),
                priority=priority,
                block_hash_ids=block_hash_ids,
                session_id=session_id,
                cohort=cohort,
                thinking_depth=thinking_depth,
                tool_call_latency=row_tool_call_latency,
                thinking_round_plans=row_thinking_round_plans,
            )

            requests.append(request)

        return requests
