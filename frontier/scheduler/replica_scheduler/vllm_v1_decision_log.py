"""Optional JSONL log of vLLM V1 scheduling decisions.

The log is enabled by ``FRONTIER_VLLM_V1_SCHED_DECISION_LOG_PATH`` and is a
debugging aid only; when the variable is unset every call is a no-op.  It lives
in its own module so that the handler is configured exactly once, whichever
scheduler component logs first.
"""

import json
import logging
import os
from typing import Any, Dict, Optional


_FRONTIER_VLLM_V1_SCHED_DECISION_LOG_PATH = os.environ.get(
    "FRONTIER_VLLM_V1_SCHED_DECISION_LOG_PATH", ""
)
_frontier_vllm_v1_sched_decision_logger: Optional[logging.Logger] = None

if _FRONTIER_VLLM_V1_SCHED_DECISION_LOG_PATH:
    _frontier_vllm_v1_sched_decision_logger = logging.getLogger(
        "frontier.vllm_v1_sched_decision"
    )
    _frontier_vllm_v1_sched_decision_logger.setLevel(logging.INFO)
    _frontier_vllm_v1_sched_decision_logger.propagate = False

    _decision_log_dir = os.path.dirname(_FRONTIER_VLLM_V1_SCHED_DECISION_LOG_PATH)
    if _decision_log_dir:
        os.makedirs(_decision_log_dir, exist_ok=True)

    _decision_handler = logging.FileHandler(_FRONTIER_VLLM_V1_SCHED_DECISION_LOG_PATH)
    _decision_handler.setFormatter(logging.Formatter("%(message)s"))
    _frontier_vllm_v1_sched_decision_logger.addHandler(_decision_handler)


def _log_frontier_vllm_v1_schedule_decision(event: Dict[str, Any]) -> None:
    if _frontier_vllm_v1_sched_decision_logger is None:
        return
    _frontier_vllm_v1_sched_decision_logger.info(json.dumps(event))


def schedule_decision_logging_enabled() -> bool:
    """Return whether the decision log is configured for this process."""
    return _frontier_vllm_v1_sched_decision_logger is not None
