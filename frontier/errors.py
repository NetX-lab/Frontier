from __future__ import annotations

from typing import Any, Dict, Optional


class FrontierMemoryOOMError(RuntimeError):
    """Structured fail-fast error for Frontier GPU-memory admission failures."""

    def __init__(
        self,
        message: str,
        *,
        reason: str = "unspecified",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.reason = str(reason)
        self.details = dict(details or {})

        detail_suffix = ""
        if self.details:
            ordered_items = ", ".join(
                f"{key}={value}" for key, value in sorted(self.details.items())
            )
            detail_suffix = f" ({ordered_items})"

        super().__init__(
            f"[FRONTIER_MEMORY_OOM][reason={self.reason}] {message}{detail_suffix}"
        )
