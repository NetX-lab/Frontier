"""
Trace Store for capturing granular op-level execution traces.

This module handles the buffering and persistence of detailed execution traces
for compute operations, communication primitives, and inter-cluster transfers.
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from threading import Lock

from frontier.logger import init_logger
from frontier.types import ClusterType

logger = init_logger(__name__)


@dataclass
class TraceEvent:
    """A single trace event representing an operation."""

    type: str  # COMPUTE, COMM, TRANSFER, OVERHEAD
    name: str  # e.g., "attn_pre_proj", "all_reduce", "m2n_transfer"
    ts_start: float  # Start timestamp (simulation time in seconds)
    duration_ms: float  # Duration in milliseconds
    cluster: str  # Cluster name/type

    # Optional context fields
    replica_id: Optional[int] = None
    batch_id: Optional[str] = None
    request_id: Optional[str] = None  # For per-request ops
    layer_id: Optional[int] = None

    # For transfers
    target_cluster: Optional[str] = None

    # Flexible metadata
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, filtering out None values."""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


class TraceStore:
    """
    Centralized store for op-level traces.

    Features:
    - Thread-safe event buffering
    - Periodic flushing to disk (JSONL format)
    - Metadata management
    """

    def __init__(
        self, output_dir: str, enabled: bool = False, filename: str = "op_traces.jsonl"
    ):
        self._enabled = enabled
        self._output_dir = output_dir
        self._filename = filename
        self._filepath = os.path.join(output_dir, filename)

        self._buffer: List[TraceEvent] = []
        self._buffer_lock = Lock()
        self._buffer_size_limit = 5000  # Flush every 5k events (reduced I/O)

        self._initialized = False

    def initialize(self, simulation_info: Dict[str, Any]):
        """Initialize the trace file with metadata."""
        if not self._enabled:
            return

        os.makedirs(self._output_dir, exist_ok=True)

        # Write metadata header
        metadata = {
            "meta": {
                "timestamp": time.time(),
                "simulation_info": simulation_info,
                "version": "1.0",
            }
        }

        with open(self._filepath, "w") as f:
            f.write(json.dumps(metadata) + "\n")

        self._initialized = True
        logger.info(f"TraceStore initialized at {self._filepath}")

    def log_event(self, event: TraceEvent):
        """Log a single trace event."""
        if not self._enabled:
            return

        with self._buffer_lock:
            self._buffer.append(event)
            should_flush = len(self._buffer) >= self._buffer_size_limit

        if should_flush:
            self.flush()

    def flush(self):
        """Flush buffered events to disk."""
        if not self._enabled or not self._initialized:
            return

        with self._buffer_lock:
            if not self._buffer:
                return
            events_to_write = self._buffer
            self._buffer = []

        try:
            with open(self._filepath, "a") as f:
                for event in events_to_write:
                    f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to flush trace events: {e}")
            with self._buffer_lock:
                self._buffer = events_to_write + self._buffer
            raise

    def close(self):
        """Flush remaining events and close."""
        self.flush()
        if self._enabled:
            logger.info(f"TraceStore closed. Trace written to {self._filepath}")
