"""
Cluster Event Logger for detailed event tracking in disaggregated clusters.
"""

import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
from frontier.types import ClusterType
from frontier.logger import get_cluster_logger

class ClusterEventLogger:
    """Logger for tracking detailed cluster events."""

    def __init__(self, cluster_type: ClusterType, log_dir: str = "logs/cluster_events",
                 enabled: bool = True, log_level: str = "INFO"):
        self.cluster_type = cluster_type
        self._logger = get_cluster_logger(__name__, cluster_type.name)
        self.log_dir = log_dir
        self.enabled = enabled
        self.log_level = log_level.upper()
        self.event_count = 0
        self.error_count = 0
        self.start_time = time.time()

        # Event type statistics
        self.event_stats = {}

        # Log level hierarchy (more verbose -> smaller number)
        # NOTE: "ALL" means emit everything (equivalent to DEBUG for this file logger).
        self.log_levels = {
            "ALL": 0,
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4,
        }
        self.current_log_level = self.log_levels.get(self.log_level, self.log_levels["INFO"])

        if not self.enabled:
            self._logger.info(f"ClusterEventLogger for {cluster_type.name} is DISABLED")
            self.log_file_path = None
            return

        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(
            self.log_dir,
            f"{cluster_type.name.lower()}_{timestamp}.log"
        )

        # Initialize log file
        self._initialize_log_file()

        self._logger.info(f"ClusterEventLogger initialized for {cluster_type.name}")
        self._logger.info(f"Log file: {self.log_file_path}")
        self._logger.info(f"Log level: {self.log_level}")

    def _initialize_log_file(self):
        """Initialize the log file with header information."""
        if not self.enabled or not self.log_file_path:
            return

        with open(self.log_file_path, 'w') as f:
            f.write(f"=== VIDUR CLUSTER EVENT LOG ===\n")
            f.write(f"Cluster Type: {self.cluster_type.name}\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Log Level: {self.log_level}\n")
            f.write(f"Log File: {self.log_file_path}\n")
            f.write("=" * 50 + "\n\n")
    
    def log_event_start(self, event_type: str, event_id: str, details: Optional[Dict[str, Any]] = None):
        """Log the start of an event."""
        # Always update statistics even if logging is disabled
        self.event_count += 1
        if event_type not in self.event_stats:
            self.event_stats[event_type] = {'count': 0, 'errors': 0}
        self.event_stats[event_type]['count'] += 1

        # Skip file logging if disabled or log level too low
        if not self.enabled or not self.log_file_path or self.current_log_level > self.log_levels.get("INFO", 1):
            return

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_entry = f"[{timestamp}] START {event_type} | ID: {event_id}"

        if details:
            detail_str = " | ".join([f"{k}: {v}" for k, v in details.items()])
            log_entry += f" | {detail_str}"

        log_entry += "\n"

        with open(self.log_file_path, 'a') as f:
            f.write(log_entry)
    
    def log_event_complete(self, event_type: str, event_id: str, duration_ms: float, details: Optional[Dict[str, Any]] = None):
        """Log the completion of an event."""
        # Skip file logging if disabled or log level too low
        if not self.enabled or not self.log_file_path or self.current_log_level > self.log_levels.get("INFO", 1):
            return

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_entry = f"[{timestamp}] COMPLETE {event_type} | ID: {event_id} | Duration: {duration_ms:.3f}ms"

        if details:
            detail_str = " | ".join([f"{k}: {v}" for k, v in details.items()])
            log_entry += f" | {detail_str}"

        log_entry += "\n"

        with open(self.log_file_path, 'a') as f:
            f.write(log_entry)
    
    def log_event_error(self, event_type: str, event_id: str, error_msg: str, details: Optional[Dict[str, Any]] = None):
        """Log an event error."""
        # Always update error statistics even if logging is disabled
        self.error_count += 1
        if event_type in self.event_stats:
            self.event_stats[event_type]['errors'] += 1

        # Skip file logging if disabled or log level too high (ERROR level should always be logged)
        if not self.enabled or not self.log_file_path or self.current_log_level > self.log_levels.get("ERROR", 3):
            return

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_entry = f"[{timestamp}] ERROR {event_type} | ID: {event_id} | Error: {error_msg}"

        if details:
            detail_str = " | ".join([f"{k}: {v}" for k, v in details.items()])
            log_entry += f" | {detail_str}"

        log_entry += "\n"

        with open(self.log_file_path, 'a') as f:
            f.write(log_entry)
    
    def log_batch_info(self, batch_id: str, batch_size: int, num_tokens: int, replica_id: int):
        """Log batch processing information."""
        # Skip file logging if disabled or log level too low (DEBUG level)
        if not self.enabled or not self.log_file_path or self.current_log_level > self.log_levels.get("DEBUG", 0):
            return

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_entry = f"[{timestamp}] BATCH | ID: {batch_id} | Size: {batch_size} | Tokens: {num_tokens} | Replica: {replica_id}\n"

        with open(self.log_file_path, 'a') as f:
            f.write(log_entry)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current event processing statistics."""
        total_time = time.time() - self.start_time
        
        return {
            'cluster_type': self.cluster_type.name,
            'total_events': self.event_count,
            'total_errors': self.error_count,
            'total_time_seconds': total_time,
            'events_per_second': self.event_count / total_time if total_time > 0 else 0,
            'error_rate': self.error_count / self.event_count if self.event_count > 0 else 0,
            'event_type_stats': self.event_stats,
            'log_file': self.log_file_path
        }
    
    def write_summary(self):
        """Write a summary of event processing to the log file."""
        stats = self.get_statistics()

        # Always write summary if enabled, regardless of log level
        if self.enabled and self.log_file_path:
            with open(self.log_file_path, 'a') as f:
                f.write("\n" + "=" * 50 + "\n")
                f.write("=== EVENT PROCESSING SUMMARY ===\n")
                f.write(f"Cluster Type: {stats['cluster_type']}\n")
                f.write(f"Total Events Processed: {stats['total_events']}\n")
                f.write(f"Total Errors: {stats['total_errors']}\n")
                f.write(f"Total Processing Time: {stats['total_time_seconds']:.3f} seconds\n")
                f.write(f"Events Per Second: {stats['events_per_second']:.2f}\n")
                f.write(f"Error Rate: {stats['error_rate']:.4f}\n")
                f.write("\n=== EVENT TYPE BREAKDOWN ===\n")

                for event_type, type_stats in stats['event_type_stats'].items():
                    f.write(f"{event_type}: {type_stats['count']} events, {type_stats['errors']} errors\n")

                f.write("\n" + "=" * 50 + "\n")
                f.write(f"Log completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            self._logger.info(f"{stats['cluster_type']} event summary written to {self.log_file_path}")
        else:
            self._logger.info(f"{stats['cluster_type']} event logging was disabled - no summary file written")

        return stats

    def is_enabled(self) -> bool:
        """
        Check if event logging is enabled.

        Returns:
            bool: True if logging is enabled, False otherwise
        """
        return self.enabled

    def get_logger(self):
        """
        Get the logger instance for this cluster.

        Returns:
            Logger instance for cluster-specific logging
        """
        return self._logger
