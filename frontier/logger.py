# Adapted from
# https://github.com/skypilot-org/skypilot/blob/86dc0f6283a335e4aa37b3c10716f90999f48ab6/sky/sky_logging.py
"""Logging configuration for Sarathi."""
import logging
import sys
from typing import Optional

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"

# Optimized formats for cluster-aware logging
_CLUSTER_FORMAT = "%(cluster_prefix)s%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_SHORT_DATE_FORMAT = "%H:%M:%S"


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


class ClusterAwareFormatter(NewLineFormatter):
    """
    Custom formatter that supports cluster-aware logging with filtering.

    Features:
    - Adds cluster type prefix to log entries
    - Supports cluster-based log filtering
    - Optimized timestamp format
    """

    def __init__(self, fmt, datefmt=None, cluster_filter=None, enable_cluster_prefix=True):
        super().__init__(fmt, datefmt)
        self.cluster_filter = self._parse_cluster_filter(cluster_filter)
        self.enable_cluster_prefix = enable_cluster_prefix

    def _parse_cluster_filter(self, cluster_filter):
        """Parse cluster filter string into a set of allowed cluster types.
        
        Supports:
        - Single cluster: "PREFILL", "DECODE", etc.
        - Multiple clusters: "PREFILL,DECODE", "DECODE_ATTN,DECODE_FFN"
        - All clusters: "ALL"
        """
        if not cluster_filter:
            return None

        # Handle "ALL" keyword - return all valid cluster types
        if cluster_filter.strip().upper() == "ALL":
            return {'PREFILL', 'DECODE', 'DECODE_ATTN', 'DECODE_FFN', 'MONOLITHIC', 'TRANS'}

        # Parse comma-separated cluster types
        valid_clusters = {'PREFILL', 'DECODE', 'DECODE_ATTN', 'DECODE_FFN', 'MONOLITHIC', 'TRANS'}
        allowed_clusters = set()
        for cluster_name in cluster_filter.split(','):
            cluster_name = cluster_name.strip().upper()
            if cluster_name in valid_clusters:
                allowed_clusters.add(cluster_name)

        return allowed_clusters if allowed_clusters else None

    def format(self, record):
        # Add cluster prefix if enabled and cluster info is available
        if self.enable_cluster_prefix and hasattr(record, 'cluster_type'):
            cluster_name = getattr(record, 'cluster_type', '').upper()

            # Apply cluster filtering if configured
            if self.cluster_filter and cluster_name not in self.cluster_filter:
                return None  # Filter out this log entry

            record.cluster_prefix = f"[{cluster_name}] "
        else:
            record.cluster_prefix = ""

        # Call parent format method
        msg = super().format(record)
        return msg


class ClusterFilterHandler(logging.StreamHandler):
    """
    Custom handler that filters log records based on cluster type.
    
    Supports:
    - Single cluster: "PREFILL", "DECODE", etc.
    - Multiple clusters: "PREFILL,DECODE", "DECODE_ATTN,DECODE_FFN"
    - All clusters: "ALL"
    """

    def __init__(self, stream=None, cluster_filter=None):
        super().__init__(stream)
        self.cluster_filter = self._parse_cluster_filter(cluster_filter)

    def _parse_cluster_filter(self, cluster_filter):
        """Parse cluster filter string into a set of allowed cluster types.
        
        Supports:
        - Single cluster: "PREFILL", "DECODE", etc.
        - Multiple clusters: "PREFILL,DECODE", "DECODE_ATTN,DECODE_FFN"
        - All clusters: "ALL"
        """
        if not cluster_filter:
            return None

        # Handle "ALL" keyword - return all valid cluster types
        if cluster_filter.strip().upper() == "ALL":
            return {'PREFILL', 'DECODE', 'DECODE_ATTN', 'DECODE_FFN', 'MONOLITHIC', 'TRANS'}

        # Parse comma-separated cluster types
        valid_clusters = {'PREFILL', 'DECODE', 'DECODE_ATTN', 'DECODE_FFN', 'MONOLITHIC', 'TRANS'}
        allowed_clusters = set()
        for cluster_name in cluster_filter.split(','):
            cluster_name = cluster_name.strip().upper()
            if cluster_name in valid_clusters:
                allowed_clusters.add(cluster_name)

        return allowed_clusters if allowed_clusters else None

    def filter(self, record):
        """Filter log records based on cluster type AND any attached handler filters.
        This combines the cluster-type check with handler.filters (e.g., exact-level filter).
        """
        cluster_type = getattr(record, 'cluster_type', '').upper()

        # NEW default behavior: when no cluster filter is provided, suppress ALL
        # cluster-tagged logs. This ensures "no filter = no cluster logs".
        if not self.cluster_filter and cluster_type:
            return False

        # Cluster-type check when an explicit filter is provided
        if self.cluster_filter and cluster_type and (cluster_type not in self.cluster_filter):
            return False

        # Apply any externally attached filters (e.g., ClusterEventExactLevelFilter)
        # Only apply exact-level filtering to cluster-tagged records when a filter is explicitly provided
        for f in getattr(self, 'filters', []):
            if f is self:
                continue
            # If this is a cluster-tagged record and no explicit cluster filter is set,
            # skip external filters to avoid unintended leakage
            if cluster_type and not self.cluster_filter:
                continue
            if not f.filter(record):
                return False
        return True

    def emit(self, record):
        """Emit a log record, but only if it passes the cluster filter."""
        if self.filter(record):
            # Format the record using the formatter
            msg = self.format(record)
            if msg is not None:  # ClusterAwareFormatter may return None for filtered records
                self.stream.write(msg + '\n')
                self.flush()


class ClusterEventExactLevelFilter(logging.Filter):
    """
    Filter that allows ONLY cluster-tagged records (with 'cluster_type') that match specified log level(s).
    Non-cluster records are unaffected (always pass).
    
    Supports:
    - Single level: "INFO", "DEBUG", "WARNING", "ERROR"
    - Multiple levels: "INFO,DEBUG", "WARNING,ERROR"
    - All levels: "ALL"
    """

    def __init__(self, exact_level_name: Optional[str]):
        super().__init__()
        self._allowed_levels = None
        self.set_exact_level(exact_level_name)

    def set_exact_level(self, exact_level_name: Optional[str]):
        """
        Set the allowed log level(s) for cluster-tagged records.
        
        Args:
            exact_level_name: Single level ("INFO"), multiple levels ("INFO,DEBUG"), 
                             "ALL" for all levels, or None to disable filtering.
        """
        if exact_level_name is None:
            self._allowed_levels = None
            return
        
        # Handle "ALL" keyword - allow all log levels
        if exact_level_name.strip().upper() == "ALL":
            self._allowed_levels = {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR}
            return
        
        # Parse comma-separated levels
        mapping = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
        }
        
        self._allowed_levels = set()
        for level_name in exact_level_name.split(','):
            level_name = level_name.strip().upper()
            if level_name in mapping:
                self._allowed_levels.add(mapping[level_name])
        
        # If no valid levels found, default to None (allow all)
        if not self._allowed_levels:
            self._allowed_levels = None

    def filter(self, record: logging.LogRecord) -> bool:
        # If record is not cluster-tagged, do not filter it here
        if not hasattr(record, 'cluster_type'):
            return True
        # If no levels configured, allow as normal
        if self._allowed_levels is None:
            return True
        # Allow if record level is in allowed set
        return record.levelno in self._allowed_levels


_root_logger = logging.getLogger("frontier")
_default_handler = None
_error_handler = None  # Dedicated handler for ERROR-level logs to stderr
_cluster_log_config = {
    'cluster_filter': None,
    'enable_cluster_prefix': True,
    'use_short_timestamp': True,
}

# Single shared filter instance; handler attaches this once and we mutate it on config changes
_cluster_event_level_filter = ClusterEventExactLevelFilter(None)


def _setup_logger():
    _root_logger.setLevel(logging.DEBUG)
    global _default_handler, _error_handler

    if _default_handler is None:
        # Use ClusterFilterHandler for cluster-aware filtering
        _default_handler = ClusterFilterHandler(
            stream=sys.stdout,
            cluster_filter=_cluster_log_config['cluster_filter']
        )
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _default_handler.setLevel(logging.INFO)
        # Attach exact-level filter for cluster-tagged records
        _default_handler.addFilter(_cluster_event_level_filter)
        _root_logger.addHandler(_default_handler)

    # Add dedicated ERROR handler to stderr for immediate visibility
    if _error_handler is None:
        _error_handler = logging.StreamHandler(sys.stderr)
        _error_handler.setLevel(logging.ERROR)
        _error_handler.flush = sys.stderr.flush  # type: ignore

        # Use simple format for error messages (no cluster filtering)
        error_fmt = NewLineFormatter(
            "⚠️ ERROR %(asctime)s %(filename)s:%(lineno)d] %(message)s",
            datefmt=_SHORT_DATE_FORMAT
        )
        _error_handler.setFormatter(error_fmt)
        _root_logger.addHandler(_error_handler)

    # Choose format and date format based on configuration
    log_format = _CLUSTER_FORMAT if _cluster_log_config['enable_cluster_prefix'] else _FORMAT
    date_format = _SHORT_DATE_FORMAT if _cluster_log_config['use_short_timestamp'] else _DATE_FORMAT

    fmt = ClusterAwareFormatter(
        log_format,
        datefmt=date_format,
        cluster_filter=_cluster_log_config['cluster_filter'],
        enable_cluster_prefix=_cluster_log_config['enable_cluster_prefix']
    )
    _default_handler.setFormatter(fmt)
    # Setting this will avoid the message
    # being propagated to the parent logger.
    _root_logger.propagate = False


# The logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_setup_logger()


def init_logger(name: str):
    return logging.getLogger(name)


def set_log_level(level: str):
    """
    Set the logging level for the Frontier logger.

    Args:
        level: Logging level ('debug', 'info', 'warning', 'error')
    """
    level_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
    }

    log_level = level_map.get(level.lower(), logging.INFO)
    _default_handler.setLevel(log_level)
    _root_logger.setLevel(log_level)


def enable_detailed_logging():
    """Enable detailed logging for debugging workflow issues."""
    set_log_level('debug')


def enable_minimal_logging():
    """Enable minimal logging for production runs."""
    set_log_level('info')


def configure_cluster_logging(cluster_filter=None, enable_cluster_prefix=True, use_short_timestamp=True, cluster_event_log_level=None):
    """
    Configure cluster-aware logging settings.

    Args:
        cluster_filter: Comma-separated list of cluster types to show (e.g., 'PREFILL,DECODE', 'PREFILL,DECODE_ATTN,DECODE_FFN')
        enable_cluster_prefix: Whether to add cluster type prefix to log entries
        use_short_timestamp: Whether to use short timestamp format (HH:MM:SS)
        cluster_event_log_level: Exact level to show for cluster-tagged logs only ('DEBUG'|'INFO'|'WARNING'|'ERROR');
                                 if None, no exact-level filtering is applied.
    """
    global _cluster_log_config, _default_handler, _cluster_event_level_filter

    # Update configuration
    _cluster_log_config['cluster_filter'] = cluster_filter
    _cluster_log_config['enable_cluster_prefix'] = enable_cluster_prefix
    _cluster_log_config['use_short_timestamp'] = use_short_timestamp

    # Update exact-level filter for cluster-tagged records ONLY when a cluster filter is explicitly provided
    if _cluster_event_level_filter is not None:
        if cluster_filter:
            _cluster_event_level_filter.set_exact_level(cluster_event_log_level)
        else:
            # Disable exact-level filtering when no cluster filter is provided
            _cluster_event_level_filter.set_exact_level(None)

    # Reconfigure the handler and formatter
    if _default_handler is not None:
        # Update handler's cluster filter
        if hasattr(_default_handler, 'cluster_filter'):
            _default_handler.cluster_filter = _default_handler._parse_cluster_filter(cluster_filter)

        # Choose format and date format based on new configuration
        log_format = _CLUSTER_FORMAT if enable_cluster_prefix else _FORMAT
        date_format = _SHORT_DATE_FORMAT if use_short_timestamp else _DATE_FORMAT

        # Create new formatter with updated settings
        fmt = ClusterAwareFormatter(
            log_format,
            datefmt=date_format,
            cluster_filter=cluster_filter,
            enable_cluster_prefix=enable_cluster_prefix
        )
        _default_handler.setFormatter(fmt)


def get_cluster_logger(name: str, cluster_type: str = None):
    """
    Get a logger with cluster type information.

    Args:
        name: Logger name
        cluster_type: Cluster type (PREFILL, DECODE_ATTN, DECODE_FFN)

    Returns:
        Logger instance with cluster type context
    """
    logger = logging.getLogger(name)

    if cluster_type:
        # Create a custom LoggerAdapter that adds cluster_type to all log records
        class ClusterLoggerAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                # Add cluster_type to the log record
                if 'extra' not in kwargs:
                    kwargs['extra'] = {}
                kwargs['extra']['cluster_type'] = cluster_type
                return msg, kwargs

        return ClusterLoggerAdapter(logger, {'cluster_type': cluster_type})

    return logger
