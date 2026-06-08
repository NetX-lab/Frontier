"""
Performance profiler for Vidur simulator.

This module provides comprehensive performance monitoring and analysis capabilities
for identifying bottlenecks in the simulation execution.
"""

import time
import threading
from collections import defaultdict
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import json


@dataclass
class TimingRecord:
    """Record for a single timing measurement."""
    name: str
    start_time: float
    end_time: float
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration * 1000,
            "metadata": self.metadata
        }


@dataclass
class ComponentStats:
    """Statistics for a component."""
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    
    def update(self, duration: float):
        self.total_time += duration
        self.call_count += 1
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
    
    @property
    def avg_time(self) -> float:
        return self.total_time / self.call_count if self.call_count > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_time_ms": self.total_time * 1000,
            "call_count": self.call_count,
            "avg_time_ms": self.avg_time * 1000,
            "min_time_ms": self.min_time * 1000 if self.min_time != float('inf') else 0.0,
            "max_time_ms": self.max_time * 1000,
        }


class PerformanceProfiler:
    """
    Thread-safe performance profiler for Vidur simulator.
    
    Tracks execution time of various components and operations to identify bottlenecks.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._lock = threading.Lock()
        
        # Component-level statistics
        self._component_stats: Dict[str, ComponentStats] = defaultdict(ComponentStats)
        
        # Hierarchical timing (for nested operations)
        self._timing_stack: Dict[int, List[tuple]] = defaultdict(list)  # thread_id -> stack
        
        # Detailed records (optional, can be disabled for performance)
        self._detailed_records: List[TimingRecord] = []
        self._record_details = False
        
        # Event type breakdown
        self._event_type_stats: Dict[str, ComponentStats] = defaultdict(ComponentStats)
        
        # Scheduler level breakdown
        self._scheduler_stats: Dict[str, ComponentStats] = defaultdict(ComponentStats)
        
        # Cluster-specific stats
        self._cluster_stats: Dict[str, ComponentStats] = defaultdict(ComponentStats)
        
        # Phase timing (high-level phases)
        self._phase_times: Dict[str, float] = {}
        
        # Start time for overall profiling
        self._profiling_start_time = time.perf_counter()
        
    def enable_detailed_recording(self):
        """Enable detailed record keeping (may impact performance)."""
        self._record_details = True
    
    def disable_detailed_recording(self):
        """Disable detailed record keeping for better performance."""
        self._record_details = False
    
    @contextmanager
    def profile(self, component_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for profiling a code block.
        
        Usage:
            with profiler.profile("component_name"):
                # code to profile
        """
        if not self.enabled:
            yield
            return
        
        start_time = time.perf_counter()
        thread_id = threading.get_ident()
        
        # Push to stack for hierarchical tracking
        with self._lock:
            self._timing_stack[thread_id].append((component_name, start_time, metadata or {}))
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            with self._lock:
                # Pop from stack
                if self._timing_stack[thread_id]:
                    self._timing_stack[thread_id].pop()
                
                # Update component stats
                self._component_stats[component_name].update(duration)
                
                # Record detailed timing if enabled
                if self._record_details:
                    record = TimingRecord(
                        name=component_name,
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        metadata=metadata or {}
                    )
                    self._detailed_records.append(record)
    
    def record_event_processing(self, event_type: str, duration: float):
        """Record event processing time by event type."""
        if not self.enabled:
            return
        
        with self._lock:
            self._event_type_stats[event_type].update(duration)
    
    def record_scheduler_operation(self, scheduler_level: str, operation: str, duration: float):
        """Record scheduler operation time."""
        if not self.enabled:
            return
        
        key = f"{scheduler_level}.{operation}"
        with self._lock:
            self._scheduler_stats[key].update(duration)
    
    def record_cluster_operation(self, cluster_type: str, operation: str, duration: float):
        """Record cluster-specific operation time."""
        if not self.enabled:
            return
        
        key = f"{cluster_type}.{operation}"
        with self._lock:
            self._cluster_stats[key].update(duration)
    
    def record_phase(self, phase_name: str, duration: float):
        """Record high-level phase timing."""
        if not self.enabled:
            return
        
        with self._lock:
            self._phase_times[phase_name] = duration
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.enabled:
            return {"enabled": False}
        
        total_profiling_time = time.perf_counter() - self._profiling_start_time
        
        with self._lock:
            # Sort components by total time
            sorted_components = sorted(
                self._component_stats.items(),
                key=lambda x: x[1].total_time,
                reverse=True
            )
            
            # Sort event types by total time
            sorted_events = sorted(
                self._event_type_stats.items(),
                key=lambda x: x[1].total_time,
                reverse=True
            )
            
            # Sort scheduler operations by total time
            sorted_schedulers = sorted(
                self._scheduler_stats.items(),
                key=lambda x: x[1].total_time,
                reverse=True
            )
            
            # Sort cluster operations by total time
            sorted_clusters = sorted(
                self._cluster_stats.items(),
                key=lambda x: x[1].total_time,
                reverse=True
            )
            
            summary = {
                "total_profiling_time_s": total_profiling_time,
                "component_breakdown": {
                    name: stats.to_dict() for name, stats in sorted_components
                },
                "event_type_breakdown": {
                    name: stats.to_dict() for name, stats in sorted_events
                },
                "scheduler_breakdown": {
                    name: stats.to_dict() for name, stats in sorted_schedulers
                },
                "cluster_breakdown": {
                    name: stats.to_dict() for name, stats in sorted_clusters
                },
                "phase_times": {
                    name: time_s * 1000 for name, time_s in self._phase_times.items()
                },
                "top_10_bottlenecks": [
                    {
                        "component": name,
                        "total_time_ms": stats.total_time * 1000,
                        "percentage": (stats.total_time / total_profiling_time * 100) if total_profiling_time > 0 else 0,
                        "call_count": stats.call_count,
                        "avg_time_ms": stats.avg_time * 1000
                    }
                    for name, stats in sorted_components[:10]
                ]
            }
            
            return summary
    
    def print_summary(self):
        """Print formatted performance summary."""
        summary = self.get_summary()
        
        if not summary.get("enabled", True):
            print("Performance profiling is disabled")
            return
        
        print("\n" + "="*80)
        print("VIDUR SIMULATOR PERFORMANCE ANALYSIS")
        print("="*80)
        
        print(f"\nTotal Profiling Time: {summary['total_profiling_time_s']:.2f}s")
        
        print("\n" + "-"*80)
        print("TOP 10 BOTTLENECKS")
        print("-"*80)
        print(f"{'Component':<40} {'Time (ms)':<12} {'%':<8} {'Calls':<10} {'Avg (ms)':<10}")
        print("-"*80)
        
        for item in summary['top_10_bottlenecks']:
            print(f"{item['component']:<40} {item['total_time_ms']:>10.2f}  "
                  f"{item['percentage']:>6.2f}% {item['call_count']:>8}  {item['avg_time_ms']:>8.4f}")
        
        # Event type breakdown
        if summary['event_type_breakdown']:
            print("\n" + "-"*80)
            print("EVENT TYPE BREAKDOWN")
            print("-"*80)
            print(f"{'Event Type':<40} {'Time (ms)':<12} {'Calls':<10} {'Avg (ms)':<10}")
            print("-"*80)
            
            for event_type, stats in list(summary['event_type_breakdown'].items())[:10]:
                print(f"{event_type:<40} {stats['total_time_ms']:>10.2f}  "
                      f"{stats['call_count']:>8}  {stats['avg_time_ms']:>8.4f}")
        
        # Phase times
        if summary['phase_times']:
            print("\n" + "-"*80)
            print("PHASE TIMING")
            print("-"*80)
            for phase, time_ms in summary['phase_times'].items():
                print(f"{phase:<40} {time_ms:>10.2f} ms")
        
        print("\n" + "="*80 + "\n")
    
    def save_to_file(self, filepath: str):
        """Save performance summary to JSON file."""
        summary = self.get_summary()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Performance summary saved to: {filepath}")
    
    def reset(self):
        """Reset all profiling data."""
        with self._lock:
            self._component_stats.clear()
            self._event_type_stats.clear()
            self._scheduler_stats.clear()
            self._cluster_stats.clear()
            self._phase_times.clear()
            self._detailed_records.clear()
            self._timing_stack.clear()
            self._profiling_start_time = time.perf_counter()


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_global_profiler() -> PerformanceProfiler:
    """Get or create the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler(enabled=False)
    return _global_profiler


def enable_profiling():
    """Enable global performance profiling."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler(enabled=True)
    else:
        _global_profiler.enabled = True


def disable_profiling():
    """Disable global performance profiling."""
    global _global_profiler
    if _global_profiler is not None:
        _global_profiler.enabled = False

