"""
Performance monitoring and metrics collection service.

Provides metrics collection, performance tracking, and monitoring capabilities
for the GenAI PCB Design Platform.

Requirements: 13.1, 13.3, 13.5
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class OperationMetrics:
    """Metrics for a specific operation."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: str = "in_progress"
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Performance monitoring service for tracking system metrics and operation performance.
    
    Provides:
    - Operation timing and tracking
    - Metrics collection and aggregation
    - Performance statistics
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self._metrics: List[PerformanceMetric] = []
        self._operations: Dict[str, OperationMetrics] = {}
        self._operation_stats: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        logger.info("Performance monitor initialized")
    
    def start_operation(self, operation_id: str, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Start tracking an operation.
        
        Args:
            operation_id: Unique identifier for the operation
            operation_name: Name/type of the operation
            metadata: Optional metadata about the operation
        """
        with self._lock:
            self._operations[operation_id] = OperationMetrics(
                operation_name=operation_name,
                start_time=time.time(),
                metadata=metadata or {}
            )
        logger.debug(f"Started tracking operation: {operation_id} ({operation_name})")
    
    def end_operation(self, operation_id: str, status: str = "completed", metadata: Optional[Dict[str, Any]] = None) -> float:
        """
        End tracking an operation and record its duration.
        
        Args:
            operation_id: Unique identifier for the operation
            status: Final status of the operation (completed, failed, cancelled)
            metadata: Optional additional metadata
            
        Returns:
            Duration of the operation in seconds
            
        Raises:
            ValueError: If operation_id not found
        """
        with self._lock:
            if operation_id not in self._operations:
                raise ValueError(f"Operation {operation_id} not found")
            
            operation = self._operations[operation_id]
            operation.end_time = time.time()
            operation.duration = operation.end_time - operation.start_time
            operation.status = status
            
            if metadata:
                operation.metadata.update(metadata)
            
            # Record duration for statistics
            self._operation_stats[operation.operation_name].append(operation.duration)
            
            # Record metric
            self.record_metric(
                f"operation.{operation.operation_name}.duration",
                operation.duration,
                tags={"status": status}
            )
            
            logger.info(f"Operation {operation_id} ({operation.operation_name}) {status} in {operation.duration:.2f}s")
            
            return operation.duration
    
    def get_operation_status(self, operation_id: str) -> Optional[OperationMetrics]:
        """
        Get the current status of an operation.
        
        Args:
            operation_id: Unique identifier for the operation
            
        Returns:
            OperationMetrics if found, None otherwise
        """
        with self._lock:
            return self._operations.get(operation_id)
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for categorization
        """
        with self._lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self._metrics.append(metric)
        
        logger.debug(f"Recorded metric: {name}={value} {tags or {}}")
    
    def get_operation_statistics(self, operation_name: str) -> Dict[str, float]:
        """
        Get statistics for a specific operation type.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Dictionary with min, max, avg, count statistics
        """
        with self._lock:
            durations = self._operation_stats.get(operation_name, [])
            
            if not durations:
                return {
                    "count": 0,
                    "min": 0.0,
                    "max": 0.0,
                    "avg": 0.0
                }
            
            return {
                "count": len(durations),
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations)
            }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all operation types.
        
        Returns:
            Dictionary mapping operation names to their statistics
        """
        with self._lock:
            return {
                op_name: self.get_operation_statistics(op_name)
                for op_name in self._operation_stats.keys()
            }
    
    def get_recent_metrics(self, name: Optional[str] = None, limit: int = 100) -> List[PerformanceMetric]:
        """
        Get recent metrics, optionally filtered by name.
        
        Args:
            name: Optional metric name to filter by
            limit: Maximum number of metrics to return
            
        Returns:
            List of recent metrics
        """
        with self._lock:
            metrics = self._metrics
            
            if name:
                metrics = [m for m in metrics if m.name == name]
            
            return metrics[-limit:]
    
    def clear_old_metrics(self, max_age_hours: int = 24) -> int:
        """
        Clear metrics older than specified age.
        
        Args:
            max_age_hours: Maximum age of metrics to keep in hours
            
        Returns:
            Number of metrics cleared
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self._lock:
            original_count = len(self._metrics)
            self._metrics = [m for m in self._metrics if m.timestamp > cutoff_time]
            cleared_count = original_count - len(self._metrics)
        
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} old metrics")
        
        return cleared_count


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor
