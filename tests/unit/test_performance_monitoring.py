"""
Unit tests for performance monitoring service.

Tests performance tracking, metrics collection, and statistics.
"""

import pytest
import time
from datetime import datetime, timedelta

from src.services.performance_monitoring import (
    PerformanceMonitor,
    PerformanceMetric,
    OperationMetrics,
    get_performance_monitor
)


class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor."""
    
    def test_start_and_end_operation(self):
        """Test starting and ending an operation."""
        monitor = PerformanceMonitor()
        
        # Start operation
        monitor.start_operation("op1", "test_operation", {"key": "value"})
        
        # Check operation is tracked
        status = monitor.get_operation_status("op1")
        assert status is not None
        assert status.operation_name == "test_operation"
        assert status.status == "in_progress"
        assert status.metadata == {"key": "value"}
        
        # End operation
        time.sleep(0.1)  # Simulate work
        duration = monitor.end_operation("op1", "completed")
        
        # Check operation completed
        assert duration >= 0.1
        status = monitor.get_operation_status("op1")
        assert status.status == "completed"
        assert status.duration >= 0.1
    
    def test_end_nonexistent_operation(self):
        """Test ending an operation that doesn't exist."""
        monitor = PerformanceMonitor()
        
        with pytest.raises(ValueError, match="Operation .* not found"):
            monitor.end_operation("nonexistent")
    
    def test_record_metric(self):
        """Test recording metrics."""
        monitor = PerformanceMonitor()
        
        # Record metrics
        monitor.record_metric("test.metric", 42.5, {"tag": "value"})
        monitor.record_metric("test.metric", 50.0)
        
        # Get recent metrics
        metrics = monitor.get_recent_metrics("test.metric")
        assert len(metrics) == 2
        assert metrics[0].name == "test.metric"
        assert metrics[0].value == 42.5
        assert metrics[0].tags == {"tag": "value"}
        assert metrics[1].value == 50.0
    
    def test_operation_statistics(self):
        """Test operation statistics calculation."""
        monitor = PerformanceMonitor()
        
        # Record multiple operations
        for i in range(5):
            monitor.start_operation(f"op{i}", "test_op")
            time.sleep(0.01 * (i + 1))  # Variable duration
            monitor.end_operation(f"op{i}", "completed")
        
        # Get statistics
        stats = monitor.get_operation_statistics("test_op")
        assert stats["count"] == 5
        assert stats["min"] > 0
        assert stats["max"] > stats["min"]
        assert stats["avg"] > 0
    
    def test_statistics_for_nonexistent_operation(self):
        """Test statistics for operation with no data."""
        monitor = PerformanceMonitor()
        
        stats = monitor.get_operation_statistics("nonexistent")
        assert stats["count"] == 0
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
        assert stats["avg"] == 0.0
    
    def test_get_all_statistics(self):
        """Test getting statistics for all operations."""
        monitor = PerformanceMonitor()
        
        # Record operations of different types
        monitor.start_operation("op1", "type_a")
        monitor.end_operation("op1", "completed")
        
        monitor.start_operation("op2", "type_b")
        monitor.end_operation("op2", "completed")
        
        # Get all statistics
        all_stats = monitor.get_all_statistics()
        assert "type_a" in all_stats
        assert "type_b" in all_stats
        assert all_stats["type_a"]["count"] == 1
        assert all_stats["type_b"]["count"] == 1
    
    def test_get_recent_metrics_with_limit(self):
        """Test getting recent metrics with limit."""
        monitor = PerformanceMonitor()
        
        # Record many metrics
        for i in range(20):
            monitor.record_metric("test.metric", float(i))
        
        # Get limited results
        metrics = monitor.get_recent_metrics("test.metric", limit=5)
        assert len(metrics) == 5
        assert metrics[-1].value == 19.0  # Most recent
    
    def test_clear_old_metrics(self):
        """Test clearing old metrics."""
        monitor = PerformanceMonitor()
        
        # Record metrics
        for i in range(10):
            monitor.record_metric("test.metric", float(i))
        
        # Manually set some metrics to old timestamps
        old_time = datetime.now() - timedelta(hours=25)
        for i in range(5):
            monitor._metrics[i].timestamp = old_time
        
        # Clear old metrics
        cleared = monitor.clear_old_metrics(max_age_hours=24)
        assert cleared == 5
        
        # Check remaining metrics
        metrics = monitor.get_recent_metrics()
        assert len(metrics) == 5
    
    def test_operation_with_failure_status(self):
        """Test operation with failure status."""
        monitor = PerformanceMonitor()
        
        monitor.start_operation("op1", "test_op")
        duration = monitor.end_operation("op1", "failed", {"error": "test error"})
        
        status = monitor.get_operation_status("op1")
        assert status.status == "failed"
        assert status.metadata["error"] == "test error"
        assert duration > 0
    
    def test_concurrent_operations(self):
        """Test tracking multiple concurrent operations."""
        monitor = PerformanceMonitor()
        
        # Start multiple operations
        monitor.start_operation("op1", "type_a")
        monitor.start_operation("op2", "type_b")
        monitor.start_operation("op3", "type_a")
        
        # Check all are tracked
        assert monitor.get_operation_status("op1") is not None
        assert monitor.get_operation_status("op2") is not None
        assert monitor.get_operation_status("op3") is not None
        
        # End operations
        monitor.end_operation("op1", "completed")
        monitor.end_operation("op2", "completed")
        monitor.end_operation("op3", "completed")
        
        # Check statistics
        stats = monitor.get_operation_statistics("type_a")
        assert stats["count"] == 2


class TestGetPerformanceMonitor:
    """Test global performance monitor instance."""
    
    def test_get_performance_monitor_singleton(self):
        """Test that get_performance_monitor returns singleton."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        
        assert monitor1 is monitor2
    
    def test_get_performance_monitor_returns_instance(self):
        """Test that get_performance_monitor returns PerformanceMonitor."""
        monitor = get_performance_monitor()
        assert isinstance(monitor, PerformanceMonitor)


class TestPerformanceMetric:
    """Test PerformanceMetric dataclass."""
    
    def test_create_metric(self):
        """Test creating a performance metric."""
        timestamp = datetime.now()
        metric = PerformanceMetric(
            name="test.metric",
            value=42.5,
            timestamp=timestamp,
            tags={"env": "test"}
        )
        
        assert metric.name == "test.metric"
        assert metric.value == 42.5
        assert metric.timestamp == timestamp
        assert metric.tags == {"env": "test"}
    
    def test_metric_with_default_tags(self):
        """Test metric with default empty tags."""
        metric = PerformanceMetric(
            name="test.metric",
            value=10.0,
            timestamp=datetime.now()
        )
        
        assert metric.tags == {}


class TestOperationMetrics:
    """Test OperationMetrics dataclass."""
    
    def test_create_operation_metrics(self):
        """Test creating operation metrics."""
        start_time = time.time()
        metrics = OperationMetrics(
            operation_name="test_op",
            start_time=start_time,
            metadata={"key": "value"}
        )
        
        assert metrics.operation_name == "test_op"
        assert metrics.start_time == start_time
        assert metrics.end_time is None
        assert metrics.duration is None
        assert metrics.status == "in_progress"
        assert metrics.metadata == {"key": "value"}
    
    def test_operation_metrics_with_defaults(self):
        """Test operation metrics with default values."""
        metrics = OperationMetrics(
            operation_name="test_op",
            start_time=time.time()
        )
        
        assert metrics.metadata == {}
        assert metrics.status == "in_progress"
