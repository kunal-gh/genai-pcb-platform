"""
Unit tests for request queue service.

Tests request queuing, priority scheduling, and wait time estimation.
"""

import pytest
import time
import threading

from src.services.request_queue import (
    RequestQueue,
    RequestPriority,
    QueuedRequest,
    QueueStatus,
    get_request_queue
)


class TestRequestQueue:
    """Test suite for RequestQueue."""
    
    def test_enqueue_request(self):
        """Test enqueueing a request."""
        queue = RequestQueue(max_workers=2)
        
        def dummy_handler():
            return "result"
        
        request_id = queue.enqueue(
            "test_operation",
            dummy_handler,
            priority=RequestPriority.NORMAL
        )
        
        assert request_id is not None
        
        # Check status
        status = queue.get_status(request_id)
        assert status is not None
        assert status.status == "queued"
        assert status.position is not None
        assert status.estimated_wait_time is not None
    
    def test_priority_ordering(self):
        """Test that high priority requests are processed first."""
        queue = RequestQueue(max_workers=1)
        queue.start()
        
        results = []
        
        def handler(value):
            time.sleep(0.1)
            results.append(value)
        
        # Enqueue with different priorities
        queue.enqueue("op", handler, args=("low",), priority=RequestPriority.LOW)
        queue.enqueue("op", handler, args=("high",), priority=RequestPriority.HIGH)
        queue.enqueue("op", handler, args=("normal",), priority=RequestPriority.NORMAL)
        
        # Wait for processing
        time.sleep(0.5)
        queue.stop()
        
        # High priority should be processed first (after any already processing)
        assert "high" in results
        assert results.index("high") < results.index("normal")
        assert results.index("normal") < results.index("low")
    
    def test_request_processing(self):
        """Test that requests are processed correctly."""
        queue = RequestQueue(max_workers=2)
        queue.start()
        
        result_container = []
        
        def handler(value):
            result_container.append(value)
            return value * 2
        
        request_id = queue.enqueue("test_op", handler, args=(5,))
        
        # Wait for processing
        time.sleep(0.5)
        
        # Check status
        status = queue.get_status(request_id)
        assert status.status == "completed"
        assert status.result == 10
        assert 5 in result_container
        
        queue.stop()
    
    def test_request_failure(self):
        """Test handling of failed requests."""
        queue = RequestQueue(max_workers=2)
        queue.start()
        
        def failing_handler():
            raise ValueError("Test error")
        
        request_id = queue.enqueue("test_op", failing_handler)
        
        # Wait for processing
        time.sleep(0.5)
        
        # Check status
        status = queue.get_status(request_id)
        assert status.status == "failed"
        assert "Test error" in status.error
        
        queue.stop()
    
    def test_cancel_request(self):
        """Test cancelling a queued request."""
        queue = RequestQueue(max_workers=1)
        
        def slow_handler():
            time.sleep(1.0)
        
        # Enqueue multiple requests
        request_id1 = queue.enqueue("op", slow_handler)
        request_id2 = queue.enqueue("op", slow_handler)
        
        # Cancel second request
        cancelled = queue.cancel_request(request_id2)
        assert cancelled is True
        
        # Check status
        status = queue.get_status(request_id2)
        assert status.status == "cancelled"
    
    def test_cannot_cancel_processing_request(self):
        """Test that processing requests cannot be cancelled."""
        queue = RequestQueue(max_workers=1)
        queue.start()
        
        def slow_handler():
            time.sleep(0.5)
        
        request_id = queue.enqueue("op", slow_handler)
        
        # Wait for processing to start
        time.sleep(0.1)
        
        # Try to cancel
        cancelled = queue.cancel_request(request_id)
        assert cancelled is False
        
        queue.stop()
    
    def test_get_queue_info(self):
        """Test getting queue information."""
        queue = RequestQueue(max_workers=3)
        
        def dummy_handler():
            pass
        
        # Enqueue some requests
        queue.enqueue("op", dummy_handler)
        queue.enqueue("op", dummy_handler)
        
        info = queue.get_queue_info()
        assert info["max_workers"] == 3
        assert info["queued_requests"] >= 0
        assert "queue_size" in info
        assert "active_workers" in info
    
    def test_wait_time_estimation(self):
        """Test wait time estimation."""
        queue = RequestQueue(max_workers=2)
        queue.start()
        
        def timed_handler():
            time.sleep(0.1)
        
        # Process some requests to build history
        for _ in range(3):
            queue.enqueue("timed_op", timed_handler)
        
        time.sleep(0.5)
        
        # Enqueue new request and check estimate
        request_id = queue.enqueue("timed_op", timed_handler)
        status = queue.get_status(request_id)
        
        # Should have an estimate based on history
        assert status.estimated_wait_time is not None
        assert status.estimated_wait_time >= 0
        
        queue.stop()
    
    def test_concurrent_workers(self):
        """Test multiple workers processing concurrently."""
        queue = RequestQueue(max_workers=3)
        queue.start()
        
        processed = []
        lock = threading.Lock()
        
        def handler(value):
            time.sleep(0.1)
            with lock:
                processed.append(value)
        
        # Enqueue multiple requests
        for i in range(6):
            queue.enqueue("op", handler, args=(i,))
        
        # Wait for processing
        time.sleep(0.5)
        
        # All should be processed
        assert len(processed) == 6
        
        queue.stop()
    
    def test_status_updates_during_processing(self):
        """Test that status updates correctly during processing."""
        queue = RequestQueue(max_workers=1)
        queue.start()
        
        def handler():
            time.sleep(0.2)
        
        request_id = queue.enqueue("op", handler)
        
        # Initially queued
        status = queue.get_status(request_id)
        assert status.status == "queued"
        
        # Wait for processing to start
        time.sleep(0.1)
        status = queue.get_status(request_id)
        assert status.status == "processing"
        assert status.started_at is not None
        
        # Wait for completion
        time.sleep(0.3)
        status = queue.get_status(request_id)
        assert status.status == "completed"
        assert status.completed_at is not None
        
        queue.stop()
    
    def test_nonexistent_request_status(self):
        """Test getting status for nonexistent request."""
        queue = RequestQueue()
        
        status = queue.get_status("nonexistent")
        assert status is None


class TestGetRequestQueue:
    """Test global request queue instance."""
    
    def test_get_request_queue_singleton(self):
        """Test that get_request_queue returns singleton."""
        queue1 = get_request_queue()
        queue2 = get_request_queue()
        
        assert queue1 is queue2
    
    def test_get_request_queue_auto_starts(self):
        """Test that get_request_queue auto-starts the queue."""
        queue = get_request_queue()
        assert isinstance(queue, RequestQueue)
        assert queue._running is True


class TestQueuedRequest:
    """Test QueuedRequest dataclass."""
    
    def test_priority_ordering(self):
        """Test that requests are ordered by priority."""
        def dummy():
            pass
        
        high = QueuedRequest(
            priority=RequestPriority.HIGH.value,
            timestamp=time.time(),
            request_id="1",
            operation_name="op",
            handler=dummy
        )
        
        low = QueuedRequest(
            priority=RequestPriority.LOW.value,
            timestamp=time.time(),
            request_id="2",
            operation_name="op",
            handler=dummy
        )
        
        # High priority should be "less than" low priority (processed first)
        assert high < low


class TestQueueStatus:
    """Test QueueStatus dataclass."""
    
    def test_create_queue_status(self):
        """Test creating queue status."""
        status = QueueStatus(
            request_id="test-id",
            status="queued",
            position=5,
            estimated_wait_time=30.0
        )
        
        assert status.request_id == "test-id"
        assert status.status == "queued"
        assert status.position == 5
        assert status.estimated_wait_time == 30.0
        assert status.queued_at is not None
