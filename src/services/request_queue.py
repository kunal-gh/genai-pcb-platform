"""
Request queuing service with wait time estimates.

Provides request queuing, priority scheduling, and wait time estimation
for the GenAI PCB Design Platform.

Requirements: 13.3
"""

import time
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
from queue import PriorityQueue, Empty
import uuid

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels."""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


@dataclass(order=True)
class QueuedRequest:
    """A queued request with priority."""
    priority: int
    timestamp: float = field(compare=False)
    request_id: str = field(compare=False)
    operation_name: str = field(compare=False)
    handler: Callable = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: Dict[str, Any] = field(default_factory=dict, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass
class QueueStatus:
    """Status of a queued request."""
    request_id: str
    status: str  # queued, processing, completed, failed, cancelled
    position: Optional[int] = None
    estimated_wait_time: Optional[float] = None
    queued_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class RequestQueue:
    """
    Request queue with priority scheduling and wait time estimation.
    
    Provides:
    - Priority-based request queuing
    - Wait time estimation based on historical data
    - Request status tracking
    - Concurrent request processing
    """
    
    def __init__(self, max_workers: int = 5):
        """
        Initialize request queue.
        
        Args:
            max_workers: Maximum number of concurrent workers
        """
        self._queue: PriorityQueue = PriorityQueue()
        self._max_workers = max_workers
        self._active_workers = 0
        self._request_status: Dict[str, QueueStatus] = {}
        self._processing_times: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._workers: List[threading.Thread] = []
        self._running = False
        
        logger.info(f"Request queue initialized with {max_workers} workers")
    
    def start(self) -> None:
        """Start the queue workers."""
        if self._running:
            logger.warning("Request queue already running")
            return
        
        self._running = True
        
        # Start worker threads
        for i in range(self._max_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"QueueWorker-{i}", daemon=True)
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"Started {self._max_workers} queue workers")
    
    def stop(self) -> None:
        """Stop the queue workers."""
        self._running = False
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=5.0)
        
        self._workers.clear()
        logger.info("Stopped queue workers")
    
    def enqueue(
        self,
        operation_name: str,
        handler: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Enqueue a request for processing.
        
        Args:
            operation_name: Name of the operation
            handler: Callable to execute
            args: Positional arguments for handler
            kwargs: Keyword arguments for handler
            priority: Request priority
            metadata: Optional metadata
            
        Returns:
            Request ID for tracking
        """
        request_id = str(uuid.uuid4())
        timestamp = time.time()
        
        request = QueuedRequest(
            priority=priority.value,
            timestamp=timestamp,
            request_id=request_id,
            operation_name=operation_name,
            handler=handler,
            args=args,
            kwargs=kwargs or {},
            metadata=metadata or {}
        )
        
        # Calculate queue position and wait time
        queue_size = self._queue.qsize()
        position = queue_size + 1
        estimated_wait = self._estimate_wait_time(operation_name, position)
        
        # Create status entry
        status = QueueStatus(
            request_id=request_id,
            status="queued",
            position=position,
            estimated_wait_time=estimated_wait
        )
        
        with self._lock:
            self._request_status[request_id] = status
        
        # Add to queue
        self._queue.put(request)
        
        logger.info(
            f"Enqueued request {request_id} ({operation_name}) "
            f"with priority {priority.name}, position {position}, "
            f"estimated wait {estimated_wait:.1f}s"
        )
        
        return request_id
    
    def get_status(self, request_id: str) -> Optional[QueueStatus]:
        """
        Get the status of a queued request.
        
        Args:
            request_id: Request ID
            
        Returns:
            QueueStatus if found, None otherwise
        """
        with self._lock:
            status = self._request_status.get(request_id)
            
            # Update position for queued requests
            if status and status.status == "queued":
                status.position = self._calculate_position(request_id)
                status.estimated_wait_time = self._estimate_wait_time(
                    self._get_operation_name(request_id),
                    status.position or 0
                )
            
            return status
    
    def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a queued request.
        
        Args:
            request_id: Request ID
            
        Returns:
            True if cancelled, False if not found or already processing
        """
        with self._lock:
            status = self._request_status.get(request_id)
            
            if not status:
                return False
            
            if status.status != "queued":
                logger.warning(f"Cannot cancel request {request_id} with status {status.status}")
                return False
            
            status.status = "cancelled"
            status.completed_at = datetime.now()
            
            logger.info(f"Cancelled request {request_id}")
            return True
    
    def get_queue_info(self) -> Dict[str, Any]:
        """
        Get information about the queue.
        
        Returns:
            Dictionary with queue statistics
        """
        with self._lock:
            queued_count = sum(1 for s in self._request_status.values() if s.status == "queued")
            processing_count = sum(1 for s in self._request_status.values() if s.status == "processing")
            completed_count = sum(1 for s in self._request_status.values() if s.status == "completed")
            failed_count = sum(1 for s in self._request_status.values() if s.status == "failed")
            
            return {
                "queue_size": self._queue.qsize(),
                "active_workers": self._active_workers,
                "max_workers": self._max_workers,
                "queued_requests": queued_count,
                "processing_requests": processing_count,
                "completed_requests": completed_count,
                "failed_requests": failed_count
            }
    
    def _worker_loop(self) -> None:
        """Worker thread loop for processing requests."""
        while self._running:
            try:
                # Get request from queue with timeout
                request = self._queue.get(timeout=1.0)
                
                # Update status
                with self._lock:
                    self._active_workers += 1
                    status = self._request_status.get(request.request_id)
                    if status:
                        if status.status == "cancelled":
                            # Skip cancelled requests
                            self._active_workers -= 1
                            self._queue.task_done()
                            continue
                        
                        status.status = "processing"
                        status.started_at = datetime.now()
                        status.position = None
                
                # Process request
                start_time = time.time()
                try:
                    result = request.handler(*request.args, **request.kwargs)
                    
                    # Update status - success
                    with self._lock:
                        if status:
                            status.status = "completed"
                            status.completed_at = datetime.now()
                            status.result = result
                    
                    logger.info(f"Completed request {request.request_id} ({request.operation_name})")
                    
                except Exception as e:
                    # Update status - failure
                    with self._lock:
                        if status:
                            status.status = "failed"
                            status.completed_at = datetime.now()
                            status.error = str(e)
                    
                    logger.error(f"Failed request {request.request_id} ({request.operation_name}): {e}")
                
                finally:
                    # Record processing time
                    duration = time.time() - start_time
                    with self._lock:
                        if request.operation_name not in self._processing_times:
                            self._processing_times[request.operation_name] = []
                        self._processing_times[request.operation_name].append(duration)
                        
                        # Keep only recent times (last 100)
                        if len(self._processing_times[request.operation_name]) > 100:
                            self._processing_times[request.operation_name] = \
                                self._processing_times[request.operation_name][-100:]
                        
                        self._active_workers -= 1
                    
                    self._queue.task_done()
                    
            except Empty:
                # Timeout, continue loop
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
    
    def _estimate_wait_time(self, operation_name: str, position: int) -> float:
        """
        Estimate wait time for a request.
        
        Args:
            operation_name: Name of the operation
            position: Position in queue
            
        Returns:
            Estimated wait time in seconds
        """
        with self._lock:
            # Get average processing time for this operation
            times = self._processing_times.get(operation_name, [])
            
            if not times:
                # Default estimate if no historical data
                avg_time = 30.0
            else:
                avg_time = sum(times) / len(times)
            
            # Estimate based on position and available workers
            available_workers = max(1, self._max_workers - self._active_workers)
            estimated_wait = (position / available_workers) * avg_time
            
            return estimated_wait
    
    def _calculate_position(self, request_id: str) -> int:
        """Calculate current position in queue for a request."""
        # This is approximate since PriorityQueue doesn't expose position
        # In production, consider using a custom queue implementation
        return self._queue.qsize()
    
    def _get_operation_name(self, request_id: str) -> str:
        """Get operation name for a request."""
        with self._lock:
            status = self._request_status.get(request_id)
            return status.metadata.get("operation_name", "unknown") if status else "unknown"


# Global request queue instance
_request_queue: Optional[RequestQueue] = None


def get_request_queue() -> RequestQueue:
    """Get the global request queue instance."""
    global _request_queue
    if _request_queue is None:
        _request_queue = RequestQueue()
        _request_queue.start()
    return _request_queue
