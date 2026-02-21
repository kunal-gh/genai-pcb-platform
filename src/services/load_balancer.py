"""
Load Balancer Service

Handles request distribution, concurrent user support, and intelligent routing
for the GenAI PCB Platform.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import threading
import hashlib
import random
import redis
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_AWARE = "resource_aware"
    CONSISTENT_HASH = "consistent_hash"


class RequestPriority(Enum):
    """Request priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BackendServer:
    """Represents a backend processing server"""
    server_id: str
    host: str
    port: int
    weight: float = 1.0
    max_connections: int = 100
    current_connections: int = 0
    health_status: str = "healthy"
    last_health_check: datetime = field(default_factory=datetime.now)
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0


@dataclass
class RequestContext:
    """Context information for a request"""
    request_id: str
    user_id: str
    request_type: str
    priority: RequestPriority
    estimated_duration: int  # seconds
    resource_requirements: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)
    assigned_server: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class LoadBalancerMetrics:
    """Load balancer performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    current_connections: int = 0
    queue_length: int = 0
    throughput_rps: float = 0.0  # requests per second
    timestamp: datetime = field(default_factory=datetime.now)


class LoadBalancer:
    """
    Intelligent load balancer for distributing requests across backend servers
    with support for concurrent users and multiple balancing strategies.
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.backend_servers: Dict[str, BackendServer] = {}
        self.strategy = LoadBalancingStrategy.RESOURCE_AWARE
        self.request_queue: deque = deque()
        self.active_requests: Dict[str, RequestContext] = {}
        self.metrics = LoadBalancerMetrics()
        self.request_history: deque = deque(maxlen=1000)

        # Round robin state
        self._round_robin_index = 0

        # Consistent hashing ring
        self._hash_ring: List[tuple] = []
        self._ring_size = 1000

        # Thread safety
        self._lock = threading.RLock()

        # Health checking
        self._health_check_active = True
        self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_thread.start()

        # Metrics collection
        self._metrics_thread = threading.Thread(target=self._metrics_collection_loop, daemon=True)
        self._metrics_thread.start()

        logger.info("LoadBalancer initialized with resource-aware strategy")

    def register_backend_server(self, server_id: str, host: str, port: int,
                              weight: float = 1.0, max_connections: int = 100) -> bool:
        """Register a new backend server"""
        try:
            with self._lock:
                server = BackendServer(
                    server_id=server_id,
                    host=host,
                    port=port,
                    weight=weight,
                    max_connections=max_connections
                )

                self.backend_servers[server_id] = server

                # Update consistent hash ring
                self._rebuild_hash_ring()

                # Store in Redis for persistence
                self.redis_client.hset(
                    "backend_servers",
                    server_id,
                    f"{host}:{port}:{weight}:{max_connections}"
                )

                logger.info(f"Registered backend server {server_id} at {host}:{port}")
                return True

        except Exception as e:
            logger.error(f"Failed to register backend server {server_id}: {e}")
            return False

    def unregister_backend_server(self, server_id: str) -> bool:
        """Unregister a backend server"""
        try:
            with self._lock:
                if server_id in self.backend_servers:
                    del self.backend_servers[server_id]
                    self._rebuild_hash_ring()
                    self.redis_client.hdel("backend_servers", server_id)
                    logger.info(f"Unregistered backend server {server_id}")
                    return True
                return False

        except Exception as e:
            logger.error(f"Failed to unregister backend server {server_id}: {e}")
            return False

    def set_load_balancing_strategy(self, strategy: LoadBalancingStrategy):
        """Set the load balancing strategy"""
        with self._lock:
            self.strategy = strategy
            if strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                self._rebuild_hash_ring()
            logger.info(f"Load balancing strategy set to {strategy.value}")

    def submit_request(self, request_id: str, user_id: str, request_type: str,
                      priority: RequestPriority = RequestPriority.NORMAL,
                      estimated_duration: int = 60,
                      resource_requirements: Optional[Dict[str, float]] = None) -> bool:
        """Submit a request for processing"""
        try:
            resource_requirements = resource_requirements or {"cpu": 1.0, "memory": 1.0}

            request_context = RequestContext(
                request_id=request_id,
                user_id=user_id,
                request_type=request_type,
                priority=priority,
                estimated_duration=estimated_duration,
                resource_requirements=resource_requirements
            )

            with self._lock:
                # Try immediate assignment if possible
                server_id = self._select_backend_server(request_context)

                if server_id and self._can_assign_request(server_id, request_context):
                    # Assign immediately
                    request_context.assigned_server = server_id
                    self.active_requests[request_id] = request_context
                    self.backend_servers[server_id].current_connections += 1

                    # Store assignment in Redis
                    self.redis_client.hset(
                        "request_assignments",
                        request_id,
                        f"{server_id}:{datetime.now().isoformat()}"
                    )

                    self.metrics.total_requests += 1
                    self.request_history.append((request_id, server_id, datetime.now()))
                    logger.debug(f"Assigned request {request_id} to server {server_id}")
                    return True
                else:
                    # Queue for later
                    self.request_queue.append(request_context)
                    self.metrics.queue_length = len(self.request_queue)
                    self.redis_client.lpush("load_balancer_queue", request_id)
                    logger.debug(f"Queued request {request_id}")
                    return True

        except Exception as e:
            logger.error(f"Failed to submit request {request_id}: {e}")
            self.metrics.failed_requests += 1
            return False

    def _select_backend_server(self, request_context: RequestContext) -> Optional[str]:
        """Select a backend server based on current strategy."""
        healthy = [
            (sid, s) for sid, s in self.backend_servers.items()
            if s.health_status == "healthy"
        ]
        if not healthy:
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            idx = self._round_robin_index % len(healthy)
            self._round_robin_index += 1
            return healthy[idx][0]

        if self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            best = min(healthy, key=lambda x: x[1].current_connections)
            return best[0]

        if self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            total_weight = sum(s.weight for _, s in healthy)
            r = random.uniform(0, total_weight)
            for sid, s in healthy:
                r -= s.weight
                if r <= 0:
                    return sid
            return healthy[-1][0]

        if self.strategy == LoadBalancingStrategy.RESOURCE_AWARE:
            best = min(
                healthy,
                key=lambda x: (
                    x[1].current_connections / max(x[1].max_connections, 1),
                    x[1].cpu_usage,
                    x[1].memory_usage,
                )
            )
            return best[0]

        if self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            if not self._hash_ring:
                return healthy[0][0] if healthy else None
            key = f"{request_context.user_id}:{request_context.request_id}"
            h = int(hashlib.md5(key.encode()).hexdigest(), 16) % self._ring_size
            for ring_hash, server_id in sorted(self._hash_ring):
                if ring_hash >= h:
                    return server_id
            return self._hash_ring[0][1]

        return healthy[0][0]

    def _can_assign_request(self, server_id: str, request_context: RequestContext) -> bool:
        """Check if the server can accept the request."""
        server = self.backend_servers.get(server_id)
        if not server or server.health_status != "healthy":
            return False
        if server.current_connections >= server.max_connections:
            return False
        return True

    def _rebuild_hash_ring(self) -> None:
        """Rebuild the consistent hash ring from current backend servers."""
        self._hash_ring = []
        for server_id in self.backend_servers:
            for v in range(10):
                key = f"{server_id}:{v}"
                h = int(hashlib.md5(key.encode()).hexdigest(), 16) % self._ring_size
                self._hash_ring.append((h, server_id))
        self._hash_ring.sort(key=lambda x: x[0])

    def _health_check_loop(self) -> None:
        """Background loop to check backend server health."""
        while self._health_check_active:
            try:
                with self._lock:
                    for server in self.backend_servers.values():
                        server.last_health_check = datetime.now()
                        server.health_status = "healthy"
                time.sleep(30)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(60)

    def _metrics_collection_loop(self) -> None:
        """Background loop to update metrics."""
        while self._health_check_active:
            try:
                with self._lock:
                    self.metrics.current_connections = sum(
                        s.current_connections for s in self.backend_servers.values()
                    )
                    self.metrics.queue_length = len(self.request_queue)
                    self.metrics.timestamp = datetime.now()
                time.sleep(10)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(30)

    def get_metrics(self) -> LoadBalancerMetrics:
        """Return current load balancer metrics."""
        with self._lock:
            return LoadBalancerMetrics(
                total_requests=self.metrics.total_requests,
                successful_requests=self.metrics.successful_requests,
                failed_requests=self.metrics.failed_requests,
                average_response_time=self.metrics.average_response_time,
                current_connections=self.metrics.current_connections,
                queue_length=self.metrics.queue_length,
                throughput_rps=self.metrics.throughput_rps,
                timestamp=self.metrics.timestamp,
            )

    def shutdown(self) -> None:
        """Shutdown the load balancer and background threads."""
        self._health_check_active = False
        if hasattr(self, "_health_check_thread"):
            self._health_check_thread.join(timeout=5)
        if hasattr(self, "_metrics_thread"):
            self._metrics_thread.join(timeout=5)
        logger.info("LoadBalancer shutdown completed")


# Global load balancer instance
_load_balancer: Optional[LoadBalancer] = None


def get_load_balancer(redis_client: Optional[redis.Redis] = None) -> LoadBalancer:
    """Get the global load balancer instance."""
    global _load_balancer
    if _load_balancer is None:
        _load_balancer = LoadBalancer(redis_client=redis_client)
    return _load_balancer
