"""
Unit tests for load balancer service.

Tests load balancing strategies, request distribution, and concurrent user support.
"""

import pytest
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import redis

from src.services.load_balancer import (
    LoadBalancer,
    LoadBalancingStrategy,
    RequestPriority,
    BackendServer,
    RequestContext,
    LoadBalancerMetrics
)


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock = MagicMock(spec=redis.Redis)
    mock.hset.return_value = True
    mock.hdel.return_value = True
    mock.lpush.return_value = 1
    mock.llen.return_value = 0
    mock.keys.return_value = []
    return mock


@pytest.fixture
def load_balancer(mock_redis):
    """Create a load balancer instance for testing."""
    lb = LoadBalancer(redis_client=mock_redis)
    # Give threads time to start
    time.sleep(0.1)
    return lb


def test_load_balancer_initialization(load_balancer):
    """Test load balancer initializes correctly."""
    assert load_balancer is not None
    assert load_balancer.strategy == LoadBalancingStrategy.RESOURCE_AWARE
    assert len(load_balancer.backend_servers) == 0
    assert load_balancer.metrics.total_requests == 0


def test_register_backend_server(load_balancer):
    """Test registering a backend server."""
    success = load_balancer.register_backend_server(
        server_id="server1",
        host="localhost",
        port=8001,
        weight=1.0,
        max_connections=100
    )
    
    assert success is True
    assert "server1" in load_balancer.backend_servers
    
    server = load_balancer.backend_servers["server1"]
    assert server.host == "localhost"
    assert server.port == 8001
    assert server.weight == 1.0
    assert server.max_connections == 100
    assert server.health_status == "healthy"


def test_unregister_backend_server(load_balancer):
    """Test unregistering a backend server."""
    # Register first
    load_balancer.register_backend_server("server1", "localhost", 8001)
    assert "server1" in load_balancer.backend_servers
    
    # Unregister
    success = load_balancer.unregister_backend_server("server1")
    assert success is True
    assert "server1" not in load_balancer.backend_servers


def test_round_robin_strategy(load_balancer):
    """Test round-robin load balancing strategy."""
    load_balancer.set_load_balancing_strategy(LoadBalancingStrategy.ROUND_ROBIN)
    
    # Register multiple servers
    load_balancer.register_backend_server("server1", "localhost", 8001)
    load_balancer.register_backend_server("server2", "localhost", 8002)
    load_balancer.register_backend_server("server3", "localhost", 8003)
    
    # Submit requests and check distribution
    request_context = RequestContext(
        request_id="req1",
        user_id="user1",
        request_type="test",
        priority=RequestPriority.NORMAL,
        estimated_duration=60,
        resource_requirements={"cpu": 1.0, "memory": 1.0}
    )
    
    # Get server selections (should cycle through servers)
    servers = []
    for i in range(6):
        server_id = load_balancer._select_backend_server(request_context)
        servers.append(server_id)
    
    # Should cycle through all servers
    assert "server1" in servers
    assert "server2" in servers
    assert "server3" in servers


def test_least_connections_strategy(load_balancer):
    """Test least connections load balancing strategy."""
    load_balancer.set_load_balancing_strategy(LoadBalancingStrategy.LEAST_CONNECTIONS)
    
    # Register servers with different connection counts
    load_balancer.register_backend_server("server1", "localhost", 8001)
    load_balancer.register_backend_server("server2", "localhost", 8002)
    
    # Set different connection counts
    load_balancer.backend_servers["server1"].current_connections = 5
    load_balancer.backend_servers["server2"].current_connections = 2
    
    request_context = RequestContext(
        request_id="req1",
        user_id="user1",
        request_type="test",
        priority=RequestPriority.NORMAL,
        estimated_duration=60,
        resource_requirements={"cpu": 1.0, "memory": 1.0}
    )
    
    # Should select server with least connections
    server_id = load_balancer._select_backend_server(request_context)
    assert server_id == "server2"


def test_resource_aware_strategy(load_balancer):
    """Test resource-aware load balancing strategy."""
    load_balancer.set_load_balancing_strategy(LoadBalancingStrategy.RESOURCE_AWARE)
    
    # Register servers with different resource usage
    load_balancer.register_backend_server("server1", "localhost", 8001, max_connections=100)
    load_balancer.register_backend_server("server2", "localhost", 8002, max_connections=100)
    
    # Set different resource usage
    load_balancer.backend_servers["server1"].current_connections = 80
    load_balancer.backend_servers["server1"].cpu_usage = 90.0
    load_balancer.backend_servers["server2"].current_connections = 30
    load_balancer.backend_servers["server2"].cpu_usage = 40.0
    
    request_context = RequestContext(
        request_id="req1",
        user_id="user1",
        request_type="test",
        priority=RequestPriority.NORMAL,
        estimated_duration=60,
        resource_requirements={"cpu": 1.0, "memory": 1.0}
    )
    
    # Should select server with lower resource usage
    server_id = load_balancer._select_backend_server(request_context)
    assert server_id == "server2"


def test_submit_request_immediate_assignment(load_balancer):
    """Test submitting a request with immediate assignment."""
    # Register a server
    load_balancer.register_backend_server("server1", "localhost", 8001, max_connections=100)
    
    # Submit request
    success = load_balancer.submit_request(
        request_id="req1",
        user_id="user1",
        request_type="test",
        priority=RequestPriority.NORMAL,
        estimated_duration=60,
        resource_requirements={"cpu": 1.0, "memory": 1.0}
    )
    
    assert success is True
    assert "req1" in load_balancer.active_requests
    assert load_balancer.active_requests["req1"].assigned_server == "server1"
    assert load_balancer.backend_servers["server1"].current_connections == 1


def test_submit_request_queuing(load_balancer):
    """Test submitting a request when servers are at capacity."""
    # Register a server with low capacity
    load_balancer.register_backend_server("server1", "localhost", 8001, max_connections=1)
    
    # Fill capacity
    load_balancer.submit_request("req1", "user1", "test")
    
    # Submit another request (should be queued)
    success = load_balancer.submit_request("req2", "user2", "test")
    
    assert success is True
    assert len(load_balancer.request_queue) == 1
    assert load_balancer.metrics.queue_length == 1


def test_request_priority_handling(load_balancer):
    """Test that high priority requests are handled appropriately."""
    load_balancer.register_backend_server("server1", "localhost", 8001)
    
    # Submit requests with different priorities
    load_balancer.submit_request(
        "req1", "user1", "test",
        priority=RequestPriority.LOW
    )
    load_balancer.submit_request(
        "req2", "user2", "test",
        priority=RequestPriority.HIGH
    )
    load_balancer.submit_request(
        "req3", "user3", "test",
        priority=RequestPriority.CRITICAL
    )
    
    # All should be accepted
    assert load_balancer.metrics.total_requests == 3


def test_consistent_hash_strategy(load_balancer):
    """Test consistent hashing strategy for user affinity."""
    load_balancer.set_load_balancing_strategy(LoadBalancingStrategy.CONSISTENT_HASH)
    
    # Register servers
    load_balancer.register_backend_server("server1", "localhost", 8001)
    load_balancer.register_backend_server("server2", "localhost", 8002)
    
    # Same user should get same server (using same request_id for consistency)
    request1 = RequestContext(
        request_id="req1",
        user_id="user1",
        request_type="test",
        priority=RequestPriority.NORMAL,
        estimated_duration=60,
        resource_requirements={"cpu": 1.0, "memory": 1.0}
    )
    request2 = RequestContext(
        request_id="req1",  # Same request_id to ensure same hash
        user_id="user1",
        request_type="test",
        priority=RequestPriority.NORMAL,
        estimated_duration=60,
        resource_requirements={"cpu": 1.0, "memory": 1.0}
    )
    
    server1 = load_balancer._select_backend_server(request1)
    server2 = load_balancer._select_backend_server(request2)
    
    # Should get same server for same user+request combination
    assert server1 == server2
    
    # Different user should potentially get different server
    request3 = RequestContext(
        request_id="req1",
        user_id="user2",  # Different user
        request_type="test",
        priority=RequestPriority.NORMAL,
        estimated_duration=60,
        resource_requirements={"cpu": 1.0, "memory": 1.0}
    )
    
    server3 = load_balancer._select_backend_server(request3)
    # server3 may or may not be different from server1, but should be consistent
    assert server3 in ["server1", "server2"]


def test_get_metrics(load_balancer):
    """Test getting load balancer metrics."""
    load_balancer.register_backend_server("server1", "localhost", 8001)
    load_balancer.submit_request("req1", "user1", "test")
    
    metrics = load_balancer.get_metrics()
    
    assert isinstance(metrics, LoadBalancerMetrics)
    assert metrics.total_requests >= 1
    assert metrics.current_connections >= 0


def test_unhealthy_server_exclusion(load_balancer):
    """Test that unhealthy servers are excluded from selection."""
    load_balancer.register_backend_server("server1", "localhost", 8001)
    load_balancer.register_backend_server("server2", "localhost", 8002)
    
    # Mark server1 as unhealthy
    load_balancer.backend_servers["server1"].health_status = "unhealthy"
    
    request_context = RequestContext(
        request_id="req1",
        user_id="user1",
        request_type="test",
        priority=RequestPriority.NORMAL,
        estimated_duration=60,
        resource_requirements={"cpu": 1.0, "memory": 1.0}
    )
    
    # Should only select healthy server
    server_id = load_balancer._select_backend_server(request_context)
    assert server_id == "server2"


def test_weighted_round_robin_strategy(load_balancer):
    """Test weighted round-robin strategy."""
    load_balancer.set_load_balancing_strategy(LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN)
    
    # Register servers with different weights
    load_balancer.register_backend_server("server1", "localhost", 8001, weight=1.0)
    load_balancer.register_backend_server("server2", "localhost", 8002, weight=3.0)
    
    request_context = RequestContext(
        request_id="req1",
        user_id="user1",
        request_type="test",
        priority=RequestPriority.NORMAL,
        estimated_duration=60,
        resource_requirements={"cpu": 1.0, "memory": 1.0}
    )
    
    # Get multiple selections
    selections = []
    for i in range(100):
        server_id = load_balancer._select_backend_server(request_context)
        selections.append(server_id)
    
    # Server2 should be selected more often due to higher weight
    server2_count = selections.count("server2")
    server1_count = selections.count("server1")
    
    # With weight 3:1, server2 should be selected roughly 3x more
    assert server2_count > server1_count


def test_shutdown(load_balancer):
    """Test graceful shutdown of load balancer."""
    load_balancer.shutdown()
    
    # Threads should stop
    assert load_balancer._health_check_active is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
