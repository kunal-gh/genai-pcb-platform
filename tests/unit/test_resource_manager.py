"""
Unit tests for resource manager service.

Tests auto-scaling, resource allocation, and worker node management.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import redis

from src.services.resource_manager import (
    ResourceManager,
    ResourceType,
    ScalingAction,
    ResourceMetrics,
    ScalingPolicy,
    WorkerNode
)


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock = MagicMock(spec=redis.Redis)
    mock.hset.return_value = True
    mock.hdel.return_value = True
    mock.llen.return_value = 0
    mock.keys.return_value = []
    return mock


@pytest.fixture
def resource_manager(mock_redis):
    """Create a resource manager instance for testing."""
    with patch('src.services.resource_manager.config'):
        rm = ResourceManager(redis_client=mock_redis)
        # Give monitoring thread time to start
        time.sleep(0.1)
        return rm


def test_resource_manager_initialization(resource_manager):
    """Test resource manager initializes correctly."""
    assert resource_manager is not None
    assert isinstance(resource_manager.scaling_policy, ScalingPolicy)
    assert len(resource_manager.worker_nodes) == 0


def test_register_worker_node(resource_manager):
    """Test registering a worker node."""
    success = resource_manager.register_worker_node(
        node_id="worker1",
        cpu_capacity=100.0,
        memory_capacity=16.0,
        gpu_capacity=1.0
    )
    
    assert success is True
    assert "worker1" in resource_manager.worker_nodes
    
    worker = resource_manager.worker_nodes["worker1"]
    assert worker.status == "active"
    assert worker.cpu_capacity == 100.0
    assert worker.memory_capacity == 16.0
    assert worker.gpu_capacity == 1.0


def test_get_resource_metrics(resource_manager):
    """Test getting current resource metrics."""
    metrics = resource_manager.get_resource_metrics()
    
    assert isinstance(metrics, ResourceMetrics)
    assert metrics.cpu_percent >= 0
    assert metrics.memory_percent >= 0
    assert metrics.storage_percent >= 0
    assert isinstance(metrics.timestamp, datetime)


def test_determine_scaling_action_maintain(resource_manager):
    """Test scaling decision when resources are balanced."""
    # Add some metrics with moderate usage
    for i in range(5):
        metrics = ResourceMetrics(
            cpu_percent=50.0,
            memory_percent=60.0,
            active_requests=5,
            queue_length=2
        )
        resource_manager.resource_metrics.append(metrics)
    
    # Wait for cooldown
    resource_manager.last_scaling_action = datetime.now() - timedelta(seconds=400)
    
    action = resource_manager.determine_scaling_action(metrics)
    assert action == ScalingAction.MAINTAIN


def test_determine_scaling_action_scale_up(resource_manager):
    """Test scaling up when resources are high."""
    # Add metrics with high usage
    for i in range(5):
        metrics = ResourceMetrics(
            cpu_percent=90.0,
            memory_percent=85.0,
            active_requests=20,
            queue_length=15
        )
        resource_manager.resource_metrics.append(metrics)
    
    # Wait for cooldown
    resource_manager.last_scaling_action = datetime.now() - timedelta(seconds=400)
    
    action = resource_manager.determine_scaling_action(metrics)
    assert action == ScalingAction.SCALE_UP


def test_determine_scaling_action_scale_down(resource_manager):
    """Test scaling down when resources are low."""
    # Add metrics with low usage
    for i in range(5):
        metrics = ResourceMetrics(
            cpu_percent=15.0,
            memory_percent=20.0,
            active_requests=1,
            queue_length=0
        )
        resource_manager.resource_metrics.append(metrics)
    
    # Wait for cooldown
    resource_manager.last_scaling_action = datetime.now() - timedelta(seconds=400)
    
    action = resource_manager.determine_scaling_action(metrics)
    assert action == ScalingAction.SCALE_DOWN


def test_cooldown_period(resource_manager):
    """Test that cooldown period prevents rapid scaling."""
    # Add metrics that would trigger scale up
    for i in range(5):
        metrics = ResourceMetrics(
            cpu_percent=95.0,
            memory_percent=90.0,
            active_requests=25,
            queue_length=20
        )
        resource_manager.resource_metrics.append(metrics)
    
    # Recent scaling action (within cooldown)
    resource_manager.last_scaling_action = datetime.now() - timedelta(seconds=60)
    
    action = resource_manager.determine_scaling_action(metrics)
    assert action == ScalingAction.MAINTAIN


def test_assign_task_to_worker(resource_manager):
    """Test assigning a task to a worker node."""
    # Register workers
    resource_manager.register_worker_node("worker1", 100.0, 16.0)
    resource_manager.register_worker_node("worker2", 100.0, 16.0)
    
    # Assign task
    worker_id = resource_manager.assign_task_to_worker(
        task_id="task1",
        task_type="design_processing",
        resource_requirements={"cpu": 10.0, "memory": 2.0}
    )
    
    assert worker_id is not None
    assert worker_id in ["worker1", "worker2"]
    
    worker = resource_manager.worker_nodes[worker_id]
    assert "task1" in worker.active_tasks
    assert worker.current_load >= 10.0


def test_assign_task_load_balancing(resource_manager):
    """Test that tasks are assigned to least loaded worker."""
    # Register workers with different loads
    resource_manager.register_worker_node("worker1", 100.0, 16.0)
    resource_manager.register_worker_node("worker2", 100.0, 16.0)
    
    # Load worker1
    resource_manager.worker_nodes["worker1"].current_load = 80.0
    resource_manager.worker_nodes["worker1"].active_tasks = ["task1", "task2"]
    
    # Assign new task
    worker_id = resource_manager.assign_task_to_worker(
        task_id="task3",
        task_type="design_processing",
        resource_requirements={"cpu": 10.0, "memory": 2.0}
    )
    
    # Should assign to less loaded worker2
    assert worker_id == "worker2"


def test_assign_task_insufficient_resources(resource_manager):
    """Test task assignment when no worker has sufficient resources."""
    # Register worker with limited capacity
    resource_manager.register_worker_node("worker1", 10.0, 2.0)
    
    # Try to assign task requiring more resources
    worker_id = resource_manager.assign_task_to_worker(
        task_id="task1",
        task_type="design_processing",
        resource_requirements={"cpu": 20.0, "memory": 4.0}
    )
    
    # Should return None
    assert worker_id is None


def test_complete_task(resource_manager):
    """Test completing a task and freeing resources."""
    # Register worker and assign task
    resource_manager.register_worker_node("worker1", 100.0, 16.0)
    worker_id = resource_manager.assign_task_to_worker(
        task_id="task1",
        task_type="design_processing",
        resource_requirements={"cpu": 10.0, "memory": 2.0}
    )
    
    initial_load = resource_manager.worker_nodes[worker_id].current_load
    
    # Complete task
    success = resource_manager.complete_task("task1", worker_id)
    
    assert success is True
    assert "task1" not in resource_manager.worker_nodes[worker_id].active_tasks
    assert resource_manager.worker_nodes[worker_id].current_load < initial_load


def test_get_cluster_status(resource_manager):
    """Test getting cluster status."""
    # Register workers
    resource_manager.register_worker_node("worker1", 100.0, 16.0)
    resource_manager.register_worker_node("worker2", 100.0, 16.0)
    
    status = resource_manager.get_cluster_status()
    
    assert "timestamp" in status
    assert "resource_metrics" in status
    assert "cluster_metrics" in status
    assert "scaling_policy" in status
    
    assert status["cluster_metrics"]["total_workers"] == 2
    assert status["cluster_metrics"]["active_workers"] == 2


def test_scale_local_workers_up(resource_manager):
    """Test scaling up local workers."""
    # Register initial workers
    resource_manager.register_worker_node("worker1", 100.0, 16.0)
    
    # Scale up
    success = resource_manager._scale_local_workers(3)
    
    assert success is True
    active_workers = [w for w in resource_manager.worker_nodes.values() if w.status == "active"]
    assert len(active_workers) == 3


def test_scale_local_workers_down(resource_manager):
    """Test scaling down local workers."""
    # Register workers
    resource_manager.register_worker_node("worker1", 100.0, 16.0)
    resource_manager.register_worker_node("worker2", 100.0, 16.0)
    resource_manager.register_worker_node("worker3", 100.0, 16.0)
    
    # Scale down
    success = resource_manager._scale_local_workers(1)
    
    assert success is True
    active_workers = [w for w in resource_manager.worker_nodes.values() if w.status == "active"]
    # Should have at least 1 active worker
    assert len(active_workers) >= 1


def test_execute_scaling_action_scale_up(resource_manager):
    """Test executing scale up action."""
    # Register initial worker
    resource_manager.register_worker_node("worker1", 100.0, 16.0)
    
    # Set policy
    resource_manager.scaling_policy.max_replicas = 5
    resource_manager.scaling_policy.scale_up_factor = 2.0
    
    # Disable Kubernetes client to use local scaling
    resource_manager.k8s_client = None
    
    # Execute scale up
    success = resource_manager.execute_scaling_action(ScalingAction.SCALE_UP)
    
    assert success is True
    assert resource_manager.last_scaling_action is not None


def test_execute_scaling_action_scale_down(resource_manager):
    """Test executing scale down action."""
    # Register workers
    for i in range(4):
        resource_manager.register_worker_node(f"worker{i}", 100.0, 16.0)
    
    # Set policy
    resource_manager.scaling_policy.min_replicas = 1
    resource_manager.scaling_policy.scale_down_factor = 0.5
    
    # Disable Kubernetes client to use local scaling
    resource_manager.k8s_client = None
    
    # Execute scale down
    success = resource_manager.execute_scaling_action(ScalingAction.SCALE_DOWN)
    
    assert success is True


def test_execute_scaling_action_maintain(resource_manager):
    """Test executing maintain action (no-op)."""
    resource_manager.register_worker_node("worker1", 100.0, 16.0)
    
    success = resource_manager.execute_scaling_action(ScalingAction.MAINTAIN)
    
    assert success is True


def test_scaling_policy_limits(resource_manager):
    """Test that scaling respects min/max replica limits."""
    # Set strict limits
    resource_manager.scaling_policy.min_replicas = 2
    resource_manager.scaling_policy.max_replicas = 5
    
    # Register initial workers
    resource_manager.register_worker_node("worker1", 100.0, 16.0)
    
    # Try to scale up beyond max
    resource_manager.scaling_policy.scale_up_factor = 10.0
    resource_manager.execute_scaling_action(ScalingAction.SCALE_UP)
    
    active_workers = [w for w in resource_manager.worker_nodes.values() if w.status == "active"]
    assert len(active_workers) <= resource_manager.scaling_policy.max_replicas


def test_worker_heartbeat_update(resource_manager):
    """Test worker heartbeat updates."""
    # Register worker
    resource_manager.register_worker_node("worker1", 100.0, 16.0)
    
    # Set old heartbeat
    resource_manager.worker_nodes["worker1"].last_heartbeat = datetime.now() - timedelta(minutes=10)
    
    # Update heartbeats (should remove stale worker)
    resource_manager._update_worker_heartbeats()
    
    # Worker should be removed
    assert "worker1" not in resource_manager.worker_nodes


def test_concurrent_task_assignment(resource_manager):
    """Test concurrent task assignments to multiple workers."""
    # Register multiple workers
    for i in range(3):
        resource_manager.register_worker_node(f"worker{i}", 100.0, 16.0)
    
    # Assign multiple tasks
    assigned_workers = []
    for i in range(6):
        worker_id = resource_manager.assign_task_to_worker(
            task_id=f"task{i}",
            task_type="design_processing",
            resource_requirements={"cpu": 10.0, "memory": 2.0}
        )
        if worker_id:
            assigned_workers.append(worker_id)
    
    # All tasks should be assigned
    assert len(assigned_workers) == 6
    
    # Tasks should be distributed across workers
    unique_workers = set(assigned_workers)
    assert len(unique_workers) > 1


def test_shutdown(resource_manager):
    """Test graceful shutdown of resource manager."""
    resource_manager.shutdown()
    
    # Monitoring should stop
    assert resource_manager._monitoring_active is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
