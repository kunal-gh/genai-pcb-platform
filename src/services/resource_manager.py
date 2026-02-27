"""
Resource Manager Service

Handles auto-scaling, resource allocation, and load balancing for the GenAI PCB Platform.
Provides concurrent user support with intelligent resource management.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import redis
from kubernetes import client, config
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources that can be managed"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"


class ScalingAction(Enum):
    """Scaling actions that can be performed"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics"""
    cpu_percent: float
    memory_percent: float
    gpu_percent: float = 0.0
    storage_percent: float = 0.0
    active_requests: int = 0
    queue_length: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingPolicy:
    """Configuration for auto-scaling behavior"""
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 80.0
    scale_up_threshold: float = 85.0
    scale_down_threshold: float = 30.0
    cooldown_period: int = 300  # seconds
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7


@dataclass
class WorkerNode:
    """Represents a worker node in the cluster"""
    node_id: str
    status: str
    cpu_capacity: float
    memory_capacity: float
    gpu_capacity: float = 0.0
    current_load: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    active_tasks: List[str] = field(default_factory=list)


class ResourceManager:
    """
    Manages resource allocation, auto-scaling, and load balancing
    for concurrent user support and optimal performance.
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.scaling_policy = ScalingPolicy()
        self.worker_nodes: Dict[str, WorkerNode] = {}
        self.resource_metrics: List[ResourceMetrics] = []
        self.last_scaling_action = datetime.now()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.RLock()

        # Initialize Kubernetes client if available
        self.k8s_client = None
        self._init_kubernetes()

        # Start monitoring thread
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()

        logger.info("ResourceManager initialized with auto-scaling enabled")

    def _init_kubernetes(self):
        """Initialize Kubernetes client for auto-scaling"""
        try:
            # Try in-cluster config first, then local config
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()

            self.k8s_client = client.AppsV1Api()
            logger.info("Kubernetes client initialized successfully")
        except Exception as e:
            logger.warning(f"Kubernetes not available, using local scaling: {e}")

    def register_worker_node(self, node_id: str, cpu_capacity: float,
                           memory_capacity: float, gpu_capacity: float = 0.0) -> bool:
        """Register a new worker node in the cluster"""
        try:
            with self._lock:
                worker = WorkerNode(
                    node_id=node_id,
                    status="active",
                    cpu_capacity=cpu_capacity,
                    memory_capacity=memory_capacity,
                    gpu_capacity=gpu_capacity
                )
                self.worker_nodes[node_id] = worker

                # Store in Redis for persistence
                self.redis_client.hset(
                    "worker_nodes",
                    node_id,
                    f"{cpu_capacity},{memory_capacity},{gpu_capacity}"
                )

                logger.info(f"Registered worker node {node_id} with capacity: "
                          f"CPU={cpu_capacity}, Memory={memory_capacity}, GPU={gpu_capacity}")
                return True

        except Exception as e:
            logger.error(f"Failed to register worker node {node_id}: {e}")
            return False

    def get_resource_metrics(self) -> ResourceMetrics:
        """Get current system resource utilization"""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Get GPU metrics if available
            gpu_percent = 0.0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_percent = sum(gpu.load * 100 for gpu in gpus) / len(gpus)
            except ImportError:
                pass

            # Get storage metrics (use C: on Windows if / not available)
            try:
                disk = psutil.disk_usage('/')
            except OSError:
                try:
                    disk = psutil.disk_usage('C:\\')
                except OSError:
                    disk = None
            storage_percent = (disk.used / disk.total * 100) if (disk and getattr(disk, 'total', 0)) else 0.0

            # Get queue metrics from Redis
            queue_length = self.redis_client.llen("processing_queue") or 0
            active_requests = len(self.redis_client.keys("request:*:status")) or 0

            metrics = ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                gpu_percent=gpu_percent,
                storage_percent=storage_percent,
                active_requests=active_requests,
                queue_length=queue_length
            )

            # Store metrics history
            with self._lock:
                self.resource_metrics.append(metrics)
                # Keep only last 100 metrics
                if len(self.resource_metrics) > 100:
                    self.resource_metrics = self.resource_metrics[-100:]

            return metrics

        except Exception as e:
            logger.error(f"Failed to get resource metrics: {e}")
            return ResourceMetrics(cpu_percent=0, memory_percent=0)

    def determine_scaling_action(self, metrics: ResourceMetrics) -> ScalingAction:
        """Determine if scaling action is needed based on current metrics"""
        try:
            # Check cooldown period
            if datetime.now() - self.last_scaling_action < timedelta(seconds=self.scaling_policy.cooldown_period):
                return ScalingAction.MAINTAIN

            # Calculate average metrics over last 5 minutes
            recent_metrics = [m for m in self.resource_metrics
                            if datetime.now() - m.timestamp < timedelta(minutes=5)]

            if not recent_metrics:
                return ScalingAction.MAINTAIN

            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            avg_queue = sum(m.queue_length for m in recent_metrics) / len(recent_metrics)

            # Scale up conditions
            if (avg_cpu > self.scaling_policy.scale_up_threshold or
                avg_memory > self.scaling_policy.scale_up_threshold or
                avg_queue > 10):  # Queue threshold
                return ScalingAction.SCALE_UP

            # Scale down conditions
            if (avg_cpu < self.scaling_policy.scale_down_threshold and
                avg_memory < self.scaling_policy.scale_down_threshold and
                avg_queue < 2):
                return ScalingAction.SCALE_DOWN

            return ScalingAction.MAINTAIN

        except Exception as e:
            logger.error(f"Failed to determine scaling action: {e}")
            return ScalingAction.MAINTAIN

    def execute_scaling_action(self, action: ScalingAction) -> bool:
        """Execute the determined scaling action"""
        try:
            if action == ScalingAction.MAINTAIN:
                return True

            current_replicas = len([w for w in self.worker_nodes.values() if w.status == "active"])

            if action == ScalingAction.SCALE_UP:
                target_replicas = min(
                    int(current_replicas * self.scaling_policy.scale_up_factor),
                    self.scaling_policy.max_replicas
                )
            else:  # SCALE_DOWN
                target_replicas = max(
                    int(current_replicas * self.scaling_policy.scale_down_factor),
                    self.scaling_policy.min_replicas
                )

            if target_replicas == current_replicas:
                return True

            # Execute scaling via Kubernetes if available
            if self.k8s_client:
                success = self._scale_kubernetes_deployment(target_replicas)
            else:
                success = self._scale_local_workers(target_replicas)

            if success:
                self.last_scaling_action = datetime.now()
                logger.info(f"Scaled {action.value} from {current_replicas} to {target_replicas} replicas")

            return success

        except Exception as e:
            logger.error(f"Failed to execute scaling action {action}: {e}")
            return False

    def _scale_kubernetes_deployment(self, target_replicas: int) -> bool:
        """Scale Kubernetes deployment to target replica count"""
        try:
            # Update deployment replica count
            deployment_name = "genai-pcb-workers"
            namespace = "default"

            # Get current deployment
            deployment = self.k8s_client.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )

            # Update replica count
            deployment.spec.replicas = target_replicas

            # Apply the update
            self.k8s_client.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )

            logger.info(f"Kubernetes deployment scaled to {target_replicas} replicas")
            return True

        except ApiException as e:
            logger.error(f"Kubernetes API error during scaling: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to scale Kubernetes deployment: {e}")
            return False

    def _scale_local_workers(self, target_replicas: int) -> bool:
        """Scale local worker processes (fallback when Kubernetes unavailable)"""
        try:
            current_active = len([w for w in self.worker_nodes.values() if w.status == "active"])

            if target_replicas > current_active:
                # Scale up - activate idle workers or create new ones
                idle_workers = [w for w in self.worker_nodes.values() if w.status == "idle"]
                needed = target_replicas - current_active

                for i, worker in enumerate(idle_workers[:needed]):
                    worker.status = "active"
                    worker.last_heartbeat = datetime.now()

                # If still need more, create new workers
                remaining = needed - len(idle_workers)
                for i in range(remaining):
                    node_id = f"local-worker-{len(self.worker_nodes) + i}"
                    self.register_worker_node(node_id, 100.0, 100.0)

            else:
                # Scale down - deactivate excess workers
                active_workers = [w for w in self.worker_nodes.values() if w.status == "active"]
                to_deactivate = current_active - target_replicas

                for worker in active_workers[:to_deactivate]:
                    if not worker.active_tasks:  # Only deactivate idle workers
                        worker.status = "idle"

            return True

        except Exception as e:
            logger.error(f"Failed to scale local workers: {e}")
            return False

    def assign_task_to_worker(self, task_id: str, task_type: str,
                            resource_requirements: Dict[str, float]) -> Optional[str]:
        """Assign a task to the best available worker node"""
        try:
            with self._lock:
                # Find best worker based on current load and capacity
                best_worker = None
                best_score = float('inf')

                for worker in self.worker_nodes.values():
                    if worker.status != "active":
                        continue

                    # Check if worker can handle resource requirements
                    cpu_req = resource_requirements.get('cpu', 0)
                    memory_req = resource_requirements.get('memory', 0)

                    if (worker.current_load + cpu_req > worker.cpu_capacity or
                        memory_req > worker.memory_capacity):
                        continue

                    # Calculate load score (lower is better)
                    load_score = worker.current_load / worker.cpu_capacity
                    task_count_score = len(worker.active_tasks) / 10.0
                    total_score = load_score + task_count_score

                    if total_score < best_score:
                        best_score = total_score
                        best_worker = worker

                if best_worker:
                    # Assign task to worker
                    best_worker.active_tasks.append(task_id)
                    best_worker.current_load += resource_requirements.get('cpu', 0)

                    # Store assignment in Redis
                    self.redis_client.hset(
                        "task_assignments",
                        task_id,
                        f"{best_worker.node_id},{task_type},{datetime.now().isoformat()}"
                    )

                    logger.info(f"Assigned task {task_id} to worker {best_worker.node_id}")
                    return best_worker.node_id

                logger.warning(f"No available worker for task {task_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to assign task {task_id}: {e}")
            return None

    def complete_task(self, task_id: str, worker_id: str) -> bool:
        """Mark a task as completed and free up worker resources"""
        try:
            with self._lock:
                worker = self.worker_nodes.get(worker_id)
                if not worker:
                    logger.warning(f"Worker {worker_id} not found for task completion")
                    return False

                if task_id in worker.active_tasks:
                    worker.active_tasks.remove(task_id)

                    # Get task resource requirements from Redis
                    assignment = self.redis_client.hget("task_assignments", task_id)
                    if assignment:
                        # Free up resources (simplified - in real implementation,
                        # would track actual resource usage)
                        worker.current_load = max(0, worker.current_load - 10.0)

                        # Remove assignment
                        self.redis_client.hdel("task_assignments", task_id)

                logger.info(f"Task {task_id} completed on worker {worker_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to complete task {task_id}: {e}")
            return False

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status and metrics"""
        try:
            metrics = self.get_resource_metrics()

            active_workers = [w for w in self.worker_nodes.values() if w.status == "active"]
            total_capacity = sum(w.cpu_capacity for w in active_workers)
            total_load = sum(w.current_load for w in active_workers)

            return {
                "timestamp": datetime.now().isoformat(),
                "resource_metrics": {
                    "cpu_percent": metrics.cpu_percent,
                    "memory_percent": metrics.memory_percent,
                    "gpu_percent": metrics.gpu_percent,
                    "storage_percent": metrics.storage_percent,
                    "active_requests": metrics.active_requests,
                    "queue_length": metrics.queue_length
                },
                "cluster_metrics": {
                    "total_workers": len(self.worker_nodes),
                    "active_workers": len(active_workers),
                    "total_capacity": total_capacity,
                    "total_load": total_load,
                    "utilization_percent": (total_load / total_capacity * 100) if total_capacity > 0 else 0
                },
                "scaling_policy": {
                    "min_replicas": self.scaling_policy.min_replicas,
                    "max_replicas": self.scaling_policy.max_replicas,
                    "target_cpu_percent": self.scaling_policy.target_cpu_percent,
                    "last_scaling_action": self.last_scaling_action.isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Failed to get cluster status: {e}")
            return {"error": str(e)}

    def _monitoring_loop(self):
        """Background monitoring loop for auto-scaling"""
        while self._monitoring_active:
            try:
                # Get current metrics
                metrics = self.get_resource_metrics()

                # Determine scaling action
                action = self.determine_scaling_action(metrics)

                # Execute scaling if needed
                if action != ScalingAction.MAINTAIN:
                    self.execute_scaling_action(action)

                # Update worker heartbeats
                self._update_worker_heartbeats()

                # Sleep for monitoring interval
                time.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error

    def _update_worker_heartbeats(self):
        """Update worker node heartbeats and remove stale nodes"""
        try:
            current_time = datetime.now()
            stale_threshold = timedelta(minutes=5)

            with self._lock:
                stale_workers = []
                for worker_id, worker in self.worker_nodes.items():
                    if current_time - worker.last_heartbeat > stale_threshold:
                        stale_workers.append(worker_id)

                # Remove stale workers
                for worker_id in stale_workers:
                    logger.warning(f"Removing stale worker {worker_id}")
                    del self.worker_nodes[worker_id]
                    self.redis_client.hdel("worker_nodes", worker_id)

        except Exception as e:
            logger.error(f"Failed to update worker heartbeats: {e}")

    def shutdown(self):
        """Shutdown the resource manager"""
        try:
            self._monitoring_active = False
            if hasattr(self, '_monitoring_thread'):
                self._monitoring_thread.join(timeout=5)

            self.executor.shutdown(wait=True)
            logger.info("ResourceManager shutdown completed")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Global resource manager instance
_resource_manager = None


def get_resource_manager() -> ResourceManager:
    """Get the global resource manager instance"""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager
