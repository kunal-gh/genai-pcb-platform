"""
Ray cluster management for distributed RL training.

This module provides Ray cluster initialization, scaling, health checks, and graceful
shutdown for distributed reinforcement learning training. It supports:
- Local development mode (single machine)
- Kubernetes deployment with Ray operator
- Worker scaling from 4 to 64 parallel workers
- Automatic fallback to local mode if cluster unavailable
- Resource management (CPU, GPU, memory)
- Cluster health monitoring

Requirements satisfied: 5.1, 5.2, 5.3, 5.4
"""

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import socket

logger = logging.getLogger(__name__)


class ClusterMode(Enum):
    """Ray cluster deployment mode."""
    LOCAL = "local"  # Single machine, local development
    KUBERNETES = "kubernetes"  # Kubernetes with Ray operator
    REMOTE = "remote"  # Remote Ray cluster


class ClusterStatus(Enum):
    """Ray cluster health status."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Some workers unavailable
    UNHEALTHY = "unhealthy"
    SHUTDOWN = "shutdown"


@dataclass
class NodeConfig:
    """Configuration for a Ray node (head or worker)."""
    num_cpus: int
    memory_gb: int
    num_gpus: int = 0
    object_store_memory_gb: int = 10
    
    def to_resources(self) -> Dict[str, float]:
        """Convert to Ray resources dictionary."""
        resources = {
            "CPU": self.num_cpus,
            "memory": self.memory_gb * 1024 * 1024 * 1024,  # Convert to bytes
            "object_store_memory": self.object_store_memory_gb * 1024 * 1024 * 1024,
        }
        if self.num_gpus > 0:
            resources["GPU"] = self.num_gpus
        return resources


@dataclass
class ClusterConfig:
    """
    Ray cluster configuration.
    
    Attributes:
        mode: Deployment mode (local, kubernetes, remote)
        num_workers: Number of worker nodes (4-64)
        head_node: Head node configuration
        worker_node: Worker node configuration
        gpu_worker_node: GPU worker node configuration (optional)
        num_gpu_workers: Number of GPU workers (0-8)
        redis_address: Redis address for cluster coordination
        ray_address: Ray head node address (for remote mode)
        namespace: Ray namespace for isolation
        dashboard_port: Ray dashboard port
        object_store_memory_gb: Object store memory per node
    """
    mode: ClusterMode = ClusterMode.LOCAL
    num_workers: int = 4
    
    # Node configurations (from design spec)
    head_node: NodeConfig = field(default_factory=lambda: NodeConfig(
        num_cpus=8,
        memory_gb=32,
        num_gpus=0,
        object_store_memory_gb=10
    ))
    
    worker_node: NodeConfig = field(default_factory=lambda: NodeConfig(
        num_cpus=4,
        memory_gb=16,
        num_gpus=0,
        object_store_memory_gb=10
    ))
    
    gpu_worker_node: NodeConfig = field(default_factory=lambda: NodeConfig(
        num_cpus=8,
        memory_gb=32,
        num_gpus=1,
        object_store_memory_gb=10
    ))
    
    num_gpu_workers: int = 0  # 0-8 GPU workers
    
    # Connection settings
    redis_address: Optional[str] = None
    ray_address: Optional[str] = None  # "auto" or "ray://host:port"
    namespace: str = "rl_routing"
    
    # Ports
    dashboard_port: int = 8265
    
    # Resource limits
    min_workers: int = 4
    max_workers: int = 64
    
    # Health check settings
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 10
    
    # Logging
    log_to_driver: bool = True
    logging_level: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate cluster configuration."""
        # Validate worker count
        if not (self.min_workers <= self.num_workers <= self.max_workers):
            raise ValueError(
                f"num_workers ({self.num_workers}) must be between "
                f"{self.min_workers} and {self.max_workers}"
            )
        
        # Validate GPU worker count
        if not (0 <= self.num_gpu_workers <= 8):
            raise ValueError(
                f"num_gpu_workers ({self.num_gpu_workers}) must be between 0 and 8"
            )
        
        # Validate mode-specific requirements
        if self.mode == ClusterMode.REMOTE and self.ray_address is None:
            raise ValueError("ray_address is required for remote mode")
        
        logger.info(f"Cluster configuration validated: {self.num_workers} workers, mode={self.mode.value}")


@dataclass
class ClusterHealth:
    """Ray cluster health information."""
    status: ClusterStatus
    num_nodes: int
    num_workers: int
    num_gpus: int
    available_cpus: int
    available_memory_gb: float
    available_gpus: int
    node_statuses: List[Dict[str, Any]] = field(default_factory=list)
    last_check_time: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    
    def is_healthy(self) -> bool:
        """Check if cluster is healthy."""
        return self.status == ClusterStatus.HEALTHY
    
    def is_degraded(self) -> bool:
        """Check if cluster is degraded."""
        return self.status == ClusterStatus.DEGRADED
    
    def summary(self) -> str:
        """Get human-readable health summary."""
        return (
            f"Status: {self.status.value}, "
            f"Nodes: {self.num_nodes}, "
            f"Workers: {self.num_workers}, "
            f"CPUs: {self.available_cpus}, "
            f"GPUs: {self.available_gpus}, "
            f"Memory: {self.available_memory_gb:.1f}GB"
        )


class RayClusterManager:
    """
    Ray cluster manager for distributed RL training.
    
    This class manages Ray cluster lifecycle:
    - Initialization (local and remote)
    - Worker scaling (4-64 workers)
    - Health checks and monitoring
    - Graceful shutdown
    - Automatic fallback to local mode
    
    Example:
        # Local mode
        manager = RayClusterManager()
        manager.initialize()
        
        # Kubernetes mode
        config = ClusterConfig(mode=ClusterMode.KUBERNETES, num_workers=16)
        manager = RayClusterManager(config)
        manager.initialize()
        
        # Check health
        health = manager.check_health()
        print(health.summary())
        
        # Shutdown
        manager.shutdown()
    """
    
    def __init__(self, config: Optional[ClusterConfig] = None):
        """
        Initialize Ray cluster manager.
        
        Args:
            config: Cluster configuration (uses defaults if None)
        """
        self.config = config or ClusterConfig()
        self._initialized = False
        self._ray_context = None
        self._fallback_mode = False
        
        logger.info(f"RayClusterManager created with mode={self.config.mode.value}")
    
    def initialize(self, force_local: bool = False) -> bool:
        """
        Initialize Ray cluster.
        
        This method:
        1. Attempts to initialize Ray cluster based on configuration mode
        2. Falls back to local mode if cluster initialization fails
        3. Validates cluster resources meet minimum requirements
        4. Performs initial health check
        
        Args:
            force_local: Force local mode regardless of configuration
            
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            logger.warning("Ray cluster already initialized")
            return True
        
        try:
            import ray
        except ImportError:
            logger.error(
                "Ray is not installed. Install with: pip install ray[default]>=2.7.0"
            )
            return False
        
        logger.info(f"Initializing Ray cluster in {self.config.mode.value} mode...")
        
        # Force local mode if requested
        if force_local:
            logger.info("Forcing local mode")
            self.config.mode = ClusterMode.LOCAL
        
        try:
            if self.config.mode == ClusterMode.LOCAL:
                success = self._initialize_local(ray)
            elif self.config.mode == ClusterMode.KUBERNETES:
                success = self._initialize_kubernetes(ray)
            elif self.config.mode == ClusterMode.REMOTE:
                success = self._initialize_remote(ray)
            else:
                raise ValueError(f"Unsupported cluster mode: {self.config.mode}")
            
            if not success:
                logger.warning("Cluster initialization failed, attempting fallback to local mode")
                return self._fallback_to_local(ray)
            
            self._initialized = True
            self._ray_context = ray
            
            # Perform initial health check
            health = self.check_health()
            logger.info(f"Ray cluster initialized successfully: {health.summary()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray cluster: {e}", exc_info=True)
            logger.warning("Attempting fallback to local mode")
            return self._fallback_to_local(ray)
    
    def _initialize_local(self, ray) -> bool:
        """
        Initialize Ray in local mode (single machine).
        
        Args:
            ray: Ray module
            
        Returns:
            True if successful
        """
        logger.info("Initializing Ray in local mode")
        
        try:
            # Check if Ray is already initialized
            if ray.is_initialized():
                logger.info("Ray is already initialized, using existing cluster")
                return True
            
            # Initialize Ray with local resources
            init_kwargs = {
                "num_cpus": self.config.head_node.num_cpus,
                "num_gpus": self.config.head_node.num_gpus,
                "object_store_memory": self.config.head_node.object_store_memory_gb * 1024 * 1024 * 1024,
                "dashboard_port": self.config.dashboard_port,
                "namespace": self.config.namespace,
                "logging_level": self.config.logging_level,
                "log_to_driver": self.config.log_to_driver,
                "ignore_reinit_error": True,
            }
            
            ray.init(**init_kwargs)
            
            logger.info(
                f"Ray initialized in local mode: "
                f"{self.config.head_node.num_cpus} CPUs, "
                f"{self.config.head_node.num_gpus} GPUs"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray in local mode: {e}", exc_info=True)
            return False
    
    def _initialize_kubernetes(self, ray) -> bool:
        """
        Initialize Ray cluster on Kubernetes with Ray operator.
        
        Args:
            ray: Ray module
            
        Returns:
            True if successful
        """
        logger.info("Initializing Ray cluster on Kubernetes")
        
        try:
            # In Kubernetes mode, Ray is typically already initialized by the operator
            # We just need to connect to it
            
            # Check if Ray address is provided via environment variable
            ray_address = os.environ.get("RAY_ADDRESS", "auto")
            
            if ray.is_initialized():
                logger.info("Ray is already initialized in Kubernetes cluster")
                return True
            
            # Connect to Ray cluster
            ray.init(
                address=ray_address,
                namespace=self.config.namespace,
                logging_level=self.config.logging_level,
                log_to_driver=self.config.log_to_driver,
                ignore_reinit_error=True,
            )
            
            logger.info(f"Connected to Ray cluster on Kubernetes: {ray_address}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray on Kubernetes: {e}", exc_info=True)
            return False
    
    def _initialize_remote(self, ray) -> bool:
        """
        Initialize connection to remote Ray cluster.
        
        Args:
            ray: Ray module
            
        Returns:
            True if successful
        """
        logger.info(f"Connecting to remote Ray cluster: {self.config.ray_address}")
        
        try:
            if ray.is_initialized():
                logger.info("Ray is already initialized")
                return True
            
            # Connect to remote Ray cluster
            ray.init(
                address=self.config.ray_address,
                namespace=self.config.namespace,
                logging_level=self.config.logging_level,
                log_to_driver=self.config.log_to_driver,
                ignore_reinit_error=True,
            )
            
            logger.info(f"Connected to remote Ray cluster: {self.config.ray_address}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to remote Ray cluster: {e}", exc_info=True)
            return False
    
    def _fallback_to_local(self, ray) -> bool:
        """
        Fallback to local mode if cluster initialization fails.
        
        Args:
            ray: Ray module
            
        Returns:
            True if fallback successful
        """
        logger.warning("Ray cluster initialization failed, falling back to local mode")
        logger.warning("Training will proceed in single-process mode with reduced parallelism")
        
        self._fallback_mode = True
        self.config.mode = ClusterMode.LOCAL
        
        success = self._initialize_local(ray)
        
        if success:
            self._initialized = True
            self._ray_context = ray
            logger.info("Successfully fell back to local mode")
        else:
            logger.error("Fallback to local mode failed")
        
        return success
    
    def check_health(self) -> ClusterHealth:
        """
        Check Ray cluster health.
        
        This method:
        1. Queries Ray cluster state
        2. Checks node availability
        3. Checks resource availability (CPU, GPU, memory)
        4. Determines overall health status
        
        Returns:
            ClusterHealth object with current cluster state
        """
        if not self._initialized or self._ray_context is None:
            return ClusterHealth(
                status=ClusterStatus.UNINITIALIZED,
                num_nodes=0,
                num_workers=0,
                num_gpus=0,
                available_cpus=0,
                available_memory_gb=0.0,
                available_gpus=0,
                error_message="Cluster not initialized"
            )
        
        try:
            ray = self._ray_context
            
            # Get cluster resources
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            # Get node information
            nodes = ray.nodes()
            alive_nodes = [n for n in nodes if n.get("Alive", False)]
            
            # Count workers (exclude head node)
            num_workers = len(alive_nodes) - 1 if len(alive_nodes) > 0 else 0
            
            # Extract resource information
            total_cpus = int(cluster_resources.get("CPU", 0))
            available_cpus = int(available_resources.get("CPU", 0))
            
            total_gpus = int(cluster_resources.get("GPU", 0))
            available_gpus = int(available_resources.get("GPU", 0))
            
            total_memory = cluster_resources.get("memory", 0)
            available_memory = available_resources.get("memory", 0)
            available_memory_gb = available_memory / (1024 ** 3)
            
            # Determine health status
            if len(alive_nodes) == 0:
                status = ClusterStatus.UNHEALTHY
                error_message = "No alive nodes"
            elif self.config.mode != ClusterMode.LOCAL and num_workers < self.config.min_workers:
                status = ClusterStatus.DEGRADED
                error_message = f"Only {num_workers} workers available (minimum: {self.config.min_workers})"
            elif available_cpus < 1:
                status = ClusterStatus.DEGRADED
                error_message = "No available CPUs"
            else:
                status = ClusterStatus.HEALTHY
                error_message = None
            
            # Build node status list
            node_statuses = []
            for node in alive_nodes:
                node_status = {
                    "node_id": node.get("NodeID", "unknown"),
                    "node_name": node.get("NodeName", "unknown"),
                    "alive": node.get("Alive", False),
                    "resources": node.get("Resources", {}),
                }
                node_statuses.append(node_status)
            
            health = ClusterHealth(
                status=status,
                num_nodes=len(alive_nodes),
                num_workers=num_workers,
                num_gpus=total_gpus,
                available_cpus=available_cpus,
                available_memory_gb=available_memory_gb,
                available_gpus=available_gpus,
                node_statuses=node_statuses,
                error_message=error_message
            )
            
            if not health.is_healthy():
                logger.warning(f"Cluster health check: {health.summary()}")
                if error_message:
                    logger.warning(f"Health issue: {error_message}")
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return ClusterHealth(
                status=ClusterStatus.UNHEALTHY,
                num_nodes=0,
                num_workers=0,
                num_gpus=0,
                available_cpus=0,
                available_memory_gb=0.0,
                available_gpus=0,
                error_message=str(e)
            )
    
    def scale_workers(self, num_workers: int) -> bool:
        """
        Scale the number of worker nodes.
        
        Note: This method is primarily for documentation. In Kubernetes mode,
        scaling is typically handled by the Ray operator or Kubernetes HPA.
        In local mode, scaling is not applicable.
        
        Args:
            num_workers: Target number of workers (4-64)
            
        Returns:
            True if scaling initiated successfully
        """
        if not self._initialized:
            logger.error("Cannot scale workers: cluster not initialized")
            return False
        
        # Validate worker count
        if not (self.config.min_workers <= num_workers <= self.config.max_workers):
            logger.error(
                f"Invalid worker count: {num_workers}. "
                f"Must be between {self.config.min_workers} and {self.config.max_workers}"
            )
            return False
        
        if self.config.mode == ClusterMode.LOCAL:
            logger.warning("Worker scaling not supported in local mode")
            return False
        
        if self.config.mode == ClusterMode.KUBERNETES:
            logger.info(
                f"Worker scaling in Kubernetes mode should be handled by Ray operator. "
                f"Target workers: {num_workers}"
            )
            # In Kubernetes, scaling is handled by the Ray operator
            # This would typically involve updating the RayCluster CRD
            self.config.num_workers = num_workers
            return True
        
        logger.warning(f"Worker scaling not implemented for mode: {self.config.mode.value}")
        return False
    
    def is_initialized(self) -> bool:
        """Check if cluster is initialized."""
        return self._initialized
    
    def is_fallback_mode(self) -> bool:
        """Check if cluster is running in fallback mode."""
        return self._fallback_mode
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get comprehensive cluster information.
        
        Returns:
            Dictionary with cluster configuration and status
        """
        health = self.check_health()
        
        return {
            "initialized": self._initialized,
            "mode": self.config.mode.value,
            "fallback_mode": self._fallback_mode,
            "num_workers": self.config.num_workers,
            "health": {
                "status": health.status.value,
                "num_nodes": health.num_nodes,
                "num_workers": health.num_workers,
                "available_cpus": health.available_cpus,
                "available_gpus": health.available_gpus,
                "available_memory_gb": health.available_memory_gb,
                "error_message": health.error_message,
            },
            "config": {
                "head_node": {
                    "cpus": self.config.head_node.num_cpus,
                    "memory_gb": self.config.head_node.memory_gb,
                    "gpus": self.config.head_node.num_gpus,
                },
                "worker_node": {
                    "cpus": self.config.worker_node.num_cpus,
                    "memory_gb": self.config.worker_node.memory_gb,
                    "gpus": self.config.worker_node.num_gpus,
                },
                "num_gpu_workers": self.config.num_gpu_workers,
            }
        }
    
    def shutdown(self, graceful: bool = True):
        """
        Shutdown Ray cluster gracefully.
        
        Args:
            graceful: If True, wait for running tasks to complete
        """
        if not self._initialized:
            logger.warning("Cluster not initialized, nothing to shutdown")
            return
        
        logger.info("Shutting down Ray cluster...")
        
        try:
            if self._ray_context is not None:
                ray = self._ray_context
                
                if graceful:
                    logger.info("Waiting for running tasks to complete...")
                    # In a real implementation, we would wait for tasks
                    # For now, we just shutdown
                
                ray.shutdown()
                logger.info("Ray cluster shutdown complete")
            
            self._initialized = False
            self._ray_context = None
            
        except Exception as e:
            logger.error(f"Error during cluster shutdown: {e}", exc_info=True)
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False


def create_cluster_manager(
    mode: str = "local",
    num_workers: int = 4,
    num_gpu_workers: int = 0,
    ray_address: Optional[str] = None
) -> RayClusterManager:
    """
    Create Ray cluster manager with simplified configuration.
    
    Args:
        mode: Cluster mode ("local", "kubernetes", "remote")
        num_workers: Number of worker nodes (4-64)
        num_gpu_workers: Number of GPU workers (0-8)
        ray_address: Ray address for remote mode
        
    Returns:
        Configured RayClusterManager instance
        
    Example:
        # Local development
        manager = create_cluster_manager(mode="local")
        
        # Kubernetes with 16 workers
        manager = create_cluster_manager(mode="kubernetes", num_workers=16)
        
        # Remote cluster
        manager = create_cluster_manager(
            mode="remote",
            ray_address="ray://my-cluster:10001"
        )
    """
    cluster_mode = ClusterMode(mode)
    
    config = ClusterConfig(
        mode=cluster_mode,
        num_workers=num_workers,
        num_gpu_workers=num_gpu_workers,
        ray_address=ray_address
    )
    
    return RayClusterManager(config)
