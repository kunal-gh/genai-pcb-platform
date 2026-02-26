"""
Distributed RL training pipeline for PCB routing agent.

This module implements distributed reinforcement learning training using Ray RLlib
with PPO (Proximal Policy Optimization) algorithm. It supports:
- Distributed rollout collection across Ray cluster workers
- Gradient aggregation from distributed workers
- GPU acceleration for policy updates
- Checkpointing every 100k steps
- Comprehensive metric logging (TensorBoard, console)
- Final policy saving to model registry

Requirements satisfied: 5.5, 5.6, 5.7, 5.8, 5.9, 5.10
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RLTrainingConfig:
    """
    Configuration for RL routing agent training.
    
    This configuration combines PPO hyperparameters with training-specific settings
    like total timesteps, checkpoint frequency, and logging configuration.
    
    Attributes:
        total_timesteps: Total environment steps to train (default: 10M)
        checkpoint_frequency: Steps between checkpoints (default: 100k)
        checkpoint_dir: Directory for saving checkpoints
        log_dir: Directory for TensorBoard logs
        eval_frequency: Steps between evaluation runs
        eval_episodes: Number of episodes per evaluation
        save_final_model: Whether to save final model to registry
        model_version: Version string for model registry
        dataset_version: Dataset version for metadata
    """
    # Training duration
    total_timesteps: int = 10_000_000  # 10M steps
    
    # Checkpointing
    checkpoint_frequency: int = 100_000  # 100k steps
    checkpoint_dir: str = "checkpoints/rl_routing"
    
    # Logging
    log_dir: str = "logs/rl_routing"
    log_frequency: int = 1000  # Log every N steps
    
    # Evaluation
    eval_frequency: int = 50_000  # Evaluate every 50k steps
    eval_episodes: int = 10
    
    # Model registry
    save_final_model: bool = True
    model_version: str = "v1.0.0"
    dataset_version: str = "circuitnet_v2.0"
    
    # Early stopping
    early_stopping_patience: int = 10  # Stop if no improvement for N evals
    early_stopping_threshold: float = 0.01  # Minimum improvement threshold
    
    # Resource configuration
    use_gpu: bool = True
    num_gpus: int = 1
    
    def __post_init__(self):
        """Create directories after initialization."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingMetrics:
    """
    Training metrics collected during RL training.
    
    Tracks episode-level and step-level metrics for monitoring training progress.
    """
    # Episode metrics
    episode_reward_mean: float = 0.0
    episode_reward_min: float = 0.0
    episode_reward_max: float = 0.0
    episode_length_mean: float = 0.0
    
    # Routing-specific metrics
    routing_success_rate: float = 0.0
    via_count_mean: float = 0.0
    trace_length_mean: float = 0.0
    drc_violations_mean: float = 0.0
    
    # Training metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    kl_divergence: float = 0.0
    
    # Timestep tracking
    timesteps_total: int = 0
    episodes_total: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary for logging."""
        return {
            "episode_reward_mean": self.episode_reward_mean,
            "episode_reward_min": self.episode_reward_min,
            "episode_reward_max": self.episode_reward_max,
            "episode_length_mean": self.episode_length_mean,
            "routing_success_rate": self.routing_success_rate,
            "via_count_mean": self.via_count_mean,
            "trace_length_mean": self.trace_length_mean,
            "drc_violations_mean": self.drc_violations_mean,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "kl_divergence": self.kl_divergence,
            "timesteps_total": self.timesteps_total,
            "episodes_total": self.episodes_total,
        }


@dataclass
class TrainingResult:
    """
    Result of RL training run.
    
    Contains final metrics, training statistics, and paths to saved artifacts.
    """
    success: bool
    final_metrics: TrainingMetrics
    total_training_time: float
    checkpoint_path: str
    model_registry_path: Optional[str] = None
    tensorboard_log_dir: str = ""
    training_history: List[Dict[str, float]] = field(default_factory=list)
    error_message: Optional[str] = None
    
    def summary(self) -> str:
        """Get human-readable training summary."""
        lines = [
            "=" * 80,
            "RL Training Summary",
            "=" * 80,
            f"Success: {self.success}",
            f"Total Training Time: {self.total_training_time:.2f}s ({self.total_training_time/3600:.2f}h)",
            f"Total Timesteps: {self.final_metrics.timesteps_total}",
            f"Total Episodes: {self.final_metrics.episodes_total}",
            "",
            "Final Metrics:",
            f"  Episode Reward (mean): {self.final_metrics.episode_reward_mean:.2f}",
            f"  Routing Success Rate: {self.final_metrics.routing_success_rate:.2%}",
            f"  Via Count (mean): {self.final_metrics.via_count_mean:.2f}",
            f"  Trace Length (mean): {self.final_metrics.trace_length_mean:.2f}",
            f"  DRC Violations (mean): {self.final_metrics.drc_violations_mean:.2f}",
            "",
            f"Checkpoint: {self.checkpoint_path}",
        ]
        
        if self.model_registry_path:
            lines.append(f"Model Registry: {self.model_registry_path}")
        
        if self.tensorboard_log_dir:
            lines.append(f"TensorBoard Logs: {self.tensorboard_log_dir}")
        
        if self.error_message:
            lines.extend([
                "",
                f"Error: {self.error_message}"
            ])
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


class RLRoutingTrainer:
    """
    Distributed RL training for routing agent using Ray RLlib.
    
    This class implements the complete training pipeline:
    1. Initialize Ray cluster
    2. Create routing environment
    3. Configure PPO algorithm
    4. Distribute training across workers
    5. Training loop:
       - Collect rollouts from workers (distributed)
       - Aggregate experiences
       - Update policy using PPO
       - Log metrics (TensorBoard, console)
       - Checkpoint every 100k steps
    6. Save final policy to model registry
    
    The trainer supports:
    - Distributed rollout collection across Ray cluster workers
    - Gradient aggregation from distributed workers
    - GPU acceleration for policy updates
    - Automatic checkpointing
    - Comprehensive metric logging
    - Early stopping based on performance
    
    Example:
        from src.training.ray_cluster import RayClusterManager, ClusterConfig
        from src.training.ppo_config import PPOConfig
        from src.training.routing_environment import RoutingEnvironment
        
        # Initialize Ray cluster
        cluster_manager = RayClusterManager()
        cluster_manager.initialize()
        
        # Create trainer
        trainer = RLRoutingTrainer(
            cluster_manager=cluster_manager,
            ppo_config=PPOConfig(),
            training_config=RLTrainingConfig()
        )
        
        # Train
        result = trainer.train()
        print(result.summary())
    """
    
    def __init__(
        self,
        cluster_manager,  # RayClusterManager
        ppo_config,  # PPOConfig
        training_config: Optional[RLTrainingConfig] = None,
        env_creator=None,
        model_registry=None
    ):
        """
        Initialize RL routing trainer.
        
        Args:
            cluster_manager: RayClusterManager instance
            ppo_config: PPOConfig instance with PPO hyperparameters
            training_config: RLTrainingConfig instance (uses defaults if None)
            env_creator: Optional function to create routing environment
            model_registry: Optional ModelRegistry instance for saving final model
        """
        self.cluster_manager = cluster_manager
        self.ppo_config = ppo_config
        self.training_config = training_config or RLTrainingConfig()
        self.env_creator = env_creator
        self.model_registry = model_registry
        
        # Training state
        self.trainer = None
        self.tensorboard_writer = None
        self.training_start_time = None
        self.best_reward = float('-inf')
        self.no_improvement_count = 0
        
        # Metrics history
        self.metrics_history: List[Dict[str, float]] = []
        
        logger.info("RLRoutingTrainer initialized")
    
    def train(self) -> TrainingResult:
        """
        Train RL routing agent using distributed PPO or single-process fallback.
        
        This method implements the complete training loop:
        1. Initialize Ray cluster (if not already initialized)
        2. Detect if Ray cluster is in fallback mode
        3. Create routing environment
        4. Configure PPO algorithm (distributed or single-process)
        5. Training loop:
           - Collect rollouts (distributed or single-process)
           - Aggregate gradients
           - Update policy
           - Log metrics
           - Checkpoint periodically
           - Evaluate periodically
        6. Save final policy
        
        Returns:
            TrainingResult with final metrics and paths to artifacts
        """
        logger.info("Starting RL routing agent training")
        self.training_start_time = time.time()
        
        try:
            # Step 1: Initialize Ray cluster
            if not self.cluster_manager.is_initialized():
                logger.info("Initializing Ray cluster...")
                success = self.cluster_manager.initialize()
                if not success:
                    raise RuntimeError("Failed to initialize Ray cluster")
            
            # Step 2: Check if Ray cluster is in fallback mode
            if self.cluster_manager.is_fallback_mode():
                logger.warning(
                    "Ray cluster initialization failed, falling back to single-process training"
                )
                logger.warning(
                    "Training will proceed with reduced parallelism (no distributed workers)"
                )
                return self._train_single_process()
            
            # Check cluster health
            health = self.cluster_manager.check_health()
            logger.info(f"Cluster health: {health.summary()}")
            
            # Step 3: Create routing environment
            logger.info("Setting up routing environment...")
            self._setup_environment()
            
            # Step 4: Configure PPO algorithm
            logger.info("Configuring PPO algorithm...")
            self._setup_ppo_trainer()
            
            # Step 5: Setup logging
            logger.info("Setting up TensorBoard logging...")
            self._setup_tensorboard()
            
            # Step 6: Training loop
            logger.info("Starting training loop...")
            final_metrics = self._training_loop()
            
            # Step 7: Save final policy
            logger.info("Saving final policy...")
            model_registry_path = self._save_final_policy()
            
            # Calculate total training time
            total_time = time.time() - self.training_start_time
            
            # Create training result
            result = TrainingResult(
                success=True,
                final_metrics=final_metrics,
                total_training_time=total_time,
                checkpoint_path=str(Path(self.training_config.checkpoint_dir) / "final"),
                model_registry_path=model_registry_path,
                tensorboard_log_dir=self.training_config.log_dir,
                training_history=self.metrics_history
            )
            
            logger.info("Training completed successfully")
            logger.info(result.summary())
            
            return result
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            
            total_time = time.time() - self.training_start_time if self.training_start_time else 0.0
            
            return TrainingResult(
                success=False,
                final_metrics=TrainingMetrics(),
                total_training_time=total_time,
                checkpoint_path="",
                error_message=str(e)
            )
        
        finally:
            # Cleanup
            self._cleanup()
    
    def _setup_environment(self):
        """
        Setup routing environment for training.
        
        This method creates the environment creator function that will be used
        by Ray workers to instantiate routing environments.
        """
        if self.env_creator is None:
            # Create default environment creator
            def default_env_creator(env_config):
                from src.training.routing_environment import RoutingEnvironment
                from src.models.circuit_graph import CircuitGraph
                from src.models.pcb_state import BoardConstraints
                
                # Create simple test circuit for training
                # In production, this would load from a dataset
                circuit_graph = CircuitGraph(
                    nodes=[],
                    edges=[],
                    board_size=(100.0, 100.0),
                    num_layers=2,
                    design_rules={}
                )
                
                board_constraints = BoardConstraints(
                    width=100.0,
                    height=100.0,
                    num_layers=2,
                    keepout_zones=[]
                )
                
                return RoutingEnvironment(
                    circuit_graph=circuit_graph,
                    board_constraints=board_constraints,
                    grid_resolution=0.5
                )
            
            self.env_creator = default_env_creator
        
        logger.info("Environment creator configured")
    
    def _setup_ppo_trainer(self):
        """
        Setup PPO trainer with Ray RLlib.
        
        This method:
        1. Registers the routing environment with Ray
        2. Configures PPO algorithm with distributed workers
        3. Sets up GPU acceleration if available
        4. Creates the PPO trainer instance
        """
        try:
            import ray
            from ray import tune
            from ray.rllib.algorithms.ppo import PPO, PPOConfig as RLlibPPOConfig
            from ray.tune.registry import register_env
        except ImportError:
            raise ImportError(
                "Ray RLlib is not installed. Install with: pip install 'ray[rllib]>=2.7.0'"
            )
        
        # Register environment
        register_env("routing_env", self.env_creator)
        logger.info("Registered routing environment with Ray")
        
        # Create RLlib PPO config
        rllib_config_dict = self.ppo_config.to_rllib_config()
        
        # Override with training-specific settings
        rllib_config_dict.update({
            "num_gpus": self.training_config.num_gpus if self.training_config.use_gpu else 0,
        })
        
        # Create PPO config
        ppo_config = RLlibPPOConfig()
        ppo_config.update_from_dict(rllib_config_dict)
        ppo_config.environment("routing_env")
        
        # Configure custom model if needed
        # For now, use default CNN policy
        ppo_config.training(
            model={
                "conv_filters": [
                    [32, [3, 3], 1],
                    [64, [3, 3], 1],
                    [128, [3, 3], 1],
                    [256, [3, 3], 1],
                ],
                "fcnet_hiddens": list(self.ppo_config.hidden_dims),
                "fcnet_activation": "relu",
            }
        )
        
        # Build trainer
        self.trainer = ppo_config.build()
        
        logger.info("PPO trainer configured with distributed workers")
        logger.info(f"  Workers: {self.ppo_config.num_workers}")
        logger.info(f"  Envs per worker: {self.ppo_config.num_envs_per_worker}")
        logger.info(f"  GPUs: {rllib_config_dict['num_gpus']}")
    
    def _setup_tensorboard(self):
        """
        Setup TensorBoard logging.
        
        Creates TensorBoard SummaryWriter for logging training metrics.
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            log_dir = Path(self.training_config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            self.tensorboard_writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"TensorBoard logging enabled: {log_dir}")
            
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.tensorboard_writer = None
    
    def _training_loop(self) -> TrainingMetrics:
        """
        Main training loop.
        
        This loop:
        1. Collects rollouts from distributed workers
        2. Aggregates experiences
        3. Updates policy using PPO
        4. Logs metrics
        5. Checkpoints periodically
        6. Evaluates periodically
        7. Checks for early stopping
        
        Returns:
            Final training metrics
        """
        logger.info(f"Training for {self.training_config.total_timesteps} timesteps")
        
        timesteps_trained = 0
        iteration = 0
        
        while timesteps_trained < self.training_config.total_timesteps:
            iteration += 1
            
            # Train one iteration (collects rollouts and updates policy)
            result = self.trainer.train()
            
            # Extract metrics
            metrics = self._extract_metrics(result)
            timesteps_trained = metrics.timesteps_total
            
            # Log metrics
            if iteration % (self.training_config.log_frequency // 1000) == 0:
                self._log_metrics(metrics, iteration)
            
            # Checkpoint periodically
            if timesteps_trained % self.training_config.checkpoint_frequency < self.ppo_config.train_batch_size:
                self._save_checkpoint(timesteps_trained)
            
            # Evaluate periodically
            if timesteps_trained % self.training_config.eval_frequency < self.ppo_config.train_batch_size:
                self._evaluate(metrics)
            
            # Check early stopping
            if self._should_stop_early(metrics):
                logger.info(f"Early stopping triggered at {timesteps_trained} timesteps")
                break
            
            # Store metrics history
            self.metrics_history.append(metrics.to_dict())
        
        logger.info(f"Training loop completed: {timesteps_trained} timesteps, {iteration} iterations")
        
        return metrics
    
    def _extract_metrics(self, result: Dict[str, Any]) -> TrainingMetrics:
        """
        Extract training metrics from Ray RLlib result.
        
        Args:
            result: Training result dictionary from Ray RLlib
        
        Returns:
            TrainingMetrics object
        """
        # Extract episode metrics
        episode_reward_mean = result.get("episode_reward_mean", 0.0)
        episode_reward_min = result.get("episode_reward_min", 0.0)
        episode_reward_max = result.get("episode_reward_max", 0.0)
        episode_length_mean = result.get("episode_len_mean", 0.0)
        
        # Extract custom metrics (routing-specific)
        custom_metrics = result.get("custom_metrics", {})
        routing_success_rate = custom_metrics.get("routing_success_rate_mean", 0.0)
        via_count_mean = custom_metrics.get("via_count_mean", 0.0)
        trace_length_mean = custom_metrics.get("trace_length_mean", 0.0)
        drc_violations_mean = custom_metrics.get("drc_violations_mean", 0.0)
        
        # Extract training metrics
        info = result.get("info", {})
        learner_info = info.get("learner", {})
        default_policy = learner_info.get("default_policy", {})
        
        policy_loss = default_policy.get("policy_loss", 0.0)
        value_loss = default_policy.get("vf_loss", 0.0)
        entropy = default_policy.get("entropy", 0.0)
        kl_divergence = default_policy.get("kl", 0.0)
        
        # Extract timestep tracking
        timesteps_total = result.get("timesteps_total", 0)
        episodes_total = result.get("episodes_total", 0)
        
        return TrainingMetrics(
            episode_reward_mean=episode_reward_mean,
            episode_reward_min=episode_reward_min,
            episode_reward_max=episode_reward_max,
            episode_length_mean=episode_length_mean,
            routing_success_rate=routing_success_rate,
            via_count_mean=via_count_mean,
            trace_length_mean=trace_length_mean,
            drc_violations_mean=drc_violations_mean,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            kl_divergence=kl_divergence,
            timesteps_total=timesteps_total,
            episodes_total=episodes_total,
        )
    
    def _log_metrics(self, metrics: TrainingMetrics, iteration: int):
        """
        Log training metrics to console and TensorBoard.
        
        Args:
            metrics: Training metrics to log
            iteration: Current training iteration
        """
        # Console logging
        logger.info(
            f"Iteration {iteration} | "
            f"Timesteps: {metrics.timesteps_total} | "
            f"Reward: {metrics.episode_reward_mean:.2f} | "
            f"Success Rate: {metrics.routing_success_rate:.2%} | "
            f"Via Count: {metrics.via_count_mean:.2f} | "
            f"Trace Length: {metrics.trace_length_mean:.2f}"
        )
        
        # TensorBoard logging
        if self.tensorboard_writer is not None:
            timestep = metrics.timesteps_total
            
            # Episode metrics
            self.tensorboard_writer.add_scalar(
                "episode/reward_mean", metrics.episode_reward_mean, timestep
            )
            self.tensorboard_writer.add_scalar(
                "episode/reward_min", metrics.episode_reward_min, timestep
            )
            self.tensorboard_writer.add_scalar(
                "episode/reward_max", metrics.episode_reward_max, timestep
            )
            self.tensorboard_writer.add_scalar(
                "episode/length_mean", metrics.episode_length_mean, timestep
            )
            
            # Routing metrics
            self.tensorboard_writer.add_scalar(
                "routing/success_rate", metrics.routing_success_rate, timestep
            )
            self.tensorboard_writer.add_scalar(
                "routing/via_count_mean", metrics.via_count_mean, timestep
            )
            self.tensorboard_writer.add_scalar(
                "routing/trace_length_mean", metrics.trace_length_mean, timestep
            )
            self.tensorboard_writer.add_scalar(
                "routing/drc_violations_mean", metrics.drc_violations_mean, timestep
            )
            
            # Training metrics
            self.tensorboard_writer.add_scalar(
                "training/policy_loss", metrics.policy_loss, timestep
            )
            self.tensorboard_writer.add_scalar(
                "training/value_loss", metrics.value_loss, timestep
            )
            self.tensorboard_writer.add_scalar(
                "training/entropy", metrics.entropy, timestep
            )
            self.tensorboard_writer.add_scalar(
                "training/kl_divergence", metrics.kl_divergence, timestep
            )
            
            self.tensorboard_writer.flush()
    
    def _save_checkpoint(self, timesteps: int):
        """
        Save training checkpoint.
        
        Args:
            timesteps: Current timestep count
        """
        checkpoint_dir = Path(self.training_config.checkpoint_dir)
        checkpoint_path = checkpoint_dir / f"checkpoint_{timesteps}"
        
        self.trainer.save(str(checkpoint_path))
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _evaluate(self, metrics: TrainingMetrics):
        """
        Evaluate current policy.
        
        Args:
            metrics: Current training metrics
        """
        # Check if performance improved
        if metrics.episode_reward_mean > self.best_reward + self.training_config.early_stopping_threshold:
            self.best_reward = metrics.episode_reward_mean
            self.no_improvement_count = 0
            logger.info(f"New best reward: {self.best_reward:.2f}")
        else:
            self.no_improvement_count += 1
    
    def _should_stop_early(self, metrics: TrainingMetrics) -> bool:
        """
        Check if training should stop early.
        
        Args:
            metrics: Current training metrics
        
        Returns:
            True if training should stop
        """
        if self.no_improvement_count >= self.training_config.early_stopping_patience:
            logger.info(
                f"No improvement for {self.no_improvement_count} evaluations. "
                f"Best reward: {self.best_reward:.2f}"
            )
            return True
        
        return False
    
    def _save_final_policy(self) -> Optional[str]:
        """
        Save final policy to model registry.
        
        Returns:
            Path to saved model in registry, or None if not saved
        """
        if not self.training_config.save_final_model:
            logger.info("Skipping model registry save (save_final_model=False)")
            return None
        
        if self.model_registry is None:
            logger.warning("Model registry not provided, cannot save final model")
            return None
        
        try:
            # Save final checkpoint
            final_checkpoint_dir = Path(self.training_config.checkpoint_dir) / "final"
            self.trainer.save(str(final_checkpoint_dir))
            
            # Extract policy weights
            policy = self.trainer.get_policy()
            model = policy.model
            
            # Register with model registry
            metadata = {
                "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_version": self.training_config.dataset_version,
                "performance_metrics": {
                    "episode_reward_mean": self.best_reward,
                    "routing_success_rate": self.metrics_history[-1].get("routing_success_rate", 0.0) if self.metrics_history else 0.0,
                    "via_count_mean": self.metrics_history[-1].get("via_count_mean", 0.0) if self.metrics_history else 0.0,
                    "trace_length_mean": self.metrics_history[-1].get("trace_length_mean", 0.0) if self.metrics_history else 0.0,
                },
                "hyperparameters": self.ppo_config.to_dict(),
                "description": f"RL routing agent trained for {self.training_config.total_timesteps} timesteps"
            }
            
            model_path = self.model_registry.register_model(
                model_type="rl_routing",
                model=model,
                version=self.training_config.model_version,
                metadata=metadata
            )
            
            logger.info(f"Final model saved to registry: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to save final model to registry: {e}", exc_info=True)
            return None
    
    def _cleanup(self):
        """Cleanup resources after training."""
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
            logger.info("TensorBoard writer closed")
        
        if self.trainer is not None:
            self.trainer.stop()
            logger.info("Trainer stopped")
    
    def _train_single_process(self) -> TrainingResult:
        """
        Train RL routing agent in single-process mode (fallback).
        
        This method implements single-process training without Ray RLlib:
        1. Create routing environment
        2. Create ActorCriticNetwork
        3. Setup optimizer
        4. Training loop:
           - Collect rollouts (single process)
           - Compute advantages using GAE
           - Update policy using PPO
           - Log metrics
           - Checkpoint periodically
        5. Save final policy
        
        Returns:
            TrainingResult with final metrics and paths to artifacts
        """
        logger.info("Starting single-process training (fallback mode)")
        
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from src.services.rl_routing_agent import ActorCriticNetwork
            from src.training.routing_environment import RoutingEnvironment
            from src.models.circuit_graph import CircuitGraph
            from src.models.pcb_state import BoardConstraints
            
            # Setup device
            device = torch.device("cuda" if torch.cuda.is_available() and self.training_config.use_gpu else "cpu")
            logger.info(f"Using device: {device}")
            
            # Create routing environment
            logger.info("Creating routing environment...")
            circuit_graph = CircuitGraph(
                nodes=[],
                edges=[],
                board_size=(100.0, 100.0),
                num_layers=2,
                design_rules={}
            )
            
            board_constraints = BoardConstraints(
                width=100.0,
                height=100.0,
                num_layers=2,
                keepout_zones=[]
            )
            
            env = RoutingEnvironment(
                circuit_graph=circuit_graph,
                board_constraints=board_constraints,
                grid_resolution=0.5
            )
            
            # Get state and action dimensions
            state_shape = env.get_state_shape()
            action_dim = env.get_action_space_size()
            
            logger.info(f"State shape: {state_shape}, Action dim: {action_dim}")
            
            # Create ActorCriticNetwork
            logger.info("Creating ActorCriticNetwork...")
            network = ActorCriticNetwork(
                state_shape=state_shape,
                action_dim=action_dim,
                hidden_dims=list(self.ppo_config.hidden_dims)
            ).to(device)
            
            # Create optimizer
            optimizer = optim.Adam(
                network.parameters(),
                lr=self.ppo_config.learning_rate
            )
            
            # Setup TensorBoard
            self._setup_tensorboard()
            
            # Training loop
            logger.info("Starting single-process training loop...")
            timesteps_trained = 0
            episodes_trained = 0
            
            # Training metrics
            episode_rewards = []
            episode_lengths = []
            routing_success_rates = []
            via_counts = []
            trace_lengths = []
            
            while timesteps_trained < self.training_config.total_timesteps:
                # Collect rollout
                states, actions, rewards, values, log_probs = self._collect_rollout_single_process(
                    env, network, device
                )
                
                if len(states) == 0:
                    logger.warning("Empty rollout collected, skipping update")
                    continue
                
                # Compute advantages using GAE
                advantages, returns = self._compute_gae(
                    rewards, values, gamma=self.ppo_config.gamma, gae_lambda=self.ppo_config.gae_lambda
                )
                
                # Update policy using PPO
                policy_loss, value_loss, entropy = self._update_policy_single_process(
                    network, optimizer, states, actions, log_probs, advantages, returns, device
                )
                
                # Update counters
                timesteps_trained += len(states)
                episodes_trained += 1
                
                # Track metrics
                episode_reward = sum(rewards)
                episode_length = len(states)
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Log metrics periodically
                if episodes_trained % 10 == 0:
                    metrics = TrainingMetrics(
                        episode_reward_mean=np.mean(episode_rewards[-100:]) if episode_rewards else 0.0,
                        episode_reward_min=np.min(episode_rewards[-100:]) if episode_rewards else 0.0,
                        episode_reward_max=np.max(episode_rewards[-100:]) if episode_rewards else 0.0,
                        episode_length_mean=np.mean(episode_lengths[-100:]) if episode_lengths else 0.0,
                        routing_success_rate=0.0,  # Would need to track from environment
                        via_count_mean=0.0,
                        trace_length_mean=0.0,
                        drc_violations_mean=0.0,
                        policy_loss=policy_loss,
                        value_loss=value_loss,
                        entropy=entropy,
                        kl_divergence=0.0,
                        timesteps_total=timesteps_trained,
                        episodes_total=episodes_trained
                    )
                    
                    self._log_metrics(metrics, episodes_trained)
                    self.metrics_history.append(metrics.to_dict())
                
                # Checkpoint periodically
                if timesteps_trained % self.training_config.checkpoint_frequency < episode_length:
                    self._save_checkpoint_single_process(network, optimizer, timesteps_trained)
            
            # Save final model
            logger.info("Saving final model...")
            final_checkpoint_path = self._save_checkpoint_single_process(
                network, optimizer, timesteps_trained, is_final=True
            )
            
            # Create final metrics
            final_metrics = TrainingMetrics(
                episode_reward_mean=np.mean(episode_rewards[-100:]) if episode_rewards else 0.0,
                episode_reward_min=np.min(episode_rewards[-100:]) if episode_rewards else 0.0,
                episode_reward_max=np.max(episode_rewards[-100:]) if episode_rewards else 0.0,
                episode_length_mean=np.mean(episode_lengths[-100:]) if episode_lengths else 0.0,
                routing_success_rate=0.0,
                via_count_mean=0.0,
                trace_length_mean=0.0,
                drc_violations_mean=0.0,
                policy_loss=0.0,
                value_loss=0.0,
                entropy=0.0,
                kl_divergence=0.0,
                timesteps_total=timesteps_trained,
                episodes_total=episodes_trained
            )
            
            # Calculate total training time
            total_time = time.time() - self.training_start_time
            
            # Create training result
            result = TrainingResult(
                success=True,
                final_metrics=final_metrics,
                total_training_time=total_time,
                checkpoint_path=final_checkpoint_path,
                model_registry_path=None,  # Could register with model registry if needed
                tensorboard_log_dir=self.training_config.log_dir,
                training_history=self.metrics_history
            )
            
            logger.info("Single-process training completed successfully")
            logger.info(result.summary())
            
            return result
            
        except Exception as e:
            logger.error(f"Single-process training failed: {e}", exc_info=True)
            
            total_time = time.time() - self.training_start_time if self.training_start_time else 0.0
            
            return TrainingResult(
                success=False,
                final_metrics=TrainingMetrics(),
                total_training_time=total_time,
                checkpoint_path="",
                error_message=str(e)
            )
    
    def _collect_rollout_single_process(
        self, env, network, device
    ) -> Tuple[List[torch.Tensor], List[int], List[float], List[float], List[torch.Tensor]]:
        """
        Collect a single rollout in single-process mode.
        
        Args:
            env: Routing environment
            network: ActorCriticNetwork
            device: PyTorch device
        
        Returns:
            Tuple of (states, actions, rewards, values, log_probs)
        """
        import torch
        
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        
        # Reset environment
        state = env.reset()
        done = False
        
        # Collect rollout
        max_steps = self.ppo_config.rollout_fragment_length
        steps = 0
        
        while not done and steps < max_steps:
            # Convert state to tensor
            state_tensor = torch.from_numpy(state.grid).float().unsqueeze(0).to(device)
            
            # Get action and value from network
            with torch.no_grad():
                action_logits, value = network(state_tensor)
                action_probs = torch.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action.item())
            
            # Store transition
            states.append(state_tensor)
            actions.append(action.item())
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob)
            
            # Update state
            state = next_state
            steps += 1
        
        return states, actions, rewards, values, log_probs
    
    def _compute_gae(
        self, rewards: List[float], values: List[float], gamma: float, gae_lambda: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = np.zeros(len(rewards))
        returns = np.zeros(len(rewards))
        
        gae = 0
        next_value = 0  # Assume terminal state has value 0
        
        # Compute advantages backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def _update_policy_single_process(
        self,
        network,
        optimizer,
        states: List[torch.Tensor],
        actions: List[int],
        old_log_probs: List[torch.Tensor],
        advantages: np.ndarray,
        returns: np.ndarray,
        device
    ) -> Tuple[float, float, float]:
        """
        Update policy using PPO in single-process mode.
        
        Args:
            network: ActorCriticNetwork
            optimizer: Optimizer
            states: List of state tensors
            actions: List of actions
            old_log_probs: List of old log probabilities
            advantages: Advantage estimates
            returns: Return estimates
            device: PyTorch device
        
        Returns:
            Tuple of (policy_loss, value_loss, entropy)
        """
        import torch
        import torch.nn as nn
        
        # Convert to tensors
        states_tensor = torch.cat(states, dim=0).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
        old_log_probs_tensor = torch.stack(old_log_probs).to(device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
        
        # Normalize advantages
        if self.ppo_config.normalize_advantage:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO update for multiple epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        for epoch in range(self.ppo_config.n_epochs):
            # Forward pass
            action_logits, values = network(states_tensor)
            
            # Compute action probabilities
            action_probs = torch.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            
            # Compute log probabilities
            log_probs = action_dist.log_prob(actions_tensor)
            
            # Compute entropy
            entropy = action_dist.entropy().mean()
            
            # Compute ratio for PPO
            ratio = torch.exp(log_probs - old_log_probs_tensor)
            
            # Compute surrogate losses
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_config.clip_range, 1.0 + self.ppo_config.clip_range) * advantages_tensor
            
            # Policy loss (negative because we want to maximize)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns_tensor)
            
            # Total loss
            loss = (
                policy_loss
                + self.ppo_config.value_coef * value_loss
                - self.ppo_config.entropy_coef * entropy
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.ppo_config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(network.parameters(), self.ppo_config.max_grad_norm)
            
            optimizer.step()
            
            # Track losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        # Average losses over epochs
        avg_policy_loss = total_policy_loss / self.ppo_config.n_epochs
        avg_value_loss = total_value_loss / self.ppo_config.n_epochs
        avg_entropy = total_entropy / self.ppo_config.n_epochs
        
        return avg_policy_loss, avg_value_loss, avg_entropy
    
    def _save_checkpoint_single_process(
        self, network, optimizer, timesteps: int, is_final: bool = False
    ) -> str:
        """
        Save checkpoint in single-process mode.
        
        Args:
            network: ActorCriticNetwork
            optimizer: Optimizer
            timesteps: Current timestep count
            is_final: Whether this is the final checkpoint
        
        Returns:
            Path to saved checkpoint
        """
        import torch
        
        checkpoint_dir = Path(self.training_config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if is_final:
            checkpoint_path = checkpoint_dir / "final_single_process.pt"
        else:
            checkpoint_path = checkpoint_dir / f"checkpoint_single_process_{timesteps}.pt"
        
        # Save checkpoint
        torch.save({
            "timesteps": timesteps,
            "network_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "ppo_config": self.ppo_config.to_dict(),
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return str(checkpoint_path)


def create_rl_trainer(
    num_workers: int = 16,
    num_gpu_workers: int = 0,
    total_timesteps: int = 10_000_000,
    checkpoint_frequency: int = 100_000,
    use_gpu: bool = True,
    config_path: Optional[str] = None
) -> RLRoutingTrainer:
    """
    Create RL routing trainer with simplified configuration.
    
    This is a convenience function for creating a trainer with common settings.
    
    Args:
        num_workers: Number of Ray workers for distributed training
        num_gpu_workers: Number of GPU workers
        total_timesteps: Total training timesteps
        checkpoint_frequency: Steps between checkpoints
        use_gpu: Whether to use GPU acceleration
        config_path: Optional path to PPO config YAML file
    
    Returns:
        Configured RLRoutingTrainer instance
    
    Example:
        # Create trainer with 16 workers
        trainer = create_rl_trainer(num_workers=16, total_timesteps=10_000_000)
        
        # Train
        result = trainer.train()
        print(result.summary())
    """
    from src.training.ray_cluster import create_cluster_manager
    from src.training.ppo_config import load_ppo_config
    from src.services.model_registry import ModelRegistry
    
    # Create Ray cluster manager
    cluster_manager = create_cluster_manager(
        mode="local",  # Will auto-detect Kubernetes if available
        num_workers=num_workers,
        num_gpu_workers=num_gpu_workers
    )
    
    # Load PPO config
    ppo_config = load_ppo_config(config_path=config_path)
    ppo_config.num_workers = num_workers
    ppo_config.use_gpu = use_gpu
    
    # Create training config
    training_config = RLTrainingConfig(
        total_timesteps=total_timesteps,
        checkpoint_frequency=checkpoint_frequency,
        use_gpu=use_gpu
    )
    
    # Create model registry
    model_registry = ModelRegistry()
    
    # Create trainer
    trainer = RLRoutingTrainer(
        cluster_manager=cluster_manager,
        ppo_config=ppo_config,
        training_config=training_config,
        model_registry=model_registry
    )
    
    return trainer
