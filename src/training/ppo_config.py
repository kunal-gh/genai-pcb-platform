"""
PPO (Proximal Policy Optimization) configuration for RL routing agent.

This module provides PPO algorithm configuration for both Ray RLlib and Stable Baselines3,
with the hyperparameters specified in the design document:
- Learning rate: 3e-4
- Discount factor (γ): 0.99
- GAE lambda (λ): 0.95
- PPO clip range (ε): 0.2
- Value function coefficient: 0.5
- Entropy coefficient: 0.01
- Batch size: 256
- Mini-batch size: 64
- Epochs per update: 10
"""

import os
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


logger = logging.getLogger(__name__)


# Hyperparameter validation ranges for PPO
PPO_VALIDATION_RANGES = {
    # Core PPO hyperparameters
    "learning_rate": (1e-6, 1e-2),
    "gamma": (0.9, 0.999),
    "gae_lambda": (0.8, 1.0),
    "clip_range": (0.1, 0.5),
    "value_coef": (0.1, 1.0),
    "entropy_coef": (0.0, 0.1),
    
    # Batch sizes
    "batch_size": (32, 4096),
    "minibatch_size": (16, 512),
    "n_epochs": (1, 20),
    
    # Network architecture
    "hidden_dims": (64, 1024),
    
    # Training
    "max_grad_norm": (0.1, 10.0),
    "target_kl": (0.001, 0.1),
}


@dataclass
class PPOConfig:
    """
    PPO algorithm configuration compatible with Ray RLlib and Stable Baselines3.
    
    This configuration class provides:
    - Default hyperparameters matching the design specification
    - Support for both Ray RLlib and Stable Baselines3 backends
    - YAML configuration loading
    - Environment variable overrides
    - Hyperparameter validation
    
    Attributes:
        learning_rate: Learning rate for policy and value networks
        gamma: Discount factor for future rewards
        gae_lambda: GAE (Generalized Advantage Estimation) lambda parameter
        clip_range: PPO clipping parameter (epsilon)
        value_coef: Value function loss coefficient
        entropy_coef: Entropy bonus coefficient for exploration
        batch_size: Total batch size for training
        minibatch_size: Mini-batch size for SGD updates
        n_epochs: Number of epochs per training iteration
        max_grad_norm: Maximum gradient norm for clipping
        target_kl: Target KL divergence for early stopping
        normalize_advantage: Whether to normalize advantages
        use_gae: Whether to use Generalized Advantage Estimation
        backend: RL framework backend ("rllib" or "stable_baselines3")
    """
    
    # Core PPO hyperparameters (from design specification)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Batch configuration (from design specification)
    batch_size: int = 256
    minibatch_size: int = 64
    n_epochs: int = 10
    
    # Additional PPO parameters
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None  # None means no early stopping
    normalize_advantage: bool = True
    use_gae: bool = True
    
    # Network architecture
    hidden_dims: tuple = (512, 256)
    shared_features: bool = True
    
    # Backend selection
    backend: str = "rllib"  # "rllib" or "stable_baselines3"
    
    # Training configuration
    num_workers: int = 16
    num_envs_per_worker: int = 4
    rollout_fragment_length: int = 200
    train_batch_size: int = 4000
    sgd_minibatch_size: int = 256
    num_sgd_iter: int = 10
    
    # Device configuration
    use_gpu: bool = True
    num_gpus: int = 1
    num_gpus_per_worker: float = 0.0
    
    # Checkpointing
    checkpoint_frequency: int = 100000  # steps
    checkpoint_dir: str = "checkpoints/rl_routing"
    
    # Logging
    log_dir: str = "logs/rl_routing"
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate hyperparameters after initialization."""
        self._validate_hyperparameters()
        self._validate_backend()
    
    def _validate_hyperparameters(self):
        """
        Validate all hyperparameters against acceptable ranges.
        
        Raises:
            ValueError: If any hyperparameter is outside acceptable range
        """
        for param_name, (min_val, max_val) in PPO_VALIDATION_RANGES.items():
            if hasattr(self, param_name):
                value = getattr(self, param_name)
                
                # Handle tuple parameters (like hidden_dims)
                if isinstance(value, (tuple, list)):
                    for v in value:
                        if not isinstance(v, (int, float)):
                            continue
                        if not (min_val <= v <= max_val):
                            raise ValueError(
                                f"Hyperparameter '{param_name}' value {v} is outside "
                                f"acceptable range [{min_val}, {max_val}]"
                            )
                elif isinstance(value, (int, float)):
                    if not (min_val <= value <= max_val):
                        raise ValueError(
                            f"Hyperparameter '{param_name}' value {value} is outside "
                            f"acceptable range [{min_val}, {max_val}]"
                        )
        
        # Validate batch size relationships
        if self.minibatch_size > self.batch_size:
            raise ValueError(
                f"Mini-batch size ({self.minibatch_size}) cannot be larger than "
                f"batch size ({self.batch_size})"
            )
        
        if self.batch_size % self.minibatch_size != 0:
            logger.warning(
                f"Batch size ({self.batch_size}) is not evenly divisible by "
                f"mini-batch size ({self.minibatch_size}). This may cause issues."
            )
        
        logger.info("All PPO hyperparameters validated successfully")
    
    def _validate_backend(self):
        """
        Validate backend selection.
        
        Raises:
            ValueError: If backend is not supported
        """
        valid_backends = ["rllib", "stable_baselines3"]
        if self.backend not in valid_backends:
            raise ValueError(
                f"Backend '{self.backend}' is not supported. "
                f"Valid options: {valid_backends}"
            )
    
    def log_hyperparameters(self):
        """Log all active hyperparameters for reproducibility."""
        logger.info("=" * 80)
        logger.info("PPO Configuration:")
        logger.info("=" * 80)
        
        # Group parameters by category
        categories = {
            "Core PPO Parameters": [
                "learning_rate", "gamma", "gae_lambda", "clip_range",
                "value_coef", "entropy_coef"
            ],
            "Batch Configuration": [
                "batch_size", "minibatch_size", "n_epochs"
            ],
            "Additional PPO Parameters": [
                "max_grad_norm", "target_kl", "normalize_advantage", "use_gae"
            ],
            "Network Architecture": [
                "hidden_dims", "shared_features"
            ],
            "Backend": [
                "backend"
            ],
            "Distributed Training": [
                "num_workers", "num_envs_per_worker", "rollout_fragment_length",
                "train_batch_size", "sgd_minibatch_size", "num_sgd_iter"
            ],
            "Device Configuration": [
                "use_gpu", "num_gpus", "num_gpus_per_worker"
            ],
            "Checkpointing": [
                "checkpoint_frequency", "checkpoint_dir"
            ],
            "Logging": [
                "log_dir", "log_level"
            ],
        }
        
        for category, params in categories.items():
            logger.info(f"\n{category}:")
            for param in params:
                if hasattr(self, param):
                    value = getattr(self, param)
                    logger.info(f"  {param}: {value}")
        
        logger.info("=" * 80)
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "PPOConfig":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            PPOConfig instance with values from YAML file
            
        Raises:
            FileNotFoundError: If config file does not exist
            yaml.YAMLError: If config file is not valid YAML
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Loading PPO configuration from: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            config_dict = {}
        
        # Create config instance from YAML data
        config = cls(**config_dict)
        logger.info(f"PPO configuration loaded successfully from {config_path}")
        
        return config
    
    @classmethod
    def from_yaml_with_overrides(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        env_prefix: str = "PPO_"
    ) -> "PPOConfig":
        """
        Load configuration from YAML file with environment variable overrides.
        
        This method:
        1. Loads base configuration from YAML file (if provided)
        2. Applies environment variable overrides for critical hyperparameters
        3. Uses default values when config file is not provided
        
        Environment variable naming convention:
        - PPO_LEARNING_RATE -> learning_rate
        - PPO_GAMMA -> gamma
        - PPO_BATCH_SIZE -> batch_size
        
        Args:
            config_path: Optional path to YAML configuration file
            env_prefix: Prefix for environment variables (default: "PPO_")
            
        Returns:
            PPOConfig instance with YAML values and environment overrides
        """
        # Start with defaults or YAML config
        if config_path is not None:
            config_path = Path(config_path)
            if config_path.exists():
                config = cls.from_yaml(config_path)
                config_dict = {f.name: getattr(config, f.name) for f in fields(config)}
            else:
                logger.warning(
                    f"Configuration file not found: {config_path}. "
                    "Using default PPO configuration."
                )
                config_dict = {}
        else:
            logger.info("No configuration file provided. Using default PPO configuration.")
            config_dict = {}
        
        # Apply environment variable overrides
        overrides = {}
        for field_info in fields(cls):
            env_var_name = f"{env_prefix}{field_info.name.upper()}"
            env_value = os.environ.get(env_var_name)
            
            if env_value is not None:
                # Convert environment variable to appropriate type
                try:
                    field_type = field_info.type
                    
                    # Handle Optional types
                    if hasattr(field_type, '__origin__'):
                        if field_type.__origin__ is Union:
                            # Get the non-None type from Optional
                            field_type = [t for t in field_type.__args__ if t is not type(None)][0]
                    
                    if field_type == bool:
                        converted_value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif field_type == int:
                        converted_value = int(env_value)
                    elif field_type == float:
                        converted_value = float(env_value)
                    elif field_type == tuple:
                        # Parse tuple from comma-separated string
                        converted_value = tuple(int(x.strip()) for x in env_value.split(','))
                    else:
                        converted_value = env_value
                    
                    overrides[field_info.name] = converted_value
                    logger.info(
                        f"Environment override: {field_info.name} = {converted_value} "
                        f"(from {env_var_name})"
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse environment variable {env_var_name}={env_value}: {e}. "
                        "Ignoring override."
                    )
        
        # Merge config_dict with overrides
        config_dict.update(overrides)
        
        # Create final config instance
        config = cls(**config_dict)
        
        return config
    
    def to_yaml(self, output_path: Union[str, Path]):
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Path where YAML file should be saved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        
        # Convert tuples to lists for YAML serialization
        for key, value in config_dict.items():
            if isinstance(value, tuple):
                config_dict[key] = list(value)
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"PPO configuration saved to: {output_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary with all configuration parameters
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}
    
    def to_rllib_config(self) -> Dict[str, Any]:
        """
        Convert to Ray RLlib PPO configuration dictionary.
        
        Returns:
            Dictionary compatible with Ray RLlib PPOConfig
        """
        if self.backend != "rllib":
            logger.warning(
                f"Converting to RLlib config but backend is set to '{self.backend}'"
            )
        
        config = {
            # PPO-specific parameters
            "lr": self.learning_rate,
            "gamma": self.gamma,
            "lambda": self.gae_lambda,
            "clip_param": self.clip_range,
            "vf_loss_coeff": self.value_coef,
            "entropy_coeff": self.entropy_coef,
            
            # Training parameters
            "train_batch_size": self.train_batch_size,
            "sgd_minibatch_size": self.sgd_minibatch_size,
            "num_sgd_iter": self.num_sgd_iter,
            
            # Rollout parameters
            "num_workers": self.num_workers,
            "num_envs_per_worker": self.num_envs_per_worker,
            "rollout_fragment_length": self.rollout_fragment_length,
            
            # GAE parameters
            "use_gae": self.use_gae,
            
            # Gradient clipping
            "grad_clip": self.max_grad_norm,
            
            # KL divergence
            "kl_coeff": 0.2,  # Initial KL coefficient
            "kl_target": self.target_kl if self.target_kl is not None else 0.01,
            
            # Advantage normalization
            "normalize_advantages": self.normalize_advantage,
            
            # GPU configuration
            "num_gpus": self.num_gpus if self.use_gpu else 0,
            "num_gpus_per_worker": self.num_gpus_per_worker,
            
            # Framework
            "framework": "torch",
            
            # Model configuration
            "model": {
                "custom_model": None,  # Will be set by trainer
                "custom_model_config": {
                    "hidden_dims": self.hidden_dims,
                    "shared_features": self.shared_features,
                },
            },
        }
        
        return config
    
    def to_stable_baselines3_config(self) -> Dict[str, Any]:
        """
        Convert to Stable Baselines3 PPO configuration dictionary.
        
        Returns:
            Dictionary compatible with Stable Baselines3 PPO
        """
        if self.backend != "stable_baselines3":
            logger.warning(
                f"Converting to Stable Baselines3 config but backend is set to '{self.backend}'"
            )
        
        config = {
            # PPO-specific parameters
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "vf_coef": self.value_coef,
            "ent_coef": self.entropy_coef,
            
            # Training parameters
            "n_steps": self.rollout_fragment_length,
            "batch_size": self.minibatch_size,
            "n_epochs": self.n_epochs,
            
            # Gradient clipping
            "max_grad_norm": self.max_grad_norm,
            
            # KL divergence
            "target_kl": self.target_kl,
            
            # Advantage normalization
            "normalize_advantage": self.normalize_advantage,
            
            # Use GAE
            "use_sde": False,  # State-dependent exploration (not used)
            
            # Device
            "device": "cuda" if self.use_gpu else "cpu",
            
            # Logging
            "verbose": 1 if self.log_level == "INFO" else 0,
        }
        
        return config


def create_ppo_trainer(
    config: PPOConfig,
    env_creator,
    model_class=None
):
    """
    Create PPO trainer instance based on backend configuration.
    
    This function creates a PPO trainer using either Ray RLlib or Stable Baselines3
    based on the backend specified in the configuration.
    
    Args:
        config: PPO configuration
        env_creator: Function that creates the environment
        model_class: Optional custom model class (for RLlib)
    
    Returns:
        Trainer instance (Ray RLlib Trainer or Stable Baselines3 PPO)
    
    Raises:
        ImportError: If required backend library is not installed
        ValueError: If backend is not supported
    """
    config.log_hyperparameters()
    
    if config.backend == "rllib":
        try:
            from ray import tune
            from ray.rllib.algorithms.ppo import PPO, PPOConfig as RLlibPPOConfig
            from ray.tune.registry import register_env
        except ImportError:
            raise ImportError(
                "Ray RLlib is not installed. Install with: pip install ray[rllib]"
            )
        
        # Register environment
        register_env("routing_env", env_creator)
        
        # Create RLlib PPO config
        rllib_config = config.to_rllib_config()
        
        # Create PPO trainer
        ppo_config = RLlibPPOConfig()
        ppo_config.update_from_dict(rllib_config)
        ppo_config.environment("routing_env")
        
        # Set custom model if provided
        if model_class is not None:
            from ray.rllib.models import ModelCatalog
            ModelCatalog.register_custom_model("routing_policy", model_class)
            ppo_config.training(model={"custom_model": "routing_policy"})
        
        trainer = ppo_config.build()
        
        logger.info("Created Ray RLlib PPO trainer")
        return trainer
    
    elif config.backend == "stable_baselines3":
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            raise ImportError(
                "Stable Baselines3 is not installed. Install with: pip install stable-baselines3"
            )
        
        # Create vectorized environment
        env = DummyVecEnv([env_creator])
        
        # Create Stable Baselines3 PPO config
        sb3_config = config.to_stable_baselines3_config()
        
        # Create PPO trainer
        trainer = PPO(
            policy="CnnPolicy" if model_class is None else model_class,
            env=env,
            **sb3_config
        )
        
        logger.info("Created Stable Baselines3 PPO trainer")
        return trainer
    
    else:
        raise ValueError(f"Unsupported backend: {config.backend}")


def load_ppo_config(
    config_path: Optional[Union[str, Path]] = None,
    env_prefix: str = "PPO_"
) -> PPOConfig:
    """
    Load PPO configuration with YAML and environment variable support.
    
    This is the main entry point for loading PPO configuration. It:
    1. Loads from YAML file if provided and exists
    2. Uses default values if config file is missing
    3. Applies environment variable overrides for critical hyperparameters
    4. Validates all hyperparameters
    5. Logs all active hyperparameters
    
    Args:
        config_path: Optional path to YAML configuration file
        env_prefix: Prefix for environment variables (default: "PPO_")
        
    Returns:
        Validated PPOConfig instance
        
    Example:
        # Load with defaults
        config = load_ppo_config()
        
        # Load from YAML
        config = load_ppo_config("config/ppo.yaml")
        
        # Load with environment overrides
        # Set: PPO_LEARNING_RATE=0.0003, PPO_BATCH_SIZE=256
        config = load_ppo_config("config/ppo.yaml")
    """
    config = PPOConfig.from_yaml_with_overrides(
        config_path=config_path,
        env_prefix=env_prefix
    )
    
    # Log all hyperparameters for reproducibility
    config.log_hyperparameters()
    
    return config
