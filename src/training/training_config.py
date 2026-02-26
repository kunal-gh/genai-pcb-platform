"""
Training configuration management with YAML loading, environment variable overrides,
and hyperparameter validation.

This module provides a flexible configuration system that:
- Loads hyperparameters from YAML configuration files
- Supports environment variable overrides for critical hyperparameters
- Validates all hyperparameters against acceptable ranges
- Uses sensible defaults when configuration file is not provided
- Logs all active hyperparameters at startup for reproducibility
"""

import os
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
import torch


logger = logging.getLogger(__name__)


# Hyperparameter validation ranges
VALIDATION_RANGES = {
    # Dataset
    "batch_size": (1, 256),
    "num_workers": (0, 32),
    
    # Model architecture
    "node_feature_dim": (8, 512),
    "edge_feature_dim": (8, 512),
    "hidden_dim": (32, 1024),
    "num_layers": (1, 16),
    "dropout": (0.0, 0.9),
    
    # Optimization
    "learning_rate": (1e-6, 1e-1),
    "beta1": (0.0, 1.0),
    "beta2": (0.0, 1.0),
    "weight_decay": (0.0, 1.0),
    
    # Learning rate scheduling
    "lr_scheduler_patience": (1, 100),
    "lr_scheduler_factor": (0.01, 0.99),
    "lr_scheduler_min_lr": (1e-10, 1e-3),
    
    # Training
    "num_epochs": (1, 1000),
    "early_stopping_patience": (1, 200),
    "checkpoint_every": (1, 100),
    
    # Data augmentation
    "augmentation_prob": (0.0, 1.0),
}


@dataclass
class TrainingConfig:
    """
    Training configuration with YAML loading and environment variable overrides.
    
    This configuration class supports:
    - Loading from YAML files
    - Environment variable overrides (prefix: FALCON_)
    - Hyperparameter validation
    - Default values when config file is missing
    """
    
    # Dataset
    dataset_path: str = "data/circuitnet"
    batch_size: int = 32
    num_workers: int = 4
    
    # Model architecture
    node_feature_dim: int = 64
    edge_feature_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1
    
    # Optimization
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    
    # Learning rate scheduling
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-6
    
    # Training
    num_epochs: int = 100
    early_stopping_patience: int = 20
    checkpoint_every: int = 5
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # Logging
    log_dir: str = "logs/falcon_gnn"
    checkpoint_dir: str = "checkpoints/falcon_gnn"
    
    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Model registry
    model_registry_path: str = "models"
    model_version: str = "v1.0.0"
    
    def __post_init__(self):
        """Validate hyperparameters after initialization."""
        self._validate_hyperparameters()
    
    def _validate_hyperparameters(self):
        """
        Validate all hyperparameters against acceptable ranges.
        
        Raises:
            ValueError: If any hyperparameter is outside acceptable range
        """
        for param_name, (min_val, max_val) in VALIDATION_RANGES.items():
            if hasattr(self, param_name):
                value = getattr(self, param_name)
                if not isinstance(value, (int, float)):
                    continue
                    
                if not (min_val <= value <= max_val):
                    raise ValueError(
                        f"Hyperparameter '{param_name}' value {value} is outside "
                        f"acceptable range [{min_val}, {max_val}]"
                    )
        
        logger.info("All hyperparameters validated successfully")
    
    def log_hyperparameters(self):
        """Log all active hyperparameters for reproducibility."""
        logger.info("=" * 80)
        logger.info("Active Training Configuration:")
        logger.info("=" * 80)
        
        # Group parameters by category
        categories = {
            "Dataset": ["dataset_path", "batch_size", "num_workers"],
            "Model Architecture": ["node_feature_dim", "edge_feature_dim", "hidden_dim", "num_layers", "dropout"],
            "Optimization": ["learning_rate", "beta1", "beta2", "weight_decay"],
            "Learning Rate Scheduling": ["lr_scheduler_patience", "lr_scheduler_factor", "lr_scheduler_min_lr"],
            "Training": ["num_epochs", "early_stopping_patience", "checkpoint_every"],
            "Data Augmentation": ["use_augmentation", "augmentation_prob"],
            "Logging": ["log_dir", "checkpoint_dir"],
            "Device": ["device"],
            "Model Registry": ["model_registry_path", "model_version"],
        }
        
        for category, params in categories.items():
            logger.info(f"\n{category}:")
            for param in params:
                if hasattr(self, param):
                    value = getattr(self, param)
                    logger.info(f"  {param}: {value}")
        
        logger.info("=" * 80)
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "TrainingConfig":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            TrainingConfig instance with values from YAML file
            
        Raises:
            FileNotFoundError: If config file does not exist
            yaml.YAMLError: If config file is not valid YAML
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Loading configuration from: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            config_dict = {}
        
        # Create config instance from YAML data
        config = cls(**config_dict)
        logger.info(f"Configuration loaded successfully from {config_path}")
        
        return config
    
    @classmethod
    def from_yaml_with_overrides(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        env_prefix: str = "FALCON_"
    ) -> "TrainingConfig":
        """
        Load configuration from YAML file with environment variable overrides.
        
        This method:
        1. Loads base configuration from YAML file (if provided)
        2. Applies environment variable overrides for critical hyperparameters
        3. Uses default values when config file is not provided
        
        Environment variable naming convention:
        - FALCON_LEARNING_RATE -> learning_rate
        - FALCON_BATCH_SIZE -> batch_size
        - FALCON_HIDDEN_DIM -> hidden_dim
        
        Args:
            config_path: Optional path to YAML configuration file
            env_prefix: Prefix for environment variables (default: "FALCON_")
            
        Returns:
            TrainingConfig instance with YAML values and environment overrides
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
                    "Using default configuration."
                )
                config_dict = {}
        else:
            logger.info("No configuration file provided. Using default configuration.")
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
                        field_type = field_type.__args__[0]
                    
                    if field_type == bool:
                        converted_value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif field_type == int:
                        converted_value = int(env_value)
                    elif field_type == float:
                        converted_value = float(env_value)
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
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to: {output_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary with all configuration parameters
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}


def load_training_config(
    config_path: Optional[Union[str, Path]] = None,
    env_prefix: str = "FALCON_"
) -> TrainingConfig:
    """
    Load training configuration with YAML and environment variable support.
    
    This is the main entry point for loading training configuration. It:
    1. Loads from YAML file if provided and exists
    2. Uses default values if config file is missing
    3. Applies environment variable overrides for critical hyperparameters
    4. Validates all hyperparameters
    5. Logs all active hyperparameters
    
    Args:
        config_path: Optional path to YAML configuration file
        env_prefix: Prefix for environment variables (default: "FALCON_")
        
    Returns:
        Validated TrainingConfig instance
        
    Example:
        # Load with defaults
        config = load_training_config()
        
        # Load from YAML
        config = load_training_config("config/training.yaml")
        
        # Load with environment overrides
        # Set: FALCON_LEARNING_RATE=0.001, FALCON_BATCH_SIZE=64
        config = load_training_config("config/training.yaml")
    """
    config = TrainingConfig.from_yaml_with_overrides(
        config_path=config_path,
        env_prefix=env_prefix
    )
    
    # Log all hyperparameters for reproducibility
    config.log_hyperparameters()
    
    return config
