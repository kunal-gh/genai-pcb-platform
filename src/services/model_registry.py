"""
Model registry for versioning and managing trained AI models.

Handles model storage, versioning, loading, and metadata management
for FALCON GNN and RL Routing Engine models.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import torch

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Model version metadata."""
    model_type: str  # "falcon_gnn" or "rl_routing"
    version: str  # e.g., "v1.0.0"
    training_date: str
    dataset_version: str
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    git_commit_hash: Optional[str] = None
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary."""
        return cls(**data)


class ModelRegistry:
    """
    Model registry for versioning and managing trained models.
    
    Directory structure:
    models/
    ├── falcon_gnn/
    │   ├── v1.0.0/
    │   │   ├── model.pt
    │   │   ├── metadata.json
    │   │   └── config.yaml
    │   ├── v1.1.0/
    │   └── latest -> v1.1.0
    └── rl_routing/
        ├── v1.0.0/
        ├── v1.1.0/
        └── latest -> v1.1.0
    """
    
    def __init__(self, registry_path: str = "models"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to model registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized model registry at {self.registry_path}")
    
    def register_model(
        self,
        model_type: str,
        model: torch.nn.Module,
        version: str,
        metadata: Dict[str, Any],
    ) -> str:
        """
        Register a trained model with version metadata.
        
        Args:
            model_type: Type of model ("falcon_gnn" or "rl_routing")
            model: PyTorch model to register
            version: Version string (e.g., "v1.0.0")
            metadata: Model metadata including:
                - training_date
                - dataset_version
                - performance_metrics
                - hyperparameters
                - git_commit_hash (optional)
                - description (optional)
                
        Returns:
            Path to registered model directory
        """
        if model_type not in ["falcon_gnn", "rl_routing"]:
            raise ValueError(f"Invalid model_type: {model_type}")
        
        # Create version directory
        model_dir = self.registry_path / model_type / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_path = model_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model weights to {model_path}")
        
        # Create model version metadata
        model_version = ModelVersion(
            model_type=model_type,
            version=version,
            training_date=metadata.get('training_date', datetime.now().isoformat()),
            dataset_version=metadata.get('dataset_version', 'unknown'),
            performance_metrics=metadata.get('performance_metrics', {}),
            hyperparameters=metadata.get('hyperparameters', {}),
            git_commit_hash=metadata.get('git_commit_hash'),
            description=metadata.get('description', ''),
        )
        
        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_version.to_dict(), f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Update latest symlink
        self._update_latest_link(model_type, version)
        
        # Cleanup old versions (keep last 10)
        self._cleanup_old_versions(model_type, keep=10)
        
        return str(model_dir)
    
    def load_model(
        self,
        model_type: str,
        model_class: type,
        version: Optional[str] = None,
        **model_kwargs
    ) -> torch.nn.Module:
        """
        Load model by type and version.
        
        Args:
            model_type: Type of model ("falcon_gnn" or "rl_routing")
            model_class: Model class to instantiate
            version: Version string (default: latest)
            **model_kwargs: Arguments to pass to model constructor
            
        Returns:
            Loaded PyTorch model
        """
        if model_type not in ["falcon_gnn", "rl_routing"]:
            raise ValueError(f"Invalid model_type: {model_type}")
        
        # Use latest version if not specified
        if version is None:
            version = self._get_latest_version(model_type)
            if version is None:
                raise FileNotFoundError(f"No models found for {model_type}")
        
        model_dir = self.registry_path / model_type / version
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")
        
        # Load metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        model_version = ModelVersion.from_dict(metadata)
        
        # Validate architecture compatibility
        self._validate_architecture_compatibility(model_version, model_kwargs)
        
        # Instantiate model
        try:
            model = model_class(**model_kwargs)
        except Exception as e:
            logger.error(f"Failed to instantiate model: {e}")
            # Try fallback to previous version
            return self._load_fallback_model(model_type, model_class, version, **model_kwargs)
        
        # Load weights
        model_path = model_dir / "model.pt"
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            logger.info(f"Loaded model {model_type} {version}")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            # Try fallback to previous version
            return self._load_fallback_model(model_type, model_class, version, **model_kwargs)
        
        return model
    
    def list_versions(
        self,
        model_type: str,
        limit: int = 10
    ) -> List[ModelVersion]:
        """
        List available model versions with metadata.
        
        Args:
            model_type: Type of model ("falcon_gnn" or "rl_routing")
            limit: Maximum number of versions to return
            
        Returns:
            List of ModelVersion objects, sorted by date (newest first)
        """
        if model_type not in ["falcon_gnn", "rl_routing"]:
            raise ValueError(f"Invalid model_type: {model_type}")
        
        model_type_dir = self.registry_path / model_type
        if not model_type_dir.exists():
            return []
        
        versions = []
        for version_dir in model_type_dir.iterdir():
            if version_dir.is_dir() and version_dir.name != "latest":
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    versions.append(ModelVersion.from_dict(metadata))
        
        # Sort by training date (newest first)
        versions.sort(key=lambda v: v.training_date, reverse=True)
        
        return versions[:limit]
    
    def get_model_info(
        self,
        model_type: str,
        version: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """
        Get model version information.
        
        Args:
            model_type: Type of model
            version: Version string (default: latest)
            
        Returns:
            ModelVersion object or None if not found
        """
        if version is None:
            version = self._get_latest_version(model_type)
            if version is None:
                return None
        
        metadata_path = self.registry_path / model_type / version / "metadata.json"
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return ModelVersion.from_dict(metadata)
    
    def _get_latest_version(self, model_type: str) -> Optional[str]:
        """Get latest version for model type."""
        latest_link = self.registry_path / model_type / "latest"
        if latest_link.exists() and latest_link.is_symlink():
            return latest_link.readlink().name
        
        # Fallback: find most recent version
        versions = self.list_versions(model_type, limit=1)
        if versions:
            return versions[0].version
        
        return None
    
    def _update_latest_link(self, model_type: str, version: str):
        """Update 'latest' symlink to point to specified version."""
        latest_link = self.registry_path / model_type / "latest"
        target = Path(version)
        
        # Remove existing symlink
        if latest_link.exists():
            latest_link.unlink()
        
        # Create new symlink
        try:
            latest_link.symlink_to(target)
            logger.info(f"Updated latest link for {model_type} to {version}")
        except OSError as e:
            logger.warning(f"Failed to create symlink: {e}")
    
    def _cleanup_old_versions(self, model_type: str, keep: int = 10):
        """Remove old model versions, keeping only the most recent."""
        versions = self.list_versions(model_type, limit=1000)
        
        if len(versions) <= keep:
            return
        
        # Remove oldest versions
        for version in versions[keep:]:
            version_dir = self.registry_path / model_type / version.version
            if version_dir.exists():
                shutil.rmtree(version_dir)
                logger.info(f"Removed old version: {version.version}")
    
    def _validate_architecture_compatibility(
        self,
        model_version: ModelVersion,
        model_kwargs: Dict[str, Any]
    ):
        """Validate that model architecture is compatible with current code."""
        # Check critical hyperparameters match
        stored_params = model_version.hyperparameters
        
        # For now, just log warnings if parameters don't match
        for key, value in model_kwargs.items():
            if key in stored_params and stored_params[key] != value:
                logger.warning(
                    f"Parameter mismatch for {key}: "
                    f"stored={stored_params[key]}, current={value}"
                )
    
    def _load_fallback_model(
        self,
        model_type: str,
        model_class: type,
        failed_version: str,
        **model_kwargs
    ) -> torch.nn.Module:
        """Load fallback model when primary loading fails."""
        logger.warning(f"Attempting to load fallback model for {model_type}")
        
        # Get all versions except the failed one
        versions = self.list_versions(model_type, limit=10)
        for version in versions:
            if version.version != failed_version:
                try:
                    return self.load_model(
                        model_type,
                        model_class,
                        version.version,
                        **model_kwargs
                    )
                except Exception as e:
                    logger.error(f"Fallback to {version.version} failed: {e}")
                    continue
        
        raise RuntimeError(f"All fallback attempts failed for {model_type}")
