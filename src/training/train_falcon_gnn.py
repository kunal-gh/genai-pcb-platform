"""
FALCON GNN training pipeline.

Implements the training loop for the FALCON GNN model on CircuitNet dataset,
including data loading, augmentation, optimization, checkpointing, and logging.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch
import numpy as np

from src.services.falcon_gnn import FalconGNN, create_falcon_gnn
from src.training.dataset_loader import CircuitNetDataset, create_dataloaders
from src.training.data_augmentation import augment_circuit
from src.services.model_registry import ModelRegistry
from src.training.training_config import TrainingConfig, load_training_config

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Training result."""
    success: bool
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    best_epoch: int
    total_epochs: int
    training_time: float
    model_path: str
    test_metrics: Dict[str, float]


class FalconGNNTrainer:
    """
    Training pipeline for FALCON GNN model.
    
    Implements:
    - Data loading and augmentation
    - Training loop with forward/backward passes
    - Adam optimizer with learning rate scheduling
    - MSE loss for parasitic prediction
    - Early stopping
    - Checkpoint saving
    - TensorBoard logging
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Create directories
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Initialize model registry
        self.model_registry = ModelRegistry(config.model_registry_path)
        
        # Device setup with CUDA detection
        self.device = self._setup_device(config.device)
        self._log_device_info()
        
        # Model, optimizer, scheduler (initialized in train())
        self.model: Optional[FalconGNN] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        logger.info("Initialized FalconGNNTrainer")
    
    def _setup_device(self, device_str: str) -> torch.device:
        """
        Set up compute device with CUDA detection.
        
        Args:
            device_str: Device string ('cuda', 'cpu', or 'cuda:0', etc.)
            
        Returns:
            torch.device object
        """
        # Check if CUDA is requested
        if 'cuda' in device_str.lower():
            if torch.cuda.is_available():
                # CUDA is available, use GPU
                device = torch.device(device_str)
                return device
            else:
                # CUDA requested but not available, fall back to CPU
                logger.info("CUDA requested but not available. Falling back to CPU.")
                return torch.device('cpu')
        else:
            # CPU explicitly requested
            return torch.device(device_str)
    
    def _log_device_info(self):
        """Log detailed device information at startup."""
        if self.device.type == 'cuda':
            # GPU information
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_memory = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)  # GB
            logger.info(f"Using GPU: {gpu_name}")
            logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")
            
            # Log current GPU memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
                reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
                logger.info(f"GPU Memory Allocated: {allocated:.2f} GB")
                logger.info(f"GPU Memory Reserved: {reserved:.2f} GB")
        else:
            # CPU information
            logger.info(f"Using device: CPU")
            logger.info("GPU acceleration not available. Training will use CPU.")

    
    def train(self) -> TrainingResult:
        """
        Train FALCON GNN model on CircuitNet dataset.
        
        Returns:
            TrainingResult with training metrics and model path
        """
        start_time = time.time()
        
        logger.info("Starting FALCON GNN training")
        logger.info(f"Configuration: {self.config}")
        
        # Load datasets
        logger.info("Loading datasets...")
        train_dataset, val_dataset, test_dataset = create_dataloaders(
            dataset_path=self.config.dataset_path,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
        
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty")
        
        logger.info(f"Dataset sizes: train={len(train_dataset)}, "
                   f"val={len(val_dataset)}, test={len(test_dataset)}")
        
        # Print dataset statistics
        train_stats = train_dataset.get_statistics()
        logger.info(f"Training dataset statistics: {train_stats}")
        
        # Initialize model
        logger.info("Initializing model...")
        self.model = create_falcon_gnn(
            node_feature_dim=self.config.node_feature_dim,
            edge_feature_dim=self.config.edge_feature_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        self.model = self.model.to(self.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model has {num_params:,} parameters")
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config.lr_scheduler_patience,
            factor=self.config.lr_scheduler_factor,
            min_lr=self.config.lr_scheduler_min_lr,
            verbose=True,
        )
        
        # Training loop
        logger.info("Starting training loop...")
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss, train_metrics = self._train_epoch(train_dataset)
            
            # Validate
            val_loss, val_metrics = self._validate_epoch(val_dataset)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self._log_metrics(epoch, train_loss, val_loss, train_metrics, val_metrics, current_lr)
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                
                # Save best model
                self._save_checkpoint(epoch, is_best=True)
                logger.info(f"New best model at epoch {epoch} with val_loss={val_loss:.6f}")
            else:
                self.epochs_without_improvement += 1
            
            # Checkpoint saving
            if (epoch + 1) % self.config.checkpoint_every == 0:
                self._save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best model for final evaluation
        self._load_best_checkpoint()
        
        # Final evaluation on test set
        logger.info("Evaluating on test set...")
        test_loss, test_metrics = self._validate_epoch(test_dataset)
        logger.info(f"Test loss: {test_loss:.6f}")
        logger.info(f"Test metrics: {test_metrics}")
        
        # Register model in model registry
        logger.info("Registering model in model registry...")
        model_path = self._register_model(test_metrics)
        
        # Training complete
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        result = TrainingResult(
            success=True,
            final_train_loss=train_loss,
            final_val_loss=val_loss,
            best_val_loss=self.best_val_loss,
            best_epoch=self.best_epoch,
            total_epochs=self.current_epoch + 1,
            training_time=training_time,
            model_path=model_path,
            test_metrics=test_metrics,
        )
        
        # Close TensorBoard writer
        self.writer.close()
        
        return result
    
    def _train_epoch(self, dataset: CircuitNetDataset) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            dataset: Training dataset
            
        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        total_mse_c = 0.0
        total_mse_l = 0.0
        
        for i, sample in enumerate(dataset):
            # Apply data augmentation
            circuit = sample.circuit_graph
            if self.config.use_augmentation and np.random.rand() < self.config.augmentation_prob:
                augmented_circuits = augment_circuit(
                    circuit,
                    rotation=True,
                    mirroring=True,
                    permutation=False,  # Skip permutation for now
                    noise=True,
                    noise_level=0.05,
                )
                # Use random augmented version
                circuit = np.random.choice(augmented_circuits)
            
            # Convert to PyTorch Geometric format
            data = self.model._circuit_to_pyg_data(circuit)
            data = data.to(self.device)
            
            # Get ground truth parasitics
            target_parasitics = self._get_target_parasitics(sample, circuit)
            if target_parasitics is None:
                continue
            target_parasitics = target_parasitics.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            _, predicted_parasitics = self.model(
                data.x,
                data.edge_index,
                data.edge_attr,
            )
            
            # Compute MSE loss
            loss = nn.functional.mse_loss(predicted_parasitics, target_parasitics)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_samples += 1
            
            # Compute per-component MSE
            mse_c = nn.functional.mse_loss(predicted_parasitics[:, 0], target_parasitics[:, 0])
            mse_l = nn.functional.mse_loss(predicted_parasitics[:, 1], target_parasitics[:, 1])
            total_mse_c += mse_c.item()
            total_mse_l += mse_l.item()
        
        avg_loss = total_loss / max(total_samples, 1)
        avg_mse_c = total_mse_c / max(total_samples, 1)
        avg_mse_l = total_mse_l / max(total_samples, 1)
        
        metrics = {
            'mse_capacitance': avg_mse_c,
            'mse_inductance': avg_mse_l,
        }
        
        return avg_loss, metrics
    
    def _validate_epoch(self, dataset: CircuitNetDataset) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.
        
        Args:
            dataset: Validation dataset
            
        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        total_mse_c = 0.0
        total_mse_l = 0.0
        total_mae_c = 0.0
        total_mae_l = 0.0
        
        with torch.no_grad():
            for sample in dataset:
                circuit = sample.circuit_graph
                
                # Convert to PyTorch Geometric format
                data = self.model._circuit_to_pyg_data(circuit)
                data = data.to(self.device)
                
                # Get ground truth parasitics
                target_parasitics = self._get_target_parasitics(sample, circuit)
                if target_parasitics is None:
                    continue
                target_parasitics = target_parasitics.to(self.device)
                
                # Forward pass
                _, predicted_parasitics = self.model(
                    data.x,
                    data.edge_index,
                    data.edge_attr,
                )
                
                # Compute MSE loss
                loss = nn.functional.mse_loss(predicted_parasitics, target_parasitics)
                
                # Accumulate metrics
                total_loss += loss.item()
                total_samples += 1
                
                # Compute per-component metrics
                mse_c = nn.functional.mse_loss(predicted_parasitics[:, 0], target_parasitics[:, 0])
                mse_l = nn.functional.mse_loss(predicted_parasitics[:, 1], target_parasitics[:, 1])
                mae_c = torch.mean(torch.abs(predicted_parasitics[:, 0] - target_parasitics[:, 0]))
                mae_l = torch.mean(torch.abs(predicted_parasitics[:, 1] - target_parasitics[:, 1]))
                
                total_mse_c += mse_c.item()
                total_mse_l += mse_l.item()
                total_mae_c += mae_c.item()
                total_mae_l += mae_l.item()
        
        avg_loss = total_loss / max(total_samples, 1)
        avg_mse_c = total_mse_c / max(total_samples, 1)
        avg_mse_l = total_mse_l / max(total_samples, 1)
        avg_mae_c = total_mae_c / max(total_samples, 1)
        avg_mae_l = total_mae_l / max(total_samples, 1)
        
        metrics = {
            'mse_capacitance': avg_mse_c,
            'mse_inductance': avg_mse_l,
            'mae_capacitance': avg_mae_c,
            'mae_inductance': avg_mae_l,
        }
        
        return avg_loss, metrics
    
    def _get_target_parasitics(
        self,
        sample: Any,
        circuit: Any
    ) -> Optional[torch.Tensor]:
        """
        Extract target parasitics from sample.
        
        Args:
            sample: Dataset sample
            circuit: Circuit graph
            
        Returns:
            Tensor of shape [num_edges, 2] with capacitance and inductance values
        """
        target_values = []
        
        for edge in circuit.edges:
            if edge.net_name in sample.measured_parasitics:
                parasitics = sample.measured_parasitics[edge.net_name]
                c = parasitics.get('C', 0.0)
                l = parasitics.get('L', 0.0)
                target_values.append([c, l])
            else:
                # Skip edges without measured parasitics
                return None
        
        if not target_values:
            return None
        
        return torch.tensor(target_values, dtype=torch.float32)
    
    def _log_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
    ):
        """Log metrics to TensorBoard and console."""
        # TensorBoard logging
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('LearningRate', learning_rate, epoch)
        
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        # Console logging
        logger.info(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}, "
            f"lr={learning_rate:.2e}"
        )
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def _load_best_checkpoint(self):
        """Load best model checkpoint."""
        best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
        if not best_path.exists():
            logger.warning("Best model checkpoint not found")
            return
        
        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    def _register_model(self, test_metrics: Dict[str, float]) -> str:
        """Register trained model in model registry."""
        metadata = {
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_version': 'CircuitNet 2.0',
            'performance_metrics': {
                'test_mse_capacitance': test_metrics['mse_capacitance'],
                'test_mse_inductance': test_metrics['mse_inductance'],
                'test_mae_capacitance': test_metrics['mae_capacitance'],
                'test_mae_inductance': test_metrics['mae_inductance'],
                'best_val_loss': self.best_val_loss,
            },
            'hyperparameters': {
                'node_feature_dim': self.config.node_feature_dim,
                'edge_feature_dim': self.config.edge_feature_dim,
                'hidden_dim': self.config.hidden_dim,
                'num_layers': self.config.num_layers,
                'dropout': self.config.dropout,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
            },
            'description': f'FALCON GNN trained for {self.current_epoch + 1} epochs',
        }
        
        model_path = self.model_registry.register_model(
            model_type='falcon_gnn',
            model=self.model,
            version=self.config.model_version,
            metadata=metadata,
        )
        
        logger.info(f"Registered model at {model_path}")
        return model_path


def train_falcon_gnn(config: TrainingConfig) -> TrainingResult:
    """
    Train FALCON GNN model.
    
    Args:
        config: Training configuration
        
    Returns:
        TrainingResult with training metrics
    """
    trainer = FalconGNNTrainer(config)
    result = trainer.train()
    return result
