"""
CircuitNet dataset loader for training FALCON GNN.

This module loads and parses PCB layouts from the CircuitNet 2.0 dataset,
extracting component positions, connections, and measured parasitics.
"""

import os
import json
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from src.models.circuit_graph import CircuitNode, CircuitEdge, CircuitGraph

logger = logging.getLogger(__name__)


@dataclass
class CircuitNetSample:
    """Single sample from CircuitNet dataset."""
    layout_id: str
    circuit_graph: CircuitGraph
    measured_parasitics: Dict[str, Dict[str, float]]  # net_name -> {C, L, R}
    metadata: Dict[str, Any]


class CircuitNetDataset:
    """
    CircuitNet 2.0 dataset loader.
    
    Loads PCB layouts from CircuitNet dataset and converts them to
    circuit graph representations for training.
    """
    
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
    ):
        """
        Initialize CircuitNet dataset loader.
        
        Args:
            dataset_path: Path to CircuitNet dataset directory
            split: Dataset split ("train", "val", or "test")
            train_ratio: Ratio of training data (default: 0.8)
            val_ratio: Ratio of validation data (default: 0.1)
            test_ratio: Ratio of test data (default: 0.1)
            random_seed: Random seed for reproducible splits
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Validate ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        # Load dataset
        self.samples: List[CircuitNetSample] = []
        self._load_dataset()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_dataset(self):
        """Load and split dataset."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        # Find all layout files
        layout_files = list(self.dataset_path.glob("**/*.json"))
        
        if not layout_files:
            logger.warning(f"No layout files found in {self.dataset_path}")
            return
        
        # Sort for reproducibility
        layout_files = sorted(layout_files)
        
        # Split dataset
        np.random.seed(self.random_seed)
        indices = np.random.permutation(len(layout_files))
        
        train_end = int(len(layout_files) * self.train_ratio)
        val_end = train_end + int(len(layout_files) * self.val_ratio)
        
        if self.split == "train":
            selected_indices = indices[:train_end]
        elif self.split == "val":
            selected_indices = indices[train_end:val_end]
        elif self.split == "test":
            selected_indices = indices[val_end:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        # Load selected files
        for idx in selected_indices:
            layout_file = layout_files[idx]
            try:
                sample = self._parse_layout_file(layout_file)
                if sample:
                    self.samples.append(sample)
            except Exception as e:
                logger.error(f"Error parsing {layout_file}: {e}")
    
    def _parse_layout_file(self, layout_file: Path) -> Optional[CircuitNetSample]:
        """
        Parse a CircuitNet layout file.
        
        Args:
            layout_file: Path to layout JSON file
            
        Returns:
            CircuitNetSample or None if parsing fails
        """
        try:
            with open(layout_file, 'r') as f:
                data = json.load(f)
            
            layout_id = data.get('layout_id', layout_file.stem)
            
            # Parse components (nodes)
            nodes = []
            for comp_data in data.get('components', []):
                node = CircuitNode(
                    id=comp_data['id'],
                    component_type=comp_data.get('type', 'unknown'),
                    value=comp_data.get('value'),
                    package=comp_data.get('package', ''),
                    dimensions=tuple(comp_data.get('dimensions', [0.0, 0.0, 0.0])),
                    pin_count=comp_data.get('pin_count', 2),
                    position=tuple(comp_data.get('position', [0.0, 0.0])),
                    orientation=comp_data.get('orientation', 0.0),
                    layer=comp_data.get('layer', 0),
                    electrical_params=comp_data.get('electrical_params', {}),
                )
                nodes.append(node)
            
            # Parse connections (edges)
            edges = []
            for conn_data in data.get('connections', []):
                edge = CircuitEdge(
                    id=conn_data['id'],
                    source_node=conn_data['source'],
                    target_node=conn_data['target'],
                    net_name=conn_data.get('net_name', ''),
                    signal_type=conn_data.get('signal_type', 'signal'),
                    routing_constraints=conn_data.get('routing_constraints', {}),
                    current_length=conn_data.get('current_length', 0.0),
                    layer_transitions=conn_data.get('layer_transitions', 0),
                    measured_parasitics=conn_data.get('measured_parasitics'),
                )
                edges.append(edge)
            
            # Create circuit graph
            circuit_graph = CircuitGraph(
                nodes=nodes,
                edges=edges,
                board_size=tuple(data.get('board_size', [100.0, 100.0])),
                num_layers=data.get('num_layers', 2),
                design_rules=data.get('design_rules', {}),
            )
            
            # Validate graph
            errors = circuit_graph.validate()
            if errors:
                logger.warning(f"Validation errors in {layout_file}: {errors}")
                return None
            
            # Extract measured parasitics
            measured_parasitics = {}
            for edge in edges:
                if edge.measured_parasitics:
                    measured_parasitics[edge.net_name] = edge.measured_parasitics
            
            # Create sample
            sample = CircuitNetSample(
                layout_id=layout_id,
                circuit_graph=circuit_graph,
                measured_parasitics=measured_parasitics,
                metadata=data.get('metadata', {}),
            )
            
            return sample
            
        except Exception as e:
            logger.error(f"Error parsing layout file {layout_file}: {e}")
            return None
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> CircuitNetSample:
        """Get sample by index."""
        return self.samples[idx]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.samples:
            return {}
        
        num_components = [len(s.circuit_graph.nodes) for s in self.samples]
        num_connections = [len(s.circuit_graph.edges) for s in self.samples]
        board_areas = [s.circuit_graph.board_size[0] * s.circuit_graph.board_size[1] 
                      for s in self.samples]
        
        stats = {
            'num_samples': len(self.samples),
            'components': {
                'mean': np.mean(num_components),
                'std': np.std(num_components),
                'min': np.min(num_components),
                'max': np.max(num_components),
            },
            'connections': {
                'mean': np.mean(num_connections),
                'std': np.std(num_connections),
                'min': np.min(num_connections),
                'max': np.max(num_connections),
            },
            'board_area': {
                'mean': np.mean(board_areas),
                'std': np.std(board_areas),
                'min': np.min(board_areas),
                'max': np.max(board_areas),
            },
        }
        
        return stats


def create_dataloaders(
    dataset_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    random_seed: int = 42,
) -> Tuple[CircuitNetDataset, CircuitNetDataset, CircuitNetDataset]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        dataset_path: Path to CircuitNet dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = CircuitNetDataset(
        dataset_path=dataset_path,
        split="train",
        random_seed=random_seed,
    )
    
    val_dataset = CircuitNetDataset(
        dataset_path=dataset_path,
        split="val",
        random_seed=random_seed,
    )
    
    test_dataset = CircuitNetDataset(
        dataset_path=dataset_path,
        split="test",
        random_seed=random_seed,
    )
    
    logger.info(f"Created dataloaders: train={len(train_dataset)}, "
                f"val={len(val_dataset)}, test={len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset
