"""
FALCON GNN - Graph Neural Network for Component Placement.

Implements a graph neural network that learns circuit topology and parasitics
to optimize component placement on PCB layouts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional
import logging

from src.models.circuit_graph import CircuitGraph, CircuitNode, CircuitEdge

logger = logging.getLogger(__name__)


class FalconGNN(nn.Module):
    """
    Graph Neural Network for parasitic prediction and placement optimization.
    
    Architecture:
    - Input: Circuit graph with component and connection features
    - 4 Graph Convolutional layers with residual connections
    - Hidden dimension: 256
    - Output: Parasitic predictions (capacitance, inductance) per edge
    """
    
    def __init__(
        self,
        node_feature_dim: int = 64,
        edge_feature_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize FALCON GNN model.
        
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            dropout: Dropout probability
        """
        super(FalconGNN, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Node encoder: component properties → hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Edge encoder: connection properties → hidden_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # GNN layers: 4x GraphConv with ReLU and residual connections
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Layer normalization for each GNN layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Parasitic predictor: hidden_dim → 2 (capacitance, inductance)
        self.parasitic_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concatenate source and target node features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # Output: [capacitance, inductance]
        )
        
        logger.info(f"Initialized FALCON GNN with {num_layers} layers, hidden_dim={hidden_dim}")
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GNN.
        
        Args:
            node_features: Node feature matrix [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_features: Edge feature matrix [num_edges, edge_feature_dim]
            batch: Batch assignment vector [num_nodes] (optional)
            
        Returns:
            Tuple of (node_embeddings, parasitic_predictions)
            - node_embeddings: [num_nodes, hidden_dim]
            - parasitic_predictions: [num_edges, 2] (capacitance, inductance)
        """
        # Encode node features
        x = self.node_encoder(node_features)  # [num_nodes, hidden_dim]
        
        # Encode edge features (not used in message passing, but stored for later)
        edge_emb = self.edge_encoder(edge_features)  # [num_edges, hidden_dim]
        
        # Message passing through GNN layers with residual connections
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            # Store input for residual connection
            x_residual = x
            
            # GNN layer
            x = gnn_layer(x, edge_index)
            
            # Layer normalization
            x = layer_norm(x)
            
            # ReLU activation
            x = F.relu(x)
            
            # Residual connection
            x = x + x_residual
            
            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Predict parasitics for each edge
        # Concatenate source and target node embeddings
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        source_emb = x[source_nodes]  # [num_edges, hidden_dim]
        target_emb = x[target_nodes]  # [num_edges, hidden_dim]
        
        edge_input = torch.cat([source_emb, target_emb], dim=1)  # [num_edges, hidden_dim * 2]
        
        # Predict parasitics
        parasitics = self.parasitic_predictor(edge_input)  # [num_edges, 2]
        
        # Apply activation to ensure positive values
        parasitics = F.softplus(parasitics)  # Ensures positive capacitance and inductance
        
        return x, parasitics
    
    def predict_parasitics(
        self,
        circuit_graph: CircuitGraph,
        device: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Predict parasitics for a circuit graph.
        
        Args:
            circuit_graph: Input circuit graph
            device: Device to run inference on ('cpu' or 'cuda'). If None, uses model's current device.
            
        Returns:
            Dictionary mapping net_name to parasitic values {C, L}
        """
        self.eval()
        
        # Determine device
        if device is None:
            # Use the device the model is currently on
            device = next(self.parameters()).device
        else:
            device = torch.device(device)
        
        # Convert circuit graph to PyTorch Geometric format
        data = self._circuit_to_pyg_data(circuit_graph)
        data = data.to(device)
        
        with torch.no_grad():
            _, parasitics = self.forward(
                data.x,
                data.edge_index,
                data.edge_attr,
            )
        
        # Convert predictions to dictionary
        predictions = {}
        for i, edge in enumerate(circuit_graph.edges):
            capacitance = parasitics[i, 0].item()  # pF
            inductance = parasitics[i, 1].item()  # nH
            
            predictions[edge.net_name] = {
                'C': capacitance,
                'L': inductance,
            }
        
        return predictions
    
    def _circuit_to_pyg_data(self, circuit: CircuitGraph) -> Data:
        """
        Convert CircuitGraph to PyTorch Geometric Data object.
        
        Args:
            circuit: Circuit graph
            
        Returns:
            PyTorch Geometric Data object
        """
        # Create node feature matrix
        node_features = []
        node_id_to_idx = {}
        
        for i, node in enumerate(circuit.nodes):
            node_id_to_idx[node.id] = i
            features = self._encode_node_features(node)
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float32)
        
        # Create edge index and edge features
        edge_indices = []
        edge_features = []
        
        for edge in circuit.edges:
            source_idx = node_id_to_idx[edge.source_node]
            target_idx = node_id_to_idx[edge.target_node]
            
            edge_indices.append([source_idx, target_idx])
            features = self._encode_edge_features(edge)
            edge_features.append(features)
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def _encode_node_features(self, node: CircuitNode) -> List[float]:
        """
        Encode node (component) features.
        
        Returns 64-dimensional feature vector:
        - Component type (one-hot, 20 categories)
        - Physical dimensions (3)
        - Pin count (1)
        - Position (2)
        - Orientation (1)
        - Layer (1)
        - Electrical parameters (up to 36)
        """
        features = []
        
        # Component type (one-hot encoding, 20 categories)
        component_types = [
            'resistor', 'capacitor', 'inductor', 'diode', 'transistor',
            'ic', 'connector', 'switch', 'led', 'crystal',
            'fuse', 'relay', 'transformer', 'battery', 'motor',
            'sensor', 'display', 'microcontroller', 'memory', 'unknown'
        ]
        type_idx = component_types.index(node.component_type) if node.component_type in component_types else 19
        type_onehot = [1.0 if i == type_idx else 0.0 for i in range(20)]
        features.extend(type_onehot)
        
        # Physical dimensions (normalized)
        width, height, thickness = node.dimensions
        features.extend([width / 100.0, height / 100.0, thickness / 10.0])
        
        # Pin count (normalized)
        features.append(node.pin_count / 100.0)
        
        # Position (normalized by board size)
        x, y = node.position
        features.extend([x / 200.0, y / 200.0])
        
        # Orientation (normalized to [0, 1])
        features.append(node.orientation / 360.0)
        
        # Layer (normalized)
        features.append(node.layer / 10.0)
        
        # Electrical parameters (pad to 36 dimensions)
        param_values = list(node.electrical_params.values())[:36]
        param_values += [0.0] * (36 - len(param_values))
        features.extend(param_values)
        
        return features[:64]  # Ensure exactly 64 dimensions
    
    def _encode_edge_features(self, edge: CircuitEdge) -> List[float]:
        """
        Encode edge (connection) features.
        
        Returns 32-dimensional feature vector:
        - Signal type (one-hot, 4 categories)
        - Current length (1)
        - Layer transitions (1)
        - Routing constraints (up to 26)
        """
        features = []
        
        # Signal type (one-hot encoding)
        signal_types = ['power', 'ground', 'signal', 'differential']
        type_idx = signal_types.index(edge.signal_type) if edge.signal_type in signal_types else 2
        type_onehot = [1.0 if i == type_idx else 0.0 for i in range(4)]
        features.extend(type_onehot)
        
        # Current length (normalized)
        features.append(edge.current_length / 100.0)
        
        # Layer transitions (normalized)
        features.append(edge.layer_transitions / 10.0)
        
        # Routing constraints (pad to 26 dimensions)
        constraint_values = list(edge.routing_constraints.values())[:26]
        constraint_values += [0.0] * (26 - len(constraint_values))
        features.extend(constraint_values)
        
        return features[:32]  # Ensure exactly 32 dimensions


def create_falcon_gnn(
    node_feature_dim: int = 64,
    edge_feature_dim: int = 32,
    hidden_dim: int = 256,
    num_layers: int = 4,
    dropout: float = 0.1,
) -> FalconGNN:
    """
    Factory function to create FALCON GNN model.
    
    Args:
        node_feature_dim: Dimension of node features
        edge_feature_dim: Dimension of edge features
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers
        dropout: Dropout probability
        
    Returns:
        Initialized FALCON GNN model
    """
    model = FalconGNN(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    
    return model
