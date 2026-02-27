"""
Differentiable cost model for placement optimization.

Computes placement quality based on predicted parasitics, trace length,
congestion, and constraint violations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging

from src.models.circuit_graph import CircuitGraph
from src.models.pcb_state import BoardConstraints

logger = logging.getLogger(__name__)


class PlacementCostModel:
    """
    Differentiable cost model for placement optimization.
    
    Cost = w1 * trace_length_cost 
         + w2 * parasitic_cost
         + w3 * congestion_cost
         + w4 * constraint_violation_cost
    
    Weights: w1=1.0, w2=2.0, w3=1.5, w4=10.0
    """
    
    def __init__(
        self,
        trace_length_weight: float = 1.0,
        parasitic_weight: float = 2.0,
        congestion_weight: float = 1.5,
        constraint_weight: float = 10.0,
    ):
        """
        Initialize cost model.
        
        Args:
            trace_length_weight: Weight for trace length cost
            parasitic_weight: Weight for parasitic cost
            congestion_weight: Weight for congestion cost
            constraint_weight: Weight for constraint violation cost
        """
        self.w1 = trace_length_weight
        self.w2 = parasitic_weight
        self.w3 = congestion_weight
        self.w4 = constraint_weight
        
        logger.info(f"Initialized cost model with weights: "
                   f"trace={self.w1}, parasitic={self.w2}, "
                   f"congestion={self.w3}, constraint={self.w4}")
    
    def compute_cost(
        self,
        positions: torch.Tensor,
        circuit_graph: CircuitGraph,
        gnn_predictions: Optional[Dict[str, Dict[str, float]]] = None,
        constraints: Optional[BoardConstraints] = None,
    ) -> torch.Tensor:
        """
        Compute total placement cost.
        
        Args:
            positions: Component positions [num_components, 2]
            circuit_graph: Circuit graph with connectivity
            gnn_predictions: Predicted parasitics from GNN (optional)
            constraints: Board constraints (optional)
            
        Returns:
            Total cost (scalar tensor with gradient)
        """
        # Compute individual cost components
        trace_cost = self.compute_trace_length_cost(positions, circuit_graph)
        parasitic_cost = self.compute_parasitic_cost(positions, circuit_graph, gnn_predictions)
        congestion_cost = self.compute_congestion_cost(positions, circuit_graph)
        constraint_cost = self.compute_constraint_violation_cost(positions, circuit_graph, constraints)
        
        # Weighted sum
        total_cost = (
            self.w1 * trace_cost +
            self.w2 * parasitic_cost +
            self.w3 * congestion_cost +
            self.w4 * constraint_cost
        )
        
        return total_cost
    
    def compute_trace_length_cost(
        self,
        positions: torch.Tensor,
        circuit_graph: CircuitGraph,
    ) -> torch.Tensor:
        """
        Compute trace length cost using half-perimeter wirelength (HPWL).
        
        Args:
            positions: Component positions [num_components, 2]
            circuit_graph: Circuit graph
            
        Returns:
            Trace length cost (scalar tensor)
        """
        # Create node ID to index mapping
        node_id_to_idx = {node.id: i for i, node in enumerate(circuit_graph.nodes)}
        
        total_length = torch.tensor(0.0, requires_grad=True)
        
        # Compute HPWL for each net
        for edge in circuit_graph.edges:
            source_idx = node_id_to_idx[edge.source_node]
            target_idx = node_id_to_idx[edge.target_node]
            
            source_pos = positions[source_idx]
            target_pos = positions[target_idx]
            
            # Manhattan distance (L1 norm)
            distance = torch.abs(source_pos[0] - target_pos[0]) + torch.abs(source_pos[1] - target_pos[1])
            
            total_length = total_length + distance
        
        return total_length
    
    def compute_parasitic_cost(
        self,
        positions: torch.Tensor,
        circuit_graph: CircuitGraph,
        gnn_predictions: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> torch.Tensor:
        """
        Compute parasitic cost based on GNN predictions.
        
        Parasitics degrade signal integrity, so we penalize high parasitic values.
        
        Args:
            positions: Component positions [num_components, 2]
            circuit_graph: Circuit graph
            gnn_predictions: Predicted parasitics {net_name: {C, L}}
            
        Returns:
            Parasitic cost (scalar tensor)
        """
        if gnn_predictions is None:
            # If no predictions, use distance-based approximation
            return self._approximate_parasitic_cost(positions, circuit_graph)
        
        total_parasitic_cost = torch.tensor(0.0, requires_grad=True)
        
        # Sum weighted parasitics for all nets
        for net_name, parasitics in gnn_predictions.items():
            capacitance = parasitics.get('C', 0.0)  # pF
            inductance = parasitics.get('L', 0.0)  # nH
            
            # Weight parasitics by their impact on signal integrity
            # Higher capacitance and inductance are worse
            parasitic_penalty = torch.tensor(capacitance * 0.1 + inductance * 0.05)
            
            total_parasitic_cost = total_parasitic_cost + parasitic_penalty
        
        return total_parasitic_cost
    
    def _approximate_parasitic_cost(
        self,
        positions: torch.Tensor,
        circuit_graph: CircuitGraph,
    ) -> torch.Tensor:
        """
        Approximate parasitic cost based on trace length.
        
        Longer traces have higher parasitics.
        """
        # Use trace length as proxy for parasitics
        trace_length = self.compute_trace_length_cost(positions, circuit_graph)
        return trace_length * 0.01  # Scale factor
    
    def compute_congestion_cost(
        self,
        positions: torch.Tensor,
        circuit_graph: CircuitGraph,
        grid_size: int = 20,
    ) -> torch.Tensor:
        """
        Compute congestion cost based on component density.
        
        Penalizes overlapping routing regions to avoid routing congestion.
        
        Args:
            positions: Component positions [num_components, 2]
            circuit_graph: Circuit graph
            grid_size: Grid resolution for congestion estimation
            
        Returns:
            Congestion cost (scalar tensor)
        """
        board_width, board_height = circuit_graph.board_size
        
        # Create congestion grid
        grid = torch.zeros((grid_size, grid_size))
        
        # Map components to grid cells
        for i, node in enumerate(circuit_graph.nodes):
            pos = positions[i]
            
            # Convert position to grid coordinates
            grid_x = int((pos[0] / board_width) * (grid_size - 1))
            grid_y = int((pos[1] / board_height) * (grid_size - 1))
            
            # Clamp to grid boundaries
            grid_x = max(0, min(grid_size - 1, grid_x))
            grid_y = max(0, min(grid_size - 1, grid_y))
            
            # Increment congestion
            grid[grid_y, grid_x] += 1.0
        
        # Compute congestion cost (penalize high-density regions)
        # Use squared congestion to heavily penalize overlaps
        congestion_cost = torch.sum(grid ** 2)
        
        return congestion_cost
    
    def compute_constraint_violation_cost(
        self,
        positions: torch.Tensor,
        circuit_graph: CircuitGraph,
        constraints: Optional[BoardConstraints] = None,
    ) -> torch.Tensor:
        """
        Compute constraint violation cost.
        
        Penalizes violations of:
        - Board boundaries
        - Keepout zones
        - Component clearances
        
        Args:
            positions: Component positions [num_components, 2]
            circuit_graph: Circuit graph
            constraints: Board constraints
            
        Returns:
            Constraint violation cost (scalar tensor)
        """
        if constraints is None:
            constraints = BoardConstraints(
                width=circuit_graph.board_size[0],
                height=circuit_graph.board_size[1],
            )
        
        total_violation = torch.tensor(0.0, requires_grad=True)
        
        # Check board boundary violations
        for i, node in enumerate(circuit_graph.nodes):
            pos = positions[i]
            width, height, _ = node.dimensions
            
            # Check if component extends beyond board
            if pos[0] - width/2 < 0:
                total_violation = total_violation + torch.abs(pos[0] - width/2)
            if pos[0] + width/2 > constraints.width:
                total_violation = total_violation + (pos[0] + width/2 - constraints.width)
            if pos[1] - height/2 < 0:
                total_violation = total_violation + torch.abs(pos[1] - height/2)
            if pos[1] + height/2 > constraints.height:
                total_violation = total_violation + (pos[1] + height/2 - constraints.height)
        
        # Check keepout zone violations
        for zone in constraints.keepout_zones:
            x1, y1, x2, y2 = zone
            for i, node in enumerate(circuit_graph.nodes):
                pos = positions[i]
                
                # Check if component overlaps with keepout zone
                if x1 <= pos[0] <= x2 and y1 <= pos[1] <= y2:
                    # Penalize based on distance to zone boundary
                    dist_to_boundary = min(
                        pos[0] - x1,
                        x2 - pos[0],
                        pos[1] - y1,
                        y2 - pos[1],
                    )
                    total_violation = total_violation + torch.tensor(10.0) - dist_to_boundary
        
        # Check component clearance violations
        clearance = constraints.component_clearance
        for i, node_i in enumerate(circuit_graph.nodes):
            pos_i = positions[i]
            width_i, height_i, _ = node_i.dimensions
            
            for j, node_j in enumerate(circuit_graph.nodes):
                if i >= j:
                    continue
                
                pos_j = positions[j]
                width_j, height_j, _ = node_j.dimensions
                
                # Compute distance between components
                distance = torch.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                
                # Required clearance
                required_clearance = clearance + (width_i + width_j) / 2
                
                # Penalize if too close
                if distance < required_clearance:
                    violation = required_clearance - distance
                    total_violation = total_violation + violation
        
        return total_violation


def compute_placement_cost(
    positions: torch.Tensor,
    circuit_graph: CircuitGraph,
    gnn_predictions: Optional[Dict[str, Dict[str, float]]] = None,
    constraints: Optional[BoardConstraints] = None,
    weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Convenience function to compute placement cost.
    
    Args:
        positions: Component positions [num_components, 2]
        circuit_graph: Circuit graph
        gnn_predictions: Predicted parasitics from GNN
        constraints: Board constraints
        weights: Custom cost weights
        
    Returns:
        Total cost (scalar tensor with gradient)
    """
    if weights is None:
        weights = {}
    
    cost_model = PlacementCostModel(
        trace_length_weight=weights.get('trace_length', 1.0),
        parasitic_weight=weights.get('parasitic', 2.0),
        congestion_weight=weights.get('congestion', 1.5),
        constraint_weight=weights.get('constraint', 10.0),
    )
    
    return cost_model.compute_cost(positions, circuit_graph, gnn_predictions, constraints)
