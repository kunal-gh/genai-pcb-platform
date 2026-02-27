"""
Placement optimizer using FALCON GNN and gradient-based optimization.

Optimizes component placement to minimize trace length, parasitics,
congestion, and constraint violations.
"""

import torch
import torch.optim as optim
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from src.models.circuit_graph import CircuitGraph
from src.models.pcb_state import PlacementResult, BoardConstraints, OptimizationConfig
from src.services.falcon_gnn import FalconGNN
from src.services.placement_cost import PlacementCostModel

logger = logging.getLogger(__name__)


@dataclass
class OptimizationState:
    """State during optimization."""
    iteration: int
    cost: float
    cost_history: List[float]
    best_cost: float
    best_positions: torch.Tensor
    converged: bool
    no_improvement_count: int


class PlacementOptimizer:
    """
    Gradient-based placement optimization using FALCON GNN.
    
    Algorithm:
    1. Initialize positions using force-directed layout
    2. Convert netlist to graph representation
    3. Iteratively:
       a. Forward pass through GNN to predict parasitics
       b. Compute cost function
       c. Backpropagate gradients
       d. Update positions using Adam optimizer
       e. Project positions to satisfy hard constraints
    4. Return optimized placement
    """
    
    def __init__(
        self,
        gnn_model: Optional[FalconGNN] = None,
        cost_model: Optional[PlacementCostModel] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize placement optimizer.
        
        Args:
            gnn_model: Trained FALCON GNN model (optional)
            cost_model: Cost model for placement evaluation (optional)
            device: Device to run optimization on ('cpu' or 'cuda'). If None, auto-detects.
        """
        self.gnn_model = gnn_model
        self.cost_model = cost_model or PlacementCostModel()
        
        # Setup device with CUDA detection
        self.device = self._setup_device(device)
        self._log_device_info()
        
        if self.gnn_model is not None:
            self.gnn_model.to(self.device)
            self.gnn_model.eval()
    
    def _setup_device(self, device_str: Optional[str]) -> torch.device:
        """
        Set up compute device with CUDA detection.
        
        Args:
            device_str: Device string ('cuda', 'cpu', or None for auto-detect)
            
        Returns:
            torch.device object
        """
        if device_str is None:
            # Auto-detect: use CUDA if available, otherwise CPU
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info("Auto-detected CUDA. Using GPU for placement optimization.")
            else:
                device = torch.device('cpu')
                logger.info("CUDA not available. Using CPU for placement optimization.")
            return device
        
        # Check if CUDA is requested
        if 'cuda' in device_str.lower():
            if torch.cuda.is_available():
                # CUDA is available, use GPU
                device = torch.device(device_str)
                return device
            else:
                # CUDA requested but not available, fall back to CPU
                logger.info("CUDA requested but not available. Falling back to CPU for placement optimization.")
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
            logger.info(f"Placement Optimizer using GPU: {gpu_name}")
            logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        else:
            # CPU information
            logger.info("Placement Optimizer using CPU")
    
    def optimize_placement(
        self,
        circuit_graph: CircuitGraph,
        board_constraints: Optional[BoardConstraints] = None,
        config: Optional[OptimizationConfig] = None,
    ) -> PlacementResult:
        """
        Optimize component placement.
        
        Args:
            circuit_graph: Circuit graph with components and connections
            board_constraints: Board constraints (optional)
            config: Optimization configuration (optional)
            
        Returns:
            PlacementResult with optimized positions and metrics
        """
        start_time = time.time()
        
        # Use default config if not provided
        if config is None:
            config = OptimizationConfig()
        
        # Use default constraints if not provided
        if board_constraints is None:
            board_constraints = BoardConstraints(
                width=circuit_graph.board_size[0],
                height=circuit_graph.board_size[1],
                num_layers=circuit_graph.num_layers,
            )
        
        # Validate configuration
        config_errors = config.validate()
        if config_errors:
            return PlacementResult(
                success=False,
                error_message=f"Invalid configuration: {config_errors}",
            )
        
        # Validate circuit graph
        graph_errors = circuit_graph.validate()
        if graph_errors:
            return PlacementResult(
                success=False,
                error_message=f"Invalid circuit graph: {graph_errors}",
            )
        
        try:
            # Initialize positions using force-directed layout
            logger.info("Initializing positions with force-directed layout")
            positions = self._initialize_positions(circuit_graph, board_constraints)
            positions = positions.to(self.device)
            positions.requires_grad = True
            
            # Set up optimizer
            optimizer = optim.Adam([positions], lr=config.learning_rate)
            
            # Initialize optimization state
            state = OptimizationState(
                iteration=0,
                cost=float('inf'),
                cost_history=[],
                best_cost=float('inf'),
                best_positions=positions.clone().detach(),
                converged=False,
                no_improvement_count=0,
            )
            
            # Optimization loop
            logger.info(f"Starting optimization (max_iterations={config.max_iterations})")
            
            for iteration in range(config.max_iterations):
                state.iteration = iteration
                
                # Check timeout
                if config.timeout_seconds > 0:
                    elapsed = time.time() - start_time
                    if elapsed > config.timeout_seconds:
                        logger.warning(f"Optimization timeout after {elapsed:.1f}s")
                        break
                
                # Forward pass: predict parasitics using GNN
                gnn_predictions = None
                if self.gnn_model is not None:
                    gnn_predictions = self._predict_parasitics(circuit_graph, positions)
                
                # Compute cost
                cost = self.cost_model.compute_cost(
                    positions,
                    circuit_graph,
                    gnn_predictions,
                    board_constraints,
                )
                
                state.cost = cost.item()
                state.cost_history.append(state.cost)
                
                # Track best solution
                if state.cost < state.best_cost:
                    state.best_cost = state.cost
                    state.best_positions = positions.clone().detach()
                    state.no_improvement_count = 0
                else:
                    state.no_improvement_count += 1
                
                # Check convergence
                if self._check_convergence(state, config):
                    state.converged = True
                    logger.info(f"Converged at iteration {iteration}")
                    break
                
                # Backward pass
                optimizer.zero_grad()
                cost.backward()
                
                # Gradient clipping to prevent instability
                torch.nn.utils.clip_grad_norm_([positions], max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                
                # Project positions to satisfy hard constraints
                with torch.no_grad():
                    positions.data = self._project_to_constraints(
                        positions.data,
                        circuit_graph,
                        board_constraints,
                    )
                
                # Log progress
                if iteration % 100 == 0:
                    logger.info(f"Iteration {iteration}: cost={state.cost:.4f}, "
                              f"best={state.best_cost:.4f}")
            
            # Use best positions found
            final_positions = state.best_positions.cpu()
            
            # Compute final quality metrics
            quality_metrics = self._compute_quality_metrics(
                final_positions,
                circuit_graph,
                board_constraints,
            )
            
            # Create result
            positions_dict = {}
            orientations_dict = {}
            for i, node in enumerate(circuit_graph.nodes):
                pos = final_positions[i].tolist()
                positions_dict[node.id] = tuple(pos)
                orientations_dict[node.id] = node.orientation
            
            optimization_time = time.time() - start_time
            
            result = PlacementResult(
                success=True,
                positions=positions_dict,
                orientations=orientations_dict,
                quality_metrics=quality_metrics,
                optimization_time=optimization_time,
                iterations=state.iteration + 1,
                converged=state.converged,
            )
            
            logger.info(f"Optimization complete: {state.iteration + 1} iterations, "
                       f"{optimization_time:.2f}s, cost={state.best_cost:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            return PlacementResult(
                success=False,
                error_message=str(e),
                optimization_time=time.time() - start_time,
            )
    
    def _initialize_positions(
        self,
        circuit_graph: CircuitGraph,
        constraints: BoardConstraints,
    ) -> torch.Tensor:
        """
        Initialize component positions using force-directed layout.
        
        Args:
            circuit_graph: Circuit graph
            constraints: Board constraints
            
        Returns:
            Initial positions [num_components, 2]
        """
        num_components = len(circuit_graph.nodes)
        
        # Start with random positions
        positions = torch.rand(num_components, 2)
        positions[:, 0] *= constraints.width
        positions[:, 1] *= constraints.height
        
        # Apply force-directed layout for a few iterations
        for _ in range(50):
            forces = torch.zeros_like(positions)
            
            # Attractive forces between connected components
            node_id_to_idx = {node.id: i for i, node in enumerate(circuit_graph.nodes)}
            
            for edge in circuit_graph.edges:
                i = node_id_to_idx[edge.source_node]
                j = node_id_to_idx[edge.target_node]
                
                diff = positions[j] - positions[i]
                distance = torch.norm(diff) + 1e-6
                
                # Spring force (attractive)
                force = diff * 0.01
                forces[i] += force
                forces[j] -= force
            
            # Repulsive forces between all components
            for i in range(num_components):
                for j in range(i + 1, num_components):
                    diff = positions[j] - positions[i]
                    distance = torch.norm(diff) + 1e-6
                    
                    # Coulomb force (repulsive)
                    force = -diff / (distance ** 2) * 10.0
                    forces[i] += force
                    forces[j] -= force
            
            # Update positions
            positions += forces * 0.1
            
            # Clamp to board boundaries
            positions[:, 0] = torch.clamp(positions[:, 0], 0, constraints.width)
            positions[:, 1] = torch.clamp(positions[:, 1], 0, constraints.height)
        
        return positions
    
    def _predict_parasitics(
        self,
        circuit_graph: CircuitGraph,
        positions: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """
        Predict parasitics using GNN model.
        
        Args:
            circuit_graph: Circuit graph
            positions: Current positions
            
        Returns:
            Predicted parasitics {net_name: {C, L}}
        """
        if self.gnn_model is None:
            return {}
        
        # Update circuit graph with current positions
        updated_graph = self._update_graph_positions(circuit_graph, positions)
        
        # Predict parasitics (model will use its current device)
        with torch.no_grad():
            predictions = self.gnn_model.predict_parasitics(updated_graph, device=None)
        
        return predictions
    
    def _update_graph_positions(
        self,
        circuit_graph: CircuitGraph,
        positions: torch.Tensor,
    ) -> CircuitGraph:
        """Update circuit graph with new positions."""
        import copy
        updated_graph = copy.deepcopy(circuit_graph)
        
        for i, node in enumerate(updated_graph.nodes):
            pos = positions[i].detach().cpu().tolist()
            node.position = tuple(pos)
        
        return updated_graph
    
    def _project_to_constraints(
        self,
        positions: torch.Tensor,
        circuit_graph: CircuitGraph,
        constraints: BoardConstraints,
    ) -> torch.Tensor:
        """
        Project positions to satisfy hard constraints.
        
        Args:
            positions: Current positions
            circuit_graph: Circuit graph
            constraints: Board constraints
            
        Returns:
            Projected positions
        """
        # Clamp to board boundaries
        positions[:, 0] = torch.clamp(positions[:, 0], 0, constraints.width)
        positions[:, 1] = torch.clamp(positions[:, 1], 0, constraints.height)
        
        # TODO: Handle keepout zones and component clearances
        # For now, just enforce board boundaries
        
        return positions
    
    def _check_convergence(
        self,
        state: OptimizationState,
        config: OptimizationConfig,
    ) -> bool:
        """
        Check if optimization has converged.
        
        Args:
            state: Current optimization state
            config: Optimization configuration
            
        Returns:
            True if converged
        """
        # Need at least 10 iterations
        if state.iteration < 10:
            return False
        
        # Check if cost change is below threshold for last 10 iterations
        if len(state.cost_history) >= 10:
            recent_costs = state.cost_history[-10:]
            cost_change = max(recent_costs) - min(recent_costs)
            
            if cost_change < config.convergence_threshold:
                return True
        
        return False
    
    def _compute_quality_metrics(
        self,
        positions: torch.Tensor,
        circuit_graph: CircuitGraph,
        constraints: BoardConstraints,
    ) -> Dict[str, Any]:
        """
        Compute quality metrics for final placement.
        
        Args:
            positions: Final positions
            circuit_graph: Circuit graph
            constraints: Board constraints
            
        Returns:
            Dictionary of quality metrics
        """
        # Compute total trace length
        node_id_to_idx = {node.id: i for i, node in enumerate(circuit_graph.nodes)}
        total_trace_length = 0.0
        
        for edge in circuit_graph.edges:
            i = node_id_to_idx[edge.source_node]
            j = node_id_to_idx[edge.target_node]
            
            pos_i = positions[i]
            pos_j = positions[j]
            
            distance = torch.abs(pos_i[0] - pos_j[0]) + torch.abs(pos_i[1] - pos_j[1])
            total_trace_length += distance.item()
        
        # Check constraint violations
        constraint_violations = []
        for i, node in enumerate(circuit_graph.nodes):
            pos = positions[i]
            
            # Check board boundaries
            if pos[0] < 0 or pos[0] > constraints.width:
                constraint_violations.append(f"Component {node.id} outside board (x)")
            if pos[1] < 0 or pos[1] > constraints.height:
                constraint_violations.append(f"Component {node.id} outside board (y)")
        
        # Predict parasitics if GNN available
        parasitic_estimates = {}
        if self.gnn_model is not None:
            updated_graph = self._update_graph_positions(circuit_graph, positions)
            parasitic_estimates = self.gnn_model.predict_parasitics(updated_graph, device=None)
        
        metrics = {
            'total_trace_length': total_trace_length,
            'num_components': len(circuit_graph.nodes),
            'num_connections': len(circuit_graph.edges),
            'constraint_violations': constraint_violations,
            'parasitic_estimates': parasitic_estimates,
        }
        
        return metrics
