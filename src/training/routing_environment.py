"""
PCB routing environment for reinforcement learning.

This module implements the routing problem as a Markov Decision Process (MDP)
with multi-channel grid state representation and discrete action space.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from enum import IntEnum

from src.models.pcb_state import PCBState, BoardConstraints
from src.models.circuit_graph import CircuitGraph, CircuitNode


class RoutingAction(IntEnum):
    """
    Discrete action space for PCB routing.
    
    8 total actions:
    - 4 movement directions (North, South, East, West)
    - 2 layer transitions (Up, Down)
    - 1 via placement
    - 1 net completion
    """
    NORTH = 0      # Move north (decrease y)
    SOUTH = 1      # Move south (increase y)
    EAST = 2       # Move east (increase x)
    WEST = 3       # Move west (decrease x)
    LAYER_UP = 4   # Transition to upper layer
    LAYER_DOWN = 5 # Transition to lower layer
    PLACE_VIA = 6  # Place via at current position
    FINISH_NET = 7 # Complete current net routing


@dataclass
class RoutingState:
    """
    Internal routing state tracking.
    
    Attributes:
        current_position: Current routing position (x, y, layer)
        current_layer: Current PCB layer
        trace_path: List of positions in current trace
        target_pins: Target pin positions for current net
        obstacles: Set of obstacle positions
    """
    current_position: Tuple[int, int, int] = (0, 0, 0)  # x, y, layer
    current_layer: int = 0
    trace_path: List[Tuple[int, int, int]] = None
    target_pins: List[Tuple[int, int]] = None
    obstacles: set = None
    
    def __post_init__(self):
        if self.trace_path is None:
            self.trace_path = []
        if self.target_pins is None:
            self.target_pins = []
        if self.obstacles is None:
            self.obstacles = set()


class RoutingEnvironment:
    """
    PCB routing environment implementing MDP interface.
    
    This environment represents the routing problem as a Markov Decision Process:
    - State: Multi-channel grid (H × W × C) encoding PCB layout
    - Actions: 8 discrete actions (4 directions, 2 layer transitions, via, finish)
    - Rewards: Based on trace length, via count, DRC violations, net completion
    - Transitions: Deterministic state updates based on actions
    
    Attributes:
        circuit_graph: Circuit topology and component information
        board_constraints: PCB board constraints and design rules
        grid_resolution: Grid resolution in mm per cell
        state: Current PCB state
        routing_state: Internal routing state tracking
    """
    
    def __init__(
        self,
        circuit_graph: CircuitGraph,
        board_constraints: BoardConstraints,
        grid_resolution: float = 0.5  # mm per grid cell
    ):
        """
        Initialize routing environment.
        
        Args:
            circuit_graph: Circuit graph with components and connections
            board_constraints: Board constraints and design rules
            grid_resolution: Grid resolution in mm per cell
        """
        self.circuit_graph = circuit_graph
        self.board_constraints = board_constraints
        self.grid_resolution = grid_resolution
        
        # Calculate grid dimensions
        self.grid_height = int(np.ceil(board_constraints.height / grid_resolution))
        self.grid_width = int(np.ceil(board_constraints.width / grid_resolution))
        
        # Calculate number of channels
        # Base channels: component occupancy, existing traces, current net, 
        # target pins, obstacles, layer assignment
        self.num_base_channels = 6
        self.num_channels = self.num_base_channels + board_constraints.num_layers
        
        # Initialize state
        self.state: Optional[PCBState] = None
        self.routing_state: Optional[RoutingState] = None
        
        # Action space
        self.action_space_size = 8
        
        # Episode tracking
        self.episode_step = 0
        self.max_steps_per_episode = 10000
        
    def reset(self) -> PCBState:
        """
        Reset environment to initial state.
        
        Returns:
            Initial PCB state
        """
        # Initialize multi-channel grid
        grid = np.zeros(
            (self.grid_height, self.grid_width, self.num_channels),
            dtype=np.float32
        )
        
        # Channel 0: Component occupancy
        grid[:, :, 0] = self._create_component_occupancy_channel()
        
        # Channel 1: Existing traces (initially empty)
        grid[:, :, 1] = 0.0
        
        # Channel 2: Current net (initially empty)
        grid[:, :, 2] = 0.0
        
        # Channel 3: Target pins (will be set when routing starts)
        grid[:, :, 3] = 0.0
        
        # Channel 4: Obstacles and keepout zones
        grid[:, :, 4] = self._create_obstacles_channel()
        
        # Channel 5: Layer assignment (initially layer 0)
        grid[:, :, 5] = 0.0
        
        # Channels 6+: Per-layer trace information
        for layer_idx in range(self.board_constraints.num_layers):
            grid[:, :, 6 + layer_idx] = 0.0
        
        # Get list of nets to route
        nets = self._get_nets_from_graph()
        
        # Create PCB state
        self.state = PCBState(
            grid=grid,
            current_net="",
            routed_nets=[],
            unrouted_nets=nets,
            via_count=0,
            total_trace_length=0.0,
            drc_violations=[]
        )
        
        # Initialize routing state
        self.routing_state = RoutingState()
        
        # Start routing first net
        if self.state.unrouted_nets:
            self._start_routing_net(self.state.unrouted_nets[0])
        
        self.episode_step = 0
        
        return self.state
    
    def step(self, action: int) -> Tuple[PCBState, float, bool, Dict[str, Any]]:
        """
        Execute action and return next state, reward, done flag, and info.
        
        Args:
            action: Action to execute (0-7)
        
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Store previous state for reward computation
        prev_state = self._copy_state(self.state)
        
        # Execute action
        action_enum = RoutingAction(action)
        valid_action = self._execute_action(action_enum)
        
        # Compute reward
        reward = self._compute_reward(prev_state, self.state, action_enum, valid_action)
        
        # Check if episode is done
        done = self._is_done()
        
        # Increment step counter
        self.episode_step += 1
        
        # Check for max steps
        if self.episode_step >= self.max_steps_per_episode:
            done = True
        
        # Gather info
        info = {
            "valid_action": valid_action,
            "current_net": self.state.current_net,
            "routed_nets_count": len(self.state.routed_nets),
            "unrouted_nets_count": len(self.state.unrouted_nets),
            "via_count": self.state.via_count,
            "trace_length": self.state.total_trace_length,
            "drc_violations": len(self.state.drc_violations),
            "episode_step": self.episode_step
        }
        
        return self.state, reward, done, info
    
    def _execute_action(self, action: RoutingAction) -> bool:
        """
        Execute routing action and update state.
        
        Args:
            action: Action to execute
        
        Returns:
            True if action was valid and executed, False otherwise
        """
        if action == RoutingAction.NORTH:
            return self._move_north()
        elif action == RoutingAction.SOUTH:
            return self._move_south()
        elif action == RoutingAction.EAST:
            return self._move_east()
        elif action == RoutingAction.WEST:
            return self._move_west()
        elif action == RoutingAction.LAYER_UP:
            return self._transition_layer_up()
        elif action == RoutingAction.LAYER_DOWN:
            return self._transition_layer_down()
        elif action == RoutingAction.PLACE_VIA:
            return self._place_via()
        elif action == RoutingAction.FINISH_NET:
            return self._finish_net()
        else:
            return False
    
    def _move_north(self) -> bool:
        """Move routing position north (decrease y)."""
        x, y, layer = self.routing_state.current_position
        new_y = y - 1
        
        if new_y < 0:
            return False  # Out of bounds
        
        new_pos = (x, new_y, layer)
        if self._is_position_valid(new_pos):
            self.routing_state.current_position = new_pos
            self.routing_state.trace_path.append(new_pos)
            self._update_trace_channel(new_pos)
            self._update_current_net_channel(new_pos)
            return True
        return False
    
    def _move_south(self) -> bool:
        """Move routing position south (increase y)."""
        x, y, layer = self.routing_state.current_position
        new_y = y + 1
        
        if new_y >= self.grid_height:
            return False  # Out of bounds
        
        new_pos = (x, new_y, layer)
        if self._is_position_valid(new_pos):
            self.routing_state.current_position = new_pos
            self.routing_state.trace_path.append(new_pos)
            self._update_trace_channel(new_pos)
            self._update_current_net_channel(new_pos)
            return True
        return False
    
    def _move_east(self) -> bool:
        """Move routing position east (increase x)."""
        x, y, layer = self.routing_state.current_position
        new_x = x + 1
        
        if new_x >= self.grid_width:
            return False  # Out of bounds
        
        new_pos = (new_x, y, layer)
        if self._is_position_valid(new_pos):
            self.routing_state.current_position = new_pos
            self.routing_state.trace_path.append(new_pos)
            self._update_trace_channel(new_pos)
            self._update_current_net_channel(new_pos)
            return True
        return False
    
    def _move_west(self) -> bool:
        """Move routing position west (decrease x)."""
        x, y, layer = self.routing_state.current_position
        new_x = x - 1
        
        if new_x < 0:
            return False  # Out of bounds
        
        new_pos = (new_x, y, layer)
        if self._is_position_valid(new_pos):
            self.routing_state.current_position = new_pos
            self.routing_state.trace_path.append(new_pos)
            self._update_trace_channel(new_pos)
            self._update_current_net_channel(new_pos)
            return True
        return False
    
    def _transition_layer_up(self) -> bool:
        """Transition to upper layer."""
        x, y, layer = self.routing_state.current_position
        new_layer = layer - 1
        
        if new_layer < 0:
            return False  # No upper layer
        
        new_pos = (x, y, new_layer)
        self.routing_state.current_position = new_pos
        self.routing_state.current_layer = new_layer
        self.routing_state.trace_path.append(new_pos)
        self._update_layer_channel(new_layer)
        return True
    
    def _transition_layer_down(self) -> bool:
        """Transition to lower layer."""
        x, y, layer = self.routing_state.current_position
        new_layer = layer + 1
        
        if new_layer >= self.board_constraints.num_layers:
            return False  # No lower layer
        
        new_pos = (x, y, new_layer)
        self.routing_state.current_position = new_pos
        self.routing_state.current_layer = new_layer
        self.routing_state.trace_path.append(new_pos)
        self._update_layer_channel(new_layer)
        return True
    
    def _place_via(self) -> bool:
        """Place via at current position."""
        x, y, layer = self.routing_state.current_position
        
        # Check if via can be placed (not on obstacle, not too close to other vias)
        if not self._can_place_via((x, y)):
            return False
        
        # Place via
        self.state.via_count += 1
        
        # Mark via in grid (could add via channel if needed)
        # For now, vias are implicit in layer transitions
        
        return True
    
    def _finish_net(self) -> bool:
        """Finish routing current net."""
        if not self.state.current_net:
            return False
        
        # Check if we reached a target pin
        x, y, _ = self.routing_state.current_position
        if not self._is_at_target_pin((x, y)):
            return False  # Must be at target to finish
        
        # Mark net as routed
        self.state.routed_nets.append(self.state.current_net)
        self.state.unrouted_nets.remove(self.state.current_net)
        
        # Update trace length
        trace_length = len(self.routing_state.trace_path) * self.grid_resolution
        self.state.total_trace_length += trace_length
        
        # Commit current trace to existing traces channel
        self._commit_trace_to_grid()
        
        # Clear current net channel
        self.state.grid[:, :, 2] = 0.0
        
        # Start routing next net if available
        if self.state.unrouted_nets:
            self._start_routing_net(self.state.unrouted_nets[0])
        else:
            self.state.current_net = ""
        
        return True
    
    def _compute_reward(
        self,
        prev_state: PCBState,
        next_state: PCBState,
        action: RoutingAction,
        valid_action: bool
    ) -> float:
        """
        Compute reward for state transition.
        
        Reward function:
        - Penalize trace length increase: -1.0 per unit
        - Penalize via placement: -2.0 per via
        - Penalize layer transitions: -0.5 per transition
        - Large penalty for DRC violations: -100.0
        - Large reward for net completion: +100.0
        - Invalid action penalty: -10.0
        
        Args:
            prev_state: Previous state
            next_state: Next state
            action: Action taken
            valid_action: Whether action was valid
        
        Returns:
            Reward value
        """
        if not valid_action:
            return -10.0  # Penalty for invalid action
        
        reward = 0.0
        
        # Penalize trace length increase
        trace_length_delta = next_state.total_trace_length - prev_state.total_trace_length
        reward -= 1.0 * trace_length_delta
        
        # Penalize via placement
        via_count_delta = next_state.via_count - prev_state.via_count
        reward -= 2.0 * via_count_delta
        
        # Penalize layer transitions
        if action in [RoutingAction.LAYER_UP, RoutingAction.LAYER_DOWN]:
            reward -= 0.5
        
        # Large penalty for DRC violations
        drc_violations_delta = len(next_state.drc_violations) - len(prev_state.drc_violations)
        reward -= 100.0 * drc_violations_delta
        
        # Large reward for net completion
        routed_nets_delta = len(next_state.routed_nets) - len(prev_state.routed_nets)
        reward += 100.0 * routed_nets_delta
        
        # Episode completion bonus
        if len(next_state.unrouted_nets) == 0 and len(prev_state.unrouted_nets) > 0:
            reward += 100.0
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if episode is complete."""
        # Episode is done when all nets are routed or no current net
        return len(self.state.unrouted_nets) == 0 and not self.state.current_net
    
    def _create_component_occupancy_channel(self) -> np.ndarray:
        """
        Create component occupancy channel.
        
        Returns:
            Binary grid indicating component positions
        """
        channel = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        
        for node in self.circuit_graph.nodes:
            x, y = node.position
            width, height, _ = node.dimensions
            
            # Convert to grid coordinates
            grid_x = int(x / self.grid_resolution)
            grid_y = int(y / self.grid_resolution)
            grid_w = max(1, int(width / self.grid_resolution))
            grid_h = max(1, int(height / self.grid_resolution))
            
            # Mark component area
            x_start = max(0, grid_x)
            x_end = min(self.grid_width, grid_x + grid_w)
            y_start = max(0, grid_y)
            y_end = min(self.grid_height, grid_y + grid_h)
            
            channel[y_start:y_end, x_start:x_end] = 1.0
        
        return channel
    
    def _create_obstacles_channel(self) -> np.ndarray:
        """
        Create obstacles and keepout zones channel.
        
        Returns:
            Binary grid indicating obstacle positions
        """
        channel = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        
        for zone in self.board_constraints.keepout_zones:
            x1, y1, x2, y2 = zone
            
            # Convert to grid coordinates
            grid_x1 = int(x1 / self.grid_resolution)
            grid_y1 = int(y1 / self.grid_resolution)
            grid_x2 = int(x2 / self.grid_resolution)
            grid_y2 = int(y2 / self.grid_resolution)
            
            # Mark keepout area
            x_start = max(0, grid_x1)
            x_end = min(self.grid_width, grid_x2)
            y_start = max(0, grid_y1)
            y_end = min(self.grid_height, grid_y2)
            
            channel[y_start:y_end, x_start:x_end] = 1.0
        
        return channel
    
    def _get_nets_from_graph(self) -> List[str]:
        """
        Extract list of nets from circuit graph.
        
        Returns:
            List of net names
        """
        nets = set()
        for edge in self.circuit_graph.edges:
            nets.add(edge.net_name)
        return sorted(list(nets))
    
    def _start_routing_net(self, net_name: str):
        """
        Start routing a new net.
        
        Args:
            net_name: Name of net to route
        """
        self.state.current_net = net_name
        
        # Find edges for this net
        net_edges = [e for e in self.circuit_graph.edges if e.net_name == net_name]
        
        if not net_edges:
            return
        
        # Get source and target nodes
        # For simplicity, use first edge's source as start, target as end
        edge = net_edges[0]
        source_node = self.circuit_graph.get_node_by_id(edge.source_node)
        target_node = self.circuit_graph.get_node_by_id(edge.target_node)
        
        if not source_node or not target_node:
            return
        
        # Set starting position
        start_x = int(source_node.position[0] / self.grid_resolution)
        start_y = int(source_node.position[1] / self.grid_resolution)
        start_layer = source_node.layer
        
        self.routing_state.current_position = (start_x, start_y, start_layer)
        self.routing_state.current_layer = start_layer
        self.routing_state.trace_path = [(start_x, start_y, start_layer)]
        
        # Set target pins
        target_x = int(target_node.position[0] / self.grid_resolution)
        target_y = int(target_node.position[1] / self.grid_resolution)
        self.routing_state.target_pins = [(target_x, target_y)]
        
        # Update target pins channel
        self.state.grid[:, :, 3] = 0.0
        if 0 <= target_y < self.grid_height and 0 <= target_x < self.grid_width:
            self.state.grid[target_y, target_x, 3] = 1.0
        
        # Update current net channel
        if 0 <= start_y < self.grid_height and 0 <= start_x < self.grid_width:
            self.state.grid[start_y, start_x, 2] = 1.0
    
    def _is_position_valid(self, position: Tuple[int, int, int]) -> bool:
        """
        Check if position is valid for routing.
        
        Args:
            position: (x, y, layer) position
        
        Returns:
            True if position is valid
        """
        x, y, layer = position
        
        # Check bounds
        if x < 0 or x >= self.grid_width:
            return False
        if y < 0 or y >= self.grid_height:
            return False
        if layer < 0 or layer >= self.board_constraints.num_layers:
            return False
        
        # Check obstacles
        if self.state.grid[y, x, 4] > 0.5:  # Obstacle channel
            return False
        
        # Check component occupancy (allow routing over components for now)
        # In a more sophisticated version, this could be restricted
        
        return True
    
    def _can_place_via(self, position: Tuple[int, int]) -> bool:
        """
        Check if via can be placed at position.
        
        Args:
            position: (x, y) position
        
        Returns:
            True if via can be placed
        """
        x, y = position
        
        # Check bounds
        if x < 0 or x >= self.grid_width:
            return False
        if y < 0 or y >= self.grid_height:
            return False
        
        # Check obstacles
        if self.state.grid[y, x, 4] > 0.5:
            return False
        
        # Could add more sophisticated via placement rules here
        
        return True
    
    def _is_at_target_pin(self, position: Tuple[int, int]) -> bool:
        """
        Check if current position is at a target pin.
        
        Args:
            position: (x, y) position
        
        Returns:
            True if at target pin
        """
        return position in self.routing_state.target_pins
    
    def _update_trace_channel(self, position: Tuple[int, int, int]):
        """Update existing traces channel with new trace segment."""
        x, y, layer = position
        if 0 <= y < self.grid_height and 0 <= x < self.grid_width:
            self.state.grid[y, x, 1] = 1.0  # Existing traces channel
            # Also update per-layer channel
            if 6 + layer < self.num_channels:
                self.state.grid[y, x, 6 + layer] = 1.0
    
    def _update_current_net_channel(self, position: Tuple[int, int, int]):
        """Update current net channel."""
        x, y, _ = position
        if 0 <= y < self.grid_height and 0 <= x < self.grid_width:
            self.state.grid[y, x, 2] = 1.0  # Current net channel
    
    def _update_layer_channel(self, layer: int):
        """Update layer assignment channel."""
        # Set layer channel to indicate current layer
        self.state.grid[:, :, 5] = float(layer) / max(1, self.board_constraints.num_layers - 1)
    
    def _commit_trace_to_grid(self):
        """Commit current trace path to existing traces channel."""
        for x, y, layer in self.routing_state.trace_path:
            if 0 <= y < self.grid_height and 0 <= x < self.grid_width:
                self.state.grid[y, x, 1] = 1.0
                if 6 + layer < self.num_channels:
                    self.state.grid[y, x, 6 + layer] = 1.0
    
    def _copy_state(self, state: PCBState) -> PCBState:
        """Create a copy of PCB state for reward computation."""
        return PCBState(
            grid=state.grid.copy(),
            current_net=state.current_net,
            routed_nets=state.routed_nets.copy(),
            unrouted_nets=state.unrouted_nets.copy(),
            via_count=state.via_count,
            total_trace_length=state.total_trace_length,
            drc_violations=state.drc_violations.copy()
        )
    
    def get_state_shape(self) -> Tuple[int, int, int]:
        """
        Get shape of state representation.
        
        Returns:
            Tuple of (height, width, channels)
        """
        return (self.grid_height, self.grid_width, self.num_channels)
    
    def get_action_space_size(self) -> int:
        """
        Get size of action space.
        
        Returns:
            Number of discrete actions (8)
        """
        return self.action_space_size
    
    def render(self) -> str:
        """
        Render current state as string for debugging.
        
        Returns:
            String representation of state
        """
        if self.state is None:
            return "Environment not initialized"
        
        lines = []
        lines.append(f"Current Net: {self.state.current_net}")
        lines.append(f"Routed Nets: {len(self.state.routed_nets)}/{len(self.state.routed_nets) + len(self.state.unrouted_nets)}")
        lines.append(f"Via Count: {self.state.via_count}")
        lines.append(f"Trace Length: {self.state.total_trace_length:.2f} mm")
        lines.append(f"DRC Violations: {len(self.state.drc_violations)}")
        lines.append(f"Episode Step: {self.episode_step}")
        
        if self.routing_state:
            x, y, layer = self.routing_state.current_position
            lines.append(f"Current Position: ({x}, {y}, layer {layer})")
            lines.append(f"Trace Path Length: {len(self.routing_state.trace_path)}")
        
        return "\n".join(lines)
