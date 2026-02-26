"""
Data augmentation utilities for PCB circuit graphs.

Implements rotation, mirroring, component permutation, and noise injection
to increase training data diversity while preserving electrical equivalence.
"""

import numpy as np
import copy
from typing import List, Tuple
from src.models.circuit_graph import CircuitNode, CircuitEdge, CircuitGraph


def rotate_circuit(circuit: CircuitGraph, angle: float) -> CircuitGraph:
    """
    Rotate circuit by specified angle.
    
    Args:
        circuit: Input circuit graph
        angle: Rotation angle in degrees (0, 90, 180, 270)
        
    Returns:
        Rotated circuit graph
    """
    if angle not in [0.0, 90.0, 180.0, 270.0]:
        raise ValueError(f"Angle must be 0, 90, 180, or 270 degrees, got {angle}")
    
    # Create deep copy
    rotated = copy.deepcopy(circuit)
    
    if angle == 0.0:
        return rotated
    
    board_width, board_height = circuit.board_size
    
    # Rotate each component
    for node in rotated.nodes:
        x, y = node.position
        
        if angle == 90.0:
            # Rotate 90° clockwise: (x, y) -> (y, board_width - x)
            new_x = y
            new_y = board_width - x
            new_orientation = (node.orientation + 90.0) % 360.0
        elif angle == 180.0:
            # Rotate 180°: (x, y) -> (board_width - x, board_height - y)
            new_x = board_width - x
            new_y = board_height - y
            new_orientation = (node.orientation + 180.0) % 360.0
        else:  # 270.0
            # Rotate 270° clockwise: (x, y) -> (board_height - y, x)
            new_x = board_height - y
            new_y = x
            new_orientation = (node.orientation + 270.0) % 360.0
        
        node.position = (new_x, new_y)
        node.orientation = new_orientation
    
    # Update board size for 90° and 270° rotations
    if angle in [90.0, 270.0]:
        rotated.board_size = (board_height, board_width)
    
    return rotated


def mirror_circuit(circuit: CircuitGraph, axis: str = "horizontal") -> CircuitGraph:
    """
    Mirror circuit along specified axis.
    
    Args:
        circuit: Input circuit graph
        axis: Mirror axis ("horizontal" or "vertical")
        
    Returns:
        Mirrored circuit graph
    """
    if axis not in ["horizontal", "vertical"]:
        raise ValueError(f"Axis must be 'horizontal' or 'vertical', got {axis}")
    
    # Create deep copy
    mirrored = copy.deepcopy(circuit)
    
    board_width, board_height = circuit.board_size
    
    # Mirror each component
    for node in mirrored.nodes:
        x, y = node.position
        
        if axis == "horizontal":
            # Mirror horizontally: (x, y) -> (board_width - x, y)
            new_x = board_width - x
            new_y = y
            # Mirror orientation
            new_orientation = (180.0 - node.orientation) % 360.0
        else:  # vertical
            # Mirror vertically: (x, y) -> (x, board_height - y)
            new_x = x
            new_y = board_height - y
            # Mirror orientation
            new_orientation = (360.0 - node.orientation) % 360.0
        
        node.position = (new_x, new_y)
        node.orientation = new_orientation
    
    return mirrored


def permute_components(circuit: CircuitGraph, random_seed: int = None) -> CircuitGraph:
    """
    Randomly permute equivalent components.
    
    Components of the same type with the same value are considered equivalent
    and can be permuted without changing circuit behavior.
    
    Args:
        circuit: Input circuit graph
        random_seed: Random seed for reproducibility
        
    Returns:
        Circuit with permuted components
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Create deep copy
    permuted = copy.deepcopy(circuit)
    
    # Group components by type and value
    component_groups = {}
    for i, node in enumerate(permuted.nodes):
        key = (node.component_type, node.value)
        if key not in component_groups:
            component_groups[key] = []
        component_groups[key].append(i)
    
    # Permute positions within each group
    for group_indices in component_groups.values():
        if len(group_indices) > 1:
            # Get positions of components in this group
            positions = [permuted.nodes[i].position for i in group_indices]
            orientations = [permuted.nodes[i].orientation for i in group_indices]
            
            # Shuffle positions
            shuffled_indices = np.random.permutation(len(positions))
            
            # Assign shuffled positions
            for i, shuffled_i in enumerate(shuffled_indices):
                permuted.nodes[group_indices[i]].position = positions[shuffled_i]
                permuted.nodes[group_indices[i]].orientation = orientations[shuffled_i]
    
    return permuted


def add_position_noise(
    circuit: CircuitGraph,
    noise_level: float = 0.05,
    random_seed: int = None
) -> CircuitGraph:
    """
    Add random noise to component positions.
    
    Args:
        circuit: Input circuit graph
        noise_level: Noise level as fraction of board size (default: 0.05 = 5%)
        random_seed: Random seed for reproducibility
        
    Returns:
        Circuit with noisy positions
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Create deep copy
    noisy = copy.deepcopy(circuit)
    
    board_width, board_height = circuit.board_size
    
    # Add noise to each component position
    for node in noisy.nodes:
        x, y = node.position
        
        # Generate noise
        noise_x = np.random.uniform(-noise_level * board_width, noise_level * board_width)
        noise_y = np.random.uniform(-noise_level * board_height, noise_level * board_height)
        
        # Add noise and clip to board boundaries
        new_x = np.clip(x + noise_x, 0, board_width)
        new_y = np.clip(y + noise_y, 0, board_height)
        
        node.position = (new_x, new_y)
    
    return noisy


def augment_circuit(
    circuit: CircuitGraph,
    rotation: bool = True,
    mirroring: bool = True,
    permutation: bool = True,
    noise: bool = True,
    noise_level: float = 0.05,
    random_seed: int = None
) -> List[CircuitGraph]:
    """
    Apply multiple augmentations to create diverse training samples.
    
    Args:
        circuit: Input circuit graph
        rotation: Whether to apply rotation augmentation
        mirroring: Whether to apply mirroring augmentation
        permutation: Whether to apply component permutation
        noise: Whether to add position noise
        noise_level: Noise level for position noise
        random_seed: Random seed for reproducibility
        
    Returns:
        List of augmented circuit graphs
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    augmented = [circuit]  # Include original
    
    # Rotation augmentation
    if rotation:
        for angle in [90.0, 180.0, 270.0]:
            augmented.append(rotate_circuit(circuit, angle))
    
    # Mirroring augmentation
    if mirroring:
        augmented.append(mirror_circuit(circuit, "horizontal"))
        augmented.append(mirror_circuit(circuit, "vertical"))
    
    # Component permutation
    if permutation:
        augmented.append(permute_components(circuit, random_seed))
    
    # Position noise
    if noise:
        augmented.append(add_position_noise(circuit, noise_level, random_seed))
    
    return augmented


def validate_augmentation(original: CircuitGraph, augmented: CircuitGraph) -> bool:
    """
    Validate that augmentation preserves circuit connectivity.
    
    Args:
        original: Original circuit graph
        augmented: Augmented circuit graph
        
    Returns:
        True if augmentation is valid (preserves connectivity)
    """
    # Check same number of nodes and edges
    if len(original.nodes) != len(augmented.nodes):
        return False
    
    if len(original.edges) != len(augmented.edges):
        return False
    
    # Check that all node IDs are present
    original_ids = {node.id for node in original.nodes}
    augmented_ids = {node.id for node in augmented.nodes}
    if original_ids != augmented_ids:
        return False
    
    # Check that all edges are preserved
    original_edges = {(edge.source_node, edge.target_node, edge.net_name) 
                     for edge in original.edges}
    augmented_edges = {(edge.source_node, edge.target_node, edge.net_name) 
                      for edge in augmented.edges}
    if original_edges != augmented_edges:
        return False
    
    # Check that components are within board boundaries
    board_width, board_height = augmented.board_size
    for node in augmented.nodes:
        x, y = node.position
        if not (0 <= x <= board_width and 0 <= y <= board_height):
            return False
    
    return True
