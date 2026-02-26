"""
RL-based router for PCB trace routing.

This module implements the RLRouter class that uses a trained policy network
to route PCB nets using reinforcement learning.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

from src.models.pcb_state import PCBState, RoutingResult, BoardConstraints
from src.training.routing_environment import RoutingEnvironment, RoutingAction
from src.services.rl_routing_agent import RoutingPolicyNetwork


logger = logging.getLogger(__name__)


class RLRouter:
    """
    RL-based routing using trained policy network.
    
    This class implements routing using a trained reinforcement learning policy.
    It integrates with the RoutingEnvironment and RoutingPolicyNetwork to perform
    action selection and routing execution.
    
    Attributes:
        policy_network: Trained policy network for action selection
        routing_env: Routing environment for state management and action execution
        device: PyTorch device (cuda/cpu) for inference
        max_steps_per_net: Maximum steps allowed per net routing
    """
    
    def __init__(
        self,
        policy_network: RoutingPolicyNetwork,
        routing_env: RoutingEnvironment,
        device: str = "cuda"
    ):
        """
        Initialize RL router.
        
        Args:
            policy_network: Trained policy network
            routing_env: Routing environment
            device: Device for inference (cuda/cpu)
        """
        self.policy_network = policy_network
        self.routing_env = routing_env
        
        # Set device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            if device == "cuda":
                logger.warning("CUDA requested but not available, falling back to CPU")
        
        # Move policy network to device
        self.policy_network.to(self.device)
        self.policy_network.eval()  # Set to evaluation mode
        
        self.max_steps_per_net = 10000
        
        logger.info(f"RLRouter initialized with device: {self.device}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load policy network from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True
            )
            
            # Extract state dict (handle different checkpoint formats)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load state dict into policy network
            self.policy_network.load_state_dict(state_dict)
            self.policy_network.eval()
            
            logger.info("Checkpoint loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Checkpoint loading failed: {e}")
    
    def route(
        self,
        pcb_state: PCBState,
        nets: List[str],
        max_steps: int = 10000
    ) -> RoutingResult:
        """
        Route nets using RL policy.
        
        This method routes the specified nets using the trained policy network.
        It initializes the routing environment with the PCB state, then iteratively
        selects actions using the policy network and executes them in the environment
        until all nets are routed or the maximum step limit is reached.
        
        Args:
            pcb_state: Current PCB state
            nets: List of nets to route
            max_steps: Maximum steps per net
            
        Returns:
            RoutingResult with routed traces and statistics
        """
        start_time = time.time()
        
        # Handle empty nets list
        if not nets:
            logger.info("No nets to route, returning empty result")
            return RoutingResult(
                success=True,
                routed_pcb={'traces': {}, 'pcb_state': pcb_state},
                routing_stats={
                    'total_trace_length': 0.0,
                    'via_count': 0,
                    'nets_routed': 0,
                    'nets_total': 0,
                    'drc_violation_count': 0,
                    'total_steps': 0,
                    'routing_time': 0.0
                },
                routing_time=0.0,
                routing_success_rate=0.0,
                unrouted_nets=[],
                drc_violations=[],
                error_message=""
            )
        
        # Update max steps
        self.max_steps_per_net = max_steps
        self.routing_env.max_steps_per_episode = max_steps * len(nets)
        
        # Initialize environment with PCB state
        logger.info(f"Starting RL routing for {len(nets)} nets")
        current_state = self._initialize_environment(pcb_state, nets)
        
        # Track routing statistics
        routed_traces: Dict[str, List[Dict[str, Any]]] = {}
        total_steps = 0
        nets_routed = 0
        
        # Route all nets
        done = False
        while not done and total_steps < self.routing_env.max_steps_per_episode:
            # Select action using policy network
            action = self._select_action(current_state)
            
            # Execute action in environment
            next_state, reward, done, info = self.routing_env.step(action)
            
            # Update state
            current_state = next_state
            total_steps += 1
            
            # Check if a net was completed
            if info.get('valid_action') and action == RoutingAction.FINISH_NET:
                completed_net = info.get('current_net')
                if completed_net and completed_net not in routed_traces:
                    # Extract trace for completed net
                    trace = self._extract_trace_from_state(current_state, completed_net)
                    routed_traces[completed_net] = trace
                    nets_routed += 1
                    logger.info(f"Completed routing net {completed_net} ({nets_routed}/{len(nets)})")
            
            # Log progress periodically
            if total_steps % 1000 == 0:
                logger.debug(f"Routing step {total_steps}, nets completed: {nets_routed}/{len(nets)}")
        
        # Calculate routing time
        routing_time = time.time() - start_time
        
        # Determine unrouted nets
        unrouted_nets = [net for net in nets if net not in routed_traces]
        
        # Calculate success rate
        routing_success_rate = nets_routed / len(nets) if nets else 0.0
        
        # Build routing statistics
        routing_stats = {
            'total_trace_length': current_state.total_trace_length,
            'via_count': current_state.via_count,
            'nets_routed': nets_routed,
            'nets_total': len(nets),
            'drc_violation_count': len(current_state.drc_violations),
            'total_steps': total_steps,
            'routing_time': routing_time
        }
        
        # Create routing result
        result = RoutingResult(
            success=(routing_success_rate == 1.0),
            routed_pcb={'traces': routed_traces, 'pcb_state': current_state},
            routing_stats=routing_stats,
            routing_time=routing_time,
            routing_success_rate=routing_success_rate,
            unrouted_nets=unrouted_nets,
            drc_violations=current_state.drc_violations,
            error_message="" if routing_success_rate == 1.0 else f"Failed to route {len(unrouted_nets)} nets"
        )
        
        logger.info(
            f"RL routing completed: {nets_routed}/{len(nets)} nets routed "
            f"in {routing_time:.2f}s ({total_steps} steps)"
        )
        
        return result
    
    def _initialize_environment(
        self,
        pcb_state: PCBState,
        nets: List[str]
    ) -> PCBState:
        """
        Initialize routing environment with PCB state and nets.
        
        Args:
            pcb_state: Initial PCB state
            nets: List of nets to route
            
        Returns:
            Initialized PCB state from environment
        """
        # Reset environment
        env_state = self.routing_env.reset()
        
        # Update environment state with provided PCB state if needed
        # For now, we use the environment's reset state
        # In a more sophisticated implementation, we would merge the states
        
        # Update unrouted nets list
        env_state.unrouted_nets = nets.copy()
        env_state.routed_nets = []
        
        # Clear current_net if no nets to route
        if not nets:
            env_state.current_net = ""
        
        return env_state
    
    def _select_action(self, state: PCBState) -> int:
        """
        Select action using policy network.
        
        Args:
            state: Current PCB state
            
        Returns:
            Selected action index (0-7)
        """
        # Convert state to tensor
        state_tensor = state.to_tensor()
        state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
        state_tensor = state_tensor.to(self.device)
        
        # Get action from policy network
        with torch.no_grad():
            action_logits = self.policy_network(state_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)
            
            # Sample action from distribution
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
        
        return action.item()
    
    def _extract_trace_from_state(
        self,
        state: PCBState,
        net_name: str
    ) -> List[Dict[str, Any]]:
        """
        Extract trace path for a completed net from the PCB state.
        
        Args:
            state: Current PCB state
            net_name: Name of the net
            
        Returns:
            List of trace segments with positions and metadata
        """
        # Extract trace path from routing environment's internal state
        if hasattr(self.routing_env, 'routing_state') and self.routing_env.routing_state:
            trace_path = self.routing_env.routing_state.trace_path.copy()
        else:
            trace_path = []
        
        # Convert trace path to trace segments
        traces = []
        if trace_path:
            trace_segment = {
                'net_name': net_name,
                'path': trace_path,
                'width': self.routing_env.board_constraints.trace_width,
                'length': len(trace_path) * self.routing_env.grid_resolution
            }
            traces.append(trace_segment)
        
        return traces
