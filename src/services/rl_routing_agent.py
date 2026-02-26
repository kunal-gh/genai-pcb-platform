"""
RL routing agent with policy network for PCB trace routing.

This module implements the reinforcement learning agent for automated PCB routing,
including the policy network architecture and value network for actor-critic methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class RoutingPolicyNetwork(nn.Module):
    """
    Convolutional neural network for routing policy.
    
    This network processes the spatial PCB state representation and outputs
    action logits for the routing agent. It uses a CNN architecture to capture
    spatial patterns in the PCB layout.
    
    Architecture:
    - Input: PCB state (C × H × W) where C is number of channels
    - 4 Conv2D layers: [32, 64, 128, 256] channels
    - Kernel size: 3×3, stride: 1, padding: 1
    - Activation: ReLU
    - 2 FC layers: [512, 256]
    - Output: Action logits (8 actions)
    
    The network is designed to work with PPO (Proximal Policy Optimization)
    and can be integrated with Ray RLlib or Stable Baselines3.
    """
    
    def __init__(
        self,
        input_channels: int,
        action_dim: int = 8,
        hidden_dims: Tuple[int, int] = (512, 256)
    ):
        """
        Initialize the routing policy network.
        
        Args:
            input_channels: Number of input channels in the state representation
            action_dim: Number of discrete actions (default: 8)
            hidden_dims: Dimensions of fully connected layers (default: (512, 256))
        """
        super(RoutingPolicyNetwork, self).__init__()
        
        self.input_channels = input_channels
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Convolutional layers
        # Conv1: input_channels -> 32 channels
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Conv2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Conv3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Conv4: 128 -> 256 channels
        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Adaptive pooling to handle variable input sizes
        # This ensures consistent feature size regardless of input spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        # After adaptive pooling: 256 channels * 4 * 4 = 4096 features
        self.fc1 = nn.Linear(256 * 4 * 4, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.action_head = nn.Linear(hidden_dims[1], action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights using He initialization for ReLU activations.
        
        He initialization is optimal for ReLU activations as it maintains
        variance across layers during forward and backward passes.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # He initialization for convolutional layers
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.Linear):
                # He initialization for linear layers
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the policy network.
        
        Args:
            state: Input state tensor of shape (batch_size, C, H, W)
                   or (C, H, W) for single state
        
        Returns:
            Action logits of shape (batch_size, action_dim) or (action_dim,)
        """
        # Handle single state (add batch dimension)
        single_state = False
        if state.dim() == 3:
            state = state.unsqueeze(0)
            single_state = True
        
        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Adaptive pooling to handle variable input sizes
        x = self.adaptive_pool(x)
        
        # Flatten for fully connected layers
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Action logits (no activation)
        action_logits = self.action_head(x)
        
        # Remove batch dimension if input was single state
        if single_state:
            action_logits = action_logits.squeeze(0)
        
        return action_logits
    
    def get_action_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities using softmax.
        
        Args:
            state: Input state tensor
        
        Returns:
            Action probabilities (sums to 1)
        """
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)
    
    def sample_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Sample an action from the policy distribution.
        
        Args:
            state: Input state tensor (C, H, W)
        
        Returns:
            Tuple of (action_index, log_probability)
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        # Sample action from categorical distribution
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob


class RoutingValueNetwork(nn.Module):
    """
    Value network for actor-critic RL algorithms (e.g., PPO).
    
    This network estimates the state value V(s), which represents the expected
    cumulative reward from a given state. It shares the same CNN architecture
    as the policy network but outputs a single scalar value.
    
    Architecture:
    - Input: PCB state (C × H × W)
    - 4 Conv2D layers: [32, 64, 128, 256] channels
    - Kernel size: 3×3, stride: 1, padding: 1
    - Activation: ReLU
    - 2 FC layers: [512, 256]
    - Output: State value (scalar)
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_dims: Tuple[int, int] = (512, 256)
    ):
        """
        Initialize the value network.
        
        Args:
            input_channels: Number of input channels in the state representation
            hidden_dims: Dimensions of fully connected layers (default: (512, 256))
        """
        super(RoutingValueNetwork, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_dims = hidden_dims
        
        # Convolutional layers (same as policy network)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.value_head = nn.Linear(hidden_dims[1], 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the value network.
        
        Args:
            state: Input state tensor of shape (batch_size, C, H, W)
                   or (C, H, W) for single state
        
        Returns:
            State value of shape (batch_size, 1) or (1,)
        """
        # Handle single state (add batch dimension)
        single_state = False
        if state.dim() == 3:
            state = state.unsqueeze(0)
            single_state = True
        
        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Adaptive pooling to handle variable input sizes
        x = self.adaptive_pool(x)
        
        # Flatten for fully connected layers
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Value output (no activation)
        value = self.value_head(x)
        
        # Remove batch dimension if input was single state
        if single_state:
            value = value.squeeze(0)
        
        return value


class ActorCriticNetwork(nn.Module):
    """
    Combined actor-critic network for PPO.
    
    This network combines both policy (actor) and value (critic) networks,
    which is the standard architecture for PPO. It can share convolutional
    features between the two heads for efficiency.
    """
    
    def __init__(
        self,
        input_channels: int,
        action_dim: int = 8,
        hidden_dims: Tuple[int, int] = (512, 256),
        shared_features: bool = True
    ):
        """
        Initialize the actor-critic network.
        
        Args:
            input_channels: Number of input channels in the state representation
            action_dim: Number of discrete actions (default: 8)
            hidden_dims: Dimensions of fully connected layers (default: (512, 256))
            shared_features: Whether to share convolutional features (default: True)
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.input_channels = input_channels
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.shared_features = shared_features
        
        if shared_features:
            # Shared convolutional layers
            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            
            # Adaptive pooling to handle variable input sizes
            self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
            
            # Separate FC layers for actor and critic
            self.actor_fc1 = nn.Linear(256 * 4 * 4, hidden_dims[0])
            self.actor_fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
            self.actor_head = nn.Linear(hidden_dims[1], action_dim)
            
            self.critic_fc1 = nn.Linear(256 * 4 * 4, hidden_dims[0])
            self.critic_fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
            self.critic_head = nn.Linear(hidden_dims[1], 1)
        else:
            # Separate networks
            self.actor = RoutingPolicyNetwork(input_channels, action_dim, hidden_dims)
            self.critic = RoutingValueNetwork(input_channels, hidden_dims)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        if self.shared_features:
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        module.weight,
                        mode='fan_out',
                        nonlinearity='relu'
                    )
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                
                elif isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(
                        module.weight,
                        mode='fan_out',
                        nonlinearity='relu'
                    )
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both actor and critic.
        
        Args:
            state: Input state tensor of shape (batch_size, C, H, W)
                   or (C, H, W) for single state
        
        Returns:
            Tuple of (action_logits, state_value)
        """
        if not self.shared_features:
            action_logits = self.actor(state)
            value = self.critic(state)
            return action_logits, value
        
        # Handle single state (add batch dimension)
        single_state = False
        if state.dim() == 3:
            state = state.unsqueeze(0)
            single_state = True
        
        # Shared convolutional features
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Adaptive pooling to handle variable input sizes
        x = self.adaptive_pool(x)
        
        # Flatten
        batch_size = x.size(0)
        features = x.view(batch_size, -1)
        
        # Actor path
        actor_x = F.relu(self.actor_fc1(features))
        actor_x = F.relu(self.actor_fc2(actor_x))
        action_logits = self.actor_head(actor_x)
        
        # Critic path
        critic_x = F.relu(self.critic_fc1(features))
        critic_x = F.relu(self.critic_fc2(critic_x))
        value = self.critic_head(critic_x)
        
        # Remove batch dimension if input was single state
        if single_state:
            action_logits = action_logits.squeeze(0)
            value = value.squeeze(0)
        
        return action_logits, value
    
    def get_action_and_value(
        self,
        state: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action and get value for a state.
        
        Args:
            state: Input state tensor (C, H, W)
        
        Returns:
            Tuple of (action_index, log_probability, state_value)
        """
        action_logits, value = self.forward(state)
        probs = F.softmax(action_logits, dim=-1)
        
        # Sample action
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob, value
