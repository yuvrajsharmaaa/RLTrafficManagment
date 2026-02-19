"""
Deep Q-Network (DQN) Agent for Traffic Signal Control.

Implements DQN with:
- Experience replay
- Target network
- Epsilon-greedy exploration
- Optional double DQN
- Optional dueling architecture
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import os
import json

from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class QNetwork(nn.Module):
    """
    Q-Network for estimating action values.
    
    Standard feedforward network with configurable hidden layers.
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_layers: List[int] = [256, 256],
                 activation: str = "relu"):
        """
        Initialize Q-Network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'leaky_relu')
        """
        super(QNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.01))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to get Q-values for all actions."""
        return self.network(state)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture.
    
    Separates state value (V) and advantage (A) estimation:
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_layers: List[int] = [256, 256],
                 activation: str = "relu"):
        super(DuelingQNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.LeakyReLU(0.01)
        
        # Shared feature layers
        self.feature_layers = nn.ModuleList()
        prev_dim = state_dim
        for hidden_dim in hidden_layers[:-1]:
            self.feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_layers[-1]),
            self.activation,
            nn.Linear(hidden_layers[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_layers[-1]),
            self.activation,
            nn.Linear(hidden_layers[-1], action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Shared features
        x = state
        for layer in self.feature_layers:
            x = self.activation(layer(x))
        
        # Value and advantage
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class DQNAgent:
    """
    DQN Agent for traffic signal control.
    
    Features:
    - Standard DQN or Dueling DQN
    - Experience replay (standard or prioritized)
    - Target network with soft/hard updates
    - Epsilon-greedy exploration with decay
    - Double DQN option
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 100000,
        target_update_freq: int = 100,
        hidden_layers: List[int] = [256, 256],
        activation: str = "relu",
        use_dueling: bool = False,
        use_double: bool = True,
        use_per: bool = False,
        tau: float = 0.005,
        seed: Optional[int] = None,
        device: Optional[str] = None
    ):
        """
        Initialize DQN Agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per step
            batch_size: Training batch size
            buffer_size: Replay buffer capacity
            target_update_freq: Steps between target updates
            hidden_layers: Hidden layer sizes
            activation: Activation function
            use_dueling: Whether to use dueling architecture
            use_double: Whether to use double DQN
            use_per: Whether to use prioritized replay
            tau: Soft update coefficient
            seed: Random seed
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double = use_double
        self.use_per = use_per
        self.tau = tau
        
        # Set seeds
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        NetworkClass = DuelingQNetwork if use_dueling else QNetwork
        self.q_network = NetworkClass(
            state_dim, action_dim, hidden_layers, activation
        ).to(self.device)
        self.target_network = NetworkClass(
            state_dim, action_dim, hidden_layers, activation
        ).to(self.device)
        
        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size, state_dim, seed=seed)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size, state_dim, seed=seed)
        
        # Training stats
        self.train_steps = 0
        self.episode_count = 0
        self.losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (use exploration)
            
        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Sample batch
        if self.use_per:
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            if self.use_double:
                # Double DQN: use online network to select actions
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN
                next_q = self.target_network(next_states).max(dim=1)[0]
            
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute TD errors
        td_errors = current_q - target_q
        
        # Update priorities if using PER
        if self.use_per:
            self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # Compute loss
        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self._update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def _update_target_network(self, soft: bool = True) -> None:
        """Update target network (soft or hard update)."""
        if soft:
            for target_param, param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
        else:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath: str) -> None:
        """Save agent state to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
            'episode_count': self.episode_count,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
                'use_double': self.use_double,
                'tau': self.tau
            }
        }
        torch.save(checkpoint, filepath)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load agent state from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_steps = checkpoint['train_steps']
        self.episode_count = checkpoint['episode_count']
        
        print(f"Agent loaded from {filepath}")
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given a state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_network(state_tensor).cpu().numpy()[0]
    
    def reset_epsilon(self) -> None:
        """Reset epsilon to starting value."""
        self.epsilon = self.epsilon_start
    
    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
            'episode_count': self.episode_count,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0
        }
