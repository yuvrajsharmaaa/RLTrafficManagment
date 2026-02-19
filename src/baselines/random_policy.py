"""
Random Policy Baseline for Traffic Signal Control.

Simple baseline that selects actions uniformly at random.
"""

import numpy as np
from typing import Optional


class RandomPolicy:
    """
    Random Policy: Selects actions uniformly at random.
    
    Used as a baseline to compare against learned policies.
    Performance should be poor compared to learned or fixed-time policies.
    """
    
    def __init__(self, action_dim: int, seed: Optional[int] = None):
        """
        Initialize random policy.
        
        Args:
            action_dim: Number of discrete actions (phases)
            seed: Random seed for reproducibility
        """
        self.action_dim = action_dim
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        self.name = "Random Policy"
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select a random action (ignores state).
        
        Args:
            state: Current state (ignored)
            
        Returns:
            Random action index
        """
        return np.random.randint(self.action_dim)
    
    def reset(self) -> None:
        """Reset policy (no-op for random policy)."""
        pass
    
    def __repr__(self) -> str:
        return f"RandomPolicy(action_dim={self.action_dim})"
