"""
Baseline Algorithms
UCB1, Epsilon-Greedy, Fixed Strategy, and Random baselines
"""

import numpy as np
from typing import Dict, Any
from .base_algorithm import BaseAlgorithm


class UCB1(BaseAlgorithm):
    """
    Upper Confidence Bound algorithm for bargaining.
    
    Standard UCB1 adapted to opponent type selection.
    """
    
    def __init__(self, n_types: int, c: float = np.sqrt(2)):
        """
        Args:
            n_types: Number of opponent types
            c: Exploration parameter (default: sqrt(2))
        """
        super().__init__(n_types, "UCB1")
        self.c = c
        
        # Track counts and rewards for each type
        self.counts = np.zeros(n_types, dtype=int)
        self.rewards = np.zeros(n_types)
        self.total_counts = 0
    
    def select_type(self, episode: int) -> int:
        """
        Select type with highest UCB score.
        
        UCB_k = mu_k + c * sqrt(2 * log(t) / N_k)
        """
        # Ensure all types tried at least once
        for k in range(self.n_types):
            if self.counts[k] == 0:
                return k
        
        # Compute UCB scores
        t = self.total_counts + 1
        avg_rewards = self.rewards / (self.counts + 1e-10)
        exploration_bonus = self.c * np.sqrt(2 * np.log(t) / (self.counts + 1e-10))
        ucb_scores = avg_rewards + exploration_bonus
        
        return int(np.argmax(ucb_scores))
    
    def update(self, outcome: Dict[str, Any]):
        """Update type statistics."""
        type_idx = outcome.get('type_believed', 0)
        utility = outcome.get('utility', 0.0)
        
        self.counts[type_idx] += 1
        self.rewards[type_idx] += utility
        self.total_counts += 1
        self.episode_count += 1
    
    def reset(self):
        """Reset to initial state."""
        super().reset()
        self.counts = np.zeros(self.n_types, dtype=int)
        self.rewards = np.zeros(self.n_types)
        self.total_counts = 0


class EpsilonGreedy(BaseAlgorithm):
    """
    Epsilon-Greedy algorithm for bargaining.
    """
    
    def __init__(self, n_types: int, epsilon: float = 0.1, 
                 decay: bool = True):
        """
        Args:
            n_types: Number of opponent types
            epsilon: Exploration probability
            decay: Whether to decay epsilon over time
        """
        super().__init__(n_types, "EpsilonGreedy")
        self.epsilon = epsilon
        self.decay = decay
        
        # Track average rewards
        self.counts = np.zeros(n_types, dtype=int)
        self.rewards = np.zeros(n_types)
    
    def select_type(self, episode: int) -> int:
        """
        Select type: explore with prob epsilon, exploit otherwise.
        """
        # Compute current epsilon
        if self.decay:
            current_epsilon = min(1.0, self.epsilon * self.n_types / (episode + 1))
        else:
            current_epsilon = self.epsilon
        
        # Explore
        if np.random.random() < current_epsilon:
            return np.random.randint(self.n_types)
        
        # Exploit: select best average reward
        avg_rewards = self.rewards / (self.counts + 1e-10)
        return int(np.argmax(avg_rewards))
    
    def update(self, outcome: Dict[str, Any]):
        """Update statistics."""
        type_idx = outcome.get('type_believed', 0)
        utility = outcome.get('utility', 0.0)
        
        self.counts[type_idx] += 1
        self.rewards[type_idx] += utility
        self.episode_count += 1
    
    def reset(self):
        """Reset to initial state."""
        super().reset()
        self.counts = np.zeros(self.n_types, dtype=int)
        self.rewards = np.zeros(self.n_types)


class FixedStrategy(BaseAlgorithm):
    """
    Fixed strategy - always assumes most common opponent type.
    
    Baseline with no learning.
    """
    
    def __init__(self, n_types: int, fixed_type: int = 0):
        """
        Args:
            n_types: Number of opponent types
            fixed_type: Type to always assume (default: 0)
        """
        super().__init__(n_types, "FixedStrategy")
        self.fixed_type = fixed_type
    
    def select_type(self, episode: int) -> int:
        """Always return fixed type."""
        return self.fixed_type
    
    def update(self, outcome: Dict[str, Any]):
        """No update - strategy is fixed."""
        self.episode_count += 1


class RandomBaseline(BaseAlgorithm):
    """
    Random baseline - selects opponent type uniformly at random.
    
    Lower bound for performance comparison.
    """
    
    def __init__(self, n_types: int):
        """
        Args:
            n_types: Number of opponent types
        """
        super().__init__(n_types, "Random")
    
    def select_type(self, episode: int) -> int:
        """Select random type uniformly."""
        return np.random.randint(self.n_types)
    
    def update(self, outcome: Dict[str, Any]):
        """No learning."""
        self.episode_count += 1


# Dictionary for easy access
ALGORITHMS = {
    'TSB': ThompsonSamplingBargaining,
    'UCB1': UCB1,
    'EpsilonGreedy': EpsilonGreedy,
    'FixedStrategy': FixedStrategy,
    'Random': RandomBaseline
}
