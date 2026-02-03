"""
Base Algorithm Module
Abstract base class for all learning algorithms
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAlgorithm(ABC):
    """
    Abstract base class for all bargaining learning algorithms.
    
    All algorithms must implement:
    - select_type(): Choose which opponent type to assume
    - update(): Update internal state given outcome
    """
    
    def __init__(self, n_types: int, name: str = "base"):
        """
        Initialize algorithm.
        
        Args:
            n_types: Number of possible opponent types
            name: Algorithm identifier
        """
        self.n_types = n_types
        self.name = name
        self.episode_count = 0
    
    @abstractmethod
    def select_type(self, episode: int) -> int:
        """
        Select which opponent type to assume for this episode.
        
        Args:
            episode: Current episode number
            
        Returns:
            Index of selected opponent type (0 to n_types-1)
        """
        pass
    
    @abstractmethod
    def update(self, outcome: Dict[str, Any]):
        """
        Update algorithm state given episode outcome.
        
        Args:
            outcome: Dictionary containing:
                - 'utility': Agent's utility in this episode
                - 'type_true': True opponent type index
                - 'history': Negotiation history (optional)
                - 'rounds': Number of rounds taken (optional)
        """
        pass
    
    def reset(self):
        """Reset algorithm to initial state."""
        self.episode_count = 0
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get algorithm state information for debugging.
        
        Returns:
            Dictionary with algorithm-specific info
        """
        return {
            'name': self.name,
            'n_types': self.n_types,
            'episodes': self.episode_count
        }
