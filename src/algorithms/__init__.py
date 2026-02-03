"""
Algorithms package
"""

from .base_algorithm import BaseAlgorithm
from .thompson_bargaining import ThompsonSamplingBargaining
from .baselines import (
    UCB1,
    EpsilonGreedy,
    FixedStrategy,
    RandomBaseline,
    ALGORITHMS
)

__all__ = [
    'BaseAlgorithm',
    'ThompsonSamplingBargaining',
    'UCB1',
    'EpsilonGreedy',
    'FixedStrategy',
    'RandomBaseline',
    'ALGORITHMS'
]
