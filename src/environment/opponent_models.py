"""
Opponent Models Module
Implements 4 opponent types from TECHNICAL_SPEC.md Section 2.2
"""

import numpy as np
from typing import List, Dict
from abc import ABC, abstractmethod


class OpponentModel(ABC):
    """
    Abstract base class for opponent types.
    
    Each opponent has:
    - Utility function u_2(x; theta) = w_theta^T x
    - Strategy sigma(theta) mapping history to offers
    """
    
    def __init__(self, utility_weights: List[float], name: str = "generic"):
        """
        Initialize opponent model.
        
        Args:
            utility_weights: Weights for opponent's utility function
            name: Opponent type identifier
        """
        self.weights = np.array(utility_weights)
        self.name = name
        self.n_issues = len(utility_weights)
    
    def compute_utility(self, offer: np.ndarray) -> float:
        """
        Compute opponent's utility for given offer.
        
        Args:
            offer: Resource allocation (n_issues,)
            
        Returns:
            Utility value
        """
        return float(np.dot(self.weights, offer))
    
    @abstractmethod
    def generate_offer(self, round: int, history: List[Dict]) -> np.ndarray:
        """
        Generate opponent's offer at given round.
        
        Args:
            round: Current round number (1-indexed)
            history: List of previous rounds' offers
            
        Returns:
            Offer as numpy array
        """
        pass
    
    def accept_offer(self, offer: np.ndarray, round: int, 
                    history: List[Dict], beta: float = 5.0) -> bool:
        """
        Decide whether to accept agent's offer.
        
        Uses logistic acceptance model (TECHNICAL_SPEC.md Assumption 4.1):
        Pr[accept] = 1 / (1 + exp(-beta * (u_2(offer) - reservation)))
        
        Args:
            offer: Agent's offer
            round: Current round
            history: Negotiation history
            beta: Rationality parameter
            
        Returns:
            True if accepts, False otherwise
        """
        utility = self.compute_utility(offer)
        reservation = self._get_reservation_utility(round)
        
        # Logistic acceptance probability
        prob_accept = 1.0 / (1.0 + np.exp(-beta * (utility - reservation)))
        
        return np.random.random() < prob_accept
    
    @abstractmethod
    def _get_reservation_utility(self, round: int) -> float:
        """
        Get reservation utility at given round.
        
        Reservation utility typically decreases over time as deadline
        approaches (concession behavior).
        
        Args:
            round: Current round
            
        Returns:
            Reservation utility threshold
        """
        pass
    
    def _normalize_offer(self, offer: np.ndarray) -> np.ndarray:
        """Normalize offer to sum to 1."""
        offer = np.maximum(offer, 0)
        total = offer.sum()
        if total > 0:
            return offer / total
        return np.ones(self.n_issues) / self.n_issues


class ConcederOpponent(OpponentModel):
    """
    Conceder opponent type.
    
    Concedes linearly over time with parameter alpha.
    From EXPERIMENTS.md Table 1.
    """
    
    def __init__(self, utility_weights: List[float] = [0.2, 0.5, 0.3],
                 alpha: float = 1.5, T_max: int = 20):
        super().__init__(utility_weights, "conceder")
        self.alpha = alpha
        self.T_max = T_max
    
    def generate_offer(self, round: int, history: List[Dict]) -> np.ndarray:
        """
        Generate offer with linear concession.
        
        From TECHNICAL_SPEC.md Section 1.2:
        x_t = x_max * (1 - t/T_max)^alpha
        
        where x_max is the offer maximizing opponent's utility (their weights).
        """
        t_norm = round / self.T_max
        
        # Maximum demand offer (proportional to weights)
        x_max = self.weights / self.weights.sum()
        
        # Apply concession formula directly (not interpolation)
        # As t increases, concession_factor decreases
        concession_factor = (1 - t_norm) ** self.alpha
        offer = x_max * concession_factor
        
        return self._normalize_offer(offer)
    
    def _get_reservation_utility(self, round: int) -> float:
        """Reservation decreases as (1 - t/T_max)^alpha."""
        t_norm = round / self.T_max
        max_util = self.compute_utility(self.weights / self.weights.sum())
        return max_util * (1 - t_norm) ** self.alpha


class HardlinerOpponent(OpponentModel):
    """
    Hardliner opponent type.
    
    Maintains high demand until near deadline, then sharp concession.
    From EXPERIMENTS.md Table 1.
    """
    
    def __init__(self, utility_weights: List[float] = [0.3, 0.2, 0.5],
                 beta: float = 5.0, T_max: int = 20, threshold: float = 0.8):
        super().__init__(utility_weights, "hardliner")
        self.beta = beta
        self.T_max = T_max
        self.threshold = threshold  # Start conceding after this fraction
    
    def generate_offer(self, round: int, history: List[Dict]) -> np.ndarray:
        """
        Generate offer: hold firm until threshold, then sharp concession.
        
        From TECHNICAL_SPEC.md Section 1.2:
        x_t = x_max                      if t < 0.8*T_max
        x_t = x_max * exp(-beta*(t - 0.8*T_max))  if t >= 0.8*T_max
        
        Note: Uses ABSOLUTE time difference, not normalized.
        """
        t_norm = round / self.T_max
        
        if t_norm < self.threshold:
            # High demand phase
            offer = self.weights / self.weights.sum()
        else:
            # Sharp concession phase
            # Use ABSOLUTE time: (t - 0.8*T_max), not (t_norm - 0.8)
            time_after_abs = round - self.threshold * self.T_max
            concession = np.exp(-self.beta * time_after_abs)
            
            # Direct formula from spec (not interpolation)
            x_max = self.weights / self.weights.sum()
            offer = x_max * concession
        
        return self._normalize_offer(offer)
    
    def _get_reservation_utility(self, round: int) -> float:
        """High reservation until threshold, then drops."""
        t_norm = round / self.T_max
        max_util = self.compute_utility(self.weights / self.weights.sum())
        
        if t_norm < self.threshold:
            # Very high reservation (90% of max)
            return max_util * 0.9
        else:
            # Sharp drop after threshold - use ABSOLUTE time
            time_after_abs = round - self.threshold * self.T_max
            return max_util * np.exp(-self.beta * time_after_abs)


class TitForTatOpponent(OpponentModel):
    """
    Tit-for-Tat opponent type.
    
    Mirrors agent's concession pattern.
    From EXPERIMENTS.md Table 1.
    """
    
    def __init__(self, utility_weights: List[float] = [0.4, 0.3, 0.3],
                 gamma: float = 0.9, T_max: int = 20):
        super().__init__(utility_weights, "tit_for_tat")
        self.gamma = gamma  # How much to mirror
        self.T_max = T_max
    
    def generate_offer(self, round: int, history: List[Dict]) -> np.ndarray:
        """
        Generate offer by mirroring agent's recent concessions.
        """
        if len(history) < 2:
            # Initial offer: moderate demand
            return self.weights / self.weights.sum()
        
        # Get agent's last two offers
        agent_offer_t = history[-1]['agent_offer']
        agent_offer_t_prev = history[-2]['agent_offer']
        
        # Compute agent's concession
        # If agent_offer_t < agent_offer_t_prev, agent conceded (negative concession)
        agent_concession = agent_offer_t - agent_offer_t_prev
        
        # Get our last offer
        if 'opponent_offer' in history[-1]:
            our_last = history[-1]['opponent_offer']
        else:
            our_last = self.weights / self.weights.sum()
        
        # Mirror agent's concession with factor gamma
        # If agent conceded (negative concession), we also concede (reduce our demand)
        # Formula: x_t = x_{t-1} + gamma * (agent's concession)
        our_offer = our_last + self.gamma * agent_concession
        
        return self._normalize_offer(our_offer)
    
    def _get_reservation_utility(self, round: int) -> float:
        """Reservation follows simple decreasing pattern."""
        t_norm = round / self.T_max
        max_util = self.compute_utility(self.weights / self.weights.sum())
        return max_util * (1 - 0.7 * t_norm)  # Linear decrease


class BoulwareOpponent(OpponentModel):
    """
    Boulware opponent type.
    
    Concedes slowly at first, then sharply near deadline.
    From EXPERIMENTS.md Table 1.
    """
    
    def __init__(self, utility_weights: List[float] = [0.25, 0.35, 0.4],
                 T_max: int = 20):
        super().__init__(utility_weights, "boulware")
        self.T_max = T_max
    
    def generate_offer(self, round: int, history: List[Dict]) -> np.ndarray:
        """
        Generate offer with cubic concession function.
        
        From TECHNICAL_SPEC.md Section 1.2:
        x_t = x_max * (1 - (t/T_max)^3)
        
        Cubic concession: starts slow, accelerates near deadline.
        """
        t_norm = round / self.T_max
        
        # Maximum demand offer
        x_max = self.weights / self.weights.sum()
        
        # Cubic concession factor (direct formula, not interpolation)
        concession = 1 - (t_norm ** 3)
        offer = x_max * concession
        
        return self._normalize_offer(offer)
    
    def _get_reservation_utility(self, round: int) -> float:
        """Reservation follows cubic curve."""
        t_norm = round / self.T_max
        max_util = self.compute_utility(self.weights / self.weights.sum())
        # Cubic decrease: slow at first, fast near end
        return max_util * (1 - 0.9 * (t_norm ** 3))


def create_opponent_models(n_issues: int = 3, T_max: int = 20) -> List[OpponentModel]:
    """
    Factory function to create all 4 opponent types.
    
    Args:
        n_issues: Number of issues
        T_max: Deadline
        
    Returns:
        List of opponent model instances
    """
    return [
        ConcederOpponent(T_max=T_max),
        HardlinerOpponent(T_max=T_max),
        TitForTatOpponent(T_max=T_max),
        BoulwareOpponent(T_max=T_max)
    ]


# Utility function helpers
def compute_bargaining_structure(opponent_models: List[OpponentModel],
                                n_issues: int, T_max: int) -> float:
    """
    Compute bargaining structure parameter B.
    
    B = (d * K) / E[Var_t[u_2(x_t; theta)]]
    
    From TECHNICAL_SPEC.md Definition 2.3.
    
    Args:
        opponent_models: List of opponent models
        n_issues: Number of issues (d)
        T_max: Deadline
        
    Returns:
        Structure parameter B
    """
    K = len(opponent_models)
    
    # Compute temporal variance for each opponent
    temporal_variances = []
    
    for opponent in opponent_models:
        utilities = []
        for t in range(1, T_max + 1):
            offer = opponent.generate_offer(t, [])
            utility = opponent.compute_utility(offer)
            utilities.append(utility)
        
        variance = np.var(utilities)
        temporal_variances.append(variance)
    
    avg_variance = np.mean(temporal_variances)
    
    # Compute B
    if avg_variance > 0:
        B = (n_issues * K) / avg_variance
    else:
        B = float('inf')  # No variance means infinite structure
    
    return B
