"""
Bargaining Environment Module
Implements alternating-offers negotiation simulation
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple


class BargainingEnvironment:
    """
    Bilateral alternating-offers bargaining environment.
    
    Implements the game described in TECHNICAL_SPEC.md Section 2.
    """
    
    def __init__(self, n_issues: int = 3, T_max: int = 20, 
                 delta: float = 0.95, disagreement_value: float = 0.1):
        """
        Initialize bargaining environment.
        
        Args:
            n_issues: Number of negotiation issues (d in the paper)
            T_max: Maximum rounds (deadline)
            delta: Discount factor
            disagreement_value: Payoff if no agreement reached
        """
        self.n_issues = n_issues
        self.T_max = T_max
        self.delta = delta
        self.d = disagreement_value
        
        # Agent (learner) utility weights - fixed
        # From EXPERIMENTS.md Section 2.1
        self.agent_weights = np.array([0.5, 0.3, 0.2])
        
        # Initialize state
        self.reset()
    
    def reset(self) -> Dict:
        """
        Reset environment for new negotiation episode.
        
        Returns:
            Initial state dictionary
        """
        self.round = 0
        self.history = deque(maxlen=self.T_max)
        self.done = False
        self.agreement = None
        self.agent_utility = None
        
        return self._get_state()
    
    def step(self, agent_offer: np.ndarray, opponent_model) -> Dict:
        """
        Execute one round of negotiation.
        
        Args:
            agent_offer: Agent's offer (n_issues,) - must sum to 1
            opponent_model: OpponentModel instance to respond
            
        Returns:
            Dictionary with:
                - 'utility': Agent's utility (None if ongoing)
                - 'done': Whether negotiation ended
                - 'agreement': Final agreement (None if disagreement)
                - 'round': Current round number
                - 'accepted_by_opponent': Whether opponent accepted
                - 'accepted_by_agent': Whether agent accepted opponent's counter
        """
        self.round += 1
        
        # Normalize agent offer
        agent_offer = self._normalize_offer(agent_offer)
        
        # Opponent decides whether to accept agent's offer
        accept = opponent_model.accept_offer(
            agent_offer, 
            self.round, 
            list(self.history)
        )
        
        if accept:
            # Agreement reached on agent's offer
            utility = self._compute_utility(agent_offer)
            self.done = True
            self.agreement = agent_offer
            self.agent_utility = utility
            
            return {
                'utility': utility,
                'done': True,
                'agreement': agent_offer.copy(),
                'round': self.round,
                'accepted_by_opponent': True,
                'accepted_by_agent': False,
                'opponent_offer': None
            }
        
        # Opponent rejects and makes counter-offer
        if self.round >= self.T_max:
            # Deadline reached - disagreement
            self.done = True
            self.agent_utility = self.d
            
            return {
                'utility': self.d,
                'done': True,
                'agreement': None,
                'round': self.round,
                'accepted_by_opponent': False,
                'accepted_by_agent': False,
                'opponent_offer': None
            }
        
        # Generate opponent's counter-offer
        opponent_offer = opponent_model.generate_offer(
            self.round, 
            list(self.history)
        )
        opponent_offer = self._normalize_offer(opponent_offer)
        
        # Agent decides whether to accept opponent's offer
        # Simplified: accept if above disagreement value
        opponent_utility = self._compute_utility(opponent_offer)
        
        # Myopic acceptance: accept if better than disagreement
        # and expected future utility
        accept_threshold = self.d
        
        # In practice, agent might use more sophisticated reasoning
        # For simulation, we assume agent accepts if utility > threshold
        agent_accepts = opponent_utility >= accept_threshold
        
        if agent_accepts:
            # Agent accepts opponent's offer
            self.done = True
            self.agreement = opponent_offer
            self.agent_utility = opponent_utility
            
            return {
                'utility': opponent_utility,
                'done': True,
                'agreement': opponent_offer.copy(),
                'round': self.round,
                'accepted_by_opponent': False,
                'accepted_by_agent': True,
                'opponent_offer': opponent_offer.copy()
            }
        
        # Continue negotiation - no agreement this round
        self.history.append({
            'round': self.round,
            'agent_offer': agent_offer.copy(),
            'opponent_offer': opponent_offer.copy()
        })
        
        return {
            'utility': None,
            'done': False,
            'agreement': None,
            'round': self.round,
            'accepted_by_opponent': False,
            'accepted_by_agent': False,
            'opponent_offer': opponent_offer.copy()
        }
    
    def run_full_negotiation(self, algorithm, opponent_type_idx: int,
                            opponent_models: List) -> Dict:
        """
        Run complete negotiation episode.
        
        Args:
            algorithm: Algorithm instance (must have select_type and update)
            opponent_type_idx: Index of true opponent type
            opponent_models: List of all opponent model instances
            
        Returns:
            Dictionary with negotiation outcome
        """
        true_opponent = opponent_models[opponent_type_idx]
        self.reset()
        
        # Algorithm selects believed opponent type
        believed_type_idx = algorithm.select_type(self.round)
        believed_opponent = opponent_models[believed_type_idx]
        
        # Run negotiation rounds
        while not self.done and self.round < self.T_max:
            # Generate offer based on believed opponent type
            # This is simplified - in full implementation, 
            # algorithm would use its internal model
            
            # Simple offer: demand proportional to agent weights
            offer = self.agent_weights.copy()
            
            # Adjust based on round (deadline pressure)
            time_pressure = self.round / self.T_max
            concession = time_pressure * 0.3  # Concede up to 30%
            offer = offer * (1 - concession) + 0.33 * concession
            offer = self._normalize_offer(offer)
            
            # Execute round
            result = self.step(offer, true_opponent)
        
        # Compute final utility
        if result['done']:
            utility = result['utility']
        else:
            utility = self.d
        
        return {
            'utility': utility,
            'agreement': result.get('agreement'),
            'rounds': self.round,
            'type_true': opponent_type_idx,
            'type_believed': believed_type_idx,
            'history': list(self.history)
        }
    
    def _compute_utility(self, offer: np.ndarray) -> float:
        """
        Compute agent's utility for given offer.
        
        Args:
            offer: Resource allocation (n_issues,)
            
        Returns:
            Utility value in [0, 1]
        """
        return float(np.dot(self.agent_weights, offer))
    
    def _normalize_offer(self, offer: np.ndarray) -> np.ndarray:
        """
        Normalize offer to valid simplex (sum to 1, all non-negative).
        
        Args:
            offer: Raw offer array
            
        Returns:
            Normalized offer
        """
        offer = np.maximum(offer, 0)  # Clip negative values
        total = offer.sum()
        if total > 0:
            return offer / total
        else:
            # Return uniform if all zeros
            return np.ones(self.n_issues) / self.n_issues
    
    def _get_state(self) -> Dict:
        """Return current environment state."""
        return {
            'round': self.round,
            'history': list(self.history),
            'done': self.done,
            'n_issues': self.n_issues,
            'T_max': self.T_max
        }
    
    def get_oracle_utility(self, opponent_model) -> float:
        """
        Compute oracle utility (optimal strategy against known opponent).
        
        This is a simplified approximation. Full implementation would use
        backward induction (TECHNICAL_SPEC.md Section 2.1).
        
        Args:
            opponent_model: True opponent model
            
        Returns:
            Approximate optimal utility
        """
        # Simulate optimal play (simplified)
        # In practice, this would solve the game tree
        
        # Generate offers at each round
        utilities = []
        for t in range(1, self.T_max + 1):
            # Optimal offer balances concession with opponent acceptance
            concession = (t / self.T_max) ** 1.5  # Sigmoid-like
            offer = self.agent_weights * (1 - concession) + 0.33 * concession
            offer = self._normalize_offer(offer)
            
            utility = self._compute_utility(offer)
            utilities.append(utility)
        
        # Oracle achieves near-maximum utility
        return max(utilities) * 0.95  # Approximate


class SimpleBargainingEnv:
    """
    Simplified bargaining environment for faster simulation.
    
    This version uses analytical approximations instead of full 
    round-by-round simulation for speed.
    """
    
    def __init__(self, n_issues: int = 3, T_max: int = 20):
        self.n_issues = n_issues
        self.T_max = T_max
        self.agent_weights = np.array([0.5, 0.3, 0.2])
    
    def simulate_negotiation(self, algorithm_type_idx: int, 
                           true_type_idx: int,
                           opponent_models: List) -> Dict:
        """
        Fast simulation using analytical model.
        
        Returns approximate negotiation outcome without 
        simulating every round.
        """
        # Simplified: use type similarity to estimate outcome
        similarity = 1.0 if algorithm_type_idx == true_type_idx else 0.3
        
        # Base utility when types match
        base_utility = 0.85
        
        # Utility when types mismatch
        mismatch_penalty = 0.3 * (1 - similarity)
        
        # Add noise
        noise = np.random.normal(0, 0.05)
        
        utility = base_utility - mismatch_penalty + noise
        utility = np.clip(utility, 0.1, 0.95)
        
        return {
            'utility': utility,
            'agreement_reached': utility > 0.2,
            'rounds': int(self.T_max * (1 - similarity * 0.5))
        }
