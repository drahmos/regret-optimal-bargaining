"""
Thompson Sampling for Bargaining (TSB)
Main algorithm implementation from TECHNICAL_SPEC.md Section 4
"""

import numpy as np
from typing import Dict, Any, List, Optional
from .base_algorithm import BaseAlgorithm


class ThompsonSamplingBargaining(BaseAlgorithm):
    """
    Thompson Sampling for Bargaining (TSB).
    
    Uses Dirichlet posterior over opponent types with Bayesian updates.
    
    Reference: TECHNICAL_SPEC.md Algorithm 1
    """
    
    def __init__(self, n_types: int, prior_alpha: Optional[np.ndarray] = None,
                 likelihood_method: str = "outcome"):
        """
        Initialize TSB algorithm.
        
        Args:
            n_types: Number of opponent types
            prior_alpha: Dirichlet prior parameters (default: uniform)
            likelihood_method: "outcome" or "trajectory" (how to compute likelihood)
        """
        super().__init__(n_types, "TSB")
        
        if prior_alpha is None:
            # Uniform prior
            self.alpha = np.ones(n_types)
        else:
            self.alpha = np.array(prior_alpha, dtype=float)
        
        self.likelihood_method = likelihood_method
        
        # Track posterior for analysis
        self.posterior_history = []
        
        # Track which types were selected
        self.type_history = []
    
    def select_type(self, episode: int) -> int:
        """
        Sample opponent type from posterior.
        
        Algorithm 1, Step 1: Sample from Dirichlet posterior
        Algorithm 1, Step 2: Sample type from categorical
        
        Args:
            episode: Current episode number
            
        Returns:
            Selected opponent type index
        """
        # Sample from Dirichlet posterior
        pi_sample = np.random.dirichlet(self.alpha)
        
        # Sample type from categorical distribution
        type_idx = np.random.choice(self.n_types, p=pi_sample)
        
        self.type_history.append(type_idx)
        
        return type_idx
    
    def update(self, outcome: Dict[str, Any]):
        """
        Update beliefs based on episode outcome.
        
        Algorithm 1, Step 4: Bayesian update
        
        Args:
            outcome: Dictionary with:
                - 'utility': Agent's utility
                - 'type_true': True opponent type
                - 'history': Negotiation history
                - 'rounds': Number of rounds
                - 'agreement_reached': Whether agreement was reached
        """
        # Compute likelihoods for each opponent type
        likelihoods = self._compute_likelihoods(outcome)
        
        # Bayesian update: alpha_k <- alpha_k + L(theta_k | outcome)
        self.alpha += likelihoods
        
        # Store posterior for analysis
        posterior = self.alpha / self.alpha.sum()
        self.posterior_history.append(posterior.copy())
        
        self.episode_count += 1
    
    def _compute_likelihoods(self, outcome: Dict[str, Any]) -> np.ndarray:
        """
        Compute likelihood L(theta_k | outcome) for each type.
        
        From TECHNICAL_SPEC.md Section 4.2.
        
        Args:
            outcome: Episode outcome
            
        Returns:
            Array of likelihoods for each opponent type
        """
        if self.likelihood_method == "outcome":
            return self._compute_likelihood_outcome(outcome)
        elif self.likelihood_method == "trajectory":
            return self._compute_likelihood_trajectory(outcome)
        else:
            raise ValueError(f"Unknown likelihood method: {self.likelihood_method}")
    
    def _compute_likelihood_outcome(self, outcome: Dict[str, Any]) -> np.ndarray:
        """
        Compute likelihood based on final outcome using generative model.
        
        From TECHNICAL_SPEC.md Assumption 4.1:
        Pr[accept | x, theta, t] = 1 / (1 + exp(-beta * (u_2(x; theta) - reservation)))
        
        This uses the actual opponent model to compute probability of observed outcome.
        """
        utility = outcome['utility']
        rounds = outcome.get('rounds', 10)
        agreement = outcome.get('agreement_reached', utility > 0.2)
        
        # Import opponent models here to avoid circular dependency
        # In practice, these would be passed to the constructor
        try:
            from ..environment.opponent_models import (
                ConcederOpponent, HardlinerOpponent, 
                TitForTatOpponent, BoulwareOpponent
            )
            opponent_classes = [
                ConcederOpponent, HardlinerOpponent,
                TitForTatOpponent, BoulwareOpponent
            ]
        except ImportError:
            # Fallback to heuristic if imports fail
            return self._compute_likelihood_heuristic(outcome)
        
        likelihoods = np.zeros(self.n_types)
        
        for k in range(self.n_types):
            # Create instance of opponent type k
            opponent = opponent_classes[k % len(opponent_classes)]()
            
            # Compute likelihood based on generative model
            # Likelihood = Pr[outcome | theta_k] 
            
            # For agreement case: likelihood based on reservation probability
            # For disagreement case: likelihood based on disagreement probability
            
            if agreement:
                # Agreement reached: compute probability of accepting at this utility/round
                # Higher likelihood if opponent's reservation was compatible
                reservation = opponent._get_reservation_utility(rounds)
                
                # Logistic acceptance model with beta=5
                beta = 5.0
                prob_accept = 1.0 / (1.0 + np.exp(-beta * (utility - reservation)))
                
                # Also consider if utility is in typical range for this opponent
                # Conceders: moderate utility, quick
                # Hardliners: lower utility or late agreement
                # etc.
                if k == 0:  # Conceder
                    type_score = 1.0 if (0.6 < utility < 0.9 and rounds < 10) else 0.5
                elif k == 1:  # Hardliner
                    type_score = 1.0 if (rounds > 15 or utility < 0.75) else 0.3
                elif k == 2:  # Tit-for-Tat
                    type_score = 1.0 if (0.5 < utility < 0.85 and 8 < rounds < 14) else 0.5
                elif k == 3:  # Boulware
                    type_score = 1.0 if (rounds > 12 or utility > 0.8) else 0.4
                else:
                    type_score = 0.5
                
                likelihoods[k] = prob_accept * type_score
            else:
                # Disagreement: opponent likely rejected or agent rejected
                # Higher likelihood for opponents with high reservation utility
                reservation_at_deadline = opponent._get_reservation_utility(20)
                prob_reject = 1.0 / (1.0 + np.exp(5.0 * (utility - reservation_at_deadline)))
                
                # Hardliners more likely to cause disagreement
                if k == 1:  # Hardliner
                    likelihoods[k] = prob_reject * 2.0  # Boost hardliner for disagreements
                else:
                    likelihoods[k] = prob_reject
        
        # Normalize to sum to 1 (soft update, as in specification)
        total = likelihoods.sum()
        if total > 0:
            likelihoods = likelihoods / total
        else:
            likelihoods = np.ones(self.n_types) / self.n_types
        
        return likelihoods
    
    def _compute_likelihood_heuristic(self, outcome: Dict[str, Any]) -> np.ndarray:
        """
        Fallback heuristic likelihood (when imports fail).
        Uses simpler but still principled approach.
        """
        utility = outcome['utility']
        rounds = outcome.get('rounds', 10)
        agreement = outcome.get('agreement_reached', utility > 0.2)
        
        likelihoods = np.zeros(self.n_types)
        
        # Use logistic model with estimated parameters
        for k in range(self.n_types):
            # Reservation utility estimate based on opponent type
            if k == 0:  # Conceder: low reservation, decreases quickly
                reservation = 0.6 * (1 - rounds / 20)
            elif k == 1:  # Hardliner: high reservation
                reservation = 0.85 if rounds < 16 else 0.3
            elif k == 2:  # TFT: moderate
                reservation = 0.7 * (1 - 0.5 * rounds / 20)
            else:  # Boulware: moderate-low, drops late
                reservation = 0.75 * (1 - 0.9 * (rounds / 20) ** 3)
            
            # Logistic acceptance probability
            beta = 5.0
            if agreement:
                prob = 1.0 / (1.0 + np.exp(-beta * (utility - reservation)))
            else:
                prob = 1.0 - 1.0 / (1.0 + np.exp(-beta * (utility - reservation)))
            
            likelihoods[k] = max(0.1, prob)
        
        # Normalize
        total = likelihoods.sum()
        if total > 0:
            likelihoods = likelihoods / total
        else:
            likelihoods = np.ones(self.n_types) / self.n_types
        
        return likelihoods
    
    def _compute_likelihood_trajectory(self, outcome: Dict[str, Any]) -> np.ndarray:
        """
        Compute likelihood based on full negotiation trajectory.
        
        More informative but requires detailed history.
        """
        history = outcome.get('history', [])
        
        if not history:
            # Fall back to outcome-based
            return self._compute_likelihood_outcome(outcome)
        
        likelihoods = np.zeros(self.n_types)
        
        # Analyze concession patterns in history
        # This is a simplified version
        
        for k in range(self.n_types):
            # Count how many rounds match expected behavior
            match_count = 0
            
            for i, round_data in enumerate(history):
                if 'opponent_offer' in round_data:
                    opp_offer = round_data['opponent_offer']
                    
                    # Expected offers based on type
                    if k == 0:  # Conceder: decreasing over time
                        if i > 0 and opp_offer.mean() < 0.5:
                            match_count += 1
                    elif k == 1:  # Hardliner: high initially
                        if i < len(history) / 2 and opp_offer.mean() > 0.6:
                            match_count += 1
                    elif k == 2:  # TFT: mirrors agent
                        if 'agent_offer' in round_data:
                            match_count += 1  # Simplified
                    elif k == 3:  # Boulware: slow concession
                        if i > len(history) * 0.7:
                            match_count += 1
            
            likelihoods[k] = match_count + 0.1  # Add small constant
        
        # Normalize
        total = likelihoods.sum()
        if total > 0:
            likelihoods = likelihoods / total
        
        return likelihoods
    
    def get_posterior(self) -> np.ndarray:
        """Get current posterior distribution."""
        return self.alpha / self.alpha.sum()
    
    def get_info(self) -> Dict[str, Any]:
        """Get algorithm state for debugging."""
        info = super().get_info()
        info.update({
            'alpha': self.alpha.copy(),
            'posterior': self.get_posterior(),
            'most_likely_type': int(np.argmax(self.alpha)),
            'posterior_entropy': -np.sum(
                self.get_posterior() * np.log(self.get_posterior() + 1e-10)
            )
        })
        return info
    
    def reset(self):
        """Reset to initial state."""
        super().reset()
        self.alpha = np.ones(self.n_types)
        self.posterior_history = []
        self.type_history = []
