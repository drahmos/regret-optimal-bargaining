"""
Main Experiment Runner
Runs complete experimental pipeline
"""

import numpy as np
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import pickle
import os
from datetime import datetime

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.bargaining_env import BargainingEnvironment
from environment.opponent_models import create_opponent_models, compute_bargaining_structure
from algorithms.thompson_bargaining import ThompsonSamplingBargaining
from algorithms.baselines import UCB1, EpsilonGreedy, FixedStrategy, RandomBaseline
from utils.metrics import compute_cumulative_regret, aggregate_results, create_results_dataframe
from utils.visualization import create_all_plots
from utils.statistical_tests import compare_all_algorithms, print_comparison_table


class ExperimentRunner:
    """
    Main class for running experiments.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize experiment runner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results = {}
        
        # Create opponent models
        self.opponent_models = create_opponent_models(
            n_issues=config.get('n_issues', 3),
            T_max=config.get('T_max', 20)
        )
        
        # Compute bargaining structure parameter B
        self.B = compute_bargaining_structure(
            self.opponent_models,
            n_issues=config.get('n_issues', 3),
            T_max=config.get('T_max', 20)
        )
        
        print(f"Experiment Configuration:")
        print(f"  Number of issues: {config.get('n_issues', 3)}")
        print(f"  Deadline (T_max): {config.get('T_max', 20)}")
        print(f"  Horizon (T): {config.get('T', 1000)}")
        print(f"  Bargaining structure B: {self.B:.2f}")
        print(f"  Seeds: {config.get('n_seeds', 10)}")
    
    def run_single_experiment(self, algorithm_class, seed: int) -> Dict[str, Any]:
        """
        Run single experiment with given algorithm and seed.
        
        Args:
            algorithm_class: Algorithm class to instantiate
            seed: Random seed
            
        Returns:
            Results dictionary
        """
        # Set random seed
        np.random.seed(seed)
        
        # Initialize environment and algorithm
        env = BargainingEnvironment(
            n_issues=self.config.get('n_issues', 3),
            T_max=self.config.get('T_max', 20)
        )
        
        algorithm = algorithm_class(n_types=len(self.opponent_models))
        
        # Get opponent distribution
        opponent_dist = self.config.get('opponent_dist', [0.25, 0.25, 0.25, 0.25])
        
        # Run T episodes
        T = self.config.get('T', 1000)
        
        utilities = []
        oracle_utilities = []
        agreements = []
        rounds_list = []
        types_selected = []
        types_true = []
        
        for episode in range(T):
            # Sample true opponent type
            true_type_idx = np.random.choice(len(opponent_dist), p=opponent_dist)
            true_opponent = self.opponent_models[true_type_idx]
            
            # Algorithm selects believed type
            believed_type_idx = algorithm.select_type(episode)
            
            # Run negotiation
            env.reset()
            episode_utility = 0
            agreement_reached = False
            rounds_taken = 0
            
            # Run actual negotiation using BargainingEnvironment
            # This implements the alternating-offers protocol from the paper
            believed_opponent = self.opponent_models[believed_type_idx]
            
            # Reset environment for this negotiation
            env.reset()
            
            # Run round-by-round negotiation
            max_rounds = env.T_max
            agreement_reached = False
            episode_utility = env.d  # Default: disagreement payoff
            rounds_taken = max_rounds
            
            for round_num in range(1, max_rounds + 1):
                # Generate offer based on believed opponent type
                # Use concession strategy: start high, concede toward deadline
                time_pressure = round_num / max_rounds
                concession = time_pressure ** 1.5  # Non-linear concession
                
                # Compute offer: weighted average of agent preferences and fair split
                agent_weights = env.agent_weights
                fair_split = np.ones(len(agent_weights)) / len(agent_weights)
                offer = agent_weights * (1 - concession) + fair_split * concession
                offer = offer / offer.sum()  # Normalize to simplex
                
                # Execute negotiation round
                result = env.step(offer, true_opponent)
                
                if result['done']:
                    agreement_reached = result.get('accepted_by_opponent', False) or result.get('accepted_by_agent', False)
                    episode_utility = result['utility'] if result['utility'] is not None else env.d
                    rounds_taken = round_num
                    break
            
            # Compute oracle utility (optimal strategy against known type)
            # This should depend on the actual opponent type
            oracle_util = env.get_oracle_utility(true_opponent)
            
            # Store results
            utilities.append(episode_utility)
            oracle_utilities.append(oracle_util)
            agreements.append(agreement_reached)
            rounds_list.append(rounds_taken)
            types_selected.append(believed_type_idx)
            types_true.append(true_type_idx)
            
            # Update algorithm
            outcome = {
                'utility': episode_utility,
                'type_true': true_type_idx,
                'type_believed': believed_type_idx,
                'agreement_reached': agreement_reached,
                'rounds': rounds_taken
            }
            algorithm.update(outcome)
        
        # Compute cumulative regret
        cumulative_regret = compute_cumulative_regret(utilities, oracle_utilities)
        
        return {
            'utilities': utilities,
            'oracle_utilities': oracle_utilities,
            'cumulative_regret': cumulative_regret.tolist(),
            'agreements': agreements,
            'rounds': rounds_list,
            'types_selected': types_selected,
            'types_true': types_true,
            'final_regret': float(cumulative_regret[-1]),
            'algorithm': algorithm.name,
            'seed': seed
        }
    
    def run_algorithm_suite(self, algorithm_classes: List, 
                           n_seeds: int = 10) -> Dict[str, List[Dict]]:
        """
        Run full suite of algorithms with multiple seeds.
        
        Args:
            algorithm_classes: List of algorithm classes
            n_seeds: Number of random seeds
            
        Returns:
            Dictionary of results
        """
        results = {}
        
        for alg_class in algorithm_classes:
            alg_name = alg_class.__name__
            print(f"\nRunning {alg_name}...")
            
            alg_results = []
            for seed in tqdm(range(n_seeds), desc=f"{alg_name} seeds"):
                result = self.run_single_experiment(alg_class, seed)
                alg_results.append(result)
            
            results[alg_name] = alg_results
        
        self.results = results
        return results
    
    def save_results(self, output_dir: str = "results"):
        """
        Save results to disk.
        
        Args:
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.pkl"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'results': self.results,
                'config': self.config,
                'B': self.B
            }, f)
        
        print(f"\nResults saved to {filepath}")
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze and print results.
        
        Returns:
            Analysis summary
        """
        if not self.results:
            print("No results to analyze. Run experiments first.")
            return {}
        
        print("\n" + "="*80)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        
        # Aggregate results
        summary = {}
        for alg_name, results_list in self.results.items():
            final_regrets = [r['final_regret'] for r in results_list]
            
            summary[alg_name] = {
                'mean_regret': np.mean(final_regrets),
                'std_regret': np.std(final_regrets),
                'se_regret': np.std(final_regrets) / np.sqrt(len(final_regrets)),
                'min_regret': np.min(final_regrets),
                'max_regret': np.max(final_regrets)
            }
        
        # Print comparison table
        print("\nFinal Cumulative Regret Comparison:")
        print("-" * 80)
        print(f"{'Algorithm':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 80)
        
        for alg_name, stats in sorted(summary.items(), 
                                     key=lambda x: x[1]['mean_regret']):
            print(f"{alg_name:<20} "
                  f"{stats['mean_regret']:>11.2f} "
                  f"{stats['std_regret']:>11.2f} "
                  f"{stats['min_regret']:>11.2f} "
                  f"{stats['max_regret']:>11.2f}")
        
        print("-" * 80)
        
        # Statistical tests
        if len(self.results) > 1:
            print("\nStatistical Tests (vs UCB1 baseline):")
            print("-" * 80)
            
            # Extract final regrets for each algorithm
            regrets_dict = {}
            for alg_name, results_list in self.results.items():
                regrets_dict[alg_name] = [r['final_regret'] for r in results_list]
            
            if 'UCB1' in regrets_dict:
                comparisons = compare_all_algorithms(regrets_dict, 'UCB1')
                print_comparison_table(comparisons)
        
        return summary


def run_main_experiments(T: int = 1000, n_seeds: int = 10, 
                        output_dir: str = "results"):
    """
    Run main experiment suite (Experiment 1 from EXPERIMENTS.md).
    
    Args:
        T: Number of episodes
        n_seeds: Number of random seeds
        output_dir: Output directory
    """
    print("="*80)
    print("REGRET-OPTIMAL BARGAINING - MAIN EXPERIMENTS")
    print("="*80)
    
    # Configuration
    config = {
        'n_issues': 3,
        'T_max': 20,
        'T': T,
        'n_seeds': n_seeds,
        'opponent_dist': [0.25, 0.25, 0.25, 0.25]  # Uniform
    }
    
    # Initialize runner
    runner = ExperimentRunner(config)
    
    # Define algorithms
    algorithms = [
        ThompsonSamplingBargaining,
        UCB1,
        EpsilonGreedy,
        FixedStrategy,
        RandomBaseline
    ]
    
    # Run experiments
    results = runner.run_algorithm_suite(algorithms, n_seeds)
    
    # Analyze results
    summary = runner.analyze_results()
    
    # Save results
    runner.save_results(output_dir)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_dir = os.path.join(output_dir, "plots")
    try:
        create_all_plots(results, plot_dir)
    except Exception as e:
        print(f"Note: Could not generate plots (error: {e})")
    
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    
    return runner, results, summary


def quick_test():
    """
    Quick test to verify everything works (5 minutes).
    """
    print("\n" + "="*80)
    print("QUICK TEST (T=100, 2 seeds)")
    print("="*80)
    
    runner, results, summary = run_main_experiments(
        T=100,
        n_seeds=2,
        output_dir="results/quick_test"
    )
    
    print("\n✓ Quick test passed!")
    
    return runner, results, summary


if __name__ == "__main__":
    # Run full experiments
    runner, results, summary = run_main_experiments(
        T=1000,
        n_seeds=10,
        output_dir="results"
    )
