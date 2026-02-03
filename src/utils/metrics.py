"""
Metrics and Utilities Module
Evaluation metrics and helper functions
"""

import numpy as np
from typing import List, Dict, Any
import pandas as pd


def compute_cumulative_regret(utilities: List[float], 
                              oracle_utilities: List[float]) -> np.ndarray:
    """
    Compute cumulative regret over time.
    
    R_T = sum_{t=1}^T [V*(theta_t) - u_t]
    
    Args:
        utilities: Achieved utilities per episode
        oracle_utilities: Oracle optimal utilities per episode
        
    Returns:
        Cumulative regret array
    """
    utilities = np.array(utilities)
    oracle_utilities = np.array(oracle_utilities)
    
    # Per-episode regret
    per_episode_regret = oracle_utilities - utilities
    
    # Cumulative regret
    cumulative_regret = np.cumsum(per_episode_regret)
    
    return cumulative_regret


def compute_regret_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute comprehensive regret metrics.
    
    Args:
        results: Dictionary with experiment results
        
    Returns:
        Dictionary with metrics
    """
    utilities = results.get('utilities', [])
    oracle_utilities = results.get('oracle_utilities', [])
    
    if not utilities:
        return {}
    
    # Cumulative regret
    cum_regret = compute_cumulative_regret(utilities, oracle_utilities)
    
    metrics = {
        'total_regret': float(cum_regret[-1]),
        'avg_regret_per_episode': float(cum_regret[-1] / len(utilities)),
        'final_regret': float(cum_regret[-1]),
        'max_regret': float(np.max(cum_regret)),
        'min_utility': float(np.min(utilities)),
        'max_utility': float(np.max(utilities)),
        'mean_utility': float(np.mean(utilities)),
        'std_utility': float(np.std(utilities)),
    }
    
    return metrics


def compute_agreement_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute agreement-related metrics.
    
    Args:
        results: Dictionary with experiment results
        
    Returns:
        Dictionary with agreement metrics
    """
    agreements = results.get('agreements', [])
    rounds = results.get('rounds', [])
    utilities = results.get('utilities', [])
    
    if not agreements:
        return {}
    
    agreements_arr = np.array(agreements)
    
    metrics = {
        'agreement_rate': float(np.mean(agreements_arr)),
        'num_agreements': int(np.sum(agreements_arr)),
        'num_disagreements': int(np.sum(~agreements_arr)),
    }
    
    # Rounds to agreement (when agreement reached)
    if rounds and np.any(agreements_arr):
        rounds_arr = np.array(rounds)
        agreement_rounds = rounds_arr[agreements_arr]
        metrics['mean_rounds_to_agreement'] = float(np.mean(agreement_rounds))
        metrics['median_rounds_to_agreement'] = float(np.median(agreement_rounds))
        metrics['std_rounds_to_agreement'] = float(np.std(agreement_rounds))
    
    # Utilities by agreement status
    if utilities:
        utilities_arr = np.array(utilities)
        metrics['mean_utility_when_agreement'] = float(
            np.mean(utilities_arr[agreements_arr])
        ) if np.any(agreements_arr) else 0.0
        metrics['mean_utility_when_disagreement'] = float(
            np.mean(utilities_arr[~agreements_arr])
        ) if np.any(~agreements_arr) else 0.0
    
    return metrics


def compute_type_identification_accuracy(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute accuracy of opponent type identification.
    
    Args:
        results: Dictionary with type selections
        
    Returns:
        Dictionary with accuracy metrics
    """
    types_selected = results.get('types_selected', [])
    types_true = results.get('types_true', [])
    
    if not types_selected or not types_true:
        return {}
    
    types_selected = np.array(types_selected)
    types_true = np.array(types_true)
    
    correct = types_selected == types_true
    
    metrics = {
        'type_accuracy': float(np.mean(correct)),
        'num_correct': int(np.sum(correct)),
        'num_incorrect': int(np.sum(~correct)),
    }
    
    # Per-type accuracy
    unique_types = np.unique(types_true)
    for t in unique_types:
        mask = types_true == t
        if np.any(mask):
            acc = np.mean(correct[mask])
            metrics[f'accuracy_type_{t}'] = float(acc)
    
    return metrics


def aggregate_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results across multiple seeds.
    
    Args:
        results_list: List of result dictionaries (one per seed)
        
    Returns:
        Aggregated results with mean and std
    """
    if not results_list:
        return {}
    
    # Collect metrics across seeds
    all_metrics = {
        'total_regret': [],
        'avg_regret_per_episode': [],
        'agreement_rate': [],
        'mean_utility': [],
        'type_accuracy': [],
    }
    
    for results in results_list:
        regret_metrics = compute_regret_metrics(results)
        agreement_metrics = compute_agreement_metrics(results)
        type_metrics = compute_type_identification_accuracy(results)
        
        if regret_metrics:
            all_metrics['total_regret'].append(regret_metrics.get('total_regret', 0))
            all_metrics['avg_regret_per_episode'].append(
                regret_metrics.get('avg_regret_per_episode', 0)
            )
            all_metrics['mean_utility'].append(regret_metrics.get('mean_utility', 0))
        
        if agreement_metrics:
            all_metrics['agreement_rate'].append(
                agreement_metrics.get('agreement_rate', 0)
            )
        
        if type_metrics:
            all_metrics['type_accuracy'].append(type_metrics.get('type_accuracy', 0))
    
    # Compute statistics
    aggregated = {}
    for metric_name, values in all_metrics.items():
        if values:
            aggregated[f'{metric_name}_mean'] = float(np.mean(values))
            aggregated[f'{metric_name}_std'] = float(np.std(values))
            aggregated[f'{metric_name}_se'] = float(np.std(values) / np.sqrt(len(values)))
    
    return aggregated


def create_results_dataframe(results_dict: Dict[str, List[Dict]]) -> pd.DataFrame:
    """
    Create pandas DataFrame from results dictionary.
    
    Args:
        results_dict: {algorithm_name: [results_per_seed]}
        
    Returns:
        DataFrame with aggregated results
    """
    rows = []
    
    for alg_name, results_list in results_dict.items():
        aggregated = aggregate_results(results_list)
        aggregated['algorithm'] = alg_name
        rows.append(aggregated)
    
    return pd.DataFrame(rows)
