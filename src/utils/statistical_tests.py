"""
Statistical Tests Module
Statistical significance testing and effect sizes
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional


def paired_ttest(method1_results: List[float], 
                method2_results: List[float]) -> Tuple[float, float, float]:
    """
    Run paired t-test between two methods.
    
    Args:
        method1_results: Results from method 1 (e.g., TSB)
        method2_results: Results from method 2 (e.g., UCB1)
        
    Returns:
        (t_statistic, p_value, cohen_d)
    """
    method1_results = np.array(method1_results)
    method2_results = np.array(method2_results)
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(method1_results, method2_results)
    
    # Cohen's d for effect size
    diff = method1_results - method2_results
    cohen_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
    
    return t_stat, p_val, cohen_d


def wilcoxon_signed_rank(method1_results: List[float],
                         method2_results: List[float]) -> Tuple[float, float]:
    """
    Run Wilcoxon signed-rank test (non-parametric alternative to t-test).
    
    Args:
        method1_results: Results from method 1
        method2_results: Results from method 2
        
    Returns:
        (statistic, p_value)
    """
    statistic, p_val = stats.wilcoxon(method1_results, method2_results)
    return statistic, p_val


def compute_cohens_d(group1: List[float], 
                    group2: List[float],
                    paired: bool = True) -> float:
    """
    Compute Cohen's d effect size.
    
    Args:
        group1: First group of results
        group2: Second group of results
        paired: Whether samples are paired
        
    Returns:
        Cohen's d
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    if paired:
        # Paired Cohen's d
        diff = group1 - group2
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        return mean_diff / (std_diff + 1e-10)
    else:
        # Independent Cohen's d
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        return (mean1 - mean2) / (pooled_std + 1e-10)


def compute_confidence_interval(data: List[float], 
                               confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval for mean.
    
    Args:
        data: Sample data
        confidence: Confidence level (default: 0.95)
        
    Returns:
        (lower_bound, upper_bound)
    """
    data = np.array(data)
    mean = np.mean(data)
    se = stats.sem(data)  # Standard error
    
    # t-distribution critical value
    h = se * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    
    return mean - h, mean + h


def bonferroni_correction(p_values: List[float], 
                         alpha: float = 0.05) -> List[bool]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values
        alpha: Significance level (default: 0.05)
        
    Returns:
        List of booleans indicating significance after correction
    """
    n = len(p_values)
    corrected_alpha = alpha / n
    
    return [p < corrected_alpha for p in p_values]


def compare_all_algorithms(results_dict: Dict[str, List[float]],
                          baseline_name: str = 'UCB1') -> Dict[str, Dict]:
    """
    Compare all algorithms against a baseline.
    
    Args:
        results_dict: {algorithm_name: [results_per_seed]}
        baseline_name: Name of baseline algorithm
        
    Returns:
        Dictionary with comparison results
    """
    if baseline_name not in results_dict:
        raise ValueError(f"Baseline {baseline_name} not found in results")
    
    baseline_results = results_dict[baseline_name]
    comparisons = {}
    
    for alg_name, alg_results in results_dict.items():
        if alg_name == baseline_name:
            continue
        
        # Run tests
        t_stat, p_val, cohen_d = paired_ttest(alg_results, baseline_results)
        
        comparisons[alg_name] = {
            'vs': baseline_name,
            't_statistic': t_stat,
            'p_value': p_val,
            'cohen_d': cohen_d,
            'significant': p_val < 0.05,
            'effect_size': interpret_cohens_d(abs(cohen_d)),
            'mean_diff': np.mean(alg_results) - np.mean(baseline_results),
            'percent_improvement': (np.mean(baseline_results) - np.mean(alg_results)) / np.mean(baseline_results) * 100
        }
    
    return comparisons


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
        
    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def format_test_result(t_stat: float, p_val: float, cohen_d: float) -> str:
    """
    Format statistical test result as string.
    
    Args:
        t_stat: t-statistic
        p_val: p-value
        cohen_d: Cohen's d
        
    Returns:
        Formatted string
    """
    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    effect = interpret_cohens_d(cohen_d)
    
    return (f"t={t_stat:.3f}, p={p_val:.4f} {significance}, "
            f"d={cohen_d:.3f} ({effect})")


def print_comparison_table(comparisons: Dict[str, Dict], 
                           metric_name: str = "Regret"):
    """
    Print formatted comparison table.
    
    Args:
        comparisons: Output from compare_all_algorithms
        metric_name: Name of metric being compared
    """
    print(f"\n{metric_name} Comparison Results")
    print("=" * 80)
    print(f"{'Algorithm':<20} {'Mean Diff':<12} {'% Improv':<10} {'p-value':<12} {'Effect Size':<15}")
    print("-" * 80)
    
    for alg_name, comp in comparisons.items():
        sig_marker = "***" if comp['p_value'] < 0.001 else "**" if comp['p_value'] < 0.01 else "*" if comp['p_value'] < 0.05 else ""
        print(f"{alg_name:<20} "
              f"{comp['mean_diff']:>+11.2f} "
              f"{comp['percent_improvement']:>9.1f}% "
              f"{comp['p_value']:.4f}{sig_marker:<3} "
              f"{comp['effect_size']:<15}")
    
    print("=" * 80)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05")
