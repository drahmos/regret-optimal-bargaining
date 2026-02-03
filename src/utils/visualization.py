"""
Visualization Module
Plotting functions for experimental results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_regret_curves(results_dict: Dict[str, List[Dict]], 
                      save_path: Optional[str] = None,
                      show_std: bool = True):
    """
    Plot cumulative regret curves for multiple algorithms.
    
    Args:
        results_dict: {algorithm_name: [results_per_seed]}
        save_path: Optional path to save figure
        show_std: Whether to show standard deviation shading
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for idx, (alg_name, results_list) in enumerate(results_dict.items()):
        # Extract cumulative regrets across seeds
        all_regrets = []
        for results in results_list:
            if 'cumulative_regret' in results:
                all_regrets.append(results['cumulative_regret'])
            elif 'utilities' in results and 'oracle_utilities' in results:
                from .metrics import compute_cumulative_regret
                cum_reg = compute_cumulative_regret(
                    results['utilities'],
                    results['oracle_utilities']
                )
                all_regrets.append(cum_reg)
        
        if not all_regrets:
            continue
        
        # Convert to array
        max_len = max(len(r) for r in all_regrets)
        regrets_array = np.zeros((len(all_regrets), max_len))
        for i, r in enumerate(all_regrets):
            regrets_array[i, :len(r)] = r
        
        # Compute mean and std
        mean_regret = np.mean(regrets_array, axis=0)
        std_regret = np.std(regrets_array, axis=0)
        
        # Plot
        episodes = np.arange(len(mean_regret))
        ax.plot(episodes, mean_regret, label=alg_name, 
               linewidth=2, color=colors[idx])
        
        if show_std:
            ax.fill_between(episodes, 
                          mean_regret - std_regret,
                          mean_regret + std_regret,
                          alpha=0.2, color=colors[idx])
    
    ax.set_xlabel('Episode', fontsize=14)
    ax.set_ylabel('Cumulative Regret', fontsize=14)
    ax.set_title('Cumulative Regret Comparison', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    plt.close()


def plot_final_regret_comparison(summary_df, save_path: Optional[str] = None):
    """
    Plot bar chart comparing final regret across algorithms.
    
    Args:
        summary_df: DataFrame with aggregated results
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = summary_df['algorithm'].values
    
    # Get metrics
    if 'total_regret_mean' in summary_df.columns:
        means = summary_df['total_regret_mean'].values
        stds = summary_df.get('total_regret_std', np.zeros(len(means))).values
    else:
        means = summary_df['final_regret_mean'].values
        stds = summary_df.get('final_regret_std', np.zeros(len(means))).values
    
    # Compute 95% confidence intervals
    cis = 1.96 * stds / np.sqrt(10)  # Assuming 10 seeds
    
    x_pos = np.arange(len(algorithms))
    bars = ax.bar(x_pos, means, yerr=cis, capsize=5, alpha=0.7, 
                 color='steelblue', edgecolor='black')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Final Cumulative Regret', fontsize=14)
    ax.set_title('Algorithm Comparison (Final Regret)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mean:.1f}',
               ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    plt.close()


def plot_regret_scaling(horizons: List[int], 
                       regrets_dict: Dict[str, List[float]],
                       save_path: Optional[str] = None):
    """
    Plot regret scaling with horizon (log-log).
    
    Args:
        horizons: List of horizon values (T)
        regrets_dict: {algorithm: [regret_per_horizon]}
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for alg_name, regrets in regrets_dict.items():
        ax.loglog(horizons, regrets, marker='o', label=alg_name, linewidth=2)
    
    # Add reference lines for O(sqrt(T)) and O(T)
    T_ref = np.array(horizons)
    sqrt_line = regrets_dict[list(regrets_dict.keys())[0]][0] * np.sqrt(T_ref / T_ref[0])
    linear_line = regrets_dict[list(regrets_dict.keys())[0]][0] * (T_ref / T_ref[0])
    
    ax.loglog(T_ref, sqrt_line, 'k--', alpha=0.5, label='O(√T)')
    ax.loglog(T_ref, linear_line, 'r--', alpha=0.5, label='O(T)')
    
    ax.set_xlabel('Horizon (T)', fontsize=14)
    ax.set_ylabel('Cumulative Regret', fontsize=14)
    ax.set_title('Regret Scaling with Horizon', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    plt.close()


def plot_structure_exploitation(B_values: List[float],
                                 regrets_dict: Dict[str, List[float]],
                                 save_path: Optional[str] = None):
    """
    Plot how regret changes with bargaining structure parameter B.
    
    Args:
        B_values: List of B values
        regrets_dict: {algorithm: [regret_per_B]}
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for alg_name, regrets in regrets_dict.items():
        ax.plot(B_values, regrets, marker='o', label=alg_name, linewidth=2)
    
    ax.set_xlabel('Bargaining Structure Parameter (B)', fontsize=14)
    ax.set_ylabel('Cumulative Regret', fontsize=14)
    ax.set_title('Effect of Bargaining Structure on Regret', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    plt.close()


def plot_agreement_rates(results_dict: Dict[str, List[Dict]],
                        save_path: Optional[str] = None):
    """
    Plot agreement rates for different algorithms.
    
    Args:
        results_dict: {algorithm_name: [results_per_seed]}
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = []
    rates = []
    errors = []
    
    for alg_name, results_list in results_dict.items():
        # Extract agreement rates
        rates_per_seed = []
        for results in results_list:
            if 'agreements' in results:
                agreements = np.array(results['agreements'])
                rates_per_seed.append(np.mean(agreements))
        
        if rates_per_seed:
            algorithms.append(alg_name)
            rates.append(np.mean(rates_per_seed))
            errors.append(1.96 * np.std(rates_per_seed) / np.sqrt(len(rates_per_seed)))
    
    x_pos = np.arange(len(algorithms))
    bars = ax.bar(x_pos, rates, yerr=errors, capsize=5, alpha=0.7, 
                 color='forestgreen', edgecolor='black')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.set_ylabel('Agreement Rate', fontsize=14)
    ax.set_title('Agreement Rate by Algorithm', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.2%}',
               ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    plt.close()


def create_all_plots(results_dict: Dict[str, List[Dict]],
                    output_dir: str = "results/plots"):
    """
    Generate all standard plots.
    
    Args:
        results_dict: {algorithm_name: [results_per_seed]}
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating plots...")
    
    # Plot 1: Regret curves
    plot_regret_curves(
        results_dict,
        save_path=os.path.join(output_dir, "fig1_regret_curves.png")
    )
    
    # Plot 2: Final regret comparison
    from .metrics import create_results_dataframe
    summary_df = create_results_dataframe(results_dict)
    plot_final_regret_comparison(
        summary_df,
        save_path=os.path.join(output_dir, "fig2_final_regret_comparison.png")
    )
    
    # Plot 3: Agreement rates
    plot_agreement_rates(
        results_dict,
        save_path=os.path.join(output_dir, "fig3_agreement_rates.png")
    )
    
    print(f"All plots saved to {output_dir}")
