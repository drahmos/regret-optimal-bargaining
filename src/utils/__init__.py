"""
Utilities package
"""

from .metrics import (
    compute_cumulative_regret,
    compute_regret_metrics,
    compute_agreement_metrics,
    compute_type_identification_accuracy,
    aggregate_results,
    create_results_dataframe
)

from .visualization import (
    plot_regret_curves,
    plot_final_regret_comparison,
    plot_regret_scaling,
    plot_structure_exploitation,
    plot_agreement_rates,
    create_all_plots
)

from .statistical_tests import (
    paired_ttest,
    wilcoxon_signed_rank,
    compute_cohens_d,
    compute_confidence_interval,
    bonferroni_correction,
    compare_all_algorithms,
    interpret_cohens_d,
    format_test_result,
    print_comparison_table
)

__all__ = [
    # Metrics
    'compute_cumulative_regret',
    'compute_regret_metrics',
    'compute_agreement_metrics',
    'compute_type_identification_accuracy',
    'aggregate_results',
    'create_results_dataframe',
    # Visualization
    'plot_regret_curves',
    'plot_final_regret_comparison',
    'plot_regret_scaling',
    'plot_structure_exploitation',
    'plot_agreement_rates',
    'create_all_plots',
    # Statistical tests
    'paired_ttest',
    'wilcoxon_signed_rank',
    'compute_cohens_d',
    'compute_confidence_interval',
    'bonferroni_correction',
    'compare_all_algorithms',
    'interpret_cohens_d',
    'format_test_result',
    'print_comparison_table'
]
