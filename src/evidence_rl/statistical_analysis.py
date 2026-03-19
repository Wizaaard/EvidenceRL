"""
Statistical Analysis Module for EvidenceRL Metrics

Provides confidence intervals, significance testing, and effect size computation
for robust model comparison.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """Results of statistical comparison between two models."""
    mean_diff: float
    p_value: float
    significant: bool
    test_name: str
    effect_size: float
    ci_lower: float
    ci_upper: float

    def __str__(self):
        return (f"{self.test_name}: "
                f"Δ={self.mean_diff:.4f}, "
                f"p={self.p_value:.4f} {'*' if self.significant else ''}, "
                f"CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
                f"d={self.effect_size:.4f}")


def bootstrap_ci(values: np.ndarray,
                 num_bootstrap: int = 1000,
                 confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        values: Array of metric values
        num_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95)

    Returns:
        (lower_bound, upper_bound) of confidence interval
    """
    bootstrap_means = []
    n = len(values)

    for _ in range(num_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

    return lower, upper


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Effect size (standardized mean difference)
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compare_two_models(metrics_a: List[float],
                       metrics_b: List[float],
                       alpha: float = 0.05) -> ComparisonResult:
    """
    Compare two models with appropriate statistical test.

    Automatically selects between t-test (if normal) and Mann-Whitney U (if non-normal).

    Args:
        metrics_a: Metrics from model A
        metrics_b: Metrics from model B
        alpha: Significance level (default 0.05)

    Returns:
        ComparisonResult with test statistics
    """
    arr_a = np.array(metrics_a)
    arr_b = np.array(metrics_b)

    # Test normality (if n >= 3)
    normal_a = True
    normal_b = True

    if len(arr_a) >= 3:
        _, p_norm_a = stats.shapiro(arr_a)
        normal_a = p_norm_a > alpha

    if len(arr_b) >= 3:
        _, p_norm_b = stats.shapiro(arr_b)
        normal_b = p_norm_b > alpha

    # Choose test
    if normal_a and normal_b:
        # Parametric test
        stat, p_value = stats.ttest_ind(arr_a, arr_b)
        test_name = "Independent t-test"
    else:
        # Non-parametric test
        stat, p_value = stats.mannwhitneyu(arr_a, arr_b, alternative='two-sided')
        test_name = "Mann-Whitney U"

    # Effect size
    effect_size = cohens_d(arr_a, arr_b)

    # Confidence interval on difference
    diff = arr_a - arr_b[:len(arr_a)] if len(arr_a) == len(arr_b) else None
    if diff is not None:
        ci_lower, ci_upper = bootstrap_ci(diff)
    else:
        # Different lengths - use bootstrap on mean difference
        bootstrap_diffs = []
        for _ in range(1000):
            sample_a = np.random.choice(arr_a, size=len(arr_a), replace=True)
            sample_b = np.random.choice(arr_b, size=len(arr_b), replace=True)
            bootstrap_diffs.append(np.mean(sample_a) - np.mean(sample_b))
        ci_lower, ci_upper = np.percentile(bootstrap_diffs, [2.5, 97.5])

    return ComparisonResult(
        mean_diff=np.mean(arr_a) - np.mean(arr_b),
        p_value=p_value,
        significant=p_value < alpha,
        test_name=test_name,
        effect_size=effect_size,
        ci_lower=ci_lower,
        ci_upper=ci_upper
    )


def compute_metric_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute comprehensive statistics for a single metric.

    Args:
        values: List of metric values

    Returns:
        Dictionary with mean, median, std, CI, etc.
    """
    arr = np.array(values)
    ci_lower, ci_upper = bootstrap_ci(arr)

    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "ci_width": float(ci_upper - ci_lower)
    }


def compare_multiple_models(model_metrics: Dict[str, List[float]],
                           alpha: float = 0.05) -> Dict[str, any]:
    """
    Compare multiple models with ANOVA/Kruskal-Wallis and post-hoc tests.

    Args:
        model_metrics: Dictionary mapping model_name -> list of metric values
        alpha: Significance level

    Returns:
        Dictionary with overall test results and pairwise comparisons
    """
    model_names = list(model_metrics.keys())
    model_arrays = [np.array(model_metrics[name]) for name in model_names]

    # Test normality for all groups
    all_normal = True
    for arr in model_arrays:
        if len(arr) >= 3:
            _, p_norm = stats.shapiro(arr)
            if p_norm <= alpha:
                all_normal = False
                break

    # Overall test
    if all_normal:
        # One-way ANOVA
        stat, p_value = stats.f_oneway(*model_arrays)
        test_name = "One-way ANOVA"
    else:
        # Kruskal-Wallis H-test
        stat, p_value = stats.kruskal(*model_arrays)
        test_name = "Kruskal-Wallis H"

    results = {
        "overall_test": test_name,
        "overall_statistic": float(stat),
        "overall_p_value": float(p_value),
        "significant_difference": p_value < alpha,
        "pairwise_comparisons": {}
    }

    # Pairwise post-hoc comparisons (if overall significant)
    if p_value < alpha:
        from itertools import combinations

        pairwise_p_values = []
        for name_a, name_b in combinations(model_names, 2):
            comparison = compare_two_models(
                model_metrics[name_a],
                model_metrics[name_b],
                alpha=alpha
            )
            results["pairwise_comparisons"][f"{name_a}_vs_{name_b}"] = {
                "mean_diff": comparison.mean_diff,
                "p_value": comparison.p_value,
                "significant": comparison.significant,
                "effect_size": comparison.effect_size
            }
            pairwise_p_values.append(comparison.p_value)

        # Bonferroni correction for multiple comparisons
        n_comparisons = len(pairwise_p_values)
        corrected_alpha = alpha / n_comparisons

        results["bonferroni_corrected_alpha"] = corrected_alpha
        results["n_significant_after_correction"] = sum(
            1 for p in pairwise_p_values if p < corrected_alpha
        )

    return results


def compute_correlation_matrix(metrics_dict: Dict[str, List[float]]) -> np.ndarray:
    """
    Compute correlation matrix for multiple metrics.

    Args:
        metrics_dict: Dictionary mapping metric_name -> list of values

    Returns:
        Correlation matrix (Pearson correlation)
    """
    metric_names = list(metrics_dict.keys())

    # Ensure all lists have same length
    min_length = min(len(v) for v in metrics_dict.values())
    data = np.array([metrics_dict[name][:min_length] for name in metric_names])

    return np.corrcoef(data)


if __name__ == "__main__":
    # Example usage
    import json

    # Simulate some metrics
    np.random.seed(42)
    model_a_precision = np.random.beta(8, 2, 100)  # Mean ~0.8
    model_b_precision = np.random.beta(7, 3, 100)  # Mean ~0.7

    print("=== Single Metric Statistics ===")
    stats_a = compute_metric_statistics(model_a_precision.tolist())
    print(f"Model A: mean={stats_a['mean']:.4f}, "
          f"95% CI=[{stats_a['ci_lower']:.4f}, {stats_a['ci_upper']:.4f}]")

    print("\n=== Two Model Comparison ===")
    comparison = compare_two_models(
        model_a_precision.tolist(),
        model_b_precision.tolist()
    )
    print(comparison)

    print("\n=== Multiple Model Comparison ===")
    model_c_precision = np.random.beta(6, 4, 100)  # Mean ~0.6
    multi_results = compare_multiple_models({
        "Model A": model_a_precision.tolist(),
        "Model B": model_b_precision.tolist(),
        "Model C": model_c_precision.tolist()
    })
    print(json.dumps(multi_results, indent=2))
