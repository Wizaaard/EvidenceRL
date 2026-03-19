#!/usr/bin/env python3
"""
Enhanced Metrics Analysis Script

Demonstrates usage of new statistical analysis, validated judge, and
parse error classification modules.

Usage:
    python enhanced_metrics_analysis.py \
        --metrics-file metrics_output/model_a_metrics.json \
        --compare-with metrics_output/model_b_metrics.json \
        --output-dir analysis_output/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evidence_rl.statistical_analysis import (
    compute_metric_statistics,
    compare_two_models,
    compare_multiple_models,
    bootstrap_ci
)
from evidence_rl.parse_error_analysis import (
    ParseErrorAnalyzer,
    ParsingErrorType
)


def load_metrics_file(file_path: str) -> Dict:
    """Load metrics JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_metric_values(metrics_data: Dict, metric_name: str) -> List[float]:
    """
    Extract metric values from metrics file.

    Args:
        metrics_data: Loaded metrics JSON
        metric_name: e.g., "precision@3", "reward_grounding@3"

    Returns:
        List of metric values across patients
    """
    values = []

    # Handle different metric formats
    if "patient_metrics" in metrics_data:
        for patient in metrics_data["patient_metrics"]:
            if metric_name in patient:
                values.append(patient[metric_name])
            elif "@" in metric_name:
                # Handle @k metrics
                base_metric, k = metric_name.split("@")
                k = int(k)
                if base_metric in patient:
                    metric_dict = patient[base_metric]
                    if isinstance(metric_dict, dict) and k in metric_dict:
                        values.append(metric_dict[k])

    return values


def analyze_single_model(metrics_file: str, output_dir: str):
    """Analyze a single model's metrics with statistical rigor."""
    print(f"\n{'='*60}")
    print(f"SINGLE MODEL ANALYSIS: {Path(metrics_file).name}")
    print(f"{'='*60}\n")

    # Load metrics
    data = load_metrics_file(metrics_file)

    # Key metrics to analyze
    metrics_to_analyze = [
        "precision@3",
        "recall@3",
        "reward_grounding@3",
        "reward_accuracy@3",
        "reward_combined@3"
    ]

    results = {}

    for metric_name in metrics_to_analyze:
        values = extract_metric_values(data, metric_name)

        if not values:
            print(f"⚠️ Metric '{metric_name}' not found in data")
            continue

        # Compute comprehensive statistics
        stats = compute_metric_statistics(values)
        results[metric_name] = stats

        # Print results
        print(f"📊 {metric_name}:")
        print(f"   Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"   95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
        print(f"   Median: {stats['median']:.4f}")
        print(f"   Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print()

    # Parse error analysis (if generation results available)
    if "results" in data or "summary" in data:
        print("\n📝 Parse Error Analysis:")
        analyzer = ParseErrorAnalyzer()

        if "results" in data:
            for result in data["results"]:
                raw_output = result.get("structured_output", {}).get("raw_output", "")
                parse_success = result.get("structured_output", {}).get("parse_success", False)
                hadm_id = result.get("hadm_id")

                # Simulate error tracking
                if not parse_success and raw_output:
                    # We don't have the actual error, but can classify based on output
                    analyzer.track_attempt(
                        raw_output=raw_output,
                        error=None,
                        hadm_id=hadm_id,
                        success=False
                    )
                else:
                    analyzer.track_attempt(
                        raw_output=raw_output,
                        error=None,
                        hadm_id=hadm_id,
                        success=True
                    )

        elif "summary" in data:
            # Use summary stats
            parse_rate = data["summary"].get("parse_success_rate", 0.0)
            print(f"   Parse success rate: {parse_rate:.1%}")

        # Get error summary
        error_summary = analyzer.get_summary()
        print(error_summary)

        print("\n💡 Recommendations:")
        for rec in analyzer.get_recommendations():
            print(f"   {rec}")

        # Export detailed errors
        output_path = Path(output_dir) / "parse_errors.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        analyzer.export_error_examples(str(output_path))

    # Save results
    output_path = Path(output_dir) / "statistical_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {output_path}")


def compare_two_model_runs(file_a: str, file_b: str, output_dir: str):
    """Compare two models statistically."""
    print(f"\n{'='*60}")
    print(f"TWO MODEL COMPARISON")
    print(f"{'='*60}\n")

    print(f"Model A: {Path(file_a).name}")
    print(f"Model B: {Path(file_b).name}")
    print()

    # Load both files
    data_a = load_metrics_file(file_a)
    data_b = load_metrics_file(file_b)

    # Metrics to compare
    metrics_to_compare = [
        "precision@3",
        "recall@3",
        "reward_grounding@3",
        "reward_accuracy@3",
        "reward_combined@3"
    ]

    comparison_results = {}

    for metric_name in metrics_to_compare:
        values_a = extract_metric_values(data_a, metric_name)
        values_b = extract_metric_values(data_b, metric_name)

        if not values_a or not values_b:
            print(f"⚠️ Skipping '{metric_name}' - insufficient data")
            continue

        # Statistical comparison
        comparison = compare_two_models(values_a, values_b)

        comparison_results[metric_name] = {
            "mean_a": float(sum(values_a) / len(values_a)),
            "mean_b": float(sum(values_b) / len(values_b)),
            "mean_diff": comparison.mean_diff,
            "p_value": comparison.p_value,
            "significant": comparison.significant,
            "effect_size": comparison.effect_size,
            "ci_lower": comparison.ci_lower,
            "ci_upper": comparison.ci_upper,
            "test": comparison.test_name
        }

        # Print results
        print(f"📊 {metric_name}:")
        print(f"   Model A: {comparison_results[metric_name]['mean_a']:.4f}")
        print(f"   Model B: {comparison_results[metric_name]['mean_b']:.4f}")
        print(f"   {comparison}")

        if comparison.significant:
            winner = "A" if comparison.mean_diff > 0 else "B"
            print(f"   🏆 Model {winner} is significantly better (p < 0.05)")
        else:
            print(f"   ↔️  No significant difference")

        print()

    # Save results
    output_path = Path(output_dir) / "model_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)

    print(f"✅ Comparison results saved to {output_path}")


def compare_multiple_model_runs(files: List[str], output_dir: str):
    """Compare multiple models with ANOVA/Kruskal-Wallis."""
    print(f"\n{'='*60}")
    print(f"MULTIPLE MODEL COMPARISON ({len(files)} models)")
    print(f"{'='*60}\n")

    # Load all files
    all_data = {}
    for file_path in files:
        model_name = Path(file_path).stem  # Use filename as model name
        all_data[model_name] = load_metrics_file(file_path)
        print(f"   Loaded: {model_name}")

    print()

    # Metrics to compare
    metrics_to_compare = [
        "precision@3",
        "reward_combined@3"
    ]

    for metric_name in metrics_to_compare:
        print(f"📊 Comparing {metric_name} across {len(files)} models:\n")

        # Extract values for each model
        model_metrics = {}
        for model_name, data in all_data.items():
            values = extract_metric_values(data, metric_name)
            if values:
                model_metrics[model_name] = values
                print(f"   {model_name}: mean={sum(values)/len(values):.4f} (n={len(values)})")

        if len(model_metrics) < 2:
            print(f"   ⚠️ Insufficient models with this metric")
            continue

        print()

        # Statistical comparison
        comparison_results = compare_multiple_models(model_metrics)

        print(f"   {comparison_results['overall_test']}:")
        print(f"   Statistic: {comparison_results['overall_statistic']:.4f}")
        print(f"   p-value: {comparison_results['overall_p_value']:.4e}")

        if comparison_results['significant_difference']:
            print(f"   ✅ Significant difference detected (p < 0.05)")

            # Show pairwise comparisons
            if comparison_results.get('pairwise_comparisons'):
                print(f"\n   Pairwise Comparisons:")
                for pair_name, pair_result in comparison_results['pairwise_comparisons'].items():
                    sig_marker = "**" if pair_result['significant'] else ""
                    print(f"      {pair_name}: Δ={pair_result['mean_diff']:.4f}, "
                          f"p={pair_result['p_value']:.4f} {sig_marker}")

            print(f"\n   Bonferroni-corrected α: {comparison_results['bonferroni_corrected_alpha']:.4f}")
            print(f"   Significant pairs after correction: {comparison_results['n_significant_after_correction']}")
        else:
            print(f"   ↔️  No significant difference between models")

        print("\n" + "-"*60 + "\n")

    print(f"✅ Multi-model comparison complete")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced metrics analysis with statistical rigor"
    )
    parser.add_argument(
        "--metrics-file",
        required=True,
        help="Path to metrics JSON file"
    )
    parser.add_argument(
        "--compare-with",
        nargs="+",
        help="Additional metrics files to compare with"
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_output",
        help="Directory for output files"
    )

    args = parser.parse_args()

    # Single model analysis
    analyze_single_model(args.metrics_file, args.output_dir)

    # Two-model comparison
    if args.compare_with and len(args.compare_with) == 1:
        compare_two_model_runs(
            args.metrics_file,
            args.compare_with[0],
            args.output_dir
        )

    # Multi-model comparison
    elif args.compare_with and len(args.compare_with) > 1:
        all_files = [args.metrics_file] + args.compare_with
        compare_multiple_model_runs(all_files, args.output_dir)


if __name__ == "__main__":
    main()
