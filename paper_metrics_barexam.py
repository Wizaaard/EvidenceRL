#!/usr/bin/env python3
"""Compute publication-ready BarExam metrics from enhanced_metrics.json files.

Produces a multi-model LaTeX table analogous to the medical domain's paper_metrics.py:
  - Accuracy (replaces F1@k since BarExam is MCQ)
  - Grounding: G_max (combined_sentence_max), G_avg (combined_sentence_avg)
  - 3x2 Taxonomy: EB, RF, H, W, LG, Faithfulness
  - 95% bootstrap confidence intervals

Usage:
    # All models (auto-discover):
    python paper_metrics_barexam.py --version v1.2

    # Single file:
    python paper_metrics_barexam.py --metrics-json path/to/enhanced_metrics.json

    # Custom thresholds:
    python paper_metrics_barexam.py --version v1.2 --threshold-high 0.5 --threshold-low 0.1

    # Use legal dimension instead of combined:
    python paper_metrics_barexam.py --version v1.2 --grounding-dim legal
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parent.parent
BAREXAM_OUTPUT = DATA_ROOT / "barexam_output" / "test"

MODELS = [
    "gemma-3-4b",
    "gemma-3-12b",
    "gemma-3-27b",
    "Llama-3.2-3B",
    "Llama-3.1-8B",
    "Llama-3.3-70B",
    "gpt-oss-20b",
    "gpt-oss-120b",
]

# Display names for LaTeX
MODEL_DISPLAY = {
    "gemma-3-4b": ("Gemma-3", "4B"),
    "gemma-3-12b": ("Gemma-3", "12B"),
    "gemma-3-27b": ("Gemma-3", "27B"),
    "Llama-3.2-3B": ("Llama-3.2", "3B"),
    "Llama-3.1-8B": ("Llama-3.1", "8B"),
    "Llama-3.3-70B": ("Llama-3.3", "70B"),
    "gpt-oss-20b": ("GPT-OSS", "20B"),
    "gpt-oss-120b": ("GPT-OSS", "120B"),
}

TAXONOMY_LABELS = {
    ("correct", "grounded"): "Evidence-Based",
    ("correct", "weak"): "Weakly Supported",
    ("correct", "contradicted"): "Lucky Guess",
    ("incorrect", "grounded"): "Reasoning Failure",
    ("incorrect", "weak"): "Unsupported Error",
    ("incorrect", "contradicted"): "Hallucination",
}


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper)."""
    arr = np.array(values)
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return mean, mean, mean

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap)
    n = len(arr)
    for i in range(n_bootstrap):
        sample = arr[rng.randint(0, n, size=n)]
        boot_means[i] = sample.mean()

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_means, 100 * alpha))
    ci_upper = float(np.percentile(boot_means, 100 * (1.0 - alpha)))
    return mean, ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

def classify_case(
    correct: bool,
    grounding_score: float | None,
    threshold: float = 0.5,
) -> str:
    """Classify a single BarExam case into the 3x2 taxonomy.

    For scores on [-1, 1] scale (entailment - contradiction):
      - Grounded:     score > threshold
      - Weak:         -threshold <= score <= threshold
      - Contradicted: score < -threshold

    Same logic as medical domain's classify_diagnosis().
    """
    correctness = "correct" if correct else "incorrect"
    if grounding_score is None:
        grounding = "weak"
    elif grounding_score > threshold:
        grounding = "grounded"
    elif grounding_score < -threshold:
        grounding = "contradicted"
    else:
        grounding = "weak"
    return TAXONOMY_LABELS[(correctness, grounding)]


def compute_taxonomy(
    cases: List[Dict[str, Any]],
    grounding_key: str,
    threshold: float,
) -> Dict[str, Any]:
    """Compute the 3x2 taxonomy for all cases."""
    counts: Counter = Counter()
    total = 0

    for case in cases:
        correct = case.get("correct", False)
        g_val = case.get(grounding_key)
        label = classify_case(correct, g_val, threshold)
        counts[label] += 1
        total += 1

    rates = {}
    for label in TAXONOMY_LABELS.values():
        rates[label] = counts[label] / total if total > 0 else 0.0

    eb = counts["Evidence-Based"]
    ws = counts["Weakly Supported"]
    lg = counts["Lucky Guess"]
    n_correct = eb + ws + lg

    return {
        "counts": dict(counts),
        "rates": rates,
        "total": total,
        "eb_pct": rates["Evidence-Based"],
        "rf_pct": rates["Reasoning Failure"],
        "h_pct": rates["Hallucination"],
        "lg_pct": rates["Lucky Guess"],
        "weak_pct": rates["Weakly Supported"] + rates["Unsupported Error"],
        "faithfulness": eb / n_correct if n_correct > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Load and process one model
# ---------------------------------------------------------------------------

def process_model(
    metrics_path: Path,
    taxonomy_key: str,
    threshold: float,
    n_bootstrap: int,
) -> Dict[str, Any]:
    """Process a single model's enhanced_metrics.json.

    Supports both old format (combined_sentence_max_score, etc.) and
    new format (grounding_max, grounding_avg) from the updated pipeline.
    """
    with metrics_path.open() as f:
        data = json.load(f)

    cases = data["per_case"]
    metrics = data["metrics"]

    # Detect format: new pipeline has "grounding_max"/"grounding_avg",
    # old pipeline has "{dim}_sentence_{max,avg}_score"
    sample = cases[0] if cases else {}
    new_format = "grounding_max" in sample

    if new_format:
        g_max_key = "grounding_max"
        g_avg_key = "grounding_avg"
    else:
        # Fall back to old format (default: combined dimension)
        g_max_key = "combined_sentence_max_score"
        g_avg_key = "combined_sentence_avg_score"

    # Override taxonomy key if it matches the detected keys
    if taxonomy_key == "grounding_avg":
        tax_key = g_avg_key
    elif taxonomy_key == "grounding_max":
        tax_key = g_max_key
    else:
        tax_key = taxonomy_key

    # Accuracy with bootstrap CI
    acc_values = [1.0 if c.get("correct", False) else 0.0 for c in cases]
    acc_mean, acc_lo, acc_hi = bootstrap_ci(acc_values, n_bootstrap=n_bootstrap)

    # Grounding scores (skip None)
    g_max_values = [c[g_max_key] for c in cases if c.get(g_max_key) is not None]
    g_avg_values = [c[g_avg_key] for c in cases if c.get(g_avg_key) is not None]

    g_max_mean, g_max_lo, g_max_hi = bootstrap_ci(g_max_values, n_bootstrap=n_bootstrap) if g_max_values else (0, 0, 0)
    g_avg_mean, g_avg_lo, g_avg_hi = bootstrap_ci(g_avg_values, n_bootstrap=n_bootstrap) if g_avg_values else (0, 0, 0)

    # Taxonomy
    taxonomy = compute_taxonomy(cases, tax_key, threshold)

    return {
        "n_cases": len(cases),
        "accuracy": (acc_mean, acc_lo, acc_hi),
        "g_max": (g_max_mean, g_max_lo, g_max_hi),
        "g_avg": (g_avg_mean, g_avg_lo, g_avg_hi),
        "taxonomy": taxonomy,
        "parse_rate": metrics.get("parse_rate", 0),
        "format": "new" if new_format else "legacy",
    }


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def fmt_pct_ci(mean: float, lo: float, hi: float) -> str:
    """Format as XX.X for table cell."""
    return f"{mean * 100:.1f}"


def fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}"


def print_multi_model_table(
    results: Dict[str, Dict[str, Any]],
    taxonomy_key: str,
    threshold: float,
) -> None:
    """Print the paper-ready multi-model LaTeX table."""

    tax_label = f"$G_{{\\text{{avg}}}}$" if "avg" in taxonomy_key else f"$G_{{\\max}}$"

    print()
    print("=" * 100)
    print(f"BAREXAM TABLE (taxonomy on: {taxonomy_key}, "
          f"threshold: ±{threshold})")
    print("=" * 100)

    # Table header
    print()
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{BarExam MBE: Accuracy, Evidence Grounding, and Diagnostic Taxonomy across backbone models (RAG). "
          f"Taxonomy computed using {tax_label} with threshold $\\tau = {threshold}$.}}")
    print("\\label{tab:barexam_results}")
    print("\\resizebox{\\textwidth}{!}{%")
    print("\\begin{tabular}{ll|c|cc|ccccccc}")
    print("\\toprule")
    print("\\multirow{2}{*}{\\textbf{Family}} & \\multirow{2}{*}{\\textbf{Size}} & "
          "\\multirow{2}{*}{\\textbf{Acc.}($\\uparrow$)} & "
          "\\textbf{$G_{\\text{avg}}$}($\\uparrow$) & "
          "\\textbf{$G_{\\max}$}($\\uparrow$) & "
          "\\textbf{EB}($\\uparrow$) & "
          "\\textbf{RF}($\\downarrow$) & "
          "\\textbf{H}($\\downarrow$) & "
          "\\textbf{W}($\\downarrow$) & "
          "\\textbf{LG}($\\downarrow$) & "
          "\\textbf{F}($\\uparrow$) \\\\")
    print(" & & & \\multicolumn{2}{c|}{\\footnotesize NLI Grounding} & "
          "\\multicolumn{6}{c}{\\footnotesize Diagnostic Taxonomy} \\\\")
    print("\\midrule")

    # Sort models by family then size for clean grouping
    model_order = [
        "gemma-3-4b", "gemma-3-12b", "gemma-3-27b",
        "Llama-3.2-3B", "Llama-3.1-8B", "Llama-3.3-70B",
        "gpt-oss-20b", "gpt-oss-120b",
    ]

    prev_family = None
    for model_id in model_order:
        if model_id not in results:
            continue

        r = results[model_id]
        family, size = MODEL_DISPLAY[model_id]
        tax = r["taxonomy"]

        acc_str = fmt_pct_ci(*r["accuracy"])
        g_avg_str = f"{r['g_avg'][0]:.3f}"
        g_max_str = f"{r['g_max'][0]:.3f}"

        eb_str = fmt_pct(tax["eb_pct"])
        rf_str = fmt_pct(tax["rf_pct"])
        h_str = fmt_pct(tax["h_pct"])
        w_str = fmt_pct(tax["weak_pct"])
        lg_str = fmt_pct(tax["lg_pct"])
        f_str = fmt_pct(tax["faithfulness"])

        # Add midrule between families
        if prev_family is not None and family != prev_family:
            print("\\midrule")

        print(f"{family} & {size} & "
              f"{acc_str} & {g_avg_str} & {g_max_str} & "
              f"{eb_str} & {rf_str} & {h_str} & {w_str} & {lg_str} & {f_str} \\\\")

        prev_family = family

    print("\\bottomrule")
    print("\\end{tabular}}")
    print("\\end{table}")

    # Also print with CIs and raw numbers
    print()
    print("=" * 100)
    print("DETAILED RESULTS (with 95% Bootstrap CIs)")
    print("=" * 100)

    for model_id in model_order:
        if model_id not in results:
            continue
        r = results[model_id]
        tax = r["taxonomy"]
        family, size = MODEL_DISPLAY[model_id]

        acc_m, acc_lo, acc_hi = r["accuracy"]
        gm_m, gm_lo, gm_hi = r["g_max"]
        ga_m, ga_lo, ga_hi = r["g_avg"]

        print(f"\n{family} {size} (n={r['n_cases']}, parse={r['parse_rate']:.1%}):")
        print(f"  Accuracy:  {acc_m:.3f} [{acc_lo:.3f}, {acc_hi:.3f}]")
        print(f"  G_max:     {gm_m:.3f} [{gm_lo:.3f}, {gm_hi:.3f}]")
        print(f"  G_avg:     {ga_m:.3f} [{ga_lo:.3f}, {ga_hi:.3f}]")
        print(f"  Taxonomy:  EB={tax['eb_pct']:.1%}  RF={tax['rf_pct']:.1%}  "
              f"H={tax['h_pct']:.1%}  W={tax['weak_pct']:.1%}  "
              f"LG={tax['lg_pct']:.1%}  Faith={tax['faithfulness']:.1%}")
        print(f"  Counts:    {tax['counts']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BarExam paper-ready metrics")
    parser.add_argument("--metrics-json", type=str, default=None,
                        help="Single enhanced_metrics.json file")
    parser.add_argument("--version", type=str, default="v1.2",
                        help="Version tag for auto-discovery (default: v1.2)")
    parser.add_argument("--mode-tag", type=str, default="rag",
                        help="Mode tag (default: rag)")
    parser.add_argument("--metrics-subdir", type=str, default="metrics",
                        help="Metrics subdirectory name (default: metrics, use metrics_v2 for new pipeline)")
    parser.add_argument("--taxonomy-key", type=str, default="grounding_avg",
                        choices=["grounding_avg", "grounding_max"],
                        help="Grounding metric for taxonomy (default: grounding_avg)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Symmetric threshold ±τ for grounded/contradicted (default: 0.5, same as medical)")
    parser.add_argument("--n-bootstrap", type=int, default=10_000,
                        help="Bootstrap samples for CIs (default: 10000)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results as JSON")
    args = parser.parse_args()

    all_results = {}

    if args.metrics_json:
        # Single file mode
        path = Path(args.metrics_json)
        model_id = path.parent.parent.name.split(f"_{args.mode_tag}")[0]
        print(f"Loading: {path} (model: {model_id})")
        all_results[model_id] = process_model(
            path, args.taxonomy_key, args.threshold, args.n_bootstrap
        )
    else:
        # Auto-discover all models
        print(f"Auto-discovering models: version={args.version}, mode={args.mode_tag}")
        print(f"Looking in: {BAREXAM_OUTPUT}")

        for model_id in MODELS:
            result_dir = BAREXAM_OUTPUT / f"{model_id}_{args.mode_tag}-{args.version}"
            metrics_path = result_dir / args.metrics_subdir / "enhanced_metrics.json"

            if not metrics_path.exists():
                print(f"  SKIP: {model_id} (no metrics at {metrics_path})")
                continue

            print(f"  Loading: {model_id}")
            all_results[model_id] = process_model(
                metrics_path, args.taxonomy_key, args.threshold, args.n_bootstrap
            )

    if not all_results:
        print("No results found!")
        return

    print(f"\nLoaded {len(all_results)} models")

    print(f"Taxonomy: based on {args.taxonomy_key}, threshold: ±{args.threshold}")

    print_multi_model_table(all_results, args.taxonomy_key, args.threshold)

    # Grounding distribution analysis
    print()
    print("=" * 100)
    print("GROUNDING DISTRIBUTION ANALYSIS")
    print("=" * 100)

    for model_id in MODELS:
        if model_id not in all_results:
            continue
        r = all_results[model_id]
        # Reload to get per-case scores for distribution
        if args.metrics_json:
            path = Path(args.metrics_json)
        else:
            path = BAREXAM_OUTPUT / f"{model_id}_{args.mode_tag}-{args.version}" / args.metrics_subdir / "enhanced_metrics.json"

        with path.open() as f:
            data = json.load(f)

        # Try new format first, fall back to old
        sample = data["per_case"][0] if data["per_case"] else {}
        g_key = "grounding_max" if "grounding_max" in sample else "combined_sentence_max_score"
        scores = [c[g_key] for c in data["per_case"] if c.get(g_key) is not None]
        if scores:
            arr = np.array(scores)
            τ = args.threshold
            pct_g = np.mean(arr > τ) * 100
            pct_w = np.mean((arr >= -τ) & (arr <= τ)) * 100
            pct_c = np.mean(arr < -τ) * 100
            family, size = MODEL_DISPLAY[model_id]
            print(f"  {family:10s} {size:4s}: "
                  f"mean={np.mean(arr):.3f}  med={np.median(arr):.3f}  "
                  f"grounded={pct_g:.0f}%  weak={pct_w:.0f}%  contradicted={pct_c:.0f}%")

    # Optional JSON output
    if args.output_json:
        output = {
            "config": {
                "version": args.version,
                "taxonomy_key": args.taxonomy_key,
                "threshold": args.threshold,
            },
            "models": {},
        }
        for model_id, r in all_results.items():
            output["models"][model_id] = {
                "n_cases": r["n_cases"],
                "accuracy": {"mean": r["accuracy"][0], "ci_lower": r["accuracy"][1], "ci_upper": r["accuracy"][2]},
                "g_max": {"mean": r["g_max"][0], "ci_lower": r["g_max"][1], "ci_upper": r["g_max"][2]},
                "g_avg": {"mean": r["g_avg"][0], "ci_lower": r["g_avg"][1], "ci_upper": r["g_avg"][2]},
                "taxonomy": {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in r["taxonomy"].items()
                },
            }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to: {args.output_json}")


if __name__ == "__main__":
    main()
