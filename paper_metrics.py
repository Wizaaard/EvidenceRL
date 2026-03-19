#!/usr/bin/env python3
"""Compute publication-ready metrics from an EvidenceRL enhanced_metrics.json file.

Produces:
  - Table 1: Diagnostic accuracy (P@k, R@k, F1@k) and grounding (G_max@k, G_avg@k)
             with 95% bootstrap confidence intervals
  - Table 2: 3×2 Diagnostic Taxonomy (Grounded/Weak/Contradicted × Correct/Incorrect)
  - Derived metrics: EB%, RF%, H%, LG%, Weak%, Faithfulness
  - LaTeX-formatted output ready for copy-paste

Usage:
    python paper_metrics.py <path_to_enhanced_metrics.json> [--k 1 3 5] [--threshold 0.5]
           [--sentence-level]  # use grounding_avg instead of grounding_max for taxonomy
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute mean and bootstrap confidence interval.

    Returns (mean, ci_lower, ci_upper).
    """
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
# Taxonomy helpers
# ---------------------------------------------------------------------------

TAXONOMY_LABELS = {
    ("correct", "grounded"): "Evidence-Based",
    ("correct", "weak"): "Weakly Supported",
    ("correct", "contradicted"): "Lucky Guess",
    ("incorrect", "grounded"): "Reasoning Failure",
    ("incorrect", "weak"): "Unsupported Error",
    ("incorrect", "contradicted"): "Hallucination",
}


def classify_diagnosis(
    correct: bool,
    grounding_avg: float,
    threshold: float = 0.5,
) -> str:
    """Classify a single diagnosis into the 3×2 taxonomy."""
    correctness = "correct" if correct else "incorrect"
    if grounding_avg > threshold:
        grounding = "grounded"
    elif grounding_avg < -threshold:
        grounding = "contradicted"
    else:
        grounding = "weak"
    return TAXONOMY_LABELS[(correctness, grounding)]


# ---------------------------------------------------------------------------
# Core metric extraction
# ---------------------------------------------------------------------------

def extract_per_patient_metrics(
    patient: Dict[str, Any],
    k_values: List[int],
) -> Dict[str, Any]:
    """Extract per-patient metric values for the requested k values."""
    out = {}
    for k in k_values:
        sk = str(k)
        out[f"p@{k}"] = patient["precision_at_k"].get(sk, 0.0)
        out[f"r@{k}"] = patient["recall_at_k"].get(sk, 0.0)
        out[f"f1@{k}"] = patient["f1_at_k"].get(sk, 0.0)
        out[f"g_max@{k}"] = patient["reward_grounding_max_at_k"].get(sk, 0.0)
        out[f"g_avg@{k}"] = patient["reward_grounding_avg_at_k"].get(sk, 0.0)
    return out


def compute_taxonomy(
    patients: List[Dict[str, Any]],
    k: int,
    threshold: float,
    sentence_level: bool = False,
) -> Dict[str, Any]:
    """Compute the 3×2 diagnostic taxonomy at a given k.

    For each patient, classifies the top-k diagnoses.
    When sentence_level=True, uses grounding_avg (mean over sentences);
    otherwise uses grounding_max (best single sentence).
    """
    grounding_key = "grounding_avg" if sentence_level else "grounding_max"
    counts: Counter = Counter()
    total = 0

    for patient in patients:
        rewards = patient.get("per_diagnosis_rewards", [])
        verdicts = patient.get("verdicts", [])

        top_k = min(k, len(rewards))
        for i in range(top_k):
            correct = verdicts[i] if i < len(verdicts) else False
            g_val = rewards[i].get(grounding_key, 0.0)
            label = classify_diagnosis(correct, g_val, threshold)
            counts[label] += 1
            total += 1

    # Compute rates
    rates = {}
    for label in TAXONOMY_LABELS.values():
        rates[label] = counts[label] / total if total > 0 else 0.0

    # Derived metrics
    eb = counts["Evidence-Based"]
    ws = counts["Weakly Supported"]
    lg = counts["Lucky Guess"]
    h = counts["Hallucination"]
    ue = counts["Unsupported Error"]
    rf = counts["Reasoning Failure"]

    n_correct = eb + ws + lg
    faithfulness = eb / n_correct if n_correct > 0 else 0.0

    return {
        "counts": dict(counts),
        "rates": rates,
        "total": total,
        "eb_pct": rates["Evidence-Based"],
        "rf_pct": rates["Reasoning Failure"],
        "h_pct": rates["Hallucination"],
        "lg_pct": rates["Lucky Guess"],
        "weak_pct": rates["Weakly Supported"] + rates["Unsupported Error"],
        "faithfulness": faithfulness,
    }


# ---------------------------------------------------------------------------
# LaTeX formatting
# ---------------------------------------------------------------------------

def fmt(val: float, decimals: int = 3) -> str:
    return f"{val:.{decimals}f}"


def fmt_ci(mean: float, lo: float, hi: float, decimals: int = 3) -> str:
    return f"{mean:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}\\%"


def print_latex_table1(
    metrics_with_ci: Dict[str, Tuple[float, float, float]],
    k_values: List[int],
) -> None:
    """Print Table 1: Diagnostic accuracy and grounding."""
    print()
    print("=" * 72)
    print("TABLE 1: Diagnostic Accuracy & Evidence Grounding")
    print("=" * 72)

    # Header
    k_header = " & ".join([f"$k={k}$" for k in k_values])
    print(f"\n\\begin{{tabular}}{{l{'c' * len(k_values)}}}")
    print("\\toprule")
    print(f"Metric & {k_header} \\\\")
    print("\\midrule")

    for metric_name, display_name in [
        ("p", "Precision@$k$"),
        ("r", "Recall@$k$"),
        ("f1", "F1@$k$"),
        ("g_max", "$r_g^{\\max}$@$k$"),
        ("g_avg", "$r_g^{\\mathrm{avg}}$@$k$"),
    ]:
        cells = []
        for k in k_values:
            key = f"{metric_name}@{k}"
            mean, lo, hi = metrics_with_ci[key]
            cells.append(f"{fmt(mean)}")
        print(f"{display_name} & {' & '.join(cells)} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")

    # Also print with CIs for the main text
    print("\n--- With 95% Bootstrap CIs ---")
    for metric_name, display_name in [
        ("p", "P"),
        ("r", "R"),
        ("f1", "F1"),
        ("g_max", "G_max"),
        ("g_avg", "G_avg"),
    ]:
        for k in k_values:
            key = f"{metric_name}@{k}"
            mean, lo, hi = metrics_with_ci[key]
            print(f"  {display_name}@{k}: {fmt_ci(mean, lo, hi)}")


def print_latex_table2(taxonomy: Dict[str, Any], k: int, threshold: float) -> None:
    """Print Table 2: Diagnostic taxonomy."""
    rates = taxonomy["rates"]
    counts = taxonomy["counts"]
    total = taxonomy["total"]

    print()
    print("=" * 72)
    print(f"TABLE 2: Diagnostic Taxonomy (k={k}, threshold={threshold})")
    print("=" * 72)

    print(f"\nTotal predictions classified: {total}")
    print()

    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("& \\textbf{Correct} & \\textbf{Incorrect} \\\\")
    print("\\midrule")

    for grounding_label, gt_thresh in [
        (f"Grounded ($r_g^{{\\max}} > {threshold}$)", "grounded"),
        (f"Weak ($r_g^{{\\max}} \\in [-{threshold}, {threshold}]$)", "weak"),
        (f"Contradicted ($r_g^{{\\max}} < -{threshold}$)", "contradicted"),
    ]:
        correct_key = TAXONOMY_LABELS[("correct", gt_thresh)]
        incorrect_key = TAXONOMY_LABELS[("incorrect", gt_thresh)]
        c_count = counts.get(correct_key, 0)
        i_count = counts.get(incorrect_key, 0)
        c_pct = c_count / total * 100 if total > 0 else 0
        i_pct = i_count / total * 100 if total > 0 else 0
        print(
            f"{grounding_label} & "
            f"{correct_key} {c_pct:.1f}\\% ({c_count}) & "
            f"{incorrect_key} {i_pct:.1f}\\% ({i_count}) \\\\"
        )

    print("\\bottomrule")
    print("\\end{tabular}")

    # Key derived metrics
    print(f"\n--- Key Ratios ---")
    print(f"  Evidence-Based     (EB%):  {taxonomy['eb_pct'] * 100:.1f}%")
    print(f"  Reasoning Failure  (RF%):  {taxonomy['rf_pct'] * 100:.1f}%")
    print(f"  Hallucination      (H%):   {taxonomy['h_pct'] * 100:.1f}%")
    print(f"  Lucky Guess        (LG%):  {taxonomy['lg_pct'] * 100:.1f}%")
    print(f"  Weak               (W%):   {taxonomy['weak_pct'] * 100:.1f}%")
    print(f"  Faithfulness:              {taxonomy['faithfulness'] * 100:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute publication-ready metrics from enhanced_metrics.json"
    )
    parser.add_argument(
        "metrics_json",
        help="Path to enhanced_metrics.json",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="k values for @k metrics (default: 1 3 5)",
    )
    parser.add_argument(
        "--taxonomy-k",
        type=int,
        default=3,
        help="k for taxonomy computation (default: 3)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Grounding threshold for taxonomy (default: 0.5)",
    )
    parser.add_argument(
        "--sentence-level",
        action="store_true",
        default=False,
        help="Use grounding_avg (sentence-level mean) instead of grounding_max for taxonomy classification",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10_000,
        help="Number of bootstrap samples for CIs (default: 10000)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save computed paper metrics as JSON",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading: {args.metrics_json}")
    with open(args.metrics_json) as f:
        data = json.load(f)

    patients = data["patient_metrics"]
    config = data.get("config", {})
    n = len(patients)

    print(f"Patients: {n}")
    print(f"Judge: {config.get('judge_model', 'N/A')}")
    print(f"NLI:   {config.get('nli_model', 'N/A')}")

    # ----- Table 1: Per-patient metrics with bootstrap CIs -----
    # Collect per-patient values for each metric
    metric_values: Dict[str, List[float]] = {}
    for k in args.k:
        for prefix in ["p", "r", "f1", "g_max", "g_avg"]:
            metric_values[f"{prefix}@{k}"] = []

    for patient in patients:
        pm = extract_per_patient_metrics(patient, args.k)
        for key, val in pm.items():
            metric_values[key].append(val)

    # Compute bootstrap CIs
    metrics_with_ci: Dict[str, Tuple[float, float, float]] = {}
    for key, values in metric_values.items():
        metrics_with_ci[key] = bootstrap_ci(values, n_bootstrap=args.n_bootstrap)

    print_latex_table1(metrics_with_ci, args.k)

    # ----- Table 2: Diagnostic taxonomy -----
    grounding_mode = "grounding_avg (sentence-level)" if args.sentence_level else "grounding_max (diagnosis-level)"
    print(f"Taxonomy grounding: {grounding_mode}")
    taxonomy = compute_taxonomy(patients, k=args.taxonomy_k, threshold=args.threshold, sentence_level=args.sentence_level)
    print_latex_table2(taxonomy, k=args.taxonomy_k, threshold=args.threshold)

    # ----- Summary for abstract -----
    tk = args.taxonomy_k
    print()
    print("=" * 72)
    print(f"ABSTRACT-READY (k={tk})")
    print("=" * 72)

    # Accuracy & grounding row with ±CI
    def pm(key: str) -> str:
        """Format as XX.X ± X.X (half-width of 95% CI, percentage scale)."""
        m, lo, hi = metrics_with_ci[key]
        margin = (hi - lo) / 2
        return f"{m * 100:.1f} $\\pm$ {margin * 100:.1f}"

    header = (f"Pre@{tk}($\\uparrow$) & Rec@{tk}($\\uparrow$) & "
              f"F1@{tk}($\\uparrow$) & G$_{{max}}$@{tk}($\\uparrow$) & "
              f"G$_{{avg}}$@{tk}($\\uparrow$)")
    values = (f"{pm(f'p@{tk}')} & {pm(f'r@{tk}')} & {pm(f'f1@{tk}')} & "
              f"{pm(f'g_max@{tk}')} & {pm(f'g_avg@{tk}')}")

    print(f"\n{header} \\\\")
    print(f"{values} \\\\")

    # Taxonomy row
    tax_header = (f"Evidence Based($\\uparrow$) & Reasoning Failure($\\downarrow$) & "
                  f"Hallucination($\\downarrow$) & Lucky Guess($\\downarrow$) & "
                  f"Weak($\\downarrow$) & Faithfulness($\\uparrow$)")
    tax_values = (f"{taxonomy['eb_pct']*100:.1f}\\% & "
                  f"{taxonomy['rf_pct']*100:.1f}\\% & "
                  f"{taxonomy['h_pct']*100:.1f}\\% & "
                  f"{taxonomy['lg_pct']*100:.1f}\\% & "
                  f"{taxonomy['weak_pct']*100:.1f}\\% & "
                  f"{taxonomy['faithfulness']*100:.1f}\\%")

    print(f"\n{tax_header} \\\\")
    print(f"{tax_values} \\\\")

    # ----- Grounding distribution analysis -----
    print()
    print("=" * 72)
    print("GROUNDING DISTRIBUTION (all diagnoses)")
    print("=" * 72)
    all_g_max = []
    for patient in patients:
        for r in patient.get("per_diagnosis_rewards", []):
            all_g_max.append(r.get("grounding_max", 0.0))

    arr = np.array(all_g_max)
    print(f"  N diagnoses:  {len(arr)}")
    print(f"  Mean:         {np.mean(arr):.4f}")
    print(f"  Std:          {np.std(arr):.4f}")
    print(f"  Median:       {np.median(arr):.4f}")
    print(f"  Min / Max:    {np.min(arr):.4f} / {np.max(arr):.4f}")

    pct_grounded = np.mean(arr > args.threshold) * 100
    pct_weak = np.mean((arr >= -args.threshold) & (arr <= args.threshold)) * 100
    pct_contradicted = np.mean(arr < -args.threshold) * 100
    print(f"  Grounded (>{args.threshold}):      {pct_grounded:.1f}%")
    print(f"  Weak ([-{args.threshold},{args.threshold}]):   {pct_weak:.1f}%")
    print(f"  Contradicted (<-{args.threshold}): {pct_contradicted:.1f}%")

    # ----- Optional JSON output -----
    if args.output_json:
        output = {
            "config": {
                "source_file": args.metrics_json,
                "k_values": args.k,
                "taxonomy_k": args.taxonomy_k,
                "threshold": args.threshold,
                "n_patients": n,
            },
            "table1": {
                key: {"mean": m, "ci_lower": lo, "ci_upper": hi}
                for key, (m, lo, hi) in metrics_with_ci.items()
            },
            "taxonomy": {
                "counts": taxonomy["counts"],
                "rates": {k: round(v, 4) for k, v in taxonomy["rates"].items()},
                "total": taxonomy["total"],
                "eb_pct": round(taxonomy["eb_pct"], 4),
                "rf_pct": round(taxonomy["rf_pct"], 4),
                "h_pct": round(taxonomy["h_pct"], 4),
                "lg_pct": round(taxonomy["lg_pct"], 4),
                "weak_pct": round(taxonomy["weak_pct"], 4),
                "faithfulness": round(taxonomy["faithfulness"], 4),
            },
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved paper metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
