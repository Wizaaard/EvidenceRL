#!/usr/bin/env python3
"""Plot revision results showing grounding and verdict changes.

This script generates visualization plots for the revision pipeline results,
comparing original vs revised diagnoses.

Usage:
    python plot_revision_results.py --metrics-json <path> --output-dir <dir>
    python plot_revision_results.py --metrics-json merged_metrics.json --output-dir plots/

Plots generated:
1. grounding_scatter.png - Scatter plot of original vs revised grounding scores
2. grounding_distribution.png - Distribution of grounding scores before/after revision
3. revision_summary.png - Summary statistics (counts, case types)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def check_dependencies() -> None:
    """Check that required dependencies are available."""
    if not MATPLOTLIB_AVAILABLE:
        print("ERROR: matplotlib is required for plotting. Install with: pip install matplotlib")
        sys.exit(1)
    if not NUMPY_AVAILABLE:
        print("ERROR: numpy is required for plotting. Install with: pip install numpy")
        sys.exit(1)


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Load metrics JSON file."""
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_diagnosis_data(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract per-diagnosis data from metrics.

    Returns list of dicts with:
    - hadm_id: patient ID
    - diagnosis_idx: index within patient
    - diagnosis_name: name of diagnosis
    - grounding: current grounding score
    - grounding_old: original grounding score (if available)
    - verdict: True if correct, False if incorrect
    - revised: True if this diagnosis was revised
    """
    diagnosis_data = []

    patient_metrics = metrics.get("patient_metrics", [])

    for patient in patient_metrics:
        hadm_id = patient.get("hadm_id", "unknown")
        verdicts = patient.get("verdicts", [])
        per_diagnosis_rewards = patient.get("per_diagnosis_rewards", [])
        predicted_diagnoses = patient.get("predicted_diagnoses_with_reasoning", [])

        for idx, reward_data in enumerate(per_diagnosis_rewards):
            grounding = reward_data.get("grounding", 0.0)
            grounding_old = reward_data.get("grounding_old", grounding)  # Default to current if no old
            revised = reward_data.get("revised", False)
            verdict = verdicts[idx] if idx < len(verdicts) else False

            # Get diagnosis name if available
            diagnosis_name = ""
            if idx < len(predicted_diagnoses):
                diagnosis_name = predicted_diagnoses[idx].get("diagnosis", f"Diagnosis {idx+1}")

            diagnosis_data.append({
                "hadm_id": hadm_id,
                "diagnosis_idx": idx,
                "diagnosis_name": diagnosis_name,
                "grounding": grounding,
                "grounding_old": grounding_old,
                "verdict": verdict,
                "revised": revised,
            })

    return diagnosis_data


def plot_grounding_scatter(
    diagnosis_data: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Grounding: Original vs Revised",
) -> Tuple[int, int]:
    """Create scatter plot of original vs revised grounding scores.

    Returns (total_revised, total_unchanged) counts.
    """
    # Separate revised and non-revised diagnoses
    revised_data = [d for d in diagnosis_data if d["revised"]]
    unchanged_data = [d for d in diagnosis_data if not d["revised"]]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot unchanged diagnoses (gray, smaller)
    if unchanged_data:
        x_unchanged = [d["grounding_old"] for d in unchanged_data]
        y_unchanged = [d["grounding"] for d in unchanged_data]
        ax.scatter(x_unchanged, y_unchanged, c='lightgray', alpha=0.3, s=20, label=f'Unchanged (n={len(unchanged_data)})')

    # Plot revised diagnoses by verdict
    revised_correct = [d for d in revised_data if d["verdict"]]
    revised_incorrect = [d for d in revised_data if not d["verdict"]]

    if revised_correct:
        x_correct = [d["grounding_old"] for d in revised_correct]
        y_correct = [d["grounding"] for d in revised_correct]
        ax.scatter(x_correct, y_correct, c='green', alpha=0.7, s=50,
                   label=f'Revised Correct (Case A, n={len(revised_correct)})', marker='o')

    if revised_incorrect:
        x_incorrect = [d["grounding_old"] for d in revised_incorrect]
        y_incorrect = [d["grounding"] for d in revised_incorrect]
        ax.scatter(x_incorrect, y_incorrect, c='red', alpha=0.7, s=50,
                   label=f'Revised Incorrect (Case B, n={len(revised_incorrect)})', marker='x')

    # Add diagonal line (no change)
    ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.5, label='No change line')

    # Add horizontal lines for ambiguous region
    ax.axhline(y=0.25, color='orange', linestyle=':', alpha=0.5)
    ax.axhline(y=-0.25, color='orange', linestyle=':', alpha=0.5)
    ax.axvline(x=0.25, color='orange', linestyle=':', alpha=0.5)
    ax.axvline(x=-0.25, color='orange', linestyle=':', alpha=0.5)

    # Add shaded ambiguous region
    ax.axhspan(-0.25, 0.25, alpha=0.1, color='orange', label='Ambiguous region')

    ax.set_xlabel('Original Grounding Score', fontsize=12)
    ax.set_ylabel('Revised Grounding Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[plot] Saved grounding scatter plot to: {output_path}")
    return len(revised_data), len(unchanged_data)


def plot_grounding_distribution(
    diagnosis_data: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Grounding Score Distribution",
) -> None:
    """Create distribution plot comparing original vs revised grounding scores."""
    # Get only revised diagnoses for comparison
    revised_data = [d for d in diagnosis_data if d["revised"]]

    if not revised_data:
        print("[plot] No revised diagnoses found, skipping distribution plot")
        return

    original_scores = [d["grounding_old"] for d in revised_data]
    revised_scores = [d["grounding"] for d in revised_data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram comparison
    ax1 = axes[0]
    bins = np.linspace(-1, 1, 21)
    ax1.hist(original_scores, bins=bins, alpha=0.6, label='Original', color='blue', edgecolor='black')
    ax1.hist(revised_scores, bins=bins, alpha=0.6, label='Revised', color='green', edgecolor='black')
    ax1.axvline(x=0.25, color='orange', linestyle='--', label='Ambiguous boundary')
    ax1.axvline(x=-0.25, color='orange', linestyle='--')
    ax1.set_xlabel('Grounding Score', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Grounding Scores (Revised Diagnoses)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot comparison by verdict
    ax2 = axes[1]

    revised_correct = [d for d in revised_data if d["verdict"]]
    revised_incorrect = [d for d in revised_data if not d["verdict"]]

    box_data = []
    box_labels = []
    box_colors = []

    if revised_correct:
        box_data.append([d["grounding_old"] for d in revised_correct])
        box_labels.append(f'Case A\nOriginal\n(n={len(revised_correct)})')
        box_colors.append('lightblue')

        box_data.append([d["grounding"] for d in revised_correct])
        box_labels.append(f'Case A\nRevised')
        box_colors.append('green')

    if revised_incorrect:
        box_data.append([d["grounding_old"] for d in revised_incorrect])
        box_labels.append(f'Case B\nOriginal\n(n={len(revised_incorrect)})')
        box_colors.append('lightyellow')

        box_data.append([d["grounding"] for d in revised_incorrect])
        box_labels.append(f'Case B\nRevised')
        box_colors.append('red')

    if box_data:
        bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5)
        ax2.axhline(y=-0.25, color='orange', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Grounding Score', fontsize=12)
        ax2.set_title('Grounding by Case Type (Before/After)', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[plot] Saved grounding distribution plot to: {output_path}")


def plot_revision_summary(
    diagnosis_data: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_path: Path,
    title: str = "Revision Summary",
) -> None:
    """Create summary plot showing revision statistics."""
    # Get revision stats from metrics if available
    revision_stats = metrics.get("revision_stats", {})

    # Calculate stats from data
    total_diagnoses = len(diagnosis_data)
    revised_data = [d for d in diagnosis_data if d["revised"]]
    total_revised = len(revised_data)

    case_a_count = len([d for d in revised_data if d["verdict"]])
    case_b_count = len([d for d in revised_data if not d["verdict"]])

    # Calculate grounding changes
    grounding_changes = [d["grounding"] - d["grounding_old"] for d in revised_data]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Revision counts
    ax1 = axes[0, 0]
    categories = ['Total\nDiagnoses', 'Revised', 'Case A\n(Correct)', 'Case B\n(Incorrect)']
    counts = [total_diagnoses, total_revised, case_a_count, case_b_count]
    colors = ['steelblue', 'orange', 'green', 'red']
    bars = ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Diagnosis Counts', fontsize=12)
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Revision rate pie chart
    ax2 = axes[0, 1]
    if total_revised > 0:
        pie_data = [case_a_count, case_b_count]
        pie_labels = [f'Case A (Well-grounded)\n{case_a_count} ({100*case_a_count/total_revised:.1f}%)',
                      f'Case B (Contradictory)\n{case_b_count} ({100*case_b_count/total_revised:.1f}%)']
        pie_colors = ['green', 'red']
        result = ax2.pie(pie_data, labels=pie_labels, colors=pie_colors, startangle=90)
        wedges = result[0]  # First element is always wedges
        for wedge in wedges:
            wedge.set_alpha(0.7)
        ax2.set_title(f'Revised Diagnoses by Case Type\n(Total: {total_revised})', fontsize=12)
    else:
        ax2.text(0.5, 0.5, 'No revisions', ha='center', va='center', fontsize=14)
        ax2.set_title('Revised Diagnoses by Case Type', fontsize=12)

    # Plot 3: Grounding change histogram
    ax3 = axes[1, 0]
    if grounding_changes:
        ax3.hist(grounding_changes, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        mean_change = np.mean(grounding_changes)
        ax3.axvline(x=mean_change, color='red', linestyle='-', alpha=0.7,
                    label=f'Mean: {mean_change:.3f}')
        ax3.set_xlabel('Grounding Change (Revised - Original)', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_title('Distribution of Grounding Changes', fontsize=12)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No revisions', ha='center', va='center', fontsize=14)
        ax3.set_title('Distribution of Grounding Changes', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Statistics text
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Calculate statistics
    stats_text = f"""
REVISION SUMMARY STATISTICS
{'='*40}

Total Diagnoses:        {total_diagnoses:,}
Total Revised:          {total_revised:,} ({100*total_revised/total_diagnoses:.1f}%)

Case A (Correct):       {case_a_count:,} ({100*case_a_count/max(total_revised,1):.1f}% of revised)
Case B (Incorrect):     {case_b_count:,} ({100*case_b_count/max(total_revised,1):.1f}% of revised)
"""

    if grounding_changes:
        stats_text += f"""
GROUNDING CHANGES (Revised Only)
{'='*40}
Mean Change:            {np.mean(grounding_changes):+.4f}
Std Dev:                {np.std(grounding_changes):.4f}
Min Change:             {np.min(grounding_changes):+.4f}
Max Change:             {np.max(grounding_changes):+.4f}

Improved (>0):          {sum(1 for c in grounding_changes if c > 0):,}
Worsened (<0):          {sum(1 for c in grounding_changes if c < 0):,}
Unchanged (=0):         {sum(1 for c in grounding_changes if c == 0):,}
"""

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[plot] Saved revision summary plot to: {output_path}")


def plot_grounding_verdict_scatter(
    diagnosis_data: List[Dict[str, Any]],
    output_dir: Path,
    title_prefix: str = "",
) -> None:
    """Create two scatter plots: grounding (X) vs verdict (Y) for pre and post revision.

    This shows how revision improves the separation between correct (verdict=1)
    and incorrect (verdict=0) diagnoses based on grounding scores.
    """
    revised_data = [d for d in diagnosis_data if d["revised"]]

    if not revised_data:
        print("[plot] No revised diagnoses found, skipping grounding-verdict scatter plots")
        return

    # Add jitter to verdict for visibility (small random offset)
    np.random.seed(42)  # For reproducibility
    jitter_amount = 0.08

    # Extract data
    grounding_old = np.array([d["grounding_old"] for d in revised_data])
    grounding_new = np.array([d["grounding"] for d in revised_data])
    verdicts = np.array([1 if d["verdict"] else 0 for d in revised_data])
    verdict_jittered = verdicts + np.random.uniform(-jitter_amount, jitter_amount, len(verdicts))

    # Create PRE-revision scatter plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # Color by verdict
    colors = ['green' if v == 1 else 'red' for v in verdicts]
    ax1.scatter(grounding_old, verdict_jittered, c=colors, alpha=0.6, s=40, edgecolors='black', linewidths=0.5)

    # Add reference lines
    ax1.axvline(x=0.25, color='orange', linestyle='--', alpha=0.7, label='Ambiguous boundary')
    ax1.axvline(x=-0.25, color='orange', linestyle='--', alpha=0.7)
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    # Labels and formatting
    ax1.set_xlabel('Grounding Score (Original)', fontsize=12)
    ax1.set_ylabel('Verdict (0=Incorrect, 1=Correct)', fontsize=12)
    ax1.set_title(f'{title_prefix}PRE-Revision: Grounding vs Verdict', fontsize=14)
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Incorrect (0)', 'Correct (1)'])
    ax1.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Correct diagnosis'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Incorrect diagnosis'),
    ]
    ax1.legend(handles=legend_elements, loc='center right')

    # Add annotation for expected pattern
    ax1.text(0.7, 0.85, 'Expected:\nCorrect → High grounding', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax1.text(-0.9, 0.15, 'Expected:\nIncorrect → Low grounding', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_dir / "grounding_verdict_pre.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot] Saved PRE-revision scatter to: {output_dir / 'grounding_verdict_pre.png'}")

    # Create POST-revision scatter plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.scatter(grounding_new, verdict_jittered, c=colors, alpha=0.6, s=40, edgecolors='black', linewidths=0.5)

    # Add reference lines
    ax2.axvline(x=0.25, color='orange', linestyle='--', alpha=0.7, label='Ambiguous boundary')
    ax2.axvline(x=-0.25, color='orange', linestyle='--', alpha=0.7)
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    # Labels and formatting
    ax2.set_xlabel('Grounding Score (Revised)', fontsize=12)
    ax2.set_ylabel('Verdict (0=Incorrect, 1=Correct)', fontsize=12)
    ax2.set_title(f'{title_prefix}POST-Revision: Grounding vs Verdict', fontsize=14)
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Incorrect (0)', 'Correct (1)'])
    ax2.grid(True, alpha=0.3)

    ax2.legend(handles=legend_elements, loc='center right')

    # Add annotation for expected pattern
    ax2.text(0.7, 0.85, 'Expected:\nCorrect → High grounding', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax2.text(-0.9, 0.15, 'Expected:\nIncorrect → Low grounding', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_dir / "grounding_verdict_post.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot] Saved POST-revision scatter to: {output_dir / 'grounding_verdict_post.png'}")


def plot_verdict_comparison(
    diagnosis_data: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Verdict vs Grounding",
) -> None:
    """Create plot showing verdict (correct/incorrect) vs grounding scores."""
    revised_data = [d for d in diagnosis_data if d["revised"]]

    if not revised_data:
        print("[plot] No revised diagnoses found, skipping verdict comparison plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Original grounding by verdict
    ax1 = axes[0]
    correct_original = [d["grounding_old"] for d in revised_data if d["verdict"]]
    incorrect_original = [d["grounding_old"] for d in revised_data if not d["verdict"]]

    positions_orig = [1, 2]
    data_orig = [correct_original, incorrect_original] if correct_original and incorrect_original else []

    if data_orig:
        parts1 = ax1.violinplot(data_orig, positions=positions_orig, showmeans=True, showmedians=True)
        parts1['bodies'][0].set_facecolor('green')
        parts1['bodies'][0].set_alpha(0.5)
        if len(parts1['bodies']) > 1:
            parts1['bodies'][1].set_facecolor('red')
            parts1['bodies'][1].set_alpha(0.5)

    ax1.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='Ambiguous boundary')
    ax1.axhline(y=-0.25, color='orange', linestyle='--', alpha=0.5)
    ax1.set_xticks(positions_orig)
    ax1.set_xticklabels([f'Correct\n(n={len(correct_original)})', f'Incorrect\n(n={len(incorrect_original)})'])
    ax1.set_ylabel('Grounding Score', fontsize=12)
    ax1.set_title('ORIGINAL Grounding by Verdict', fontsize=12)
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Revised grounding by verdict
    ax2 = axes[1]
    correct_revised = [d["grounding"] for d in revised_data if d["verdict"]]
    incorrect_revised = [d["grounding"] for d in revised_data if not d["verdict"]]

    positions_rev = [1, 2]
    data_rev = [correct_revised, incorrect_revised] if correct_revised and incorrect_revised else []

    if data_rev:
        parts2 = ax2.violinplot(data_rev, positions=positions_rev, showmeans=True, showmedians=True)
        parts2['bodies'][0].set_facecolor('green')
        parts2['bodies'][0].set_alpha(0.5)
        if len(parts2['bodies']) > 1:
            parts2['bodies'][1].set_facecolor('red')
            parts2['bodies'][1].set_alpha(0.5)

    ax2.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='Ambiguous boundary')
    ax2.axhline(y=-0.25, color='orange', linestyle='--', alpha=0.5)
    ax2.set_xticks(positions_rev)
    ax2.set_xticklabels([f'Correct\n(n={len(correct_revised)})', f'Incorrect\n(n={len(incorrect_revised)})'])
    ax2.set_ylabel('Grounding Score', fontsize=12)
    ax2.set_title('REVISED Grounding by Verdict', fontsize=12)
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add legend
    green_patch = mpatches.Patch(color='green', alpha=0.5, label='Correct (Case A: well-grounded)')
    red_patch = mpatches.Patch(color='red', alpha=0.5, label='Incorrect (Case B: contradictory)')
    fig.legend(handles=[green_patch, red_patch], loc='upper center', ncol=2, fontsize=10)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[plot] Saved verdict comparison plot to: {output_path}")


def generate_all_plots(
    metrics_path: Path,
    output_dir: Path,
    title_prefix: str = "",
) -> Dict[str, Any]:
    """Generate all revision plots.

    Returns statistics dict with:
    - total_diagnoses
    - total_revised
    - case_a_count
    - case_b_count
    """
    check_dependencies()

    # Load data
    print(f"[plot] Loading metrics from: {metrics_path}")
    metrics = load_metrics(metrics_path)

    # Extract diagnosis data
    diagnosis_data = extract_diagnosis_data(metrics)
    print(f"[plot] Found {len(diagnosis_data)} total diagnoses")

    revised_count = sum(1 for d in diagnosis_data if d["revised"])
    print(f"[plot] Found {revised_count} revised diagnoses")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    prefix = f"{title_prefix} - " if title_prefix else ""

    # 1. Grounding scatter plot
    total_revised, total_unchanged = plot_grounding_scatter(
        diagnosis_data,
        output_dir / "grounding_scatter.png",
        title=f"{prefix}Grounding: Original vs Revised",
    )

    # 2. Grounding distribution plot
    plot_grounding_distribution(
        diagnosis_data,
        output_dir / "grounding_distribution.png",
        title=f"{prefix}Grounding Score Distribution",
    )

    # 3. Revision summary plot
    plot_revision_summary(
        diagnosis_data,
        metrics,
        output_dir / "revision_summary.png",
        title=f"{prefix}Revision Summary",
    )

    # 4. Verdict comparison plot
    plot_verdict_comparison(
        diagnosis_data,
        output_dir / "verdict_comparison.png",
        title=f"{prefix}Verdict vs Grounding",
    )

    # 5. Grounding vs Verdict scatter plots (pre and post)
    plot_grounding_verdict_scatter(
        diagnosis_data,
        output_dir,
        title_prefix=f"{prefix}" if prefix else "",
    )

    # Calculate return stats
    revised_data = [d for d in diagnosis_data if d["revised"]]
    stats = {
        "total_diagnoses": len(diagnosis_data),
        "total_revised": len(revised_data),
        "case_a_count": len([d for d in revised_data if d["verdict"]]),
        "case_b_count": len([d for d in revised_data if not d["verdict"]]),
        "plots_generated": [
            str(output_dir / "grounding_scatter.png"),
            str(output_dir / "grounding_distribution.png"),
            str(output_dir / "revision_summary.png"),
            str(output_dir / "verdict_comparison.png"),
            str(output_dir / "grounding_verdict_pre.png"),
            str(output_dir / "grounding_verdict_post.png"),
        ],
    }

    print(f"[plot] All plots saved to: {output_dir}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Plot revision results showing grounding and verdict changes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python plot_revision_results.py --metrics-json merged_metrics.json --output-dir plots/
    python plot_revision_results.py --metrics-json metrics.json --output-dir plots/ --title "Model A"
        """,
    )
    parser.add_argument(
        "--metrics-json",
        required=True,
        help="Path to metrics JSON file (merged_metrics.json or metrics_worker*.json)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save plot images",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional title prefix for plots",
    )

    args = parser.parse_args()

    metrics_path = Path(args.metrics_json)
    output_dir = Path(args.output_dir)

    if not metrics_path.exists():
        print(f"ERROR: Metrics file not found: {metrics_path}")
        sys.exit(1)

    stats = generate_all_plots(metrics_path, output_dir, args.title)

    print("\n" + "="*50)
    print("PLOTTING COMPLETE")
    print("="*50)
    print(f"Total diagnoses:  {stats['total_diagnoses']}")
    print(f"Total revised:    {stats['total_revised']}")
    print(f"Case A (correct): {stats['case_a_count']}")
    print(f"Case B (incorrect): {stats['case_b_count']}")
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
