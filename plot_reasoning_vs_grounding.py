#!/usr/bin/env python3
"""
Plot reasoning length vs grounding score.

Creates scatter plots showing the relationship between reasoning length
and grounding reward at the per-diagnosis level.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

# Publication-quality settings
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['legend.fontsize'] = 9
rcParams['pdf.fonttype'] = 42

# Paths — code/plot_reasoning_vs_grounding.py → parent = evidenceRL project root
METRICS_BASE = Path(__file__).resolve().parent.parent

METRICS_PATHS = {
    "baseline": METRICS_BASE / "metrics_baseline_output",
    "baseline_rag": METRICS_BASE / "metrics_baseline_rag_output",
    "evidencerl": METRICS_BASE / "metrics_dpo_output",
}

# Models with EvidenceRL results
MODELS_WITH_ERL = ["Llama-3.1-8B", "gemma-3-4b", "gemma-3-12b"]

# Colors (colorblind-friendly)
COLORS = {
    "baseline": "#E69F00",      # Orange
    "baseline_rag": "#56B4E9",  # Sky blue
    "evidencerl": "#009E73",    # Teal
}

LABELS = {
    "baseline": "Baseline",
    "baseline_rag": "Baseline + RAG",
    "evidencerl": "EvidenceRL",
}

MODEL_DISPLAY = {
    "Llama-3.1-8B": "Llama 3.1 8B",
    "gemma-3-4b": "Gemma 3 4B",
    "gemma-3-12b": "Gemma 3 12B",
}


def load_metrics_file(filepath: Path) -> dict:
    """Load metrics JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_diagnosis_data(data: dict) -> List[Tuple[int, float, bool]]:
    """
    Extract (reasoning_length, grounding, is_correct) for each diagnosis.
    """
    results = []

    for patient in data.get("patient_metrics", []):
        diagnoses_with_reasoning = patient.get("predicted_diagnoses_with_reasoning", [])
        per_diagnosis_rewards = patient.get("per_diagnosis_rewards", [])
        verdicts = patient.get("verdicts", [])

        for i, diag in enumerate(diagnoses_with_reasoning):
            reasoning = diag.get("reasoning", "")
            reasoning_len = len(reasoning)

            # Get grounding score
            if i < len(per_diagnosis_rewards):
                grounding = per_diagnosis_rewards[i].get("grounding", 0.0)
            else:
                grounding = 0.0

            # Get correctness
            is_correct = verdicts[i] if i < len(verdicts) else False

            # Only include if there's actual reasoning
            if reasoning_len > 0:
                results.append((reasoning_len, grounding, is_correct))

    return results


def find_metrics_file(base_dir: Path, model_id: str, condition: str) -> Path:
    """Find the merged metrics file for a model."""
    model_dir = base_dir / f"{model_id}_v1.0"

    # Try different naming patterns
    patterns = [
        f"{model_id}_{condition}_metrics-v1.0.json",
        f"{model_id}_dpo_metrics-v1.0.json",
        f"{model_id}_baseline_metrics-v1.0.json",
        f"{model_id}_baseline_rag_metrics-v1.0.json",
        f"{model_id}_metrics-v1.0.json",
    ]

    for pattern in patterns:
        filepath = model_dir / pattern
        if filepath.exists():
            return filepath

    # Try glob
    for f in model_dir.glob("*_metrics*.json"):
        if "worker" not in f.name:
            return f

    return None


def plot_reasoning_vs_grounding_single(
    model_id: str,
    output_path: Path,
    figsize: Tuple[float, float] = (10, 4),
):
    """Create reasoning vs grounding plot for a single model across conditions."""

    conditions = ["baseline", "baseline_rag", "evidencerl"]

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    display_name = MODEL_DISPLAY.get(model_id, model_id)

    for ax_idx, condition in enumerate(conditions):
        ax = axes[ax_idx]

        # Handle DPO model naming
        if condition == "evidencerl":
            search_id = f"{model_id}-dpo"
        else:
            search_id = model_id

        base_dir = METRICS_PATHS[condition]
        filepath = find_metrics_file(base_dir, search_id, condition)

        if filepath is None or not filepath.exists():
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{LABELS[condition]}", fontweight='bold')
            continue

        data = load_metrics_file(filepath)
        diagnosis_data = extract_diagnosis_data(data)

        if not diagnosis_data:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{LABELS[condition]}", fontweight='bold')
            continue

        lengths = np.array([d[0] for d in diagnosis_data])
        groundings = np.array([d[1] for d in diagnosis_data])
        correct = np.array([d[2] for d in diagnosis_data])

        # Plot correct and incorrect with different markers
        ax.scatter(lengths[correct], groundings[correct],
                   alpha=0.5, s=20, c='#2ca02c', label='Correct', marker='o')
        ax.scatter(lengths[~correct], groundings[~correct],
                   alpha=0.5, s=20, c='#d62728', label='Incorrect', marker='x')

        # Add regression line
        if len(lengths) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(lengths, groundings)
            x_line = np.linspace(lengths.min(), lengths.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=COLORS[condition], linewidth=2,
                   label=f'r={r_value:.3f}')

        ax.set_xlabel("Reasoning Length (chars)", fontweight='bold')
        if ax_idx == 0:
            ax.set_ylabel("Grounding Score", fontweight='bold')
        ax.set_title(f"{LABELS[condition]}", fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Stats annotation
        mean_len = np.mean(lengths)
        mean_ground = np.mean(groundings)
        ax.text(0.05, 0.95, f'μ len={mean_len:.0f}\nμ gr={mean_ground:.3f}',
               transform=ax.transAxes, fontsize=8, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(f"{display_name}: Reasoning Length vs Grounding", fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_reasoning_vs_grounding_combined(
    output_path: Path,
    figsize: Tuple[float, float] = (12, 8),
):
    """Create combined plot for all models with EvidenceRL."""

    fig, axes = plt.subplots(len(MODELS_WITH_ERL), 3, figsize=figsize,
                             sharex='col', sharey='row')

    conditions = ["baseline", "baseline_rag", "evidencerl"]

    for row_idx, model_id in enumerate(MODELS_WITH_ERL):
        display_name = MODEL_DISPLAY.get(model_id, model_id)

        for col_idx, condition in enumerate(conditions):
            ax = axes[row_idx, col_idx]

            # Handle DPO model naming
            if condition == "evidencerl":
                search_id = f"{model_id}-dpo"
            else:
                search_id = model_id

            base_dir = METRICS_PATHS[condition]
            filepath = find_metrics_file(base_dir, search_id, condition)

            if filepath is None or not filepath.exists():
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                if row_idx == 0:
                    ax.set_title(f"{LABELS[condition]}", fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(f"{display_name}\nGrounding", fontweight='bold')
                continue

            data = load_metrics_file(filepath)
            diagnosis_data = extract_diagnosis_data(data)

            if not diagnosis_data:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                continue

            lengths = np.array([d[0] for d in diagnosis_data])
            groundings = np.array([d[1] for d in diagnosis_data])
            correct = np.array([d[2] for d in diagnosis_data])

            # Plot
            ax.scatter(lengths[correct], groundings[correct],
                       alpha=0.4, s=15, c='#2ca02c', marker='o')
            ax.scatter(lengths[~correct], groundings[~correct],
                       alpha=0.4, s=15, c='#d62728', marker='x')

            # Regression line
            if len(lengths) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(lengths, groundings)
                x_line = np.linspace(0, max(lengths.max(), 1000), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color=COLORS[condition], linewidth=2)

                # Correlation annotation
                ax.text(0.95, 0.05, f'r={r_value:.2f}',
                       transform=ax.transAxes, fontsize=9, ha='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.grid(True, alpha=0.3)

            # Labels
            if row_idx == 0:
                ax.set_title(f"{LABELS[condition]}", fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"{display_name}\nGrounding", fontweight='bold')
            if row_idx == len(MODELS_WITH_ERL) - 1:
                ax.set_xlabel("Reasoning Length (chars)", fontweight='bold')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markersize=8, label='Correct'),
        Line2D([0], [0], marker='x', color='#d62728', markersize=8,
               label='Incorrect', linestyle='None'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    fig.suptitle("Reasoning Length vs Grounding Score (per diagnosis)",
                 fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Plotting Reasoning Length vs Grounding")
    print("=" * 60)

    # Combined plot for all models
    plot_reasoning_vs_grounding_combined(
        output_path=output_dir / "rq4_reasoning_length.png"
    )

    # Individual plots per model
    for model_id in MODELS_WITH_ERL:
        safe_name = model_id.replace("-", "_").replace(".", "")
        plot_reasoning_vs_grounding_single(
            model_id=model_id,
            output_path=output_dir / f"reasoning_vs_grounding_{safe_name}.png"
        )

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
