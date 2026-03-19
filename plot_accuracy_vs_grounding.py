#!/usr/bin/env python3
"""Plot diagnosis accuracy vs grounding reward for baseline, RAG, and EvidenceRL."""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Publication-quality settings
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['legend.fontsize'] = 9
rcParams['pdf.fonttype'] = 42

# code/plot_accuracy_vs_grounding.py → parent = evidenceRL project root
METRICS_BASE = Path(__file__).resolve().parent.parent

MODELS = ["Llama-3.1-8B", "gemma-3-12b"]
MODEL_DISPLAY = {"Llama-3.1-8B": "Llama 3.1 8B", "gemma-3-12b": "Gemma 3 12B"}

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

def load_file(path):
    with open(path, 'r') as f:
        return json.load(f)

def find_metrics_file(base_dir):
    for f in base_dir.glob("*_metrics*.json"):
        if "worker" not in f.name:
            return f
    return None

def extract_per_diagnosis(data):
    """Extract precision and grounding for each diagnosis."""
    results = []
    for patient in data.get("patient_metrics", []):
        rewards = patient.get("per_diagnosis_rewards", [])
        verdicts = patient.get("verdicts", [])

        for i, reward in enumerate(rewards):
            precision = reward.get("precision", 0.0)
            grounding = reward.get("grounding", 0.0)
            is_correct = verdicts[i] if i < len(verdicts) else (precision == 1.0)
            results.append((precision, grounding, is_correct))
    return results

# Create figure - bar chart showing grounding by correctness
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for ax_idx, model in enumerate(MODELS):
    ax = axes[ax_idx]

    conditions_data = {}

    # Load all three conditions
    paths = {
        "baseline": METRICS_BASE / f"metrics_baseline_output/{model}_v1.0",
        "baseline_rag": METRICS_BASE / f"metrics_baseline_rag_output/{model}_v1.0",
        "evidencerl": METRICS_BASE / f"metrics_dpo_output/{model}-dpo_v1.0",
    }

    for cond, path in paths.items():
        if not path.exists():
            continue
        f = find_metrics_file(path)
        if f is None:
            continue
        data = load_file(f)
        diags = extract_per_diagnosis(data)

        prec = np.array([d[0] for d in diags])
        ground = np.array([d[1] for d in diags])

        conditions_data[cond] = {
            "incorrect": np.mean(ground[prec == 0]) if sum(prec == 0) > 0 else 0,
            "correct": np.mean(ground[prec == 1]) if sum(prec == 1) > 0 else 0,
            "overall": np.mean(ground),
            "incorrect_err": np.std(ground[prec == 0])/np.sqrt(sum(prec == 0)) if sum(prec == 0) > 0 else 0,
            "correct_err": np.std(ground[prec == 1])/np.sqrt(sum(prec == 1)) if sum(prec == 1) > 0 else 0,
            "overall_err": np.std(ground)/np.sqrt(len(ground)),
        }

    # Plot grouped bars
    categories = ['Incorrect', 'Correct', 'Overall']
    x = np.arange(len(categories))
    width = 0.25

    for i, cond in enumerate(["baseline", "baseline_rag", "evidencerl"]):
        if cond not in conditions_data:
            continue
        d = conditions_data[cond]
        vals = [d["incorrect"], d["correct"], d["overall"]]
        errs = [d["incorrect_err"], d["correct_err"], d["overall_err"]]

        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, yerr=errs, label=LABELS[cond],
                     color=COLORS[cond], edgecolor='black', linewidth=0.5, capsize=3)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Diagnosis Category", fontweight='bold')
    ax.set_ylabel("Mean Grounding Score", fontweight='bold')
    ax.set_title(MODEL_DISPLAY[model], fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)

fig.suptitle("Grounding Score by Diagnosis Correctness", fontweight='bold', y=1.02)
plt.tight_layout()
FIGURE_DIR = Path(__file__).resolve().parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)
plt.savefig(FIGURE_DIR / "accuracy_vs_grounding.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {FIGURE_DIR / 'accuracy_vs_grounding.png'}")

# ============================================================================
# SCATTER PLOT VERSION - Distribution focused with violin plots
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_idx, model in enumerate(MODELS):
    ax = axes[ax_idx]

    paths = {
        "baseline": METRICS_BASE / f"metrics_baseline_output/{model}_v1.0",
        "baseline_rag": METRICS_BASE / f"metrics_baseline_rag_output/{model}_v1.0",
        "evidencerl": METRICS_BASE / f"metrics_dpo_output/{model}-dpo_v1.0",
    }

    all_data = []
    positions = []
    colors_list = []

    for i, (cond, path) in enumerate(paths.items()):
        if not path.exists():
            continue
        f = find_metrics_file(path)
        if f is None:
            continue

        data = load_file(f)
        diags = extract_per_diagnosis(data)
        ground = np.array([d[1] for d in diags])

        all_data.append(ground)
        positions.append(i)
        colors_list.append(COLORS[cond])

    # Create violin plot
    parts = ax.violinplot(all_data, positions=positions, showmeans=True, showmedians=True)

    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.7)

    # Style the lines
    for partname in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxs']:
        if partname in parts:
            parts[partname].set_color('black')
            parts[partname].set_linewidth(1.5)

    # Add individual points with jitter (only significant ones: |grounding| > 0.3)
    for i, (ground, cond) in enumerate(zip(all_data, ["baseline", "baseline_rag", "evidencerl"])):
        significant = np.abs(ground) > 0.3
        jitter = np.random.normal(0, 0.05, sum(significant))
        ax.scatter(np.full(sum(significant), i) + jitter, ground[significant],
                   alpha=0.6, s=15, c=colors_list[i], edgecolor='black', linewidth=0.3)

    # Reference line at 0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)

    # Shade regions
    ax.axhspan(-1.5, 0, alpha=0.05, color='red', label='Negative grounding')
    ax.axhspan(0, 1.5, alpha=0.05, color='green', label='Positive grounding')

    ax.set_xticks(range(3))
    ax.set_xticklabels([LABELS[c] for c in ["baseline", "baseline_rag", "evidencerl"]])
    ax.set_ylabel("Grounding Score", fontweight='bold')
    ax.set_title(MODEL_DISPLAY[model], fontweight='bold', fontsize=12)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, axis='y', alpha=0.3)

    # Add mean annotations
    for i, ground in enumerate(all_data):
        mean_val = np.mean(ground)
        ax.annotate(f'μ={mean_val:.3f}', xy=(i, mean_val), xytext=(i+0.3, mean_val+0.15),
                    fontsize=9, ha='left', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

fig.suptitle("Grounding Distribution Shift: Baseline → RAG → EvidenceRL",
             fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "accuracy_vs_grounding_violin.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {FIGURE_DIR / 'accuracy_vs_grounding_violin.png'}")

# ============================================================================
# SCATTER showing only extreme points (|grounding| > 0.3)
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for row_idx, model in enumerate(MODELS):
    paths = {
        "baseline": METRICS_BASE / f"metrics_baseline_output/{model}_v1.0",
        "baseline_rag": METRICS_BASE / f"metrics_baseline_rag_output/{model}_v1.0",
        "evidencerl": METRICS_BASE / f"metrics_dpo_output/{model}-dpo_v1.0",
    }

    for col_idx, (cond, path) in enumerate(paths.items()):
        ax = axes[row_idx, col_idx]

        if not path.exists():
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue

        f = find_metrics_file(path)
        if f is None:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue

        data = load_file(f)
        diags = extract_per_diagnosis(data)

        prec = np.array([d[0] for d in diags])
        ground = np.array([d[1] for d in diags])

        # Only show significant points (|grounding| > 0.2)
        significant = np.abs(ground) > 0.2

        # Add jitter to precision for visibility
        jitter = np.random.normal(0, 0.08, len(prec))

        # Plot all points faintly
        ax.scatter(prec + jitter, ground, alpha=0.1, s=10, c='gray')

        # Plot significant incorrect (red) and correct (green)
        incorrect_mask = (prec == 0) & significant
        correct_mask = (prec == 1) & significant

        ax.scatter(prec[incorrect_mask] + jitter[incorrect_mask], ground[incorrect_mask],
                   alpha=0.7, s=30, c='#d62728', label='Incorrect', marker='x', linewidths=1.5)
        ax.scatter(prec[correct_mask] + jitter[correct_mask], ground[correct_mask],
                   alpha=0.7, s=30, c='#2ca02c', label='Correct', marker='o', edgecolor='black', linewidth=0.5)

        # Add mean markers
        mean_all = np.mean(ground)
        ax.axhline(y=mean_all, color=COLORS[cond], linestyle='-', linewidth=2, alpha=0.8)

        # Reference line
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Incorrect', 'Correct'])

        if row_idx == 0:
            ax.set_title(LABELS[cond], fontweight='bold', fontsize=12)
        if col_idx == 0:
            ax.set_ylabel(f"{MODEL_DISPLAY[model]}\nGrounding", fontweight='bold')
        if row_idx == 1:
            ax.set_xlabel("Diagnosis Correctness", fontweight='bold')

        ax.grid(True, axis='y', alpha=0.3)

        # Stats annotation
        n_sig = sum(significant)
        n_total = len(ground)
        pct_positive = sum(ground > 0) / len(ground) * 100
        ax.text(0.95, 0.95, f'μ={mean_all:.3f}\n{pct_positive:.0f}% positive\nn={n_total}',
                transform=ax.transAxes, fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='x', color='w', markerfacecolor='#d62728',
           markeredgecolor='#d62728', markersize=8, label='Incorrect (|g|>0.2)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
           markersize=8, label='Correct (|g|>0.2)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=6, alpha=0.3, label='All diagnoses'),
    Line2D([0], [0], linestyle='-', color='gray', linewidth=2, label='Mean'),
]
fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99))

fig.suptitle("Diagnosis Grounding: Significant Points Only (|grounding| > 0.2)",
             fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "accuracy_vs_grounding_scatter.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {FIGURE_DIR / 'accuracy_vs_grounding_scatter.png'}")

# Print statistics
print("\n" + "=" * 80)
print("ACCURACY vs GROUNDING STATISTICS")
print("=" * 80)

for model in MODELS:
    print(f"\n{MODEL_DISPLAY[model]}:")
    print("-" * 70)
    print(f"  {'Condition':<15} {'Incorrect':>12} {'Correct':>12} {'Overall':>12}")
    print("  " + "-" * 55)

    paths = {
        "baseline": METRICS_BASE / f"metrics_baseline_output/{model}_v1.0",
        "baseline_rag": METRICS_BASE / f"metrics_baseline_rag_output/{model}_v1.0",
        "evidencerl": METRICS_BASE / f"metrics_dpo_output/{model}-dpo_v1.0",
    }

    for cond, path in paths.items():
        if not path.exists():
            continue
        f = find_metrics_file(path)
        if f is None:
            continue
        data = load_file(f)
        diags = extract_per_diagnosis(data)

        prec = np.array([d[0] for d in diags])
        ground = np.array([d[1] for d in diags])

        inc = np.mean(ground[prec == 0]) if sum(prec == 0) > 0 else 0
        cor = np.mean(ground[prec == 1]) if sum(prec == 1) > 0 else 0
        ovr = np.mean(ground)

        print(f"  {LABELS[cond]:<15} {inc:>12.4f} {cor:>12.4f} {ovr:>12.4f}")
