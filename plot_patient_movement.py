#!/usr/bin/env python3
"""Plot patient-level grounding movement across conditions."""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import defaultdict

# Publication-quality settings
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['legend.fontsize'] = 9
rcParams['pdf.fonttype'] = 42

# code/plot_patient_movement.py → parent = evidenceRL project root
METRICS_BASE = Path(__file__).resolve().parent.parent

MODELS = ["Llama-3.1-8B", "gemma-3-12b"]
MODEL_DISPLAY = {"Llama-3.1-8B": "Llama 3.1 8B", "gemma-3-12b": "Gemma 3 12B"}

COLORS = {
    "baseline": "#E69F00",
    "baseline_rag": "#56B4E9",
    "evidencerl": "#009E73",
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

def extract_patient_grounding(data):
    """Extract average grounding per patient."""
    patient_grounding = {}

    for patient in data.get("patient_metrics", []):
        hadm_id = patient.get("hadm_id", "")
        rewards = patient.get("per_diagnosis_rewards", [])

        if rewards:
            avg_grounding = np.mean([r.get("grounding", 0.0) for r in rewards])
            patient_grounding[hadm_id] = avg_grounding

    return patient_grounding

# ============================================================================
# PATIENT MOVEMENT PLOT - Lines connecting same patient across conditions
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

SIGNIFICANT_THRESHOLD = 0.15  # Movement threshold to highlight

for ax_idx, model in enumerate(MODELS):
    ax = axes[ax_idx]

    paths = {
        "baseline": METRICS_BASE / f"metrics_baseline_output/{model}_v1.0",
        "baseline_rag": METRICS_BASE / f"metrics_baseline_rag_output/{model}_v1.0",
        "evidencerl": METRICS_BASE / f"metrics_dpo_output/{model}-dpo_v1.0",
    }

    # Load patient grounding for each condition
    condition_data = {}
    for cond, path in paths.items():
        if not path.exists():
            continue
        f = find_metrics_file(path)
        if f is None:
            continue
        data = load_file(f)
        condition_data[cond] = extract_patient_grounding(data)

    # Find patients present in all conditions
    all_patients = set(condition_data.get("baseline", {}).keys())
    all_patients &= set(condition_data.get("baseline_rag", {}).keys())
    all_patients &= set(condition_data.get("evidencerl", {}).keys())

    print(f"\n{MODEL_DISPLAY[model]}: {len(all_patients)} patients in all conditions")

    # Categorize patients by movement
    improved_significant = []  # grounding improved > threshold
    worsened_significant = []  # grounding worsened > threshold
    stable = []  # minimal change

    for patient_id in all_patients:
        baseline_g = condition_data["baseline"].get(patient_id, 0)
        rag_g = condition_data["baseline_rag"].get(patient_id, 0)
        erl_g = condition_data["evidencerl"].get(patient_id, 0)

        total_change = erl_g - baseline_g

        if total_change > SIGNIFICANT_THRESHOLD:
            improved_significant.append((patient_id, baseline_g, rag_g, erl_g))
        elif total_change < -SIGNIFICANT_THRESHOLD:
            worsened_significant.append((patient_id, baseline_g, rag_g, erl_g))
        else:
            stable.append((patient_id, baseline_g, rag_g, erl_g))

    print(f"  Improved significantly: {len(improved_significant)}")
    print(f"  Worsened significantly: {len(worsened_significant)}")
    print(f"  Stable: {len(stable)}")

    # Plot positions
    x_pos = [0, 1, 2]

    # Plot stable patients (gray, faint)
    for patient_id, b, r, e in stable:
        ax.plot(x_pos, [b, r, e], color='gray', alpha=0.15, linewidth=0.5)

    # Plot worsened patients (red)
    for patient_id, b, r, e in worsened_significant:
        ax.plot(x_pos, [b, r, e], color='#d62728', alpha=0.5, linewidth=1)

    # Plot improved patients (green)
    for patient_id, b, r, e in improved_significant:
        ax.plot(x_pos, [b, r, e], color='#2ca02c', alpha=0.5, linewidth=1)

    # Plot mean trajectory
    mean_baseline = np.mean([condition_data["baseline"][p] for p in all_patients])
    mean_rag = np.mean([condition_data["baseline_rag"][p] for p in all_patients])
    mean_erl = np.mean([condition_data["evidencerl"][p] for p in all_patients])

    ax.plot(x_pos, [mean_baseline, mean_rag, mean_erl],
            color='black', linewidth=3, marker='o', markersize=10,
            label=f'Mean trajectory', zorder=10)

    # Add annotations for mean
    for i, (val, label) in enumerate(zip([mean_baseline, mean_rag, mean_erl],
                                          ["Baseline", "RAG", "EvidenceRL"])):
        ax.annotate(f'{val:.3f}', xy=(i, val), xytext=(i+0.15, val+0.08),
                    fontsize=10, fontweight='bold')

    # Reference line at 0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # Shade regions
    ax.axhspan(-1.5, 0, alpha=0.03, color='red')
    ax.axhspan(0, 1.5, alpha=0.03, color='green')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([LABELS[c] for c in ["baseline", "baseline_rag", "evidencerl"]])
    ax.set_ylabel("Average Grounding Score", fontweight='bold')
    ax.set_xlabel("Condition", fontweight='bold')
    ax.set_title(MODEL_DISPLAY[model], fontweight='bold', fontsize=14)
    ax.set_ylim(-0.8, 0.8)
    ax.grid(True, axis='y', alpha=0.3)

    # Stats box
    pct_improved = len(improved_significant) / len(all_patients) * 100
    pct_worsened = len(worsened_significant) / len(all_patients) * 100
    ax.text(0.02, 0.98,
            f'n={len(all_patients)} patients\n'
            f'Improved (Δ>{SIGNIFICANT_THRESHOLD}): {len(improved_significant)} ({pct_improved:.0f}%)\n'
            f'Worsened (Δ<-{SIGNIFICANT_THRESHOLD}): {len(worsened_significant)} ({pct_worsened:.0f}%)',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#2ca02c', linewidth=2, alpha=0.7, label=f'Improved (Δ>{SIGNIFICANT_THRESHOLD})'),
    Line2D([0], [0], color='#d62728', linewidth=2, alpha=0.7, label=f'Worsened (Δ<-{SIGNIFICANT_THRESHOLD})'),
    Line2D([0], [0], color='gray', linewidth=1, alpha=0.3, label='Stable'),
    Line2D([0], [0], color='black', linewidth=3, marker='o', label='Mean'),
]
fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.95))

fig.suptitle("Patient-Level Grounding Movement: Baseline → RAG → EvidenceRL",
             fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
FIGURE_DIR = Path(__file__).resolve().parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)
plt.savefig(FIGURE_DIR / "patient_movement.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: {FIGURE_DIR / 'patient_movement.png'}")

# ============================================================================
# SANKEY-STYLE: Show flow from negative to positive grounding
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_idx, model in enumerate(MODELS):
    ax = axes[ax_idx]

    paths = {
        "baseline": METRICS_BASE / f"metrics_baseline_output/{model}_v1.0",
        "baseline_rag": METRICS_BASE / f"metrics_baseline_rag_output/{model}_v1.0",
        "evidencerl": METRICS_BASE / f"metrics_dpo_output/{model}-dpo_v1.0",
    }

    condition_data = {}
    for cond, path in paths.items():
        if not path.exists():
            continue
        f = find_metrics_file(path)
        if f is None:
            continue
        data = load_file(f)
        condition_data[cond] = extract_patient_grounding(data)

    all_patients = set(condition_data.get("baseline", {}).keys())
    all_patients &= set(condition_data.get("baseline_rag", {}).keys())
    all_patients &= set(condition_data.get("evidencerl", {}).keys())

    # Categorize by start and end state
    categories = {
        "neg_to_pos": 0,  # Started negative, ended positive
        "neg_to_neg": 0,  # Started negative, stayed negative
        "pos_to_pos": 0,  # Started positive, stayed positive
        "pos_to_neg": 0,  # Started positive, became negative
    }

    for patient_id in all_patients:
        baseline_g = condition_data["baseline"].get(patient_id, 0)
        erl_g = condition_data["evidencerl"].get(patient_id, 0)

        start_pos = baseline_g > 0
        end_pos = erl_g > 0

        if not start_pos and end_pos:
            categories["neg_to_pos"] += 1
        elif not start_pos and not end_pos:
            categories["neg_to_neg"] += 1
        elif start_pos and end_pos:
            categories["pos_to_pos"] += 1
        else:
            categories["pos_to_neg"] += 1

    # Create stacked bar showing the flow
    total = len(all_patients)

    labels = ['Negative→Positive\n(improved)', 'Stayed Positive',
              'Stayed Negative', 'Positive→Negative\n(worsened)']
    values = [categories["neg_to_pos"], categories["pos_to_pos"],
              categories["neg_to_neg"], categories["pos_to_neg"]]
    colors_bar = ['#2ca02c', '#90EE90', '#FFB6C1', '#d62728']

    bars = ax.bar(range(4), values, color=colors_bar, edgecolor='black', linewidth=0.5)

    # Add percentage labels
    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}\n({pct:.0f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Number of Patients", fontweight='bold')
    ax.set_title(MODEL_DISPLAY[model], fontweight='bold', fontsize=12)
    ax.set_ylim(0, max(values) * 1.3)

    # Summary stat
    improvement_rate = (categories["neg_to_pos"] + categories["pos_to_pos"]) / total * 100
    ax.text(0.98, 0.98, f'Positive grounding after EvidenceRL:\n{improvement_rate:.0f}% of patients',
            transform=ax.transAxes, fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

fig.suptitle("Grounding State Transition: Baseline → EvidenceRL",
             fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "grounding_flow.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {FIGURE_DIR / 'grounding_flow.png'}")

# ============================================================================
# Print detailed statistics
# ============================================================================
print("\n" + "=" * 80)
print("PATIENT MOVEMENT STATISTICS")
print("=" * 80)

for model in MODELS:
    print(f"\n{MODEL_DISPLAY[model]}:")
    print("-" * 60)

    paths = {
        "baseline": METRICS_BASE / f"metrics_baseline_output/{model}_v1.0",
        "baseline_rag": METRICS_BASE / f"metrics_baseline_rag_output/{model}_v1.0",
        "evidencerl": METRICS_BASE / f"metrics_dpo_output/{model}-dpo_v1.0",
    }

    condition_data = {}
    for cond, path in paths.items():
        f = find_metrics_file(path)
        if f:
            data = load_file(f)
            condition_data[cond] = extract_patient_grounding(data)

    all_patients = set(condition_data.get("baseline", {}).keys())
    all_patients &= set(condition_data.get("baseline_rag", {}).keys())
    all_patients &= set(condition_data.get("evidencerl", {}).keys())

    changes = []
    for p in all_patients:
        b = condition_data["baseline"].get(p, 0)
        e = condition_data["evidencerl"].get(p, 0)
        changes.append(e - b)

    changes = np.array(changes)

    print(f"  Patients tracked: {len(all_patients)}")
    print(f"  Mean change: {np.mean(changes):+.4f}")
    print(f"  Median change: {np.median(changes):+.4f}")
    print(f"  Std of change: {np.std(changes):.4f}")
    print(f"  % improved (any): {sum(changes > 0) / len(changes) * 100:.1f}%")
    print(f"  % improved (>0.1): {sum(changes > 0.1) / len(changes) * 100:.1f}%")
    print(f"  % improved (>0.2): {sum(changes > 0.2) / len(changes) * 100:.1f}%")
    print(f"  % worsened (any): {sum(changes < 0) / len(changes) * 100:.1f}%")
    print(f"  % worsened (<-0.1): {sum(changes < -0.1) / len(changes) * 100:.1f}%")
