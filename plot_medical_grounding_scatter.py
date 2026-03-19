#!/usr/bin/env python3
"""Scatter plot: per-diagnosis grounding score vs correctness for all medical models.

Mirrors the BarExam scatter plot but for the medical domain.
Each dot = one predicted diagnosis, colored by verdict (correct/incorrect).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parent.parent
MEDICAL_OUTPUT = DATA_ROOT / "metrics_baseline_output" / "paper_ready"
FIGURE_DIR = DATA_ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

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

MODEL_DISPLAY = {
    "gemma-3-4b": "Gemma-3 4B",
    "gemma-3-12b": "Gemma-3 12B",
    "gemma-3-27b": "Gemma-3 27B",
    "Llama-3.2-3B": "Llama-3.2 3B",
    "Llama-3.1-8B": "Llama-3.1 8B",
    "Llama-3.3-70B": "Llama-3.3 70B",
    "gpt-oss-20b": "GPT-OSS 20B",
    "gpt-oss-120b": "GPT-OSS 120B",
}

SUFFIX = "_v1.0-TEST_llama70"
THRESHOLD = 0.5  # Medical domain uses 0.5 threshold


def load_model_data(model_id: str) -> list[dict] | None:
    """Load per-diagnosis (verdict, grounding_avg) pairs."""
    path = MEDICAL_OUTPUT / f"{model_id}{SUFFIX}" / "enhanced_metrics.json"
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)

    results = []
    for patient in data["patient_metrics"]:
        verdicts = patient.get("verdicts", [])
        rewards = patient.get("per_diagnosis_rewards", [])
        for i, (verdict, reward) in enumerate(zip(verdicts, rewards)):
            g_avg = reward.get("grounding_avg")
            if g_avg is not None:
                results.append({"correct": verdict, "grounding_avg": g_avg})
    return results


def main():
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
    axes = axes.flatten()

    colors = {"correct": "#2196F3", "incorrect": "#F44336"}
    tau = THRESHOLD

    for idx, model_id in enumerate(MODELS):
        ax = axes[idx]
        cases = load_model_data(model_id)
        if cases is None:
            ax.set_title(f"{MODEL_DISPLAY[model_id]} (no data)")
            continue

        correct_scores = [c["grounding_avg"] for c in cases if c["correct"]]
        incorrect_scores = [c["grounding_avg"] for c in cases if not c["correct"]]

        rng = np.random.RandomState(42)

        for label, scores, color, x_center in [
            ("Incorrect", incorrect_scores, colors["incorrect"], 0),
            ("Correct", correct_scores, colors["correct"], 1),
        ]:
            if not scores:
                continue
            arr = np.array(scores)
            jitter = rng.uniform(-0.25, 0.25, size=len(arr))
            ax.scatter(
                x_center + jitter, arr,
                c=color, alpha=0.3, s=12, edgecolors="none",
                zorder=3,
            )
            # Mean marker
            mean_val = np.mean(arr)
            ax.plot(
                x_center, mean_val, marker="D", color="black",
                markersize=8, zorder=5, markeredgecolor="white", markeredgewidth=1,
            )

        # Threshold bands
        ax.axhspan(-tau, tau, alpha=0.08, color="gray", zorder=1)
        ax.axhline(tau, color="green", linestyle="--", alpha=0.5, linewidth=1, zorder=2)
        ax.axhline(-tau, color="red", linestyle="--", alpha=0.5, linewidth=1, zorder=2)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.3, linewidth=0.5, zorder=2)

        n_correct = len(correct_scores)
        n_incorrect = len(incorrect_scores)
        n_total = n_correct + n_incorrect
        ax.set_title(
            f"{MODEL_DISPLAY[model_id]}\n(n={n_total} diagnoses, "
            f"{n_correct} correct / {n_incorrect} incorrect)",
            fontsize=10,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Incorrect", "Correct"], fontsize=9)
        ax.set_ylim(-1.05, 1.05)

        if idx % 4 == 0:
            ax.set_ylabel("$G_{\\mathrm{avg}}$ (entailment − contradiction)", fontsize=10)

        if idx == 3 or idx == 7:
            ax.annotate("Grounded", xy=(1.05, 0.85), xycoords="axes fraction",
                       fontsize=8, color="green", fontweight="bold", ha="left")
            ax.annotate("Weak", xy=(1.05, 0.5), xycoords="axes fraction",
                       fontsize=8, color="gray", fontweight="bold", ha="left")
            ax.annotate("Contradicted", xy=(1.05, 0.15), xycoords="axes fraction",
                       fontsize=8, color="red", fontweight="bold", ha="left")

    legend_elements = [
        mpatches.Patch(facecolor=colors["correct"], alpha=0.6, label="Correct Diagnosis"),
        mpatches.Patch(facecolor=colors["incorrect"], alpha=0.6, label="Incorrect Diagnosis"),
        plt.Line2D([0], [0], marker="D", color="black", linestyle="None",
                   markersize=6, label="Mean"),
        mpatches.Patch(facecolor="gray", alpha=0.15, label=f"Weak zone (±{tau})"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=4, fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Medical Domain: Per-Diagnosis Grounding Score Distribution by Correctness\n"
        f"(NLI: PubMedBERT-MNLI-MedNLI, scoring: entailment − contradiction, τ = ±{tau})",
        fontsize=14, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    out_path = FIGURE_DIR / "medical_grounding_scatter.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

    out_pdf = FIGURE_DIR / "medical_grounding_scatter.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")

    plt.close()

    # Print summary stats
    print("\n=== Summary Statistics ===")
    for model_id in MODELS:
        cases = load_model_data(model_id)
        if cases is None:
            continue
        correct_scores = [c["grounding_avg"] for c in cases if c["correct"]]
        incorrect_scores = [c["grounding_avg"] for c in cases if not c["correct"]]
        all_scores = [c["grounding_avg"] for c in cases]
        arr = np.array(all_scores)
        print(f"  {MODEL_DISPLAY[model_id]:16s}: n={len(cases):5d}  "
              f"mean={np.mean(arr):.3f}  "
              f"correct_mean={np.mean(correct_scores):.3f}  "
              f"incorrect_mean={np.mean(incorrect_scores):.3f}  "
              f"grounded={np.mean(arr > tau)*100:.0f}%  "
              f"weak={np.mean((arr >= -tau) & (arr <= tau))*100:.0f}%  "
              f"contradicted={np.mean(arr < -tau)*100:.0f}%")


if __name__ == "__main__":
    main()
