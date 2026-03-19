#!/usr/bin/env python3
"""Scatter plot: per-case grounding score vs correctness for all BarExam models.

Creates a strip/swarm plot showing each case's G_avg score, colored by
correct/incorrect, with threshold bands marked. One panel per model.
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
BAREXAM_OUTPUT = DATA_ROOT / "barexam_output" / "test"
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

VERSION = "v1.2"
MODE = "rag"
METRICS_SUBDIR = "metrics_v4"
THRESHOLD = 0.1


def load_model_data(model_id: str) -> list[dict] | None:
    path = BAREXAM_OUTPUT / f"{model_id}_{MODE}-{VERSION}" / METRICS_SUBDIR / "enhanced_metrics.json"
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    return data["per_case"]


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

        # Extract data
        correct_scores = []
        incorrect_scores = []
        for c in cases:
            score = c.get("grounding_avg")
            if score is None:
                continue
            if c.get("correct", False):
                correct_scores.append(score)
            else:
                incorrect_scores.append(score)

        # Add jitter for visibility
        rng = np.random.RandomState(42)

        # Plot incorrect first (behind), then correct
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
                c=color, alpha=0.5, s=30, edgecolors="white", linewidths=0.3,
                zorder=3,
            )
            # Add mean marker
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

        # Labels
        n_correct = len(correct_scores)
        n_incorrect = len(incorrect_scores)
        acc = n_correct / (n_correct + n_incorrect) * 100 if (n_correct + n_incorrect) > 0 else 0
        ax.set_title(f"{MODEL_DISPLAY[model_id]}\n(Acc={acc:.1f}%, n={n_correct + n_incorrect})", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Incorrect", "Correct"], fontsize=9)
        ax.set_ylim(-1.05, 1.05)

        if idx % 4 == 0:
            ax.set_ylabel("$G_{\\mathrm{avg}}$ (entailment − contradiction)", fontsize=10)

        # Add zone labels on right side
        if idx == 3 or idx == 7:
            ax.annotate("Grounded", xy=(1.05, 0.85), xycoords="axes fraction",
                       fontsize=8, color="green", fontweight="bold", ha="left")
            ax.annotate("Weak", xy=(1.05, 0.5), xycoords="axes fraction",
                       fontsize=8, color="gray", fontweight="bold", ha="left")
            ax.annotate("Contradicted", xy=(1.05, 0.15), xycoords="axes fraction",
                       fontsize=8, color="red", fontweight="bold", ha="left")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors["correct"], alpha=0.6, label="Correct"),
        mpatches.Patch(facecolor=colors["incorrect"], alpha=0.6, label="Incorrect"),
        plt.Line2D([0], [0], marker="D", color="black", linestyle="None",
                   markersize=6, label="Mean"),
        mpatches.Patch(facecolor="gray", alpha=0.15, label=f"Weak zone (±{tau})"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=4, fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "BarExam: Per-Case Grounding Score Distribution by Correctness\n"
        f"(NLI: DeBERTa-v3-large, scoring: entailment − contradiction, τ = ±{tau})",
        fontsize=14, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    out_path = FIGURE_DIR / "barexam_grounding_scatter_v4.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Also save PDF for paper
    out_pdf = FIGURE_DIR / "barexam_grounding_scatter_v4.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")

    plt.close()


if __name__ == "__main__":
    main()
