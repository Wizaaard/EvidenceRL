#!/usr/bin/env python3
"""
Plot SFT training and validation curves from training logs.

Reads training_log.json files from SFT model directories and produces
publication-quality plots of loss curves for visual inspection.

Usage:
    # Plot all SFT models
    python plot_sft_training.py

    # Plot specific mode
    python plot_sft_training.py --mode rag

    # Plot specific model
    python plot_sft_training.py --model gemma-3-4b

    # Custom output path
    python plot_sft_training.py --output plots/sft_curves.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for SLURM
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SFT_MODELS_DIR = PROJECT_ROOT / "sft_trained_models"

# Color palette for models
MODEL_COLORS = {
    "gemma-3-4b":   "#e6194b",
    "gemma-3-12b":  "#3cb44b",
    "gemma-3-27b":  "#4363d8",
    "Llama-3.2-3B": "#f58231",
    "Llama-3.1-8B": "#911eb4",
    "Llama-3.3-70B":"#42d4f4",
    "gpt-oss-20b":  "#f032e6",
    "gpt-oss-120b": "#bfef45",
}


def load_training_log(model_dir: Path) -> list:
    """Load training log history from a model directory.

    Searches for trainer_state.json in the highest-numbered checkpoint,
    then falls back to training_log.json at the model root.
    """
    # Try trainer_state.json from the latest checkpoint
    ckpt_dirs = sorted(
        model_dir.glob("checkpoint-*"),
        key=lambda d: int(d.name.split("-")[1]) if d.name.split("-")[1].isdigit() else 0,
    )
    for ckpt in reversed(ckpt_dirs):
        state_path = ckpt / "trainer_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            return state.get("log_history", [])

    # Fallback: training_log.json at model root
    log_path = model_dir / "training_log.json"
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)

    return None


def extract_curves(log_history: list) -> dict:
    """Extract train/eval loss curves from TRL log history."""
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    for entry in log_history:
        step = entry.get('step', 0)

        if 'loss' in entry:
            train_steps.append(step)
            train_loss.append(entry['loss'])

        if 'eval_loss' in entry:
            eval_steps.append(step)
            eval_loss.append(entry['eval_loss'])

    return {
        'train_steps': np.array(train_steps),
        'train_loss': np.array(train_loss),
        'eval_steps': np.array(eval_steps),
        'eval_loss': np.array(eval_loss),
    }


def plot_single_model(ax_train, ax_eval, model_name, curves, color):
    """Plot train and eval curves for a single model."""
    if len(curves['train_steps']) > 0:
        ax_train.plot(
            curves['train_steps'], curves['train_loss'],
            color=color, alpha=0.8, linewidth=1.5, label=model_name,
        )

    if len(curves['eval_steps']) > 0:
        ax_eval.plot(
            curves['eval_steps'], curves['eval_loss'],
            color=color, alpha=0.8, linewidth=1.5, label=model_name,
            marker='o', markersize=3,
        )


def plot_combined(all_curves: dict, mode: str, output_path: str):
    """Plot all models' curves on a combined figure."""
    fig, (ax_train, ax_eval) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"SFT Training Curves — {mode.upper()} Mode", fontsize=14, fontweight='bold')

    for model_name, curves in all_curves.items():
        color = MODEL_COLORS.get(model_name, '#808080')
        plot_single_model(ax_train, ax_eval, model_name, curves, color)

    # Train loss
    ax_train.set_title("Training Loss", fontsize=12)
    ax_train.set_xlabel("Step")
    ax_train.set_ylabel("Loss")
    ax_train.legend(fontsize=8, loc='upper right')
    ax_train.grid(True, alpha=0.3)

    # Eval loss
    ax_eval.set_title("Validation Loss", fontsize=12)
    ax_eval.set_xlabel("Step")
    ax_eval.set_ylabel("Loss")
    ax_eval.legend(fontsize=8, loc='upper right')
    ax_eval.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_individual(model_name: str, curves: dict, mode: str, output_dir: Path):
    """Plot individual model with train + eval on same axes."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f"SFT Training — {model_name} ({mode.upper()})", fontsize=14, fontweight='bold')

    if len(curves['train_steps']) > 0:
        ax.plot(
            curves['train_steps'], curves['train_loss'],
            color='#2196F3', alpha=0.6, linewidth=1.0, label='Train Loss',
        )

    if len(curves['eval_steps']) > 0:
        ax.plot(
            curves['eval_steps'], curves['eval_loss'],
            color='#F44336', alpha=0.9, linewidth=2.0, label='Val Loss',
            marker='o', markersize=4,
        )

        # Annotate best eval loss
        best_idx = np.argmin(curves['eval_loss'])
        best_step = curves['eval_steps'][best_idx]
        best_loss = curves['eval_loss'][best_idx]
        ax.annotate(
            f'Best: {best_loss:.4f}\n(step {best_step})',
            xy=(best_step, best_loss),
            xytext=(20, 20), textcoords='offset points',
            fontsize=9, color='#F44336',
            arrowprops=dict(arrowstyle='->', color='#F44336', alpha=0.7),
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"sft_{model_name}_{mode}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot SFT training curves")
    parser.add_argument("--models-dir", type=str, default=str(SFT_MODELS_DIR),
                        help="Directory containing SFT trained model directories")
    parser.add_argument("--mode", type=str, choices=["rag", "no-rag"],
                        help="Filter by mode (default: both)")
    parser.add_argument("--model", type=str, help="Filter by model name")
    parser.add_argument("--output", type=str, help="Output path for combined plot")
    parser.add_argument("--output-dir", type=str, help="Output directory for individual plots")
    parser.add_argument("--individual", action="store_true",
                        help="Also generate individual per-model plots")

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"ERROR: Models directory not found: {models_dir}")
        return

    # Default output directory
    plot_dir = Path(args.output_dir) if args.output_dir else models_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Discover model directories
    modes = [args.mode] if args.mode else ["rag", "no-rag"]

    for mode in modes:
        print(f"\n{'=' * 50}")
        print(f"Mode: {mode.upper()}")
        print(f"{'=' * 50}")

        all_curves = {}
        mode_suffix = f"sft_{mode.replace('-', '_')}" if mode == "no-rag" else f"sft_{mode}"

        for model_dir in sorted(models_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            # Match exact mode suffix: _sft_rag vs _sft_no-rag
            if mode == "rag" and not model_dir.name.endswith("_sft_rag"):
                continue
            if mode == "no-rag" and not model_dir.name.endswith("_sft_no-rag"):
                continue

            # Extract model name from dir name (e.g., "gemma-3-4b_sft_rag" -> "gemma-3-4b")
            model_name = model_dir.name.replace(f"_sft_{mode}", "")

            if args.model and model_name != args.model:
                continue

            log_history = load_training_log(model_dir)
            if log_history is None:
                print(f"  {model_name}: No training log found, skipping")
                continue

            curves = extract_curves(log_history)
            print(f"  {model_name}: {len(curves['train_steps'])} train steps, {len(curves['eval_steps'])} eval points")

            all_curves[model_name] = curves

            # Individual plot
            if args.individual:
                plot_individual(model_name, curves, mode, plot_dir)

        if all_curves:
            # Combined plot
            combined_path = args.output if args.output else str(plot_dir / f"sft_curves_{mode}.png")
            plot_combined(all_curves, mode, combined_path)
        else:
            print("  No training logs found.")

    print(f"\nAll plots saved to: {plot_dir}/")


if __name__ == "__main__":
    main()
