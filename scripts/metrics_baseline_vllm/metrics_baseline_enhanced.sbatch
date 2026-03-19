#!/usr/bin/env bash
#SBATCH -J evidencerl_baseline_enhanced_metrics
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH -o logs_evidencerl_baseline/baseline_enhanced_metrics_%j.out
#SBATCH -e logs_evidencerl_baseline/baseline_enhanced_metrics_%j.err

# ============================================================================
# EvidenceRL Baseline Enhanced Metrics (vLLM Version)
# ============================================================================
# Computes ALL enhanced metrics on BASELINE (no-RAG) generation output:
# - F1@k, Macro-F1, Micro-F1
# - Retrieval Success@k, MRR
# - Statistical confidence intervals
# - Per-diagnosis performance breakdown
#
# Uses the same compute_evidencerl_metrics_vllm_enhanced.py as the RAG version.
# ============================================================================

set -euo pipefail

: "${GENERATION_FILE:?Need GENERATION_FILE}"
: "${JUDGE_MODEL:?Need JUDGE_MODEL}"
: "${OUTPUT_JSON:?Need OUTPUT_JSON}"
: "${CONDA_ENV:=ragcon}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_K="${MAX_K:-5}"
NLI_MODEL="${NLI_MODEL:-pritamdeka/PubMedBERT-MNLI-MedNLI}"

# Use absolute path to project root (sbatch runs from /var/lib/slurm/)
PROJECT_ROOT="/home/hice1/jtamo3/bmed-sp-wang/BenoitData/Restricted/evidenceRL/code"

echo "============================================================"
echo "EvidenceRL Baseline Enhanced Metrics (vLLM)"
echo "============================================================"
echo "MODE:            BASELINE (no-RAG)"
echo "GENERATION_FILE: ${GENERATION_FILE}"
echo "JUDGE_MODEL:     ${JUDGE_MODEL}"
echo "NLI_MODEL:       ${NLI_MODEL}"
echo "OUTPUT_JSON:     ${OUTPUT_JSON}"
echo "TENSOR_PARALLEL: ${TENSOR_PARALLEL_SIZE}"
echo "CONDA_ENV:       ${CONDA_ENV}"
echo "============================================================"

# Check GPU
echo "Checking GPU availability..."
if ! nvidia-smi &>/dev/null; then
    echo "[ERROR] GPU not available"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Set CUDA_VISIBLE_DEVICES
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    [ -n "${SLURM_JOB_GPUS:-}" ] && export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
    [ -n "${SLURM_STEP_GPUS:-}" ] && export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS}"
fi

# Initialize conda
for conda_path in "$HOME/.conda/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/mambaforge/etc/profile.d/conda.sh" "/usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh"; do
    [ -f "$conda_path" ] && { source "$conda_path"; break; }
done

echo "Activating conda environment: ${CONDA_ENV}"
set +u
eval "$(conda shell.bash hook 2>/dev/null || true)"
conda activate "${CONDA_ENV}"
set -u

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Create output directory
OUTPUT_DIR="$(dirname "${OUTPUT_JSON}")"
mkdir -p "${OUTPUT_DIR}"

echo "Starting baseline enhanced metrics computation..."
python "${PROJECT_ROOT}/compute_evidencerl_metrics_vllm_enhanced.py" \
    --results-json "${GENERATION_FILE}" \
    --output-json "${OUTPUT_JSON}" \
    --judge-model "${JUDGE_MODEL}" \
    --nli-model "${NLI_MODEL}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-k "${MAX_K}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ] && [ -f "${OUTPUT_JSON}" ]; then
    echo "============================================================"
    echo "Baseline Enhanced Metrics Completed Successfully!"
    echo "============================================================"
    echo "Output saved to: ${OUTPUT_JSON}"

    # Show quick summary if jq is available
    if command -v jq &> /dev/null; then
        echo ""
        echo "Quick Summary:"
        jq -r '.aggregate_metrics |
            "  P@3:      \(.avg_precision_at_k."3" | tonumber | . * 100 | round / 100)",
            "  R@3:      \(.avg_recall_at_k."3" | tonumber | . * 100 | round / 100)",
            "  F1@3:     \(.avg_f1_at_k."3" | tonumber | . * 100 | round / 100)",
            "  G_max@3:  \(.avg_reward_grounding_max_at_k."3" | tonumber | . * 100 | round / 100)",
            "  G_avg@3:  \(.avg_reward_grounding_avg_at_k."3" | tonumber | . * 100 | round / 100)",
            "  Reward:   \(.avg_reward_combined_at_k."3" | tonumber | . * 100 | round / 100)"
        ' "${OUTPUT_JSON}"
    fi

    exit 0
else
    echo "[ERROR] Baseline enhanced metrics computation failed (exit code: ${EXIT_CODE})"
    exit 1
fi
