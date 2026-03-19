#!/usr/bin/env bash
#SBATCH -J erl_metrics_worker
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH -o logs_evidencerl/metrics_worker_%A_%a.out
#SBATCH -e logs_evidencerl/metrics_worker_%A_%a.err

# ============================================================================
# EvidenceRL Metrics Computation Worker
# ============================================================================
# Computes reward metrics for a subset of patients from generation output.
#
# Models loaded:
#   - Judge model (LLM): medgemma-27b-it (fixed) for precision/accuracy evaluation
#   - NLI model (CrossEncoder): For grounding evaluation
#   - Embedding model: For Focus-Then-Verify section selection
#
# Required environment variables:
#   MODEL_ID        - Short model identifier
#   GENERATION_FILE - Path to generation JSON output
#   JUDGE_MODEL     - Path to judge model (fixed: medgemma-27b-it)
#   PATIENT_START   - Start index for this worker's patient subset
#   PATIENT_END     - End index for this worker's patient subset
#   WORKER_ID       - Worker ID
#   VERSION         - Version string
#   OUTPUT_BASE_DIR - Base directory for outputs
#
# Optional environment variables:
#   SIGNAL_DIR      - Directory for completion signals
#   NLI_MODEL       - NLI model name (default: cross-encoder/nli-deberta-v3-base)
#   EMBEDDING_MODEL - Embedding model (default: sentence-transformers/all-MiniLM-L6-v2)
#   REWARD_WEIGHT   - Weight for grounding in combined reward (default: 0.5)
#   BATCH_SIZE      - Batch size for judge inference (default: 48 for 27B model on 2x H200)
# ============================================================================

set -euo pipefail

# Validate required environment variables
: "${MODEL_ID:?Need MODEL_ID}"
: "${GENERATION_FILE:?Need GENERATION_FILE}"
: "${JUDGE_MODEL:?Need JUDGE_MODEL}"
: "${PATIENT_START:?Need PATIENT_START}"
: "${PATIENT_END:?Need PATIENT_END}"
: "${WORKER_ID:?Need WORKER_ID}"
: "${VERSION:=1.0}"
: "${OUTPUT_BASE_DIR:?Need OUTPUT_BASE_DIR}"

# Configuration
CONDA_ENV="${CONDA_ENV:-evidencerl}"
PROJECT_ROOT="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/code"

# Model defaults
NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-base}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
REWARD_WEIGHT="${REWARD_WEIGHT:-0.5}"
# Default batch size 48 for medgemma-27b-it judge on 2x H200 GPUs (54GB model, 286GB VRAM total)
# Judge inference uses max_new_tokens=32 (not 1024), so KV cache is small and larger batches are safe
# Tested: batch=24 uses ~26% VRAM, batch=48 uses ~37% VRAM - safe with good margin
BATCH_SIZE="${BATCH_SIZE:-48}"
MAX_K="${MAX_K:-5}"

# Output directories
SIGNAL_DIR="${SIGNAL_DIR:-${OUTPUT_BASE_DIR}/signals}"

# Create directories
mkdir -p logs_evidencerl "${OUTPUT_BASE_DIR}" "${SIGNAL_DIR}"

# GPU health check and visibility
echo "[metrics worker ${WORKER_ID}] Checking GPU availability..."
echo "[metrics worker ${WORKER_ID}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "[metrics worker ${WORKER_ID}] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-not set}"
echo "[metrics worker ${WORKER_ID}] SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-not set}"
echo "[metrics worker ${WORKER_ID}] GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-not set}"

if ! nvidia-smi &>/dev/null; then
    echo "[ERROR] GPU not available on this node. Exiting for retry on different node."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[metrics worker ${WORKER_ID}] GPU check passed."

# Ensure CUDA_VISIBLE_DEVICES is set from Slurm allocation
# This fixes issues where multiple jobs on same node compete for GPUs
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
        echo "[metrics worker ${WORKER_ID}] Set CUDA_VISIBLE_DEVICES from SLURM_JOB_GPUS: ${CUDA_VISIBLE_DEVICES}"
    elif [ -n "${SLURM_STEP_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS}"
        echo "[metrics worker ${WORKER_ID}] Set CUDA_VISIBLE_DEVICES from SLURM_STEP_GPUS: ${CUDA_VISIBLE_DEVICES}"
    elif [ -n "${GPU_DEVICE_ORDINAL:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${GPU_DEVICE_ORDINAL}"
        echo "[metrics worker ${WORKER_ID}] Set CUDA_VISIBLE_DEVICES from GPU_DEVICE_ORDINAL: ${CUDA_VISIBLE_DEVICES}"
    fi
fi

# Initialize conda
CONDA_INIT_SCRIPT=""
for conda_path in \
    "$HOME/.conda/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/mambaforge/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh"; do
    if [ -f "$conda_path" ]; then
        CONDA_INIT_SCRIPT="$conda_path"
        break
    fi
done

if [ -n "$CONDA_INIT_SCRIPT" ]; then
    source "$CONDA_INIT_SCRIPT"
else
    CONDA_EXE_PATH="$(command -v conda 2>/dev/null || true)"
    if [ -n "$CONDA_EXE_PATH" ] && [ -x "$CONDA_EXE_PATH" ]; then
        CONDA_BASE="$("$CONDA_EXE_PATH" info --base)"
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "[ERROR] Could not find conda. Exiting."
        exit 2
    fi
fi

set +u
conda activate "${CONDA_ENV}"
set -u
echo "[metrics worker ${WORKER_ID}] Activated conda env: ${CONDA_ENV}"

# Ensure conda's libstdc++ takes precedence
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Set environment
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Output file for this worker
JSON_OUTPUT="${OUTPUT_BASE_DIR}/${MODEL_ID}_metrics_worker${WORKER_ID}_v${VERSION}.json"
SIGNAL_FILE="${SIGNAL_DIR}/${MODEL_ID}_metrics_worker${WORKER_ID}.done"

# Check if already completed (idempotency)
if [ -f "${SIGNAL_FILE}" ]; then
    echo "[metrics worker ${WORKER_ID}] Already completed (signal file exists). Skipping."
    exit 0
fi

# Check if output already exists and is valid
if [ -f "${JSON_OUTPUT}" ]; then
    if python3 -c "import json; d=json.load(open('${JSON_OUTPUT}')); assert len(d.get('patient_metrics',[])) > 0" 2>/dev/null; then
        echo "[metrics worker ${WORKER_ID}] Output already exists and is valid. Marking as complete."
        touch "${SIGNAL_FILE}"
        exit 0
    else
        echo "[metrics worker ${WORKER_ID}] Output exists but invalid. Re-running."
    fi
fi

echo "[metrics worker ${WORKER_ID}] Starting metrics computation..."
echo "[metrics worker ${WORKER_ID}] MODEL_ID=${MODEL_ID}"
echo "[metrics worker ${WORKER_ID}] GENERATION_FILE=${GENERATION_FILE}"
echo "[metrics worker ${WORKER_ID}] JUDGE_MODEL=${JUDGE_MODEL} (fixed: medgemma-27b-it)"
echo "[metrics worker ${WORKER_ID}] PATIENT_RANGE=[${PATIENT_START}:${PATIENT_END}]"
echo "[metrics worker ${WORKER_ID}] NLI_MODEL=${NLI_MODEL}"
echo "[metrics worker ${WORKER_ID}] EMBEDDING_MODEL=${EMBEDDING_MODEL}"
echo "[metrics worker ${WORKER_ID}] OUTPUT=${JSON_OUTPUT}"

# Run metrics computation
# Use --gpu-bind=closest to ensure proper GPU binding when multiple jobs share a node
set +e
srun --gpu-bind=closest python "${PROJECT_ROOT}/scripts/metrics/compute_metrics_worker.py" \
    --generation-file "${GENERATION_FILE}" \
    --output-json "${JSON_OUTPUT}" \
    --judge-model "${JUDGE_MODEL}" \
    --nli-model "${NLI_MODEL}" \
    --embedding-model "${EMBEDDING_MODEL}" \
    --patient-start-idx "${PATIENT_START}" \
    --patient-end-idx "${PATIENT_END}" \
    --reward-weight-grounding "${REWARD_WEIGHT}" \
    --batch-size "${BATCH_SIZE}" \
    --max-k "${MAX_K}"
METRICS_EXIT_CODE=$?
set -e

# Check if computation succeeded
if [ $METRICS_EXIT_CODE -eq 0 ] && [ -f "${JSON_OUTPUT}" ]; then
    echo "[metrics worker ${WORKER_ID}] Metrics computation completed successfully."
    touch "${SIGNAL_FILE}"
    echo "[metrics worker ${WORKER_ID}] Signal file created: ${SIGNAL_FILE}"
else
    echo "[ERROR] Metrics computation failed for worker ${WORKER_ID} (exit code: ${METRICS_EXIT_CODE})"
    exit 1
fi
