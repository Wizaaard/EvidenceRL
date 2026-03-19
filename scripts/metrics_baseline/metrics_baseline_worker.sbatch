#!/usr/bin/env bash
#SBATCH -J erl_baseline_metrics_worker
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH -o logs_evidencerl_baseline/metrics_baseline_worker_%A_%a.out
#SBATCH -e logs_evidencerl_baseline/metrics_baseline_worker_%A_%a.err

# ============================================================================
# EvidenceRL Baseline Metrics Computation Worker
# ============================================================================
# Computes reward metrics for BASELINE generation output (no-RAG).
#
# Required environment variables:
#   MODEL_ID        - Short model identifier
#   GENERATION_FILE - Path to baseline generation JSON output
#   JUDGE_MODEL     - Path to judge model (fixed: medgemma-27b-it)
#   PATIENT_START   - Start index for this worker's patient subset
#   PATIENT_END     - End index for this worker's patient subset
#   WORKER_ID       - Worker ID
#   VERSION         - Version string
#   OUTPUT_BASE_DIR - Base directory for outputs
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
BATCH_SIZE="${BATCH_SIZE:-48}"
MAX_K="${MAX_K:-5}"

# Output directories
SIGNAL_DIR="${SIGNAL_DIR:-${OUTPUT_BASE_DIR}/signals}"

# Create directories
mkdir -p logs_evidencerl_baseline "${OUTPUT_BASE_DIR}" "${SIGNAL_DIR}"

# GPU health check
echo "[baseline metrics worker ${WORKER_ID}] Checking GPU availability..."
echo "[baseline metrics worker ${WORKER_ID}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"

if ! nvidia-smi &>/dev/null; then
    echo "[ERROR] GPU not available on this node. Exiting for retry on different node."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[baseline metrics worker ${WORKER_ID}] GPU check passed."

# Ensure CUDA_VISIBLE_DEVICES is set from Slurm allocation
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
    elif [ -n "${SLURM_STEP_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS}"
    elif [ -n "${GPU_DEVICE_ORDINAL:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${GPU_DEVICE_ORDINAL}"
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
echo "[baseline metrics worker ${WORKER_ID}] Activated conda env: ${CONDA_ENV}"

# Ensure conda's libstdc++ takes precedence
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Set environment
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Output file for this worker - includes "baseline" in name
JSON_OUTPUT="${OUTPUT_BASE_DIR}/${MODEL_ID}_baseline_metrics_worker${WORKER_ID}_v${VERSION}.json"
SIGNAL_FILE="${SIGNAL_DIR}/${MODEL_ID}_baseline_metrics_worker${WORKER_ID}.done"

# Check if already completed (idempotency)
if [ -f "${SIGNAL_FILE}" ]; then
    echo "[baseline metrics worker ${WORKER_ID}] Already completed (signal file exists). Skipping."
    exit 0
fi

# Check if output already exists and is valid
if [ -f "${JSON_OUTPUT}" ]; then
    if python3 -c "import json; d=json.load(open('${JSON_OUTPUT}')); assert len(d.get('patient_metrics',[])) > 0" 2>/dev/null; then
        echo "[baseline metrics worker ${WORKER_ID}] Output already exists and is valid. Marking as complete."
        touch "${SIGNAL_FILE}"
        exit 0
    else
        echo "[baseline metrics worker ${WORKER_ID}] Output exists but invalid. Re-running."
    fi
fi

echo "[baseline metrics worker ${WORKER_ID}] Starting BASELINE metrics computation..."
echo "[baseline metrics worker ${WORKER_ID}] MODE=BASELINE (no-RAG evaluation)"
echo "[baseline metrics worker ${WORKER_ID}] MODEL_ID=${MODEL_ID}"
echo "[baseline metrics worker ${WORKER_ID}] GENERATION_FILE=${GENERATION_FILE}"
echo "[baseline metrics worker ${WORKER_ID}] JUDGE_MODEL=${JUDGE_MODEL}"
echo "[baseline metrics worker ${WORKER_ID}] PATIENT_RANGE=[${PATIENT_START}:${PATIENT_END}]"
echo "[baseline metrics worker ${WORKER_ID}] OUTPUT=${JSON_OUTPUT}"

# Run metrics computation - reuse the same worker script (works for both RAG and baseline)
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
    echo "[baseline metrics worker ${WORKER_ID}] Baseline metrics computation completed successfully."
    touch "${SIGNAL_FILE}"
    echo "[baseline metrics worker ${WORKER_ID}] Signal file created: ${SIGNAL_FILE}"
else
    echo "[ERROR] Baseline metrics computation failed for worker ${WORKER_ID} (exit code: ${METRICS_EXIT_CODE})"
    exit 1
fi
