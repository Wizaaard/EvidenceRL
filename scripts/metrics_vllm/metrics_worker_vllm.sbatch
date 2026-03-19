#!/usr/bin/env bash
#SBATCH -J erl_metrics_vllm_worker
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH -o logs_evidencerl/metrics_vllm_worker_%A_%a.out
#SBATCH -e logs_evidencerl/metrics_vllm_worker_%A_%a.err

# ============================================================================
# EvidenceRL Metrics Worker (vLLM Version)
# ============================================================================
# High-performance metrics computation using vLLM for the judge.
#
# Key differences from HuggingFace version:
# - Uses vLLM for judge inference (14-24x faster)
# - PagedAttention for efficient memory management
# - Reduced time limit (2h vs 4h) due to faster inference
#
# Note: The NLI model (CrossEncoder) remains unchanged as it's already
# optimized and doesn't benefit from vLLM.
#
# Required environment variables:
#   GENERATION_JSON  - Path to generation JSON file
#   OUTPUT_DIR       - Output directory for worker results
#   PATIENT_START    - Start index for this worker's patient subset
#   PATIENT_END      - End index for this worker's patient subset
#   WORKER_ID        - Worker ID
#
# Optional environment variables:
#   JUDGE_MODEL           - Judge model path (default: medgemma-27b-it)
#   TENSOR_PARALLEL_SIZE  - GPUs for tensor parallelism (default: 2)
#   GPU_MEMORY_UTILIZATION - Fraction of GPU memory (default: 0.90)
#   SIGNAL_DIR            - Directory for completion signals
# ============================================================================

set -euo pipefail

# Validate required environment variables
: "${GENERATION_JSON:?Need GENERATION_JSON}"
: "${OUTPUT_DIR:?Need OUTPUT_DIR}"
: "${PATIENT_START:?Need PATIENT_START}"
: "${PATIENT_END:?Need PATIENT_END}"
: "${WORKER_ID:?Need WORKER_ID}"

# Configuration
CONDA_ENV="${CONDA_ENV:-evidencerl}"
PROJECT_ROOT="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/code"
MODEL_DIR="//storage/ice-shared/bmed-sp-wang/Models"

# Model defaults
JUDGE_MODEL="${JUDGE_MODEL:-${MODEL_DIR}/medgemma-27b-it}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-base}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

# Signal directory
SIGNAL_DIR="${SIGNAL_DIR:-${OUTPUT_DIR}/signals}"

# Create directories
mkdir -p logs_evidencerl "${OUTPUT_DIR}" "${SIGNAL_DIR}"

# GPU health check
echo "[vllm worker ${WORKER_ID}] Checking GPU availability..."
echo "[vllm worker ${WORKER_ID}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "[vllm worker ${WORKER_ID}] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-not set}"

if ! nvidia-smi &>/dev/null; then
    echo "[ERROR] GPU not available on this node. Exiting for retry."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[vllm worker ${WORKER_ID}] GPU check passed."

# Ensure CUDA_VISIBLE_DEVICES is set
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
        echo "[vllm worker ${WORKER_ID}] Set CUDA_VISIBLE_DEVICES from SLURM_JOB_GPUS: ${CUDA_VISIBLE_DEVICES}"
    elif [ -n "${SLURM_STEP_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS}"
        echo "[vllm worker ${WORKER_ID}] Set CUDA_VISIBLE_DEVICES from SLURM_STEP_GPUS: ${CUDA_VISIBLE_DEVICES}"
    elif [ -n "${GPU_DEVICE_ORDINAL:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${GPU_DEVICE_ORDINAL}"
        echo "[vllm worker ${WORKER_ID}] Set CUDA_VISIBLE_DEVICES from GPU_DEVICE_ORDINAL: ${CUDA_VISIBLE_DEVICES}"
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
echo "[vllm worker ${WORKER_ID}] Activated conda env: ${CONDA_ENV}"

# Ensure conda's libstdc++ takes precedence
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Set environment
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# vLLM specific settings
# Note: Do NOT set VLLM_ATTENTION_BACKEND - let vLLM auto-select based on model
# FLASH_ATTN doesn't support multimodal models like medgemma

# Model/version defaults for consistent naming (from master or extracted)
MODEL_ID="${MODEL_ID:-model}"
VERSION="${VERSION:-1.0}"

# Output files for this worker (consistent with HuggingFace naming)
METRICS_OUTPUT="${OUTPUT_DIR}/${MODEL_ID}_metrics_worker${WORKER_ID}_v${VERSION}.json"
SIGNAL_FILE="${SIGNAL_DIR}/metrics_vllm_worker${WORKER_ID}.done"

# Check if already completed (idempotency)
if [ -f "${SIGNAL_FILE}" ]; then
    echo "[vllm worker ${WORKER_ID}] Already completed (signal file exists). Skipping."
    exit 0
fi

# Check if output already exists and is valid
if [ -f "${METRICS_OUTPUT}" ]; then
    if python3 -c "import json; d=json.load(open('${METRICS_OUTPUT}')); assert 'patient_metrics' in d" 2>/dev/null; then
        echo "[vllm worker ${WORKER_ID}] Output already exists and is valid. Marking as complete."
        touch "${SIGNAL_FILE}"
        exit 0
    else
        echo "[vllm worker ${WORKER_ID}] Output exists but invalid. Re-running."
    fi
fi

echo "[vllm worker ${WORKER_ID}] Starting vLLM metrics computation..."
echo "[vllm worker ${WORKER_ID}] GENERATION_JSON=${GENERATION_JSON}"
echo "[vllm worker ${WORKER_ID}] JUDGE_MODEL=${JUDGE_MODEL}"
echo "[vllm worker ${WORKER_ID}] PATIENT_RANGE=[${PATIENT_START}:${PATIENT_END}]"
echo "[vllm worker ${WORKER_ID}] TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
echo "[vllm worker ${WORKER_ID}] GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "[vllm worker ${WORKER_ID}] METRICS_OUTPUT=${METRICS_OUTPUT}"

# Run metrics computation with vLLM
set +e
srun --gpu-bind=closest python "${PROJECT_ROOT}/scripts/metrics_vllm/compute_metrics_worker_vllm.py" \
    --generation-json "${GENERATION_JSON}" \
    --output-json "${METRICS_OUTPUT}" \
    --judge-model "${JUDGE_MODEL}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --nli-model "${NLI_MODEL}" \
    --embedding-model "${EMBEDDING_MODEL}" \
    --patient-start "${PATIENT_START}" \
    --patient-end "${PATIENT_END}" \
    --worker-id "${WORKER_ID}"
METRICS_EXIT_CODE=$?
set -e

# Check if computation succeeded
if [ $METRICS_EXIT_CODE -eq 0 ] && [ -f "${METRICS_OUTPUT}" ]; then
    echo "[vllm worker ${WORKER_ID}] Metrics computation completed successfully."
    touch "${SIGNAL_FILE}"
    echo "[vllm worker ${WORKER_ID}] Signal file created: ${SIGNAL_FILE}"
else
    echo "[ERROR] Metrics computation failed for worker ${WORKER_ID} (exit code: ${METRICS_EXIT_CODE})"
    exit 1
fi
