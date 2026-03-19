#!/usr/bin/env bash
#SBATCH -J erl_baseline_vllm_worker
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH -o logs_evidencerl_baseline/generation_baseline_vllm_worker_%A_%a.out
#SBATCH -e logs_evidencerl_baseline/generation_baseline_vllm_worker_%A_%a.err

# ============================================================================
# EvidenceRL Baseline Generation Worker (vLLM Version)
# ============================================================================
# High-performance BASELINE generation (no-RAG) using vLLM.
# Uses --no-rag flag for zero-shot generation WITHOUT retrieval.
#
# This is FASTER than RAG generation because:
# - No embedding computation
# - No FAISS index loading
# - No document retrieval
# Plus vLLM's 14-24x speedup over HuggingFace.
#
# Required environment variables:
#   MODEL_ID          - Short identifier for the model
#   MODEL_NAME        - Full path to the model
#   PATIENT_START     - Start index for this worker's patient subset
#   PATIENT_END       - End index for this worker's patient subset
#   WORKER_ID         - Worker ID
#   VERSION           - Version string for output naming
#   WORKER_OUTPUT_DIR - Directory for worker outputs (workers/ subdirectory)
#
# Optional environment variables:
#   SIGNAL_DIR      - Directory for completion signals
#   NUM_GPUS        - Number of GPUs for tensor parallelism (default: 2)
#   EXTRACTOR_MODEL - Path to separate model for fallback extraction (optional)
# ============================================================================

set -euo pipefail

# Validate required environment variables
: "${MODEL_ID:?Need MODEL_ID}"
: "${MODEL_NAME:?Need MODEL_NAME}"
: "${PATIENT_START:?Need PATIENT_START}"
: "${PATIENT_END:?Need PATIENT_END}"
: "${WORKER_ID:?Need WORKER_ID}"
: "${VERSION:=1.0}"
: "${WORKER_OUTPUT_DIR:?Need WORKER_OUTPUT_DIR}"

# Configuration
CONDA_ENV="${CONDA_ENV:-evidencerl}"
PROJECT_ROOT="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/code"
PATIENT_DATA_PATH="//storage/ice-shared/bmed-sp-wang/Ben/Data/physionet.org/files/mimic-iv-ext-cardiac-disease/1.0.0"

# vLLM-specific settings
NUM_GPUS="${NUM_GPUS:-2}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
EXTRACTOR_MODEL="${EXTRACTOR_MODEL:-}"

# Signal directory (should be passed from master, but provide fallback)
SIGNAL_DIR="${SIGNAL_DIR:-}"

# Create directories
mkdir -p logs_evidencerl_baseline "${WORKER_OUTPUT_DIR}" "${SIGNAL_DIR}"

# GPU health check
echo "[baseline worker ${WORKER_ID}] Checking GPU availability..."
echo "[baseline worker ${WORKER_ID}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "[baseline worker ${WORKER_ID}] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-not set}"

if ! nvidia-smi &>/dev/null; then
    echo "[ERROR] GPU not available on this node. Exiting for retry."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[baseline worker ${WORKER_ID}] GPU check passed."

# Ensure CUDA_VISIBLE_DEVICES is set
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
        echo "[baseline worker ${WORKER_ID}] Set CUDA_VISIBLE_DEVICES from SLURM_JOB_GPUS: ${CUDA_VISIBLE_DEVICES}"
    elif [ -n "${SLURM_STEP_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS}"
        echo "[baseline worker ${WORKER_ID}] Set CUDA_VISIBLE_DEVICES from SLURM_STEP_GPUS: ${CUDA_VISIBLE_DEVICES}"
    elif [ -n "${GPU_DEVICE_ORDINAL:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${GPU_DEVICE_ORDINAL}"
        echo "[baseline worker ${WORKER_ID}] Set CUDA_VISIBLE_DEVICES from GPU_DEVICE_ORDINAL: ${CUDA_VISIBLE_DEVICES}"
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

# Explicitly set CONDA_PREFIX if conda activate didn't set it
# Check common conda environment locations
if [ -z "${CONDA_PREFIX:-}" ] || [ ! -d "${CONDA_PREFIX}" ]; then
    for conda_base in \
        "$HOME/scratch/conda_envs" \
        "$HOME/.conda/envs" \
        "$HOME/miniconda3/envs" \
        "$HOME/anaconda3/envs" \
        "/storage/ice1/8/6/$USER/conda_envs"; do
        if [ -d "${conda_base}/${CONDA_ENV}" ]; then
            export CONDA_PREFIX="${conda_base}/${CONDA_ENV}"
            echo "[baseline worker ${WORKER_ID}] Set CONDA_PREFIX to: ${CONDA_PREFIX}"
            break
        fi
    done
fi

# Verify CONDA_PREFIX is set and valid
if [ -z "${CONDA_PREFIX:-}" ] || [ ! -d "${CONDA_PREFIX}" ]; then
    echo "[ERROR] Could not locate conda environment: ${CONDA_ENV}"
    echo "[ERROR] CONDA_PREFIX=${CONDA_PREFIX:-not set}"
    exit 2
fi

echo "[baseline worker ${WORKER_ID}] Using conda env: ${CONDA_ENV} at ${CONDA_PREFIX}"

# Verify python executable exists
if [ ! -x "${CONDA_PREFIX}/bin/python" ]; then
    echo "[ERROR] Python executable not found at: ${CONDA_PREFIX}/bin/python"
    exit 2
fi

# Ensure conda's libstdc++ takes precedence
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Isolate conda environment from user site-packages
# This prevents ~/.local packages from interfering
export PYTHONNOUSERSITE=1
unset PYTHONUSERBASE

# Set environment
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# HuggingFace settings - avoid API calls for missing optional files
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

# Output file for this worker
JSON_OUTPUT="${WORKER_OUTPUT_DIR}/${MODEL_ID}_baseline_vllm_worker${WORKER_ID}_v${VERSION}.json"
SIGNAL_FILE="${SIGNAL_DIR}/${MODEL_ID}_baseline_vllm_worker${WORKER_ID}.done"

# Check if already completed (idempotency)
if [ -f "${SIGNAL_FILE}" ]; then
    echo "[baseline worker ${WORKER_ID}] Already completed (signal file exists). Skipping."
    exit 0
fi

# Check if output already exists and is valid
if [ -f "${JSON_OUTPUT}" ]; then
    if "${CONDA_PREFIX}/bin/python" -c "import json; d=json.load(open('${JSON_OUTPUT}')); assert len(d.get('results',[])) > 0" 2>/dev/null; then
        echo "[baseline worker ${WORKER_ID}] Output already exists and is valid. Marking as complete."
        touch "${SIGNAL_FILE}"
        exit 0
    else
        echo "[baseline worker ${WORKER_ID}] Output exists but invalid. Re-running."
    fi
fi

echo "[baseline worker ${WORKER_ID}] Starting vLLM BASELINE generation (no-RAG)..."
echo "[baseline worker ${WORKER_ID}] MODEL_ID=${MODEL_ID}"
echo "[baseline worker ${WORKER_ID}] MODEL_PATH=${MODEL_NAME}"
echo "[baseline worker ${WORKER_ID}] PATIENT_RANGE=[${PATIENT_START}:${PATIENT_END}]"
echo "[baseline worker ${WORKER_ID}] NUM_GPUS=${NUM_GPUS}"
echo "[baseline worker ${WORKER_ID}] MAX_TOKENS=${MAX_TOKENS}"
echo "[baseline worker ${WORKER_ID}] GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "[baseline worker ${WORKER_ID}] EXTRACTOR_MODEL=${EXTRACTOR_MODEL:-same as generator}"
echo "[baseline worker ${WORKER_ID}] OUTPUT=${JSON_OUTPUT}"
echo "[baseline worker ${WORKER_ID}] MODE=BASELINE (--no-rag)"

# Build command arguments - KEY DIFFERENCE: --no-rag flag
CMD_ARGS=(
    --model-name "${MODEL_NAME}"
    --patient-data-path "${PATIENT_DATA_PATH}"
    --patient-start-idx "${PATIENT_START}"
    --patient-end-idx "${PATIENT_END}"
    --json-output "${JSON_OUTPUT}"
    --tensor-parallel-size "${NUM_GPUS}"
    --max-tokens "${MAX_TOKENS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --no-rag
)

# Add extractor model if specified
if [ -n "${EXTRACTOR_MODEL}" ]; then
    echo "[baseline worker ${WORKER_ID}] Using separate extractor model: ${EXTRACTOR_MODEL}"
    CMD_ARGS+=(--extractor-model "${EXTRACTOR_MODEL}")
fi

# Quick vLLM availability check
if ! "${CONDA_PREFIX}/bin/python" -c "import vllm" 2>/dev/null; then
    echo "[ERROR] vLLM not available in conda environment"
    exit 2
fi

# Run vLLM baseline generation
# Note: Running directly without srun since we're already in an sbatch job with GPUs allocated
# The --gpu-bind would have been handled by sbatch's --gres allocation
set +e
"${CONDA_PREFIX}/bin/python" "${PROJECT_ROOT}/run_evidence_rl_vllm.py" "${CMD_ARGS[@]}"
GEN_EXIT_CODE=$?
set -e

# Check if generation succeeded
if [ $GEN_EXIT_CODE -eq 0 ] && [ -f "${JSON_OUTPUT}" ]; then
    echo "[baseline worker ${WORKER_ID}] Baseline generation completed successfully."
    touch "${SIGNAL_FILE}"
    echo "[baseline worker ${WORKER_ID}] Signal file created: ${SIGNAL_FILE}"
else
    echo "[ERROR] Baseline generation failed for worker ${WORKER_ID} (exit code: ${GEN_EXIT_CODE})"
    exit 1
fi
