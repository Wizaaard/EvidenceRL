#!/usr/bin/env bash
#SBATCH -J erl_sc_worker
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH -o logs_evidencerl_sc/generation_sc_worker_%A_%a.out
#SBATCH -e logs_evidencerl_sc/generation_sc_worker_%A_%a.err

# ============================================================================
# EvidenceRL Self-Consistency Generation Worker (vLLM Version)
# ============================================================================
# SC worker: generates N diverse samples per patient and aggregates via
# embedding-based majority voting.  No retrieval is needed.
#
# Time limit is 6h (generating N=10 samples per patient via vLLM n parameter).
#
# Required environment variables:
#   MODEL_ID          - Short identifier for the model
#   MODEL_NAME        - Full path to the model
#   PATIENT_START     - Start index for this worker's patient subset
#   PATIENT_END       - End index for this worker's patient subset
#   WORKER_ID         - Worker ID
#   VERSION           - Version string for output naming
#   WORKER_OUTPUT_DIR - Directory for worker outputs
#
# Optional environment variables:
#   SIGNAL_DIR              - Directory for completion signals
#   NUM_GPUS                - Number of GPUs for tensor parallelism (default: 2)
#   SC_NUM_SAMPLES          - Number of diverse samples per patient (default: 10)
#   SC_TEMPERATURE          - Temperature for diverse sampling (default: 0.9)
#   SC_SIMILARITY_THRESHOLD - Cosine sim threshold for clustering (default: 0.85)
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

# Self-Consistency parameters
SC_NUM_SAMPLES="${SC_NUM_SAMPLES:-10}"
SC_TEMPERATURE="${SC_TEMPERATURE:-0.9}"
SC_SIMILARITY_THRESHOLD="${SC_SIMILARITY_THRESHOLD:-0.85}"

# Signal directory
SIGNAL_DIR="${SIGNAL_DIR:-}"

# Create directories
mkdir -p logs_evidencerl_sc "${WORKER_OUTPUT_DIR}" "${SIGNAL_DIR}"

# GPU health check
echo "[sc worker ${WORKER_ID}] Checking GPU availability..."
echo "[sc worker ${WORKER_ID}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "[sc worker ${WORKER_ID}] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-not set}"

if ! nvidia-smi &>/dev/null; then
    echo "[ERROR] GPU not available on this node. Exiting for retry."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[sc worker ${WORKER_ID}] GPU check passed."

# Ensure CUDA_VISIBLE_DEVICES is set
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
        echo "[sc worker ${WORKER_ID}] Set CUDA_VISIBLE_DEVICES from SLURM_JOB_GPUS: ${CUDA_VISIBLE_DEVICES}"
    elif [ -n "${SLURM_STEP_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS}"
        echo "[sc worker ${WORKER_ID}] Set CUDA_VISIBLE_DEVICES from SLURM_STEP_GPUS: ${CUDA_VISIBLE_DEVICES}"
    elif [ -n "${GPU_DEVICE_ORDINAL:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${GPU_DEVICE_ORDINAL}"
        echo "[sc worker ${WORKER_ID}] Set CUDA_VISIBLE_DEVICES from GPU_DEVICE_ORDINAL: ${CUDA_VISIBLE_DEVICES}"
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
if [ -z "${CONDA_PREFIX:-}" ] || [ ! -d "${CONDA_PREFIX}" ]; then
    for conda_base in \
        "$HOME/scratch/conda_envs" \
        "$HOME/.conda/envs" \
        "$HOME/miniconda3/envs" \
        "$HOME/anaconda3/envs" \
        "/storage/ice1/8/6/$USER/conda_envs"; do
        if [ -d "${conda_base}/${CONDA_ENV}" ]; then
            export CONDA_PREFIX="${conda_base}/${CONDA_ENV}"
            echo "[sc worker ${WORKER_ID}] Set CONDA_PREFIX to: ${CONDA_PREFIX}"
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

echo "[sc worker ${WORKER_ID}] Using conda env: ${CONDA_ENV} at ${CONDA_PREFIX}"

# Verify python executable exists
if [ ! -x "${CONDA_PREFIX}/bin/python" ]; then
    echo "[ERROR] Python executable not found at: ${CONDA_PREFIX}/bin/python"
    exit 2
fi

# Ensure conda's libstdc++ takes precedence
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Isolate conda environment from user site-packages
export PYTHONNOUSERSITE=1
unset PYTHONUSERBASE

# Set environment
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# HuggingFace settings
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

# Use per-worker vLLM cache root to prevent torch.compile cache corruption from concurrent writes
export VLLM_CACHE_ROOT="/tmp/vllm_cache_${SLURM_JOB_ID}"
echo "[sc worker ${WORKER_ID}] VLLM_CACHE_ROOT: ${VLLM_CACHE_ROOT}"

# Output file for this worker
JSON_OUTPUT="${WORKER_OUTPUT_DIR}/${MODEL_ID}_worker${WORKER_ID}_v${VERSION}.json"
SIGNAL_FILE="${SIGNAL_DIR}/${MODEL_ID}_sc_worker${WORKER_ID}.done"

# Check if already completed (idempotency)
if [ -f "${SIGNAL_FILE}" ]; then
    echo "[sc worker ${WORKER_ID}] Already completed (signal file exists). Skipping."
    exit 0
fi

# Check if output already exists and is valid
if [ -f "${JSON_OUTPUT}" ]; then
    if "${CONDA_PREFIX}/bin/python" -c "import json; d=json.load(open('${JSON_OUTPUT}')); assert len(d.get('results',[])) > 0" 2>/dev/null; then
        echo "[sc worker ${WORKER_ID}] Output already exists and is valid. Marking as complete."
        touch "${SIGNAL_FILE}"
        exit 0
    else
        echo "[sc worker ${WORKER_ID}] Output exists but invalid. Re-running."
    fi
fi

echo "[sc worker ${WORKER_ID}] Starting Self-Consistency vLLM generation..."
echo "[sc worker ${WORKER_ID}] MODEL_ID=${MODEL_ID}"
echo "[sc worker ${WORKER_ID}] MODEL_PATH=${MODEL_NAME}"
echo "[sc worker ${WORKER_ID}] PATIENT_RANGE=[${PATIENT_START}:${PATIENT_END}]"
echo "[sc worker ${WORKER_ID}] NUM_GPUS=${NUM_GPUS}"
echo "[sc worker ${WORKER_ID}] MAX_TOKENS=${MAX_TOKENS}"
echo "[sc worker ${WORKER_ID}] GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "[sc worker ${WORKER_ID}] SC_NUM_SAMPLES=${SC_NUM_SAMPLES}"
echo "[sc worker ${WORKER_ID}] SC_TEMPERATURE=${SC_TEMPERATURE}"
echo "[sc worker ${WORKER_ID}] SC_SIMILARITY_THRESHOLD=${SC_SIMILARITY_THRESHOLD}"
echo "[sc worker ${WORKER_ID}] OUTPUT=${JSON_OUTPUT}"

# Build command arguments
CMD_ARGS=(
    --model-name "${MODEL_NAME}"
    --patient-data-path "${PATIENT_DATA_PATH}"
    --patient-start-idx "${PATIENT_START}"
    --patient-end-idx "${PATIENT_END}"
    --json-output "${JSON_OUTPUT}"
    --tensor-parallel-size "${NUM_GPUS}"
    --max-tokens "${MAX_TOKENS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --self-consistency
    --sc-num-samples "${SC_NUM_SAMPLES}"
    --sc-temperature "${SC_TEMPERATURE}"
    --sc-similarity-threshold "${SC_SIMILARITY_THRESHOLD}"
)

# Add extractor model if specified
if [ -n "${EXTRACTOR_MODEL:-}" ]; then
    echo "[sc worker ${WORKER_ID}] Using separate extractor model: ${EXTRACTOR_MODEL}"
    CMD_ARGS+=(--extractor-model "${EXTRACTOR_MODEL}")
fi

# Add SC embedding model if specified (default: FremyCompany/BioLORD-2023)
if [ -n "${SC_EMBEDDING_MODEL:-}" ]; then
    echo "[sc worker ${WORKER_ID}] Using SC embedding model: ${SC_EMBEDDING_MODEL}"
    CMD_ARGS+=(--sc-embedding-model "${SC_EMBEDDING_MODEL}")
fi

# Quick vLLM availability check
if ! "${CONDA_PREFIX}/bin/python" -c "import vllm" 2>/dev/null; then
    echo "[ERROR] vLLM not available in conda environment"
    exit 2
fi

# Run Self-Consistency vLLM generation
set +e
"${CONDA_PREFIX}/bin/python" "${PROJECT_ROOT}/run_evidence_rl_vllm.py" "${CMD_ARGS[@]}"
GEN_EXIT_CODE=$?
set -e

# Check if generation succeeded
if [ $GEN_EXIT_CODE -eq 0 ] && [ -f "${JSON_OUTPUT}" ]; then
    echo "[sc worker ${WORKER_ID}] Self-Consistency generation completed successfully."
    touch "${SIGNAL_FILE}"
    echo "[sc worker ${WORKER_ID}] Signal file created: ${SIGNAL_FILE}"
else
    echo "[ERROR] Self-Consistency generation failed for worker ${WORKER_ID} (exit code: ${GEN_EXIT_CODE})"
    exit 1
fi
