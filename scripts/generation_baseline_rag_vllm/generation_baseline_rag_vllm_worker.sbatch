#!/usr/bin/env bash
#SBATCH -J erl_baseline_rag_vllm_worker
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH -o logs_evidencerl_baseline_rag/generation_baseline_rag_vllm_worker_%A_%a.out
#SBATCH -e logs_evidencerl_baseline_rag/generation_baseline_rag_vllm_worker_%A_%a.err

# ============================================================================
# EvidenceRL Baseline RAG Generation Worker (vLLM Version)
# ============================================================================
# High-performance BASELINE RAG generation (with retrieval) using vLLM.
# Uses FAISS index for document retrieval on held-out test patients.
#
# Required environment variables:
#   MODEL_ID        - Short identifier for the model
#   MODEL_NAME      - Full path to the model
#   PATIENT_START   - Start index for this worker's patient subset
#   PATIENT_END     - End index for this worker's patient subset
#   WORKER_ID       - Worker ID
#   VERSION         - Version string for output naming
#   OUTPUT_BASE_DIR - Base directory for outputs
#   FAISS_INDEX_DIR - Path to pre-built FAISS index
# ============================================================================

set -euo pipefail

# Validate required environment variables
: "${MODEL_ID:?Need MODEL_ID}"
: "${MODEL_NAME:?Need MODEL_NAME}"
: "${PATIENT_START:?Need PATIENT_START}"
: "${PATIENT_END:?Need PATIENT_END}"
: "${WORKER_ID:?Need WORKER_ID}"
: "${VERSION:=1.0}"
: "${OUTPUT_BASE_DIR:?Need OUTPUT_BASE_DIR}"
: "${FAISS_INDEX_DIR:?Need FAISS_INDEX_DIR}"

# Configuration
CONDA_ENV="${CONDA_ENV:-evidencerl}"
PROJECT_ROOT="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/code"
PATIENT_DATA_PATH="//storage/ice-shared/bmed-sp-wang/Ben/Data/physionet.org/files/mimic-iv-ext-cardiac-disease/1.0.0"

# vLLM-specific settings
NUM_GPUS="${NUM_GPUS:-2}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# Generation parameters (same as regular RAG generation)
TOP_K_PRE=3
CHUNK_SIZE=200
CHUNK_OVERLAP=50

# Signal directory
SIGNAL_DIR="${SIGNAL_DIR:-${OUTPUT_BASE_DIR}/signals}"

# Create directories
mkdir -p logs_evidencerl_baseline_rag "${OUTPUT_BASE_DIR}" "${SIGNAL_DIR}"

# GPU health check
echo "[vllm baseline_rag worker ${WORKER_ID}] Checking GPU availability..."
echo "[vllm baseline_rag worker ${WORKER_ID}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "[vllm baseline_rag worker ${WORKER_ID}] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-not set}"

if ! nvidia-smi &>/dev/null; then
    echo "[ERROR] GPU not available on this node. Exiting for retry."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[vllm baseline_rag worker ${WORKER_ID}] GPU check passed."

# Ensure CUDA_VISIBLE_DEVICES is set
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
        echo "[vllm baseline_rag worker ${WORKER_ID}] Set CUDA_VISIBLE_DEVICES from SLURM_JOB_GPUS: ${CUDA_VISIBLE_DEVICES}"
    elif [ -n "${SLURM_STEP_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS}"
        echo "[vllm baseline_rag worker ${WORKER_ID}] Set CUDA_VISIBLE_DEVICES from SLURM_STEP_GPUS: ${CUDA_VISIBLE_DEVICES}"
    elif [ -n "${GPU_DEVICE_ORDINAL:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${GPU_DEVICE_ORDINAL}"
        echo "[vllm baseline_rag worker ${WORKER_ID}] Set CUDA_VISIBLE_DEVICES from GPU_DEVICE_ORDINAL: ${CUDA_VISIBLE_DEVICES}"
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
echo "[vllm baseline_rag worker ${WORKER_ID}] Activated conda env: ${CONDA_ENV}"

# Ensure conda's libstdc++ takes precedence
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Set environment
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Output file for this worker - includes "baseline_rag_vllm" in name
JSON_OUTPUT="${OUTPUT_BASE_DIR}/${MODEL_ID}_baseline_rag_vllm_worker${WORKER_ID}_v${VERSION}.json"
SIGNAL_FILE="${SIGNAL_DIR}/${MODEL_ID}_baseline_rag_vllm_worker${WORKER_ID}.done"

# Check if already completed (idempotency)
if [ -f "${SIGNAL_FILE}" ]; then
    echo "[vllm baseline_rag worker ${WORKER_ID}] Already completed (signal file exists). Skipping."
    exit 0
fi

# Check if output already exists and is valid
if [ -f "${JSON_OUTPUT}" ]; then
    if python3 -c "import json; d=json.load(open('${JSON_OUTPUT}')); assert len(d.get('results',[])) > 0" 2>/dev/null; then
        echo "[vllm baseline_rag worker ${WORKER_ID}] Output already exists and is valid. Marking as complete."
        touch "${SIGNAL_FILE}"
        exit 0
    else
        echo "[vllm baseline_rag worker ${WORKER_ID}] Output exists but invalid. Re-running."
    fi
fi

echo "[vllm baseline_rag worker ${WORKER_ID}] Starting vLLM BASELINE RAG generation..."
echo "[vllm baseline_rag worker ${WORKER_ID}] MODE=BASELINE RAG (with retrieval)"
echo "[vllm baseline_rag worker ${WORKER_ID}] MODEL_ID=${MODEL_ID}"
echo "[vllm baseline_rag worker ${WORKER_ID}] MODEL_PATH=${MODEL_NAME}"
echo "[vllm baseline_rag worker ${WORKER_ID}] FAISS_INDEX=${FAISS_INDEX_DIR}"
echo "[vllm baseline_rag worker ${WORKER_ID}] PATIENT_RANGE=[${PATIENT_START}:${PATIENT_END}]"
echo "[vllm baseline_rag worker ${WORKER_ID}] NUM_GPUS=${NUM_GPUS}"
echo "[vllm baseline_rag worker ${WORKER_ID}] MAX_TOKENS=${MAX_TOKENS}"
echo "[vllm baseline_rag worker ${WORKER_ID}] GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "[vllm baseline_rag worker ${WORKER_ID}] OUTPUT=${JSON_OUTPUT}"

# Build command arguments - uses FAISS index for RAG (no --no-rag flag)
CMD_ARGS=(
    --model-name "${MODEL_NAME}"
    --patient-data-path "${PATIENT_DATA_PATH}"
    --patient-start-idx "${PATIENT_START}"
    --patient-end-idx "${PATIENT_END}"
    --top-k-pre "${TOP_K_PRE}"
    --chunk-size "${CHUNK_SIZE}"
    --chunk-overlap "${CHUNK_OVERLAP}"
    --json-output "${JSON_OUTPUT}"
    --tensor-parallel-size "${NUM_GPUS}"
    --max-tokens "${MAX_TOKENS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --faiss-index-dir "${FAISS_INDEX_DIR}"
)

# Run vLLM RAG generation
set +e
srun --gpu-bind=closest python "${PROJECT_ROOT}/run_evidence_rl_vllm.py" "${CMD_ARGS[@]}"
GEN_EXIT_CODE=$?
set -e

# Check if generation succeeded
if [ $GEN_EXIT_CODE -eq 0 ] && [ -f "${JSON_OUTPUT}" ]; then
    echo "[vllm baseline_rag worker ${WORKER_ID}] Baseline RAG generation completed successfully."
    touch "${SIGNAL_FILE}"
    echo "[vllm baseline_rag worker ${WORKER_ID}] Signal file created: ${SIGNAL_FILE}"
else
    echo "[ERROR] Baseline RAG generation failed for worker ${WORKER_ID} (exit code: ${GEN_EXIT_CODE})"
    exit 1
fi
