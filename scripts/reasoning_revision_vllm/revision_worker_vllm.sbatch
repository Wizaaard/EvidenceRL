#!/usr/bin/env bash
#SBATCH -J erl_revision_vllm_worker
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH -o logs_evidencerl/revision_vllm_worker_%A_%a.out
#SBATCH -e logs_evidencerl/revision_vllm_worker_%A_%a.err

# ============================================================================
# EvidenceRL Reasoning Revision Worker (vLLM Version)
# ============================================================================
# High-performance reasoning revision using vLLM for 14-24x faster inference.
#
# Key differences from HuggingFace version:
# - Uses vLLM with tensor parallelism (2 GPUs)
# - PagedAttention for efficient memory management
# - Continuous batching for optimal throughput
# - Reduced time limit (2h vs 4h) due to faster inference
#
# Required environment variables:
#   GENERATION_JSON  - Path to generation JSON file
#   METRICS_JSON     - Path to metrics JSON file
#   OUTPUT_DIR       - Output directory for worker results
#   PATIENT_START    - Start index for this worker's patient subset
#   PATIENT_END      - End index for this worker's patient subset
#   WORKER_ID        - Worker ID
#
# Optional environment variables:
#   MODEL_NAME            - Revision model path (default: medgemma-27b-it)
#   TENSOR_PARALLEL_SIZE  - GPUs for tensor parallelism (default: 2)
#   MAX_TOKENS            - Max tokens to generate (default: 256)
#   GPU_MEMORY_UTILIZATION - Fraction of GPU memory (default: 0.90)
#   GROUNDING_MIN         - Min grounding for ambiguous range (default: -0.25)
#   GROUNDING_MAX         - Max grounding for ambiguous range (default: 0.25)
#   SEED                  - Random seed (default: 42)
#   SIGNAL_DIR            - Directory for completion signals
# ============================================================================

set -euo pipefail

# Validate required environment variables
: "${GENERATION_JSON:?Need GENERATION_JSON}"
: "${METRICS_JSON:?Need METRICS_JSON}"
: "${OUTPUT_DIR:?Need OUTPUT_DIR}"
: "${PATIENT_START:?Need PATIENT_START}"
: "${PATIENT_END:?Need PATIENT_END}"
: "${WORKER_ID:?Need WORKER_ID}"

# Configuration
CONDA_ENV="${CONDA_ENV:-evidencerl}"
PROJECT_ROOT="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/code"
MODEL_DIR="//storage/ice-shared/bmed-sp-wang/Models"

# Model defaults
MODEL_NAME="${MODEL_NAME:-${MODEL_DIR}/medgemma-27b-it}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_TOKENS="${MAX_TOKENS:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
GROUNDING_MIN="${GROUNDING_MIN:--0.25}"
GROUNDING_MAX="${GROUNDING_MAX:-0.25}"
SEED="${SEED:-42}"

# NLI model settings for grounding recalculation
NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-base}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
REWARD_WEIGHT_GROUNDING="${REWARD_WEIGHT_GROUNDING:-0.5}"
MAX_K="${MAX_K:-5}"
SKIP_NLI="${SKIP_NLI:-}"  # Set to "1" or "true" to skip NLI recalculation

# Signal directory
SIGNAL_DIR="${SIGNAL_DIR:-${OUTPUT_DIR}/signals}"

# Create directories
mkdir -p logs_evidencerl "${OUTPUT_DIR}" "${SIGNAL_DIR}"

# GPU health check and visibility
echo "[vllm worker ${WORKER_ID}] Checking GPU availability..."
echo "[vllm worker ${WORKER_ID}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "[vllm worker ${WORKER_ID}] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-not set}"
echo "[vllm worker ${WORKER_ID}] SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-not set}"
echo "[vllm worker ${WORKER_ID}] GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-not set}"

if ! nvidia-smi &>/dev/null; then
    echo "[ERROR] GPU not available on this node. Exiting for retry on different node."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[vllm worker ${WORKER_ID}] GPU check passed."

# Ensure CUDA_VISIBLE_DEVICES is set from Slurm allocation
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
# FLASH_ATTN doesn't support multimodal models like medgemma-27b-it

# Output files for this worker
GEN_OUTPUT="${OUTPUT_DIR}/generation_worker${WORKER_ID}.json"
METRICS_OUTPUT="${OUTPUT_DIR}/metrics_worker${WORKER_ID}.json"
SIGNAL_FILE="${SIGNAL_DIR}/revision_vllm_worker${WORKER_ID}.done"

# Check if already completed (idempotency)
if [ -f "${SIGNAL_FILE}" ]; then
    echo "[vllm worker ${WORKER_ID}] Already completed (signal file exists). Skipping."
    exit 0
fi

# Check if output already exists and is valid
if [ -f "${GEN_OUTPUT}" ] && [ -f "${METRICS_OUTPUT}" ]; then
    if python3 -c "import json; d=json.load(open('${GEN_OUTPUT}')); assert len(d.get('results',[])) >= 0" 2>/dev/null; then
        echo "[vllm worker ${WORKER_ID}] Output already exists and is valid. Marking as complete."
        touch "${SIGNAL_FILE}"
        exit 0
    else
        echo "[vllm worker ${WORKER_ID}] Output exists but invalid. Re-running."
    fi
fi

echo "[vllm worker ${WORKER_ID}] Starting reasoning revision with vLLM..."
echo "[vllm worker ${WORKER_ID}] GENERATION_JSON=${GENERATION_JSON}"
echo "[vllm worker ${WORKER_ID}] METRICS_JSON=${METRICS_JSON}"
echo "[vllm worker ${WORKER_ID}] MODEL_NAME=${MODEL_NAME}"
echo "[vllm worker ${WORKER_ID}] PATIENT_RANGE=[${PATIENT_START}:${PATIENT_END}]"
echo "[vllm worker ${WORKER_ID}] TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
echo "[vllm worker ${WORKER_ID}] MAX_TOKENS=${MAX_TOKENS}"
echo "[vllm worker ${WORKER_ID}] GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "[vllm worker ${WORKER_ID}] GROUNDING_RANGE=[${GROUNDING_MIN}, ${GROUNDING_MAX}]"
echo "[vllm worker ${WORKER_ID}] GEN_OUTPUT=${GEN_OUTPUT}"
echo "[vllm worker ${WORKER_ID}] METRICS_OUTPUT=${METRICS_OUTPUT}"

# Build command arguments
CMD_ARGS=(
    --generation-json "${GENERATION_JSON}"
    --metrics-json "${METRICS_JSON}"
    --output-dir "${OUTPUT_DIR}"
    --model-name "${MODEL_NAME}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --max-tokens "${MAX_TOKENS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --grounding-min "${GROUNDING_MIN}"
    --grounding-max "${GROUNDING_MAX}"
    --seed "${SEED}"
    --patient-start "${PATIENT_START}"
    --patient-end "${PATIENT_END}"
    --worker-id "${WORKER_ID}"
    --nli-model "${NLI_MODEL}"
    --embedding-model "${EMBEDDING_MODEL}"
    --reward-weight-grounding "${REWARD_WEIGHT_GROUNDING}"
    --max-k "${MAX_K}"
)

# Add --skip-nli flag if requested
if [ -n "${SKIP_NLI}" ] && [ "${SKIP_NLI}" != "0" ] && [ "${SKIP_NLI}" != "false" ]; then
    echo "[vllm worker ${WORKER_ID}] NLI recalculation will be SKIPPED"
    CMD_ARGS+=(--skip-nli)
else
    echo "[vllm worker ${WORKER_ID}] NLI_MODEL=${NLI_MODEL}"
    echo "[vllm worker ${WORKER_ID}] EMBEDDING_MODEL=${EMBEDDING_MODEL}"
fi

# Run reasoning revision with vLLM
set +e
srun --gpu-bind=closest python "${PROJECT_ROOT}/scripts/reasoning_revision_vllm/revision_worker_vllm.py" "${CMD_ARGS[@]}"
REVISION_EXIT_CODE=$?
set -e

# Check if computation succeeded
if [ $REVISION_EXIT_CODE -eq 0 ] && [ -f "${GEN_OUTPUT}" ]; then
    echo "[vllm worker ${WORKER_ID}] Reasoning revision completed successfully."
    touch "${SIGNAL_FILE}"
    echo "[vllm worker ${WORKER_ID}] Signal file created: ${SIGNAL_FILE}"
else
    echo "[ERROR] Reasoning revision failed for worker ${WORKER_ID} (exit code: ${REVISION_EXIT_CODE})"
    exit 1
fi
