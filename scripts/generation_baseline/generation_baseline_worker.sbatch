#!/usr/bin/env bash
#SBATCH -J erl_baseline_worker
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH -o logs_evidencerl_baseline/worker_%A_%a.out
#SBATCH -e logs_evidencerl_baseline/worker_%A_%a.err

# ============================================================================
# EvidenceRL Baseline Generation Worker (No-RAG)
# ============================================================================
# This script processes a subset of patients for distributed BASELINE generation.
# Uses --no-rag flag for zero-shot generation WITHOUT retrieval.
#
# This is FASTER than RAG generation because:
# - No embedding computation
# - No FAISS index loading
# - No document retrieval
#
# Required environment variables (passed via --export):
#   MODEL_ID        - Short identifier for the model (e.g., "gemma-3-4b")
#   MODEL_NAME      - Full path to the model
#   PATIENT_START   - Start index for this worker's patient subset
#   PATIENT_END     - End index for this worker's patient subset
#   WORKER_ID       - Worker ID
#   VERSION         - Version string for output naming
#   OUTPUT_BASE_DIR - Base directory for outputs
#
# Optional environment variables:
#   SIGNAL_DIR      - Directory for completion signals (default: OUTPUT_BASE_DIR/signals)
#   BATCH_SIZE      - Override auto-detected batch size
#   NUM_GPUS        - Number of GPUs (for batch size auto-detection, default: 2)
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

# Configuration
CONDA_ENV="${CONDA_ENV:-evidencerl}"
PROJECT_ROOT="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/code"
PATIENT_DATA_PATH="//storage/ice-shared/bmed-sp-wang/Ben/Data/physionet.org/files/mimic-iv-ext-cardiac-disease/1.0.0"
MODEL_DIR="//storage/ice-shared/bmed-sp-wang/Models"

# Batch size: use provided value or auto-detect based on model and GPU count
# Conservative batch sizes for H200 GPUs (141GB VRAM each) to avoid OOM with long sequences
NUM_GPUS="${NUM_GPUS:-2}"

# Helper function to calculate batch size based on GPU count
get_batch_size() {
    local base=$1  # Batch size for 2 GPUs
    echo $((base * NUM_GPUS / 2))
}

if [ -z "${BATCH_SIZE:-}" ]; then
    # Auto-detect batch size based on model ID
    # Base values are for 2 GPUs, scaled by GPU count
    case "${MODEL_ID}" in
        # Gemma family (including MedGemma)
        *[Gg]emma*270[Mm]*)
            BATCH_SIZE=$(get_batch_size 256)
            ;;
        *[Gg]emma*1[Bb]*)
            BATCH_SIZE=$(get_batch_size 128)
            ;;
        *[Gg]emma*4[Bb]*)
            BATCH_SIZE=$(get_batch_size 64)
            ;;
        *[Gg]emma*12[Bb]*)
            BATCH_SIZE=$(get_batch_size 24)
            ;;
        *[Gg]emma*27[Bb]*)
            BATCH_SIZE=$(get_batch_size 12)
            ;;
        # Llama family (including Med42)
        [Ll]lama*3.2*1[Bb]*|[Ll]lama*1[Bb]*)
            BATCH_SIZE=$(get_batch_size 128)
            ;;
        *[Mm]ed42*8[Bb]*|[Ll]lama*8[Bb]*)
            BATCH_SIZE=$(get_batch_size 32)
            ;;
        *[Mm]ed42*70[Bb]*|[Ll]lama*70[Bb]*)
            BATCH_SIZE=$(get_batch_size 4)
            ;;
        # Llama 4 MoE models
        [Ll]lama*4*[Mm]averick*)
            BATCH_SIZE=$(get_batch_size 16)
            ;;
        [Ll]lama*4*[Ss]cout*)
            BATCH_SIZE=$(get_batch_size 16)
            ;;
        # GPT-OSS family
        *gpt*oss*20[Bb]*)
            BATCH_SIZE=$(get_batch_size 16)
            ;;
        *gpt*oss*120[Bb]*)
            BATCH_SIZE=$(get_batch_size 2)
            ;;
        # Qwen family
        [Qq]wen*7[Bb]*)
            BATCH_SIZE=$(get_batch_size 32)
            ;;
        [Qq]wen*14[Bb]*)
            BATCH_SIZE=$(get_batch_size 16)
            ;;
        [Qq]wen*72[Bb]*)
            BATCH_SIZE=$(get_batch_size 4)
            ;;
        # Mistral family
        [Mm]istral*7[Bb]*)
            BATCH_SIZE=$(get_batch_size 32)
            ;;
        [Mm]ixtral*)
            BATCH_SIZE=$(get_batch_size 8)
            ;;
        *)
            # Default conservative batch size (scales with GPUs)
            BATCH_SIZE=$(get_batch_size 16)
            echo "[baseline worker ${WORKER_ID}] Unknown model '${MODEL_ID}', using default batch_size=${BATCH_SIZE}"
            ;;
    esac
    echo "[baseline worker ${WORKER_ID}] Auto-detected batch_size=${BATCH_SIZE} for model=${MODEL_ID} with ${NUM_GPUS} GPUs"
fi

# Output directories
SIGNAL_DIR="${SIGNAL_DIR:-${OUTPUT_BASE_DIR}/signals}"

# Create directories
mkdir -p logs_evidencerl_baseline "${OUTPUT_BASE_DIR}" "${SIGNAL_DIR}"

# GPU health check and visibility
echo "[baseline worker ${WORKER_ID}] Checking GPU availability..."
echo "[baseline worker ${WORKER_ID}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "[baseline worker ${WORKER_ID}] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-not set}"

if ! nvidia-smi &>/dev/null; then
    echo "[ERROR] GPU not available on this node. Exiting for retry on different node."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[baseline worker ${WORKER_ID}] GPU check passed."

# Ensure CUDA_VISIBLE_DEVICES is set from Slurm allocation
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

# Temporarily disable nounset for conda activation
set +u
conda activate "${CONDA_ENV}"
set -u
echo "[baseline worker ${WORKER_ID}] Activated conda env: ${CONDA_ENV}"

# Ensure conda's libstdc++ takes precedence over system libraries
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Set environment
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Determine model path
if [[ "${MODEL_NAME}" == /* ]]; then
    # Absolute path provided
    FULL_MODEL_PATH="${MODEL_NAME}"
else
    # Relative to MODEL_DIR
    FULL_MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"
fi

# Output file for this worker - includes "baseline" in name
JSON_OUTPUT="${OUTPUT_BASE_DIR}/${MODEL_ID}_baseline_worker${WORKER_ID}_v${VERSION}.json"
SIGNAL_FILE="${SIGNAL_DIR}/${MODEL_ID}_baseline_worker${WORKER_ID}.done"

# Check if already completed (for idempotency)
if [ -f "${SIGNAL_FILE}" ]; then
    echo "[baseline worker ${WORKER_ID}] Already completed (signal file exists). Skipping."
    exit 0
fi

# Check if output already exists and is valid
if [ -f "${JSON_OUTPUT}" ]; then
    if python3 -c "import json; d=json.load(open('${JSON_OUTPUT}')); assert len(d.get('results',[])) > 0" 2>/dev/null; then
        echo "[baseline worker ${WORKER_ID}] Output already exists and is valid. Marking as complete."
        touch "${SIGNAL_FILE}"
        exit 0
    else
        echo "[baseline worker ${WORKER_ID}] Output exists but invalid. Re-running."
    fi
fi

echo "[baseline worker ${WORKER_ID}] Starting BASELINE generation (no-RAG)..."
echo "[baseline worker ${WORKER_ID}] MODEL_ID=${MODEL_ID}"
echo "[baseline worker ${WORKER_ID}] MODEL_PATH=${FULL_MODEL_PATH}"
echo "[baseline worker ${WORKER_ID}] PATIENT_RANGE=[${PATIENT_START}:${PATIENT_END}]"
echo "[baseline worker ${WORKER_ID}] NUM_GPUS=${NUM_GPUS}"
echo "[baseline worker ${WORKER_ID}] BATCH_SIZE=${BATCH_SIZE}"
echo "[baseline worker ${WORKER_ID}] OUTPUT=${JSON_OUTPUT}"
echo "[baseline worker ${WORKER_ID}] MODE=BASELINE (--no-rag)"

# Build command arguments - KEY DIFFERENCE: --no-rag flag instead of knowledge sources
CMD_ARGS=(
    --model-name "${FULL_MODEL_PATH}"
    --patient-data-path "${PATIENT_DATA_PATH}"
    --patient-start-idx "${PATIENT_START}"
    --patient-end-idx "${PATIENT_END}"
    --batch-size "${BATCH_SIZE}"
    --json-output "${JSON_OUTPUT}"
    --no-rag  # THIS IS THE KEY FLAG FOR BASELINE MODE
)

# Run generation (disable errexit to capture exit code)
set +e
srun --gpu-bind=closest python "${PROJECT_ROOT}/run_evidence_rl.py" "${CMD_ARGS[@]}"
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
