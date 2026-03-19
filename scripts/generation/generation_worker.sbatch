#!/usr/bin/env bash
#SBATCH -J erl_worker
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH -o logs_evidencerl/worker_%A_%a.out
#SBATCH -e logs_evidencerl/worker_%A_%a.err

# ============================================================================
# EvidenceRL Generation Worker
# ============================================================================
# This script processes a subset of patients for distributed generation.
# It is launched by the master job with resource overrides.
#
# Default resources (in SBATCH directives): 2x H200, 128GB RAM, 16 CPUs, 4h
# Master overrides resources based on model tier:
#   Standard models (NUM_GPUS=2):    2x H200, 128GB RAM, 16 CPUs, 4h
#   Large models (NUM_GPUS=4):       4x H200, 256GB RAM, 32 CPUs, 4h
#   Extra-large models (NUM_GPUS=8): 8x H200, 256GB RAM, 64 CPUs, 2h
#
# Model tiers (auto-detected in launcher):
#   Large (4 GPU):       Llama3-Med42-70B, gpt-oss-120b, Llama-3.3-70B, Llama-4-Scout
#   Extra-large (8 GPU): Llama-4-Maverick
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
#   FAISS_INDEX_DIR - Path to pre-built FAISS index (speeds up startup by ~5-10 min)
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

# Pre-built FAISS index (set this to skip embedding/index building, saves ~5-10 min)
FAISS_INDEX_DIR="${FAISS_INDEX_DIR:-//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/faiss_index}"

# Generation parameters
TOP_K_PRE=3
CHUNK_SIZE=200
CHUNK_OVERLAP=50

# Batch size: use provided value or auto-detect based on model and GPU count
# Conservative batch sizes for H200 GPUs (141GB VRAM each) to avoid OOM with long sequences
# Memory = Model weights (fp16) + KV cache (grows with batch × seq_len) + activations
# Batch size scales roughly linearly with GPU count
NUM_GPUS="${NUM_GPUS:-2}"

# Helper function to calculate batch size based on GPU count
# Usage: get_batch_size <base_2gpu> -> returns scaled batch for current NUM_GPUS
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
        # Llama 4 MoE models - Maverick needs more VRAM due to 128 experts
        [Ll]lama*4*[Mm]averick*)
            # Maverick: 17B active params but 128 experts total (~400B params)
            # With 8 GPUs (1128GB), use batch 64 (conservative for MoE memory patterns)
            BATCH_SIZE=$(get_batch_size 16)
            ;;
        [Ll]lama*4*[Ss]cout*)
            # Scout: 17B active, 16 experts (~109B params)
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
            echo "[worker ${WORKER_ID}] Unknown model '${MODEL_ID}', using default batch_size=${BATCH_SIZE}"
            ;;
    esac
    echo "[worker ${WORKER_ID}] Auto-detected batch_size=${BATCH_SIZE} for model=${MODEL_ID} with ${NUM_GPUS} GPUs"
fi

# Knowledge source (used only if FAISS_INDEX_DIR doesn't exist)
KNOWLEDGE_DATASET="ilyassacha/cardiologyChunks"
KNOWLEDGE_DATASET_SPLIT="train"
KNOWLEDGE_DATASET_FIELD="text"
KNOWLEDGE_DATASET_MAX_RECORDS="500000"

# Output directories
SIGNAL_DIR="${SIGNAL_DIR:-${OUTPUT_BASE_DIR}/signals}"

# Create directories
mkdir -p logs_evidencerl "${OUTPUT_BASE_DIR}" "${SIGNAL_DIR}"

# GPU health check and visibility
echo "[worker ${WORKER_ID}] Checking GPU availability..."
echo "[worker ${WORKER_ID}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "[worker ${WORKER_ID}] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-not set}"
echo "[worker ${WORKER_ID}] SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-not set}"
echo "[worker ${WORKER_ID}] GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-not set}"

if ! nvidia-smi &>/dev/null; then
    echo "[ERROR] GPU not available on this node. Exiting for retry on different node."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[worker ${WORKER_ID}] GPU check passed."

# Ensure CUDA_VISIBLE_DEVICES is set from Slurm allocation
# This fixes issues where multiple jobs on same node compete for GPUs
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
        echo "[worker ${WORKER_ID}] Set CUDA_VISIBLE_DEVICES from SLURM_JOB_GPUS: ${CUDA_VISIBLE_DEVICES}"
    elif [ -n "${SLURM_STEP_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS}"
        echo "[worker ${WORKER_ID}] Set CUDA_VISIBLE_DEVICES from SLURM_STEP_GPUS: ${CUDA_VISIBLE_DEVICES}"
    elif [ -n "${GPU_DEVICE_ORDINAL:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${GPU_DEVICE_ORDINAL}"
        echo "[worker ${WORKER_ID}] Set CUDA_VISIBLE_DEVICES from GPU_DEVICE_ORDINAL: ${CUDA_VISIBLE_DEVICES}"
    fi
fi

# Initialize conda
# Check common locations and PACE cluster-specific path
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
    # Try to find conda via command path
    CONDA_EXE_PATH="$(command -v conda 2>/dev/null || true)"
    if [ -n "$CONDA_EXE_PATH" ] && [ -x "$CONDA_EXE_PATH" ]; then
        CONDA_BASE="$("$CONDA_EXE_PATH" info --base)"
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "[ERROR] Could not find conda. Exiting."
        exit 2
    fi
fi

# Temporarily disable nounset for conda activation (some activation scripts have unbound vars)
set +u
conda activate "${CONDA_ENV}"
set -u
echo "[worker ${WORKER_ID}] Activated conda env: ${CONDA_ENV}"

# Ensure conda's libstdc++ takes precedence over system libraries
# This fixes CXXABI version mismatch issues on the cluster
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

# Output file for this worker
JSON_OUTPUT="${OUTPUT_BASE_DIR}/${MODEL_ID}_worker${WORKER_ID}_v${VERSION}.json"
SIGNAL_FILE="${SIGNAL_DIR}/${MODEL_ID}_worker${WORKER_ID}.done"

# Check if already completed (for idempotency)
if [ -f "${SIGNAL_FILE}" ]; then
    echo "[worker ${WORKER_ID}] Already completed (signal file exists). Skipping."
    exit 0
fi

# Check if output already exists and is valid
if [ -f "${JSON_OUTPUT}" ]; then
    # Verify it's a valid JSON with results
    if python3 -c "import json; d=json.load(open('${JSON_OUTPUT}')); assert len(d.get('results',[])) > 0" 2>/dev/null; then
        echo "[worker ${WORKER_ID}] Output already exists and is valid. Marking as complete."
        touch "${SIGNAL_FILE}"
        exit 0
    else
        echo "[worker ${WORKER_ID}] Output exists but invalid. Re-running."
    fi
fi

echo "[worker ${WORKER_ID}] Starting generation..."
echo "[worker ${WORKER_ID}] MODEL_ID=${MODEL_ID}"
echo "[worker ${WORKER_ID}] MODEL_PATH=${FULL_MODEL_PATH}"
echo "[worker ${WORKER_ID}] PATIENT_RANGE=[${PATIENT_START}:${PATIENT_END}]"
echo "[worker ${WORKER_ID}] NUM_GPUS=${NUM_GPUS}"
echo "[worker ${WORKER_ID}] BATCH_SIZE=${BATCH_SIZE}"
echo "[worker ${WORKER_ID}] OUTPUT=${JSON_OUTPUT}"

# Build command arguments
CMD_ARGS=(
    --model-name "${FULL_MODEL_PATH}"
    --patient-data-path "${PATIENT_DATA_PATH}"
    --patient-start-idx "${PATIENT_START}"
    --patient-end-idx "${PATIENT_END}"
    --top-k-pre "${TOP_K_PRE}"
    --batch-size "${BATCH_SIZE}"
    --chunk-size "${CHUNK_SIZE}"
    --chunk-overlap "${CHUNK_OVERLAP}"
    --json-output "${JSON_OUTPUT}"
)

# Use pre-built FAISS index if available, otherwise load from HuggingFace
if [ -d "${FAISS_INDEX_DIR}" ] && [ -f "${FAISS_INDEX_DIR}/index.faiss" ]; then
    echo "[worker ${WORKER_ID}] Using pre-built FAISS index: ${FAISS_INDEX_DIR}"
    CMD_ARGS+=(--faiss-index-dir "${FAISS_INDEX_DIR}")
else
    echo "[worker ${WORKER_ID}] No pre-built index found, loading from HuggingFace dataset"
    CMD_ARGS+=(
        --knowledge-dataset "${KNOWLEDGE_DATASET}"
        --knowledge-dataset-split "${KNOWLEDGE_DATASET_SPLIT}"
        --knowledge-dataset-text-field "${KNOWLEDGE_DATASET_FIELD}"
        --knowledge-dataset-max-records "${KNOWLEDGE_DATASET_MAX_RECORDS}"
    )
fi

# Run generation (disable errexit to capture exit code)
# Use --gpu-bind=closest to ensure proper GPU binding when multiple jobs share a node
set +e
srun --gpu-bind=closest python "${PROJECT_ROOT}/run_evidence_rl.py" "${CMD_ARGS[@]}"
GEN_EXIT_CODE=$?
set -e

# Check if generation succeeded
if [ $GEN_EXIT_CODE -eq 0 ] && [ -f "${JSON_OUTPUT}" ]; then
    echo "[worker ${WORKER_ID}] Generation completed successfully."
    touch "${SIGNAL_FILE}"
    echo "[worker ${WORKER_ID}] Signal file created: ${SIGNAL_FILE}"
else
    echo "[ERROR] Generation failed for worker ${WORKER_ID} (exit code: ${GEN_EXIT_CODE})"
    exit 1
fi
