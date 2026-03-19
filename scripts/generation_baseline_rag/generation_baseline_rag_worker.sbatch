#!/usr/bin/env bash
#SBATCH -J erl_baseline_rag_worker
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH -o logs_evidencerl_baseline_rag/worker_%A_%a.out
#SBATCH -e logs_evidencerl_baseline_rag/worker_%A_%a.err

# ============================================================================
# EvidenceRL Baseline RAG Generation Worker
# ============================================================================
# Runs generation WITH RAG on held-out test patients.
# Uses FAISS index for document retrieval.
#
# Required environment variables:
#   MODEL_ID        - Short identifier for the model
#   MODEL_NAME      - Full path to the model
#   PATIENT_START   - Start index for patient subset
#   PATIENT_END     - End index for patient subset
#   WORKER_ID       - Worker ID
#   VERSION         - Version string
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
MODEL_DIR="//storage/ice-shared/bmed-sp-wang/Models"

# Generation parameters (same as regular RAG generation)
TOP_K_PRE=3
CHUNK_SIZE=200
CHUNK_OVERLAP=50

NUM_GPUS="${NUM_GPUS:-2}"

# Batch size auto-detection
get_batch_size() {
    local base=$1
    echo $((base * NUM_GPUS / 2))
}

if [ -z "${BATCH_SIZE:-}" ]; then
    case "${MODEL_ID}" in
        *[Gg]emma*270[Mm]*) BATCH_SIZE=$(get_batch_size 256) ;;
        *[Gg]emma*1[Bb]*) BATCH_SIZE=$(get_batch_size 128) ;;
        *[Gg]emma*4[Bb]*) BATCH_SIZE=$(get_batch_size 64) ;;
        *[Gg]emma*12[Bb]*) BATCH_SIZE=$(get_batch_size 24) ;;
        *[Gg]emma*27[Bb]*) BATCH_SIZE=$(get_batch_size 12) ;;
        [Ll]lama*3.2*1[Bb]*|[Ll]lama*1[Bb]*) BATCH_SIZE=$(get_batch_size 128) ;;
        *[Mm]ed42*8[Bb]*|[Ll]lama*8[Bb]*) BATCH_SIZE=$(get_batch_size 32) ;;
        *[Mm]ed42*70[Bb]*|[Ll]lama*70[Bb]*) BATCH_SIZE=$(get_batch_size 4) ;;
        [Ll]lama*4*[Mm]averick*) BATCH_SIZE=$(get_batch_size 16) ;;
        [Ll]lama*4*[Ss]cout*) BATCH_SIZE=$(get_batch_size 16) ;;
        *gpt*oss*20[Bb]*) BATCH_SIZE=$(get_batch_size 16) ;;
        *gpt*oss*120[Bb]*) BATCH_SIZE=$(get_batch_size 2) ;;
        *) BATCH_SIZE=$(get_batch_size 16) ;;
    esac
    echo "[baseline_rag worker ${WORKER_ID}] Auto-detected batch_size=${BATCH_SIZE}"
fi

# Output directories
SIGNAL_DIR="${SIGNAL_DIR:-${OUTPUT_BASE_DIR}/signals}"
mkdir -p logs_evidencerl_baseline_rag "${OUTPUT_BASE_DIR}" "${SIGNAL_DIR}"

# GPU check
echo "[baseline_rag worker ${WORKER_ID}] Checking GPU availability..."
if ! nvidia-smi &>/dev/null; then
    echo "[ERROR] GPU not available. Exiting for retry."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Set CUDA_VISIBLE_DEVICES
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
    elif [ -n "${SLURM_STEP_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS}"
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
        echo "[ERROR] Could not find conda."
        exit 2
    fi
fi

set +u
conda activate "${CONDA_ENV}"
set -u
echo "[baseline_rag worker ${WORKER_ID}] Activated conda env: ${CONDA_ENV}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Model path
if [[ "${MODEL_NAME}" == /* ]]; then
    FULL_MODEL_PATH="${MODEL_NAME}"
else
    FULL_MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"
fi

# Output file - includes "baseline_rag" in name
JSON_OUTPUT="${OUTPUT_BASE_DIR}/${MODEL_ID}_baseline_rag_worker${WORKER_ID}_v${VERSION}.json"
SIGNAL_FILE="${SIGNAL_DIR}/${MODEL_ID}_baseline_rag_worker${WORKER_ID}.done"

# Check if already completed
if [ -f "${SIGNAL_FILE}" ]; then
    echo "[baseline_rag worker ${WORKER_ID}] Already completed. Skipping."
    exit 0
fi

if [ -f "${JSON_OUTPUT}" ]; then
    if python3 -c "import json; d=json.load(open('${JSON_OUTPUT}')); assert len(d.get('results',[])) > 0" 2>/dev/null; then
        echo "[baseline_rag worker ${WORKER_ID}] Output already valid. Marking complete."
        touch "${SIGNAL_FILE}"
        exit 0
    fi
fi

echo "[baseline_rag worker ${WORKER_ID}] Starting BASELINE RAG generation..."
echo "[baseline_rag worker ${WORKER_ID}] MODE=BASELINE RAG (with retrieval)"
echo "[baseline_rag worker ${WORKER_ID}] MODEL_ID=${MODEL_ID}"
echo "[baseline_rag worker ${WORKER_ID}] FAISS_INDEX=${FAISS_INDEX_DIR}"
echo "[baseline_rag worker ${WORKER_ID}] PATIENT_RANGE=[${PATIENT_START}:${PATIENT_END}]"
echo "[baseline_rag worker ${WORKER_ID}] BATCH_SIZE=${BATCH_SIZE}"
echo "[baseline_rag worker ${WORKER_ID}] OUTPUT=${JSON_OUTPUT}"

# Build command - uses FAISS index for RAG (no --no-rag flag)
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
    --faiss-index-dir "${FAISS_INDEX_DIR}"
)

# Run generation
set +e
srun --gpu-bind=closest python "${PROJECT_ROOT}/run_evidence_rl.py" "${CMD_ARGS[@]}"
GEN_EXIT_CODE=$?
set -e

if [ $GEN_EXIT_CODE -eq 0 ] && [ -f "${JSON_OUTPUT}" ]; then
    echo "[baseline_rag worker ${WORKER_ID}] Generation completed successfully."
    touch "${SIGNAL_FILE}"
else
    echo "[ERROR] Generation failed (exit code: ${GEN_EXIT_CODE})"
    exit 1
fi
