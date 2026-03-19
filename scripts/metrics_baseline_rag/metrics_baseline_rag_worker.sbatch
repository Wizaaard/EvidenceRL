#!/usr/bin/env bash
#SBATCH -J erl_baseline_rag_metrics_worker
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH -o logs_evidencerl_baseline_rag/metrics_baseline_rag_worker_%A_%a.out
#SBATCH -e logs_evidencerl_baseline_rag/metrics_baseline_rag_worker_%A_%a.err

# ============================================================================
# EvidenceRL Baseline RAG Metrics Computation Worker
# ============================================================================

set -euo pipefail

: "${MODEL_ID:?Need MODEL_ID}"
: "${GENERATION_FILE:?Need GENERATION_FILE}"
: "${JUDGE_MODEL:?Need JUDGE_MODEL}"
: "${PATIENT_START:?Need PATIENT_START}"
: "${PATIENT_END:?Need PATIENT_END}"
: "${WORKER_ID:?Need WORKER_ID}"
: "${VERSION:=1.0}"
: "${OUTPUT_BASE_DIR:?Need OUTPUT_BASE_DIR}"

CONDA_ENV="${CONDA_ENV:-evidencerl}"
PROJECT_ROOT="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/code"

NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-base}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
REWARD_WEIGHT="${REWARD_WEIGHT:-0.5}"
BATCH_SIZE="${BATCH_SIZE:-48}"
MAX_K="${MAX_K:-5}"

SIGNAL_DIR="${SIGNAL_DIR:-${OUTPUT_BASE_DIR}/signals}"
mkdir -p logs_evidencerl_baseline_rag "${OUTPUT_BASE_DIR}" "${SIGNAL_DIR}"

echo "[baseline_rag metrics worker ${WORKER_ID}] Checking GPU..."
if ! nvidia-smi &>/dev/null; then
    echo "[ERROR] GPU not available"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    [ -n "${SLURM_JOB_GPUS:-}" ] && export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
    [ -n "${SLURM_STEP_GPUS:-}" ] && export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS}"
fi

# Initialize conda
for conda_path in "$HOME/.conda/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/mambaforge/etc/profile.d/conda.sh" "/usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh"; do
    [ -f "$conda_path" ] && { source "$conda_path"; break; }
done

set +u; conda activate "${CONDA_ENV}"; set -u
echo "[baseline_rag metrics worker ${WORKER_ID}] Activated: ${CONDA_ENV}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

JSON_OUTPUT="${OUTPUT_BASE_DIR}/${MODEL_ID}_baseline_rag_metrics_worker${WORKER_ID}_v${VERSION}.json"
SIGNAL_FILE="${SIGNAL_DIR}/${MODEL_ID}_baseline_rag_metrics_worker${WORKER_ID}.done"

[ -f "${SIGNAL_FILE}" ] && { echo "[worker] Already complete"; exit 0; }

if [ -f "${JSON_OUTPUT}" ]; then
    python3 -c "import json; d=json.load(open('${JSON_OUTPUT}')); assert len(d.get('patient_metrics',[])) > 0" 2>/dev/null && { touch "${SIGNAL_FILE}"; exit 0; }
fi

echo "[baseline_rag metrics worker ${WORKER_ID}] Starting..."
echo "[baseline_rag metrics worker ${WORKER_ID}] MODE=BASELINE RAG"
echo "[baseline_rag metrics worker ${WORKER_ID}] GENERATION_FILE=${GENERATION_FILE}"
echo "[baseline_rag metrics worker ${WORKER_ID}] PATIENT_RANGE=[${PATIENT_START}:${PATIENT_END}]"

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
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -eq 0 ] && [ -f "${JSON_OUTPUT}" ]; then
    echo "[baseline_rag metrics worker ${WORKER_ID}] Completed successfully"
    touch "${SIGNAL_FILE}"
else
    echo "[ERROR] Failed (exit code: ${EXIT_CODE})"
    exit 1
fi
