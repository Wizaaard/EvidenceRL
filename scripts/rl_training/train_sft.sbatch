#!/usr/bin/env bash
#SBATCH -J erl_sft_train
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@180
#SBATCH -o logs_sft/train_sft_%j.out
#SBATCH -e logs_sft/train_sft_%j.err

# ============================================================================
# EvidenceRL SFT Training (with checkpoint chaining)
# ============================================================================
# Fine-tunes a base model on Evidence-Based outputs using LoRA + TRL SFTTrainer.
# Automatically resumes from the latest checkpoint and re-submits itself if
# training is interrupted by the wall-time limit.
#
# Required environment variables:
#   MODEL_ID        - Short identifier (e.g., gemma-3-4b)
#   BASE_MODEL_PATH - Full path to the base model
#   DATASET_PATH    - Path to SFT dataset JSON
#   MODE            - "rag" or "no-rag"
#
# Optional environment variables:
#   CONDA_ENV       - Conda environment name (default: ragcon)
#   EPOCHS          - Number of training epochs (default: 3)
#   LR              - Learning rate (default: 2e-5)
#   BATCH_SIZE      - Per-device batch size (default: 2)
#   GRAD_ACCUM      - Gradient accumulation steps (default: 8)
#   MAX_SEQ_LEN     - Max sequence length (default: 2048)
#   MAX_CHAIN       - Max number of chained resubmissions (default: 10)
#   CHAIN_COUNT     - Current chain iteration (auto-incremented, default: 0)
# ============================================================================

set -uo pipefail

# ── Validate required env vars ───────────────────────────────────────────────
for var in MODEL_ID BASE_MODEL_PATH DATASET_PATH MODE; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: Required environment variable $var is not set"
        exit 1
    fi
done

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/hice1/jtamo3/bmed-sp-wang/BenoitData/Restricted/evidenceRL"
SCRIPT_DIR="${PROJECT_ROOT}/code/scripts/rl_training"
OUTPUT_BASE="${PROJECT_ROOT}/sft_trained_models/completion"

OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_ID}_sft_${MODE}"
LOG_DIR="${PROJECT_ROOT}/logs_sft"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# ── Chain tracking ───────────────────────────────────────────────────────────
CHAIN_COUNT="${CHAIN_COUNT:-0}"
MAX_CHAIN="${MAX_CHAIN:-10}"

# ── Signal handler for wall-time approaching ─────────────────────────────────
# SLURM sends USR1 180s before wall-time (see --signal=B:USR1@180).
# We forward SIGTERM to the python training process so HuggingFace Trainer
# can save a checkpoint before exiting.
TRAIN_PID=""
handle_timeout() {
    echo ""
    echo ">>> Wall-time approaching (USR1 received). Sending SIGTERM to training (PID=${TRAIN_PID})..."
    if [ -n "${TRAIN_PID}" ]; then
        kill -TERM "${TRAIN_PID}" 2>/dev/null || true
        wait "${TRAIN_PID}" 2>/dev/null || true
    fi
}
trap 'handle_timeout' USR1

# ── Environment setup ────────────────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-ragcon}"

echo "============================================================"
echo "SFT Training Job: ${SLURM_JOB_ID} (chain ${CHAIN_COUNT}/${MAX_CHAIN})"
echo "============================================================"
echo "  Model:        ${MODEL_ID}"
echo "  Base model:   ${BASE_MODEL_PATH}"
echo "  Dataset:      ${DATASET_PATH}"
echo "  Mode:         ${MODE}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Conda env:    ${CONDA_ENV}"
echo "  GPUs:         ${SLURM_GPUS_ON_NODE:-unknown}"
echo "  Node:         $(hostname)"
echo "  Date:         $(date)"
echo ""
echo "  Hyperparams:"
echo "    Epochs:     ${EPOCHS:-3}"
echo "    LR:         ${LR:-2e-5}"
echo "    Batch size: ${BATCH_SIZE:-2}"
echo "    Grad accum: ${GRAD_ACCUM:-8}"
echo "    Max seq:    ${MAX_SEQ_LEN:-2048}"
echo "============================================================"

# Activate conda environment
source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}"

# Ensure training dependencies are installed
pip install --quiet trl peft accelerate datasets 2>/dev/null

# Print library versions for reproducibility
python3 -c "
import trl, peft, transformers, torch, datasets
print(f'  TRL: {trl.__version__}')
print(f'  PEFT: {peft.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  PyTorch: {torch.__version__}')
print(f'  Datasets: {datasets.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')
"

# ── Launch training (auto-resume from latest checkpoint) ─────────────────────
# Use "auto" to let train_sft.py find the latest checkpoint-* dir.
# On the first run there are no checkpoints, so it starts fresh.
# Run in background so the USR1 trap can fire and forward SIGTERM.
python "${SCRIPT_DIR}/train_sft.py" \
    --base-model "${BASE_MODEL_PATH}" \
    --dataset "${DATASET_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name "${MODEL_ID}_sft_${MODE}" \
    --epochs "${EPOCHS:-3}" \
    --lr "${LR:-2e-5}" \
    --batch-size "${BATCH_SIZE:-2}" \
    --grad-accum "${GRAD_ACCUM:-8}" \
    --max-seq-length "${MAX_SEQ_LEN:-2048}" \
    --val-split 0.1 \
    --resume-from-checkpoint auto &

TRAIN_PID=$!
wait "${TRAIN_PID}" 2>/dev/null
TRAIN_EXIT=$?

# ── Check if training completed ──────────────────────────────────────────────
if [ -d "${OUTPUT_DIR}/final_model" ]; then
    echo ""
    echo "============================================================"
    echo "TRAINING COMPLETE: ${OUTPUT_DIR}"
    echo "============================================================"
    exit 0
fi

# Training did not finish (wall-time limit or error)
if [ "${TRAIN_EXIT}" -ne 0 ]; then
    echo ""
    echo "WARNING: Training exited with code ${TRAIN_EXIT}"
fi

# ── Auto-resubmit if under chain limit ───────────────────────────────────────
NEXT_CHAIN=$((CHAIN_COUNT + 1))

if [ "${NEXT_CHAIN}" -gt "${MAX_CHAIN}" ]; then
    echo ""
    echo "============================================================"
    echo "MAX CHAIN LIMIT REACHED (${MAX_CHAIN}). NOT resubmitting."
    echo "Output so far: ${OUTPUT_DIR}"
    echo "============================================================"
    exit 1
fi

echo ""
echo "============================================================"
echo "Training incomplete. Resubmitting as chain ${NEXT_CHAIN}/${MAX_CHAIN}..."
echo "============================================================"

# Build the same export vars, incrementing CHAIN_COUNT
EXPORT_VARS="MODEL_ID=${MODEL_ID}"
EXPORT_VARS="${EXPORT_VARS},BASE_MODEL_PATH=${BASE_MODEL_PATH}"
EXPORT_VARS="${EXPORT_VARS},DATASET_PATH=${DATASET_PATH}"
EXPORT_VARS="${EXPORT_VARS},MODE=${MODE}"
EXPORT_VARS="${EXPORT_VARS},CONDA_ENV=${CONDA_ENV}"
EXPORT_VARS="${EXPORT_VARS},EPOCHS=${EPOCHS:-3}"
EXPORT_VARS="${EXPORT_VARS},LR=${LR:-2e-5}"
EXPORT_VARS="${EXPORT_VARS},BATCH_SIZE=${BATCH_SIZE:-2}"
EXPORT_VARS="${EXPORT_VARS},GRAD_ACCUM=${GRAD_ACCUM:-8}"
EXPORT_VARS="${EXPORT_VARS},MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}"
EXPORT_VARS="${EXPORT_VARS},CHAIN_COUNT=${NEXT_CHAIN}"
EXPORT_VARS="${EXPORT_VARS},MAX_CHAIN=${MAX_CHAIN}"

NEXT_JOB_ID=$(sbatch \
    --export="${EXPORT_VARS}" \
    --job-name="sft_${MODEL_ID}_${MODE}" \
    --gres="gpu:h200:${SLURM_GPUS_ON_NODE:-2}" \
    --time="08:00:00" \
    "${SCRIPT_DIR}/train_sft.sbatch" | awk '{print $4}')

echo "Continuation job submitted: ${NEXT_JOB_ID}"
echo "============================================================"
