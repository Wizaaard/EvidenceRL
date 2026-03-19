#!/usr/bin/env bash
#SBATCH -J erl_dpo_train
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@180
#SBATCH -o /home/hice1/jtamo3/bmed-sp-wang/BenoitData/Restricted/evidenceRL/logs_dpo/train_dpo_%j.out
#SBATCH -e /home/hice1/jtamo3/bmed-sp-wang/BenoitData/Restricted/evidenceRL/logs_dpo/train_dpo_%j.err

# ============================================================================
# Faithfulness-DPO Training (EvidenceRL baseline)
# ============================================================================
# Trains a model on grounding-based preference pairs using DPO.
# No vLLM needed (DPO is offline — dataset is pre-built).
# Uses accelerate launch for multi-GPU data parallelism.
#
# Required environment variables:
#   MODEL_ID        - Short identifier (e.g., gemma-3-12b)
#   BASE_MODEL_PATH - Full path to the base model
#   DATASET_PATH    - Path to faithfulness_dpo.json
#
# Optional environment variables:
#   CONDA_ENV       - (default: ragcon)
#   NUM_GPUS        - (default: 2)
#   EPOCHS          - (default: 2)
#   LR              - (default: 5e-6)
#   BATCH_SIZE      - per-device (default: 1)
#   GRAD_ACCUM      - (default: 8)
#   BETA            - DPO beta (default: 0.1)
#   LOAD_IN_4BIT    - 0 or 1 (default: 0)
#   MAX_CHAIN       - max job resubmissions (default: 5)
#   CHAIN_COUNT     - current chain count (auto-incremented)
# ============================================================================

set -uo pipefail

for var in MODEL_ID BASE_MODEL_PATH DATASET_PATH; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: Required env var $var is not set"; exit 1
    fi
done

PROJECT_ROOT="/home/hice1/jtamo3/bmed-sp-wang/BenoitData/Restricted/evidenceRL"
SCRIPT_DIR="${PROJECT_ROOT}/code/scripts/rl_training"
OUTPUT_BASE="${PROJECT_ROOT}/dpo_trained_models"
OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_ID}_faithfulness_dpo"
LOG_DIR="${PROJECT_ROOT}/logs_dpo"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

CHAIN_COUNT="${CHAIN_COUNT:-0}"
MAX_CHAIN="${MAX_CHAIN:-5}"

TRAIN_PID=""
handle_timeout() {
    echo ">>> Wall-time approaching. Sending SIGTERM to training (PID=${TRAIN_PID})..."
    [ -n "${TRAIN_PID}" ] && kill -TERM "${TRAIN_PID}" 2>/dev/null; wait "${TRAIN_PID}" 2>/dev/null || true
}
trap 'handle_timeout' USR1

CONDA_ENV="${CONDA_ENV:-ragcon}"
NUM_GPUS="${NUM_GPUS:-2}"

echo "============================================================"
echo "Faithfulness-DPO Training: ${SLURM_JOB_ID} (chain ${CHAIN_COUNT}/${MAX_CHAIN})"
echo "============================================================"
echo "  Model:      ${MODEL_ID}"
echo "  Base model: ${BASE_MODEL_PATH}"
echo "  Dataset:    ${DATASET_PATH}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Epochs:     ${EPOCHS:-2}"
echo "  LR:         ${LR:-5e-6}"
echo "  Batch:      ${BATCH_SIZE:-1} x ${GRAD_ACCUM:-8} grad_accum"
echo "  DPO beta:   ${BETA:-0.1}"
echo "  4-bit:      ${LOAD_IN_4BIT:-0}"
echo "============================================================"

source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}"

python3 -c "
import trl, peft, transformers, torch
print(f'  TRL: {trl.__version__}, PEFT: {peft.__version__}')
print(f'  Transformers: {transformers.__version__}, PyTorch: {torch.__version__}')
print(f'  CUDA devices: {torch.cuda.device_count()}')
"

TRAIN_ARGS=(
    "${SCRIPT_DIR}/train_dpo.py"
    --base-model "${BASE_MODEL_PATH}"
    --dataset    "${DATASET_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --run-name   "${MODEL_ID}_faithfulness_dpo"
    --epochs     "${EPOCHS:-2}"
    --lr         "${LR:-5e-6}"
    --batch-size "${BATCH_SIZE:-1}"
    --grad-accum "${GRAD_ACCUM:-8}"
    --beta       "${BETA:-0.1}"
    --resume-from-checkpoint auto
)

if [ "${LOAD_IN_4BIT:-0}" = "1" ]; then
    TRAIN_ARGS+=(--load-in-4bit)
fi

echo ""

# Randomize MASTER_PORT to avoid conflicts when multiple jobs land on the same node
export MASTER_PORT=$((29500 + RANDOM % 1000))
echo "  MASTER_PORT=${MASTER_PORT}"

ACCEL_ARGS=(--main_process_port "${MASTER_PORT}" --num_processes "${NUM_GPUS}")

# Use FSDP config if provided via ACCELERATE_CONFIG env var
if [ -n "${ACCELERATE_CONFIG:-}" ] && [ -f "${ACCELERATE_CONFIG}" ]; then
    ACCEL_ARGS=(--config_file "${ACCELERATE_CONFIG}" --main_process_port "${MASTER_PORT}" --num_processes "${NUM_GPUS}")
    echo "Launching DPO training with accelerate FSDP (${NUM_GPUS} GPUs, sharded)..."
    echo "  Config: ${ACCELERATE_CONFIG}"
else
    echo "Launching DPO training with accelerate (${NUM_GPUS} GPUs, data parallel)..."
fi

accelerate launch "${ACCEL_ARGS[@]}" "${TRAIN_ARGS[@]}" &

TRAIN_PID=$!
wait "${TRAIN_PID}" 2>/dev/null
TRAIN_EXIT=$?

if [ -d "${OUTPUT_DIR}/final_model" ]; then
    echo "============================================================"
    echo "DPO TRAINING COMPLETE: ${OUTPUT_DIR}"
    echo "============================================================"
    exit 0
fi

[ "${TRAIN_EXIT}" -ne 0 ] && echo "WARNING: Training exited with code ${TRAIN_EXIT}"

NEXT_CHAIN=$((CHAIN_COUNT + 1))
if [ "${NEXT_CHAIN}" -gt "${MAX_CHAIN}" ]; then
    echo "MAX CHAIN LIMIT REACHED. NOT resubmitting."; exit 1
fi

echo "Training incomplete. Resubmitting as chain ${NEXT_CHAIN}/${MAX_CHAIN}..."

CHAIN_WALL="08:00:00"
[ "${NUM_GPUS}" -ge 4 ] && CHAIN_WALL="04:00:00"

EXPORT_VARS="MODEL_ID=${MODEL_ID},BASE_MODEL_PATH=${BASE_MODEL_PATH}"
EXPORT_VARS="${EXPORT_VARS},DATASET_PATH=${DATASET_PATH}"
EXPORT_VARS="${EXPORT_VARS},CONDA_ENV=${CONDA_ENV},NUM_GPUS=${NUM_GPUS}"
EXPORT_VARS="${EXPORT_VARS},EPOCHS=${EPOCHS:-2},LR=${LR:-5e-6}"
EXPORT_VARS="${EXPORT_VARS},BATCH_SIZE=${BATCH_SIZE:-1},GRAD_ACCUM=${GRAD_ACCUM:-8}"
EXPORT_VARS="${EXPORT_VARS},BETA=${BETA:-0.1},LOAD_IN_4BIT=${LOAD_IN_4BIT:-0}"
EXPORT_VARS="${EXPORT_VARS},CHAIN_COUNT=${NEXT_CHAIN},MAX_CHAIN=${MAX_CHAIN}"
EXPORT_VARS="${EXPORT_VARS},GPU_TYPE=${GPU_TYPE:-H100+H200}"
[ -n "${ACCELERATE_CONFIG:-}" ] && EXPORT_VARS="${EXPORT_VARS},ACCELERATE_CONFIG=${ACCELERATE_CONFIG}"

# Translate "+" to "|" for SLURM --constraint
CHAIN_CONSTRAINT="${GPU_TYPE:-H100+H200}"
CHAIN_CONSTRAINT="${CHAIN_CONSTRAINT//+/|}"

NEXT_JOB_ID=$(sbatch \
    --export="${EXPORT_VARS}" \
    --job-name="fdpo_${MODEL_ID}" \
    --gres="gpu:${NUM_GPUS}" \
    --constraint="${CHAIN_CONSTRAINT}" \
    --mem=256G --cpus-per-task=32 \
    --time="${CHAIN_WALL}" \
    "${SCRIPT_DIR}/train_dpo.sbatch" | awk '{print $4}')

echo "Continuation job submitted: ${NEXT_JOB_ID}"
