#!/usr/bin/env bash
#SBATCH -J barexam_grpo_train
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@180
#SBATCH -o /home/hice1/jtamo3/bmed-sp-wang/BenoitData/Restricted/evidenceRL/logs_barexam_grpo/train_%j.out
#SBATCH -e /home/hice1/jtamo3/bmed-sp-wang/BenoitData/Restricted/evidenceRL/logs_barexam_grpo/train_%j.err

# ============================================================================
# BarExam GRPO Training (with checkpoint chaining)
# ============================================================================
# Trains a base model with GRPO for BarExam MCQ:
#   R_format:    Valid JSON with answer + reasoning
#   R_accuracy:  Exact match on answer letter
#   R_grounding: NLI (DeBERTa-v3-large) sentence-level grounding
#
# Required env vars: MODEL_ID, BASE_MODEL_PATH, DATASET_PATH, MODE
# ============================================================================

set -uo pipefail

# ── Validate required env vars ─────────────────────────────────────────────
for var in MODEL_ID BASE_MODEL_PATH DATASET_PATH MODE; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: Required environment variable $var is not set"
        exit 1
    fi
done

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/hice1/jtamo3/bmed-sp-wang/BenoitData/Restricted/evidenceRL"
SCRIPT_DIR="${PROJECT_ROOT}/code/scripts/rl_training"
OUTPUT_BASE="${PROJECT_ROOT}/barexam_grpo_trained_models"

OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"
OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_ID}_barexam_grpo${OUTPUT_SUFFIX:+_${OUTPUT_SUFFIX}}_${MODE}"
LOG_DIR="${PROJECT_ROOT}/logs_barexam_grpo"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# ── Chain tracking ─────────────────────────────────────────────────────────
CHAIN_COUNT="${CHAIN_COUNT:-0}"
MAX_CHAIN="${MAX_CHAIN:-5}"

# ── Signal handler for wall-time ───────────────────────────────────────────
TRAIN_PID=""
handle_timeout() {
    echo ""
    echo ">>> Wall-time approaching (USR1). Sending SIGTERM to training (PID=${TRAIN_PID})..."
    if [ -n "${TRAIN_PID}" ]; then
        kill -TERM "${TRAIN_PID}" 2>/dev/null || true
        wait "${TRAIN_PID}" 2>/dev/null || true
    fi
}
trap 'handle_timeout' USR1

# ── Environment setup ──────────────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-ragcon}"
NUM_GPUS="${NUM_GPUS:-2}"

echo "============================================================"
echo "BarExam GRPO Training Job: ${SLURM_JOB_ID} (chain ${CHAIN_COUNT}/${MAX_CHAIN})"
echo "============================================================"
echo "  Model:        ${MODEL_ID}"
echo "  Base model:   ${BASE_MODEL_PATH}"
echo "  Dataset:      ${DATASET_PATH}"
echo "  Mode:         ${MODE}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Conda env:    ${CONDA_ENV}"
echo "  GPUs:         ${NUM_GPUS} (TP=${VLLM_TP:-1})"
echo "  Node:         $(hostname)"
echo "  Date:         $(date)"
echo ""
echo "  GRPO Hyperparams:"
echo "    Epochs:         ${EPOCHS:-3}"
echo "    LR:             ${LR:-5e-6}"
echo "    Batch size:     ${BATCH_SIZE:-1}"
echo "    Grad accum:     ${GRAD_ACCUM:-8}"
echo "    Num generations:${NUM_GENERATIONS:-8}"
echo "    Temperature:    ${TEMPERATURE:-0.7}"
echo "    Beta (KL):      ${BETA:-0.04}"
echo "    vLLM mode:      $( [ "${NO_VLLM:-0}" = "1" ] && echo 'disabled' || echo 'colocate' )"
echo "    vLLM TP:        ${VLLM_TP:-1}"
echo "    vLLM GPU util:  ${VLLM_GPU_UTIL:-0.4}"
echo "    Max model len:  ${MAX_MODEL_LEN:-auto}"
echo "    4-bit quant:    $( [ "${LOAD_IN_4BIT:-0}" = "1" ] && echo 'enabled (QLoRA)' || echo 'disabled' )"
echo "============================================================"

# Activate conda environment
source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}"

# Print library versions
python3 -c "
import trl, peft, transformers, torch
print(f'  TRL: {trl.__version__}')
print(f'  PEFT: {peft.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')
try:
    import vllm
    print(f'  vLLM: {vllm.__version__}')
except ImportError:
    print('  vLLM: not installed')
"

# ── Build training command ─────────────────────────────────────────────────
VLLM_TP="${VLLM_TP:-1}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

TRAIN_ARGS=(
    "${SCRIPT_DIR}/train_barexam_grpo.py"
    --base-model "${BASE_MODEL_PATH}"
    --dataset "${DATASET_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --run-name "${MODEL_ID}_barexam_grpo_${MODE}"
    --epochs "${EPOCHS:-3}"
    --lr "${LR:-5e-6}"
    --batch-size "${BATCH_SIZE:-1}"
    --grad-accum "${GRAD_ACCUM:-8}"
    --num-generations "${NUM_GENERATIONS:-8}"
    --temperature "${TEMPERATURE:-0.7}"
    --beta "${BETA:-0.04}"
    --vllm-tp "${VLLM_TP}"
    --vllm-gpu-util "${VLLM_GPU_UTIL}"
    --resume-from-checkpoint auto
)

if [ -n "${MAX_MODEL_LEN}" ]; then
    TRAIN_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi

if [ "${NO_VLLM:-0}" = "1" ]; then
    TRAIN_ARGS+=(--no-vllm)
fi

if [ "${LOAD_IN_4BIT:-0}" = "1" ]; then
    TRAIN_ARGS+=(--load-in-4bit)
fi

if [ -n "${REWARD_WEIGHT_GROUNDING:-}" ]; then
    TRAIN_ARGS+=(--reward-weight-grounding "${REWARD_WEIGHT_GROUNDING}")
fi

# ── Launch training ────────────────────────────────────────────────────────
LAUNCH_PROCS="${NUM_GPUS}"
echo ""
echo "Launching BarExam GRPO training with accelerate (${LAUNCH_PROCS} processes)..."

accelerate launch \
    --num_processes "${LAUNCH_PROCS}" \
    "${TRAIN_ARGS[@]}" &

TRAIN_PID=$!
wait "${TRAIN_PID}" 2>/dev/null
TRAIN_EXIT=$?

# ── Check if training completed ────────────────────────────────────────────
if [ -d "${OUTPUT_DIR}/final_model" ]; then
    echo ""
    echo "============================================================"
    echo "BAREXAM GRPO TRAINING COMPLETE: ${OUTPUT_DIR}"
    echo "============================================================"
    exit 0
fi

# Training did not finish
if [ "${TRAIN_EXIT}" -ne 0 ]; then
    echo ""
    echo "WARNING: Training exited with code ${TRAIN_EXIT}"
fi

# ── Auto-resubmit if under chain limit ─────────────────────────────────────
NEXT_CHAIN=$((CHAIN_COUNT + 1))

if [ "${NEXT_CHAIN}" -gt "${MAX_CHAIN}" ]; then
    echo ""
    echo "============================================================"
    echo "MAX CHAIN LIMIT REACHED (${MAX_CHAIN}). NOT resubmitting."
    echo "============================================================"
    exit 1
fi

echo ""
echo "============================================================"
echo "Training incomplete. Resubmitting as chain ${NEXT_CHAIN}/${MAX_CHAIN}..."
echo "============================================================"

EXPORT_VARS="MODEL_ID=${MODEL_ID}"
EXPORT_VARS="${EXPORT_VARS},BASE_MODEL_PATH=${BASE_MODEL_PATH}"
EXPORT_VARS="${EXPORT_VARS},DATASET_PATH=${DATASET_PATH}"
EXPORT_VARS="${EXPORT_VARS},MODE=${MODE}"
EXPORT_VARS="${EXPORT_VARS},CONDA_ENV=${CONDA_ENV}"
EXPORT_VARS="${EXPORT_VARS},NUM_GPUS=${NUM_GPUS}"
EXPORT_VARS="${EXPORT_VARS},VLLM_TP=${VLLM_TP:-1}"
EXPORT_VARS="${EXPORT_VARS},EPOCHS=${EPOCHS:-3}"
EXPORT_VARS="${EXPORT_VARS},LR=${LR:-5e-6}"
EXPORT_VARS="${EXPORT_VARS},BATCH_SIZE=${BATCH_SIZE:-1}"
EXPORT_VARS="${EXPORT_VARS},GRAD_ACCUM=${GRAD_ACCUM:-8}"
EXPORT_VARS="${EXPORT_VARS},NUM_GENERATIONS=${NUM_GENERATIONS:-8}"
EXPORT_VARS="${EXPORT_VARS},TEMPERATURE=${TEMPERATURE:-0.7}"
EXPORT_VARS="${EXPORT_VARS},BETA=${BETA:-0.04}"
EXPORT_VARS="${EXPORT_VARS},VLLM_GPU_UTIL=${VLLM_GPU_UTIL}"
EXPORT_VARS="${EXPORT_VARS},MAX_MODEL_LEN=${MAX_MODEL_LEN}"
EXPORT_VARS="${EXPORT_VARS},NO_VLLM=${NO_VLLM:-0}"
EXPORT_VARS="${EXPORT_VARS},LOAD_IN_4BIT=${LOAD_IN_4BIT:-0}"
EXPORT_VARS="${EXPORT_VARS},CHAIN_COUNT=${NEXT_CHAIN}"
EXPORT_VARS="${EXPORT_VARS},MAX_CHAIN=${MAX_CHAIN}"
EXPORT_VARS="${EXPORT_VARS},REWARD_WEIGHT_GROUNDING=${REWARD_WEIGHT_GROUNDING:-}"
EXPORT_VARS="${EXPORT_VARS},OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-}"

# Wall time for chain: 4 GPUs × 4h = 960 GPU-minutes (QOS limit)
CHAIN_WALL_TIME="08:00:00"
if [ "${NUM_GPUS}" -ge 4 ]; then
    CHAIN_WALL_TIME="04:00:00"
fi

NEXT_JOB_ID=$(sbatch \
    --export="${EXPORT_VARS}" \
    --job-name="barexam_grpo_${OUTPUT_SUFFIX:+${OUTPUT_SUFFIX}_}${MODEL_ID}_${MODE}" \
    --gres="gpu:h200:${NUM_GPUS}" \
    --mem=256G \
    --cpus-per-task=32 \
    --time="${CHAIN_WALL_TIME}" \
    "${SCRIPT_DIR}/train_barexam_grpo.sbatch" | awk '{print $4}')

echo "Continuation job submitted: ${NEXT_JOB_ID}"
echo "============================================================"
