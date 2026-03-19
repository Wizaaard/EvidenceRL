#!/usr/bin/env bash
#SBATCH -J erl_grpo_train
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@180
#SBATCH -o /home/hice1/jtamo3/bmed-sp-wang/BenoitData/Restricted/evidenceRL/logs_grpo/train_grpo_%j.out
#SBATCH -e /home/hice1/jtamo3/bmed-sp-wang/BenoitData/Restricted/evidenceRL/logs_grpo/train_grpo_%j.err

# ============================================================================
# EvidenceRL GRPO Training (with checkpoint chaining)
# ============================================================================
# Trains a base model with GRPO using LoRA + TRL GRPOTrainer.
# Uses three reward signals: format, accuracy (BioLORD-2023), grounding (NLI).
# Automatically resumes from latest checkpoint and re-submits on wall-time.
#
# vLLM runs in COLOCATE mode: shares GPUs with training via time-sharing.
# Training uses `accelerate launch` for multi-GPU distributed training.
#
# Required environment variables:
#   MODEL_ID        - Short identifier (e.g., gemma-3-12b)
#   BASE_MODEL_PATH - Full path to the base model
#   DATASET_PATH    - Path to GRPO dataset JSON
#   MODE            - "no-rag" or "rag"
#
# Optional environment variables:
#   CONDA_ENV       - Conda environment name (default: ragcon)
#   NUM_GPUS        - Number of GPUs (default: 2)
#   EPOCHS          - Number of training epochs (default: 2)
#   LR              - Learning rate (default: 5e-6)
#   BATCH_SIZE      - Per-device batch size (default: 1)
#   GRAD_ACCUM      - Gradient accumulation steps (default: 8)
#   NUM_GENERATIONS - Completions per prompt (default: 8)
#   TEMPERATURE     - Generation temperature (default: 0.7)
#   BETA            - KL penalty coefficient (default: 0.04)
#   VLLM_GPU_UTIL   - vLLM GPU memory utilization (default: 0.4)
#   MAX_MODEL_LEN   - vLLM max model context length (optional)
#   NO_VLLM         - Set to 1 to disable vLLM (default: 0)
#   MAX_CHAIN       - Max chained resubmissions (default: 5)
#   CHAIN_COUNT     - Current chain iteration (auto-incremented)
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
OUTPUT_BASE="${PROJECT_ROOT}/grpo_trained_models"

OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"
OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_ID}_grpo${OUTPUT_SUFFIX:+_${OUTPUT_SUFFIX}}_${MODE}"
LOG_DIR="${PROJECT_ROOT}/logs_grpo"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# ── Chain tracking ───────────────────────────────────────────────────────────
CHAIN_COUNT="${CHAIN_COUNT:-0}"
MAX_CHAIN="${MAX_CHAIN:-5}"

# ── Signal handler for wall-time ─────────────────────────────────────────────
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

# ── Environment setup ────────────────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-ragcon}"
NUM_GPUS="${NUM_GPUS:-2}"

echo "============================================================"
echo "GRPO Training Job: ${SLURM_JOB_ID} (chain ${CHAIN_COUNT}/${MAX_CHAIN})"
echo "============================================================"
echo "  Model:        ${MODEL_ID}"
echo "  Base model:   ${BASE_MODEL_PATH}"
echo "  Dataset:      ${DATASET_PATH}"
echo "  Mode:         ${MODE}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Conda env:    ${CONDA_ENV}"
echo "  GPUs:         ${NUM_GPUS} (TP=${VLLM_TP:-1}, DP=$((NUM_GPUS / ${VLLM_TP:-1})))"
echo "  Node:         $(hostname)"
echo "  Date:         $(date)"
echo ""
echo "  GRPO Hyperparams:"
echo "    Epochs:         ${EPOCHS:-2}"
echo "    LR:             ${LR:-5e-6}"
echo "    Batch size:     ${BATCH_SIZE:-1}"
echo "    Grad accum:     ${GRAD_ACCUM:-8}"
echo "    Num generations:${NUM_GENERATIONS:-8}"
echo "    Temperature:    ${TEMPERATURE:-0.7}"
echo "    Beta (KL):      ${BETA:-0.04}"
echo "    vLLM mode:      $( [ \"${NO_VLLM:-0}\" = \"1\" ] && echo 'disabled' || echo 'colocate' )"
echo "    vLLM TP:        ${VLLM_TP:-1}"
echo "    vLLM GPU util:  ${VLLM_GPU_UTIL:-0.4}"
echo "    Max model len:  ${MAX_MODEL_LEN:-auto}"
echo "    Num iterations: 2 (policy updates per generation)"
echo "    4-bit quant:    $( [ \"${LOAD_IN_4BIT:-0}\" = \"1\" ] && echo 'enabled (QLoRA)' || echo 'disabled' )"
echo "    Liger kernel:   enabled"
echo "    Sleep mode:     enabled"
echo "============================================================"

# Activate conda environment
source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}"

# Print library versions for reproducibility
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

# ── Build training command ───────────────────────────────────────────────────
VLLM_TP="${VLLM_TP:-1}"

VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

TRAIN_ARGS=(
    "${SCRIPT_DIR}/train_grpo.py"
    --base-model "${BASE_MODEL_PATH}"
    --dataset "${DATASET_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --run-name "${MODEL_ID}_grpo_${MODE}"
    --epochs "${EPOCHS:-2}"
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

if [ "${NO_SENTENCE_LEVEL:-0}" = "1" ]; then
    TRAIN_ARGS+=(--no-sentence-level)
fi

# ── Launch training via accelerate (colocate mode needs multi-GPU launch) ────
# In colocate mode, vLLM runs in-process sharing GPUs with training.
# accelerate launch handles distributed setup for GRPOTrainer.
echo ""
LAUNCH_PROCS="${NUM_GPUS}"
# For no-vllm + 4bit (QLoRA): use all GPUs with data parallelism
# (device_map={"": LOCAL_RANK} in train_grpo.py ensures each process loads on its own GPU)
if [ "${NO_VLLM:-0}" = "1" ] && [ "${LOAD_IN_4BIT:-0}" = "1" ]; then
    echo "  (QLoRA + no-vllm: ${LAUNCH_PROCS}-GPU data parallel, 4-bit model on each GPU)"
fi
echo "Launching training with accelerate (${LAUNCH_PROCS} processes, colocate mode)..."

accelerate launch \
    --num_processes "${LAUNCH_PROCS}" \
    "${TRAIN_ARGS[@]}" &

TRAIN_PID=$!
wait "${TRAIN_PID}" 2>/dev/null
TRAIN_EXIT=$?

# ── Check if training completed ──────────────────────────────────────────────
if [ -d "${OUTPUT_DIR}/final_model" ]; then
    echo ""
    echo "============================================================"
    echo "GRPO TRAINING COMPLETE: ${OUTPUT_DIR}"
    echo "============================================================"
    exit 0
fi

# Training did not finish
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

EXPORT_VARS="MODEL_ID=${MODEL_ID}"
EXPORT_VARS="${EXPORT_VARS},BASE_MODEL_PATH=${BASE_MODEL_PATH}"
EXPORT_VARS="${EXPORT_VARS},DATASET_PATH=${DATASET_PATH}"
EXPORT_VARS="${EXPORT_VARS},MODE=${MODE}"
EXPORT_VARS="${EXPORT_VARS},CONDA_ENV=${CONDA_ENV}"
EXPORT_VARS="${EXPORT_VARS},NUM_GPUS=${NUM_GPUS}"
EXPORT_VARS="${EXPORT_VARS},VLLM_TP=${VLLM_TP:-1}"
EXPORT_VARS="${EXPORT_VARS},EPOCHS=${EPOCHS:-2}"
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
EXPORT_VARS="${EXPORT_VARS},NO_SENTENCE_LEVEL=${NO_SENTENCE_LEVEL:-0}"
EXPORT_VARS="${EXPORT_VARS},OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-}"

# Compute wall time to stay within QOS 960 GPU-minute limit
CHAIN_WALL_TIME="08:00:00"
if [ "${NUM_GPUS}" -ge 4 ]; then
    CHAIN_WALL_TIME="04:00:00"
fi

NEXT_JOB_ID=$(sbatch \
    --export="${EXPORT_VARS}" \
    --job-name="grpo_${OUTPUT_SUFFIX:+${OUTPUT_SUFFIX}_}${MODEL_ID}_${MODE}" \
    --gres="gpu:h200:${NUM_GPUS}" \
    --mem=256G \
    --cpus-per-task=32 \
    --time="${CHAIN_WALL_TIME}" \
    "${SCRIPT_DIR}/train_grpo.sbatch" | awk '{print $4}')

echo "Continuation job submitted: ${NEXT_JOB_ID}"
echo "============================================================"
