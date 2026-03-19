#!/usr/bin/env bash
# ============================================================================
# Launch SFT training for all model backbones (completion-only loss)
# ============================================================================
#
# Trains with completion-only loss and auto-chains 8h jobs for large models.
# Outputs to sft_trained_models/completion/ to separate from old full-sequence runs.
#
# Usage:
#   ./launch_sft_training.sh                    # All models, both modes
#   ./launch_sft_training.sh --mode rag         # All models, RAG only
#   ./launch_sft_training.sh --model gemma-3-4b # Single model, both modes
#   ./launch_sft_training.sh --dry-run          # Print commands without submitting
#   ./launch_sft_training.sh --conda-env myenv  # Use custom conda env
#   ./launch_sft_training.sh --epochs 5 --lr 1e-5  # Override hyperparams
# ============================================================================

set -euo pipefail

# Configure these for your environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_BASE_DIR="${MODEL_BASE_DIR:-//storage/ice-shared/bmed-sp-wang/Models}"
SBATCH_SCRIPT="${SCRIPT_DIR}/train_sft.sbatch"
SFT_DATA_DIR="${PROJECT_ROOT}/training_data/sft"
LOG_DIR="${PROJECT_ROOT}/logs_sft"

mkdir -p "${LOG_DIR}"

# ── Defaults ─────────────────────────────────────────────────────────────────
MODE_FILTER=""
MODEL_FILTER=""
DRY_RUN=false
CONDA_ENV="ragcon"
EPOCHS="3"
LR="2e-5"
BATCH_SIZE="1"
GRAD_ACCUM="16"
MAX_SEQ_LEN="8192"
GPUS="2"
WALL_TIME="08:00:00"
MAX_CHAIN="10"

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)         MODE_FILTER="$2"; shift 2 ;;
        --model)        MODEL_FILTER="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --conda-env)    CONDA_ENV="$2"; shift 2 ;;
        --epochs)       EPOCHS="$2"; shift 2 ;;
        --lr)           LR="$2"; shift 2 ;;
        --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
        --grad-accum)   GRAD_ACCUM="$2"; shift 2 ;;
        --max-seq-len)  MAX_SEQ_LEN="$2"; shift 2 ;;
        --gpus)         GPUS="$2"; shift 2 ;;
        --wall-time)    WALL_TIME="$2"; shift 2 ;;
        --max-chain)    MAX_CHAIN="$2"; shift 2 ;;
        *)              echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Model configurations ─────────────────────────────────────────────────────
# Format: MODEL_ID|BASE_MODEL_PATH
MODELS=(
    "gemma-3-4b|${MODEL_BASE_DIR}/gemma-3-4b-it"
    "gemma-3-12b|${MODEL_BASE_DIR}/gemma-3-12b-it"
    "gemma-3-27b|${MODEL_BASE_DIR}/gemma-3-27b-it"
    "Llama-3.2-3B|${MODEL_BASE_DIR}/Llama-3.2-3B-Instruct"
    "Llama-3.1-8B|${MODEL_BASE_DIR}/Llama-3.1-8B-Instruct"
    "Llama-3.3-70B|${MODEL_BASE_DIR}/Llama-3.3-70B-Instruct"
    "gpt-oss-20b|${MODEL_BASE_DIR}/gpt-oss-20b"
    "gpt-oss-120b|${MODEL_BASE_DIR}/gpt-oss-120b"
)

# Dataset files
declare -A DATASETS
DATASETS["rag"]="${SFT_DATA_DIR}/sft_rag_eb2_cap2.json"
DATASETS["no-rag"]="${SFT_DATA_DIR}/sft_no_rag_eb2_cap2.json"

# Modes to train
if [ -n "${MODE_FILTER}" ]; then
    MODES=("${MODE_FILTER}")
else
    MODES=("rag" "no-rag")
fi

# ── Submit jobs ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "SFT Training Launch (completion-only loss)"
echo "============================================================"
echo "  Output:        sft_trained_models/completion/"
echo "  Modes:         ${MODES[*]}"
echo "  Model filter:  ${MODEL_FILTER:-all}"
echo "  Conda env:     ${CONDA_ENV}"
echo "  Epochs:        ${EPOCHS}"
echo "  Learning rate: ${LR}"
echo "  Batch size:    ${BATCH_SIZE} x ${GRAD_ACCUM} grad_accum"
echo "  Max seq len:   ${MAX_SEQ_LEN}"
echo "  GPUs:          ${GPUS} x H200"
echo "  Wall time:     ${WALL_TIME} (auto-chains up to ${MAX_CHAIN} jobs)"
echo "  Dry run:       ${DRY_RUN}"
echo "============================================================"
echo ""

JOB_COUNT=0

for mode in "${MODES[@]}"; do
    dataset="${DATASETS[$mode]}"

    if [ ! -f "${dataset}" ]; then
        echo "WARNING: Dataset not found: ${dataset}"
        continue
    fi

    echo "── Mode: ${mode} ──"
    echo ""

    for model_entry in "${MODELS[@]}"; do
        IFS='|' read -r model_id base_model_path <<< "${model_entry}"

        # Apply model filter
        if [ -n "${MODEL_FILTER}" ] && [ "${model_id}" != "${MODEL_FILTER}" ]; then
            continue
        fi

        # Check base model exists
        if [ ! -d "${base_model_path}" ]; then
            echo "  WARNING: Base model not found: ${base_model_path} (${model_id})"
            continue
        fi

        export_vars="MODEL_ID=${model_id}"
        export_vars="${export_vars},BASE_MODEL_PATH=${base_model_path}"
        export_vars="${export_vars},DATASET_PATH=${dataset}"
        export_vars="${export_vars},MODE=${mode}"
        export_vars="${export_vars},CONDA_ENV=${CONDA_ENV}"
        export_vars="${export_vars},EPOCHS=${EPOCHS}"
        export_vars="${export_vars},LR=${LR}"
        export_vars="${export_vars},BATCH_SIZE=${BATCH_SIZE}"
        export_vars="${export_vars},GRAD_ACCUM=${GRAD_ACCUM}"
        export_vars="${export_vars},MAX_SEQ_LEN=${MAX_SEQ_LEN}"
        export_vars="${export_vars},CHAIN_COUNT=0"
        export_vars="${export_vars},MAX_CHAIN=${MAX_CHAIN}"

        job_name="sft_${model_id}_${mode}"

        echo "  ${model_id} (${mode})"
        echo "    Base model:  ${base_model_path}"
        echo "    Dataset:     ${dataset}"
        echo "    Wall time:   ${WALL_TIME} (chains up to ${MAX_CHAIN}x)"
        echo "    Env vars:    MODEL_ID=${model_id}, MODE=${mode}, CONDA_ENV=${CONDA_ENV}"
        echo "                 EPOCHS=${EPOCHS}, LR=${LR}, BATCH_SIZE=${BATCH_SIZE}, GRAD_ACCUM=${GRAD_ACCUM}"

        if [ "${DRY_RUN}" = true ]; then
            echo "    [DRY RUN] sbatch --export=${export_vars} --job-name=${job_name} --gres=gpu:h200:${GPUS} --time=${WALL_TIME} ${SBATCH_SCRIPT}"
        else
            job_id=$(sbatch \
                --export="${export_vars}" \
                --job-name="${job_name}" \
                --gres="gpu:h200:${GPUS}" \
                --time="${WALL_TIME}" \
                "${SBATCH_SCRIPT}" | awk '{print $4}')
            echo "    Submitted job: ${job_id}"
        fi

        echo ""
        JOB_COUNT=$((JOB_COUNT + 1))
    done
done

echo "============================================================"
echo "Total jobs: ${JOB_COUNT}"
if [ "${DRY_RUN}" = false ]; then
    echo "Monitor with: squeue -u \$USER"
    echo ""
    echo "Jobs auto-chain: when an 8h job expires, it resubmits itself"
    echo "and resumes from the latest checkpoint. Max ${MAX_CHAIN} chains per model."
fi
echo "Logs at: ${LOG_DIR}/"
echo "Models at: ${PROJECT_ROOT}/sft_trained_models/completion/"
echo "============================================================"
