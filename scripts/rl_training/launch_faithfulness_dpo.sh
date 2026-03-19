#!/usr/bin/env bash
# ============================================================================
# Launch Faithfulness-DPO training jobs
# ============================================================================
# Trains model backbones on grounding-based DPO preference pairs.
# Dataset must be built first with build_faithfulness_dpo_dataset.py.
#
# Usage:
#   ./launch_faithfulness_dpo.sh                       # All models
#   ./launch_faithfulness_dpo.sh --model gemma-3-12b   # Single model
#   ./launch_faithfulness_dpo.sh --dry-run
# ============================================================================

set -euo pipefail

# Configure these for your environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_BASE_DIR="${MODEL_BASE_DIR:-//storage/ice-shared/bmed-sp-wang/Models}"
SBATCH_SCRIPT="${SCRIPT_DIR}/train_dpo.sbatch"
DATASET_PATH="${PROJECT_ROOT}/training_data/dpo/faithfulness_dpo.json"
LOG_DIR="${PROJECT_ROOT}/logs_dpo"

mkdir -p "${LOG_DIR}"

# ── Defaults ─────────────────────────────────────────────────────────────────
MODEL_FILTER=""
DRY_RUN=false
CONDA_ENV="ragcon"
EPOCHS="2"
LR="5e-6"
BATCH_SIZE="1"
GRAD_ACCUM="8"
BETA="0.1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)      MODEL_FILTER="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --conda-env)  CONDA_ENV="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --lr)         LR="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
        --beta)       BETA="$2"; shift 2 ;;
        *)            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Format: MODEL_ID|BASE_MODEL_PATH|NUM_GPUS|GPU_CONSTRAINT|FLAGS
# GPU_CONSTRAINT: use "+" for OR (translated to "|" for SLURM --constraint)
#   "H100+H200" = either H100 or H200 (for <=20B models)
#   "H200"      = H200 only (for >=27B models needing >80GB/GPU)
MODELS=(
    "gemma-3-4b|${MODEL_BASE_DIR}/gemma-3-4b-it|2|H100+H200|"
    "gemma-3-12b|${MODEL_BASE_DIR}/gemma-3-12b-it|2|H100+H200|"
    "gemma-3-27b|${MODEL_BASE_DIR}/gemma-3-27b-it|4|H200|"
    "Llama-3.2-3B|${MODEL_BASE_DIR}/Llama-3.2-3B-Instruct|2|H100+H200|"
    "Llama-3.1-8B|${MODEL_BASE_DIR}/Llama-3.1-8B-Instruct|2|H100+H200|"
    "Llama-3.3-70B|${MODEL_BASE_DIR}/Llama-3.3-70B-Instruct|4|H200|4bit"
    "gpt-oss-20b|${MODEL_BASE_DIR}/gpt-oss-20b|2|H100+H200|"
)

if [ ! -f "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset not found: ${DATASET_PATH}"
    echo "  Build it first:"
    echo "    python build_faithfulness_dpo_dataset.py \\"
    echo "      --output ${DATASET_PATH} \\"
    echo "      --gap-threshold 0.4 --top-k 1 --bottom-k 1"
    exit 1
fi

echo "============================================================"
echo "Faithfulness-DPO Training Launch"
echo "============================================================"
echo "  Dataset: ${DATASET_PATH}"
echo "  Epochs:  ${EPOCHS}, LR: ${LR}, Beta: ${BETA}"
echo "  Model filter: ${MODEL_FILTER:-all}"
echo "  Dry run: ${DRY_RUN}"
echo "============================================================"
echo ""

JOB_COUNT=0

for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r model_id base_model_path num_gpus gpu_type flags <<< "${model_entry}"

    if [ -n "${MODEL_FILTER}" ] && [ "${model_id}" != "${MODEL_FILTER}" ]; then
        continue
    fi

    if [ ! -d "${base_model_path}" ]; then
        echo "  WARNING: Model not found: ${base_model_path} (${model_id})"
        continue
    fi

    local_4bit="0"
    [[ "${flags:-}" == *"4bit"* ]] && local_4bit="1"

    local_mem="128G"
    local_cpus="16"
    local_wall="08:00:00"
    [ "${num_gpus}" -ge 4 ] && local_mem="256G" && local_cpus="32" && local_wall="04:00:00"

    export_vars="MODEL_ID=${model_id}"
    export_vars="${export_vars},BASE_MODEL_PATH=${base_model_path}"
    export_vars="${export_vars},DATASET_PATH=${DATASET_PATH}"
    export_vars="${export_vars},CONDA_ENV=${CONDA_ENV}"
    export_vars="${export_vars},NUM_GPUS=${num_gpus}"
    export_vars="${export_vars},EPOCHS=${EPOCHS}"
    export_vars="${export_vars},LR=${LR}"
    export_vars="${export_vars},BATCH_SIZE=${BATCH_SIZE}"
    export_vars="${export_vars},GRAD_ACCUM=${GRAD_ACCUM}"
    export_vars="${export_vars},BETA=${BETA}"
    export_vars="${export_vars},LOAD_IN_4BIT=${local_4bit}"
    export_vars="${export_vars},CHAIN_COUNT=0,MAX_CHAIN=10"
    export_vars="${export_vars},GPU_TYPE=${gpu_type}"

    # Translate "+" to "|" for SLURM --constraint (e.g. "H100+H200" → "H100|H200")
    local_constraint="${gpu_type//+/|}"

    echo "  ${model_id} — ${num_gpus}x GPU [${local_constraint}], 4bit=${local_4bit}"

    if [ "${DRY_RUN}" = true ]; then
        echo "    [DRY RUN] sbatch --export=... --gres=gpu:${num_gpus} --constraint=\"${local_constraint}\" ${SBATCH_SCRIPT}"
    else
        job_id=$(sbatch \
            --export="${export_vars}" \
            --job-name="fdpo_${model_id}" \
            --gres="gpu:${num_gpus}" \
            --constraint="${local_constraint}" \
            --mem="${local_mem}" \
            --cpus-per-task="${local_cpus}" \
            --time="${local_wall}" \
            "${SBATCH_SCRIPT}" | awk '{print $4}')
        echo "    Submitted: ${job_id}"
    fi

    JOB_COUNT=$((JOB_COUNT + 1))
done

echo ""
echo "============================================================"
echo "Total jobs submitted: ${JOB_COUNT}"
echo "Models at: ${PROJECT_ROOT}/dpo_trained_models/"
echo "Logs at:   ${LOG_DIR}/"
echo "============================================================"
