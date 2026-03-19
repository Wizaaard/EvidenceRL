#!/usr/bin/env bash
# ============================================================================
# Launch GRPO training for EvidenceRL model backbones
# ============================================================================
#
# Trains base models with GRPO (Group Relative Policy Optimization) using
# three reward signals: format, accuracy (BioLORD-2023), grounding (NLI).
# Auto-chains 12h jobs for long training runs.
#
# Usage:
#   ./launch_grpo_training.sh                      # All models, no-rag mode
#   ./launch_grpo_training.sh --mode rag           # All models, RAG mode
#   ./launch_grpo_training.sh --model gemma-3-12b  # Single model
#   ./launch_grpo_training.sh --dry-run            # Print commands only
#   ./launch_grpo_training.sh --no-vllm            # Disable vLLM generation
# ============================================================================

set -euo pipefail

# Configure these for your environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_BASE_DIR="${MODEL_BASE_DIR:-//storage/ice-shared/bmed-sp-wang/Models}"
SBATCH_SCRIPT="${SCRIPT_DIR}/train_grpo.sbatch"
GRPO_DATA_DIR="${PROJECT_ROOT}/training_data/grpo"
LOG_DIR="${PROJECT_ROOT}/logs_grpo"

mkdir -p "${LOG_DIR}"

# ── Defaults ─────────────────────────────────────────────────────────────────
MODE_FILTER=""
MODEL_FILTER=""
DRY_RUN=false
CONDA_ENV="ragcon"
EPOCHS="2"
LR="5e-6"
BATCH_SIZE="1"
GRAD_ACCUM="8"
NUM_GENERATIONS="8"
TEMPERATURE="0.7"
BETA="0.04"
GPUS="4"
WALL_TIME="04:00:00"
MAX_CHAIN="15"
NO_VLLM="0"
OUTPUT_SUFFIX=""

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)             MODE_FILTER="$2"; shift 2 ;;
        --model)            MODEL_FILTER="$2"; shift 2 ;;
        --dry-run)          DRY_RUN=true; shift ;;
        --conda-env)        CONDA_ENV="$2"; shift 2 ;;
        --epochs)           EPOCHS="$2"; shift 2 ;;
        --lr)               LR="$2"; shift 2 ;;
        --batch-size)       BATCH_SIZE="$2"; shift 2 ;;
        --grad-accum)       GRAD_ACCUM="$2"; shift 2 ;;
        --num-generations)  NUM_GENERATIONS="$2"; shift 2 ;;
        --temperature)      TEMPERATURE="$2"; shift 2 ;;
        --beta)             BETA="$2"; shift 2 ;;
        --gpus)             GPUS="$2"; shift 2 ;;
        --wall-time)        WALL_TIME="$2"; shift 2 ;;
        --max-chain)        MAX_CHAIN="$2"; shift 2 ;;
        --no-vllm)          NO_VLLM="1"; shift ;;
        --output-suffix)    OUTPUT_SUFFIX="$2"; shift 2 ;;
        *)                  echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Model configurations ─────────────────────────────────────────────────────
# Format: MODEL_ID|BASE_MODEL_PATH|NUM_GPUS|VLLM_TP_SIZE|VLLM_GPU_UTIL|MAX_MODEL_LEN|FLAGS
# NUM_GPUS = total GPUs allocated via SLURM
# VLLM_TP_SIZE = tensor parallelism for vLLM (1 = each GPU is independent DP worker)
# VLLM_GPU_UTIL = fraction of GPU memory for vLLM (lower for larger models)
# MAX_MODEL_LEN = explicit max context length (empty = auto-detect)
# FLAGS = optional comma-separated flags: "no-vllm" to disable vLLM, "4bit" for QLoRA
#
# For models that fit on 1 GPU, TP=1 gives data parallelism = NUM_GPUS (faster)
# Only 70B needs TP>1 since it doesn't fit on a single GPU
# 27B needs explicit max_model_len (default 131K exceeds available KV cache with TP=1)
# 70B: vLLM TP=4 + 4bit QLoRA (monkey-patch set_weight_attrs for BnB compatibility)
#   (70B bf16=140GB > H200 139.8GB, so 4-bit is required; device_map per LOCAL_RANK)
# gpt-oss-20b needs no-vllm: vLLM 0.12.0 _load_weights_mxfp4 has a bug
#   (TypeError: default_weight_loader() got unexpected kwarg 'weight_name')
MODELS=(
    "gemma-3-4b|${MODEL_BASE_DIR}/gemma-3-4b-it|4|1|0.4|"
    "gemma-3-12b|${MODEL_BASE_DIR}/gemma-3-12b-it|4|1|0.4|"
    "gemma-3-27b|${MODEL_BASE_DIR}/gemma-3-27b-it|4|1|0.5|10000"
    "Llama-3.2-3B|${MODEL_BASE_DIR}/Llama-3.2-3B-Instruct|4|1|0.4|"
    "Llama-3.1-8B|${MODEL_BASE_DIR}/Llama-3.1-8B-Instruct|4|1|0.4|"
    "Llama-3.3-70B|${MODEL_BASE_DIR}/Llama-3.3-70B-Instruct|4|4|0.4|10000|4bit"
    "gpt-oss-20b|${MODEL_BASE_DIR}/gpt-oss-20b|4|1|0.4||no-vllm"
)

# Dataset files (built by build_grpo_dataset.py)
declare -A DATASETS
DATASETS["no-rag"]="${GRPO_DATA_DIR}/grpo_no_rag_prompts.json"
DATASETS["rag"]="${GRPO_DATA_DIR}/grpo_rag_prompts.json"

# Modes to train
if [ -n "${MODE_FILTER}" ]; then
    MODES=("${MODE_FILTER}")
else
    MODES=("no-rag")
fi

# ── Submit jobs ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "GRPO Training Launch (EvidenceRL)"
echo "============================================================"
echo "  Output:          grpo_trained_models/"
echo "  Modes:           ${MODES[*]}"
echo "  Model filter:    ${MODEL_FILTER:-all}"
echo "  Conda env:       ${CONDA_ENV}"
echo "  Epochs:          ${EPOCHS}"
echo "  Learning rate:   ${LR}"
echo "  Batch size:      ${BATCH_SIZE} x ${GRAD_ACCUM} grad_accum"
echo "  Num generations: ${NUM_GENERATIONS} per prompt"
echo "  Temperature:     ${TEMPERATURE}"
echo "  Beta (KL):       ${BETA}"
echo "  vLLM:            $( [ "${NO_VLLM}" = "1" ] && echo 'disabled' || echo 'enabled' )"
echo "  Wall time:       ${WALL_TIME} (auto-chains up to ${MAX_CHAIN} jobs)"
echo "  Dry run:         ${DRY_RUN}"
echo "============================================================"
echo ""

JOB_COUNT=0

for mode in "${MODES[@]}"; do
    dataset="${DATASETS[$mode]}"

    if [ ! -f "${dataset}" ]; then
        echo "WARNING: Dataset not found: ${dataset}"
        echo "  Run: python build_grpo_dataset.py --output-dir ${GRPO_DATA_DIR} --modes ${mode}"
        continue
    fi

    echo "── Mode: ${mode} ──"
    echo ""

    for model_entry in "${MODELS[@]}"; do
        IFS='|' read -r model_id base_model_path model_gpus model_tp model_gpu_util model_max_len model_flags <<< "${model_entry}"

        # Apply model filter
        if [ -n "${MODEL_FILTER}" ] && [ "${model_id}" != "${MODEL_FILTER}" ]; then
            continue
        fi

        # QOS coe-ice allows 960 GPU-minutes max per job: 4 GPUs × 4h = 960
        local_gpus="${model_gpus}"
        local_tp="${model_tp}"
        local_mem="256G"
        local_cpus="32"
        local_wall="${WALL_TIME}"
        local_dp=$((local_gpus / local_tp))

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
        export_vars="${export_vars},NUM_GPUS=${local_gpus}"
        export_vars="${export_vars},VLLM_TP=${local_tp}"
        export_vars="${export_vars},EPOCHS=${EPOCHS}"
        export_vars="${export_vars},LR=${LR}"
        export_vars="${export_vars},BATCH_SIZE=${BATCH_SIZE}"
        export_vars="${export_vars},GRAD_ACCUM=${GRAD_ACCUM}"
        export_vars="${export_vars},NUM_GENERATIONS=${NUM_GENERATIONS}"
        export_vars="${export_vars},TEMPERATURE=${TEMPERATURE}"
        export_vars="${export_vars},BETA=${BETA}"
        export_vars="${export_vars},VLLM_GPU_UTIL=${model_gpu_util:-0.4}"
        export_vars="${export_vars},MAX_MODEL_LEN=${model_max_len:-}"
        # Per-model flags (comma-separated) override global settings
        local_no_vllm="${NO_VLLM}"
        local_4bit="0"
        if [[ "${model_flags:-}" == *"no-vllm"* ]]; then
            local_no_vllm="1"
        fi
        if [[ "${model_flags:-}" == *"4bit"* ]]; then
            local_4bit="1"
        fi
        export_vars="${export_vars},NO_VLLM=${local_no_vllm}"
        export_vars="${export_vars},LOAD_IN_4BIT=${local_4bit}"
        export_vars="${export_vars},CHAIN_COUNT=0"
        export_vars="${export_vars},MAX_CHAIN=${MAX_CHAIN}"
        export_vars="${export_vars},OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-}"

        job_name="grpo_${OUTPUT_SUFFIX:+${OUTPUT_SUFFIX}_}${model_id}_${mode}"

        local_vllm_info="gpu_util=${model_gpu_util:-0.4}, max_len=${model_max_len:-auto}"
        if [ "${local_no_vllm}" = "1" ]; then
            local_vllm_info="vLLM=disabled (transformers generation)"
        fi
        if [ "${local_4bit}" = "1" ]; then
            local_vllm_info="${local_vllm_info}, 4bit=QLoRA"
        fi
        echo "  ${model_id} (${mode}) — ${local_gpus}x H200, TP=${local_tp}, DP=${local_dp}, ${local_vllm_info}"
        echo "    Base model:  ${base_model_path}"
        echo "    Dataset:     ${dataset}"

        if [ "${DRY_RUN}" = true ]; then
            echo "    [DRY RUN] sbatch --export=${export_vars} --job-name=${job_name} --gres=gpu:h200:${local_gpus} --mem=${local_mem} --cpus-per-task=${local_cpus} --time=${local_wall} ${SBATCH_SCRIPT}"
        else
            job_id=$(sbatch \
                --export="${export_vars}" \
                --job-name="${job_name}" \
                --gres="gpu:h200:${local_gpus}" \
                --mem="${local_mem}" \
                --cpus-per-task="${local_cpus}" \
                --time="${local_wall}" \
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
    echo "Jobs auto-chain: when a ${WALL_TIME} job expires, it resubmits"
    echo "and resumes from the latest checkpoint. Max ${MAX_CHAIN} chains."
fi
echo "Logs at:   ${LOG_DIR}/"
echo "Models at: ${PROJECT_ROOT}/grpo_trained_models/"
echo "============================================================"
