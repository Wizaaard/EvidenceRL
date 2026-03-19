#!/usr/bin/env bash
# ============================================================================
# BarExam GRPO-Trained Model Generation Launcher
# ============================================================================
# Launch test-set generation using GRPO-trained LoRA adapters.
# Reuses the existing barexam_generation.sbatch with --lora-path.
#
# Usage:
#   ./launch_barexam_grpo_gen.sh                     # All trained models
#   ./launch_barexam_grpo_gen.sh --model gemma-3-4b  # Single model
#   ./launch_barexam_grpo_gen.sh --list               # Show available
#   ./launch_barexam_grpo_gen.sh --dry-run            # Preview only
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
# Configure these for your environment
MODEL_DIR="${MODEL_BASE_DIR:-//storage/ice-shared/bmed-sp-wang/Models}"
GRPO_MODELS_DIR="${DATA_ROOT}/barexam_grpo_trained_models"

# ── Model configurations ──────────────────────────────────────────────
# model_id | base_model_name | num_gpus | gpu_constraint
MODELS=(
    "gemma-3-4b|gemma-3-4b-it|2|H100+H200"
    "gemma-3-12b|gemma-3-12b-it|2|H100+H200"
    "Llama-3.2-3B|Llama-3.2-3B-Instruct|2|H100+H200"
    "Llama-3.1-8B|Llama-3.1-8B-Instruct|2|H100+H200"
)

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── Parse arguments ───────────────────────────────────────────────────
SPLIT="test"
VERSION="v2.0"
SINGLE_MODEL=""
DRY_RUN="false"
LORA_SUBDIR="final_model"  # Use final_model by default; set to "checkpoint-XXX" to override
GRPO_SUFFIX=""  # e.g. "t10" for temperature=1.0 runs

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split) SPLIT="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        --model) SINGLE_MODEL="$2"; shift 2 ;;
        --lora-subdir) LORA_SUBDIR="$2"; shift 2 ;;
        --grpo-suffix) GRPO_SUFFIX="$2"; shift 2 ;;
        --dry-run) DRY_RUN="true"; shift ;;
        --list|-l)
            echo -e "${GREEN}Available GRPO-trained BarExam models:${NC}"
            for entry in "${MODELS[@]}"; do
                IFS='|' read -r mid mname ngpus gtype <<< "${entry}"
                grpo_dir="${GRPO_MODELS_DIR}/${mid}_barexam_grpo_rag"
                if [ -d "${grpo_dir}/final_model" ]; then
                    echo -e "  ${GREEN}✓${NC} ${mid} (${grpo_dir}/final_model)"
                elif ls -d "${grpo_dir}"/checkpoint-* &>/dev/null; then
                    latest=$(ls -d "${grpo_dir}"/checkpoint-* | sort -t- -k2 -n | tail -1)
                    echo -e "  ${YELLOW}~${NC} ${mid} ($(basename ${latest}))"
                else
                    echo -e "  ${RED}✗${NC} ${mid} (no checkpoints)"
                fi
            done
            exit 0 ;;
        --help|-h)
            echo "Usage: $0 [--split test|validation] [--model <id>] [--version <v>] [--lora-subdir <dir>] [--dry-run]"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Create logs directory
mkdir -p "${DATA_ROOT}/logs_barexam"

# ── Launch jobs ───────────────────────────────────────────────────────
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}BarExam GRPO-Trained Model Generation${NC}"
echo -e "  Split:       ${SPLIT}"
echo -e "  Version:     ${VERSION}"
echo -e "  LoRA subdir: ${LORA_SUBDIR}"
echo -e "  Mode:        RAG (gold passage)"
[ "${DRY_RUN}" = "true" ] && echo -e "  ${YELLOW}** DRY RUN — no jobs submitted **${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo ""

LAUNCHED=0

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_id model_name num_gpus gpu_constraint <<< "${entry}"

    if [ -n "${SINGLE_MODEL}" ] && [ "${model_id}" != "${SINGLE_MODEL}" ]; then
        continue
    fi

    # Resolve LoRA path
    if [ -n "${GRPO_SUFFIX}" ]; then
        grpo_dir="${GRPO_MODELS_DIR}/${model_id}_barexam_grpo_${GRPO_SUFFIX}_rag"
    else
        grpo_dir="${GRPO_MODELS_DIR}/${model_id}_barexam_grpo_rag"
    fi
    if [ -d "${grpo_dir}/${LORA_SUBDIR}" ]; then
        lora_path="${grpo_dir}/${LORA_SUBDIR}"
    else
        # Fall back to latest checkpoint
        lora_path=$(ls -d "${grpo_dir}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
        if [ -z "${lora_path}" ]; then
            echo -e "  ${RED}SKIP: ${model_id} — no LoRA adapter found in ${grpo_dir}${NC}"
            continue
        fi
    fi

    # Validate adapter exists
    if [ ! -f "${lora_path}/adapter_config.json" ]; then
        echo -e "  ${RED}SKIP: ${model_id} — no adapter_config.json in ${lora_path}${NC}"
        continue
    fi

    full_model_path="${MODEL_DIR}/${model_name}"
    local_constraint="${gpu_constraint//+/|}"

    if [ "${num_gpus}" -ge 4 ]; then
        cpus=32; mem="256G"
    else
        cpus=16; mem="128G"
    fi

    if [ -n "${GRPO_SUFFIX}" ]; then
        MODE_TAG="grpo_${GRPO_SUFFIX}_rag"
    else
        MODE_TAG="grpo_rag"
    fi
    job_name="barexam_grpo_gen_${GRPO_SUFFIX:+${GRPO_SUFFIX}_}${model_id}"

    EXPORT_VARS="MODEL_NAME=${full_model_path}"
    EXPORT_VARS="${EXPORT_VARS},MODEL_ID=${model_id}"
    EXPORT_VARS="${EXPORT_VARS},SPLIT=${SPLIT}"
    EXPORT_VARS="${EXPORT_VARS},VERSION=${VERSION}"
    EXPORT_VARS="${EXPORT_VARS},NUM_GPUS=${num_gpus}"
    EXPORT_VARS="${EXPORT_VARS},BAREXAM_DATA=${DATA_ROOT}/data/barexam"
    EXPORT_VARS="${EXPORT_VARS},NO_RAG=false"
    EXPORT_VARS="${EXPORT_VARS},CONDA_ENV=ragcon"
    EXPORT_VARS="${EXPORT_VARS},PROJECT_ROOT=${PROJECT_ROOT}"
    EXPORT_VARS="${EXPORT_VARS},DATA_ROOT=${DATA_ROOT}"
    EXPORT_VARS="${EXPORT_VARS},LORA_PATH=${lora_path}"
    EXPORT_VARS="${EXPORT_VARS},MODE_TAG=${MODE_TAG}"

    # Extractor model for small models
    case "${model_id}" in
        gemma-3-4b|Llama-3.2-3B)
            EXPORT_VARS="${EXPORT_VARS},EXTRACTOR_MODEL=${MODEL_DIR}/gemma-3-12b-it"
            ;;
    esac

    echo -e "  ${BLUE}${model_id}${NC} (${num_gpus}x GPU, LoRA: $(basename ${lora_path}))"

    if [ "${DRY_RUN}" = "true" ]; then
        echo "    Would submit: sbatch --job-name=${job_name} --gres=gpu:${num_gpus} --constraint=${local_constraint}"
        echo "    LoRA: ${lora_path}"
    else
        sbatch \
            --job-name="${job_name}" \
            --gres="gpu:${num_gpus}" \
            --constraint="${local_constraint}" \
            --cpus-per-task="${cpus}" \
            --mem="${mem}" \
            --export="${EXPORT_VARS}" \
            "${SCRIPT_DIR}/barexam_generation.sbatch"
    fi

    LAUNCHED=$((LAUNCHED + 1))
done

echo ""
echo -e "${GREEN}Launched ${LAUNCHED} GRPO generation jobs.${NC}"
echo "Monitor: squeue -u \$USER | grep barexam_grpo_gen"
echo ""
echo "Output will be at:"
echo "  ${DATA_ROOT}/barexam_output/${SPLIT}/{MODEL_ID}_${MODE_TAG}-${VERSION}/"
echo ""
echo "After completion, run metrics:"
echo "  ./launch_barexam_metrics.sh --version ${VERSION} --mode-tag ${MODE_TAG}"
