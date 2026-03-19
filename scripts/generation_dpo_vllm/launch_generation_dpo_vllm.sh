#!/usr/bin/env bash
# ============================================================================
# EvidenceRL DPO Generation Launcher (vLLM Version)
# ============================================================================
# Post-training inference with DPO fine-tuned models using vLLM + LoRA.
# Runs on held-out test patients (1000-1099) for evaluation.
#
# Usage:
#   ./launch_generation_dpo_vllm.sh <model_id> [version]
#   ./launch_generation_dpo_vllm.sh --list
#   ./launch_generation_dpo_vllm.sh --help
#
# Output directory:
#   ${PROJECT_ROOT}/../generation_dpo_output
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Configure these for your environment
BASE_MODEL_DIR="${MODEL_BASE_DIR:-//storage/ice-shared/bmed-sp-wang/Models}"

# DPO trained models directory
DPO_MODELS_DIR="${PROJECT_ROOT}/../dpo_trained_models"

# Map model IDs to base models and DPO adapter paths
# Format: model_id -> "base_model_name:dpo_subdir"
declare -A DPO_MODELS=(
    ["gemma-3-4b-dpo"]="gemma-3-4b-it:gemma-3-4b_v1.0_master_training_set"
    ["gemma-3-12b-dpo"]="gemma-3-12b-it:gemma-3-12b_v1.0_master_training_set"
    ["gemma-3-27b-dpo"]="gemma-3-27b-it:gemma-3-27b-it_v1.0_master_training_set"
    ["Llama-3.1-8B-dpo"]="Llama-3.1-8B-Instruct:Llama-3.1-8B-Instruct_v1.0_master_training_set"
    ["gpt-oss-20b-dpo"]="gpt-oss-20b:gpt-oss-20b_v1.0_master_training_set"
    ["gpt-oss-120b-dpo"]="gpt-oss-120b:gpt-oss-120b_v1.0_master_training_set"
    # Faithfulness-DPO models (Critique 4 response)
    ["Llama-3.2-3B-fdpo"]="Llama-3.2-3B-Instruct:Llama-3.2-3B_faithfulness_dpo"
    ["Llama-3.1-8B-fdpo"]="Llama-3.1-8B-Instruct:Llama-3.1-8B_faithfulness_dpo"
    ["gemma-3-4b-fdpo"]="gemma-3-4b-it:gemma-3-4b_faithfulness_dpo"
    ["gemma-3-12b-fdpo"]="gemma-3-12b-it:gemma-3-12b_faithfulness_dpo"
    ["gemma-3-27b-fdpo"]="gemma-3-27b-it:gemma-3-27b_faithfulness_dpo"
    ["gpt-oss-20b-fdpo"]="gpt-oss-20b:gpt-oss-20b_faithfulness_dpo"
)

# Large models that require 4 GPUs
LARGE_MODELS=(
    "gemma-3-27b-dpo"
    "gpt-oss-120b-dpo"
)

is_large_model() {
    local model_id="$1"
    for large_model in "${LARGE_MODELS[@]}"; do
        if [ "$model_id" == "$large_model" ]; then
            return 0
        fi
    done
    return 1
}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
    echo -e "${GREEN}EvidenceRL DPO Generation Launcher (vLLM Version)${NC}"
    echo -e "${CYAN}Post-training inference with DPO fine-tuned models${NC}"
    echo ""
    echo "Usage:"
    echo "  $0 <model_id> [version]    Launch DPO generation for specified model"
    echo "  $0 --list                  List available DPO models"
    echo "  $0 --help                  Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  CONDA_ENV        Conda environment name (default: evidencerl)"
    echo "  NUM_GPUS         GPUs per worker (default: 2, auto-set for large models)"
    echo "  PATIENT_OFFSET   Starting patient index (default: 1000)"
    echo "  TOTAL_PATIENTS   Total patients to process (default: 100)"
    echo "  USE_RAG          Set to 1 to use RAG, 0 for no-RAG (default: 0)"
    echo "  EXTRACTOR_MODEL  Model ID for fallback extraction (e.g., gemma-3-12b for small models)"
    echo "  MAX_TOKENS       Max generation tokens (default: 2048)"
    echo ""
    echo "Note: 1 worker per 100 patients (auto-calculated)"
    echo ""
    echo "Output directory:"
    echo "  ${PROJECT_ROOT}/../generation_dpo_output"
}

list_models() {
    echo -e "${GREEN}Available DPO Models${NC}"
    echo ""
    printf "%-25s %-25s %s\n" "Model ID" "Base Model" "Status"
    echo "----------------------------------------------------------------------"
    for model_id in "${!DPO_MODELS[@]}"; do
        IFS=':' read -r base_model dpo_subdir <<< "${DPO_MODELS[$model_id]}"
        dpo_path="${DPO_MODELS_DIR}/${dpo_subdir}/final_model"
        if [ -f "${dpo_path}/adapter_config.json" ]; then
            status="${GREEN}✓ Ready${NC}"
        else
            status="${RED}✗ Not trained${NC}"
        fi
        printf "%-25s %-25s " "$model_id" "$base_model"
        echo -e "$status"
    done | sort
}

# Parse arguments
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

case "$1" in
    --help|-h)
        show_help
        exit 0
        ;;
    --list|-l)
        list_models
        exit 0
        ;;
esac

MODEL_ID="$1"
VERSION="${2:-1.0}"

# Validate model
if [ -z "${DPO_MODELS[$MODEL_ID]+isset}" ]; then
    echo -e "${RED}Error: Unknown DPO model '${MODEL_ID}'${NC}"
    echo "Run with --list to see available models"
    exit 1
fi

# Parse base model and DPO subdir
IFS=':' read -r BASE_MODEL_NAME DPO_SUBDIR <<< "${DPO_MODELS[$MODEL_ID]}"
BASE_MODEL_PATH="${BASE_MODEL_DIR}/${BASE_MODEL_NAME}"
LORA_PATH="${DPO_MODELS_DIR}/${DPO_SUBDIR}/final_model"

# Check if base model exists
if [ ! -d "${BASE_MODEL_PATH}" ]; then
    echo -e "${RED}Error: Base model directory not found: ${BASE_MODEL_PATH}${NC}"
    exit 1
fi

# Check if DPO adapter exists
if [ ! -f "${LORA_PATH}/adapter_config.json" ]; then
    echo -e "${RED}Error: DPO adapter not found: ${LORA_PATH}${NC}"
    echo "Make sure the model has been trained and final_model/ exists"
    exit 1
fi

# Create log directory
mkdir -p "${PROJECT_ROOT}/logs_evidencerl_dpo"

CONDA_ENV="${CONDA_ENV:-evidencerl}"

# DPO test set configuration:
# - Use 100 patients starting at index 1000 (held-out test set)
# - 1 worker per 100 patients
PATIENT_OFFSET="${PATIENT_OFFSET:-1000}"
TOTAL_PATIENTS="${TOTAL_PATIENTS:-100}"
PATIENTS_PER_WORKER="${PATIENTS_PER_WORKER:-100}"
NUM_WORKERS=$(( (TOTAL_PATIENTS + PATIENTS_PER_WORKER - 1) / PATIENTS_PER_WORKER ))

# RAG mode (default: no-RAG for DPO evaluation)
USE_RAG="${USE_RAG:-0}"

# Extractor model resolution (for small models like 3B/4B and gpt-oss)
EXTRACTOR_MODEL_PATH=""
MAX_TOKENS="${MAX_TOKENS:-2048}"
if [ -n "${EXTRACTOR_MODEL:-}" ]; then
    # Map short name to full path
    declare -A EXTRACTOR_MODELS=(
        ["gemma-3-12b"]="gemma-3-12b-it"
        ["gemma-3-27b"]="gemma-3-27b-it"
        ["Llama-3.1-8B"]="Llama-3.1-8B-Instruct"
        ["Llama-3.3-70B"]="Llama-3.3-70B-Instruct"
    )
    if [ -n "${EXTRACTOR_MODELS[$EXTRACTOR_MODEL]+isset}" ]; then
        EXTRACTOR_MODEL_PATH="${BASE_MODEL_DIR}/${EXTRACTOR_MODELS[$EXTRACTOR_MODEL]}"
    else
        # Assume it's already a full path or direct model name
        EXTRACTOR_MODEL_PATH="${BASE_MODEL_DIR}/${EXTRACTOR_MODEL}"
    fi
    if [ ! -d "${EXTRACTOR_MODEL_PATH}" ]; then
        echo -e "${YELLOW}Warning: Extractor model directory not found: ${EXTRACTOR_MODEL_PATH}${NC}"
    fi
fi

# Auto-detect resources based on model size
# GPU_TYPE: "h200" for large models, "h200|h100" for small models (either OK)
if [ -z "${NUM_GPUS:-}" ]; then
    if is_large_model "${MODEL_ID}"; then
        NUM_GPUS=4
        GPU_TYPE="${GPU_TYPE:-H200}"
        WORKER_CPUS="${WORKER_CPUS:-32}"
        WORKER_MEM="${WORKER_MEM:-256G}"
        WORKER_TIME="${WORKER_TIME:-02:00:00}"
    else
        NUM_GPUS=2
        GPU_TYPE="${GPU_TYPE:-H200|H100}"
        WORKER_CPUS="${WORKER_CPUS:-16}"
        WORKER_MEM="${WORKER_MEM:-128G}"
        WORKER_TIME="${WORKER_TIME:-02:00:00}"
    fi
else
    GPU_TYPE="${GPU_TYPE:-H200|H100}"
    WORKER_CPUS="${WORKER_CPUS:-16}"
    WORKER_MEM="${WORKER_MEM:-128G}"
    WORKER_TIME="${WORKER_TIME:-02:00:00}"
fi

# Determine mode string
if [ "$USE_RAG" -eq 1 ]; then
    MODE_STR="DPO + RAG"
else
    MODE_STR="DPO + BASELINE (no-RAG)"
fi

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}EvidenceRL DPO Generation Launcher (vLLM Version)${NC}"
echo -e "${CYAN}Post-training inference with DPO fine-tuned models${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "Mode:           ${YELLOW}${MODE_STR}${NC}"
if [ -n "${EXTRACTOR_MODEL_PATH}" ]; then
    echo -e "Extractor:      ${CYAN}${EXTRACTOR_MODEL}${NC} (${EXTRACTOR_MODEL_PATH})"
fi
echo -e "Max Tokens:     ${MAX_TOKENS}"
echo -e "Model ID:       ${BLUE}${MODEL_ID}${NC}"
echo -e "Base Model:     ${BASE_MODEL_PATH}"
echo -e "LoRA Adapter:   ${LORA_PATH}"
echo -e "Version:        ${VERSION}"
echo -e "Patient Range:  ${YELLOW}${PATIENT_OFFSET} to $((PATIENT_OFFSET + TOTAL_PATIENTS))${NC} (held-out test set)"
echo -e "Total Patients: ${TOTAL_PATIENTS}"
echo -e "Workers:        ${NUM_WORKERS} (${PATIENTS_PER_WORKER} patients each)"
echo -e "Resources:      ${NUM_GPUS}x GPU, ${WORKER_CPUS} CPUs, ${WORKER_MEM} RAM, ${WORKER_TIME}"
echo -e "Inference:      ${CYAN}vLLM + LoRA${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

if is_large_model "${MODEL_ID}"; then
    echo -e "[launcher] ${YELLOW}Large model detected: ${MODEL_ID}${NC}"
fi

echo "[launcher] Submitting vLLM DPO master job..."
echo "[launcher] Using conda environment: ${CONDA_ENV}"

# Build export string with all configuration
EXPORT_VARS="ALL"
EXPORT_VARS+=",MODEL_ID=${MODEL_ID}"
EXPORT_VARS+=",BASE_MODEL_PATH=${BASE_MODEL_PATH}"
EXPORT_VARS+=",LORA_PATH=${LORA_PATH}"
EXPORT_VARS+=",VERSION=${VERSION}"
EXPORT_VARS+=",CONDA_ENV=${CONDA_ENV}"
EXPORT_VARS+=",NUM_GPUS=${NUM_GPUS}"
EXPORT_VARS+=",PATIENT_OFFSET=${PATIENT_OFFSET}"
EXPORT_VARS+=",TOTAL_PATIENTS=${TOTAL_PATIENTS}"
EXPORT_VARS+=",PATIENTS_PER_WORKER=${PATIENTS_PER_WORKER}"
EXPORT_VARS+=",NUM_WORKERS=${NUM_WORKERS}"
EXPORT_VARS+=",WORKER_CPUS=${WORKER_CPUS}"
EXPORT_VARS+=",WORKER_MEM=${WORKER_MEM}"
EXPORT_VARS+=",WORKER_TIME=${WORKER_TIME}"
EXPORT_VARS+=",USE_RAG=${USE_RAG}"
EXPORT_VARS+=",MAX_TOKENS=${MAX_TOKENS}"
EXPORT_VARS+=",EXTRACTOR_MODEL=${EXTRACTOR_MODEL_PATH}"
EXPORT_VARS+=",GPU_TYPE=${GPU_TYPE}"

JOB_ID=$(sbatch --parsable --export="${EXPORT_VARS}" "${SCRIPT_DIR}/generation_dpo_vllm_master.sbatch")

echo -e "[launcher] Master job submitted: ${GREEN}${JOB_ID}${NC}"
echo ""
echo "Configuration summary:"
echo "  Mode:               ${MODE_STR}"
echo "  Patient range:      ${PATIENT_OFFSET} to $((PATIENT_OFFSET + TOTAL_PATIENTS)) (held-out test set)"
echo "  Total patients:     ${TOTAL_PATIENTS}"
echo "  Workers:            ${NUM_WORKERS} (1 worker per 100 patients)"
echo "  GPUs per worker:    ${NUM_GPUS}"
echo "  Inference engine:   vLLM + LoRA"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f ${PROJECT_ROOT}/logs_evidencerl_dpo/generation_dpo_vllm_master_${JOB_ID}.out"
echo ""
echo "Output will be saved to:"
echo "  ${PROJECT_ROOT}/../generation_dpo_output/${MODEL_ID}_v${VERSION}/"
