#!/usr/bin/env bash
# ============================================================================
# EvidenceRL SFT Generation Launcher (vLLM Version)
# ============================================================================
# Generates diagnoses using SFT-trained LoRA adapters in no-RAG (baseline) mode.
# Uses vLLM with enable_lora for efficient LoRA inference.
#
# Usage:
#   ./launch_generation_sft_vllm.sh <model_id> [version]
#   ./launch_generation_sft_vllm.sh --list
#   ./launch_generation_sft_vllm.sh --help
#
# Examples:
#   ./launch_generation_sft_vllm.sh gemma-3-12b-sft-no-rag 1.0-TEST
#   EXTRACTOR_MODEL=gemma-3-12b ./launch_generation_sft_vllm.sh gemma-3-4b-sft-no-rag 1.0-TEST
#
# Output directory:
#   ${PROJECT_ROOT}/../generation_sft_output
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# Configure these for your environment
MODEL_DIR="${MODEL_BASE_DIR:-//storage/ice-shared/bmed-sp-wang/Models}"
SFT_MODELS_DIR="${PROJECT_ROOT}/sft_trained_models/completion"

# SFT model mapping: model_id -> "base_model_name:sft_adapter_subdir"
declare -A SFT_MODELS=(
    ["gemma-3-4b-sft-no-rag"]="gemma-3-4b-it:gemma-3-4b_sft_no-rag"
    ["gemma-3-12b-sft-no-rag"]="gemma-3-12b-it:gemma-3-12b_sft_no-rag"
    ["gemma-3-27b-sft-no-rag"]="gemma-3-27b-it:gemma-3-27b_sft_no-rag"
    ["Llama-3.2-3B-sft-no-rag"]="Llama-3.2-3B-Instruct:Llama-3.2-3B_sft_no-rag"
    ["Llama-3.1-8B-sft-no-rag"]="Llama-3.1-8B-Instruct:Llama-3.1-8B_sft_no-rag"
    ["Llama-3.3-70B-sft-no-rag"]="Llama-3.3-70B-Instruct:Llama-3.3-70B_sft_no-rag"
    ["gpt-oss-20b-sft-no-rag"]="gpt-oss-20b:gpt-oss-20b_sft_no-rag"
)

# Large models requiring 4 GPUs
LARGE_MODELS=(
    "Llama-3.3-70B-sft-no-rag"
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

# Extractor model mapping (for resolving EXTRACTOR_MODEL short names to paths)
declare -A EXTRACTOR_MODELS=(
    ["gemma-3-12b"]="gemma-3-12b-it"
    ["gemma-3-27b"]="gemma-3-27b-it"
    ["Llama-3.1-8B"]="Llama-3.1-8B-Instruct"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
    echo -e "${GREEN}EvidenceRL SFT Generation Launcher (vLLM Version)${NC}"
    echo -e "${CYAN}SFT LoRA adapter inference in no-RAG mode${NC}"
    echo ""
    echo "Usage:"
    echo "  $0 <model_id> [version]    Launch SFT generation for specified model"
    echo "  $0 --list                  List available SFT models"
    echo "  $0 --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 gemma-3-12b-sft-no-rag 1.0-TEST"
    echo "  EXTRACTOR_MODEL=gemma-3-12b $0 gemma-3-4b-sft-no-rag 1.0-TEST"
    echo "  MAX_TOKENS=4096 EXTRACTOR_MODEL=gemma-3-12b $0 gpt-oss-20b-sft-no-rag 1.0-TEST"
    echo ""
    echo "Environment Variables:"
    echo "  CONDA_ENV        Conda environment name (default: evidencerl)"
    echo "  NUM_GPUS         GPUs per worker (default: 2, auto-set for large models)"
    echo "  TOTAL_PATIENTS   Total patients to process (default: 1000)"
    echo "  PATIENT_OFFSET   Starting patient index (default: 0)"
    echo "  NUM_WORKERS      Number of parallel workers (default: 5 standard, 10 large)"
    echo "  EXTRACTOR_MODEL  Model ID for fallback extraction (e.g., gemma-3-12b)"
    echo "  MAX_TOKENS       Max generation tokens (default: 2048)"
    echo ""
    echo "Configuration (defaults):"
    echo "  Standard models: 2x H200, 128GB RAM, 16 CPUs, 2h"
    echo "  Large models:    4x H200, 256GB RAM, 32 CPUs, 2h"
}

list_models() {
    echo -e "${GREEN}Available SFT Models (no-RAG)${NC}"
    echo ""
    echo -e "${BLUE}Standard Models (2x GPU):${NC}"
    echo "  gemma-3-4b-sft-no-rag, gemma-3-12b-sft-no-rag, gemma-3-27b-sft-no-rag"
    echo "  Llama-3.2-3B-sft-no-rag, Llama-3.1-8B-sft-no-rag"
    echo "  gpt-oss-20b-sft-no-rag"
    echo ""
    echo -e "${BLUE}Large Models (4x GPU):${NC}"
    echo "  Llama-3.3-70B-sft-no-rag"
}

# Parse arguments
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

case "$1" in
    --help|-h) show_help; exit 0 ;;
    --list|-l) list_models; exit 0 ;;
esac

# Get model selection
MODEL_ID="$1"
VERSION="${2:-1.0}"

# Validate model
if [ -z "${SFT_MODELS[$MODEL_ID]+isset}" ]; then
    echo -e "${RED}Error: Unknown SFT model '${MODEL_ID}'${NC}"
    echo ""
    echo "Available SFT models:"
    for key in "${!SFT_MODELS[@]}"; do
        echo "  $key"
    done | sort
    exit 1
fi

# Split model value into base model and SFT subdir
MODEL_VALUE="${SFT_MODELS[$MODEL_ID]}"
BASE_MODEL_NAME="${MODEL_VALUE%%:*}"
SFT_SUBDIR="${MODEL_VALUE##*:}"

BASE_MODEL_PATH="${MODEL_DIR}/${BASE_MODEL_NAME}"
LORA_PATH="${SFT_MODELS_DIR}/${SFT_SUBDIR}/final_model"

# Validate base model exists
if [ ! -d "${BASE_MODEL_PATH}" ]; then
    echo -e "${RED}Error: Base model not found: ${BASE_MODEL_PATH}${NC}"
    exit 1
fi

# Validate LoRA adapter exists
if [ ! -f "${LORA_PATH}/adapter_config.json" ]; then
    echo -e "${RED}Error: LoRA adapter not found: ${LORA_PATH}/adapter_config.json${NC}"
    echo "Make sure SFT training has completed for this model."
    exit 1
fi

# Resolve extractor model if specified
EXTRACTOR_MODEL_PATH=""
if [ -n "${EXTRACTOR_MODEL:-}" ]; then
    if [ -n "${EXTRACTOR_MODELS[$EXTRACTOR_MODEL]+isset}" ]; then
        EXTRACTOR_MODEL_NAME="${EXTRACTOR_MODELS[$EXTRACTOR_MODEL]}"
        EXTRACTOR_MODEL_PATH="${MODEL_DIR}/${EXTRACTOR_MODEL_NAME}"
    else
        echo -e "${RED}Error: Unknown extractor model '${EXTRACTOR_MODEL}'${NC}"
        exit 1
    fi
fi

# Create log directory
mkdir -p "${PROJECT_ROOT}/logs_evidencerl_sft"

# Conda environment
CONDA_ENV="${CONDA_ENV:-evidencerl}"

# Auto-detect resources based on model size
if [ -z "${NUM_GPUS:-}" ]; then
    if is_large_model "${MODEL_ID}"; then
        NUM_GPUS=4
        WORKER_CPUS="${WORKER_CPUS:-32}"
        WORKER_MEM="${WORKER_MEM:-256G}"
        WORKER_TIME="${WORKER_TIME:-02:00:00}"
        TOTAL_PATIENTS="${TOTAL_PATIENTS:-1000}"
        NUM_WORKERS="${NUM_WORKERS:-10}"
    else
        NUM_GPUS=2
        WORKER_CPUS="${WORKER_CPUS:-16}"
        WORKER_MEM="${WORKER_MEM:-128G}"
        WORKER_TIME="${WORKER_TIME:-02:00:00}"
        TOTAL_PATIENTS="${TOTAL_PATIENTS:-1000}"
        NUM_WORKERS="${NUM_WORKERS:-5}"
    fi
else
    WORKER_CPUS="${WORKER_CPUS:-16}"
    WORKER_MEM="${WORKER_MEM:-128G}"
    WORKER_TIME="${WORKER_TIME:-02:00:00}"
    TOTAL_PATIENTS="${TOTAL_PATIENTS:-1000}"
    NUM_WORKERS="${NUM_WORKERS:-5}"
fi

PATIENT_OFFSET="${PATIENT_OFFSET:-0}"
PATIENTS_PER_WORKER=$((TOTAL_PATIENTS / NUM_WORKERS))

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}EvidenceRL SFT Generation Launcher (vLLM Version)${NC}"
echo -e "${YELLOW}Mode: SFT LoRA (no-RAG, baseline generation)${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "Model ID:       ${BLUE}${MODEL_ID}${NC}"
echo -e "Base Model:     ${BASE_MODEL_PATH}"
echo -e "LoRA Adapter:   ${LORA_PATH}"
echo -e "Version:        ${VERSION}"
echo -e "Patient Range:  ${PATIENT_OFFSET} to $((PATIENT_OFFSET + TOTAL_PATIENTS)) (${TOTAL_PATIENTS} patients)"
echo -e "Workers:        ${NUM_WORKERS} (${PATIENTS_PER_WORKER} patients each)"
echo -e "Resources:      ${NUM_GPUS}x GPU, ${WORKER_CPUS} CPUs, ${WORKER_MEM} RAM, ${WORKER_TIME}"
if [ -n "${EXTRACTOR_MODEL_PATH}" ]; then
    echo -e "Extractor:      ${CYAN}${EXTRACTOR_MODEL}${NC} (${EXTRACTOR_MODEL_PATH})"
else
    echo -e "Extractor:      same as generator"
fi
echo -e "Inference:      ${CYAN}vLLM + LoRA${NC} (no-RAG)"
echo -e "${GREEN}============================================================${NC}"
echo ""

echo "[launcher] Submitting SFT vLLM master job..."
echo "[launcher] Using conda environment: ${CONDA_ENV}"

# Build export string
EXPORT_VARS="ALL"
EXPORT_VARS+=",MODEL_ID=${MODEL_ID}"
EXPORT_VARS+=",BASE_MODEL_PATH=${BASE_MODEL_PATH}"
EXPORT_VARS+=",LORA_PATH=${LORA_PATH}"
EXPORT_VARS+=",VERSION=${VERSION}"
EXPORT_VARS+=",CONDA_ENV=${CONDA_ENV}"
EXPORT_VARS+=",NUM_GPUS=${NUM_GPUS}"
EXPORT_VARS+=",TOTAL_PATIENTS=${TOTAL_PATIENTS}"
EXPORT_VARS+=",PATIENT_OFFSET=${PATIENT_OFFSET}"
EXPORT_VARS+=",NUM_WORKERS=${NUM_WORKERS}"
EXPORT_VARS+=",WORKER_CPUS=${WORKER_CPUS}"
EXPORT_VARS+=",WORKER_MEM=${WORKER_MEM}"
EXPORT_VARS+=",WORKER_TIME=${WORKER_TIME}"
EXPORT_VARS+=",EXTRACTOR_MODEL=${EXTRACTOR_MODEL_PATH}"
EXPORT_VARS+=",MAX_TOKENS=${MAX_TOKENS:-2048}"

JOB_ID=$(sbatch --parsable --export="${EXPORT_VARS}" "${SCRIPT_DIR}/generation_sft_vllm_master.sbatch")

echo -e "[launcher] Master job submitted: ${GREEN}${JOB_ID}${NC}"
echo ""
echo "Configuration summary:"
echo "  Mode:               SFT LoRA (no-RAG)"
echo "  Patient range:      ${PATIENT_OFFSET} to $((PATIENT_OFFSET + TOTAL_PATIENTS))"
echo "  GPUs per worker:    ${NUM_GPUS}"
echo "  CPUs per worker:    ${WORKER_CPUS}"
echo "  Memory per worker:  ${WORKER_MEM}"
echo "  Time per worker:    ${WORKER_TIME}"
if [ -n "${EXTRACTOR_MODEL_PATH}" ]; then
    echo "  Extractor model:    ${EXTRACTOR_MODEL} (${EXTRACTOR_MODEL_PATH})"
fi
echo "  Inference engine:   vLLM + LoRA (no-RAG)"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f ${PROJECT_ROOT}/logs_evidencerl_sft/generation_sft_vllm_master_${JOB_ID}.out"
echo ""
echo "Output will be saved to:"
echo "  ${PROJECT_ROOT}/../generation_sft_output/${MODEL_ID}_v${VERSION}/"
