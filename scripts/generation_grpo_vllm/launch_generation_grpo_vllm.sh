#!/usr/bin/env bash
# ============================================================================
# EvidenceRL GRPO Generation Launcher (vLLM Version)
# ============================================================================
# Generates diagnoses using GRPO-trained LoRA adapters in no-RAG mode.
# Uses vLLM with enable_lora for efficient LoRA inference.
#
# Automatically finds the latest GRPO checkpoint for the specified model.
#
# Usage:
#   ./launch_generation_grpo_vllm.sh <model_id> [version]
#   ./launch_generation_grpo_vllm.sh --list
#   ./launch_generation_grpo_vllm.sh --help
#
# Examples:
#   ./launch_generation_grpo_vllm.sh gemma-3-12b 1.0-TEST
#   EXTRACTOR_MODEL=gemma-3-12b ./launch_generation_grpo_vllm.sh gemma-3-4b 1.0-TEST
#
# Output directory:
#   ${PROJECT_ROOT}/../generation_grpo_output
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# Configure these for your environment
MODEL_DIR="${MODEL_BASE_DIR:-//storage/ice-shared/bmed-sp-wang/Models}"
GRPO_MODELS_DIR="${PROJECT_ROOT}/grpo_trained_models"

# GRPO model mapping: model_id -> base_model_name
declare -A GRPO_MODELS=(
    ["gemma-3-4b"]="gemma-3-4b-it"
    ["gemma-3-12b"]="gemma-3-12b-it"
    ["gemma-3-27b"]="gemma-3-27b-it"
    ["Llama-3.2-3B"]="Llama-3.2-3B-Instruct"
    ["Llama-3.1-8B"]="Llama-3.1-8B-Instruct"
    ["Llama-3.3-70B"]="Llama-3.3-70B-Instruct"
    ["gpt-oss-20b"]="gpt-oss-20b"
)

# Large models requiring 4 GPUs
LARGE_MODELS=(
    "Llama-3.3-70B"
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
    echo -e "${GREEN}EvidenceRL GRPO Generation Launcher (vLLM Version)${NC}"
    echo -e "${CYAN}GRPO LoRA adapter inference in no-RAG mode${NC}"
    echo ""
    echo "Usage:"
    echo "  $0 <model_id> [version]    Launch GRPO generation for specified model"
    echo "  $0 --list                  List available GRPO models"
    echo "  $0 --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 gemma-3-12b 1.0-TEST"
    echo "  EXTRACTOR_MODEL=gemma-3-12b $0 gemma-3-4b 1.0-TEST"
    echo ""
    echo "Environment Variables:"
    echo "  CONDA_ENV        Conda environment name (default: ragcon)"
    echo "  NUM_GPUS         GPUs per worker (default: 2, auto-set for large models)"
    echo "  TOTAL_PATIENTS   Total patients to process (default: 1000)"
    echo "  PATIENT_OFFSET   Starting patient index (default: 0)"
    echo "  NUM_WORKERS      Number of parallel workers (default: 5 standard, 10 large)"
    echo "  EXTRACTOR_MODEL  Model ID for fallback extraction (e.g., gemma-3-12b)"
    echo "  MAX_TOKENS       Max generation tokens (default: 2048)"
    echo "  LORA_PATH        Override auto-detected LoRA path"
    echo ""
    echo "Configuration (defaults):"
    echo "  Standard models: 2x H200, 128GB RAM, 16 CPUs, 2h"
    echo "  Large models:    4x H200, 256GB RAM, 32 CPUs, 2h"
}

list_models() {
    echo -e "${GREEN}Available GRPO Models (no-RAG)${NC}"
    echo ""
    echo -e "${BLUE}Standard Models (2x GPU):${NC}"
    for key in "${!GRPO_MODELS[@]}"; do
        if ! is_large_model "$key"; then
            local grpo_dir="${GRPO_MODELS_DIR}/${key}_grpo_no-rag"
            local latest_ckpt=$(ls -d "${grpo_dir}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
            local status="no checkpoint"
            if [ -n "$latest_ckpt" ] && [ -f "${latest_ckpt}/adapter_config.json" ]; then
                status="$(basename "$latest_ckpt")"
            fi
            echo "  ${key} (${status})"
        fi
    done
    echo ""
    echo -e "${BLUE}Large Models (4x GPU):${NC}"
    for key in "${LARGE_MODELS[@]}"; do
        local grpo_dir="${GRPO_MODELS_DIR}/${key}_grpo_no-rag"
        local latest_ckpt=$(ls -d "${grpo_dir}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
        local status="no checkpoint"
        if [ -n "$latest_ckpt" ] && [ -f "${latest_ckpt}/adapter_config.json" ]; then
            status="$(basename "$latest_ckpt")"
        fi
        echo "  ${key} (${status})"
    done
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
if [ -z "${GRPO_MODELS[$MODEL_ID]+isset}" ]; then
    echo -e "${RED}Error: Unknown GRPO model '${MODEL_ID}'${NC}"
    echo ""
    echo "Available GRPO models:"
    for key in "${!GRPO_MODELS[@]}"; do
        echo "  $key"
    done | sort
    exit 1
fi

# Resolve base model path
BASE_MODEL_NAME="${GRPO_MODELS[$MODEL_ID]}"
BASE_MODEL_PATH="${MODEL_DIR}/${BASE_MODEL_NAME}"

# Validate base model exists
if [ ! -d "${BASE_MODEL_PATH}" ]; then
    echo -e "${RED}Error: Base model not found: ${BASE_MODEL_PATH}${NC}"
    exit 1
fi

# Resolve LoRA path: use LORA_PATH env var or auto-detect latest checkpoint
if [ -z "${LORA_PATH:-}" ]; then
    GRPO_DIR="${GRPO_MODELS_DIR}/${MODEL_ID}_grpo_no-rag"
    if [ ! -d "${GRPO_DIR}" ]; then
        echo -e "${RED}Error: GRPO output directory not found: ${GRPO_DIR}${NC}"
        exit 1
    fi

    # Find the latest checkpoint
    LORA_PATH=$(ls -d "${GRPO_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -z "${LORA_PATH}" ]; then
        echo -e "${RED}Error: No checkpoints found in ${GRPO_DIR}${NC}"
        exit 1
    fi
fi

# Validate LoRA adapter exists
if [ ! -f "${LORA_PATH}/adapter_config.json" ]; then
    echo -e "${RED}Error: LoRA adapter not found: ${LORA_PATH}/adapter_config.json${NC}"
    echo "Make sure GRPO training has completed for this model."
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
mkdir -p "${PROJECT_ROOT}/code/logs_evidencerl_grpo" 2>/dev/null || true
mkdir -p "logs_evidencerl_grpo" 2>/dev/null || true

# Conda environment
CONDA_ENV="${CONDA_ENV:-ragcon}"

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
echo -e "${GREEN}EvidenceRL GRPO Generation Launcher (vLLM Version)${NC}"
echo -e "${YELLOW}Mode: GRPO LoRA (no-RAG, generation from GRPO checkpoint)${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "Model ID:       ${BLUE}${MODEL_ID}${NC}"
echo -e "Base Model:     ${BASE_MODEL_PATH}"
echo -e "LoRA Adapter:   ${CYAN}${LORA_PATH}${NC}"
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

echo "[launcher] Submitting GRPO vLLM master job..."
echo "[launcher] Using conda environment: ${CONDA_ENV}"

# Export variables for sbatch (avoid --export=ALL which can corrupt hyphenated values)
export MODEL_ID="${MODEL_ID}"
export MODEL_NAME="${BASE_MODEL_PATH}"
export LORA_PATH="${LORA_PATH}"
export MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
export VERSION="${VERSION}"
export CONDA_ENV="${CONDA_ENV}"
export NUM_GPUS="${NUM_GPUS}"
export TOTAL_PATIENTS="${TOTAL_PATIENTS}"
export PATIENT_OFFSET="${PATIENT_OFFSET}"
export NUM_WORKERS="${NUM_WORKERS}"
export WORKER_CPUS="${WORKER_CPUS}"
export WORKER_MEM="${WORKER_MEM}"
export WORKER_TIME="${WORKER_TIME}"
export EXTRACTOR_MODEL="${EXTRACTOR_MODEL_PATH}"
export MAX_TOKENS="${MAX_TOKENS:-2048}"
export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-${MODEL_ID}_v${VERSION}}"

JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/generation_grpo_vllm_master.sbatch")

echo -e "[launcher] Master job submitted: ${GREEN}${JOB_ID}${NC}"
echo ""
echo "Configuration summary:"
echo "  Mode:               GRPO LoRA (no-RAG)"
echo "  Patient range:      ${PATIENT_OFFSET} to $((PATIENT_OFFSET + TOTAL_PATIENTS))"
echo "  GPUs per worker:    ${NUM_GPUS}"
echo "  CPUs per worker:    ${WORKER_CPUS}"
echo "  Memory per worker:  ${WORKER_MEM}"
echo "  Time per worker:    ${WORKER_TIME}"
echo "  LoRA checkpoint:    $(basename ${LORA_PATH})"
if [ -n "${EXTRACTOR_MODEL_PATH}" ]; then
    echo "  Extractor model:    ${EXTRACTOR_MODEL} (${EXTRACTOR_MODEL_PATH})"
fi
echo "  Inference engine:   vLLM + LoRA (no-RAG)"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f logs_evidencerl_grpo/generation_grpo_vllm_master_${JOB_ID}.out"
echo ""
echo "Output will be saved to:"
echo "  ${PROJECT_ROOT}/../generation_grpo_output/${MODEL_ID}_v${VERSION}/"
