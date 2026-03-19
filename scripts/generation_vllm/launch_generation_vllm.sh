#!/usr/bin/env bash
# ============================================================================
# EvidenceRL Generation Launcher (vLLM Version)
# ============================================================================
# High-performance generation using vLLM for 14-24x faster inference.
#
# This script runs the generation pipeline with vLLM instead of
# HuggingFace Transformers. Key benefits:
# - 14-24x faster inference via PagedAttention and continuous batching
# - Tensor parallelism for efficient multi-GPU usage
# - Reduced time limits (2h workers vs 4h)
#
# Usage:
#   ./launch_generation_vllm.sh <model_id> [version]
#   ./launch_generation_vllm.sh --list              # List available models
#   ./launch_generation_vllm.sh --help              # Show help
#
# Examples:
#   ./launch_generation_vllm.sh gemma-3-27b          # Run gemma-3-27b-it
#   ./launch_generation_vllm.sh gemma-3-27b 1.1      # Run with version 1.1
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Configure these for your environment
MODEL_DIR="${MODEL_BASE_DIR:-//storage/ice-shared/bmed-sp-wang/Models}"

# Available models configuration
declare -A MODELS=(
    # Small models
    ["gemma-3-270m"]="gemma-3-270m-it"
    ["gemma-3-1b"]="gemma-3-1b-it"
    ["gemma-3-4b"]="gemma-3-4b-it"
    ["medgemma-4b"]="medgemma-4b-it"
    ["Llama-3.2-1B"]="Llama-3.2-1B-Instruct"
    ["Llama-3.2-3B"]="Llama-3.2-3B-Instruct"
    ["Llama-3.1-8B"]="Llama-3.1-8B-Instruct"
    ["Llama3-Med42-8B"]="Llama3-Med42-8B"
    ["gemma-3-12b"]="gemma-3-12b-it"

    # Medium models
    ["gemma-3-27b"]="gemma-3-27b-it"
    ["medgemma-27b"]="medgemma-27b-it"
    ["gpt-oss-20b"]="gpt-oss-20b"

    # Large models
    ["Llama3-Med42-70B"]="Llama3-Med42-70B"
    ["gpt-oss-120b"]="gpt-oss-120b"
    ["Llama-4-Scout"]="Llama-4-Scout-17B-16E-Instruct"
    ["Llama-4-Maverick"]="Llama-4-Maverick-17B-128E-Instruct"
    ["Llama-3.3-70B"]="Llama-3.3-70B-Instruct"

    # Extra-large models
    ["Llama-3.1-405B"]="Llama-3.1-405B"
)

# Large models that require 4 GPUs
LARGE_MODELS=(
    "Llama3-Med42-70B"
    "gpt-oss-120b"
    "Llama-3.3-70B"
    "Llama-4-Scout"
)

# Extra-large models that require 8 GPUs (8x H200, 32 CPUs, 256GB RAM)
EXTRA_LARGE_MODELS=(
    "Llama-4-Maverick"
    "Llama-3.1-405B"
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

is_extra_large_model() {
    local model_id="$1"
    for xl_model in "${EXTRA_LARGE_MODELS[@]}"; do
        if [ "$model_id" == "$xl_model" ]; then
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
    echo -e "${GREEN}EvidenceRL Generation Launcher (vLLM Version)${NC}"
    echo -e "${CYAN}14-24x faster inference than HuggingFace Transformers${NC}"
    echo ""
    echo "Usage:"
    echo "  $0 <model_id> [version]    Launch generation for specified model"
    echo "  $0 --list                  List available models"
    echo "  $0 --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 gemma-3-27b              # Run gemma-3-27b-it with vLLM"
    echo "  $0 gemma-3-27b 1.1          # Run with version 1.1"
    echo ""
    echo "Environment Variables:"
    echo "  CONDA_ENV        Conda environment name (default: evidencerl)"
    echo "  NUM_GPUS         GPUs per worker (default: 2, auto-set for large models)"
    echo "  TOTAL_PATIENTS   Total patients to process (default: 1000)"
    echo "  PATIENT_OFFSET   Starting patient index (default: 0 for training, 3700 for test)"
    echo "  NUM_WORKERS      Number of parallel workers (default: 5 standard, 10 large)"
    echo "  EXTRACTOR_MODEL  Model ID for fallback extraction (e.g., gemma-3-12b for 4b generator)"
    echo ""
    echo "vLLM Benefits:"
    echo "  - 14-24x faster than HuggingFace Transformers"
    echo "  - PagedAttention for efficient memory management"
    echo "  - Continuous batching for optimal throughput"
    echo "  - Tensor parallelism for multi-GPU inference"
    echo ""
    echo "Configuration (defaults):"
    echo "  Standard models:     2x H200, 128GB RAM, 16 CPUs, 2h"
    echo "  Large models:        4x H200, 256GB RAM, 32 CPUs, 2h"
    echo "  Extra-large models:  8x H200, 256GB RAM, 32 CPUs, 1h"
}

list_models() {
    echo -e "${GREEN}Available Models${NC}"
    echo ""
    echo -e "${BLUE}Small Models:${NC}"
    echo "  gemma-3-270m, gemma-3-1b, gemma-3-4b, medgemma-4b"
    echo "  Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, Llama3-Med42-8B, gemma-3-12b"
    echo ""
    echo -e "${BLUE}Medium Models:${NC}"
    echo "  gemma-3-27b, medgemma-27b, gpt-oss-20b"
    echo ""
    echo -e "${BLUE}Large Models (4x GPU):${NC}"
    echo "  Llama3-Med42-70B, gpt-oss-120b, Llama-4-Scout, Llama-3.3-70B"
    echo ""
    echo -e "${BLUE}Extra-Large Models (8x GPU):${NC}"
    echo "  Llama-4-Maverick, Llama-3.1-405B"
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

# Get model selection
MODEL_ID="$1"
VERSION="${2:-1.0}"

# Validate model
if [ -z "${MODELS[$MODEL_ID]+isset}" ]; then
    echo -e "${RED}Error: Unknown model '${MODEL_ID}'${NC}"
    echo ""
    echo "Available models:"
    for key in "${!MODELS[@]}"; do
        echo "  $key"
    done | sort
    exit 1
fi

MODEL_NAME="${MODELS[$MODEL_ID]}"
FULL_MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

# Check if model exists
if [ ! -d "${FULL_MODEL_PATH}" ]; then
    echo -e "${YELLOW}Warning: Model directory not found: ${FULL_MODEL_PATH}${NC}"
    echo "Make sure to download the model first."
    read -p "Continue anyway? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        exit 1
    fi
fi

# Resolve extractor model if specified
EXTRACTOR_MODEL_PATH=""
if [ -n "${EXTRACTOR_MODEL:-}" ]; then
    if [ -z "${MODELS[$EXTRACTOR_MODEL]+isset}" ]; then
        echo -e "${RED}Error: Unknown extractor model '${EXTRACTOR_MODEL}'${NC}"
        echo "Available models:"
        for key in "${!MODELS[@]}"; do
            echo "  $key"
        done | sort
        exit 1
    fi
    EXTRACTOR_MODEL_NAME="${MODELS[$EXTRACTOR_MODEL]}"
    EXTRACTOR_MODEL_PATH="${MODEL_DIR}/${EXTRACTOR_MODEL_NAME}"
    if [ ! -d "${EXTRACTOR_MODEL_PATH}" ]; then
        echo -e "${YELLOW}Warning: Extractor model directory not found: ${EXTRACTOR_MODEL_PATH}${NC}"
    fi
fi

# Create log directory
mkdir -p "${PROJECT_ROOT}/logs_evidencerl"

# Pre-built FAISS index path
FAISS_INDEX_DIR="${FAISS_INDEX_DIR:-${PROJECT_ROOT}/../faiss_index}"

# Conda environment
CONDA_ENV="${CONDA_ENV:-evidencerl}"

# Auto-detect resources based on model size
# vLLM workers are faster, so we use 2h instead of 4h
if [ -z "${NUM_GPUS:-}" ]; then
    if is_extra_large_model "${MODEL_ID}"; then
        NUM_GPUS=8
        WORKER_CPUS="${WORKER_CPUS:-32}"
        WORKER_MEM="${WORKER_MEM:-256G}"
        WORKER_TIME="${WORKER_TIME:-01:00:00}"
        TOTAL_PATIENTS="${TOTAL_PATIENTS:-100}"
        NUM_WORKERS="${NUM_WORKERS:-10}"
    elif is_large_model "${MODEL_ID}"; then
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
echo -e "${GREEN}EvidenceRL Generation Launcher (vLLM Version)${NC}"
echo -e "${CYAN}14-24x faster inference than HuggingFace Transformers${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "Model ID:       ${BLUE}${MODEL_ID}${NC}"
echo -e "Model Name:     ${MODEL_NAME}"
echo -e "Model Path:     ${FULL_MODEL_PATH}"
echo -e "Version:        ${VERSION}"
echo -e "Patient Range:  ${PATIENT_OFFSET} to $((PATIENT_OFFSET + TOTAL_PATIENTS)) (${TOTAL_PATIENTS} patients)"
echo -e "Workers:        ${NUM_WORKERS} (${PATIENTS_PER_WORKER} patients each)"
echo -e "Resources:      ${NUM_GPUS}x GPU, ${WORKER_CPUS} CPUs, ${WORKER_MEM} RAM, ${WORKER_TIME}"
if [ -n "${EXTRACTOR_MODEL_PATH}" ]; then
    echo -e "Extractor:      ${CYAN}${EXTRACTOR_MODEL}${NC} (${EXTRACTOR_MODEL_PATH})"
else
    echo -e "Extractor:      same as generator"
fi
echo -e "Inference:      ${CYAN}vLLM${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

echo "[launcher] Submitting vLLM master job..."
echo "[launcher] Using conda environment: ${CONDA_ENV}"

# Build export string
EXPORT_VARS="ALL"
EXPORT_VARS+=",MODEL_ID=${MODEL_ID}"
EXPORT_VARS+=",MODEL_NAME=${FULL_MODEL_PATH}"
EXPORT_VARS+=",VERSION=${VERSION}"
EXPORT_VARS+=",FAISS_INDEX_DIR=${FAISS_INDEX_DIR}"
EXPORT_VARS+=",CONDA_ENV=${CONDA_ENV}"
EXPORT_VARS+=",NUM_GPUS=${NUM_GPUS}"
EXPORT_VARS+=",TOTAL_PATIENTS=${TOTAL_PATIENTS}"
EXPORT_VARS+=",PATIENT_OFFSET=${PATIENT_OFFSET}"
EXPORT_VARS+=",NUM_WORKERS=${NUM_WORKERS}"
EXPORT_VARS+=",WORKER_CPUS=${WORKER_CPUS}"
EXPORT_VARS+=",WORKER_MEM=${WORKER_MEM}"
EXPORT_VARS+=",WORKER_TIME=${WORKER_TIME}"
EXPORT_VARS+=",EXTRACTOR_MODEL=${EXTRACTOR_MODEL_PATH}"

JOB_ID=$(sbatch --parsable --export="${EXPORT_VARS}" "${SCRIPT_DIR}/generation_master_vllm.sbatch")

echo -e "[launcher] Master job submitted: ${GREEN}${JOB_ID}${NC}"
echo ""
echo "Configuration summary:"
echo "  Patient range:      ${PATIENT_OFFSET} to $((PATIENT_OFFSET + TOTAL_PATIENTS))"
echo "  GPUs per worker:    ${NUM_GPUS}"
echo "  CPUs per worker:    ${WORKER_CPUS}"
echo "  Memory per worker:  ${WORKER_MEM}"
echo "  Time per worker:    ${WORKER_TIME} (reduced from 4h due to vLLM speed)"
if [ -n "${EXTRACTOR_MODEL_PATH}" ]; then
    echo "  Extractor model:   ${EXTRACTOR_MODEL} (${EXTRACTOR_MODEL_PATH})"
fi
echo "  Inference engine:   vLLM"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f ${PROJECT_ROOT}/logs_evidencerl/generation_vllm_master_${JOB_ID}.out"
echo ""
echo "Output will be saved to:"
echo "  ${PROJECT_ROOT}/../generation_output/${MODEL_ID}_v${VERSION}/"
