#!/usr/bin/env bash
# ============================================================================
# EvidenceRL Generation Launcher
# ============================================================================
# Main entry point for launching distributed generation experiments.
#
# Usage:
#   ./launch_generation.sh <model_id> [version]
#   ./launch_generation.sh --list              # List available models
#   ./launch_generation.sh --help              # Show help
#
# Examples:
#   ./launch_generation.sh gemma-3-4b          # Run gemma-3-4b-it
#   ./launch_generation.sh gemma-3-4b 1.1      # Run with version 1.1
#   ./launch_generation.sh Llama-3.1-8B        # Run Llama-3.1-8B-Instruct
#
# This script:
# 1. Validates the model selection
# 2. Submits the master job which orchestrates worker jobs
# 3. Workers process 100 patients each (10 workers for 1000 total)
# 4. Results are merged automatically when all workers complete
#
# Output directory:
#   ${PROJECT_ROOT}/../generation_output
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Configure these for your environment
MODEL_DIR="${MODEL_BASE_DIR:-//storage/ice-shared/bmed-sp-wang/Models}"

# Available models configuration
# Format: MODEL_ID:MODEL_NAME (relative to MODEL_DIR or absolute path)
declare -A MODELS=(
    # Small models
    ["gemma-3-270m"]="gemma-3-270m-it"
    ["gemma-3-1b"]="gemma-3-1b-it"
    ["gemma-3-4b"]="gemma-3-4b-it"
    ["medgemma-4b"]="medgemma-4b-it"
    ["Llama-3.2-1B"]="Llama-3.2-1B-Instruct"
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
)

# Models marked for running (from download_models.py "# do" comments)
MODELS_TO_RUN=(
    "gemma-3-4b"
    "Llama-3.1-8B"
    "gemma-3-12b"
    "gemma-3-27b"
    "gpt-oss-20b"
)

# Large models that require 4 GPUs (>100GB model size or 70B+ parameters)
# These need 4x H200 (576GB VRAM) for reasonable batch sizes
LARGE_MODELS=(
    "Llama3-Med42-70B"
    "gpt-oss-120b"
    "Llama-3.3-70B"
    "Llama-4-Scout"
)

# Extra-large models that require 8 GPUs (MoE models with many experts)
# Llama-4-Maverick has 128 experts, needs 8x H200 for efficient inference
EXTRA_LARGE_MODELS=(
    "Llama-4-Maverick"
)

# Function to check if model is large (4 GPUs)
is_large_model() {
    local model_id="$1"
    for large_model in "${LARGE_MODELS[@]}"; do
        if [ "$model_id" == "$large_model" ]; then
            return 0
        fi
    done
    return 1
}

# Function to check if model is extra-large (8 GPUs)
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
NC='\033[0m' # No Color

show_help() {
    echo -e "${GREEN}EvidenceRL Generation Launcher${NC}"
    echo ""
    echo "Usage:"
    echo "  $0 <model_id> [version]    Launch generation for specified model"
    echo "  $0 --list                  List available models"
    echo "  $0 --list-todo             List models marked for running"
    echo "  $0 --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 gemma-3-4b              # Run gemma-3-4b-it with default version"
    echo "  $0 gemma-3-4b 1.1          # Run gemma-3-4b-it with version 1.1"
    echo "  $0 Llama-3.1-8B            # Run Llama-3.1-8B-Instruct"
    echo ""
    echo "Environment Variables:"
    echo "  CONDA_ENV        Conda environment name (default: evidencerl)"
    echo "  BATCH_SIZE       Override auto-detected batch size"
    echo "  NUM_GPUS         GPUs per worker (default: 2, auto-set to 4/8 for large models)"
    echo "  TOTAL_PATIENTS   Total patients to process (default: 1000)"
    echo "  NUM_WORKERS      Number of parallel workers (default: 20)"
    echo "  WORKER_CPUS      CPUs per worker (default: 16/32/64 based on model)"
    echo "  WORKER_MEM       Memory per worker (default: 128G/256G based on model)"
    echo "  WORKER_TIME      Time limit per worker (default: 04:00:00)"
    echo ""
    echo "Configuration (defaults):"
    echo "  Standard models:     2x H200, 128GB RAM, 16 CPUs, 4h"
    echo "  Large models:        4x H200, 256GB RAM, 32 CPUs, 4h"
    echo "  Extra-large models:  8x H200, 256GB RAM, 64 CPUs, 2h"
    echo ""
    echo "Large models (auto-detected, 4 GPUs):"
    echo "  Llama3-Med42-70B, gpt-oss-120b, Llama-3.3-70B, Llama-4-Scout"
    echo ""
    echo "Extra-large models (auto-detected, 8 GPUs):"
    echo "  Llama-4-Maverick"
    echo ""
    echo "Output directory:"
    echo "  ${PROJECT_ROOT}/../generation_output"
}

list_models() {
    echo -e "${GREEN}Available Models${NC}"
    echo ""
    echo -e "${BLUE}Small Models:${NC}"
    echo "  gemma-3-270m     -> gemma-3-270m-it"
    echo "  gemma-3-1b       -> gemma-3-1b-it"
    echo "  gemma-3-4b       -> gemma-3-4b-it"
    echo "  medgemma-4b      -> medgemma-4b-it"
    echo "  Llama-3.2-1B     -> Llama-3.2-1B-Instruct"
    echo "  Llama-3.1-8B     -> Llama-3.1-8B-Instruct"
    echo "  Llama3-Med42-8B  -> Llama3-Med42-8B"
    echo "  gemma-3-12b      -> gemma-3-12b-it"
    echo ""
    echo -e "${BLUE}Medium Models:${NC}"
    echo "  gemma-3-27b      -> gemma-3-27b-it"
    echo "  medgemma-27b     -> medgemma-27b-it"
    echo "  gpt-oss-20b      -> gpt-oss-20b"
    echo ""
    echo -e "${BLUE}Large Models:${NC}"
    echo "  Llama3-Med42-70B -> Llama3-Med42-70B"
    echo "  gpt-oss-120b     -> gpt-oss-120b"
    echo "  Llama-4-Scout    -> Llama-4-Scout-17B-16E-Instruct"
    echo "  Llama-4-Maverick -> Llama-4-Maverick-17B-128E-Instruct"
    echo "  Llama-3.3-70B    -> Llama-3.3-70B-Instruct"
}

list_todo() {
    echo -e "${GREEN}Models Marked for Running${NC}"
    echo ""
    for model_id in "${MODELS_TO_RUN[@]}"; do
        model_name="${MODELS[$model_id]}"
        echo "  ${model_id} -> ${model_name}"
    done
    echo ""
    echo "Run with: $0 <model_id>"
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
    --list-todo)
        list_todo
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
    echo "Make sure to download the model first:"
    echo "  python scripts/download_models.py --model ${MODEL_NAME}"
    echo ""
    read -p "Continue anyway? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        exit 1
    fi
fi

# Create log directory
mkdir -p "${PROJECT_ROOT}/logs_evidencerl"

# Pre-built FAISS index path (optional but recommended for faster startup)
FAISS_INDEX_DIR="${FAISS_INDEX_DIR:-${PROJECT_ROOT}/../faiss_index}"

# Conda environment (allow override via environment variable)
CONDA_ENV="${CONDA_ENV:-evidencerl}"

# Auto-detect resources based on model size
# Priority: explicit env vars > model-based auto-detection > defaults
if [ -z "${NUM_GPUS:-}" ]; then
    if is_extra_large_model "${MODEL_ID}"; then
        NUM_GPUS=8
        WORKER_CPUS="${WORKER_CPUS:-64}"
        WORKER_MEM="${WORKER_MEM:-256G}"
        WORKER_TIME="${WORKER_TIME:-02:00:00}"
        TOTAL_PATIENTS="${TOTAL_PATIENTS:-100}"
        NUM_WORKERS="${NUM_WORKERS:-10}"
    elif is_large_model "${MODEL_ID}"; then
        NUM_GPUS=4
        WORKER_CPUS="${WORKER_CPUS:-32}"
        WORKER_MEM="${WORKER_MEM:-256G}"
        WORKER_TIME="${WORKER_TIME:-04:00:00}"
        TOTAL_PATIENTS="${TOTAL_PATIENTS:-1000}"
        NUM_WORKERS="${NUM_WORKERS:-20}"
    else
        NUM_GPUS=2
        WORKER_CPUS="${WORKER_CPUS:-16}"
        WORKER_MEM="${WORKER_MEM:-128G}"
        WORKER_TIME="${WORKER_TIME:-04:00:00}"
        TOTAL_PATIENTS="${TOTAL_PATIENTS:-1000}"
        NUM_WORKERS="${NUM_WORKERS:-20}"
    fi
else
    # Explicit NUM_GPUS set, use defaults for other vars if not set
    WORKER_CPUS="${WORKER_CPUS:-16}"
    WORKER_MEM="${WORKER_MEM:-128G}"
    WORKER_TIME="${WORKER_TIME:-04:00:00}"
    TOTAL_PATIENTS="${TOTAL_PATIENTS:-1000}"
    NUM_WORKERS="${NUM_WORKERS:-20}"
fi

# Calculate patients per worker
PATIENTS_PER_WORKER=$((TOTAL_PATIENTS / NUM_WORKERS))

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}EvidenceRL Generation Launcher${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "Model ID:       ${BLUE}${MODEL_ID}${NC}"
echo -e "Model Name:     ${MODEL_NAME}"
echo -e "Model Path:     ${FULL_MODEL_PATH}"
echo -e "Version:        ${VERSION}"
echo -e "Total Patients: ${TOTAL_PATIENTS}"
echo -e "Workers:        ${NUM_WORKERS} (${PATIENTS_PER_WORKER} patients each)"
echo -e "Resources:      ${NUM_GPUS}x GPU, ${WORKER_CPUS} CPUs, ${WORKER_MEM} RAM, ${WORKER_TIME}"
echo -e "${GREEN}============================================================${NC}"
echo ""

# Display model tier info
if is_extra_large_model "${MODEL_ID}"; then
    echo -e "[launcher] ${YELLOW}Extra-large model detected: ${MODEL_ID}${NC}"
    echo -e "[launcher] Using 8x H200 GPUs, ${WORKER_MEM} RAM, ${WORKER_CPUS} CPUs, ${WORKER_TIME}"
elif is_large_model "${MODEL_ID}"; then
    echo -e "[launcher] ${YELLOW}Large model detected: ${MODEL_ID}${NC}"
    echo -e "[launcher] Using 4x H200 GPUs, ${WORKER_MEM} RAM, ${WORKER_CPUS} CPUs, ${WORKER_TIME}"
fi

echo "[launcher] Submitting master job..."
echo "[launcher] Using conda environment: ${CONDA_ENV}"

# Build export string with all configuration
EXPORT_VARS="ALL"
EXPORT_VARS+=",MODEL_ID=${MODEL_ID}"
EXPORT_VARS+=",MODEL_NAME=${FULL_MODEL_PATH}"
EXPORT_VARS+=",VERSION=${VERSION}"
EXPORT_VARS+=",FAISS_INDEX_DIR=${FAISS_INDEX_DIR}"
EXPORT_VARS+=",CONDA_ENV=${CONDA_ENV}"
EXPORT_VARS+=",NUM_GPUS=${NUM_GPUS}"
EXPORT_VARS+=",TOTAL_PATIENTS=${TOTAL_PATIENTS}"
EXPORT_VARS+=",NUM_WORKERS=${NUM_WORKERS}"
EXPORT_VARS+=",WORKER_CPUS=${WORKER_CPUS}"
EXPORT_VARS+=",WORKER_MEM=${WORKER_MEM}"
EXPORT_VARS+=",WORKER_TIME=${WORKER_TIME}"
[ -n "${BATCH_SIZE:-}" ] && EXPORT_VARS+=",BATCH_SIZE=${BATCH_SIZE}"

JOB_ID=$(sbatch --parsable --export="${EXPORT_VARS}" "${SCRIPT_DIR}/generation_master.sbatch")

echo -e "[launcher] Master job submitted: ${GREEN}${JOB_ID}${NC}"
echo ""
echo "Configuration summary:"
echo "  GPUs per worker:    ${NUM_GPUS}"
echo "  CPUs per worker:    ${WORKER_CPUS}"
echo "  Memory per worker:  ${WORKER_MEM}"
echo "  Time per worker:    ${WORKER_TIME}"
echo "  Total patients:     ${TOTAL_PATIENTS}"
echo "  Number of workers:  ${NUM_WORKERS}"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f ${PROJECT_ROOT}/logs_evidencerl/master_${JOB_ID}.out"
echo ""
echo "Output will be saved to:"
echo "  ${PROJECT_ROOT}/../generation_output/${MODEL_ID}_v${VERSION}/"
