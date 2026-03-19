#!/usr/bin/env bash
# ============================================================================
# EvidenceRL Baseline RAG Generation Launcher (vLLM Version)
# ============================================================================
# High-performance BASELINE RAG generation (with retrieval) using vLLM.
# Runs on held-out test patients (1000-1099) for evaluation.
#
# This allows comparing RAG vs no-RAG performance on the same patients,
# with the speed benefits of vLLM.
#
# Usage:
#   ./launch_generation_baseline_rag_vllm.sh <model_id> [version]
#   ./launch_generation_baseline_rag_vllm.sh --list
#   ./launch_generation_baseline_rag_vllm.sh --help
#
# Output directory:
#   ${PROJECT_ROOT}/../generation_baseline_rag_output
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

# Large models that require 4 GPUs
LARGE_MODELS=(
    "Llama3-Med42-70B"
    "gpt-oss-120b"
    "Llama-3.3-70B"
    "Llama-4-Scout"
)

# Extra-large models that require 8 GPUs
EXTRA_LARGE_MODELS=(
    "Llama-4-Maverick"
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
    echo -e "${GREEN}EvidenceRL Baseline RAG Generation Launcher (vLLM Version)${NC}"
    echo -e "${CYAN}High-performance baseline RAG generation${NC}"
    echo ""
    echo "Runs generation WITH RAG on held-out test patients (1000-1099)."
    echo "Use this to compare RAG vs no-RAG performance."
    echo ""
    echo "Usage:"
    echo "  $0 <model_id> [version]    Launch baseline RAG generation"
    echo "  $0 --list                  List available models"
    echo "  $0 --help                  Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  CONDA_ENV        Conda environment name (default: evidencerl)"
    echo "  NUM_GPUS         GPUs per worker (default: 2, auto-set for large models)"
    echo "  PATIENT_OFFSET   Starting patient index (default: 1000)"
    echo "  TOTAL_PATIENTS   Total patients to process (default: 100)"
    echo ""
    echo "Note: 1 worker per 100 patients (auto-calculated)"
    echo ""
    echo "Output directory:"
    echo "  ${PROJECT_ROOT}/../generation_baseline_rag_output"
}

list_models() {
    echo -e "${GREEN}Available Models${NC}"
    echo ""
    for key in "${!MODELS[@]}"; do
        echo "  $key -> ${MODELS[$key]}"
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
if [ -z "${MODELS[$MODEL_ID]+isset}" ]; then
    echo -e "${RED}Error: Unknown model '${MODEL_ID}'${NC}"
    exit 1
fi

MODEL_NAME="${MODELS[$MODEL_ID]}"
FULL_MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

# Check if model exists
if [ ! -d "${FULL_MODEL_PATH}" ]; then
    echo -e "${YELLOW}Warning: Model directory not found: ${FULL_MODEL_PATH}${NC}"
    read -p "Continue anyway? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        exit 1
    fi
fi

# Create log directory
mkdir -p "${PROJECT_ROOT}/logs_evidencerl_baseline_rag"

# Pre-built FAISS index path (required for RAG)
FAISS_INDEX_DIR="${FAISS_INDEX_DIR:-${PROJECT_ROOT}/../faiss_index}"

# Check FAISS index exists
if [ ! -d "${FAISS_INDEX_DIR}" ] || [ ! -f "${FAISS_INDEX_DIR}/index.faiss" ]; then
    echo -e "${RED}Error: FAISS index not found at ${FAISS_INDEX_DIR}${NC}"
    echo "Build it first with: python scripts/generation/build_faiss_index.py"
    exit 1
fi

CONDA_ENV="${CONDA_ENV:-evidencerl}"

# Baseline RAG test set configuration:
# - Use 100 patients starting at index 1000 (held-out test set)
# - 1 worker per 100 patients
PATIENT_OFFSET="${PATIENT_OFFSET:-1000}"
TOTAL_PATIENTS="${TOTAL_PATIENTS:-100}"
PATIENTS_PER_WORKER=100
NUM_WORKERS=$(( (TOTAL_PATIENTS + PATIENTS_PER_WORKER - 1) / PATIENTS_PER_WORKER ))

# Auto-detect resources based on model size
if [ -z "${NUM_GPUS:-}" ]; then
    if is_extra_large_model "${MODEL_ID}"; then
        NUM_GPUS=8
        WORKER_CPUS="${WORKER_CPUS:-32}"
        WORKER_MEM="${WORKER_MEM:-256G}"
        WORKER_TIME="${WORKER_TIME:-01:00:00}"
    elif is_large_model "${MODEL_ID}"; then
        NUM_GPUS=4
        WORKER_CPUS="${WORKER_CPUS:-32}"
        WORKER_MEM="${WORKER_MEM:-256G}"
        WORKER_TIME="${WORKER_TIME:-02:00:00}"
    else
        NUM_GPUS=2
        WORKER_CPUS="${WORKER_CPUS:-16}"
        WORKER_MEM="${WORKER_MEM:-128G}"
        WORKER_TIME="${WORKER_TIME:-02:00:00}"
    fi
else
    WORKER_CPUS="${WORKER_CPUS:-16}"
    WORKER_MEM="${WORKER_MEM:-128G}"
    WORKER_TIME="${WORKER_TIME:-02:00:00}"
fi

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}EvidenceRL Baseline RAG Generation Launcher (vLLM Version)${NC}"
echo -e "${CYAN}High-performance baseline RAG generation${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "Mode:           ${YELLOW}BASELINE RAG (with retrieval)${NC}"
echo -e "Model ID:       ${BLUE}${MODEL_ID}${NC}"
echo -e "Model Name:     ${MODEL_NAME}"
echo -e "Model Path:     ${FULL_MODEL_PATH}"
echo -e "FAISS Index:    ${FAISS_INDEX_DIR}"
echo -e "Version:        ${VERSION}"
echo -e "Patient Range:  ${YELLOW}${PATIENT_OFFSET} to $((PATIENT_OFFSET + TOTAL_PATIENTS))${NC} (held-out test set)"
echo -e "Total Patients: ${TOTAL_PATIENTS}"
echo -e "Workers:        ${NUM_WORKERS} (${PATIENTS_PER_WORKER} patients each)"
echo -e "Resources:      ${NUM_GPUS}x GPU, ${WORKER_CPUS} CPUs, ${WORKER_MEM} RAM, ${WORKER_TIME}"
echo -e "Inference:      ${CYAN}vLLM${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

if is_extra_large_model "${MODEL_ID}"; then
    echo -e "[launcher] ${YELLOW}Extra-large model detected: ${MODEL_ID}${NC}"
elif is_large_model "${MODEL_ID}"; then
    echo -e "[launcher] ${YELLOW}Large model detected: ${MODEL_ID}${NC}"
fi

echo "[launcher] Submitting vLLM baseline RAG master job..."
echo "[launcher] Using conda environment: ${CONDA_ENV}"

# Build export string with all configuration
EXPORT_VARS="ALL"
EXPORT_VARS+=",MODEL_ID=${MODEL_ID}"
EXPORT_VARS+=",MODEL_NAME=${FULL_MODEL_PATH}"
EXPORT_VARS+=",VERSION=${VERSION}"
EXPORT_VARS+=",FAISS_INDEX_DIR=${FAISS_INDEX_DIR}"
EXPORT_VARS+=",CONDA_ENV=${CONDA_ENV}"
EXPORT_VARS+=",NUM_GPUS=${NUM_GPUS}"
EXPORT_VARS+=",PATIENT_OFFSET=${PATIENT_OFFSET}"
EXPORT_VARS+=",TOTAL_PATIENTS=${TOTAL_PATIENTS}"
EXPORT_VARS+=",NUM_WORKERS=${NUM_WORKERS}"
EXPORT_VARS+=",WORKER_CPUS=${WORKER_CPUS}"
EXPORT_VARS+=",WORKER_MEM=${WORKER_MEM}"
EXPORT_VARS+=",WORKER_TIME=${WORKER_TIME}"

JOB_ID=$(sbatch --parsable --export="${EXPORT_VARS}" "${SCRIPT_DIR}/generation_baseline_rag_vllm_master.sbatch")

echo -e "[launcher] Master job submitted: ${GREEN}${JOB_ID}${NC}"
echo ""
echo "Configuration summary:"
echo "  Mode:               BASELINE RAG (with retrieval)"
echo "  Patient range:      ${PATIENT_OFFSET} to $((PATIENT_OFFSET + TOTAL_PATIENTS)) (held-out test set)"
echo "  Total patients:     ${TOTAL_PATIENTS}"
echo "  Workers:            ${NUM_WORKERS} (1 worker per 100 patients)"
echo "  GPUs per worker:    ${NUM_GPUS}"
echo "  Inference engine:   vLLM"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f ${PROJECT_ROOT}/logs_evidencerl_baseline_rag/generation_baseline_rag_vllm_master_${JOB_ID}.out"
echo ""
echo "Output will be saved to:"
echo "  ${PROJECT_ROOT}/../generation_baseline_rag_output/${MODEL_ID}_v${VERSION}/"
