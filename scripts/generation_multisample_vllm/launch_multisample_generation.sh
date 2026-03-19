#!/usr/bin/env bash
# ============================================================================
# EvidenceRL Multi-Sample Generation Launcher (vLLM)
# ============================================================================
# Generates N completions per patient for per-model fdpo dataset construction.
# Uses the same pipeline as baseline generation (--no-rag) with --num-samples.
#
# Usage:
#   ./launch_multisample_generation.sh <model_id> [version]
#
# Examples:
#   NUM_SAMPLES=3 TOTAL_PATIENTS=3700 PATIENT_OFFSET=0 \
#     ./launch_multisample_generation.sh gemma-3-4b
#
#   EXTRACTOR_MODEL=gemma-3-12b NUM_SAMPLES=3 TOTAL_PATIENTS=3700 \
#     MAX_TOKENS=4096 ./launch_multisample_generation.sh gpt-oss-20b
#
# Output directory:
#   ${PROJECT_ROOT}/../permodel_fdpo_samples
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Configure these for your environment
MODEL_DIR="${MODEL_BASE_DIR:-//storage/ice-shared/bmed-sp-wang/Models}"

# Available models (only the 6 fdpo models)
declare -A MODELS=(
    ["gemma-3-4b"]="gemma-3-4b-it"
    ["gemma-3-12b"]="gemma-3-12b-it"
    ["gemma-3-27b"]="gemma-3-27b-it"
    ["Llama-3.2-3B"]="Llama-3.2-3B-Instruct"
    ["Llama-3.1-8B"]="Llama-3.1-8B-Instruct"
    ["gpt-oss-20b"]="gpt-oss-20b"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
    echo -e "${GREEN}EvidenceRL Multi-Sample Generation Launcher (vLLM)${NC}"
    echo -e "${CYAN}Generate N completions per patient for per-model fdpo${NC}"
    echo ""
    echo "Usage:"
    echo "  $0 <model_id> [version]"
    echo ""
    echo "Models (6 fdpo models only):"
    echo "  gemma-3-4b, gemma-3-12b, gemma-3-27b"
    echo "  Llama-3.2-3B, Llama-3.1-8B, gpt-oss-20b"
    echo ""
    echo "Environment Variables:"
    echo "  NUM_SAMPLES      Completions per patient (default: 3)"
    echo "  TOTAL_PATIENTS   Total patients to process (default: 3700)"
    echo "  PATIENT_OFFSET   Starting patient index (default: 0)"
    echo "  EXTRACTOR_MODEL  Extractor model ID for small/gpt models"
    echo "  MAX_TOKENS       Max tokens to generate (default: 2048, use 4096 for gpt)"
    echo "  CONDA_ENV        Conda environment name (default: ragcon)"
    echo "  NUM_WORKERS      Number of parallel workers (default: 5)"
}

if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

MODEL_ID="$1"
VERSION="${2:-1.0}"

# Validate model
if [ -z "${MODELS[$MODEL_ID]+isset}" ]; then
    echo -e "${RED}Error: Unknown model '${MODEL_ID}'${NC}"
    echo "Available: ${!MODELS[*]}"
    exit 1
fi

MODEL_NAME="${MODELS[$MODEL_ID]}"
FULL_MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

# Check model exists
if [ ! -d "${FULL_MODEL_PATH}" ]; then
    echo -e "${YELLOW}Warning: Model directory not found: ${FULL_MODEL_PATH}${NC}"
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
        exit 1
    fi
    EXTRACTOR_MODEL_NAME="${MODELS[$EXTRACTOR_MODEL]}"
    EXTRACTOR_MODEL_PATH="${MODEL_DIR}/${EXTRACTOR_MODEL_NAME}"
fi

# Configuration
CONDA_ENV="${CONDA_ENV:-ragcon}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
NUM_GPUS="${NUM_GPUS:-2}"
WORKER_CPUS="${WORKER_CPUS:-16}"
WORKER_MEM="${WORKER_MEM:-128G}"
WORKER_TIME="${WORKER_TIME:-04:00:00}"
TOTAL_PATIENTS="${TOTAL_PATIENTS:-3700}"
PATIENT_OFFSET="${PATIENT_OFFSET:-0}"
NUM_WORKERS="${NUM_WORKERS:-5}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
PATIENTS_PER_WORKER=$((TOTAL_PATIENTS / NUM_WORKERS))

# Create log directory
mkdir -p "${PROJECT_ROOT}/logs_multisample"

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}EvidenceRL Multi-Sample Generation (vLLM)${NC}"
echo -e "${YELLOW}Mode: BASELINE (no-RAG) + ${NUM_SAMPLES} samples/patient${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "Model ID:       ${BLUE}${MODEL_ID}${NC}"
echo -e "Model Path:     ${FULL_MODEL_PATH}"
echo -e "Version:        ${VERSION}"
echo -e "Num Samples:    ${CYAN}${NUM_SAMPLES}${NC}"
echo -e "Max Tokens:     ${MAX_TOKENS}"
echo -e "Patient Range:  ${PATIENT_OFFSET} to $((PATIENT_OFFSET + TOTAL_PATIENTS)) (${TOTAL_PATIENTS} patients)"
echo -e "Workers:        ${NUM_WORKERS} (${PATIENTS_PER_WORKER} patients each)"
echo -e "Resources:      ${NUM_GPUS}x GPU, ${WORKER_CPUS} CPUs, ${WORKER_MEM} RAM, ${WORKER_TIME}"
if [ -n "${EXTRACTOR_MODEL_PATH}" ]; then
    echo -e "Extractor:      ${CYAN}${EXTRACTOR_MODEL}${NC}"
else
    echo -e "Extractor:      same as generator"
fi
echo -e "${GREEN}============================================================${NC}"
echo ""

echo "[launcher] Submitting multi-sample master job..."

EXPORT_VARS="ALL"
EXPORT_VARS+=",MODEL_ID=${MODEL_ID}"
EXPORT_VARS+=",MODEL_NAME=${FULL_MODEL_PATH}"
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
EXPORT_VARS+=",NUM_SAMPLES=${NUM_SAMPLES}"
EXPORT_VARS+=",MAX_TOKENS=${MAX_TOKENS}"

JOB_ID=$(sbatch --parsable --export="${EXPORT_VARS}" "${SCRIPT_DIR}/multisample_generation_master.sbatch")

echo -e "[launcher] Master job submitted: ${GREEN}${JOB_ID}${NC}"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f ${PROJECT_ROOT}/logs_multisample/multisample_master_${JOB_ID}.out"
echo ""
echo "Output will be saved to:"
echo "  ${PROJECT_ROOT}/../permodel_fdpo_samples/${MODEL_ID}/"
