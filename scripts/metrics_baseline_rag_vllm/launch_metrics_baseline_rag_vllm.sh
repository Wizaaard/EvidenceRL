#!/usr/bin/env bash
# ============================================================================
# EvidenceRL Baseline RAG Metrics Launcher (vLLM Version)
# ============================================================================
# High-performance metrics computation using vLLM for the LLM judge.
# Runs on BASELINE RAG (with retrieval) generation output.
#
# Usage:
#   ./scripts/metrics_baseline_rag_vllm/launch_metrics_baseline_rag_vllm.sh <model_id> [version]
#   ./scripts/metrics_baseline_rag_vllm/launch_metrics_baseline_rag_vllm.sh --list
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Paths - BASELINE RAG directories
GENERATION_OUTPUT_DIR="${PROJECT_ROOT}/../generation_baseline_rag_output"
METRICS_OUTPUT_DIR="${PROJECT_ROOT}/../metrics_baseline_rag_output"
# Configure these for your environment
MODEL_DIR="${MODEL_BASE_DIR:-//storage/ice-shared/bmed-sp-wang/Models}"
LOG_DIR="${PROJECT_ROOT}/logs_evidencerl_baseline_rag"

# Fixed judge model
JUDGE_MODEL_NAME="medgemma-27b-it"
JUDGE_MODEL_PATH="${MODEL_DIR}/${JUDGE_MODEL_NAME}"

# Available models
declare -A MODELS=(
    ["gemma-3-270m"]="gemma-3-270m-it"
    ["gemma-3-1b"]="gemma-3-1b-it"
    ["gemma-3-4b"]="gemma-3-4b-it"
    ["medgemma-4b"]="medgemma-4b-it"
    ["Llama-3.2-1B"]="Llama-3.2-1B-Instruct"
    ["Llama-3.1-8B"]="Llama-3.1-8B-Instruct"
    ["Llama3-Med42-8B"]="Llama3-Med42-8B"
    ["gemma-3-12b"]="gemma-3-12b-it"
    ["gemma-3-27b"]="gemma-3-27b-it"
    ["medgemma-27b"]="medgemma-27b-it"
    ["gpt-oss-20b"]="gpt-oss-20b"
    ["Llama3-Med42-70B"]="Llama3-Med42-70B"
    ["gpt-oss-120b"]="gpt-oss-120b"
    ["Llama-4-Scout"]="Llama-4-Scout-17B-16E-Instruct"
    ["Llama-4-Maverick"]="Llama-4-Maverick-17B-128E-Instruct"
    ["Llama-3.3-70B"]="Llama-3.3-70B-Instruct"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
    cat << EOF
EvidenceRL Baseline RAG Metrics Launcher (vLLM Version)

Computes metrics on BASELINE RAG (with retrieval) generation output using vLLM.

Usage:
    $(basename "$0") <model_id> [version]    Run metrics for baseline RAG
    $(basename "$0") --list                  List available models
    $(basename "$0") --help                  Show this help

Judge Model: ${JUDGE_MODEL_NAME} (fixed, uses vLLM)
Inference:   vLLM (14-24x faster)

Input:  ${GENERATION_OUTPUT_DIR}/<model_id>_v<version>/
Output: ${METRICS_OUTPUT_DIR}/<model_id>_v<version>/
EOF
}

list_models() {
    echo -e "${GREEN}Available models for baseline RAG metrics:${NC}"
    echo ""
    printf "%-20s %s\n" "Model ID" "Baseline RAG Generation Status"
    echo "------------------------------------------------------"
    for model_id in "${!MODELS[@]}"; do
        gen_file="${GENERATION_OUTPUT_DIR}/${model_id}_v1.0/${model_id}_baseline_rag_100-v1.0.json"
        if [ -f "$gen_file" ]; then
            status="${GREEN}✓ Ready${NC}"
        else
            status="${RED}✗ No baseline RAG generation${NC}"
        fi
        printf "%-20s " "$model_id"
        echo -e "$status"
    done
}

if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

case "$1" in
    --help|-h) show_help; exit 0 ;;
    --list|-l) list_models; exit 0 ;;
esac

MODEL_ID="$1"
VERSION="${2:-1.0}"

if [ -z "${MODELS[$MODEL_ID]+_}" ]; then
    echo -e "${RED}ERROR: Unknown model_id: ${MODEL_ID}${NC}"
    exit 1
fi

MODEL_NAME="${MODELS[$MODEL_ID]}"

# Check baseline RAG generation output exists
GENERATION_FILE="${GENERATION_OUTPUT_DIR}/${MODEL_ID}_v${VERSION}/${MODEL_ID}_baseline_rag_100-v${VERSION}.json"
if [ ! -f "$GENERATION_FILE" ]; then
    echo -e "${RED}ERROR: Baseline RAG generation not found: ${GENERATION_FILE}${NC}"
    echo "Run: ./scripts/generation_baseline_rag_vllm/launch_generation_baseline_rag_vllm.sh ${MODEL_ID}"
    exit 1
fi

if [ ! -d "$JUDGE_MODEL_PATH" ]; then
    echo -e "${RED}ERROR: Judge model not found: ${JUDGE_MODEL_PATH}${NC}"
    exit 1
fi

mkdir -p "${LOG_DIR}" "${METRICS_OUTPUT_DIR}"

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}EvidenceRL Baseline RAG Metrics Launcher (vLLM Version)${NC}"
echo -e "${CYAN}14-24x faster LLM judge inference${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "Mode:            ${YELLOW}BASELINE RAG (with retrieval)${NC}"
echo -e "Model ID:        ${BLUE}${MODEL_ID}${NC}"
echo -e "Generation File: ${GENERATION_FILE}"
echo -e "Judge Model:     ${JUDGE_MODEL_PATH}"
echo -e "Inference:       ${CYAN}vLLM${NC}"
echo -e "Output Dir:      ${METRICS_OUTPUT_DIR}/${MODEL_ID}_v${VERSION}/"
echo -e "${GREEN}============================================================${NC}"

CONDA_ENV="${CONDA_ENV:-evidencerl}"

JOB_ID=$(sbatch --parsable \
    --export=ALL,MODEL_ID="${MODEL_ID}",MODEL_NAME="${MODEL_NAME}",VERSION="${VERSION}",GENERATION_FILE="${GENERATION_FILE}",JUDGE_MODEL="${JUDGE_MODEL_PATH}",CONDA_ENV="${CONDA_ENV}" \
    "${SCRIPT_DIR}/metrics_baseline_rag_vllm_master.sbatch")

echo -e "Master job submitted: ${GREEN}${JOB_ID}${NC}"
echo ""
echo "Monitor: tail -f ${LOG_DIR}/metrics_baseline_rag_vllm_master_${JOB_ID}.out"
echo "Results: ${METRICS_OUTPUT_DIR}/${MODEL_ID}_v${VERSION}/${MODEL_ID}_baseline_rag_metrics-v${VERSION}.json"
