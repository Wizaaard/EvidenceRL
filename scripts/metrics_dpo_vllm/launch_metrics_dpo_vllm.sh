#!/usr/bin/env bash
# ============================================================================
# EvidenceRL DPO Metrics Launcher (vLLM Version)
# ============================================================================
# High-performance metrics computation using vLLM for the LLM judge.
# Runs on DPO-trained model generation output.
#
# Usage:
#   ./scripts/metrics_dpo_vllm/launch_metrics_dpo_vllm.sh <model_id> [version]
#   ./scripts/metrics_dpo_vllm/launch_metrics_dpo_vllm.sh --list
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Paths - DPO directories
GENERATION_OUTPUT_DIR="${PROJECT_ROOT}/../generation_dpo_output"
METRICS_OUTPUT_DIR="${PROJECT_ROOT}/../metrics_dpo_output"
# Configure these for your environment
MODEL_DIR="${MODEL_BASE_DIR:-//storage/ice-shared/bmed-sp-wang/Models}"
DPO_MODELS_DIR="${PROJECT_ROOT}/../dpo_trained_models"
LOG_DIR="${PROJECT_ROOT}/logs_evidencerl_dpo"

# Fixed judge model
JUDGE_MODEL_NAME="medgemma-27b-it"
JUDGE_MODEL_PATH="${MODEL_DIR}/${JUDGE_MODEL_NAME}"

# Available DPO models (model_id -> base_model:dpo_subdir)
declare -A DPO_MODELS=(
    ["gemma-3-4b-dpo"]="gemma-3-4b-it:gemma-3-4b_v1.0_master_training_set"
    ["gemma-3-12b-dpo"]="gemma-3-12b-it:gemma-3-12b_v1.0_master_training_set"
    ["gemma-3-27b-dpo"]="gemma-3-27b-it:gemma-3-27b-it_v1.0_master_training_set"
    ["Llama-3.1-8B-dpo"]="Llama-3.1-8B-Instruct:Llama-3.1-8B-Instruct_v1.0_master_training_set"
    ["gpt-oss-20b-dpo"]="gpt-oss-20b:gpt-oss-20b_v1.0_master_training_set"
    ["gpt-oss-120b-dpo"]="gpt-oss-120b:gpt-oss-120b_v1.0_master_training_set"
    # Faithfulness-DPO models
    ["Llama-3.2-3B-fdpo"]="Llama-3.2-3B-Instruct:Llama-3.2-3B_faithfulness_dpo"
    ["Llama-3.1-8B-fdpo"]="Llama-3.1-8B-Instruct:Llama-3.1-8B_faithfulness_dpo"
    ["gemma-3-4b-fdpo"]="gemma-3-4b-it:gemma-3-4b_faithfulness_dpo"
    ["gemma-3-12b-fdpo"]="gemma-3-12b-it:gemma-3-12b_faithfulness_dpo"
    ["gemma-3-27b-fdpo"]="gemma-3-27b-it:gemma-3-27b_faithfulness_dpo"
    ["gpt-oss-20b-fdpo"]="gpt-oss-20b:gpt-oss-20b_faithfulness_dpo"
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
EvidenceRL DPO Metrics Launcher (vLLM Version)

Computes metrics on DPO-trained model generation output using vLLM.

Usage:
    $(basename "$0") <model_id> [version]    Run metrics for DPO model
    $(basename "$0") --list                  List available DPO models
    $(basename "$0") --help                  Show this help

Judge Model: ${JUDGE_MODEL_NAME} (fixed, uses vLLM)
Inference:   vLLM (14-24x faster)

Input:  ${GENERATION_OUTPUT_DIR}/<model_id>_v<version>/
Output: ${METRICS_OUTPUT_DIR}/<model_id>_v<version>/
EOF
}

list_models() {
    echo -e "${GREEN}Available DPO models for metrics:${NC}"
    echo ""
    printf "%-25s %-25s %s\n" "Model ID" "Base Model" "Generation Status"
    echo "------------------------------------------------------------------------"
    for model_id in "${!DPO_MODELS[@]}"; do
        IFS=':' read -r base_model dpo_subdir <<< "${DPO_MODELS[$model_id]}"
        gen_file=$(ls "${GENERATION_OUTPUT_DIR}/${model_id}_v1.0/${model_id}_dpo_"*"-v1.0.json" 2>/dev/null | head -1)
        if [ -n "$gen_file" ] && [ -f "$gen_file" ]; then
            status="${GREEN}Ready${NC}"
        else
            status="${RED}No generation${NC}"
        fi
        printf "%-25s %-25s " "$model_id" "$base_model"
        echo -e "$status"
    done | sort
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

if [ -z "${DPO_MODELS[$MODEL_ID]+_}" ]; then
    echo -e "${RED}ERROR: Unknown DPO model_id: ${MODEL_ID}${NC}"
    echo "Run with --list to see available models"
    exit 1
fi

# Parse base model info
IFS=':' read -r BASE_MODEL_NAME DPO_SUBDIR <<< "${DPO_MODELS[$MODEL_ID]}"

# Check DPO generation output exists (auto-detect patient count in filename)
GENERATION_DIR="${GENERATION_OUTPUT_DIR}/${MODEL_ID}_v${VERSION}"
GENERATION_FILE="${GENERATION_FILE:-}"
if [ -z "${GENERATION_FILE}" ]; then
    # Auto-detect: look for ${MODEL_ID}_dpo_*-v${VERSION}.json
    GENERATION_FILE=$(ls "${GENERATION_DIR}/${MODEL_ID}_dpo_"*"-v${VERSION}.json" 2>/dev/null | head -1)
fi
if [ -z "${GENERATION_FILE}" ] || [ ! -f "${GENERATION_FILE}" ]; then
    echo -e "${RED}ERROR: DPO generation not found in: ${GENERATION_DIR}/${NC}"
    echo "Run: ./scripts/generation_dpo_vllm/launch_generation_dpo_vllm.sh ${MODEL_ID}"
    exit 1
fi

if [ ! -d "$JUDGE_MODEL_PATH" ]; then
    echo -e "${RED}ERROR: Judge model not found: ${JUDGE_MODEL_PATH}${NC}"
    exit 1
fi

mkdir -p "${LOG_DIR}" "${METRICS_OUTPUT_DIR}"

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}EvidenceRL DPO Metrics Launcher (vLLM Version)${NC}"
echo -e "${CYAN}Post-training evaluation with LLM judge${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "Mode:            ${YELLOW}DPO (post-training)${NC}"
echo -e "Model ID:        ${BLUE}${MODEL_ID}${NC}"
echo -e "Base Model:      ${BASE_MODEL_NAME}"
echo -e "Generation File: ${GENERATION_FILE}"
echo -e "Judge Model:     ${JUDGE_MODEL_PATH}"
echo -e "Inference:       ${CYAN}vLLM${NC}"
echo -e "Output Dir:      ${METRICS_OUTPUT_DIR}/${MODEL_ID}_v${VERSION}/"
echo -e "${GREEN}============================================================${NC}"

CONDA_ENV="${CONDA_ENV:-evidencerl}"
NUM_WORKERS="${NUM_WORKERS:-1}"

JOB_ID=$(sbatch --parsable \
    --export=ALL,MODEL_ID="${MODEL_ID}",MODEL_NAME="${BASE_MODEL_NAME}",VERSION="${VERSION}",GENERATION_FILE="${GENERATION_FILE}",JUDGE_MODEL="${JUDGE_MODEL_PATH}",CONDA_ENV="${CONDA_ENV}",NUM_WORKERS="${NUM_WORKERS}" \
    "${SCRIPT_DIR}/metrics_dpo_vllm_master.sbatch")

echo -e "Master job submitted: ${GREEN}${JOB_ID}${NC}"
echo ""
echo "Monitor: tail -f ${LOG_DIR}/metrics_dpo_vllm_master_${JOB_ID}.out"
echo "Results: ${METRICS_OUTPUT_DIR}/${MODEL_ID}_v${VERSION}/${MODEL_ID}_dpo_metrics-v${VERSION}.json"
