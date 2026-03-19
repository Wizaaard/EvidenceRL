#!/usr/bin/env bash
# ============================================================================
# EvidenceRL Metrics Computation Launcher
# ============================================================================
# Main entry point for running distributed metrics computation on HPC cluster.
#
# Usage:
#   ./scripts/metrics/launch_metrics.sh <model_id> [version]
#   ./scripts/metrics/launch_metrics.sh --list
#   ./scripts/metrics/launch_metrics.sh --help
#
# Examples:
#   ./scripts/metrics/launch_metrics.sh gemma-3-4b          # v1.0 (default)
#   ./scripts/metrics/launch_metrics.sh gemma-3-4b 1.1      # Custom version
#   ./scripts/metrics/launch_metrics.sh --list              # List available
# ============================================================================

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Paths
GENERATION_OUTPUT_DIR="${PROJECT_ROOT}/../generation_output"
METRICS_OUTPUT_DIR="${PROJECT_ROOT}/../metrics_output"
# Configure these for your environment
MODEL_DIR="${MODEL_BASE_DIR:-//storage/ice-shared/bmed-sp-wang/Models}"
LOG_DIR="${PROJECT_ROOT}/logs_evidencerl"

# Fixed judge model for all metrics evaluation (medgemma-27b for medical domain expertise)
JUDGE_MODEL_NAME="medgemma-27b-it"
JUDGE_MODEL_PATH="${MODEL_DIR}/${JUDGE_MODEL_NAME}"

# Available models (must match generation)
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

# Help message
show_help() {
    cat << EOF
EvidenceRL Metrics Computation Launcher

Usage:
    $(basename "$0") <model_id> [version]    Run metrics for a model
    $(basename "$0") --list                  List available models
    $(basename "$0") --help                  Show this help

Arguments:
    model_id    Model identifier (e.g., gemma-3-4b, Llama-3.1-8B)
    version     Output version (default: 1.0)

Judge Model:
    Fixed: ${JUDGE_MODEL_NAME} (medical domain expertise)
    Path:  ${JUDGE_MODEL_PATH}

Environment Variables:
    CONDA_ENV       Conda environment name (default: evidencerl)
    BATCH_SIZE      Batch size for judge inference (default: 48 for 27B model)
    REWARD_WEIGHT   Grounding weight in combined reward (default: 0.5)

Available Models:
EOF
    for model_id in "${!MODELS[@]}"; do
        echo "    ${model_id} -> ${MODELS[$model_id]}"
    done
    echo ""
    echo "Output: ${METRICS_OUTPUT_DIR}/<model_id>_v<version>/"
}

# List models
list_models() {
    echo "Available models for metrics computation:"
    echo ""
    printf "%-20s %-30s %s\n" "Model ID" "Model Name" "Generation Status"
    echo "----------------------------------------------------------------------"
    for model_id in "${!MODELS[@]}"; do
        model_name="${MODELS[$model_id]}"
        gen_file="${GENERATION_OUTPUT_DIR}/${model_id}_v1.0/${model_id}_1000-v1.0.json"
        if [ -f "$gen_file" ]; then
            status="✓ Ready"
        else
            status="✗ No generation output"
        fi
        printf "%-20s %-30s %s\n" "$model_id" "$model_name" "$status"
    done
}

# Parse arguments
if [ $# -eq 0 ]; then
    show_help
    exit 1
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
if [ -z "${MODELS[$MODEL_ID]+_}" ]; then
    echo "ERROR: Unknown model_id: ${MODEL_ID}"
    echo "Available models: ${!MODELS[*]}"
    exit 1
fi

MODEL_NAME="${MODELS[$MODEL_ID]}"

# Check generation output exists
GENERATION_FILE="${GENERATION_OUTPUT_DIR}/${MODEL_ID}_v${VERSION}/${MODEL_ID}_1000-v${VERSION}.json"
if [ ! -f "$GENERATION_FILE" ]; then
    echo "ERROR: Generation output not found: ${GENERATION_FILE}"
    echo "Run generation first: ./scripts/launch_generation.sh ${MODEL_ID} ${VERSION}"
    exit 1
fi

# Check judge model exists (fixed: medgemma-27b-it)
if [ ! -d "$JUDGE_MODEL_PATH" ]; then
    echo "ERROR: Judge model not found: ${JUDGE_MODEL_PATH}"
    exit 1
fi

# Create directories
mkdir -p "${LOG_DIR}" "${METRICS_OUTPUT_DIR}"

echo "============================================================"
echo "EvidenceRL Metrics Computation"
echo "============================================================"
echo "Model ID:        ${MODEL_ID}"
echo "Model Name:      ${MODEL_NAME}"
echo "Version:         ${VERSION}"
echo "Generation File: ${GENERATION_FILE}"
echo "Judge Model:     ${JUDGE_MODEL_PATH} (fixed)"
echo "Output Dir:      ${METRICS_OUTPUT_DIR}/${MODEL_ID}_v${VERSION}/"
echo "============================================================"

# Submit master job
# Conda environment (allow override via environment variable)
CONDA_ENV="${CONDA_ENV:-evidencerl}"

echo ""
echo "Submitting metrics master job..."
echo "Using conda environment: ${CONDA_ENV}"

JOB_ID=$(sbatch --parsable \
    --export=ALL,MODEL_ID="${MODEL_ID}",MODEL_NAME="${MODEL_NAME}",VERSION="${VERSION}",GENERATION_FILE="${GENERATION_FILE}",JUDGE_MODEL="${JUDGE_MODEL_PATH}",CONDA_ENV="${CONDA_ENV}"${BATCH_SIZE:+,BATCH_SIZE="${BATCH_SIZE}"}${REWARD_WEIGHT:+,REWARD_WEIGHT="${REWARD_WEIGHT}"} \
    "${SCRIPT_DIR}/metrics_master.sbatch")

echo "Master job submitted: ${JOB_ID}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/metrics_master_${JOB_ID}.out"
echo ""
echo "Results will be at:"
echo "  ${METRICS_OUTPUT_DIR}/${MODEL_ID}_v${VERSION}/${MODEL_ID}_metrics-v${VERSION}.json"
