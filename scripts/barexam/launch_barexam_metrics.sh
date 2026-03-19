#!/usr/bin/env bash
# ============================================================================
# BarExam Metrics Launcher
# ============================================================================
# Launch NLI grounding metrics for all BarExam v1.1 results.
#
# Usage:
#   ./launch_barexam_metrics.sh                    # All models
#   ./launch_barexam_metrics.sh --model gemma-3-4b # Single model
#   ./launch_barexam_metrics.sh --no-grounding     # Accuracy only (no GPU)
#   ./launch_barexam_metrics.sh --version v1.0     # Specific version
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

# Models (same as generation)
MODELS=(
    "gemma-3-4b"
    "gemma-3-12b"
    "gemma-3-27b"
    "Llama-3.2-3B"
    "Llama-3.1-8B"
    "Llama-3.3-70B"
    "gpt-oss-20b"
    "gpt-oss-120b"
)

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
SPLIT="test"
VERSION="v1.1"
MODE_TAG="rag"
SINGLE_MODEL=""
NO_GROUNDING="false"
NLI_MODEL=""
OUTPUT_SUFFIX=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split) SPLIT="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        --mode-tag) MODE_TAG="$2"; shift 2 ;;
        --model) SINGLE_MODEL="$2"; shift 2 ;;
        --nli-model) NLI_MODEL="$2"; shift 2 ;;
        --output-suffix) OUTPUT_SUFFIX="$2"; shift 2 ;;
        --no-grounding) NO_GROUNDING="true"; shift ;;
        --list)
            echo -e "${GREEN}Available models:${NC}"
            for m in "${MODELS[@]}"; do echo "  $m"; done
            exit 0 ;;
        --help|-h)
            echo "Usage: $0 [--split test] [--version v1.1] [--model <id>] [--no-grounding]"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "${DATA_ROOT}/logs_barexam"

echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}BarExam Metrics Launcher${NC}"
echo -e "  Split:   ${SPLIT}"
echo -e "  Version: ${VERSION}"
echo -e "  Mode:    ${MODE_TAG}"
echo -e "${GREEN}════════════════════════════════════════${NC}"

LAUNCHED=0

for model_id in "${MODELS[@]}"; do
    if [ -n "${SINGLE_MODEL}" ] && [ "${model_id}" != "${SINGLE_MODEL}" ]; then
        continue
    fi

    RESULT_DIR="${DATA_ROOT}/barexam_output/${SPLIT}/${model_id}_${MODE_TAG}-${VERSION}"
    RESULT_FILE="${RESULT_DIR}/${model_id}_barexam_${MODE_TAG}-${VERSION}.json"

    if [ ! -f "${RESULT_FILE}" ]; then
        echo -e "  SKIP: ${model_id} (no results at ${RESULT_FILE})"
        continue
    fi

    if [ -n "${OUTPUT_SUFFIX}" ]; then
        OUTPUT_DIR="${RESULT_DIR}/metrics${OUTPUT_SUFFIX}"
    elif [ -n "${NLI_MODEL}" ]; then
        NLI_SUFFIX=$(echo "${NLI_MODEL}" | sed 's|.*/||')  # e.g. nli-deberta-v3-large
        OUTPUT_DIR="${RESULT_DIR}/metrics_${NLI_SUFFIX}"
    else
        OUTPUT_DIR="${RESULT_DIR}/metrics"
    fi
    job_name="barexam_metrics_${model_id}"

    EXPORT_VARS="RESULTS_JSON=${RESULT_FILE}"
    EXPORT_VARS="${EXPORT_VARS},OUTPUT_DIR=${OUTPUT_DIR}"
    EXPORT_VARS="${EXPORT_VARS},PROJECT_ROOT=${PROJECT_ROOT}"
    EXPORT_VARS="${EXPORT_VARS},CONDA_ENV=ragcon"

    if [ -n "${NLI_MODEL}" ]; then
        EXPORT_VARS="${EXPORT_VARS},NLI_MODEL=${NLI_MODEL}"
    fi

    if [ "${NO_GROUNDING}" = "true" ]; then
        EXPORT_VARS="${EXPORT_VARS},NO_GROUNDING=true"
    fi

    echo -e "${BLUE}Launching: ${model_id}${NC}"

    sbatch \
        --job-name="${job_name}" \
        --gres="gpu:1" \
        --export="${EXPORT_VARS}" \
        "${SCRIPT_DIR}/barexam_metrics.sbatch"

    LAUNCHED=$((LAUNCHED + 1))
done

echo ""
echo -e "${GREEN}Launched ${LAUNCHED} metrics jobs.${NC}"
echo "Monitor: squeue -u \$USER -n barexam_metrics"
