#!/usr/bin/env bash
# ============================================================================
# BarExam QA Benchmark Launcher
# ============================================================================
# Launch generation + metrics for BarExam QA across multiple models.
#
# Usage:
#   ./launch_barexam.sh                    # Launch all models (test split)
#   ./launch_barexam.sh --split validation # Launch on validation split
#   ./launch_barexam.sh --model gemma-3-4b # Launch single model
#   ./launch_barexam.sh --no-rag           # Closed-book baseline
#   ./launch_barexam.sh --list             # List models
#   ./launch_barexam.sh --download         # Download data first
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
# Configure these for your environment
MODEL_DIR="${MODEL_BASE_DIR:-//storage/ice-shared/bmed-sp-wang/Models}"

# Default data path
BAREXAM_DATA_DEFAULT="${DATA_ROOT}/data/barexam"

# ── Model configurations ──────────────────────────────────────────────
MODELS=(
    "gemma-3-4b|gemma-3-4b-it|2|H100+H200"
    "gemma-3-12b|gemma-3-12b-it|2|H100+H200"
    "gemma-3-27b|gemma-3-27b-it|2|H200"
    "Llama-3.2-3B|Llama-3.2-3B-Instruct|2|H100+H200"
    "Llama-3.1-8B|Llama-3.1-8B-Instruct|2|H100+H200"
    "Llama-3.3-70B|Llama-3.3-70B-Instruct|4|H200"
    "gpt-oss-20b|gpt-oss-20b|2|H100+H200"
    "gpt-oss-120b|gpt-oss-120b|4|H200"
)

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ── Parse arguments ───────────────────────────────────────────────────
SPLIT="test"
VERSION="v1.0"
NO_RAG="false"
SINGLE_MODEL=""
DO_DOWNLOAD="false"
BAREXAM_DATA="${BAREXAM_DATA_DEFAULT}"
MAX_CASES=""
FAISS_INDEX_DIR=""
MAX_PASSAGES=""
INCLUDE_GOLD="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split) SPLIT="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        --no-rag) NO_RAG="true"; shift ;;
        --model) SINGLE_MODEL="$2"; shift 2 ;;
        --data-path) BAREXAM_DATA="$2"; shift 2 ;;
        --max-cases) MAX_CASES="$2"; shift 2 ;;
        --faiss-index-dir) FAISS_INDEX_DIR="$2"; shift 2 ;;
        --max-passages) MAX_PASSAGES="$2"; shift 2 ;;
        --include-gold) INCLUDE_GOLD="true"; shift ;;
        --download) DO_DOWNLOAD="true"; shift ;;
        --list)
            echo -e "${GREEN}Available models:${NC}"
            for entry in "${MODELS[@]}"; do
                IFS='|' read -r mid mname ngpus gtype <<< "${entry}"
                echo "  ${mid} (${ngpus} GPU, ${gtype})"
            done
            exit 0 ;;
        --help|-h)
            echo "Usage: $0 [--split test|validation|train|all] [--model <id>] [--no-rag] [--faiss-index-dir <dir>] [--max-passages <k>] [--include-gold] [--download]"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Download data if requested ────────────────────────────────────────
if [ "${DO_DOWNLOAD}" = "true" ]; then
    echo -e "${BLUE}Downloading BarExam QA data...${NC}"
    eval "$(conda shell.bash hook)"
    conda activate ragcon
    python -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}/src')
from evidence_rl.barexam_data import download_barexam
download_barexam('${BAREXAM_DATA}')
"
    echo -e "${GREEN}Download complete.${NC}"
fi

# Create logs directory
mkdir -p "${DATA_ROOT}/logs_barexam"

# ── Launch jobs ───────────────────────────────────────────────────────
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}BarExam QA Benchmark Launcher${NC}"
echo -e "  Split:   ${SPLIT}"
if [ -n "${FAISS_INDEX_DIR}" ]; then
    mode_str="RAG (FAISS retrieval, k=${MAX_PASSAGES:-3})"
else
    mode_str=$([ "${NO_RAG}" = "true" ] && echo "no-RAG (closed-book)" || echo "RAG (with gold passage)")
fi
echo -e "  Mode:    ${mode_str}"
echo -e "  Version: ${VERSION}"
echo -e "  Data:    ${BAREXAM_DATA}"
echo -e "${GREEN}════════════════════════════════════════${NC}"

LAUNCHED=0

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_id model_name num_gpus gpu_constraint <<< "${entry}"

    if [ -n "${SINGLE_MODEL}" ] && [ "${model_id}" != "${SINGLE_MODEL}" ]; then
        continue
    fi

    full_model_path="${MODEL_DIR}/${model_name}"

    local_constraint="${gpu_constraint//+/|}"

    if [ "${num_gpus}" -ge 4 ]; then
        cpus=32
        mem="256G"
    else
        cpus=16
        mem="128G"
    fi

    MODE_TAG=$([ "${NO_RAG}" = "true" ] && echo "norag" || echo "rag")
    job_name="barexam_${SPLIT}_${model_id}_${MODE_TAG}"

    EXPORT_VARS="MODEL_NAME=${full_model_path}"
    EXPORT_VARS="${EXPORT_VARS},MODEL_ID=${model_id}"
    EXPORT_VARS="${EXPORT_VARS},SPLIT=${SPLIT}"
    EXPORT_VARS="${EXPORT_VARS},VERSION=${VERSION}"
    EXPORT_VARS="${EXPORT_VARS},NUM_GPUS=${num_gpus}"
    EXPORT_VARS="${EXPORT_VARS},BAREXAM_DATA=${BAREXAM_DATA}"
    EXPORT_VARS="${EXPORT_VARS},NO_RAG=${NO_RAG}"
    EXPORT_VARS="${EXPORT_VARS},CONDA_ENV=ragcon"
    EXPORT_VARS="${EXPORT_VARS},PROJECT_ROOT=${PROJECT_ROOT}"
    EXPORT_VARS="${EXPORT_VARS},DATA_ROOT=${DATA_ROOT}"

    # LLM extractor: models <= 4B use gemma-3-12b, others self-extract (no extractor flag)
    case "${model_id}" in
        gemma-3-4b|Llama-3.2-3B)
            EXPORT_VARS="${EXPORT_VARS},EXTRACTOR_MODEL=${MODEL_DIR}/gemma-3-12b-it"
            ;;
    esac

    if [ -n "${MAX_CASES}" ]; then
        EXPORT_VARS="${EXPORT_VARS},MAX_CASES=${MAX_CASES}"
    fi

    if [ -n "${FAISS_INDEX_DIR}" ]; then
        EXPORT_VARS="${EXPORT_VARS},FAISS_INDEX_DIR=${FAISS_INDEX_DIR}"
    fi

    if [ -n "${MAX_PASSAGES}" ]; then
        EXPORT_VARS="${EXPORT_VARS},MAX_PASSAGES=${MAX_PASSAGES}"
    fi

    if [ "${INCLUDE_GOLD}" = "true" ]; then
        EXPORT_VARS="${EXPORT_VARS},INCLUDE_GOLD=true"
    fi

    echo -e "${BLUE}Launching: ${model_id}${NC} (${num_gpus}x GPU, constraint=${local_constraint})"

    sbatch \
        --job-name="${job_name}" \
        --gres="gpu:${num_gpus}" \
        --constraint="${local_constraint}" \
        --cpus-per-task="${cpus}" \
        --mem="${mem}" \
        --export="${EXPORT_VARS}" \
        "${SCRIPT_DIR}/barexam_generation.sbatch"

    LAUNCHED=$((LAUNCHED + 1))
done

echo ""
echo -e "${GREEN}Launched ${LAUNCHED} generation jobs.${NC}"
echo "Monitor: squeue -u \$USER -n barexam_${SPLIT}"
echo "After completion, run compute_barexam_metrics.py on each output."
