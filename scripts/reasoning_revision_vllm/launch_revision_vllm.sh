#!/usr/bin/env bash
# ============================================================================
# Reasoning Revision Pipeline Launcher (vLLM Version)
# ============================================================================
# High-performance reasoning revision using vLLM for 14-24x faster inference.
#
# This script runs the reasoning revision pipeline with vLLM instead of
# HuggingFace Transformers. Key benefits:
# - 14-24x faster inference via PagedAttention and continuous batching
# - Tensor parallelism for efficient multi-GPU usage
# - Reduced time limits (2h workers vs 4h)
#
# Usage:
#   ./launch_revision_vllm.sh <generation_json> <metrics_json> [options]
#   ./launch_revision_vllm.sh --help
#
# Examples:
#   # Basic usage (always distributed with vLLM)
#   ./launch_revision_vllm.sh gen.json metrics.json
#
#   # With custom options
#   ./launch_revision_vllm.sh gen.json metrics.json --num-workers 5
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Configure these for your environment
MODEL_DIR="${MODEL_BASE_DIR:-//storage/ice-shared/bmed-sp-wang/Models}"

# Default configuration
DEFAULT_MODEL="medgemma-27b-it"
DEFAULT_OUTPUT_BASE="${PROJECT_ROOT}/../revised_output_vllm"
DEFAULT_GROUNDING_MIN="-0.25"
DEFAULT_GROUNDING_MAX="0.25"
DEFAULT_NUM_WORKERS_STANDARD="5"
DEFAULT_NUM_WORKERS_LARGE="10"
DEFAULT_TENSOR_PARALLEL_SIZE="2"
DEFAULT_MAX_TOKENS="256"
DEFAULT_GPU_MEMORY_UTILIZATION="0.90"
DEFAULT_SEED="42"

# Extra-large models that require 8 GPUs (8x H200, 32 CPUs, 256GB RAM)
EXTRA_LARGE_MODELS=(
    "Llama-4-Maverick"
    "Llama-3.1-405B"
)

# Large models that require 4 GPUs
LARGE_MODELS=(
    "Llama3-Med42-70B"
    "gpt-oss-120b"
    "Llama-3.3-70B"
    "Llama-4-Scout"
)

is_extra_large_model() {
    local model_name="$1"
    for xl_model in "${EXTRA_LARGE_MODELS[@]}"; do
        if [[ "$model_name" == *"$xl_model"* ]]; then
            return 0
        fi
    done
    return 1
}

is_large_model() {
    local model_name="$1"
    for large_model in "${LARGE_MODELS[@]}"; do
        if [[ "$model_name" == *"$large_model"* ]]; then
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
NC='\033[0m' # No Color

show_help() {
    echo -e "${GREEN}Reasoning Revision Pipeline Launcher (vLLM Version)${NC}"
    echo -e "${CYAN}14-24x faster inference than HuggingFace Transformers${NC}"
    echo ""
    echo "Usage:"
    echo "  $0 <generation_json> <metrics_json> [options]"
    echo "  $0 --help"
    echo ""
    echo "Arguments:"
    echo "  generation_json    Path to generation JSON file"
    echo "  metrics_json       Path to metrics JSON file"
    echo ""
    echo "Options:"
    echo "  --num-workers <n>           Number of workers (default: 5 standard, 10 large)"
    echo "  --model <name>              Model for revision (default: ${DEFAULT_MODEL})"
    echo "  --output-dir <path>         Output directory (default: auto-generated)"
    echo "  --grounding-min <val>       Min grounding for ambiguous range (default: ${DEFAULT_GROUNDING_MIN})"
    echo "  --grounding-max <val>       Max grounding for ambiguous range (default: ${DEFAULT_GROUNDING_MAX})"
    echo "  --tensor-parallel-size <n>  GPUs per worker (default: ${DEFAULT_TENSOR_PARALLEL_SIZE})"
    echo "  --max-tokens <n>            Max tokens to generate (default: ${DEFAULT_MAX_TOKENS})"
    echo "  --gpu-memory-utilization <f> GPU memory fraction (default: ${DEFAULT_GPU_MEMORY_UTILIZATION})"
    echo "  --seed <n>                  Random seed (default: ${DEFAULT_SEED})"
    echo "  --help                      Show this help message"
    echo ""
    echo "vLLM Benefits:"
    echo "  - 14-24x faster than HuggingFace Transformers"
    echo "  - PagedAttention for efficient memory management"
    echo "  - Continuous batching for optimal throughput"
    echo "  - Tensor parallelism for multi-GPU inference"
    echo ""
    echo "Resource Allocation:"
    echo "  Master: CPU-only, 4 CPUs, 16GB RAM, 6h"
    echo "  Worker: 2x H200 GPU, 16 CPUs, 128GB RAM, 2h"
    echo ""
    echo "Examples:"
    echo "  # Basic usage"
    echo "  $0 gen.json metrics.json"
    echo ""
    echo "  # With custom workers and model"
    echo "  $0 gen.json metrics.json --num-workers 5 --model gemma-3-27b-it"
    echo ""
    echo "  # Adjust grounding range"
    echo "  $0 gen.json metrics.json --grounding-min -0.3 --grounding-max 0.3"
}

# Parse arguments
POSITIONAL_ARGS=()
MODEL="${DEFAULT_MODEL}"
OUTPUT_DIR=""
GROUNDING_MIN="${DEFAULT_GROUNDING_MIN}"
GROUNDING_MAX="${DEFAULT_GROUNDING_MAX}"
NUM_WORKERS=""
TENSOR_PARALLEL_SIZE=""  # Leave empty for auto-detection based on model size
MAX_TOKENS="${DEFAULT_MAX_TOKENS}"
GPU_MEMORY_UTILIZATION="${DEFAULT_GPU_MEMORY_UTILIZATION}"
SEED="${DEFAULT_SEED}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --grounding-min)
            GROUNDING_MIN="$2"
            shift 2
            ;;
        --grounding-max)
            GROUNDING_MAX="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -*)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Check required arguments
if [ ${#POSITIONAL_ARGS[@]} -lt 2 ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo "Usage: $0 <generation_json> <metrics_json> [options]"
    echo "Use --help for more information"
    exit 1
fi

GENERATION_JSON="${POSITIONAL_ARGS[0]}"
METRICS_JSON="${POSITIONAL_ARGS[1]}"

# Convert to absolute paths if relative
if [[ "${GENERATION_JSON}" != /* ]]; then
    GENERATION_JSON="$(pwd)/${GENERATION_JSON}"
fi
if [[ "${METRICS_JSON}" != /* ]]; then
    METRICS_JSON="$(pwd)/${METRICS_JSON}"
fi

# Validate input files exist
if [ ! -f "${GENERATION_JSON}" ]; then
    echo -e "${RED}Error: Generation JSON not found: ${GENERATION_JSON}${NC}"
    exit 1
fi

if [ ! -f "${METRICS_JSON}" ]; then
    echo -e "${RED}Error: Metrics JSON not found: ${METRICS_JSON}${NC}"
    exit 1
fi

# Determine model path
if [[ "${MODEL}" == /* ]]; then
    MODEL_PATH="${MODEL}"
else
    MODEL_PATH="${MODEL_DIR}/${MODEL}"
fi

# Check if model exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo -e "${YELLOW}Warning: Model directory not found: ${MODEL_PATH}${NC}"
    echo "Make sure the model is available before running."
fi

# Auto-detect resources based on model size
if is_extra_large_model "${MODEL_PATH}"; then
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
    WORKER_CPUS="${WORKER_CPUS:-32}"
    WORKER_MEM="${WORKER_MEM:-256G}"
    WORKER_TIME="${WORKER_TIME:-01:00:00}"
    NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_NUM_WORKERS_LARGE}}"
    echo -e "${YELLOW}[launcher] Detected extra-large model: using 8x GPU, 32 CPUs, 256GB RAM, ${NUM_WORKERS} workers${NC}"
elif is_large_model "${MODEL_PATH}"; then
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
    WORKER_CPUS="${WORKER_CPUS:-32}"
    WORKER_MEM="${WORKER_MEM:-256G}"
    WORKER_TIME="${WORKER_TIME:-02:00:00}"
    NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_NUM_WORKERS_LARGE}}"
    echo -e "${YELLOW}[launcher] Detected large model: using 4x GPU, 32 CPUs, 256GB RAM, ${NUM_WORKERS} workers${NC}"
else
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
    WORKER_CPUS="${WORKER_CPUS:-16}"
    WORKER_MEM="${WORKER_MEM:-128G}"
    WORKER_TIME="${WORKER_TIME:-02:00:00}"
    NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_NUM_WORKERS_STANDARD}}"
fi

# Generate output directory if not specified
if [ -z "${OUTPUT_DIR}" ]; then
    # Extract base name from generation JSON
    GEN_BASENAME=$(basename "${GENERATION_JSON}" .json)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${DEFAULT_OUTPUT_BASE}/${GEN_BASENAME}_vllm_${TIMESTAMP}"
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Create log directory
mkdir -p "${PROJECT_ROOT}/logs_evidencerl"

# Conda environment (allow override via environment variable)
CONDA_ENV="${CONDA_ENV:-evidencerl}"

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Reasoning Revision Pipeline (vLLM Version)${NC}"
echo -e "${CYAN}14-24x faster inference than HuggingFace Transformers${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "Generation JSON:        ${BLUE}${GENERATION_JSON}${NC}"
echo -e "Metrics JSON:           ${BLUE}${METRICS_JSON}${NC}"
echo -e "Output Directory:       ${BLUE}${OUTPUT_DIR}${NC}"
echo -e "Revision Model:         ${MODEL_PATH}"
echo -e "Number of Workers:      ${NUM_WORKERS}"
echo -e "Tensor Parallel Size:   ${TENSOR_PARALLEL_SIZE}"
echo -e "Max Tokens:             ${MAX_TOKENS}"
echo -e "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo -e "Grounding Range:        [${GROUNDING_MIN}, ${GROUNDING_MAX}]"
echo -e "Random Seed:            ${SEED}"
echo -e "Inference Engine:       ${CYAN}vLLM${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

echo -e "[launcher] ${BLUE}Submitting vLLM distributed Slurm jobs...${NC}"
echo "[launcher] Using conda environment: ${CONDA_ENV}"
echo ""

# Build export string with all configuration
EXPORT_VARS="ALL"
EXPORT_VARS+=",GENERATION_JSON=${GENERATION_JSON}"
EXPORT_VARS+=",METRICS_JSON=${METRICS_JSON}"
EXPORT_VARS+=",OUTPUT_DIR=${OUTPUT_DIR}"
EXPORT_VARS+=",MODEL_NAME=${MODEL_PATH}"
EXPORT_VARS+=",NUM_WORKERS=${NUM_WORKERS}"
EXPORT_VARS+=",TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
EXPORT_VARS+=",MAX_TOKENS=${MAX_TOKENS}"
EXPORT_VARS+=",GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
EXPORT_VARS+=",GROUNDING_MIN=${GROUNDING_MIN}"
EXPORT_VARS+=",GROUNDING_MAX=${GROUNDING_MAX}"
EXPORT_VARS+=",SEED=${SEED}"
EXPORT_VARS+=",WORKER_CPUS=${WORKER_CPUS}"
EXPORT_VARS+=",WORKER_MEM=${WORKER_MEM}"
EXPORT_VARS+=",WORKER_TIME=${WORKER_TIME}"
EXPORT_VARS+=",CONDA_ENV=${CONDA_ENV}"

JOB_ID=$(sbatch --parsable --export="${EXPORT_VARS}" "${SCRIPT_DIR}/revision_master_vllm.sbatch")

echo -e "[launcher] Master job submitted: ${GREEN}${JOB_ID}${NC}"
echo ""
echo "Configuration summary:"
echo "  Number of workers:      ${NUM_WORKERS}"
echo "  GPUs per worker:        ${TENSOR_PARALLEL_SIZE}x H200"
echo "  CPUs per worker:        ${WORKER_CPUS}"
echo "  Memory per worker:      ${WORKER_MEM}"
echo "  Time per worker:        ${WORKER_TIME}"
echo "  Inference engine:       vLLM"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f ${PROJECT_ROOT}/logs_evidencerl/revision_vllm_master_${JOB_ID}.out"
echo ""
echo "Output will be saved to:"
echo "  ${OUTPUT_DIR}/"
echo ""
echo "Expected output files:"
echo "  - revised_generation.json  (merged generation with revised reasoning)"
echo "  - revised_metrics.json     (merged metrics with updated scores)"
echo "  - plots/                   (visualization plots)"
echo "      - grounding_scatter.png     (original vs revised grounding)"
echo "      - grounding_distribution.png (score distributions)"
echo "      - revision_summary.png      (counts and statistics)"
echo "      - verdict_comparison.png    (verdict vs grounding)"
