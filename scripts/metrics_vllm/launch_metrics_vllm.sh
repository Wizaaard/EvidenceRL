#!/usr/bin/env bash
# ============================================================================
# EvidenceRL Metrics Launcher (vLLM Version)
# ============================================================================
# High-performance metrics computation using vLLM for the LLM judge.
#
# This script runs the metrics pipeline with vLLM instead of HuggingFace
# Transformers for the judge model. Key benefits:
# - 14-24x faster judge inference
# - Tensor parallelism for efficient multi-GPU usage
# - Reduced time limits
#
# Note: The NLI model (CrossEncoder) remains unchanged as it's already
# optimized and doesn't benefit from vLLM.
#
# Usage:
#   ./launch_metrics_vllm.sh <model_id> [version]      # Auto-detect generation JSON
#   ./launch_metrics_vllm.sh --generation-json <path>  # Explicit path
#   ./launch_metrics_vllm.sh --list                    # List available models
#   ./launch_metrics_vllm.sh --help                    # Show help
#
# Examples:
#   ./launch_metrics_vllm.sh gemma-3-27b               # Run for gemma-3-27b v1.0
#   ./launch_metrics_vllm.sh gemma-3-27b 1.1           # Run for gemma-3-27b v1.1
#   ./launch_metrics_vllm.sh --generation-json /path/to/generation.json
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Configure these for your environment
MODEL_DIR="${MODEL_BASE_DIR:-//storage/ice-shared/bmed-sp-wang/Models}"
GENERATION_OUTPUT_DIR="${PROJECT_ROOT}/../generation_output"

# Default configuration
DEFAULT_JUDGE_MODEL="medgemma-27b-it"
DEFAULT_OUTPUT_BASE="${PROJECT_ROOT}/../metrics_output"
DEFAULT_NUM_WORKERS_STANDARD="5"
DEFAULT_NUM_WORKERS_LARGE="10"
DEFAULT_TENSOR_PARALLEL_SIZE="2"
DEFAULT_GPU_MEMORY_UTILIZATION="0.90"

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
NC='\033[0m'

show_help() {
    echo -e "${GREEN}EvidenceRL Metrics Launcher (vLLM Version)${NC}"
    echo -e "${CYAN}14-24x faster LLM judge inference${NC}"
    echo ""
    echo "Usage:"
    echo "  $0 <model_id> [version]              Run metrics for a model (auto-detect generation)"
    echo "  $0 --generation-json <path> [opts]   Run with explicit generation JSON path"
    echo "  $0 --list                            List available models"
    echo "  $0 --help                            Show this help message"
    echo ""
    echo "Arguments:"
    echo "  model_id    Model identifier (e.g., gemma-3-27b, gpt-oss-120b)"
    echo "  version     Output version (default: 1.0)"
    echo ""
    echo "Options:"
    echo "  --generation-json <path>     Explicit path to generation JSON file"
    echo "  --num-workers <n>            Number of workers (default: 5 standard, 10 large)"
    echo "  --judge-model <name>         Judge model (default: ${DEFAULT_JUDGE_MODEL})"
    echo "  --output-dir <path>          Output directory (default: auto-generated)"
    echo "  --tensor-parallel-size <n>   GPUs per worker (default: ${DEFAULT_TENSOR_PARALLEL_SIZE})"
    echo "  --gpu-memory-utilization <f> GPU memory fraction (default: ${DEFAULT_GPU_MEMORY_UTILIZATION})"
    echo ""
    echo "vLLM Benefits:"
    echo "  - 14-24x faster judge inference than HuggingFace"
    echo "  - PagedAttention for efficient memory management"
    echo "  - Tensor parallelism for multi-GPU inference"
    echo ""
    echo "Resource Allocation:"
    echo "  Standard models: 2x H200 GPU, 16 CPUs, 128GB RAM, 5 workers"
    echo "  Large models:    4x H200 GPU, 32 CPUs, 256GB RAM, 10 workers"
    echo ""
    echo "Examples:"
    echo "  $0 gemma-3-27b                       # Run for gemma-3-27b v1.0"
    echo "  $0 gpt-oss-120b 1.1                  # Run for gpt-oss-120b v1.1"
    echo "  $0 --generation-json /path/to/gen.json --judge-model gpt-oss-120b"
}

list_models() {
    echo -e "${GREEN}Available Models for Metrics Computation (vLLM)${NC}"
    echo ""
    printf "%-20s %-35s %s\n" "Model ID" "Model Name" "Generation Status"
    echo "------------------------------------------------------------------------"
    for model_id in "${!MODELS[@]}"; do
        model_name="${MODELS[$model_id]}"
        gen_file="${GENERATION_OUTPUT_DIR}/${model_id}_v1.0/${model_id}_1000-v1.0.json"
        if [ -f "$gen_file" ]; then
            status="${GREEN}✓ Ready${NC}"
        else
            status="${RED}✗ No generation${NC}"
        fi
        printf "%-20s %-35s " "$model_id" "$model_name"
        echo -e "$status"
    done | sort
}

# Parse arguments
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# Check for --list or --help first
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

# Initialize variables
POSITIONAL_ARGS=()
JUDGE_MODEL="${DEFAULT_JUDGE_MODEL}"
OUTPUT_DIR=""
NUM_WORKERS=""
TENSOR_PARALLEL_SIZE=""  # Leave empty for auto-detection based on model size
GPU_MEMORY_UTILIZATION="${DEFAULT_GPU_MEMORY_UTILIZATION}"
GENERATION_JSON=""
MODEL_ID=""
VERSION="1.0"

# Parse all arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --list|-l)
            list_models
            exit 0
            ;;
        --generation-json)
            GENERATION_JSON="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --judge-model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
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

# Determine generation JSON path
if [ -n "${GENERATION_JSON}" ]; then
    # Explicit path provided via --generation-json
    if [[ "${GENERATION_JSON}" != /* ]]; then
        GENERATION_JSON="$(pwd)/${GENERATION_JSON}"
    fi
    if [ ! -f "${GENERATION_JSON}" ]; then
        echo -e "${RED}Error: Generation JSON not found: ${GENERATION_JSON}${NC}"
        exit 1
    fi
elif [ ${#POSITIONAL_ARGS[@]} -ge 1 ]; then
    # Model ID provided as positional argument
    MODEL_ID="${POSITIONAL_ARGS[0]}"
    VERSION="${POSITIONAL_ARGS[1]:-1.0}"

    # Validate model
    if [ -z "${MODELS[$MODEL_ID]+_}" ]; then
        echo -e "${RED}Error: Unknown model_id: ${MODEL_ID}${NC}"
        echo ""
        echo "Available models:"
        for key in "${!MODELS[@]}"; do
            echo "  $key"
        done | sort
        echo ""
        echo "Use --list to see generation status"
        exit 1
    fi

    # Construct generation JSON path
    GENERATION_JSON="${GENERATION_OUTPUT_DIR}/${MODEL_ID}_v${VERSION}/${MODEL_ID}_1000-v${VERSION}.json"

    if [ ! -f "${GENERATION_JSON}" ]; then
        echo -e "${RED}Error: Generation output not found: ${GENERATION_JSON}${NC}"
        echo ""
        echo "Run generation first:"
        echo "  ./scripts/generation_vllm/launch_generation_vllm.sh ${MODEL_ID} ${VERSION}"
        exit 1
    fi
else
    echo -e "${RED}Error: Missing required argument${NC}"
    echo ""
    echo "Usage:"
    echo "  $0 <model_id> [version]              # Auto-detect generation JSON"
    echo "  $0 --generation-json <path> [opts]   # Explicit path"
    echo ""
    echo "Use --help for more information"
    exit 1
fi

# Determine judge model path
if [[ "${JUDGE_MODEL}" == /* ]]; then
    JUDGE_MODEL_PATH="${JUDGE_MODEL}"
else
    JUDGE_MODEL_PATH="${MODEL_DIR}/${JUDGE_MODEL}"
fi

# Check if judge model exists
if [ ! -d "${JUDGE_MODEL_PATH}" ]; then
    echo -e "${YELLOW}Warning: Judge model directory not found: ${JUDGE_MODEL_PATH}${NC}"
    echo "Make sure the model is available before running."
fi

# Auto-detect resources based on judge model size
if is_extra_large_model "${JUDGE_MODEL_PATH}"; then
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
    WORKER_CPUS="${WORKER_CPUS:-32}"
    WORKER_MEM="${WORKER_MEM:-256G}"
    WORKER_TIME="${WORKER_TIME:-01:00:00}"
    NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_NUM_WORKERS_LARGE}}"
    echo -e "${YELLOW}[launcher] Detected extra-large judge model: using 8x GPU, 32 CPUs, 256GB RAM, ${NUM_WORKERS} workers${NC}"
elif is_large_model "${JUDGE_MODEL_PATH}"; then
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
    WORKER_CPUS="${WORKER_CPUS:-32}"
    WORKER_MEM="${WORKER_MEM:-256G}"
    WORKER_TIME="${WORKER_TIME:-02:00:00}"
    NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_NUM_WORKERS_LARGE}}"
    echo -e "${YELLOW}[launcher] Detected large judge model: using 4x GPU, 32 CPUs, 256GB RAM, ${NUM_WORKERS} workers${NC}"
else
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
    WORKER_CPUS="${WORKER_CPUS:-16}"
    WORKER_MEM="${WORKER_MEM:-128G}"
    WORKER_TIME="${WORKER_TIME:-02:00:00}"
    NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_NUM_WORKERS_STANDARD}}"
fi

# Extract MODEL_ID and VERSION from generation JSON path if not set
if [ -z "${MODEL_ID}" ]; then
    # Try to extract from path like: .../model_vX.Y/model_1000-vX.Y.json
    GEN_BASENAME=$(basename "${GENERATION_JSON}" .json)
    GEN_DIRNAME=$(basename "$(dirname "${GENERATION_JSON}")")

    # Extract model_id from directory name (e.g., "gemma-3-27b_v1.0" -> "gemma-3-27b")
    if [[ "${GEN_DIRNAME}" =~ ^(.+)_v([0-9.]+)$ ]]; then
        MODEL_ID="${BASH_REMATCH[1]}"
        VERSION="${BASH_REMATCH[2]}"
    else
        # Fallback: use basename without version suffix
        MODEL_ID="${GEN_BASENAME%_*}"
        VERSION="1.0"
    fi
fi

# Generate output directory if not specified (consistent with HuggingFace version)
if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${DEFAULT_OUTPUT_BASE}/${MODEL_ID}_v${VERSION}"
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Create log directory
mkdir -p "${PROJECT_ROOT}/logs_evidencerl"

# Conda environment
CONDA_ENV="${CONDA_ENV:-evidencerl}"

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}EvidenceRL Metrics Pipeline (vLLM Version)${NC}"
echo -e "${CYAN}14-24x faster LLM judge inference${NC}"
echo -e "${GREEN}============================================================${NC}"
if [ -n "${MODEL_ID}" ]; then
    echo -e "Model ID:               ${BLUE}${MODEL_ID}${NC}"
    echo -e "Version:                ${VERSION}"
fi
echo -e "Generation JSON:        ${BLUE}${GENERATION_JSON}${NC}"
echo -e "Output Directory:       ${BLUE}${OUTPUT_DIR}${NC}"
echo -e "Judge Model:            ${JUDGE_MODEL_PATH}"
echo -e "Number of Workers:      ${NUM_WORKERS}"
echo -e "Tensor Parallel Size:   ${TENSOR_PARALLEL_SIZE}"
echo -e "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo -e "Inference Engine:       ${CYAN}vLLM${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

echo -e "[launcher] ${BLUE}Submitting vLLM metrics Slurm jobs...${NC}"
echo "[launcher] Using conda environment: ${CONDA_ENV}"
echo ""

# Build export string
EXPORT_VARS="ALL"
EXPORT_VARS+=",GENERATION_JSON=${GENERATION_JSON}"
EXPORT_VARS+=",OUTPUT_DIR=${OUTPUT_DIR}"
EXPORT_VARS+=",JUDGE_MODEL=${JUDGE_MODEL_PATH}"
EXPORT_VARS+=",NUM_WORKERS=${NUM_WORKERS}"
EXPORT_VARS+=",TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
EXPORT_VARS+=",GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
EXPORT_VARS+=",WORKER_CPUS=${WORKER_CPUS}"
EXPORT_VARS+=",WORKER_MEM=${WORKER_MEM}"
EXPORT_VARS+=",WORKER_TIME=${WORKER_TIME}"
EXPORT_VARS+=",CONDA_ENV=${CONDA_ENV}"
[ -n "${MODEL_ID}" ] && EXPORT_VARS+=",MODEL_ID=${MODEL_ID}"
[ -n "${VERSION}" ] && EXPORT_VARS+=",VERSION=${VERSION}"

JOB_ID=$(sbatch --parsable --export="${EXPORT_VARS}" "${SCRIPT_DIR}/metrics_master_vllm.sbatch")

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
echo "  tail -f ${PROJECT_ROOT}/logs_evidencerl/metrics_vllm_master_${JOB_ID}.out"
echo ""
echo "Output will be saved to:"
echo "  ${OUTPUT_DIR}/"
if [ -n "${MODEL_ID}" ]; then
    echo ""
    echo "Expected output file:"
    echo "  ${OUTPUT_DIR}/${MODEL_ID}_metrics-v${VERSION}.json"
fi
