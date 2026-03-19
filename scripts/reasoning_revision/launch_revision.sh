#!/usr/bin/env bash
# ============================================================================
# Reasoning Revision Pipeline Launcher
# ============================================================================
# Main entry point for running the reasoning revision pipeline.
#
# This script:
# 1. Revises reasoning for diagnoses with ambiguous grounding scores
#    - Case A: Correct diagnoses -> well-grounded reasoning
#    - Case B: Incorrect diagnoses -> contradictory reasoning
# 2. Recalculates NLI grounding scores for revised reasoning
#
# Modes:
#   Local:       Runs revision directly (for small datasets/debugging)
#   Distributed: Submits Slurm jobs with 10 workers (for large datasets)
#
# Usage:
#   ./launch_revision.sh <generation_json> <metrics_json> [options]
#   ./launch_revision.sh --help
#
# Examples:
#   # Local mode (direct execution)
#   ./launch_revision.sh gen.json metrics.json
#
#   # Distributed mode (Slurm jobs with 10 workers)
#   ./launch_revision.sh gen.json metrics.json --distributed
#
#   # With custom options
#   ./launch_revision.sh gen.json metrics.json --distributed --model gemma-3-27b-it
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Configure these for your environment
MODEL_DIR="${MODEL_BASE_DIR:-//storage/ice-shared/bmed-sp-wang/Models}"

# Default configuration
DEFAULT_MODEL="medgemma-27b-it"
DEFAULT_OUTPUT_BASE="${PROJECT_ROOT}/../revised_output"
DEFAULT_GROUNDING_MIN="-0.25"
DEFAULT_GROUNDING_MAX="0.25"
DEFAULT_BATCH_SIZE="64"
DEFAULT_SEED="42"
DEFAULT_NUM_WORKERS="10"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_help() {
    echo -e "${GREEN}Reasoning Revision Pipeline Launcher${NC}"
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
    echo "  --distributed          Run as distributed Slurm jobs (10 workers)"
    echo "  --num-workers <n>      Number of workers for distributed mode (default: ${DEFAULT_NUM_WORKERS})"
    echo "  --model <name>         Model for revision (default: ${DEFAULT_MODEL})"
    echo "  --output-dir <path>    Output directory (default: auto-generated)"
    echo "  --grounding-min <val>  Min grounding for ambiguous range (default: ${DEFAULT_GROUNDING_MIN})"
    echo "  --grounding-max <val>  Max grounding for ambiguous range (default: ${DEFAULT_GROUNDING_MAX})"
    echo "  --batch-size <n>       Batch size for generation (default: ${DEFAULT_BATCH_SIZE})"
    echo "  --seed <n>             Random seed (default: ${DEFAULT_SEED})"
    echo "  --skip-nli             Skip NLI grounding recalculation (local mode only)"
    echo "  --help                 Show this help message"
    echo ""
    echo "Modes:"
    echo "  Local (default):  Runs revision directly on current machine"
    echo "  Distributed:      Submits Slurm jobs with master/worker pattern"
    echo "                    - Master job: monitors workers, merges results"
    echo "                    - Workers: 2x H200 GPU, 16 CPUs, 128GB RAM, 4h each"
    echo ""
    echo "Examples:"
    echo "  # Local mode - basic usage"
    echo "  $0 gen.json metrics.json"
    echo ""
    echo "  # Distributed mode - 10 workers"
    echo "  $0 gen.json metrics.json --distributed"
    echo ""
    echo "  # Distributed mode with custom workers"
    echo "  $0 gen.json metrics.json --distributed --num-workers 20"
    echo ""
    echo "  # With custom model and output directory"
    echo "  $0 gen.json metrics.json --model gemma-3-27b-it --output-dir ./revised"
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
BATCH_SIZE="${DEFAULT_BATCH_SIZE}"
SEED="${DEFAULT_SEED}"
SKIP_NLI=false
DISTRIBUTED=false
NUM_WORKERS="${DEFAULT_NUM_WORKERS}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --distributed)
            DISTRIBUTED=true
            shift
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
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --skip-nli)
            SKIP_NLI=true
            shift
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

# Generate output directory if not specified
if [ -z "${OUTPUT_DIR}" ]; then
    # Extract base name from generation JSON
    GEN_BASENAME=$(basename "${GENERATION_JSON}" .json)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${DEFAULT_OUTPUT_BASE}/${GEN_BASENAME}_revised_${TIMESTAMP}"
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Create log directory
mkdir -p "${PROJECT_ROOT}/logs_evidencerl"

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Reasoning Revision Pipeline${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "Mode:             ${BLUE}$([ "$DISTRIBUTED" = true ] && echo "Distributed (${NUM_WORKERS} workers)" || echo "Local")${NC}"
echo -e "Generation JSON:  ${BLUE}${GENERATION_JSON}${NC}"
echo -e "Metrics JSON:     ${BLUE}${METRICS_JSON}${NC}"
echo -e "Output Directory: ${BLUE}${OUTPUT_DIR}${NC}"
echo -e "Revision Model:   ${MODEL_PATH}"
echo -e "Grounding Range:  [${GROUNDING_MIN}, ${GROUNDING_MAX}]"
echo -e "Batch Size:       ${BATCH_SIZE}"
echo -e "Random Seed:      ${SEED}"
echo -e "${GREEN}============================================================${NC}"
echo ""

# Conda environment (allow override via environment variable)
CONDA_ENV="${CONDA_ENV:-evidencerl}"

if [ "$DISTRIBUTED" = true ]; then
    # =========================================================================
    # DISTRIBUTED MODE: Submit Slurm master job
    # =========================================================================
    echo -e "[launcher] ${BLUE}Submitting distributed Slurm jobs...${NC}"
    echo "[launcher] Using conda environment: ${CONDA_ENV}"
    echo ""

    # Build export string with all configuration
    EXPORT_VARS="ALL"
    EXPORT_VARS+=",GENERATION_JSON=${GENERATION_JSON}"
    EXPORT_VARS+=",METRICS_JSON=${METRICS_JSON}"
    EXPORT_VARS+=",OUTPUT_DIR=${OUTPUT_DIR}"
    EXPORT_VARS+=",MODEL_NAME=${MODEL_PATH}"
    EXPORT_VARS+=",BATCH_SIZE=${BATCH_SIZE}"
    EXPORT_VARS+=",GROUNDING_MIN=${GROUNDING_MIN}"
    EXPORT_VARS+=",GROUNDING_MAX=${GROUNDING_MAX}"
    EXPORT_VARS+=",SEED=${SEED}"
    EXPORT_VARS+=",NUM_WORKERS=${NUM_WORKERS}"
    EXPORT_VARS+=",CONDA_ENV=${CONDA_ENV}"

    JOB_ID=$(sbatch --parsable --export="${EXPORT_VARS}" "${SCRIPT_DIR}/revision_master.sbatch")

    echo -e "[launcher] Master job submitted: ${GREEN}${JOB_ID}${NC}"
    echo ""
    echo "Configuration summary:"
    echo "  Number of workers:  ${NUM_WORKERS}"
    echo "  GPUs per worker:    2x H200"
    echo "  CPUs per worker:    16"
    echo "  Memory per worker:  128GB"
    echo "  Time per worker:    4 hours"
    echo ""
    echo "Monitor progress:"
    echo "  squeue -u \$USER"
    echo "  tail -f ${PROJECT_ROOT}/logs_evidencerl/revision_master_${JOB_ID}.out"
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

else
    # =========================================================================
    # LOCAL MODE: Run revision directly
    # =========================================================================

    # Initialize conda
    echo "[launcher] Initializing conda environment..."

    CONDA_INIT_SCRIPT=""
    for conda_path in \
        "$HOME/.conda/etc/profile.d/conda.sh" \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/mambaforge/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "/usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh"; do
        if [ -f "$conda_path" ]; then
            CONDA_INIT_SCRIPT="$conda_path"
            break
        fi
    done

    if [ -n "$CONDA_INIT_SCRIPT" ]; then
        source "$CONDA_INIT_SCRIPT"
    else
        CONDA_EXE_PATH="$(command -v conda 2>/dev/null || true)"
        if [ -n "$CONDA_EXE_PATH" ] && [ -x "$CONDA_EXE_PATH" ]; then
            CONDA_BASE="$("$CONDA_EXE_PATH" info --base)"
            source "$CONDA_BASE/etc/profile.d/conda.sh"
        fi
    fi

    set +u
    conda activate "${CONDA_ENV}" 2>/dev/null || true
    set -u

    # Set environment
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

    # Step 1: Run reasoning revision
    echo ""
    echo -e "[launcher] ${BLUE}Step 1: Running reasoning revision...${NC}"

    RUN_NLI_FLAG=""
    if [ "${SKIP_NLI}" = false ]; then
        RUN_NLI_FLAG="--run-nli"
    fi

    python "${SCRIPT_DIR}/revise_reasoning.py" \
        --generation-json "${GENERATION_JSON}" \
        --metrics-json "${METRICS_JSON}" \
        --output-dir "${OUTPUT_DIR}" \
        --model-name "${MODEL_PATH}" \
        --batch-size "${BATCH_SIZE}" \
        --grounding-min "${GROUNDING_MIN}" \
        --grounding-max "${GROUNDING_MAX}" \
        --seed "${SEED}" \
        ${RUN_NLI_FLAG}

    REVISION_EXIT_CODE=$?

    if [ $REVISION_EXIT_CODE -ne 0 ]; then
        echo -e "${RED}Error: Reasoning revision failed with exit code ${REVISION_EXIT_CODE}${NC}"
        exit 1
    fi

    # Step 2: Run NLI grounding recalculation (if not done in step 1 and not skipped)
    if [ "${SKIP_NLI}" = true ]; then
        echo ""
        echo -e "[launcher] ${YELLOW}Skipping NLI grounding recalculation (--skip-nli specified)${NC}"
    fi

    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}Pipeline Complete${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo "Output files saved to: ${OUTPUT_DIR}"
    echo ""
    echo "Files generated:"
    ls -la "${OUTPUT_DIR}"
    echo -e "${GREEN}============================================================${NC}"
fi
