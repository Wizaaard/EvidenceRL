#!/bin/bash
#SBATCH --job-name=barexam_gen
#SBATCH -o logs_barexam/gen_%j.out
#SBATCH -e logs_barexam/gen_%j.err
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00

# ═══════════════════════════════════════════════════════════
# BarExam QA Generation (vLLM)
# ═══════════════════════════════════════════════════════════
# Env vars expected from launcher:
#   MODEL_NAME     - Full model path
#   MODEL_ID       - Short model identifier
#   SPLIT          - test / validation / train / all
#   VERSION        - Output version tag
#   NUM_GPUS       - Tensor parallel size
#   BAREXAM_DATA   - Path to BarExam data directory
#   NO_RAG         - "true" for closed-book baseline
#   FAISS_INDEX_DIR- Path to FAISS index for retrieval (optional)
#   MAX_PASSAGES   - Number of passages to retrieve (default: 3)
#   INCLUDE_GOLD   - "true" to always include gold passage (optional)
#   MAX_CASES      - Max cases (optional)
#   PROJECT_ROOT   - Path to evidenceRL/code
#   DATA_ROOT      - Path to evidenceRL/
# ═══════════════════════════════════════════════════════════

set -euo pipefail

# PROJECT_ROOT and DATA_ROOT are passed from the launcher as env vars.
if [ -z "${PROJECT_ROOT:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
if [ -z "${DATA_ROOT:-}" ]; then
    DATA_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
fi
CODE_DIR="${PROJECT_ROOT}"

# Defaults
SPLIT="${SPLIT:-test}"
VERSION="${VERSION:-v1.0}"
NUM_GPUS="${NUM_GPUS:-2}"
NO_RAG="${NO_RAG:-false}"

# Conda
CONDA_ENV="${CONDA_ENV:-ragcon}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

echo "═══════════════════════════════════════════════════"
echo "BarExam QA Generation"
echo "  Model:   ${MODEL_ID} (${MODEL_NAME})"
echo "  Split:   ${SPLIT}"
echo "  Mode:    $([ "${NO_RAG}" = "true" ] && echo "no-RAG" || echo "RAG")"
echo "  GPUs:    ${NUM_GPUS}"
echo "  Version: ${VERSION}"
echo "═══════════════════════════════════════════════════"

# Build output path
# MODE_TAG can be overridden by the launcher (e.g., grpo_rag); defaults to rag/norag
if [ -z "${MODE_TAG:-}" ]; then
    MODE_TAG=$([ "${NO_RAG}" = "true" ] && echo "norag" || echo "rag")
fi
OUTPUT_DIR="${DATA_ROOT}/barexam_output/${SPLIT}/${MODEL_ID}_${MODE_TAG}-${VERSION}"
OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_ID}_barexam_${MODE_TAG}-${VERSION}.json"

mkdir -p "${OUTPUT_DIR}"

# Build command
CMD=(
    "${CONDA_PREFIX}/bin/python" "${CODE_DIR}/run_barexam_vllm.py"
    --model-name "${MODEL_NAME}"
    --barexam-data-path "${BAREXAM_DATA}"
    --split "${SPLIT}"
    --tensor-parallel-size "${NUM_GPUS}"
    --json-output "${OUTPUT_FILE}"
)

if [ "${NO_RAG}" = "true" ]; then
    CMD+=(--no-rag)
fi

if [ -n "${MAX_CASES:-}" ]; then
    CMD+=(--max-cases "${MAX_CASES}")
fi

if [ -n "${FAISS_INDEX_DIR:-}" ]; then
    CMD+=(--faiss-index-dir "${FAISS_INDEX_DIR}")
fi

if [ -n "${MAX_PASSAGES:-}" ]; then
    CMD+=(--max-passages "${MAX_PASSAGES}")
fi

if [ "${INCLUDE_GOLD:-false}" = "true" ]; then
    CMD+=(--include-gold)
fi

if [ -n "${LORA_PATH:-}" ]; then
    CMD+=(--lora-path "${LORA_PATH}")
fi

if [ -n "${EXTRACTOR_MODEL:-}" ]; then
    CMD+=(--extractor-model "${EXTRACTOR_MODEL}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "Output saved to: ${OUTPUT_FILE}"
echo "Run compute_barexam_metrics.py next."
