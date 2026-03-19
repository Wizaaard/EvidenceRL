#!/bin/bash
#SBATCH --job-name=barexam_faiss
#SBATCH -o logs_barexam/faiss_index_%j.out
#SBATCH -e logs_barexam/faiss_index_%j.err
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

if [ -z "${PROJECT_ROOT:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
if [ -z "${DATA_ROOT:-}" ]; then
    DATA_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
fi

CONDA_ENV="${CONDA_ENV:-ragcon}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

BAREXAM_DATA="${BAREXAM_DATA:-${DATA_ROOT}/data/barexam}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-${BAREXAM_DATA}/faiss_index_${SPLIT}}"

echo "═══════════════════════════════════════════════"
echo "BarExam FAISS Index Builder"
echo "  Data: ${BAREXAM_DATA}"
echo "  Split: ${SPLIT}"
echo "  Output: ${OUTPUT_DIR}"
echo "═══════════════════════════════════════════════"

"${CONDA_PREFIX}/bin/python" "${PROJECT_ROOT}/scripts/barexam/build_barexam_faiss_index.py" \
    --barexam-data-path "${BAREXAM_DATA}" \
    --split "${SPLIT}" \
    --output-dir "${OUTPUT_DIR}" \
    --embedding-batch-size 256 \
    --index-type HNSW

echo "FAISS index build complete."
