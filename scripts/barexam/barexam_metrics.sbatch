#!/bin/bash
#SBATCH --job-name=barexam_metrics
#SBATCH -o logs_barexam/metrics_%j.out
#SBATCH -e logs_barexam/metrics_%j.err
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# ═══════════════════════════════════════════════════════════
# BarExam QA Metrics (NLI Grounding)
# ═══════════════════════════════════════════════════════════
# Env vars expected from launcher:
#   RESULTS_JSON   - Path to generation results JSON
#   OUTPUT_DIR     - Output directory for metrics (optional)
#   PROJECT_ROOT   - Path to evidenceRL/code
#   NO_GROUNDING   - "true" to skip NLI (optional)
# ═══════════════════════════════════════════════════════════

set -euo pipefail

if [ -z "${PROJECT_ROOT:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
CODE_DIR="${PROJECT_ROOT}"

CONDA_ENV="${CONDA_ENV:-ragcon}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

echo "═══════════════════════════════════════════════════"
echo "BarExam QA Metrics"
echo "  Results: ${RESULTS_JSON}"
echo "  Output:  ${OUTPUT_DIR:-auto}"
echo "═══════════════════════════════════════════════════"

CMD=(
    "${CONDA_PREFIX}/bin/python" "${CODE_DIR}/compute_barexam_metrics.py"
    --results-json "${RESULTS_JSON}"
)

if [ -n "${OUTPUT_DIR:-}" ]; then
    CMD+=(--output-dir "${OUTPUT_DIR}")
fi

if [ -n "${NLI_MODEL:-}" ]; then
    CMD+=(--nli-model "${NLI_MODEL}")
fi

if [ "${NO_GROUNDING:-false}" = "true" ]; then
    CMD+=(--no-grounding)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "Metrics complete."
