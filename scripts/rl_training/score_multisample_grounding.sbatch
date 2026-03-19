#!/usr/bin/env bash
#SBATCH -J erl_score_grounding
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o logs_multisample/score_grounding_%j.out
#SBATCH -e logs_multisample/score_grounding_%j.err

# ============================================================================
# Score NLI grounding for multi-sample outputs (GPU)
# ============================================================================
# Usage:
#   sbatch --export=ALL,INPUT=...,OUTPUT=... score_multisample_grounding.sbatch
#   sbatch --export=ALL,SCORE_ALL=true score_multisample_grounding.sbatch
# ============================================================================

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-ragcon}"
PROJECT_ROOT="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/code"

echo "============================================================"
echo "NLI Grounding Scoring (GPU)"
echo "============================================================"
echo "INPUT:      ${INPUT:-SCORE_ALL mode}"
echo "OUTPUT:     ${OUTPUT:-auto}"
echo "SCORE_ALL:  ${SCORE_ALL:-false}"
echo "============================================================"

# GPU check
if ! nvidia-smi &>/dev/null; then
    echo "[ERROR] GPU not available"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# CUDA_VISIBLE_DEVICES
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    [ -n "${SLURM_JOB_GPUS:-}" ] && export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
    [ -n "${SLURM_STEP_GPUS:-}" ] && export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS}"
fi

# Conda
for conda_path in "$HOME/.conda/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/mambaforge/etc/profile.d/conda.sh" "/usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh"; do
    [ -f "$conda_path" ] && { source "$conda_path"; break; }
done

set +u
eval "$(conda shell.bash hook 2>/dev/null || true)"
conda activate "${CONDA_ENV}"
set -u

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${PROJECT_ROOT}/logs_multisample"

CMD_ARGS=()

if [ "${SCORE_ALL:-false}" = "true" ]; then
    CMD_ARGS+=(--score-all)
else
    : "${INPUT:?Need INPUT}"
    CMD_ARGS+=(--input "${INPUT}")
    if [ -n "${OUTPUT:-}" ]; then
        CMD_ARGS+=(--output "${OUTPUT}")
    fi
fi

# Full-reasoning mode (no sentence-level), GPU
CMD_ARGS+=(--no-sentence-level)

python "${PROJECT_ROOT}/scripts/rl_training/score_multisample_grounding.py" "${CMD_ARGS[@]}"

echo ""
echo "============================================================"
echo "Grounding scoring complete!"
echo "============================================================"
