#!/usr/bin/env bash
# ============================================================================
# Launch all GRPO no-RAG training jobs
# ============================================================================
# Convenience script to launch GRPO training for all 7 model backbones
# in no-RAG mode. Uses default hyperparameters.
#
# Prerequisites:
#   1. Build GRPO dataset first:
#      python build_grpo_dataset.py --output-dir training_data/grpo/ --modes no-rag
#   2. Ensure vLLM is compatible with TRL:
#      pip install vllm==0.12.0.post1
#
# Usage:
#   ./launch_all_grpo_no_rag.sh
#   ./launch_all_grpo_no_rag.sh --dry-run
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "Launching ALL GRPO no-RAG training jobs"
echo "============================================================"
echo ""

"${SCRIPT_DIR}/launch_grpo_training.sh" \
    --mode no-rag \
    --conda-env "${CONDA_ENV:-ragcon}" \
    "$@"
