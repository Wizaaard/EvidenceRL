#!/usr/bin/env bash
# ============================================================================
# Launch all GRPO no-RAG models for test-set evaluation
# ============================================================================
# Launches generation for all completed GRPO no-RAG models on the test set
# (1000 patients starting at offset 3700).
#
# Per-model settings match the baseline evaluation convention:
#   - EXTRACTOR_MODEL=gemma-3-12b for: gemma-3-4b, Llama-3.2-3B
#   - All others use defaults (self as extractor)
#
# Usage:
#   ./launch_all_grpo_no_rag.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Common settings
export CONDA_ENV="${CONDA_ENV:-ragcon}"
export TOTAL_PATIENTS="${TOTAL_PATIENTS:-1000}"
export PATIENT_OFFSET="${PATIENT_OFFSET:-3700}"
VERSION="${VERSION:-1.0-TEST}"

echo "============================================================"
echo "Launching ALL GRPO no-RAG models for test evaluation"
echo "============================================================"
echo "CONDA_ENV:       ${CONDA_ENV}"
echo "TOTAL_PATIENTS:  ${TOTAL_PATIENTS}"
echo "PATIENT_OFFSET:  ${PATIENT_OFFSET}"
echo "VERSION:         ${VERSION}"
echo "============================================================"
echo ""

# ---- Standard models (2 GPU, no special settings) ----

echo ">>> gemma-3-12b (GRPO no-RAG)"
"${SCRIPT_DIR}/launch_generation_grpo_vllm.sh" gemma-3-12b "${VERSION}"
echo ""

echo ">>> gemma-3-27b (GRPO no-RAG)"
"${SCRIPT_DIR}/launch_generation_grpo_vllm.sh" gemma-3-27b "${VERSION}"
echo ""

echo ">>> Llama-3.1-8B (GRPO no-RAG)"
"${SCRIPT_DIR}/launch_generation_grpo_vllm.sh" Llama-3.1-8B "${VERSION}"
echo ""

# ---- Models needing EXTRACTOR_MODEL=gemma-3-12b ----

echo ">>> gemma-3-4b (GRPO no-RAG, extractor: gemma-3-12b)"
EXTRACTOR_MODEL=gemma-3-12b "${SCRIPT_DIR}/launch_generation_grpo_vllm.sh" gemma-3-4b "${VERSION}"
echo ""

echo ">>> Llama-3.2-3B (GRPO no-RAG, extractor: gemma-3-12b)"
EXTRACTOR_MODEL=gemma-3-12b "${SCRIPT_DIR}/launch_generation_grpo_vllm.sh" Llama-3.2-3B "${VERSION}"
echo ""

echo "============================================================"
echo "All GRPO no-RAG jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "============================================================"
