#!/usr/bin/env bash
# ============================================================================
# Launch all SFT no-RAG models for test-set evaluation
# ============================================================================
# Launches generation for all completed SFT no-RAG models on the test set
# (1000 patients starting at offset 3700).
#
# Per-model settings match the baseline evaluation convention:
#   - EXTRACTOR_MODEL=gemma-3-12b for: gemma-3-4b, Llama-3.2-3B, gpt-oss-20b
#   - MAX_TOKENS=4096 for: gpt-oss models
#   - All others use defaults
#
# Usage:
#   ./launch_all_sft_no_rag.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Common settings
export CONDA_ENV="${CONDA_ENV:-ragcon}"
export TOTAL_PATIENTS="${TOTAL_PATIENTS:-1000}"
export PATIENT_OFFSET="${PATIENT_OFFSET:-3700}"
VERSION="${VERSION:-1.0-TEST}"

echo "============================================================"
echo "Launching ALL SFT no-RAG models for test evaluation"
echo "============================================================"
echo "CONDA_ENV:       ${CONDA_ENV}"
echo "TOTAL_PATIENTS:  ${TOTAL_PATIENTS}"
echo "PATIENT_OFFSET:  ${PATIENT_OFFSET}"
echo "VERSION:         ${VERSION}"
echo "============================================================"
echo ""

# ---- Standard models (2 GPU, no special settings) ----

echo ">>> gemma-3-12b-sft-no-rag"
"${SCRIPT_DIR}/launch_generation_sft_vllm.sh" gemma-3-12b-sft-no-rag "${VERSION}"
echo ""

echo ">>> gemma-3-27b-sft-no-rag"
"${SCRIPT_DIR}/launch_generation_sft_vllm.sh" gemma-3-27b-sft-no-rag "${VERSION}"
echo ""

echo ">>> Llama-3.1-8B-sft-no-rag"
"${SCRIPT_DIR}/launch_generation_sft_vllm.sh" Llama-3.1-8B-sft-no-rag "${VERSION}"
echo ""

# ---- Models needing EXTRACTOR_MODEL=gemma-3-12b ----

echo ">>> gemma-3-4b-sft-no-rag (extractor: gemma-3-12b)"
EXTRACTOR_MODEL=gemma-3-12b "${SCRIPT_DIR}/launch_generation_sft_vllm.sh" gemma-3-4b-sft-no-rag "${VERSION}"
echo ""

echo ">>> Llama-3.2-3B-sft-no-rag (extractor: gemma-3-12b)"
EXTRACTOR_MODEL=gemma-3-12b "${SCRIPT_DIR}/launch_generation_sft_vllm.sh" Llama-3.2-3B-sft-no-rag "${VERSION}"
echo ""

# ---- gpt-oss-20b (extractor + MAX_TOKENS=4096) ----

echo ">>> gpt-oss-20b-sft-no-rag (extractor: gemma-3-12b, max_tokens: 4096)"
EXTRACTOR_MODEL=gemma-3-12b MAX_TOKENS=4096 "${SCRIPT_DIR}/launch_generation_sft_vllm.sh" gpt-oss-20b-sft-no-rag "${VERSION}"
echo ""

# ---- Large model (4 GPU) ----

echo ">>> Llama-3.3-70B-sft-no-rag (4 GPU)"
"${SCRIPT_DIR}/launch_generation_sft_vllm.sh" Llama-3.3-70B-sft-no-rag "${VERSION}"
echo ""

echo "============================================================"
echo "All SFT no-RAG jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "============================================================"
