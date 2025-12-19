#!/usr/bin/env bash
set -euo pipefail

# ---------- CONFIGURABLE DEFAULTS ----------
MODEL_ID="gemma-3-4b"
MODEL_NAME="/home/hice1/jtamo3/bmed-sp-wang/YuxingData/Checkpoints/vllm_models/gemma-3-4b-it"
JUDGE_MODEL_NAME="/home/hice1/jtamo3/scratch/models/cache/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
PATIENT_DATA_PATH="/home/hice1/jtamo3/bmed-sp-wang/Ben/Data/physionet.org/files/mimic-iv-ext-cardiac-disease/1.0.0"
KNOWLEDGE_PATH="/home/hice1/jtamo3/bmed-sp-wang/Ben/Data/Medical-Cardiac-Knowledge"

CHUNK_SIZE=200
CHUNK_OVERLAP=50
BATCH_SIZE=16
PATIENT_PIPELINE="rag"
KNOWLEDGE_DATASET="ilyassacha/cardiologyChunks" 
KNOWLEDGE_DATASET_SPLIT="train"
KNOWLEDGE_DATASET_FIELD="text"
KNOWLEDGE_DATASET_MAX_RECORD="50000"

# Default max patients; can be overridden by first arg
MAX_PATIENTS="${1:-10}"

# Second arg = version number (default 0.1)
VERSION="${2:-0.1}"

# Default output path; can be overridden by second arg
JSON_OUTPUT="/home/hice1/jtamo3/scratch/Outputs/EvidenceRL/Runs/Baseline/${MODEL_ID}_${PATIENT_PIPELINE}_${MAX_PATIENTS}-v${VERSION}.json"

# ---------- ECHO CONFIG ----------
echo "Running RAG baseline with:"
echo "  MODEL_NAME       = $MODEL_NAME"
echo "  JUDGE_MODEL_NAME = $JUDGE_MODEL_NAME"
echo "  PATIENT_DATA     = $PATIENT_DATA_PATH"
echo "  KNOWLEDGE_PATH   = $KNOWLEDGE_PATH"
echo "  CHUNK_SIZE       = $CHUNK_SIZE"
echo "  CHUNK_OVERLAP    = $CHUNK_OVERLAP"
echo "  PATIENT_PIPELINE = $PATIENT_PIPELINE"
echo "  MAX_PATIENTS     = $MAX_PATIENTS"
echo "  BATCH_SIZE       = $BATCH_SIZE"
echo "  JSON_OUTPUT      = $JSON_OUTPUT"
echo

# ---------- RUN ----------
python main.py \
  --model-name "$MODEL_NAME" \
  --judge-model-name "$JUDGE_MODEL_NAME" \
  --patient-data-path "$PATIENT_DATA_PATH" \
  --chunk-size "$CHUNK_SIZE" \
  --chunk-overlap "$CHUNK_OVERLAP" \
  --patient-pipeline "$PATIENT_PIPELINE" \
  --max-patients "$MAX_PATIENTS" \
  --batch-size "$BATCH_SIZE" \
  --json-output "$JSON_OUTPUT" \
  --knowledge-dataset "$KNOWLEDGE_DATASET" \
  --knowledge-dataset-split "$KNOWLEDGE_DATASET_SPLIT" \
  --knowledge-dataset-text-field "$KNOWLEDGE_DATASET_FIELD" \
  --knowledge-dataset-max-records "$KNOWLEDGE_DATASET_MAX_RECORD"
