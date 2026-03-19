# EvidenceRL

A research framework for training LLMs to generate **evidence-grounded** generation using reinforcement learning. EvidenceRL combines NLI-based grounding rewards with accuracy signals to reduce hallucination and improve faithfulness across medical and legal domains.

## Overview

EvidenceRL addresses the problem of LLM hallucination in high-stakes domains by:

1. **Structured generation** — LLMs produce diagnoses with explicit clinical/legal reasoning
2. **Evidence grounding** — NLI entailment scoring measures whether reasoning is supported by retrieved documents
3. **RL fine-tuning** — Grounding and accuracy rewards drive GRPO, DPO, and SFT training to improve faithfulness

The framework supports two domains:
- **Medical** — Cardiac diagnosis from MIMIC-IV patient records
- **Legal** — Bar exam question answering

## Architecture

```
Patient/Query ──> [Retrieval (FAISS)] ──> [LLM Generation] ──> Structured Output
                                                                      │
                                              ┌───────────────────────┤
                                              ▼                       ▼
                                     NLI Grounding Score      LLM Judge Accuracy
                                        [-1, 1]                   [0, 1]
                                              │                       │
                                              └──────────┬────────────┘
                                                         ▼
                                                  Combined Reward
                                                         │
                                              ┌──────────┼──────────┐
                                              ▼          ▼          ▼
                                            GRPO        DPO        SFT
```

### Reward System

| Component | Range | Description |
|-----------|-------|-------------|
| R_grounding | [-1, 1] | NLI entailment — is reasoning supported by evidence? |
| R_accuracy | [0, 1] | LLM judge — does diagnosis match ground truth? |
| R_combined | weighted | `w * R_grounding + (1-w) * R_accuracy` |

### Hallucination Taxonomy

| Grounding | Accuracy | Interpretation |
|-----------|----------|----------------|
| High | High | **Ideal** — evidence-grounded correct diagnosis |
| High | Low | **Misinterpretation** — used evidence, wrong conclusion |
| Low | High | **Lucky guess** — correct but unfaithful reasoning |
| Low | Low | **Hallucination** — wrong and unfaithful |

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/evidencerl.git
cd evidencerl

# Option 1: Conda (recommended for GPU support)
conda env create -f environment.yml
conda activate ragcon

# Option 2: Pip
pip install -r requirements.txt

# Optional: Fast retrieval
pip install faiss-gpu  # or faiss-cpu
```

## Quick Start

### Medical Domain (vLLM)

```bash
# Single model generation (no-RAG baseline)
python run_evidence_rl_vllm.py \
    --model-name /path/to/gemma-3-12b-it \
    --patient-data-path /path/to/mimic-iv-cardiac \
    --no-rag \
    --tensor-parallel-size 2 \
    --json-output output/gemma-3-12b_baseline.json

# With RAG retrieval
python run_evidence_rl_vllm.py \
    --model-name /path/to/gemma-3-12b-it \
    --patient-data-path /path/to/mimic-iv-cardiac \
    --faiss-index-dir /path/to/faiss_index \
    --tensor-parallel-size 2 \
    --json-output output/gemma-3-12b_rag.json

# Compute metrics (grounding + accuracy)
python compute_evidencerl_metrics_vllm_enhanced.py \
    --generation-file output/gemma-3-12b_baseline.json \
    --judge-model /path/to/judge \
    --output-dir metrics/gemma-3-12b
```

### Legal Domain (Bar Exam)

```bash
python run_barexam_vllm.py \
    --model-name /path/to/gemma-3-12b-it \
    --patient-data-path /path/to/barexam-data \
    --tensor-parallel-size 2 \
    --json-output output/barexam_gemma-3-12b.json
```

### SLURM Cluster Execution

All generation and metrics scripts include SLURM launchers for distributed execution:

```bash
# Set model directory for your environment
export MODEL_BASE_DIR="/path/to/your/models"

# Launch baseline generation across workers
CONDA_ENV=ragcon TOTAL_PATIENTS=3700 PATIENT_OFFSET=0 \
    ./scripts/generation_baseline_vllm/launch_generation_baseline_vllm.sh gemma-3-12b

# Launch metrics computation
CONDA_ENV=ragcon ./scripts/launch_enhanced_metrics.sh \
    --generation-file generation_output/gemma-3-12b_v1.0/gemma-3-12b_3700-v1.0.json \
    --judge llama-3.3-70b \
    --output-dir metrics_output/gemma-3-12b
```

## Project Structure

```
src/evidence_rl/           # Core library
├── evidence_pipeline.py   # Main structured diagnosis pipeline
├── vllm_pipeline.py       # vLLM-accelerated pipeline
├── vllm_generation.py     # vLLM batch generation (tensor parallel)
├── vllm_evaluation.py     # vLLM-accelerated LLM judge
├── reward.py              # NLI grounding + accuracy rewards
├── enhanced_metrics.py    # Metrics computation
├── baseline.py            # Patient data loading
├── documents.py           # Document dataclasses
├── retrieval.py           # Semantic retrieval (embeddings)
├── faiss_retrieval.py     # FAISS-based fast retrieval
├── self_consistency_pipeline.py  # Self-consistency voting
├── self_rag_pipeline.py   # Self-RAG implementation
├── domains/               # Domain-specific modules
│   ├── medical.py         # Cardiac diagnosis
│   └── barexam.py         # Bar exam QA
└── ...

scripts/
├── generation_*_vllm/     # vLLM generation (baseline, RAG, DPO, GRPO, SFT, SC)
├── metrics_*_vllm/        # Metrics computation pipelines
├── rl_training/           # SFT, DPO, GRPO, faithfulness-DPO training
├── reasoning_revision*/   # Post-hoc reasoning revision
└── launch_enhanced_metrics.sh  # Two-phase metrics launcher

# Entry points
run_evidence_rl_vllm.py              # Main generation (medical)
run_barexam_vllm.py                  # Generation (legal/bar exam)
compute_evidencerl_metrics_vllm_enhanced.py  # Two-phase metrics
paper_metrics.py                     # Publication-ready metrics tables
```

## RL Training

EvidenceRL supports three RL training approaches:

### GRPO (Group Relative Policy Optimization)
```bash
# Build GRPO dataset
python scripts/rl_training/build_grpo_dataset.py \
    --metrics-dir metrics_output/gemma-3-12b \
    --output training_data/grpo/gemma-3-12b.json

# Train
python scripts/rl_training/train_grpo.py \
    --model-name /path/to/gemma-3-12b-it \
    --dataset training_data/grpo/gemma-3-12b.json \
    --output-dir checkpoints/gemma-3-12b-grpo
```

### DPO (Direct Preference Optimization)
```bash
# Build preference pairs from metrics
python scripts/rl_training/build_dpo_pairs.py \
    --metrics-dir metrics_output/ \
    --output training_data/dpo/

# Cross-model Faithfulness DPO (chosen/rejected picked across all models per patient)
python scripts/rl_training/build_faithfulness_dpo_dataset.py \
    --metrics-dirs metrics_output/gemma-3-4b metrics_output/gemma-3-12b ... \
    --output training_data/dpo/faithfulness_dpo_crossmodel.json

# Per-model Faithfulness DPO (within-model preference pairs, no cross-model distillation)
python scripts/rl_training/build_permodel_fdpo_dataset.py \
    --samples-file permodel_fdpo_samples/gemma-3-12b/gemma-3-12b_multisample_3x.json \
    --scores-file permodel_fdpo_samples/gemma-3-12b/grounding_scores.json \
    --output training_data/dpo/faithfulness_dpo_gemma-3-12b.json
```

### SFT (Supervised Fine-Tuning)
```bash
python scripts/rl_training/build_sft_dataset.py \
    --metrics-dir metrics_output/ \
    --output training_data/sft/

python scripts/rl_training/train_sft.py \
    --model-name /path/to/gemma-3-12b-it \
    --dataset training_data/sft/sft_dataset.json \
    --output-dir checkpoints/gemma-3-12b-sft
```

## Supported Models

The framework has been tested with 8 model backbones:

| Model | Parameters | Tensor Parallel |
|-------|-----------|-----------------|
| gemma-3-4b-it | 4B | 2 GPUs |
| gemma-3-12b-it | 12B | 2 GPUs |
| gemma-3-27b-it | 27B | 2 GPUs |
| Llama-3.2-3B-Instruct | 3B | 2 GPUs |
| Llama-3.1-8B-Instruct | 8B | 2 GPUs |
| Llama-3.3-70B-Instruct | 70B | 4 GPUs |
| gpt-oss-20b | 20B | 2 GPUs |
| gpt-oss-120b | 120B | 4 GPUs |

## Configuration

Set `MODEL_BASE_DIR` to point to your local model directory:

```bash
export MODEL_BASE_DIR="/path/to/your/models"
```

All SLURM launcher scripts will use this to resolve model paths. The project root is auto-derived from script locations — no other path configuration is needed.

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA 12.1)
- transformers 4.30+
- vLLM 0.6+ (for vLLM pipelines)
- sentence-transformers 2.2+
- FAISS (optional, for fast retrieval)



