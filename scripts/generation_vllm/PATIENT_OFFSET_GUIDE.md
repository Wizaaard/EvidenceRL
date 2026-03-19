# Patient Offset Guide for vLLM Generation

## Overview

The vLLM generation scripts now support `PATIENT_OFFSET` parameter, allowing you to process specific ranges of patients. This is useful for:
- **Training/Test splits:** Process different patient subsets for training vs evaluation
- **Distributed processing:** Run multiple independent jobs on different patient ranges
- **Resuming failed runs:** Start from where a previous run left off

## How It Works

```bash
# Patient loading is SEQUENTIAL (not random)
# If you have 4700 total patients in your dataset:
# - Patient indices: 0, 1, 2, ..., 4699

# Example 1: First 3700 patients (training set)
PATIENT_OFFSET=0 TOTAL_PATIENTS=3700
# → Loads patients [0:3700] = 0, 1, 2, ..., 3699

# Example 2: Last 1000 patients (test set)
PATIENT_OFFSET=3700 TOTAL_PATIENTS=1000
# → Loads patients [3700:4700] = 3700, 3701, ..., 4699

# Example 3: Middle range
PATIENT_OFFSET=1000 TOTAL_PATIENTS=500
# → Loads patients [1000:1500] = 1000, 1001, ..., 1499
```

## Quick Start

### Training Set (Patients 0-3699)

```bash
cd /home/hice1/jtamo3/bmed-sp-wang/BenoitData/Restricted/evidenceRL/code/scripts/generation_vllm

# Option A: Using helper script (recommended)
./launch_training_set.sh gpt-oss-120b

# Option B: Manual configuration
PATIENT_OFFSET=0 TOTAL_PATIENTS=3700 NUM_WORKERS=20 ./launch_generation_vllm.sh gpt-oss-120b 1.0
```

**Output:** `generation_output/gpt-oss-120b_v1.0/gpt-oss-120b_3700-v1.0.json`

### Test Set (Patients 3700-4699)

```bash
cd /home/hice1/jtamo3/bmed-sp-wang/BenoitData/Restricted/evidenceRL/code/scripts/generation_vllm

# Option A: Using helper script (recommended)
./launch_test_set.sh gpt-oss-120b

# Option B: Manual configuration
PATIENT_OFFSET=3700 TOTAL_PATIENTS=1000 NUM_WORKERS=10 ./launch_generation_vllm.sh gpt-oss-120b 1.0-test
```

**Output:** `generation_output/gpt-oss-120b_v1.0-test/gpt-oss-120b_1000-v1.0-test.json`

## Advanced Usage

### Custom Patient Ranges

```bash
# Process patients 1500-2499
PATIENT_OFFSET=1500 TOTAL_PATIENTS=1000 NUM_WORKERS=10 ./launch_generation_vllm.sh gemma-3-4b 1.0

# Process just 100 patients starting from 500
PATIENT_OFFSET=500 TOTAL_PATIENTS=100 NUM_WORKERS=2 ./launch_generation_vllm.sh gemma-3-12b 1.0
```

### Verification

Check that training and test sets don't overlap:

```python
#!/usr/bin/env python3
import json

# Load both datasets
with open('generation_output/gpt-oss-120b_v1.0/gpt-oss-120b_3700-v1.0.json') as f:
    train = json.load(f)
with open('generation_output/gpt-oss-120b_v1.0-test/gpt-oss-120b_1000-v1.0-test.json') as f:
    test = json.load(f)

train_ids = {r['hadm_id'] for r in train['results']}
test_ids = {r['hadm_id'] for r in test['results']}

print(f"Training: {len(train_ids)} patients")
print(f"Test: {len(test_ids)} patients")
print(f"Overlap: {len(train_ids & test_ids)} patients")
print("✅ No overlap!" if len(train_ids & test_ids) == 0 else "⚠️ Warning: Overlap detected!")
```

## Worker Scaling

The master script automatically distributes patients across workers:

```bash
# Example: 3700 patients with 20 workers
PATIENT_OFFSET=0 TOTAL_PATIENTS=3700 NUM_WORKERS=20

# Worker distribution:
# - Worker 0:  patients [0:185]
# - Worker 1:  patients [185:370]
# - Worker 2:  patients [370:555]
# - ...
# - Worker 19: patients [3515:3700]
```

## Important Notes

1. **Sequential Loading:** Patients are loaded in the order they appear in the CSV files (not randomly sampled)
2. **No Overlap Guarantee:** Different `PATIENT_OFFSET` values ensure no overlap between runs
3. **Backward Compatible:** `PATIENT_OFFSET` defaults to 0, so existing scripts work unchanged
4. **File Naming:** Output files use `TOTAL_PATIENTS` in the name (e.g., `model_3700-v1.0.json`)

## Common Workflows

### Full Dataset (4700 patients)

```bash
# All patients in one run
PATIENT_OFFSET=0 TOTAL_PATIENTS=4700 NUM_WORKERS=47 ./launch_generation_vllm.sh gpt-oss-120b 1.0
```

### 70/30 Train/Test Split

```bash
# Training: 70% = 3290 patients
PATIENT_OFFSET=0 TOTAL_PATIENTS=3290 NUM_WORKERS=33 ./launch_generation_vllm.sh gpt-oss-120b 1.0-train

# Test: 30% = 1410 patients
PATIENT_OFFSET=3290 TOTAL_PATIENTS=1410 NUM_WORKERS=14 ./launch_generation_vllm.sh gpt-oss-120b 1.0-test
```

### Resume Failed Run

If a run failed at patient 2000:

```bash
# Resume from patient 2000
PATIENT_OFFSET=2000 TOTAL_PATIENTS=1700 NUM_WORKERS=17 ./launch_generation_vllm.sh gpt-oss-120b 1.0-resume
```

Then merge the partial results manually.

## Troubleshooting

**Q: How do I know the total number of patients in my dataset?**

```bash
cd /home/hice1/jtamo3/bmed-sp-wang/BenoitData/Restricted/evidenceRL/code
python -c "from evidence_rl import load_patient_cases; print(len(load_patient_cases('//storage/ice-shared/bmed-sp-wang/Ben/Data/physionet.org/files/mimic-iv-ext-cardiac-disease/1.0.0')))"
```

**Q: What happens if PATIENT_OFFSET + TOTAL_PATIENTS exceeds the dataset size?**

The script will process all available patients up to the end of the dataset (no error).

**Q: Can I run training and test generation simultaneously?**

Yes! They use different output directories based on the `VERSION` parameter:
- Training: `gpt-oss-120b_v1.0/`
- Test: `gpt-oss-120b_v1.0-test/`
